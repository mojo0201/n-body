//! Application shell: window, wgpu, egui, and the render/physics thread split.
//!
//! Threading model
//! ---------------
//! Two threads share one `Bodies` behind a `Mutex`:
//!
//!   * the *render thread* (this file, winit's event loop) — draws, runs egui,
//!     applies UI actions. It must never block for more than one frame.
//!   * the *physics thread* — advances the sim, builds orbit lines on request.
//!
//! `Shared::render_wants` is a priority-flag: the render thread sets it to
//! `true` before taking the lock; the physics thread polls it and backs off.
//! This means a busy simulation can *never* starve a frame, at the cost of a
//! small busy-wait on the physics side (see `yield_to_render`).
//!
//! The render thread also owns the frame rate / idle policy (`idle_frames`):
//! after `IDLE_FRAMES` frames with nothing changing, it stops requesting
//! redraws and only wakes on input — so an idle sim costs ~0 % CPU.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nbody_rust::physics::{self, Bodies, BodyClass, BodyKind, BodyOrigin};
use nbody_rust::render::{self, OrbitReq, OrbitVertex, Renderer};
use nbody_rust::ui::{self, Actions, Edit, UiState, ViewInfo};
use nbody_rust::Preset;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use egui_wgpu::ScreenDescriptor;

// Zoom range for the view: 1e-3 .. 1e5 px per Gm.
const ZOOM_MIN: f32 = 1e-3;
const ZOOM_MAX: f32 = 1e5;
// A click that moves less than this many pixels is treated as a click
const CLICK_SLOP_PX: f32 = 4.0;

/// Frames to keep drawing after the last thing that changed, so egui can settle.
const IDLE_FRAMES: u32 = 3;
/// While the simulation runs, orbit lines are rebuilt at most this often.
const ORBIT_REFRESH_MS: u128 = 1000;
/// The physics thread gives the render thread the lock after this long.
const PHYS_SLICE: Duration = Duration::from_millis(4);

// ---------------------------------------------------------------------------
// State shared between the render thread and the physics thread
// ---------------------------------------------------------------------------

fn lock_ok<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

struct Shared {
    bodies: Mutex<Bodies>,
    render_wants: AtomicBool,
    quit: AtomicBool,
    paused: AtomicBool,
    speed_bits: AtomicU64,
    clear_accum: AtomicBool,
    behind: AtomicBool,
    phys_ns: AtomicU64,
    orbit_req: Mutex<Option<(u64, OrbitReq)>>,
    orbit_out: Mutex<Option<(u64, Vec<OrbitVertex>)>>,
}

impl Shared {
    fn new(bodies: Bodies, paused: bool, speed: f64) -> Self {
        Self {
            bodies: Mutex::new(bodies),
            render_wants: AtomicBool::new(false),
            quit: AtomicBool::new(false),
            paused: AtomicBool::new(paused),
            speed_bits: AtomicU64::new(speed.to_bits()),
            clear_accum: AtomicBool::new(false),
            behind: AtomicBool::new(false),
            phys_ns: AtomicU64::new(0),
            orbit_req: Mutex::new(None),
            orbit_out: Mutex::new(None),
        }
    }

    fn lock_for_render(&self) -> RenderGuard<'_> {
        self.render_wants.store(true, Ordering::Release);
        RenderGuard {
            guard: lock_ok(&self.bodies),
            wants: &self.render_wants,
        }
    }

    fn yield_to_render(&self) {
        while self.render_wants.load(Ordering::Acquire) && !self.quit.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_micros(100));
        }
    }
}

struct RenderGuard<'a> {
    guard: MutexGuard<'a, Bodies>,
    wants: &'a AtomicBool,
}

impl Deref for RenderGuard<'_> {
    type Target = Bodies;
    fn deref(&self) -> &Bodies {
        &self.guard
    }
}

impl DerefMut for RenderGuard<'_> {
    fn deref_mut(&mut self) -> &mut Bodies {
        &mut self.guard
    }
}

impl Drop for RenderGuard<'_> {
    fn drop(&mut self) {
        self.wants.store(false, Ordering::Release);
    }
}

fn physics_loop(sh: Arc<Shared>) {
    let mut last = Instant::now();
    let mut accum = 0.0_f64;
    let mut behind_until: Option<Instant> = None;

    while !sh.quit.load(Ordering::Relaxed) {
        let now = Instant::now();
        let dt_real = now.duration_since(last).as_secs_f64();
        last = now;

        if sh.clear_accum.swap(false, Ordering::AcqRel) {
            accum = 0.0;
        }
        let paused = sh.paused.load(Ordering::Relaxed);
        let speed = f64::from_bits(sh.speed_bits.load(Ordering::Relaxed)).max(1e-6);
        if !paused {
            accum += dt_real.min(0.1) * speed;
        }

        let mut worked = false;

        let req = lock_ok(&sh.orbit_req).take();
        if let Some((id, req)) = req {
            sh.yield_to_render();
            let segs = {
                let b = lock_ok(&sh.bodies);
                physics::compute_orbit_segments(&b, req.zoom, req.planets, req.debris)
            };
            let verts = render::segments_to_vertices(&segs);
            *lock_ok(&sh.orbit_out) = Some((id, verts));
            worked = true;
        }

        if !paused && accum >= physics::DT {
            sh.yield_to_render();
            let mut b = lock_ok(&sh.bodies);
            let t0 = Instant::now();
            while accum >= physics::DT {
                physics::advance(&mut b, physics::DT);
                accum -= physics::DT;
                if sh.render_wants.load(Ordering::Acquire) || t0.elapsed() >= PHYS_SLICE {
                    break;
                }
            }
            drop(b);
            sh.phys_ns
                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            worked = true;
        }

        let limit = (0.25 * speed).max(8.0 * physics::DT);
        if accum > limit {
            accum = 0.0;
            behind_until = Some(Instant::now() + Duration::from_secs(1));
        }
        let behind = !paused && behind_until.is_some_and(|t| Instant::now() < t);
        sh.behind.store(behind, Ordering::Relaxed);

        if !worked {
            let wait = if paused {
                2.0e-3
            } else {
                ((physics::DT - accum).max(0.0) / speed).clamp(2.0e-4, 2.0e-3)
            };
            std::thread::sleep(Duration::from_secs_f64(wait));
        }
    }
}

// ---------------------------------------------------------------------------
// Graphics and view state
// ---------------------------------------------------------------------------

struct GfxState {
    renderer: Renderer,
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
    egui_ctx: egui::Context,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    window: Arc<Window>,
}

struct TrailConfig {
    max_age: f64,
    max_bodies: usize,
    non_rubble_only: bool,
    enabled: bool,
}

impl Default for TrailConfig {
    fn default() -> Self {
        Self {
            max_age: 3.0,
            max_bodies: 64,
            non_rubble_only: true,
            enabled: true,
        }
    }
}

type TrailSample = (u64, f64, f32, f32, [u8; 4], f64, Option<(f32, f32)>);

struct Trail {
    history: VecDeque<(f64, f32, f32)>,
    color: [u8; 4],
}

const TRAIL_MAX_POINTS: usize = 1800;
const TRAIL_MAX_VERTS: usize = 100_000;

struct TrailState {
    history: HashMap<u64, Trail>,
    cfg: TrailConfig,
}

impl TrailState {
    fn new() -> Self {
        Self {
            history: HashMap::new(),
            cfg: TrailConfig::default(),
        }
    }

    fn clear(&mut self) {
        self.history.clear();
    }

    /// O(n) under the lock. Returns (id, sim_time, x, y, rgb, period, parent_pos).
    /// `parent_pos` is used for arc sub-sampling in `update`.
    fn snapshot(&self, b: &Bodies) -> Vec<TrailSample> {
        if !self.cfg.enabled {
            return Vec::new();
        }
        let cap = self.cfg.max_bodies.min(b.len());
        let mut out = Vec::with_capacity(cap);
        for i in 0..b.len() {
            if out.len() >= self.cfg.max_bodies {
                break;
            }
            if self.cfg.non_rubble_only && (b.origin[i].is_belt() || b.origin[i].is_rubble()) {
                continue;
            }
            let col = render::body_color(b, i);
            let color = [
                (col[0].clamp(0.0, 1.0) * 255.0) as u8,
                (col[1].clamp(0.0, 1.0) * 255.0) as u8,
                (col[2].clamp(0.0, 1.0) * 255.0) as u8,
                255,
            ];
            // Fetch orbital period and parent position for curve sub-sampling.
            let (period, parent_pos) = b
                .parent_cache
                .get(i)
                .copied()
                .flatten()
                .filter(|&p| p != i)
                .map(|p| {
                    let o = crate::physics::orbit_info(b, i, p);
                    (o.period, Some((b.x[p] as f32, b.y[p] as f32)))
                })
                .unwrap_or((f64::INFINITY, None));
            out.push((
                b.id[i],
                b.time,
                b.x[i] as f32,
                b.y[i] as f32,
                color,
                period,
                parent_pos,
            ));
        }
        out
    }

    /// Cheap; runs without the lock. Uses the snapshot to update per-body
    /// history.
    fn update(&mut self, snapshot: &[TrailSample]) {
        let max_age = self.cfg.max_age;
        let now_ids: HashSet<u64> = snapshot.iter().map(|s| s.0).collect();
        self.history.retain(|id, _| now_ids.contains(id));

        for &(id, t, x, y, color, period, parent_pos) in snapshot {
            let trail = self.history.entry(id).or_insert_with(|| Trail {
                history: VecDeque::new(),
                color,
            });
            trail.color = color;
            let is_new = trail
                .history
                .back()
                .is_none_or(|&(tl, _, _)| t > tl + 1e-12);

            if is_new {
                if let Some(&(t_last, x_last, y_last)) = trail.history.back() {
                    let dt = t - t_last;
                    // Sub-sample the path if the frame jumped a significant
                    // amount of sim time (e.g. at high simulation speeds).
                    if dt > 0.005 {
                        let steps = ((dt / 0.005).ceil() as usize).min(64);
                        for s in 1..steps {
                            let frac = s as f32 / steps as f32;
                            let (ix, iy) = if let Some((px, py)) = parent_pos {
                                // Circular arc around the parent for round trails.
                                let dx0 = x_last - px;
                                let dy0 = y_last - py;
                                let dx1 = x - px;
                                let dy1 = y - py;
                                let r0 = (dx0 * dx0 + dy0 * dy0).sqrt();
                                let r1 = (dx1 * dx1 + dy1 * dy1).sqrt();
                                if r0 < 1e-6 || r1 < 1e-6 {
                                    (x_last + (x - x_last) * frac, y_last + (y - y_last) * frac)
                                } else {
                                    let a0 = dy0.atan2(dx0);
                                    let a1 = dy1.atan2(dx1);
                                    let mut da = a1 - a0;
                                    // Shortest arc to avoid the trail looping backward
                                    if da > std::f32::consts::PI {
                                        da -= 2.0 * std::f32::consts::PI;
                                    } else if da < -std::f32::consts::PI {
                                        da += 2.0 * std::f32::consts::PI;
                                    }
                                    let r = r0 + (r1 - r0) * frac;
                                    let a = a0 + da * frac;
                                    (px + r * a.cos(), py + r * a.sin())
                                }
                            } else {
                                // Fallback to linear interpolation (stars)
                                (x_last + (x - x_last) * frac, y_last + (y - y_last) * frac)
                            };
                            trail.history.push_back((t_last + dt * frac as f64, ix, iy));
                        }
                    }
                }
                trail.history.push_back((t, x, y));
            }

            // Cap trail length to exactly one orbital period (plus 5% margin)
            let effective_age = if period.is_finite() && period > 0.0 {
                max_age.min(period * 1.05)
            } else {
                max_age
            };
            let t_newest = trail.history.back().map(|v| v.0).unwrap_or(t);
            while let Some(&(t0, _, _)) = trail.history.front() {
                if t_newest - t0 > effective_age {
                    trail.history.pop_front();
                } else {
                    break;
                }
            }
            // Global hard cap
            while trail.history.len() > TRAIL_MAX_POINTS {
                trail.history.pop_front();
            }
        }
    }

    /// Heavy; runs without the lock. Turns the per-body history into
    /// line-list vertices with alpha fading toward the tail.
    fn build_vertices(&self) -> Vec<OrbitVertex> {
        let mut out = Vec::new();
        'outer: for trail in self.history.values() {
            let n = trail.history.len();
            if n < 2 {
                continue;
            }
            let t_last = trail.history[n - 1].0;
            let t_first = trail.history[0].0;

            // Fade over the actual span of this trail, not the config max_age.
            let span = (t_last - t_first).max(1e-9);
            let rgb = [trail.color[0], trail.color[1], trail.color[2]];
            for k in 0..n - 1 {
                let (t0, x0, y0) = trail.history[k];
                let (t1, x1, y1) = trail.history[k + 1];

                // 30% alpha floor at the tail so the trail doesn't vanish.
                let a0 =
                    ((((1.0 - (t_last - t0) / span) * 0.7 + 0.3).clamp(0.0, 1.0)) * 200.0) as u8;
                let a1 =
                    ((((1.0 - (t_last - t1) / span) * 0.7 + 0.3).clamp(0.0, 1.0)) * 200.0) as u8;
                out.push(OrbitVertex {
                    pos: [x0, y0],
                    color: [rgb[0], rgb[1], rgb[2], a0],
                });
                out.push(OrbitVertex {
                    pos: [x1, y1],
                    color: [rgb[0], rgb[1], rgb[2], a1],
                });
                if out.len() >= TRAIL_MAX_VERTS {
                    break 'outer;
                }
            }
        }
        out
    }
}

/// Build comet tail and coma vertices for every active comet.
///
/// A comet is "active" when it is Icy, has a luminous host, and is
/// inside that host's snow line. Its brightness scales roughly as
/// the incident flux (1 / r²), clipped to [0, 1]. The tail is a cone
/// of lines pointing **anti-sunward** (from host to comet), tapering
/// from bright at the nucleus to transparent at the tip. The coma is
/// a small sunburst of radiating lines around the nucleus.
///
/// Returns line-list vertices in **world** coordinates (same
/// convention as trails and orbits — the orbit pipeline subtracts
/// `camera.pos` itself). Empty when there are no comets in the
/// system, so a plain Solar System without comet JSON costs nothing.
fn build_comet_tails(b: &Bodies, _cam: [f64; 2]) -> Vec<OrbitVertex> {
    let mut out = Vec::new();
    // Bail out cheaply on systems with no comets.
    if !b.origin.iter().any(|&o| o == BodyOrigin::Comet) {
        return out;
    }
    let hosts: Vec<usize> = (0..b.len())
        .filter(|&i| {
            matches!(
                b.kind[i],
                BodyKind::Star
                    | BodyKind::BrownDwarf
                    | BodyKind::WhiteDwarf
                    | BodyKind::NeutronStar
                    | BodyKind::BlackHole
            )
        })
        .collect();
    if hosts.is_empty() {
        return out;
    }
    let au = physics::au_in_gm();
    for i in 0..b.len() {
        if b.origin[i] != BodyOrigin::Comet || b.class[i] != BodyClass::Icy {
            continue;
        }
        // Nearest host.
        let mut r_min = f64::INFINITY;
        let mut h = hosts[0];
        for &k in &hosts {
            let d = (b.x[i] - b.x[k]).hypot(b.y[i] - b.y[k]);
            if d < r_min {
                r_min = d;
                h = k;
            }
        }
        let r_au = r_min / au;
        let lum = physics::luminosity_of(b, h).max(0.0);
        if lum <= 0.0 {
            continue;
        }
        let snow_au = 2.7 * lum.sqrt();
        if r_au >= snow_au || r_au < 1e-3 {
            continue;
        }
        // Activity: incident flux, clipped to [0, 1]. At 1 AU the tail
        // is already fully developed; inside 1 AU it can't grow longer
        // (clamped) but the comet stays bright.
        let activity = (1.0 / (r_au * r_au)).min(1.0);
        if activity <= 0.01 {
            continue;
        }
        // Anti-sunward unit vector.
        let dx = b.x[i] - b.x[h];
        let dy = b.y[i] - b.y[h];
        let r = dx.hypot(dy).max(1e-9);
        let (ux, uy) = (dx / r, dy / r);
        // Tail length: 0.06 AU at full activity is a realistic moderately
        // bright comet (Hale-Bopp reached ~1 AU, most visible comets
        // 0.05-0.3 AU). 0.06 AU no longer dominates the inner-system view.
        let tail_len_gm = 0.06 * au * activity;
        // Tail spread: real tails are narrow streams (aspect 20:1+), not
        // wide fans. 0.018 rad gives ~55:1 at full length — a stream.
        // An ion tail is extremely straight and narrow.
        const TAIL_LINES: usize = 3;
        const TAIL_STEPS: usize = 3;
        const TAIL_SPREAD: f64 = 0.008;
        let (r_c, g_c, b_c) = (190u8, 220u8, 255u8);
        // Coma: small fuzzy halo around the nucleus. 1.2 % of the tail
        // = 0.0007 AU = 105,000 km at full activity — realistic for a
        // bright active comet (real comas are 10^4-10^6 km radius).
        let coma_r_gm = 0.012 * tail_len_gm;
        // Start the tail rays at the coma's anti-sunward edge, NOT the
        // nucleus. Without this, all N tail rays + the coma rays converge
        // at one point, saturate, and read as a circular white blob.
        let x_start = b.x[i] + ux * coma_r_gm;
        let y_start = b.y[i] + uy * coma_r_gm;
        for k in 0..TAIL_LINES {
            let t = k as f64 / (TAIL_LINES - 1) as f64 - 0.5;
            let ang = t * TAIL_SPREAD;
            let (ca, sa) = ang.cos().sin_cos();
            let tx = ux * ca - uy * sa;
            let ty = ux * sa + uy * ca;
            for s in 0..TAIL_STEPS {
                let f0 = s as f64 / TAIL_STEPS as f64;
                let f1 = (s + 1) as f64 / TAIL_STEPS as f64;
                let x0 = x_start + tx * tail_len_gm * f0;
                let y0 = y_start + ty * tail_len_gm * f0;
                let x1 = x_start + tx * tail_len_gm * f1;
                let y1 = y_start + ty * tail_len_gm * f1;
                let a0 = ((1.0 - f0) * activity * 120.0) as u8;
                let a1 = ((1.0 - f1) * activity * 120.0) as u8;
                out.push(OrbitVertex {
                    pos: [x0 as f32, y0 as f32],
                    color: [r_c, g_c, b_c, a0],
                });
                out.push(OrbitVertex {
                    pos: [x1 as f32, y1 as f32],
                    color: [r_c, g_c, b_c, a1],
                });
            }
        }
        // Coma: 8 faint rays (was 24 bright ones). A soft glow, not a
        // sunburst. At system scale this is sub-pixel; only when zoomed
        // right in does it show structure.
        if coma_r_gm > 1e-6 {
            const COMA_RAYS: usize = 8;
            for k in 0..COMA_RAYS {
                let ang = k as f64 / COMA_RAYS as f64 * std::f64::consts::TAU;
                let (ca, sa) = ang.cos().sin_cos();
                let x0 = b.x[i];
                let y0 = b.y[i];
                let x1 = b.x[i] + ca * coma_r_gm;
                let y1 = b.y[i] + sa * coma_r_gm;
                out.push(OrbitVertex {
                    pos: [x0 as f32, y0 as f32],
                    color: [220, 240, 255, 80],
                });
                out.push(OrbitVertex {
                    pos: [x1 as f32, y1 as f32],
                    color: [220, 240, 255, 0],
                });
            }
        }
    }
    out
}

struct SimState {
    paused: bool,
    speed: f64,
    cam: [f64; 2],
    zoom: f32,
    /// Per-system body count. For `Preset::Random` this is what the user set;
    /// for `Preset::SolarSystem` this is the actual Solar preset count
    /// (Sun + 13 planets + ~200 belt bodies), taken from `b.len()` after spawn.
    body_count: usize,
    seed: u32,
    fps_timer: Instant,
    fps_frames: u32,
    fps: f32,
    steps_at_fps: u64,
    steps_per_sec: f32,
    dragging: bool,
    cursor: [f32; 2],
    press_pos: [f32; 2],
    moved_px: f32,
    show_orbits: bool,
    show_debris_orbits: bool,
    orbit_buf_valid: bool,
    last_orbit_update: Instant,
    last_orbit_zoom: f32,
    last_orbit_cam: [f64; 2],
    last_orbit_events: u64,
    orbit_posted: u64,
    orbit_applied: u64,
    selected: Option<u64>,
    follow: bool,
    preset: Preset,
    systems: usize,
    vsync: bool,
    scene_dirty: bool,
    t_window_start: Instant,
    t_window_frames: u32,
    t_frame_accum: f64,
    trail: TrailState,
    evolution_rate: f64,
    sn_kicks: bool,
    mass_transfer: bool,
    /// Slug of the library preset that `R` reloads. `None`: random seed.
    last_loaded_preset: Option<String>,
}

impl SimState {
    fn new() -> Self {
        Self {
            paused: false,
            speed: 1.0,
            cam: [0.0, 0.0],
            zoom: 0.7,
            body_count: 9,
            seed: 12345,
            fps_timer: Instant::now(),
            fps_frames: 0,
            fps: 0.0,
            steps_at_fps: 0,
            steps_per_sec: 0.0,
            dragging: false,
            cursor: [0.0; 2],
            press_pos: [0.0; 2],
            moved_px: 0.0,
            show_orbits: true,
            show_debris_orbits: false,
            orbit_buf_valid: false,
            last_orbit_update: Instant::now(),
            last_orbit_zoom: 0.7,
            last_orbit_cam: [0.0; 2],
            last_orbit_events: 0,
            orbit_posted: 0,
            orbit_applied: 0,
            selected: None,
            follow: false,
            preset: Preset::SolarSystem,
            systems: 1,
            vsync: true,
            scene_dirty: true,
            t_window_start: Instant::now(),
            t_window_frames: 0,
            t_frame_accum: 0.0,
            trail: TrailState::new(),
            evolution_rate: 0.0,
            sn_kicks: true,
            mass_transfer: true,
            last_loaded_preset: None,
        }
    }

    /// Respawn the system for the current preset.
    ///
    /// `count` is the *per-system* body count and is only used by `Random`;
    /// the Solar preset takes the body count from the spawn (13 planets +
    /// ~200 belt bodies, not the old hard-coded 9). `seed` is used by both:
    /// `Random` uses it as the RNG seed, `Solar` as the planet-phase seed.
    fn reset_with(&mut self, b: &mut Bodies, shared: &Shared, count: usize, seed: u32) {
        self.seed = seed;
        match self.preset {
            Preset::Random => {
                self.body_count = count.max(1);
                let n = self.systems.clamp(1, physics::MAX_SOLAR_SYSTEMS);
                physics::spawn_random_systems(b, self.body_count, seed, n);
            }
            Preset::SolarSystem => {
                // Solar System is a preset: always single-system now.
                // `self.systems` is left untouched so switching back to
                // Random restores the user's previous count.
                physics::spawn_solar_system_seeded(b, seed);
                self.body_count = b.len();
            }
        }
        physics::set_evolution_rate(b, self.evolution_rate);
        b.sn_kicks = self.sn_kicks;
        b.mass_transfer = self.mass_transfer;

        self.orbit_buf_valid = false;
        self.scene_dirty = true;
        self.selected = None;
        self.follow = false;
        shared.clear_accum.store(true, Ordering::Release);
        self.trail.clear();
    }

    fn reset(&mut self, b: &mut Bodies, shared: &Shared) {
        // A library preset reloads with fresh planet phases.
        if let Some(slug) = self.last_loaded_preset.clone() {
            if let Some(preset) = physics::load_preset_by_slug(&slug) {
                self.seed = self.seed.wrapping_add(1);
                physics::spawn_preset_with_seed(b, &preset, self.seed);
                physics::set_evolution_rate(b, self.evolution_rate);
                b.sn_kicks = self.sn_kicks;
                b.mass_transfer = self.mass_transfer;
                self.body_count = b.len();
                self.orbit_buf_valid = false;
                self.scene_dirty = true;
                self.selected = None;
                self.follow = false;
                shared.clear_accum.store(true, Ordering::Release);
                self.trail.clear();
                return;
            }
        }
        let seed = self.seed.wrapping_add(1);
        let count = self.body_count;
        self.reset_with(b, shared, count, seed);
    }

    fn screen_to_world(&self, p: [f32; 2], vp: [f32; 2]) -> [f64; 2] {
        [
            self.cam[0] + (p[0] as f64 - vp[0] as f64 * 0.5) / self.zoom as f64,
            self.cam[1] + (p[1] as f64 - vp[1] as f64 * 0.5) / self.zoom as f64,
        ]
    }

    fn zoom_at(&mut self, factor: f32, p: [f32; 2], vp: [f32; 2]) {
        let before = self.screen_to_world(p, vp);
        self.zoom = (self.zoom * factor).clamp(ZOOM_MIN, ZOOM_MAX);
        self.cam[0] = before[0] - (p[0] as f64 - vp[0] as f64 * 0.5) / self.zoom as f64;
        self.cam[1] = before[1] - (p[1] as f64 - vp[1] as f64 * 0.5) / self.zoom as f64;
        self.orbit_buf_valid = false;
    }

    fn fit(&mut self, b: &Bodies, vp: [f32; 2]) {
        // Multi-system ring: only Random, and not while a library preset is
        // active. Presets are always single-system.
        if self.preset == Preset::Random && self.last_loaded_preset.is_none() && self.systems > 1 {
            let mut max_r2: f64 = 0.0;
            for i in 0..b.len() {
                let r2 = b.x[i] * b.x[i] + b.y[i] * b.y[i];
                if r2 > max_r2 {
                    max_r2 = r2;
                }
            }
            let radius = (max_r2.sqrt() * 1.15).max(50.0);
            self.cam = [0.0, 0.0];
            let half = 0.5 * vp[0].min(vp[1]) as f64;
            self.zoom = ((half / radius) as f32).clamp(ZOOM_MIN, ZOOM_MAX);
            self.orbit_buf_valid = false;
            return;
        }

        // Single Solar System preset: frame the whole extent (Neptune at
        // ~30 AU ≈ 4500 Gm), matching "fit the whole system".
        if self.preset == Preset::SolarSystem && self.last_loaded_preset.is_none() {
            let radius = physics::solar_extent_gm(1);
            let half = 0.5 * vp[0].min(vp[1]) as f64;
            self.cam = [0.0, 0.0];
            self.zoom = ((half / radius) as f32).clamp(ZOOM_MIN, ZOOM_MAX);
            self.orbit_buf_valid = false;
            return;
        }

        // Single Random, or a library preset: fit to body distances from the
        // primary.
        let Some(prim) = physics::primary_index(b) else {
            return;
        };
        let mut d: Vec<f64> = (0..b.len())
            .filter(|&i| i != prim)
            .map(|i| ((b.x[i] - b.x[prim]).powi(2) + (b.y[i] - b.y[prim]).powi(2)).sqrt())
            .collect();
        let radius = if d.is_empty() {
            50.0
        } else {
            d.sort_by(|a, c| a.partial_cmp(c).unwrap_or(std::cmp::Ordering::Equal));
            let k = ((d.len() as f64 * 0.92) as usize).min(d.len() - 1);
            (d[k] * 1.15).max(5.0)
        };
        self.cam = [b.x[prim], b.y[prim]];
        let half = 0.5 * vp[0].min(vp[1]) as f64;
        self.zoom = ((half / radius) as f32).clamp(ZOOM_MIN, ZOOM_MAX);
        self.orbit_buf_valid = false;
    }

    fn pick(&self, b: &Bodies, p: [f32; 2], vp: [f32; 2]) -> Option<u64> {
        let mut best: Option<(f64, usize)> = None;
        for i in 0..b.len() {
            let sx = vp[0] as f64 * 0.5 + (b.x[i] - self.cam[0]) * self.zoom as f64;
            let sy = vp[1] as f64 * 0.5 + (b.y[i] - self.cam[1]) * self.zoom as f64;
            let dx = sx - p[0] as f64;
            let dy = sy - p[1] as f64;
            let d = (dx * dx + dy * dy).sqrt();
            let rpx = (b.r[i] * self.zoom as f64).max(8.0);
            if d <= rpx && best.is_none_or(|(s, _)| d < s) {
                best = Some((d, i));
            }
        }
        best.map(|(_, i)| b.id[i])
    }

    fn cycle_selection(&mut self, b: &Bodies, dir: i32) {
        let prim = physics::primary_index(b);
        let mut list: Vec<(f64, u64)> = (0..b.len())
            .filter(|&i| {
                matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
                    && b.origin[i] == BodyOrigin::Spawned
            })
            .map(|i| {
                let d = match prim {
                    Some(p) => ((b.x[i] - b.x[p]).powi(2) + (b.y[i] - b.y[p]).powi(2)).sqrt(),
                    None => 0.0,
                };
                (d, b.id[i])
            })
            .collect();
        if list.is_empty() {
            return;
        }
        list.sort_by(|a, c| a.0.partial_cmp(&c.0).unwrap_or(std::cmp::Ordering::Equal));
        let cur = self
            .selected
            .and_then(|id| list.iter().position(|e| e.1 == id));
        let n = list.len() as i32;
        let next = match cur {
            Some(k) => (k as i32 + dir).rem_euclid(n),
            None => {
                if dir >= 0 {
                    0
                } else {
                    n - 1
                }
            }
        };
        self.selected = Some(list[next as usize].1);
    }

    fn apply_actions(
        &mut self,
        act: Actions,
        ui_state: &mut UiState,
        vp: [f32; 2],
        b: &mut Bodies,
        shared: &Shared,
    ) {
        if act.edit.is_some()
            || act.reset
            || act.step_once
            || act.preset.is_some()
            || act.body_count.is_some()
            || act.systems.is_some()
        {
            self.scene_dirty = true;
        }

        // Systems slider: respawn immediately. Works for both presets, so
        // the user can go from 1 random system to N random systems in one
        // click without touching the preset selector.
        if let Some(n) = act.systems {
            // Only applies when Random is active (no library preset).
            if self.preset == Preset::Random && self.last_loaded_preset.is_none() {
                self.systems = n.clamp(1, physics::MAX_SOLAR_SYSTEMS);
                let count = self.body_count;
                let seed = self.seed;
                self.reset_with(b, shared, count, seed);
                self.fit(b, vp);
            }
        }

        if let Some(slug) = act.load_preset {
            if let Some(preset) = physics::load_preset_by_slug(&slug) {
                physics::spawn_from_preset(b, &preset);
                physics::set_evolution_rate(b, self.evolution_rate);
                b.sn_kicks = self.sn_kicks;
                b.mass_transfer = self.mass_transfer;
                self.body_count = b.len();
                // The preset is one system; the mode stays `Random`.
                self.preset = Preset::Random;
                self.last_loaded_preset = Some(slug.clone());
                ui_state.loaded_preset_slug = Some(slug);
                ui_state.pending_count = b.len();
                self.fit(b, vp);
                self.orbit_buf_valid = false;
                self.selected = None;
                self.follow = false;
                shared.clear_accum.store(true, Ordering::Release);
                self.trail.clear();
                self.scene_dirty = true;
            }
        }

        if let Some(p) = act.preset {
            self.preset = p;
            self.last_loaded_preset = None;
            ui_state.loaded_preset_slug = None;
            match p {
                Preset::Random => self.reset_with(
                    b,
                    shared,
                    ui_state.pending_count,
                    ui_state.pending_seed(self.seed),
                ),
                Preset::SolarSystem => {
                    let seed = self.seed;
                    self.reset_with(b, shared, 9, seed);
                }
            }
            self.fit(b, vp);
        } else if let (Some(n), Some(seed)) = (act.body_count, act.seed) {
            self.reset_with(b, shared, n, seed);
            self.fit(b, vp);
        }
        if act.reset {
            self.reset(b, shared);
            self.fit(b, vp); // ← added
            ui_state.set_pending_seed(self.seed);
        }
        if act.toggle_pause {
            self.paused = !self.paused;
        }
        if act.step_once {
            physics::advance(b, physics::DT);
        }
        if act.spawn_iso {
            if physics::spawn_interstellar_object(b).is_some() {
                self.orbit_buf_valid = false;
                self.scene_dirty = true;
            }
        }
        if act.spawn_rogue {
            if physics::spawn_rogue_planet(b).is_some() {
                self.orbit_buf_valid = false;
                self.scene_dirty = true;
            }
        }
        if let Some(s) = act.speed {
            self.speed = s.clamp(0.01, 100.0);
        }
        if let Some(v) = act.vsync {
            self.vsync = v;
        }
        if let Some(o) = act.orbits {
            self.show_orbits = o;
            self.orbit_buf_valid = false;
        }
        if let Some(o) = act.debris_orbits {
            self.show_debris_orbits = o;
            self.orbit_buf_valid = false;
        }
        if let Some(f) = act.zoom_mul {
            self.zoom_at(f, [vp[0] * 0.5, vp[1] * 0.5], vp);
        }
        if act.fit {
            self.fit(b, vp);
        }
        if let Some(sel) = act.select {
            self.selected = sel;
            if sel.is_none() {
                self.follow = false;
            }
        }
        if let Some(c) = act.center_on {
            self.cam = c;
            self.follow = false;
        }
        if let Some(f) = act.follow {
            self.follow = f && self.selected.is_some();
        }
        if act.cycle != 0 {
            self.cycle_selection(b, act.cycle);
        }
        if let Some(edit) = act.edit {
            if let Some(i) = self.selected.and_then(|id| b.index_of_id(id)) {
                match edit {
                    Edit::MassMul(f) => physics::scale_mass(b, i, f),
                    Edit::SetKind(k) => physics::set_kind(b, i, k),
                    Edit::Circularize => {
                        physics::circularize(b, i);
                    }
                    Edit::Delete => {
                        physics::delete_body(b, i);
                        self.selected = None;
                        self.follow = false;
                    }
                    Edit::SpawnMoon => {
                        let mut rng = std::mem::replace(&mut b.rng, physics::XorShift::new(1));
                        physics::spawn_moon(b, i, &mut rng);
                        b.rng = rng;
                        self.orbit_buf_valid = false;
                        self.scene_dirty = true;
                    }
                    Edit::AddRing => {
                        let mut rng = std::mem::replace(&mut b.rng, physics::XorShift::new(1));
                        physics::add_ring(b, i, 25, &mut rng);
                        b.rng = rng;
                        self.orbit_buf_valid = false;
                        self.scene_dirty = true;
                    }
                }
                self.orbit_buf_valid = false;
            }
        }
        if let Some(t) = act.trails {
            self.trail.cfg.enabled = t;
        }
        if let Some(v) = act.diag_visible {
            ui_state.diag.visible = v;
        }
        if let Some(r) = act.evolution_rate {
            physics::set_evolution_rate(b, r);
            self.evolution_rate = r;
        }
        if let Some(k) = act.sn_kicks {
            b.sn_kicks = k;
            self.sn_kicks = k;
        }
        if let Some(m) = act.mass_transfer {
            b.mass_transfer = m;
            self.mass_transfer = m;
        }
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

struct App {
    gfx: Option<GfxState>,
    sim: SimState,
    ui: UiState,
    shared: Arc<Shared>,
    phys: Option<JoinHandle<()>>,
    idle_frames: u32,
    timing_log: bool,
}

impl App {
    fn new() -> Self {
        let mut bodies = Bodies::with_capacity(64);
        physics::spawn_solar_system(&mut bodies);
        let mut sim = SimState::new();
        // The default Solar preset contains 13 planets + ~200 belt bodies,
        // not 9; keep `body_count` in sync with the spawn.
        sim.body_count = bodies.len();
        let ui = UiState::new(sim.body_count, sim.seed);
        let shared = Arc::new(Shared::new(bodies, sim.paused, sim.speed));
        Self {
            gfx: None,
            sim,
            ui,
            shared,
            phys: None,
            idle_frames: 0,
            timing_log: std::env::var_os("NBODY_TIMING").is_some(),
        }
    }

    fn shutdown(&mut self) {
        self.shared.quit.store(true, Ordering::Relaxed);
        if let Some(h) = self.phys.take() {
            let _ = h.join();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("N-Body Rust")
                        .with_inner_size(PhysicalSize::new(1600, 1000)),
                )
                .expect("failed to create window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .expect("device");

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes[0]
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: if self.sim.vsync {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            },
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let renderer = Renderer::new(&device, format);

        let egui_ctx = egui::Context::default();
        ui::setup_style(&egui_ctx);
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

        self.gfx = Some(GfxState {
            renderer,
            egui_renderer,
            egui_state,
            egui_ctx,
            surface,
            config,
            size,
            device,
            queue,
            window,
        });
        let vp = [size.width as f32, size.height as f32];
        {
            let b = self.shared.lock_for_render();
            self.sim.fit(&b, vp);
        }
        self.idle_frames = 0;

        if self.phys.is_none() {
            let sh = self.shared.clone();
            self.phys = std::thread::Builder::new()
                .name("physics".into())
                .spawn(move || physics_loop(sh))
                .ok();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gfx) = self.gfx.as_mut() else {
            return;
        };

        if !matches!(event, WindowEvent::RedrawRequested) {
            self.idle_frames = 0;
        }

        let response = gfx.egui_state.on_window_event(&gfx.window, &event);
        let vp = [gfx.config.width as f32, gfx.config.height as f32];

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(new_size) => {
                gfx.size = new_size;
                gfx.config.width = new_size.width.max(1);
                gfx.config.height = new_size.height.max(1);
                gfx.surface.configure(&gfx.device, &gfx.config);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed || gfx.egui_ctx.wants_keyboard_input() {
                    return;
                }
                let pan = 40.0 / self.sim.zoom as f64;
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Space) => self.sim.paused = !self.sim.paused,
                    Key::Named(NamedKey::ArrowUp) => self.sim.cam[1] -= pan,
                    Key::Named(NamedKey::ArrowDown) => self.sim.cam[1] += pan,
                    Key::Named(NamedKey::ArrowLeft) => self.sim.cam[0] -= pan,
                    Key::Named(NamedKey::ArrowRight) => self.sim.cam[0] += pan,
                    Key::Named(NamedKey::Escape) => {
                        self.sim.selected = None;
                        self.sim.follow = false;
                    }
                    Key::Named(NamedKey::Delete) | Key::Named(NamedKey::Backspace) => {
                        let mut b = self.shared.lock_for_render();
                        if let Some(i) = self.sim.selected.and_then(|id| b.index_of_id(id)) {
                            physics::delete_body(&mut b, i);
                            self.sim.selected = None;
                            self.sim.follow = false;
                            self.sim.orbit_buf_valid = false;
                            self.sim.scene_dirty = true;
                        }
                    }
                    Key::Character(c) => match c.to_lowercase().as_str() {
                        "r" => {
                            let mut b = self.shared.lock_for_render();
                            self.sim.reset(&mut b, &self.shared);
                            self.sim.fit(&b, vp); // ← added
                            self.ui.set_pending_seed(self.sim.seed);
                        }
                        "+" | "=" => self.sim.zoom_at(1.25, [vp[0] * 0.5, vp[1] * 0.5], vp),
                        "-" | "_" => self.sim.zoom_at(0.8, [vp[0] * 0.5, vp[1] * 0.5], vp),
                        "w" => self.sim.cam[1] -= pan,
                        "s" => self.sim.cam[1] += pan,
                        "a" => self.sim.cam[0] -= pan,
                        "d" => self.sim.cam[0] += pan,
                        "`" => self.ui.show_dock = !self.ui.show_dock,
                        "o" => {
                            self.sim.show_orbits = !self.sim.show_orbits;
                            self.sim.orbit_buf_valid = false;
                        }
                        ";" => {
                            self.sim.show_debris_orbits = !self.sim.show_debris_orbits;
                            self.sim.orbit_buf_valid = false;
                        }
                        "l" => self.ui.show_labels = !self.ui.show_labels,
                        "u" => self.ui.units_si = !self.ui.units_si,
                        "v" => self.sim.vsync = !self.sim.vsync,
                        "b" => self.ui.show_list = !self.ui.show_list,
                        "h" | "?" | "/" => self.ui.show_help = !self.ui.show_help,
                        "f" | "0" => {
                            let b = self.shared.lock_for_render();
                            self.sim.fit(&b, vp);
                        }
                        "g" => self.ui.diag.visible = !self.ui.diag.visible,
                        "c" => {
                            if self.sim.selected.is_some() {
                                self.sim.follow = !self.sim.follow;
                            }
                        }
                        "n" => {
                            let b = self.shared.lock_for_render();
                            self.sim.cycle_selection(&b, 1);
                        }
                        "p" => {
                            let b = self.shared.lock_for_render();
                            self.sim.cycle_selection(&b, -1);
                        }
                        "i" => {
                            let mut b = self.shared.lock_for_render();
                            if physics::spawn_interstellar_object(&mut b).is_some() {
                                self.sim.orbit_buf_valid = false;
                                self.sim.scene_dirty = true;
                            }
                        }
                        "t" => {
                            let mut b = self.shared.lock_for_render();
                            if physics::spawn_rogue_planet(&mut b).is_some() {
                                self.sim.orbit_buf_valid = false;
                                self.sim.scene_dirty = true;
                            }
                        }
                        "." => {
                            let mut b = self.shared.lock_for_render();
                            physics::advance(&mut b, physics::DT);
                            self.sim.scene_dirty = true;
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }

            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                if state == ElementState::Pressed {
                    if response.consumed || gfx.egui_ctx.is_pointer_over_area() {
                        return;
                    }
                    self.sim.dragging = true;
                    self.sim.press_pos = self.sim.cursor;
                    self.sim.moved_px = 0.0;
                } else if self.sim.dragging {
                    self.sim.dragging = false;
                    if self.sim.moved_px < CLICK_SLOP_PX {
                        let hit = {
                            let b = self.shared.lock_for_render();
                            self.sim.pick(&b, self.sim.cursor, vp)
                        };
                        self.sim.selected = hit;
                        if hit.is_none() {
                            self.sim.follow = false;
                        }
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let pos = [position.x as f32, position.y as f32];
                if self.sim.dragging {
                    let dx = pos[0] - self.sim.cursor[0];
                    let dy = pos[1] - self.sim.cursor[1];
                    self.sim.moved_px += dx.abs() + dy.abs();
                    if self.sim.moved_px >= CLICK_SLOP_PX {
                        self.sim.cam[0] -= dx as f64 / self.sim.zoom as f64;
                        self.sim.cam[1] -= dy as f64 / self.sim.zoom as f64;
                        self.sim.follow = false;
                    }
                }
                self.sim.cursor = pos;
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if response.consumed || gfx.egui_ctx.is_pointer_over_area() {
                    return;
                }
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 30.0,
                };
                let cursor = self.sim.cursor;
                self.sim.zoom_at((dy * 0.15).exp(), cursor, vp);
            }

            WindowEvent::RedrawRequested => {
                if gfx.size.width == 0 || gfx.size.height == 0 {
                    self.idle_frames = self.idle_frames.saturating_add(1);
                    return;
                }
                let t_frame = Instant::now();

                let want_mode = if self.sim.vsync {
                    wgpu::PresentMode::AutoVsync
                } else {
                    wgpu::PresentMode::AutoNoVsync
                };
                if gfx.config.present_mode != want_mode {
                    gfx.config.present_mode = want_mode;
                    gfx.surface.configure(&gfx.device, &gfx.config);
                }

                let frame = match gfx.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        gfx.surface.configure(&gfx.device, &gfx.config);
                        return;
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        event_loop.exit();
                        return;
                    }
                    Err(_) => return,
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut actions = Actions::default();
                let raw_input = gfx.egui_state.take_egui_input(&gfx.window);
                let full_output;
                let flashes_active;
                let n_bodies;
                let mut new_title: Option<String> = None;
                let trail_snapshot: Vec<TrailSample>;
                let comet_tail_verts: Vec<OrbitVertex>;
                // Orbit vertices arrive from the physics thread. Hoisted
                // out of the lock block so `set_orbit_vertices` (a pure
                // GPU upload) runs after the lock drops.
                #[allow(unused_assignments)]
                let mut orbit_out_taken: Option<(u64, Vec<OrbitVertex>)> = None;

                {
                    let mut b = self.shared.lock_for_render();

                    self.sim.fps_frames += 1;
                    let win = self.sim.fps_timer.elapsed().as_secs_f32();
                    if win >= 0.5 {
                        self.sim.fps = self.sim.fps_frames as f32 / win;
                        let steps = b.stats.steps;
                        self.sim.steps_per_sec =
                            (steps.saturating_sub(self.sim.steps_at_fps)) as f32 / win;
                        self.sim.steps_at_fps = steps;
                        self.sim.fps_frames = 0;
                        self.sim.fps_timer = Instant::now();
                        new_title = Some(format!(
                            "N-Body Rust — {} — {:.0} fps — {:.2}×",
                            primary_label(&b),
                            self.sim.fps,
                            self.sim.speed
                        ));
                    }

                    // One O(n) lookup shared by the follow lerp and the
                    // selection-cleanup check.
                    let selected_idx = self.sim.selected.and_then(|id| b.index_of_id(id));
                    if self.sim.follow {
                        match selected_idx {
                            Some(i) => {
                                self.sim.cam[0] += (b.x[i] - self.sim.cam[0]) * 0.2;
                                self.sim.cam[1] += (b.y[i] - self.sim.cam[1]) * 0.2;
                            }
                            None => self.sim.follow = false,
                        }
                    }
                    if self.sim.selected.is_some() && selected_idx.is_none() {
                        self.sim.selected = None;
                        self.sim.follow = false;
                    }

                    self.ui.diag.maybe_sample(&b);

                    let effective_systems = if self.sim.preset == Preset::Random
                        && self.sim.last_loaded_preset.is_none()
                    {
                        self.sim.systems
                    } else {
                        1
                    };
                    let view_info = ViewInfo {
                        paused: self.sim.paused,
                        speed: self.sim.speed,
                        fps: self.sim.fps,
                        steps_per_sec: self.sim.steps_per_sec,
                        zoom: self.sim.zoom,
                        cam: self.sim.cam,
                        viewport_px: vp,
                        ppp: gfx.egui_ctx.pixels_per_point(),
                        seed: self.sim.seed,
                        body_count: self.sim.body_count,
                        systems: effective_systems,
                        total_bodies: b.len(),
                        show_orbits: self.sim.show_orbits,
                        show_debris_orbits: self.sim.show_debris_orbits,
                        follow: self.sim.follow,
                        selected: self.sim.selected,
                        preset: self.sim.preset,
                        vsync: self.sim.vsync,
                        behind: self.shared.behind.load(Ordering::Relaxed),
                        evolution_rate: self.sim.evolution_rate,
                        sn_kicks: self.sim.sn_kicks,
                        mass_transfer: self.sim.mass_transfer,
                    };
                    full_output = gfx.egui_ctx.run(raw_input, |ctx| {
                        actions = ui::draw(ctx, &mut self.ui, &view_info, &b);
                    });
                    self.sim
                        .apply_actions(actions, &mut self.ui, vp, &mut b, &self.shared);
                    // Snapshot *after* apply_actions, so a reset that
                    // cleared the trail is captured from the *new* bodies
                    // (time 0) rather than the pre-reset ones. Otherwise
                    // the first push is stamped with the old sim time and
                    // the trail stays frozen until sim time reaches it.
                    trail_snapshot = self.sim.trail.snapshot(&b);

                    self.shared.paused.store(self.sim.paused, Ordering::Relaxed);
                    self.shared
                        .speed_bits
                        .store(self.sim.speed.to_bits(), Ordering::Relaxed);

                    orbit_out_taken = lock_ok(&self.shared.orbit_out).take();
                    let events_total = b
                        .stats
                        .merges
                        .wrapping_add(b.stats.impacts)
                        .wrapping_add(b.stats.tidal)
                        .wrapping_add(b.stats.escaped);
                    let orbits_visible = self.sim.show_orbits || self.sim.show_debris_orbits;
                    let zoom_ratio = self.sim.zoom / self.sim.last_orbit_zoom.max(1e-6);
                    let zoom_changed = (zoom_ratio - 1.0).abs() > 0.3;
                    let span_x = 0.35 * vp[0] as f64 / self.sim.zoom as f64;
                    let span_y = 0.35 * vp[1] as f64 / self.sim.zoom as f64;
                    let cam_moved = (self.sim.cam[0] - self.sim.last_orbit_cam[0]).abs() > span_x
                        || (self.sim.cam[1] - self.sim.last_orbit_cam[1]).abs() > span_y;
                    let events_changed = events_total != self.sim.last_orbit_events;
                    let stale = !self.sim.paused
                        && self.sim.last_orbit_update.elapsed().as_millis() > ORBIT_REFRESH_MS;
                    if !self.sim.orbit_buf_valid
                        || (orbits_visible
                            && (zoom_changed || cam_moved || events_changed || stale))
                    {
                        self.sim.orbit_posted += 1;
                        *lock_ok(&self.shared.orbit_req) = Some((
                            self.sim.orbit_posted,
                            OrbitReq {
                                zoom: self.sim.zoom as f64,
                                planets: self.sim.show_orbits,
                                debris: self.sim.show_debris_orbits,
                                cam: self.sim.cam,
                                viewport: vp,
                            },
                        ));
                        self.sim.last_orbit_update = Instant::now();
                        self.sim.last_orbit_zoom = self.sim.zoom;
                        self.sim.last_orbit_cam = self.sim.cam;
                        self.sim.last_orbit_events = events_total;
                        self.sim.orbit_buf_valid = true;
                    }

                    gfx.renderer
                        .build_instances(&b, self.sim.zoom, self.sim.cam, vp);
                    let _ = std::mem::take(&mut self.sim.scene_dirty);
                    comet_tail_verts = build_comet_tails(&b, self.sim.cam);
                    flashes_active = !b.flashes.is_empty();
                    n_bodies = b.len();
                }

                // ---- Everything below runs WITHOUT the bodies lock. ----
                if let Some((id, verts)) = orbit_out_taken {
                    gfx.renderer
                        .set_orbit_vertices(&gfx.device, &gfx.queue, &verts);
                    self.sim.orbit_applied = id;
                }
                gfx.renderer
                    .update_camera(&gfx.queue, self.sim.cam, self.sim.zoom, vp);
                gfx.renderer.upload_instances(&gfx.device, &gfx.queue);
                gfx.renderer
                    .set_comet_tail_vertices(&gfx.device, &gfx.queue, &comet_tail_verts);
                if self.sim.trail.cfg.enabled {
                    self.sim.trail.update(&trail_snapshot);
                    let verts = self.sim.trail.build_vertices();
                    gfx.renderer
                        .set_trail_vertices(&gfx.device, &gfx.queue, &verts);
                } else if gfx.renderer.trail_count() > 0 {
                    gfx.renderer
                        .set_trail_vertices(&gfx.device, &gfx.queue, &[]);
                }

                let repaint_now = full_output
                    .viewport_output
                    .get(&egui::ViewportId::ROOT)
                    .is_some_and(|v| v.repaint_delay.is_zero());

                gfx.egui_state
                    .handle_platform_output(&gfx.window, full_output.platform_output);
                let pixels_per_point = gfx.egui_ctx.pixels_per_point();
                let clipped = gfx
                    .egui_ctx
                    .tessellate(full_output.shapes, pixels_per_point);

                for (id, delta) in &full_output.textures_delta.set {
                    gfx.egui_renderer
                        .update_texture(&gfx.device, &gfx.queue, *id, delta);
                }

                let screen_desc = ScreenDescriptor {
                    size_in_pixels: [gfx.config.width, gfx.config.height],
                    pixels_per_point,
                };
                let mut encoder = gfx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                gfx.egui_renderer.update_buffers(
                    &gfx.device,
                    &gfx.queue,
                    &mut encoder,
                    &clipped,
                    &screen_desc,
                );

                {
                    let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("main-pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    let mut pass = pass.forget_lifetime();
                    gfx.renderer.draw(&mut pass);
                    gfx.egui_renderer.render(&mut pass, &clipped, &screen_desc);
                }

                gfx.queue.submit([encoder.finish()]);
                frame.present();

                for id in &full_output.textures_delta.free {
                    gfx.egui_renderer.free_texture(id);
                }

                if self.timing_log {
                    self.sim.t_frame_accum += t_frame.elapsed().as_secs_f64() * 1e6;
                    self.sim.t_window_frames += 1;
                    let secs = self.sim.t_window_start.elapsed().as_secs_f64();
                    if secs >= 1.0 {
                        let frames = self.sim.t_window_frames as f64;
                        let phys_busy = self.shared.phys_ns.swap(0, Ordering::Relaxed) as f64
                            / (secs * 1e9)
                            * 100.0;
                        println!(
                            "[timing] n={} fps={:.0} frame={:.2}ms physics_busy={:.0}%{}",
                            n_bodies,
                            frames / secs,
                            self.sim.t_frame_accum / frames / 1000.0,
                            phys_busy,
                            if self.shared.behind.load(Ordering::Relaxed) {
                                " (behind)"
                            } else {
                                ""
                            }
                        );
                        self.sim.t_window_start = Instant::now();
                        self.sim.t_window_frames = 0;
                        self.sim.t_frame_accum = 0.0;
                    }
                }
                if let Some(t) = new_title {
                    gfx.window.set_title(&t);
                }

                let busy = !self.sim.paused
                    || flashes_active
                    || self.sim.follow
                    || repaint_now
                    || self.sim.orbit_posted > self.sim.orbit_applied;
                self.idle_frames = if busy {
                    0
                } else {
                    self.idle_frames.saturating_add(1)
                };
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if self.idle_frames < IDLE_FRAMES {
            if let Some(g) = &self.gfx {
                g.window.request_redraw();
            }
        }
    }
}

fn primary_label(b: &physics::Bodies) -> String {
    use physics::BodyKind;
    let Some(i) = physics::primary_index(b) else {
        return "empty".into();
    };
    match b.kind[i] {
        BodyKind::Star => format!(
            "{}-type star + {} bodies",
            physics::star_class(b.m[i]),
            b.len() - 1
        ),
        BodyKind::BrownDwarf => {
            // Match `render::body_color` and `physics::class_name`: classify
            // by surface temperature, not by mass, so title/colour/panel agree.
            let t = physics::bd_state(b.m[i], b.age_gyr[i]).temperature;
            format!(
                "{}-type brown dwarf + {} bodies",
                physics::bd_class_for_temp(t),
                b.len() - 1
            )
        }
        BodyKind::WhiteDwarf => format!("white dwarf + {} bodies", b.len() - 1),
        BodyKind::NeutronStar => format!("neutron star + {} bodies", b.len() - 1),
        BodyKind::BlackHole => format!("black hole + {} bodies", b.len() - 1),
        _ => format!("{} bodies", b.len()),
    }
}

fn main() {
    let nthreads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(num_cpus::get_physical);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build_global();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run");
    app.shutdown();
}
