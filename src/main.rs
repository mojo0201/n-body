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

use nbody_rust::physics::{self, Bodies, BodyKind, BodyOrigin};
use nbody_rust::render::{self, OrbitReq, OrbitVertex, Renderer};
use nbody_rust::ui::{self, Actions, Edit, UiState, ViewInfo};
use nbody_rust::Preset;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{Window, WindowId};

use egui_wgpu::ScreenDescriptor;

// Zoom range for the view: 1e-7 .. 1e5 px per Gm.
/// Seconds in one Gyr (Julian years). Used to age stars in real time.
const SECONDS_PER_GYR: f64 = 3.15576e16;

const ZOOM_MIN: f32 = 1e-7;
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

/// Create the 4× MSAA offscreen color target used by the main render pass.
/// `format` must match the surface format.
fn create_msaa(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("msaa-color"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 4,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

// ---------------------------------------------------------------------------
// Graphics and view state
// ---------------------------------------------------------------------------

struct GfxState {
    // --- Pipelines, buffers, bind groups, egui. All hold Device refs. ---
    renderer: Renderer,
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
    egui_ctx: egui::Context,

    // --- MSAA offscreen target. Must drop before `device`. ---
    msaa_tex: wgpu::Texture,
    msaa_view: wgpu::TextureView,

    // --- Surface and its config. Must drop before `window`. ---
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,

    // --- Device & queue. Everything above has already released by now. ---
    device: wgpu::Device,
    queue: wgpu::Queue,

    // --- Window LAST. Nothing may hold a Window ref after this drops. ---
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

/// (id, sim_time, x, y, rgba, period, parent_pos). Positions stay f64 so
/// deep zoom does not jitter; they become camera-relative f32 only in
/// `build_vertices`.
type TrailSample = (u64, f64, f64, f64, [u8; 4], f64, Option<(f64, f64)>);

struct Trail {
    history: VecDeque<(f64, f64, f64)>,
    color: [u8; 4],
}

const TRAIL_MAX_POINTS: usize = 4000;
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
    fn snapshot(&self, b: &Bodies, override_pos: Option<(u64, [f64; 2])>) -> Vec<TrailSample> {
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
                    (o.period, Some((b.x[p], b.y[p])))
                })
                .unwrap_or((f64::INFINITY, None));
            let (bx, by) = match override_pos {
                Some((id, pos)) if b.id[i] == id => (pos[0], pos[1]),
                _ => (b.x[i], b.y[i]),
            };
            out.push((b.id[i], b.time, bx, by, color, period, parent_pos));
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
                    let steps = if let Some((px, py)) = parent_pos {
                        let dx0 = x_last - px;
                        let dy0 = y_last - py;
                        let dx1 = x - px;
                        let dy1 = y - py;
                        let r0 = (dx0 * dx0 + dy0 * dy0).sqrt();
                        let r1 = (dx1 * dx1 + dy1 * dy1).sqrt();
                        if r0 > 1e-6 && r1 > 1e-6 {
                            let a0 = dy0.atan2(dx0);
                            let a1 = dy1.atan2(dx1);
                            let mut da = a1 - a0;
                            if da > std::f64::consts::PI {
                                da -= 2.0 * std::f64::consts::PI;
                            } else if da < -std::f64::consts::PI {
                                da += 2.0 * std::f64::consts::PI;
                            }
                            ((da.abs() / 0.035).ceil() as usize).clamp(1, 256)
                        } else {
                            let d = ((x - x_last).powi(2) + (y - y_last).powi(2)).sqrt();
                            ((d / 0.01).ceil() as usize).clamp(1, 256)
                        }
                    } else {
                        let d = ((x - x_last).powi(2) + (y - y_last).powi(2)).sqrt();
                        ((d / 0.01).ceil() as usize).clamp(1, 256)
                    };

                    if steps > 1 {
                        for s in 1..steps {
                            let frac = s as f64 / steps as f64;
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
                                    if da > std::f64::consts::PI {
                                        da -= 2.0 * std::f64::consts::PI;
                                    } else if da < -std::f64::consts::PI {
                                        da += 2.0 * std::f64::consts::PI;
                                    }
                                    let r = r0 + (r1 - r0) * frac;
                                    let a = a0 + da * frac;
                                    (px + r * a.cos(), py + r * a.sin())
                                }
                            } else {
                                // Fallback to linear interpolation (stars)
                                (x_last + (x - x_last) * frac, y_last + (y - y_last) * frac)
                            };
                            trail.history.push_back((t_last + dt * frac, ix, iy));
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
    /// line-list vertices with alpha fading toward the tail. Positions are
    /// made relative to `cam` in f64 before the cast to f32, and the renderer
    /// draws trails with the relative camera. Trails are visited in id order
    /// and, when the vertex budget is exceeded, every trail is decimated by
    /// the same stride, so the result does not depend on hash order.
    fn build_vertices(&self, cam: [f64; 2]) -> Vec<OrbitVertex> {
        let mut ids: Vec<u64> = self.history.keys().copied().collect();
        ids.sort_unstable();
        let total_segs: usize = self
            .history
            .values()
            .map(|t| t.history.len().saturating_sub(1))
            .sum();
        let stride = (total_segs * 2).div_ceil(TRAIL_MAX_VERTS).max(1);
        let mut out = Vec::with_capacity((total_segs * 2 / stride).min(TRAIL_MAX_VERTS) + 2);
        'outer: for id in ids {
            let trail = &self.history[&id];
            let n = trail.history.len();
            if n < 2 {
                continue;
            }
            let t_last = trail.history[n - 1].0;
            let t_first = trail.history[0].0;

            // Fade over the actual span of this trail, not the config max_age.
            let span = (t_last - t_first).max(1e-9);
            let rgb = [trail.color[0], trail.color[1], trail.color[2]];
            let alpha = |t: f64| -> u8 {
                // 30% alpha floor at the tail so the trail doesn't vanish.
                ((((1.0 - (t_last - t) / span) * 0.7 + 0.3).clamp(0.0, 1.0)) * 200.0) as u8
            };
            let mut k = 0;
            while k < n - 1 {
                let k1 = (k + stride).min(n - 1);
                let (t0, x0, y0) = trail.history[k];
                let (t1, x1, y1) = trail.history[k1];
                out.push(OrbitVertex {
                    pos: [(x0 - cam[0]) as f32, (y0 - cam[1]) as f32],
                    color: [rgb[0], rgb[1], rgb[2], alpha(t0)],
                });
                out.push(OrbitVertex {
                    pos: [(x1 - cam[0]) as f32, (y1 - cam[1]) as f32],
                    color: [rgb[0], rgb[1], rgb[2], alpha(t1)],
                });
                if out.len() >= TRAIL_MAX_VERTS {
                    break 'outer;
                }
                k = k1;
            }
        }
        out
    }
}

struct SimState {
    paused: bool,
    speed: f64,
    last_frame_time: Instant,
    follow_smooth: Option<(u64, [f64; 2])>,
    cam: [f64; 2],
    zoom: f32,
    body_count: usize,
    seed: u128,
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
    t_window_start: Instant,
    t_window_frames: u32,
    t_frame_accum: f64,
    // NBODY_TIMING breakdown (microseconds, summed over the window).
    t_wait_accum: f64,
    t_held_accum: f64,
    t_inst_accum: f64,
    t_comet_accum: f64,
    last_comets: usize,
    last_comet_verts: usize,
    trail: TrailState,
    evolution_rate: f64,
    sn_kicks: bool,
    /// Stellar aging speed relative to real time.
    aging_mult: f64,
    mass_transfer: bool,
    /// Pause when a merger, impact or tidal break-up happens.
    pause_on_event: bool,
    /// Merger + impact + tidal count at the last frame.
    last_pause_events: u64,
    /// Slug of the library preset that `R` reloads. `None`: random seed.
    last_loaded_preset: Option<String>,
    /// Disk work queued while the bodies lock was held; runs after it is released.
    deferred_io: Option<Box<dyn FnOnce(&mut UiState)>>,
    /// Name of a save to load once the bodies lock is released.
    pending_load: Option<String>,
}

impl SimState {
    fn new() -> Self {
        Self {
            paused: false,
            speed: 1.0,
            last_frame_time: Instant::now(),
            follow_smooth: None,
            cam: [0.0, 0.0],
            zoom: 0.7,
            body_count: 9,
            seed: ui::random_seed(),
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
            t_window_start: Instant::now(),
            t_window_frames: 0,
            t_frame_accum: 0.0,
            t_wait_accum: 0.0,
            t_held_accum: 0.0,
            t_inst_accum: 0.0,
            t_comet_accum: 0.0,
            last_comets: 0,
            last_comet_verts: 0,
            trail: TrailState::new(),
            // Real time: one simulation time unit ages a star by exactly the
            // time it represents (`time_unit_s` seconds), so nothing visible
            // happens to a star during a session.
            evolution_rate: physics::time_unit_s() / SECONDS_PER_GYR,
            sn_kicks: true,
            aging_mult: 1.0,
            mass_transfer: true,
            pause_on_event: false,
            last_pause_events: 0,
            last_loaded_preset: None,
            deferred_io: None,
            pending_load: None,
        }
    }

    /// Respawn the system for the current preset.
    ///
    /// `count` is the *per-system* body count and is only used by `Random`;
    /// the Solar preset takes the body count from the spawn (13 planets +
    /// ~200 belt bodies, not the old hard-coded 9). `seed` is used by both:
    /// `Random` uses it as the RNG seed, `Solar` as the planet-phase seed.
    fn reset_with(&mut self, b: &mut Bodies, shared: &Shared, count: usize, seed: u128) {
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

    /// Camera, speed and selection, for a save file.
    fn app_save(&self, body_count: usize) -> physics::AppSave {
        physics::AppSave {
            cam: self.cam,
            zoom: self.zoom,
            speed: self.speed,
            paused: self.paused,
            seed: physics::seed_to_base62(self.seed),
            body_count,
            preset: match self.preset {
                Preset::Random => "random".to_string(),
                Preset::SolarSystem => "solar".to_string(),
            },
            systems: self.systems,
            library_preset: self.last_loaded_preset.clone(),
            selected: self.selected,
            follow: self.follow,
            show_orbits: self.show_orbits,
            show_debris_orbits: self.show_debris_orbits,
            aging_mult: self.aging_mult,
        }
    }

    /// Capture the whole state for `saves/<name>.nbody.json`. Reads `b` only.
    /// The disk write is queued in `deferred_io` and runs after the bodies
    /// lock is released (see `App::flush_io`); its result goes to the UI
    /// message line.
    fn save_game(&mut self, b: &Bodies, name: &str, ui_state: &mut UiState) {
        let Some(clean) = physics::sanitize_save_name(name) else {
            ui_state.set_save_msg("The name needs a letter or a digit.", false);
            return;
        };
        let captured = physics::capture_save(b, self.app_save(b.len()));
        self.deferred_io = Some(Box::new(move |ui_state: &mut UiState| {
            let result = captured.and_then(|file| physics::write_save(&clean, &file));
            match result {
                Ok(path) => ui_state.set_save_msg(format!("Saved to {}", path.display()), true),
                Err(e) => ui_state.set_save_msg(format!("Save failed: {e}"), false),
            }
            ui_state.refresh_saves();
        }));
    }

    /// Replace the running state with `saves/<name>.nbody.json`.
    /// If the file is bad, nothing changes and the UI shows the reason.
    ///
    /// The file is read and parsed *before* the bodies lock is taken, so a slow
    /// disk never stalls the physics thread. The caller must NOT hold the lock.
    fn load_game(&mut self, name: &str, ui_state: &mut UiState, shared: &Shared) {
        // Only a plain save name may reach the file system.
        if physics::sanitize_save_name(name).as_deref() != Some(name) {
            ui_state.set_save_msg("Load failed: invalid save name.", false);
            return;
        }
        let file = match physics::read_save(name) {
            Ok(f) => f,
            Err(e) => {
                ui_state.set_save_msg(format!("Load failed: {e}"), false);
                return;
            }
        };
        let mut b = shared.lock_for_render();
        let b = &mut *b;
        if let Err(e) = physics::restore_save(b, &file) {
            ui_state.set_save_msg(format!("Load failed: {e}"), false);
            return;
        }
        let a = &file.app;
        self.cam = a.cam;
        self.zoom = a.zoom.clamp(ZOOM_MIN, ZOOM_MAX);
        self.speed = a.speed.clamp(0.01, 100.0);
        self.paused = a.paused;
        if let Some(seed) = physics::seed_from_base62(&a.seed) {
            self.seed = seed;
        }
        self.body_count = b.len();
        self.preset = if a.preset == "solar" {
            Preset::SolarSystem
        } else {
            Preset::Random
        };
        self.systems = a.systems.clamp(1, physics::MAX_SOLAR_SYSTEMS);
        self.last_loaded_preset = a.library_preset.clone();
        self.show_orbits = a.show_orbits;
        self.show_debris_orbits = a.show_debris_orbits;
        self.aging_mult = a.aging_mult.max(1.0);
        // The bodies carry the true values for these three.
        self.evolution_rate = b.evolution_rate;
        self.sn_kicks = b.sn_kicks;
        self.mass_transfer = b.mass_transfer;
        self.selected = a.selected.filter(|&id| b.index_of_id(id).is_some());
        self.follow = a.follow && self.selected.is_some();
        self.follow_smooth = None;
        // Do not fire "pause on event" for events that are already counted.
        self.last_pause_events = b
            .stats
            .merges
            .wrapping_add(b.stats.impacts)
            .wrapping_add(b.stats.tidal);
        self.orbit_buf_valid = false;
        shared.clear_accum.store(true, Ordering::Release);
        self.trail.clear();
        ui_state.reset_after_load(
            b.len(),
            self.seed,
            self.systems,
            self.last_loaded_preset.clone(),
        );
        ui_state.set_save_msg(format!("Loaded {} ({} bodies)", name, b.len()), true);
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
            // Match the exact position the renderer uses (follow smooth or raw).
            let (bx, by) = match self.follow_smooth {
                Some((id, pos)) if b.id[i] == id => (pos[0], pos[1]),
                _ => (b.x[i], b.y[i]),
            };
            let sx = vp[0] as f64 * 0.5 + (bx - self.cam[0]) * self.zoom as f64;
            let sy = vp[1] as f64 * 0.5 + (by - self.cam[1]) * self.zoom as f64;
            let dx = sx - p[0] as f64;
            let dy = sy - p[1] as f64;
            let d = (dx * dx + dy * dy).sqrt();

            // Black holes are drawn with an accretion disk 6x the event horizon.
            // The minimum pick radius must match the visual size (24px).
            let rpx = match b.kind[i] {
                BodyKind::BlackHole => (b.r[i] * self.zoom as f64).max(24.0),
                _ => (b.r[i] * self.zoom as f64).max(8.0),
            };

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
        if let Some(name) = act.save {
            self.save_game(b, &name, ui_state);
        }
        if let Some(name) = act.load {
            // Read from disk after the lock is released (see `App::flush_io`).
            self.pending_load = Some(name);
        }
        if act.toggle_pause {
            self.paused = !self.paused;
        }
        if act.step_once {
            physics::advance(b, physics::DT);
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
        if let Some([fx, fy, fr]) = act.focus {
            self.cam = [fx, fy];
            let half = 0.5 * vp[0].min(vp[1]) as f64;
            self.zoom = ((half / (fr * 1.15).max(1.0)) as f32).clamp(ZOOM_MIN, ZOOM_MAX);
            self.follow = false;
            self.follow_smooth = None;
            self.orbit_buf_valid = false;
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
                    Edit::SpawnMoon(n) => {
                        let mut rng = std::mem::replace(&mut b.rng, physics::XorShift::new(1));
                        for _ in 0..n.clamp(1, 8) {
                            physics::spawn_moon(b, i, &mut rng);
                        }
                        b.rng = rng;
                        self.orbit_buf_valid = false;
                    }
                    Edit::VelMul(f) => {
                        // Scale the speed relative to the parent (or the
                        // primary); with neither, scale the absolute speed.
                        let refb = physics::find_parent(b, i)
                            .or_else(|| physics::primary_index(b).filter(|&p| p != i));
                        let (rvx, rvy) = match refb {
                            Some(p) => (b.vx[p], b.vy[p]),
                            None => (0.0, 0.0),
                        };
                        b.vx[i] = rvx + (b.vx[i] - rvx) * f;
                        b.vy[i] = rvy + (b.vy[i] - rvy) * f;
                        b.parent_cache_valid = false;
                        self.orbit_buf_valid = false;
                    }
                    Edit::Rename(name) => {
                        let t = name.trim();
                        b.name[i] = if t.is_empty() {
                            None
                        } else {
                            Some(t.to_string())
                        };
                    }
                    Edit::AddRing(n) => {
                        let mut rng = std::mem::replace(&mut b.rng, physics::XorShift::new(1));
                        physics::add_ring(b, i, n.clamp(5, 200), &mut rng);
                        b.rng = rng;
                        self.orbit_buf_valid = false;
                    }
                }
                self.orbit_buf_valid = false;
            }
        }
        if let Some((ids, edit)) = act.multi_edit {
            let idx: Vec<usize> = ids.iter().filter_map(|&id| b.index_of_id(id)).collect();
            if !idx.is_empty() {
                // One batched call each: the O(N^2) physics commit runs once
                // for the whole selection, not once per body.
                match edit {
                    Edit::MassMul(f) => {
                        let e: Vec<(usize, f64)> = idx.iter().map(|&i| (i, f)).collect();
                        physics::scale_masses(b, &e);
                    }
                    Edit::SetKind(k) => {
                        let e: Vec<(usize, BodyKind)> = idx.iter().map(|&i| (i, k)).collect();
                        physics::set_kinds(b, &e);
                    }
                    Edit::Circularize => {
                        physics::circularize_many(b, &idx);
                    }
                    Edit::Delete => {
                        if self.selected.is_some_and(|s| ids.contains(&s)) {
                            self.selected = None;
                            self.follow = false;
                        }
                        physics::delete_bodies(b, idx);
                    }
                    Edit::SpawnMoon(_) | Edit::AddRing(_) | Edit::VelMul(_) | Edit::Rename(_) => {}
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
        if let Some(v) = act.trail_age {
            self.trail.cfg.max_age = v;
        }
        if let Some(v) = act.trail_bodies {
            self.trail.cfg.max_bodies = v.max(1);
        }
        if let Some(v) = act.trail_non_rubble {
            self.trail.cfg.non_rubble_only = v;
        }
        if let Some(v) = act.sn_kicks {
            self.sn_kicks = v;
            b.sn_kicks = v;
        }
        if let Some(v) = act.mass_transfer {
            self.mass_transfer = v;
            b.mass_transfer = v;
        }
        if let Some(v) = act.aging_mult {
            self.aging_mult = v.max(1.0);
            self.evolution_rate = physics::time_unit_s() / SECONDS_PER_GYR * self.aging_mult;
            physics::set_evolution_rate(b, self.evolution_rate);
        }
        if let Some(p) = act.pause_on_event {
            self.pause_on_event = p;
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
    /// Keyboard modifiers, so Ctrl/Alt/Cmd + letter is not a shortcut.
    mods: ModifiersState,
    /// When egui wants a repaint with no input (tooltip delay, toast and
    /// confirm-button expiry). Drives `about_to_wait` while the app is idle.
    repaint_at: Option<Instant>,
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
            mods: ModifiersState::default(),
            repaint_at: None,
        }
    }

    /// Run disk work that was queued while the bodies lock was held: the
    /// save write and the save read. Must be called with the lock released.
    fn flush_io(&mut self) {
        if let Some(io) = self.sim.deferred_io.take() {
            io(&mut self.ui);
        }
        if let Some(name) = self.sim.pending_load.take() {
            self.sim.load_game(&name, &mut self.ui, &self.shared);
            self.idle_frames = 0;
        }
    }

    fn shutdown(&mut self) {
        self.shared.quit.store(true, Ordering::Relaxed);
        if let Some(h) = self.phys.take() {
            let _ = h.join();
        }
        if let Some(gfx) = &self.gfx {
            let _ = gfx.device.poll(wgpu::PollType::wait_indefinitely());
        }
        self.gfx = None;
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

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(Box::new(
            event_loop.owned_display_handle(),
        )));
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: Default::default(),
            trace: wgpu::Trace::Off,
        }))
        .expect("device");

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        // The shaders blend and shade in gamma space and egui expects a
        // non-sRGB target, so prefer a non-sRGB surface format.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
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
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);

        let (msaa_tex, msaa_view) = create_msaa(&device, format, config.width, config.height);

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

        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            format,
            egui_wgpu::RendererOptions {
                msaa_samples: 4,
                ..Default::default()
            },
        );

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
            msaa_tex,
            msaa_view,
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
            WindowEvent::CloseRequested => {
                self.shared.quit.store(true, Ordering::Relaxed);
                if let Some(h) = self.phys.take() {
                    let _ = h.join();
                }
                if let Some(gfx) = &self.gfx {
                    let _ = gfx.device.poll(wgpu::PollType::wait_indefinitely());
                }
                self.gfx = None;
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                gfx.size = new_size;
                gfx.config.width = new_size.width.max(1);
                gfx.config.height = new_size.height.max(1);
                gfx.surface.configure(&gfx.device, &gfx.config);
                // The MSAA texture must match the new surface dimensions.
                let (tex, view) = create_msaa(
                    &gfx.device,
                    gfx.config.format,
                    gfx.config.width,
                    gfx.config.height,
                );
                gfx.msaa_tex = tex;
                gfx.msaa_view = view;
            }

            WindowEvent::ModifiersChanged(m) => {
                self.mods = m.state();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed || gfx.egui_ctx.egui_wants_keyboard_input() {
                    return;
                }
                // Held keys repeat only for panning and zooming. Toggles,
                // R (new system), F5 and the like fire once per press.
                let repeatable = match event.logical_key.as_ref() {
                    Key::Named(
                        NamedKey::ArrowUp
                        | NamedKey::ArrowDown
                        | NamedKey::ArrowLeft
                        | NamedKey::ArrowRight,
                    ) => true,
                    Key::Character(c) => matches!(
                        c.to_lowercase().as_str(),
                        "w" | "a" | "s" | "d" | "+" | "=" | "-" | "_"
                    ),
                    _ => false,
                };
                if event.repeat && !repeatable {
                    return;
                }
                let pan = 40.0 / self.sim.zoom as f64;
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Space) => self.sim.paused = !self.sim.paused,
                    Key::Named(NamedKey::ArrowUp) => {
                        self.sim.cam[1] -= pan;
                        self.sim.follow = false;
                        self.sim.follow_smooth = None;
                    }
                    Key::Named(NamedKey::ArrowDown) => {
                        self.sim.cam[1] += pan;
                        self.sim.follow = false;
                        self.sim.follow_smooth = None;
                    }
                    Key::Named(NamedKey::ArrowLeft) => {
                        self.sim.cam[0] -= pan;
                        self.sim.follow = false;
                        self.sim.follow_smooth = None;
                    }
                    Key::Named(NamedKey::ArrowRight) => {
                        self.sim.cam[0] += pan;
                        self.sim.follow = false;
                        self.sim.follow_smooth = None;
                    }
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
                        }
                    }
                    Key::Named(NamedKey::F5) => {
                        {
                            let b = self.shared.lock_for_render();
                            self.sim.save_game(&b, "quicksave", &mut self.ui);
                        }
                        self.flush_io();
                    }
                    Key::Named(NamedKey::F9) => {
                        self.sim.pending_load = Some("quicksave".to_string());
                        self.flush_io();
                    }
                    Key::Character(c)
                        if !self.mods.control_key()
                            && !self.mods.alt_key()
                            && !self.mods.super_key() =>
                    {
                        match c.to_lowercase().as_str() {
                            "r" => {
                                let mut b = self.shared.lock_for_render();
                                self.sim.reset(&mut b, &self.shared);
                                self.sim.fit(&b, vp); // ← added
                                self.ui.set_pending_seed(self.sim.seed);
                            }
                            "+" | "=" => self.sim.zoom_at(1.25, [vp[0] * 0.5, vp[1] * 0.5], vp),
                            "-" | "_" => self.sim.zoom_at(0.8, [vp[0] * 0.5, vp[1] * 0.5], vp),
                            "w" => {
                                self.sim.cam[1] -= pan;
                                self.sim.follow = false;
                                self.sim.follow_smooth = None;
                            }
                            "s" => {
                                self.sim.cam[1] += pan;
                                self.sim.follow = false;
                                self.sim.follow_smooth = None;
                            }
                            "a" => {
                                self.sim.cam[0] -= pan;
                                self.sim.follow = false;
                                self.sim.follow_smooth = None;
                            }
                            "d" => {
                                self.sim.cam[0] += pan;
                                self.sim.follow = false;
                                self.sim.follow_smooth = None;
                            }
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
                            "y" => self.ui.show_systems = !self.ui.show_systems,
                            "e" => self.ui.show_events = !self.ui.show_events,
                            "k" => self.ui.show_saves = !self.ui.show_saves,
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
                            "." => {
                                let mut b = self.shared.lock_for_render();
                                physics::advance(&mut b, physics::DT);
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }

            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                if state == ElementState::Pressed {
                    if response.consumed || gfx.egui_ctx.is_pointer_over_egui() {
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
                        // ALWAYS disable follow and clear the stale smooth position
                        // when dragging the canvas.
                        self.sim.follow = false;
                        self.sim.follow_smooth = None;
                    }
                }
                self.sim.cursor = pos;
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if response.consumed || gfx.egui_ctx.is_pointer_over_egui() {
                    return;
                }
                let zoom_mod = self.mods.control_key()
                    || self.mods.super_key()
                    || self.mods.shift_key()
                    || self.mods.alt_key();
                // Convert to pixels. A mouse wheel sends whole lines on the
                // y axis only. A touchpad sends fractions or sideways motion.
                let (dx, dy, touchpad) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        (x * 30.0, y * 30.0, x != 0.0 || y.fract() != 0.0)
                    }
                    MouseScrollDelta::PixelDelta(p) => (p.x as f32, p.y as f32, true),
                };
                if zoom_mod || !touchpad {
                    // Mouse wheel, or Ctrl/Cmd with scroll (a pinch on
                    // Windows and Linux): zoom at the cursor.
                    let cursor = self.sim.cursor;
                    let dz = if dy.abs() >= dx.abs() { dy } else { dx };
                    self.sim.zoom_at((dz / 30.0 * 0.15).exp(), cursor, vp);
                } else {
                    // Two-finger scroll: pan, like a phone.
                    self.sim.cam[0] -= dx as f64 / self.sim.zoom as f64;
                    self.sim.cam[1] -= dy as f64 / self.sim.zoom as f64;
                    self.sim.follow = false;
                    self.sim.follow_smooth = None;
                }
            }

            // macOS pinch gesture. `delta` is the fractional change in scale.
            WindowEvent::PinchGesture { delta, .. } => {
                if response.consumed || gfx.egui_ctx.is_pointer_over_egui() {
                    return;
                }
                let cursor = self.sim.cursor;
                self.sim.zoom_at((delta as f32).exp(), cursor, vp);
            }

            WindowEvent::RedrawRequested => {
                if gfx.size.width == 0 || gfx.size.height == 0 {
                    self.idle_frames = self.idle_frames.saturating_add(1);
                    return;
                }
                let t_frame = Instant::now();
                let frame_dt = (t_frame - self.sim.last_frame_time).as_secs_f64().min(0.1);
                self.sim.last_frame_time = t_frame;

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
                    wgpu::CurrentSurfaceTexture::Success(f) => f,
                    wgpu::CurrentSurfaceTexture::Suboptimal(f) => {
                        gfx.surface.configure(&gfx.device, &gfx.config);
                        f
                    }
                    wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                        gfx.surface.configure(&gfx.device, &gfx.config);
                        return;
                    }
                    _ => return,
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
                    // Time spent waiting for the physics thread to release the
                    // lock, then time spent holding it (NBODY_TIMING only).
                    let t_wait0 = Instant::now();
                    let mut b = self.shared.lock_for_render();
                    let t_held0 = Instant::now();
                    self.sim.t_wait_accum += t_wait0.elapsed().as_secs_f64() * 1e6;

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

                    let selected_idx = self.sim.selected.and_then(|id| b.index_of_id(id));
                    if self.sim.follow {
                        match selected_idx {
                            Some(i) => {
                                let id = b.id[i];
                                const TAU: f64 = 0.06;
                                let alpha = 1.0 - (-frame_dt / TAU).exp();
                                let prev = match self.sim.follow_smooth {
                                    Some((prev_id, pos)) if prev_id == id => pos,
                                    _ => [b.x[i], b.y[i]],
                                };
                                let new_pos = [
                                    prev[0] + (b.x[i] - prev[0]) * alpha,
                                    prev[1] + (b.y[i] - prev[1]) * alpha,
                                ];
                                self.sim.follow_smooth = Some((id, new_pos));
                                self.sim.cam = new_pos;
                            }
                            None => {
                                self.sim.follow = false;
                                self.sim.follow_smooth = None;
                            }
                        }
                    } else {
                        self.sim.follow_smooth = None;
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
                        follow_smooth: self.sim.follow_smooth,
                        preset: self.sim.preset,
                        vsync: self.sim.vsync,
                        behind: self.shared.behind.load(Ordering::Relaxed),
                        pause_on_event: self.sim.pause_on_event,
                        trails_enabled: self.sim.trail.cfg.enabled,
                        trail_age: self.sim.trail.cfg.max_age,
                        trail_bodies: self.sim.trail.cfg.max_bodies,
                        trail_non_rubble: self.sim.trail.cfg.non_rubble_only,
                        sn_kicks: self.sim.sn_kicks,
                        mass_transfer: self.sim.mass_transfer,
                        aging_mult: self.sim.aging_mult,
                    };
                    full_output = gfx.egui_ctx.run_ui(raw_input, |ui| {
                        actions = ui::draw(ui, &mut self.ui, &view_info, &b);
                    });
                    self.sim
                        .apply_actions(actions, &mut self.ui, vp, &mut b, &self.shared);
                    trail_snapshot = self.sim.trail.snapshot(&b, self.sim.follow_smooth);

                    // Pause when the merger / impact / tidal count grows. A drop
                    // means a new system replaced the old one: no pause then.
                    let pause_events = b
                        .stats
                        .merges
                        .wrapping_add(b.stats.impacts)
                        .wrapping_add(b.stats.tidal);
                    if self.sim.pause_on_event
                        && !self.sim.paused
                        && pause_events > self.sim.last_pause_events
                    {
                        self.sim.paused = true;
                    }
                    self.sim.last_pause_events = pause_events;

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

                    let t_inst0 = Instant::now();
                    let override_body = self
                        .sim
                        .follow_smooth
                        .and_then(|(id, pos)| b.index_of_id(id).map(|idx| (idx, pos)));
                    gfx.renderer.build_instances(
                        &b,
                        self.sim.zoom,
                        self.sim.cam,
                        vp,
                        override_body,
                    );
                    self.sim.t_inst_accum += t_inst0.elapsed().as_secs_f64() * 1e6;
                    let t_comet0 = Instant::now();
                    comet_tail_verts = Vec::new();
                    self.sim.t_comet_accum += t_comet0.elapsed().as_secs_f64() * 1e6;
                    flashes_active = !b.flashes.is_empty();
                    n_bodies = b.len();
                    if self.timing_log {
                        self.sim.last_comets =
                            b.origin.iter().filter(|&&o| o == BodyOrigin::Comet).count();
                        self.sim.last_comet_verts = comet_tail_verts.len();
                    }
                    self.sim.t_held_accum += t_held0.elapsed().as_secs_f64() * 1e6;
                }

                // ---- Everything below runs WITHOUT the bodies lock. ----
                // Queued disk work (save write, save read). Inlined instead of
                // `self.flush_io()` because `gfx` still borrows `self.gfx`.
                if let Some(io) = self.sim.deferred_io.take() {
                    io(&mut self.ui);
                }
                if let Some(name) = self.sim.pending_load.take() {
                    self.sim.load_game(&name, &mut self.ui, &self.shared);
                    self.idle_frames = 0;
                }
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
                    let verts = self.sim.trail.build_vertices(self.sim.cam);
                    gfx.renderer
                        .set_trail_vertices(&gfx.device, &gfx.queue, &verts);
                } else if gfx.renderer.trail_count() > 0 {
                    gfx.renderer
                        .set_trail_vertices(&gfx.device, &gfx.queue, &[]);
                }

                let repaint_delay = full_output
                    .viewport_output
                    .get(&egui::ViewportId::ROOT)
                    .map(|v| v.repaint_delay);
                let repaint_now = repaint_delay.is_some_and(|d| d.is_zero());
                // A finite, non-zero delay (egui's `request_repaint_after`) must
                // still wake an idle app. A huge delay means "no request".
                self.repaint_at = repaint_delay
                    .filter(|d| !d.is_zero() && *d < Duration::from_secs(3600))
                    .map(|d| Instant::now() + d);

                gfx.egui_state
                    .handle_platform_output(&gfx.window, full_output.platform_output);
                let pixels_per_point = gfx.egui_ctx.pixels_per_point();
                let clipped = gfx
                    .egui_ctx
                    .tessellate(full_output.shapes, pixels_per_point);

                for (id, deltas) in &full_output.textures_delta.set {
                    for delta in deltas {
                        gfx.egui_renderer
                            .update_texture(&gfx.device, &gfx.queue, *id, delta);
                    }
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
                            view: &gfx.msaa_view,
                            depth_slice: None,
                            resolve_target: Some(&view),
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    let mut pass = pass.forget_lifetime();
                    gfx.renderer.draw(&mut pass);
                    gfx.egui_renderer.render(&mut pass, &clipped, &screen_desc);
                }

                gfx.queue.submit([encoder.finish()]);
                gfx.queue.present(frame);

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
                        let ms = |us: f64| us / frames / 1000.0;
                        let dust_busy = physics::DUST_STAT_NS.swap(0, Ordering::Relaxed) as f64
                            / (secs * 1e9)
                            * 100.0;
                        let dust_grains = physics::DUST_STAT_GRAINS.load(Ordering::Relaxed);
                        println!(
                            "[timing] n={} comets={} (tail verts {}) fps={:.0} frame={:.2}ms \
                             lock_wait={:.2}ms lock_held={:.2}ms (instances={:.2} comet_tails={:.2}) \
                             physics_busy={:.0}% (dust={:.0}%, {} grains){}",
                            n_bodies,
                            self.sim.last_comets,
                            self.sim.last_comet_verts,
                            frames / secs,
                            ms(self.sim.t_frame_accum),
                            ms(self.sim.t_wait_accum),
                            ms(self.sim.t_held_accum),
                            ms(self.sim.t_inst_accum),
                            ms(self.sim.t_comet_accum),
                            phys_busy,
                            dust_busy,
                            dust_grains,
                            if self.shared.behind.load(Ordering::Relaxed) {
                                " (behind)"
                            } else {
                                ""
                            }
                        );
                        self.sim.t_window_start = Instant::now();
                        self.sim.t_window_frames = 0;
                        self.sim.t_frame_accum = 0.0;
                        self.sim.t_wait_accum = 0.0;
                        self.sim.t_held_accum = 0.0;
                        self.sim.t_inst_accum = 0.0;
                        self.sim.t_comet_accum = 0.0;
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

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.idle_frames < IDLE_FRAMES {
            if let Some(g) = &self.gfx {
                g.window.request_redraw();
            }
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        }
        // Idle: sleep until input arrives, or until egui asked to repaint.
        match self.repaint_at {
            Some(t) if Instant::now() >= t => {
                self.repaint_at = None;
                if let Some(g) = &self.gfx {
                    g.window.request_redraw();
                }
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            Some(t) => event_loop.set_control_flow(ControlFlow::WaitUntil(t)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
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
