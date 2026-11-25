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

use nbody_rust::history::{self, History};
use nbody_rust::physics::{self, Bodies, BodyKind, BodyOrigin};
use nbody_rust::render::{self, OrbitReq, OrbitVertex, Renderer};
use nbody_rust::ui::{self, Actions, Edit, TimelineAct, TimelineInfo, UiState, ViewInfo};
use nbody_rust::view::{self, View3};
use nbody_rust::Preset;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{Window, WindowId};

use egui_wgpu::ScreenDescriptor;

/// A save or load to run once the UI frame is done (it needs the UI state).
type DeferredIo = Box<dyn FnOnce(&mut UiState)>;

mod sim_actions;
mod sim_frame;
mod sim_input;
mod sim_saves;
mod sim_thread;
use sim_thread::*;
mod trail;
use trail::*;

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

/// A timeline replay in progress: the simulation time being shown.
struct Replay {
    time: f64,
    playing: bool,
}

struct SimState {
    paused: bool,
    speed: f64,
    last_frame_time: Instant,
    follow_smooth: Option<(u64, [f64; 3])>,
    cam: [f64; 2],
    /// Focus height and view angles (radians); 0 is the old top-down view.
    cam_z: f64,
    yaw: f64,
    pitch: f64,
    /// Right mouse button held: dragging turns the view.
    rotating: bool,
    /// World point under the cursor when the rotation started, and where it
    /// sat on screen: the view turns about it so it stays under the cursor.
    rot_anchor: Option<([f64; 3], [f32; 2])>,
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
    /// Timeline replay: `Some` while viewing a recorded moment instead of the live state.
    replay: Option<Replay>,
    /// The snapshot being shown, as an owned `Bodies` swapped in while rendering.
    replay_cache: Option<(u64, Bodies)>,
    /// True while the live bodies sit in `replay_cache` (between `enter_replay`
    /// and `leave_replay`); the cache must not be dropped then.
    replay_swapped: bool,
    /// Frame id and zoom the replay orbit lines were last built for.
    replay_orbit: (u64, f32),
    last_orbit_update: Instant,
    last_orbit_zoom: f32,
    last_orbit_cam: [f64; 2],
    last_orbit_events: u64,
    last_hz: bool,
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
    deferred_io: Option<DeferredIo>,
    /// Name of a save to load once the bodies lock is released.
    pending_load: Option<String>,
}

/// What the locked part of a redraw hands to the unlocked part.
struct FrameData {
    full_output: egui::FullOutput,
    flashes_active: bool,
    n_bodies: usize,
    new_title: Option<String>,
    trail_snapshot: Vec<TrailSample>,
    comet_tail_verts: Vec<OrbitVertex>,
    /// Orbit vertices from the physics thread, applied after the lock drops.
    orbit_out: Option<(u64, Vec<OrbitVertex>)>,
    /// The recorded frame shown this redraw, if any.
    replaying: Option<u64>,
}

impl SimState {
}

impl SimState {
    fn new() -> Self {
        Self {
            paused: false,
            speed: 1.0,
            last_frame_time: Instant::now(),
            follow_smooth: None,
            cam: [0.0, 0.0],
            cam_z: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            rotating: false,
            rot_anchor: None,
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
            replay: None,
            replay_cache: None,
            replay_swapped: false,
            replay_orbit: (u64::MAX, 0.0),
            last_orbit_update: Instant::now(),
            last_orbit_zoom: 0.7,
            last_orbit_cam: [0.0; 2],
            last_orbit_events: 0,
            last_hz: false,
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

}

impl GfxState {
    /// Reconfigure the surface when the vsync setting changed.
    fn apply_present_mode(&mut self, vsync: bool) {
        let want_mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        if self.config.present_mode != want_mode {
            self.config.present_mode = want_mode;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// The next surface texture, or `None` when the surface had to be
    /// reconfigured or is unavailable this frame.
    fn acquire_frame(&mut self) -> Option<wgpu::SurfaceTexture> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) => Some(f),
            wgpu::CurrentSurfaceTexture::Suboptimal(f) => {
                self.surface.configure(&self.device, &self.config);
                Some(f)
            }
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                None
            }
            _ => None,
        }
    }

// ---------------------------------------------------------------------------
// App

    /// Push this frame's geometry to the GPU: orbit lines (when new ones
    /// arrived), camera uniforms, body instances, comet tails and trails
    /// (`None` clears any trails still on screen).
    fn upload_scene(
        &mut self,
        orbit_lines: Option<&[OrbitVertex]>,
        cam_view: &view::View3,
        vp: [f32; 2],
        comet_tails: &[OrbitVertex],
        trails: Option<&[OrbitVertex]>,
    ) {
        if let Some(verts) = orbit_lines {
            self.renderer.set_orbit_vertices(&self.device, &self.queue, verts);
        }
        self.renderer.update_camera(&self.queue, cam_view, vp);
        self.renderer.upload_instances(&self.device, &self.queue);
        self.renderer
            .set_comet_tail_vertices(&self.device, &self.queue, comet_tails);
        match trails {
            Some(verts) => self.renderer.set_trail_vertices(&self.device, &self.queue, verts),
            None if self.renderer.trail_count() > 0 => {
                self.renderer.set_trail_vertices(&self.device, &self.queue, &[]);
            }
            None => {}
        }
    }

    /// Tessellate the egui output, draw the scene and the UI into the MSAA
    /// target, resolve it into `view`, and present `frame`.
    fn present_frame(
        &mut self,
        view: &wgpu::TextureView,
        frame: wgpu::SurfaceTexture,
        full_output: egui::FullOutput,
    ) {
    self.egui_state
        .handle_platform_output(&self.window, full_output.platform_output);
    let pixels_per_point = self.egui_ctx.pixels_per_point();
    let clipped = self
        .egui_ctx
        .tessellate(full_output.shapes, pixels_per_point);

    for (id, deltas) in &full_output.textures_delta.set {
        for delta in deltas {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
    }

    let screen_desc = ScreenDescriptor {
        size_in_pixels: [self.config.width, self.config.height],
        pixels_per_point,
    };
    let mut encoder = self
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    self.egui_renderer.update_buffers(
        &self.device,
        &self.queue,
        &mut encoder,
        &clipped,
        &screen_desc,
    );

    {
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.msaa_view,
                depth_slice: None,
                resolve_target: Some(view),
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
        self.renderer.draw(&mut pass);
        self.egui_renderer.render(&mut pass, &clipped, &screen_desc);
    }

    self.queue.submit([encoder.finish()]);
    self.queue.present(frame);

    for id in &full_output.textures_delta.free {
        self.egui_renderer.free_texture(id);
    }
    }
}

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

impl App {
    /// Handle one key press. `egui_has_keyboard` is true while a text field
    /// or similar widget owns the keyboard, in which case the key is ignored.
    fn on_key(&mut self, event: winit::event::KeyEvent, vp: [f32; 2], egui_has_keyboard: bool) {
        if event.state != ElementState::Pressed || egui_has_keyboard {
            return;
        }
        if event.repeat && !is_repeatable_key(&event.logical_key) {
            return;
        }
        let pan = 40.0 / self.sim.zoom as f64;
        match event.logical_key.as_ref() {
            Key::Named(named) => self.on_named_key(named, pan),
            Key::Character(c)
                if !self.mods.control_key() && !self.mods.alt_key() && !self.mods.super_key() =>
            {
                self.on_char_key(&c.to_lowercase(), pan, vp)
            }
            _ => {}
        }
    }

    fn on_named_key(&mut self, key: NamedKey, pan: f64) {
        match key {
            NamedKey::Space => self.sim.paused = !self.sim.paused,
            NamedKey::ArrowUp => self.pan_and_unfollow(0.0, -pan),
            NamedKey::ArrowDown => self.pan_and_unfollow(0.0, pan),
            NamedKey::ArrowLeft => self.pan_and_unfollow(-pan, 0.0),
            NamedKey::ArrowRight => self.pan_and_unfollow(pan, 0.0),
            NamedKey::Escape => {
                self.sim.selected = None;
                self.sim.follow = false;
            }
            NamedKey::Delete | NamedKey::Backspace => self.delete_selected(),
            NamedKey::F5 => {
                {
                    let mut b = self.shared.lock_for_render();
                    self.shared.settle_step(&mut b);
                    self.sim.save_game(&b, "quicksave", &mut self.ui);
                }
                self.flush_io();
            }
            NamedKey::F9 => {
                self.sim.pending_load = Some("quicksave".to_string());
                self.flush_io();
            }
            _ => {}
        }
    }

    /// Letter and symbol shortcuts; `c` is already lower-cased.
    fn on_char_key(&mut self, c: &str, pan: f64, vp: [f32; 2]) {
        let centre = [vp[0] * 0.5, vp[1] * 0.5];
        match c {
            "r" => {
                let mut b = self.shared.lock_for_render();
                self.sim.reset(&mut b, &self.shared);
                self.sim.fit(&b, vp);
                self.ui.set_pending_seed(self.sim.seed);
            }
            "+" | "=" => self.sim.zoom_at(1.25, centre, vp),
            "-" | "_" => self.sim.zoom_at(0.8, centre, vp),
            "w" => self.move_cam_and_unfollow(0.0, -pan),
            "s" => self.move_cam_and_unfollow(0.0, pan),
            "a" => self.move_cam_and_unfollow(-pan, 0.0),
            "d" => self.move_cam_and_unfollow(pan, 0.0),
            "o" => {
                self.sim.show_orbits = !self.sim.show_orbits;
                self.sim.orbit_buf_valid = false;
            }
            ";" => {
                self.sim.show_debris_orbits = !self.sim.show_debris_orbits;
                self.sim.orbit_buf_valid = false;
            }
            "v" => self.sim.vsync = !self.sim.vsync,
            "f" | "0" => {
                let b = self.shared.lock_for_render();
                self.sim.fit(&b, vp);
            }
            "c" => {
                if self.sim.selected.is_some() {
                    self.sim.follow = !self.sim.follow;
                }
            }
            "t" => {
                // Back to the top-down view.
                self.sim.yaw = 0.0;
                self.sim.pitch = 0.0;
            }
            "n" | "p" => self.cycle_selection_key(if c == "n" { 1 } else { -1 }),
            "." if self.sim.replay.is_none() => {
                let mut b = self.shared.lock_for_render();
                // Finish the paused step first, so this one starts on a whole state.
                self.shared.settle_step(&mut b);
                physics::advance(&mut b, physics::DT);
            }
            _ => self.toggle_window_key(c),
        }
    }

    /// Show or hide a window or overlay.
    fn toggle_window_key(&mut self, c: &str) {
        match c {
            "`" => self.ui.show_dock = !self.ui.show_dock,
            "l" => self.ui.show_labels = !self.ui.show_labels,
            "u" => self.ui.units_si = !self.ui.units_si,
            "b" => self.ui.show_list = !self.ui.show_list,
            "y" => self.ui.show_systems = !self.ui.show_systems,
            "e" => self.ui.show_events = !self.ui.show_events,
            "k" => self.ui.show_saves = !self.ui.show_saves,
            "h" | "?" | "/" => self.ui.show_help = !self.ui.show_help,
            "g" => self.ui.diag.visible = !self.ui.diag.visible,
            "j" => self.ui.show_timeline = !self.ui.show_timeline,
            "x" => self.ui.elem.visible = !self.ui.elem.visible,
            "z" => physics::set_hz_rings(!physics::hz_rings()),
            "i" => self.ui.light.visible = !self.ui.light.visible,
            _ => {}
        }
    }

    fn pan_and_unfollow(&mut self, dx: f64, dy: f64) {
        self.sim.pan_gm(dx, dy);
        self.sim.follow = false;
        self.sim.follow_smooth = None;
    }

    fn move_cam_and_unfollow(&mut self, dx: f64, dy: f64) {
        self.sim.cam[0] += dx;
        self.sim.cam[1] += dy;
        self.sim.follow = false;
        self.sim.follow_smooth = None;
    }

    fn delete_selected(&mut self) {
        // A replay is view-only, and its selection ids belong to the snapshot.
        if self.sim.replay.is_some() {
            return;
        }
        let mut b = self.shared.lock_for_render();
        self.shared.settle_step(&mut b);
        if let Some(i) = self.sim.selected.and_then(|id| b.index_of_id(id)) {
            physics::delete_body(&mut b, i);
            self.sim.selected = None;
            self.sim.follow = false;
            self.sim.orbit_buf_valid = false;
        }
    }

    /// Select the next or previous body, from the replayed snapshot while scrubbing.
    fn cycle_selection_key(&mut self, dir: i32) {
        if self.sim.replay.is_some() {
            if let Some(cache) = self.sim.replay_cache.take() {
                self.sim.cycle_selection(&cache.1, dir);
                self.sim.replay_cache = Some(cache);
            }
        } else {
            let b = self.shared.lock_for_render();
            self.sim.cycle_selection(&b, dir);
        }
    }
}

/// Held keys repeat only for panning and zooming. Toggles, R (new system),
/// F5 and the like fire once per press.
fn is_repeatable_key(key: &Key) -> bool {
    match key.as_ref() {
        Key::Named(
            NamedKey::ArrowUp | NamedKey::ArrowDown | NamedKey::ArrowLeft | NamedKey::ArrowRight,
        ) => true,
        Key::Character(c) => {
            matches!(c.to_lowercase().as_str(), "w" | "a" | "s" | "d" | "+" | "=" | "-" | "_")
        }
        _ => false,
    }
}

impl App {
    /// Draw one frame: update the shared state under the lock, then upload and present.
    fn redraw(&mut self, vp: [f32; 2]) {
        let Some(gfx) = self.gfx.as_mut() else {
            return;
        };
        if gfx.size.width == 0 || gfx.size.height == 0 {
            self.idle_frames = self.idle_frames.saturating_add(1);
            return;
        }
        let t_frame = Instant::now();
        let frame_dt = (t_frame - self.sim.last_frame_time).as_secs_f64().min(0.1);
        self.sim.last_frame_time = t_frame;

        gfx.apply_present_mode(self.sim.vsync);
        let Some(frame) = gfx.acquire_frame() else {
            return;
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let data = self.sim.update_locked(&mut self.ui, &self.shared, gfx, vp, frame_dt, self.timing_log);

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
        if let Some((id, _)) = &data.orbit_out {
            self.sim.orbit_applied = *id;
        }
        self.sim.upload_frame(gfx, &data, vp);

        let repaint_delay = data
            .full_output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map(|v| v.repaint_delay);
        let repaint_now = repaint_delay.is_some_and(|d| d.is_zero());
        // A finite, non-zero delay (egui's `request_repaint_after`) must
        // still wake an idle app. A huge delay means "no request".
        self.repaint_at = repaint_delay
            .filter(|d| !d.is_zero() && *d < Duration::from_secs(3600))
            .map(|d| Instant::now() + d);

        let (flashes_active, n_bodies, new_title) = (data.flashes_active, data.n_bodies, data.new_title);
        gfx.present_frame(&view, frame, data.full_output);

        if self.timing_log {
            self.sim.log_frame_timing(&self.shared, t_frame, n_bodies);
        }
        if let Some(t) = new_title {
            gfx.window.set_title(&t);
        }

        let busy = !self.sim.paused
            || flashes_active
            || self.sim.follow
            || repaint_now
            || self.sim.orbit_posted > self.sim.orbit_applied;
        self.idle_frames = if busy { 0 } else { self.idle_frames.saturating_add(1) };
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
                .spawn(move || physics_supervisor(sh))
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
                let egui_has_keyboard = gfx.egui_ctx.egui_wants_keyboard_input();
                self.on_key(event, vp, egui_has_keyboard);
            }

            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                if pressed && (response.consumed || gfx.egui_ctx.is_pointer_over_egui()) {
                    return;
                }
                self.sim.on_left_button(pressed, vp, &self.shared);
            }

            // Right button: drag to turn the camera (spin and tilt).
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                if pressed && (response.consumed || gfx.egui_ctx.is_pointer_over_egui()) {
                    return;
                }
                self.sim.on_right_button(pressed, vp);
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.sim.on_cursor_moved([position.x as f32, position.y as f32], vp);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if response.consumed || gfx.egui_ctx.is_pointer_over_egui() {
                    return;
                }
                let zoom_modifier = self.mods.control_key()
                    || self.mods.super_key()
                    || self.mods.shift_key()
                    || self.mods.alt_key();
                self.sim.on_scroll(delta, zoom_modifier, vp);
            }

            // macOS pinch gesture. `delta` is the fractional change in scale.
            WindowEvent::PinchGesture { delta, .. } => {
                if response.consumed || gfx.egui_ctx.is_pointer_over_egui() {
                    return;
                }
                let cursor = self.sim.cursor;
                self.sim.zoom_at((delta as f32).exp(), cursor, vp);
            }

            WindowEvent::RedrawRequested => self.redraw(vp),
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
