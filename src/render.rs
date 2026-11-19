//! wgpu renderer for the 2D scene: instanced quads for bodies, a line-list
//! for orbit and trail vertices, instanced quads for collision flashes,
//! instanced soft points for dust grains, additive glow halos around luminous
//! bodies, and a ray-traced quad per black hole.
//!
//! Layout contract
//! ---------------
//! Every vertex/instance struct is `#[repr(C)] + Pod + Zeroable` and mirrored
//! by the `VertexAttribute` offsets below and the `@location`s in
//! `render.wgsl`. The `_pad` fields keep WGSL's 16-byte alignment; do not
//! remove them.
//!
//! Pipelines
//! ---------
//!   * `body_pipeline`  - instanced triangle strip, one quad per body.
//!   * `orbit_pipeline` - line list, used for orbit lines and trails.
//!   * `flash_pipeline` - instanced triangle strip, one ring per flash.
//!   * `dust_pipeline`  - instanced triangle strip, one ~1 px soft point per
//!     tracer grain, additive blending.
//!   * `glow_pipeline`  - instanced triangle strip, one additive halo per
//!     luminous body (stars, brown/white dwarfs, neutron stars).
//!   * `blackhole_pipeline` - instanced triangle strip, one quad per black
//!     hole; the fragment shader traces rays through Schwarzschild spacetime.
//!   * `comet_pipeline` - additive triangle list for comet tails. Reserved:
//!     `main.rs` currently uploads no tail vertices, so it never draws.
//!
//! Cameras
//! -------
//! `camera_bg` is the *relative* camera (pos = 0): bodies, flashes, dust,
//! glow, black holes and trails are pre-subtracted by the camera on the CPU
//! in f64, which keeps f32 precision when zoomed in. `orbit_camera_bg` is the
//! *world* camera, used only by orbit lines and comet tails (their vertices
//! are built elsewhere in absolute f32 coordinates).

use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};

use crate::physics::{self, Bodies, BodyClass, BodyKind, BodyOrigin};

/// Fields the dust centre uniform can hold. Must match `DustFields` in the WGSL.
pub const MAX_DUST_FIELDS: usize = 256;
/// Cap on grains drawn per frame, summed over fields (subsampled above it).
const MAX_DUST_POINTS: usize = 2_000_000;
/// A field whose extent is smaller than this on screen is not drawn.
const DUST_MIN_PX: f64 = 6.0;
/// Minimum time between point-buffer rebuilds.
const DUST_UPLOAD_MIN: Duration = Duration::from_millis(100);

// ---------------------------------------------------------------------------
// GPU structs
// ---------------------------------------------------------------------------

/// Mirrors `struct Camera` in `render.wgsl` (32 bytes, 16-aligned).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub pos: [f32; 2],
    pub zoom: f32,
    _pad0: f32,
    pub viewport: [f32; 2],
    _pad1: [f32; 2],
}

/// One body as an instanced quad. Offsets: pos 0, radius 8, lit 12,
/// color 16, light_angle 28; stride 32 (`vec3` needs 16-byte alignment).
///
/// `lit` is 1 for a body that a star lights (planets) and 0 for a body that
/// shines or is not shaded. `light_angle` is the direction from the body to
/// its star, in world axes, in radians.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BodyInstance {
    pub pos: [f32; 2],
    pub radius: f32,
    pub lit: f32,
    pub color: [f32; 3],
    pub light_angle: f32,
}

/// One endpoint of an orbit or trail segment (`Unorm8x4` colour), stride 12.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct OrbitVertex {
    pub pos: [f32; 2],
    pub color: [u8; 4],
}

/// One collision flash. Stride 16.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FlashInstance {
    pub pos: [f32; 2],
    pub radius: f32,
    pub progress: f32,
}

/// One dust grain. `pos` is relative to the field's primary at upload time,
/// `weight` is the tracer mass in Earth masses, `field` indexes the centre
/// uniform. Offsets: pos 0, weight 8, field 12; stride 16.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DustVertex {
    pub pos: [f32; 2],
    pub weight: f32,
    pub field: u32,
}

/// Everything needed to build the orbit lines for one frame. Sent to the
/// physics thread, where `compute_orbit_segments` runs.
#[derive(Clone, Copy, Debug)]
pub struct OrbitReq {
    pub zoom: f64,
    pub planets: bool,
    pub debris: bool,
    // Reserved for future culling. Currently unused by the builder.
    #[allow(dead_code)]
    pub cam: [f64; 2],
    #[allow(dead_code)]
    pub viewport: [f32; 2],
}

/// One glow halo around a luminous body. `pos` is camera-relative (same
/// convention as `BodyInstance`), `radius` is the outer edge of the halo
/// (not the body radius), `intensity` scales the peak alpha, and `color`
/// is the body's blackbody tint.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GlowInstance {
    pub pos: [f32; 2],
    pub radius: f32,
    pub intensity: f32,
    pub color: [f32; 3],
    pub _pad: f32,
}

/// One black hole. `pos` is camera-relative. `radius` is the event horizon.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BlackHoleInstance {
    pub pos: [f32; 2],
    pub radius: f32,
    /// 0..1 strength of the accretion disk; 0 draws only the shadow.
    pub accretion: f32,
}

/// Radius of the black shadow in horizon radii: 3·sqrt(3)/2. A distant
/// observer sees this disc, not the horizon itself, because light bends.
/// The shader traces rays and uses the same number (`BH_B_CRIT`).
const BH_SHADOW_IN_RS: f64 = 2.598_076;

/// Glow strength factor from luminosity `l` in L☉. A log scale: 1 at the
/// Sun, about 0.6 at 0.01 L☉, about 1.4 at 100 L☉. Clamped so a very dim
/// body keeps a faint halo and a very bright one does not wash out.
fn glow_luminosity_factor(l: f64) -> f64 {
    if !l.is_finite() || l <= 0.0 {
        return 0.25;
    }
    (1.0 + 0.2 * l.log10()).clamp(0.25, 1.8)
}

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------

/// Approximate sRGB color of a blackbody at `t_kelvin`, normalized to
/// [0, 1]. Tanner Helland's fit, accurate to a few percent over 1000 K
/// .. 40,000 K. Outside that range the endpoints are clamped — deeper
/// red than 1000 K can't be expressed in sRGB, and above 40,000 K the
/// eye can't distinguish further blue-shift.
fn blackbody_to_rgb(t_kelvin: f64) -> [f32; 3] {
    let t = (t_kelvin / 100.0).clamp(10.0, 400.0);
    let (r, g, b) = if t <= 66.0 {
        let r = 255.0;
        let g = 99.4708025861 * t.ln() - 161.1195681661;
        let b = if t <= 19.0 {
            0.0
        } else {
            138.5177312231 * (t - 10.0).ln() - 305.0447927307
        };
        (r, g, b)
    } else {
        let r = 329.698727446 * (t - 60.0).powf(-0.1332047592);
        let g = 288.1221695283 * (t - 60.0).powf(-0.0755148492);
        let b = 255.0;
        (r, g, b)
    };
    [
        (r.clamp(0.0, 255.0) / 255.0) as f32,
        (g.clamp(0.0, 255.0) / 255.0) as f32,
        (b.clamp(0.0, 255.0) / 255.0) as f32,
    ]
}

/// Equilibrium temperature (K, Bond albedo 0.3) of planet `i` from its host's
/// light alone. O(1): uses the cached parent, one hop up for moons.
fn planet_equilibrium_temp(b: &Bodies, i: usize) -> Option<f64> {
    if !b.parent_cache_valid {
        return None;
    }
    let mut p = b.parent_cache.get(i).copied().flatten()?;
    if p < b.len() && !physics::is_host_kind(b.kind[p]) {
        p = b.parent_cache.get(p).copied().flatten()?;
    }
    if p >= b.len() || !physics::is_host_kind(b.kind[p]) {
        return None;
    }
    let l = physics::luminosity_of(b, p);
    if !(l > 0.0) {
        return None;
    }
    let d_au = ((b.x[i] - b.x[p]).hypot(b.y[i] - b.y[p]) / physics::au_in_gm()).max(1e-4);
    Some(physics::equilibrium_temp(l / (d_au * d_au)))
}

/// Piecewise-linear colour ramp over `ln T`.
fn ramp(t: f64, stops: &[(f64, [f32; 3])]) -> [f32; 3] {
    if t <= stops[0].0 {
        return stops[0].1;
    }
    for w in stops.windows(2) {
        if t <= w[1].0 {
            let f = ((t / w[0].0).ln() / (w[1].0 / w[0].0).ln()) as f32;
            let (a, c) = (w[0].1, w[1].1);
            return [
                a[0] + (c[0] - a[0]) * f,
                a[1] + (c[1] - a[1]) * f,
                a[2] + (c[2] - a[2]) * f,
            ];
        }
    }
    stops[stops.len() - 1].1
}

/// Colour of a spawned planet at equilibrium temperature `t` (K).
///
/// Giants (≥ 50 M⊕) follow the Sudarsky classes: cream ammonia clouds below
/// ~150 K, white water clouds to ~250 K, clear blue-ish to ~1400 K, then dark
/// alkali-metal absorbers glowing red-orange above ~2000 K. Rocky worlds go
/// from frozen blue-white through temperate blue-green and scorched brown to
/// glowing lava. Ice giants and icy worlds keep their class colour.
fn planet_color_for_temp(t: f64, m: f64, class: BodyClass) -> Option<[f32; 3]> {
    if m >= 50.0 {
        return Some(ramp(
            t,
            &[
                (60.0, [0.80, 0.72, 0.58]),
                (150.0, [0.86, 0.78, 0.62]),
                (250.0, [0.93, 0.93, 0.90]),
                (800.0, [0.60, 0.68, 0.82]),
                (1400.0, [0.30, 0.42, 0.72]),
                (1800.0, [0.20, 0.17, 0.17]),
                (2500.0, [0.95, 0.40, 0.18]),
            ],
        ));
    }
    if class == BodyClass::Rocky {
        return Some(ramp(
            t,
            &[
                (120.0, [0.80, 0.85, 0.92]),
                (230.0, [0.62, 0.74, 0.88]),
                (300.0, [0.32, 0.52, 0.72]),
                (400.0, [0.62, 0.55, 0.42]),
                (900.0, [0.55, 0.40, 0.32]),
                (1500.0, [0.85, 0.42, 0.18]),
                (2200.0, [1.00, 0.55, 0.20]),
            ],
        ));
    }
    None
}

/// Color for a body, by kind + subclass. Stars use spectral class (giants are
/// red), brown dwarfs their temperature class, remnants fixed colours.
/// Planets are tinted by distance from their host (blue near, warm far);
/// belt and debris bodies have a fixed colour per origin.
pub fn body_color(b: &Bodies, i: usize) -> [f32; 3] {
    match b.kind[i] {
        BodyKind::Star => {
            if let Some(phase) = physics::phase_of(b, i) {
                match phase {
                    physics::StellarPhase::RedGiant => return [1.00, 0.55, 0.35],
                    physics::StellarPhase::RedSupergiant => return [1.00, 0.40, 0.25],
                    _ => {}
                }
            }

            let t = physics::temperature_of(b, i);
            if t.is_finite() && t > 0.0 {
                blackbody_to_rgb(t)
            } else {
                // Fallback for degenerate cases (temperature not
                // available for this body).
                match physics::star_class(b.m[i]) {
                    'M' => [1.0, 0.62, 0.45],
                    'K' => [1.0, 0.78, 0.55],
                    'G' => [1.0, 0.95, 0.75],
                    'F' => [1.0, 1.0, 0.92],
                    'A' => [0.85, 0.9, 1.0],
                    _ => [0.65, 0.75, 1.0],
                }
            }
        }
        BodyKind::BrownDwarf => {
            let t = physics::bd_state(b.m[i], b.age_gyr[i]).temperature;
            if t.is_finite() && t > 0.0 {
                blackbody_to_rgb(t)
            } else {
                match physics::bd_class_for_temp(t) {
                    'M' => [0.85, 0.35, 0.15],
                    'L' => [0.75, 0.25, 0.10],
                    'T' => [0.55, 0.15, 0.08],
                    'Y' => [0.35, 0.10, 0.05],
                    _ => [0.6, 0.2, 0.1],
                }
            }
        }
        BodyKind::WhiteDwarf => [0.90, 0.92, 1.0],
        BodyKind::NeutronStar => [1.0, 0.95, 0.85],
        BodyKind::BlackHole => [1.0, 0.35, 0.35],
        BodyKind::Planet | BodyKind::DwarfPlanet => {
            // Comets get their own colour regardless of the class
            // machinery below. Active (close to the host) is pale
            // cyan-white, matching C2 / dust emission. Dormant (far
            // out) fades to a dull blue-gray, matching the tarlike
            // surface of a real comet nucleus.
            if b.origin[i] == BodyOrigin::Comet {
                // Smooth blend: dark, tar-like nucleus far out, bright
                // point (coma core) when active. No hard switch.
                let d_au = physics::host_distance(b, i) / physics::au_in_gm();
                let t = (((4.0 - d_au) / 2.5).clamp(0.0, 1.0)) as f32;
                let t = t * t * (3.0 - 2.0 * t);
                let (dark, lit) = ([0.30f32, 0.31, 0.34], [0.95f32, 0.97, 1.00]);
                return [
                    dark[0] + (lit[0] - dark[0]) * t,
                    dark[1] + (lit[1] - dark[1]) * t,
                    dark[2] + (lit[2] - dark[2]) * t,
                ];
            }
            // Interstellar objects get a violet tint, distinct from
            // both comets (cyan-white) and debris (grey), so an ISO
            // visually reads as "not from around here".
            if b.origin[i] == BodyOrigin::Interstellar {
                return [0.85, 0.65, 1.00];
            }
            // Rogues get red-orange (hot, dark, radiogenic) — distinct
            // from both comets (cyan) and ISOs (violet). Readily
            // identifiable at a glance.
            if b.origin[i] == BodyOrigin::RoguePlanet {
                return [0.90, 0.42, 0.25];
            }
            // Spawned planets are coloured from their equilibrium temperature
            // (gas giants by Sudarsky class, rocky worlds from lava to ice).
            if b.origin[i] == BodyOrigin::Spawned {
                if let Some(t) = planet_equilibrium_temp(b, i) {
                    if let Some(c) = planet_color_for_temp(t, b.m[i], b.class[i]) {
                        return c;
                    }
                }
            }
            // A spawned body past 50 M⊕ is a gas giant — keep the tan
            // look instead of the pale-blue Icy class colour.
            if b.m[i] >= 50.0 && b.origin[i] == BodyOrigin::Spawned {
                return [0.85, 0.75, 0.60];
            }
            match b.class[i] {
                BodyClass::C => [0.30, 0.28, 0.26],   // carbonaceous: dark gray
                BodyClass::S => [0.55, 0.50, 0.42],   // silicaceous: tan-gray
                BodyClass::M => [0.60, 0.60, 0.62],   // metallic: silvery
                BodyClass::V => [0.55, 0.45, 0.35],   // basaltic: brown
                BodyClass::Icy => [0.68, 0.78, 0.88], // ice: pale blue-white
                BodyClass::Rocky => {
                    // Inner planets keep the existing distance tint.
                    let d = physics::host_distance(b, i) as f32;
                    let t = (d / 800.0).clamp(0.0, 1.0);
                    [0.4 + 0.4 * t, 0.6, 1.0 - 0.5 * t]
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Body instance construction
// ---------------------------------------------------------------------------

/// Comet tails: for each active comet (inside ~4 AU of its host, fading in as
/// in `body_color`) a blue ion tail pointing straight away from the host and a
/// shorter, curved, yellowish dust tail bent back along the orbit. Triangles in
/// world coordinates for the additive comet pipeline; alpha fades to zero at
/// the tip. Tail length is at least 12 px so a comet stays readable when the
/// view is zoomed out.
pub fn build_comet_tails(b: &Bodies, zoom: f32) -> Vec<OrbitVertex> {
    let mut out = Vec::new();
    if !b.parent_cache_valid {
        return out;
    }
    let au = physics::au_in_gm();
    let z = zoom.max(1e-6) as f64;
    for i in 0..b.len() {
        if b.dead[i] || b.origin[i] != BodyOrigin::Comet {
            continue;
        }
        let Some(h) = b.parent_cache.get(i).copied().flatten() else {
            continue;
        };
        if h >= b.len() || !physics::is_host_kind(b.kind[h]) {
            continue;
        }
        let dx = b.x[i] - b.x[h];
        let dy = b.y[i] - b.y[h];
        let d = dx.hypot(dy);
        if !(d > 0.0) {
            continue;
        }
        let t = (((4.0 - d / au) / 2.5).clamp(0.0, 1.0)) as f64;
        let act = t * t * (3.0 - 2.0 * t);
        if act < 0.02 {
            continue;
        }
        let (ux, uy) = (dx / d, dy / d); // away from the host
        let len = (0.5 * au).max(12.0 / z) * act;
        // Velocity direction relative to the host, for the curved dust tail.
        let (vx, vy) = (b.vx[i] - b.vx[h], b.vy[i] - b.vy[h]);
        let v = vx.hypot(vy).max(1e-30);
        let (mut tx, mut ty) = (ux - 0.4 * vx / v, uy - 0.4 * vy / v);
        let tn = tx.hypot(ty).max(1e-30);
        tx /= tn;
        ty /= tn;
        let mut tri = |dirx: f64, diry: f64, l: f64, w: f64, col: [u8; 3], a: f64| {
            let (nx, ny) = (-diry, dirx);
            let (cx, cy) = (b.x[i], b.y[i]);
            let alpha = (a * act * 255.0).clamp(0.0, 255.0) as u8;
            let base = [col[0], col[1], col[2], alpha];
            let tip = [col[0], col[1], col[2], 0];
            out.push(OrbitVertex { pos: [(cx + nx * w) as f32, (cy + ny * w) as f32], color: base });
            out.push(OrbitVertex { pos: [(cx - nx * w) as f32, (cy - ny * w) as f32], color: base });
            out.push(OrbitVertex { pos: [(cx + dirx * l) as f32, (cy + diry * l) as f32], color: tip });
        };
        let w_min = 1.0 / z;
        tri(ux, uy, len, (0.03 * len).max(w_min), [120, 170, 255], 0.45);
        tri(tx, ty, 0.65 * len, (0.10 * len).max(1.5 * w_min), [255, 238, 200], 0.30);
    }
    out
}

impl BodyInstance {
    /// Build a `BodyInstance`. The displayed radius is floored at a per-kind
    /// minimum in screen pixels so every body stays visible (black holes 4 px,
    /// stars 3, spawned planets 1.6, belt/debris 1). Non-finite inputs go
    /// off-screen with radius 0.
    pub fn new(
        kind: BodyKind,
        origin: BodyOrigin,
        color: [f32; 3],
        x: f64,
        y: f64,
        r: f64,
        zoom: f32,
    ) -> Self {
        if !x.is_finite() || !y.is_finite() || !r.is_finite() {
            return Self {
                pos: [1e12, 1e12],
                radius: 0.0,
                lit: 0.0,
                color: [0.0; 3],
                light_angle: 0.0,
            };
        }
        let z = zoom.max(1e-6);
        let rubble = origin != BodyOrigin::Spawned;
        let min_px: f64 = match kind {
            BodyKind::Star => 3.0,
            BodyKind::BrownDwarf => 2.5,
            BodyKind::WhiteDwarf => 2.0,
            BodyKind::NeutronStar => 1.5,
            BodyKind::BlackHole => 4.0,
            BodyKind::Planet => {
                if origin == BodyOrigin::Comet {
                    // A real nucleus is < 10 km across — sub-pixel at any
                    // zoom that also shows the tail. Keep it at 0.7 px so
                    // it reads as a bright point source, not an asteroid.
                    0.7
                } else if rubble {
                    1.0
                } else {
                    1.6
                }
            }
            BodyKind::DwarfPlanet => {
                if rubble {
                    1.0
                } else {
                    1.2
                }
            }
        };
        let world_min = min_px / z as f64;
        let disp_r = r.max(world_min) as f32;
        Self {
            pos: [x as f32, y as f32],
            radius: disp_r,
            lit: 0.0,
            color,
            light_angle: 0.0,
        }
    }

    /// Shade this body as a sphere lit by a star at `angle` radians (world
    /// axes, pointing from the body to the star). Planets use this.
    #[inline]
    pub fn with_light(mut self, angle: f64) -> Self {
        if angle.is_finite() {
            self.lit = 1.0;
            self.light_angle = angle as f32;
        }
        self
    }
}

// ---------------------------------------------------------------------------
// Orbit line builder (runs on the physics thread)
// ---------------------------------------------------------------------------

/// Flatten `(start, end, color)` segments into a line-list vertex buffer.
pub fn segments_to_vertices(segs: &[([f32; 2], [f32; 2], [u8; 4])]) -> Vec<OrbitVertex> {
    let mut verts = Vec::with_capacity(segs.len() * 2);
    for (p0, p1, color) in segs {
        verts.push(OrbitVertex {
            pos: *p0,
            color: *color,
        });
        verts.push(OrbitVertex {
            pos: *p1,
            color: *color,
        });
    }
    verts
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

/// All GPU resources for the scene. Buffers grow by `next_power_of_two()` and
/// are never shrunk.
pub struct Renderer {
    // Bodies.
    body_pipeline: wgpu::RenderPipeline,
    instance_buf: wgpu::Buffer,
    instance_capacity: u64,
    body_count: u32,
    instance_scratch: Vec<BodyInstance>,
    // Per-frame light sources (x, y, luminosity in L☉) and per-body
    // luminosity. Kept between frames to avoid new allocations.
    light_scratch: Vec<(f64, f64, f64)>,
    lum_scratch: Vec<f64>,
    // Body colour computed once per frame (shared by bodies and glow).
    color_scratch: Vec<[f32; 3]>,

    // Orbit lines.
    orbit_pipeline: wgpu::RenderPipeline,
    orbit_buf: wgpu::Buffer,
    orbit_capacity: u64,
    orbit_vert_count: u32,

    // Trails (same vertex format, own buffer).
    trail_buf: wgpu::Buffer,
    trail_capacity: u64,
    trail_vert_count: u32,

    // Comet tails and comas. World-space vertices (like trails). Soft
    // triangles with additive blending, so overlaps add up like light.
    comet_pipeline: wgpu::RenderPipeline,
    comet_tail_buf: wgpu::Buffer,
    comet_tail_capacity: u64,
    comet_tail_vert_count: u32,

    glow_pipeline: wgpu::RenderPipeline,
    glow_buf: wgpu::Buffer,
    glow_capacity: u64,
    glow_count: u32,
    glow_scratch: Vec<GlowInstance>,

    // Flashes.
    flash_pipeline: wgpu::RenderPipeline,
    flash_buf: wgpu::Buffer,
    flash_capacity: u64,
    flash_instance_count: u32,
    flash_scratch: Vec<FlashInstance>,

    // Cameras: relative (bodies, flashes, dust) and world (orbits, trails).
    camera_buf: wgpu::Buffer,
    camera_bg: wgpu::BindGroup,
    orbit_camera_buf: wgpu::Buffer,
    orbit_camera_bg: wgpu::BindGroup,

    // Dust grains.
    dust_pipeline: wgpu::RenderPipeline,
    dust_fields_buf: wgpu::Buffer,
    dust_fields_bg: wgpu::BindGroup,
    dust_buf: wgpu::Buffer,
    dust_capacity: u64,
    dust_count: u32,
    /// Points waiting for upload (filled under the lock).
    dust_scratch: Vec<DustVertex>,
    dust_upload_pending: bool,
    /// Slot -> index into `Bodies::dust_fields`, as of the last rebuild.
    dust_slots: Vec<usize>,
    /// (field index, version) of every field in the last rebuild.
    dust_sig: Vec<(usize, u64)>,
    dust_prev_field_count: usize,
    dust_last_upload: Instant,
    /// Camera-relative centre per slot, uploaded every frame.
    dust_centres: Vec<[f32; 4]>,

    // Black holes (event horizon + accretion disk).
    blackhole_pipeline: wgpu::RenderPipeline,
    blackhole_buf: wgpu::Buffer,
    blackhole_capacity: u64,
    blackhole_count: u32,
    blackhole_scratch: Vec<BlackHoleInstance>,
}

impl Renderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nbody-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });

        // ---- Camera bind group layout (group 0, shared by all pipelines) ----
        let cam_size = std::mem::size_of::<CameraUniform>() as u64;
        let mk_cam_buf = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: cam_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let camera_buf = mk_cam_buf("camera");
        let orbit_camera_buf = mk_cam_buf("orbit-camera");
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let mk_bg = |label, buf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &camera_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            })
        };
        let camera_bg = mk_bg("camera-bg", &camera_buf);
        let orbit_camera_bg = mk_bg("orbit-camera-bg", &orbit_camera_buf);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline-layout"),
            bind_group_layouts: &[Some(&camera_layout)],
            immediate_size: 0,
        });

        let msaa_state = wgpu::MultisampleState {
            count: 4,
            mask: !0,
            alpha_to_coverage_enabled: false,
        };

        let blend_target = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })
        };

        // ---- Body pipeline ----
        let body_instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BodyInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 8,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    // Offset 16, not 12: `lit` fills the gap and keeps
                    // `color` 16-aligned.
                    offset: 16,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 12,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 28,
                    shader_location: 4,
                },
            ],
        };
        let body_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("body-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(body_instance_layout)],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[blend_target(format)],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: msaa_state,
            multiview_mask: None,
            cache: None,
        });

        // ---- Orbit/trail pipeline ----
        let orbit_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OrbitVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Unorm8x4,
                    offset: 8,
                    shader_location: 1,
                },
            ],
        };
        let orbit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("orbit-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_orbit"),
                buffers: &[Some(orbit_layout.clone())],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_orbit"),
                targets: &[blend_target(format)],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: msaa_state,
            multiview_mask: None,
            cache: None,
        });
        // ---- Flash pipeline ----
        let flash_instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<FlashInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 8,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 12,
                    shader_location: 2,
                },
            ],
        };
        let flash_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("flash-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_flash"),
                buffers: &[Some(flash_instance_layout)],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_flash"),
                targets: &[blend_target(format)],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: msaa_state,
            multiview_mask: None,
            cache: None,
        });

        // ---- Dust pipeline ----
        // Group 1: the per-field centres, one vec4 per field.
        let fields_size = (MAX_DUST_FIELDS * std::mem::size_of::<[f32; 4]>()) as u64;
        let dust_fields_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dust-fields"),
            size: fields_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dust_fields_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dust-fields-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let dust_fields_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dust-fields-bg"),
            layout: &dust_fields_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: dust_fields_buf.as_entire_binding(),
            }],
        });
        let dust_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dust-pipeline-layout"),
            bind_group_layouts: &[Some(&camera_layout), Some(&dust_fields_layout)],
            immediate_size: 0,
        });
        let dust_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<DustVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 8,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 12,
                    shader_location: 2,
                },
            ],
        };
        // Additive: colour = src.rgb * src.a + dst, so overlapping grains
        // brighten. Destination alpha is left alone.
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let dust_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dust-pipeline"),
            layout: Some(&dust_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_dust"),
                buffers: &[Some(dust_vertex_layout)],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_dust"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(additive),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: msaa_state,
            multiview_mask: None,
            cache: None,
        });
        // ---- Comet pipeline: orbit vertex format, triangles, additive ----
        let comet_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("comet-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_orbit"),
                buffers: &[Some(orbit_layout)],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_orbit"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(additive),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: msaa_state,
            multiview_mask: None,
            cache: None,
        });

        // ---- Glow pipeline: additive soft quads behind bodies ----
        let glow_instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GlowInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 8,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 12,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 16,
                    shader_location: 3,
                },
            ],
        };
        let glow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("glow-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_glow"),
                buffers: &[Some(glow_instance_layout)],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_glow"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(additive),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: msaa_state,
            multiview_mask: None,
            cache: None,
        });

        let blackhole_instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BlackHoleInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 8,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 12,
                    shader_location: 2,
                },
            ],
        };
        let blackhole_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blackhole-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_blackhole"),
                buffers: &[Some(blackhole_instance_layout)],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_blackhole"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: msaa_state,
            multiview_mask: None,
            cache: None,
        });

        // 64k grains to start (1 MiB), grown on demand.
        let dust_capacity = 65_536 * std::mem::size_of::<DustVertex>() as u64;
        let dust_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dust"),
            size: dust_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- Initial buffers ----
        let instance_capacity = 1024 * 16;
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance"),
            size: instance_capacity * std::mem::size_of::<BodyInstance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let orbit_capacity = 1024 * 1024;
        let orbit_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orbit"),
            size: orbit_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let trail_capacity = 1024 * 16;
        let trail_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trail"),
            size: trail_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let comet_tail_capacity = 1024 * 16;
        let comet_tail_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("comet-tail"),
            size: comet_tail_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let flash_capacity = 1024 * 64;
        let flash_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flash"),
            size: flash_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Glow is one instance per luminous body, so a few hundred slots
        // is plenty at every realistic n. Grown on demand if exceeded.
        let glow_capacity: u64 = 256 * std::mem::size_of::<GlowInstance>() as u64;
        let glow_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glow"),
            size: glow_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blackhole_capacity: u64 = 64 * std::mem::size_of::<BlackHoleInstance>() as u64;
        let blackhole_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blackhole"),
            size: blackhole_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            body_pipeline,
            instance_buf,
            instance_capacity,
            body_count: 0,
            instance_scratch: Vec::new(),
            light_scratch: Vec::new(),
            lum_scratch: Vec::new(),
            color_scratch: Vec::new(),
            orbit_pipeline,
            orbit_buf,
            orbit_capacity,
            orbit_vert_count: 0,
            trail_buf,
            trail_capacity,
            trail_vert_count: 0,
            comet_pipeline,
            comet_tail_buf,
            comet_tail_capacity,
            comet_tail_vert_count: 0,
            glow_pipeline,
            glow_buf,
            glow_capacity,
            glow_count: 0,
            glow_scratch: Vec::new(),
            blackhole_pipeline,
            blackhole_buf,
            blackhole_capacity,
            blackhole_count: 0,
            blackhole_scratch: Vec::new(),
            flash_pipeline,
            flash_buf,
            flash_capacity,
            flash_instance_count: 0,
            flash_scratch: Vec::new(),
            camera_buf,
            camera_bg,
            orbit_camera_buf,
            orbit_camera_bg,
            dust_pipeline,
            dust_fields_buf,
            dust_fields_bg,
            dust_buf,
            dust_capacity,
            dust_count: 0,
            dust_scratch: Vec::new(),
            dust_upload_pending: false,
            dust_slots: Vec::new(),
            dust_sig: Vec::new(),
            dust_prev_field_count: 0,
            dust_last_upload: Instant::now(),
            dust_centres: vec![[1e12, 1e12, 0.0, 0.0]; MAX_DUST_FIELDS],
        }
    }

    /// Upload both camera uniforms: relative (pos = 0) for bodies, flashes,
    /// dust, glow, black holes and trails; world (pos = cam) for orbit lines.
    pub fn update_camera(&self, queue: &wgpu::Queue, cam: [f64; 2], zoom: f32, viewport: [f32; 2]) {
        let rel = CameraUniform {
            pos: [0.0, 0.0],
            zoom,
            _pad0: 0.0,
            viewport,
            _pad1: [0.0; 2],
        };
        let world = CameraUniform {
            pos: [cam[0] as f32, cam[1] as f32],
            ..rel
        };
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&rel));
        queue.write_buffer(&self.orbit_camera_buf, 0, bytemuck::bytes_of(&world));
    }

    pub fn build_instances(
        &mut self,
        b: &Bodies,
        zoom: f32,
        cam: [f64; 2],
        vp: [f32; 2],
        override_body: Option<(usize, [f64; 2])>,
    ) {
        // ---- Luminosity and light sources ----
        // One pass computes each luminous body's luminosity (used for the
        // glow strength below) and lists the bodies that light planets.
        let n = b.len();
        self.lum_scratch.clear();
        self.lum_scratch.resize(n, 0.0);
        self.light_scratch.clear();
        self.color_scratch.clear();
        self.color_scratch.resize(n, [0.0; 3]);
        for i in 0..n {
            if !matches!(
                b.kind[i],
                BodyKind::Star
                    | BodyKind::BrownDwarf
                    | BodyKind::WhiteDwarf
                    | BodyKind::NeutronStar
            ) {
                continue;
            }
            let l = physics::luminosity_of(b, i);
            if !(l.is_finite() && l > 0.0) {
                continue;
            }
            self.lum_scratch[i] = l;
            let (lx, ly) = match override_body {
                Some((idx, [x, y])) if idx == i => (x, y),
                _ => (b.x[i], b.y[i]),
            };
            self.light_scratch.push((lx, ly, l));
        }
        // Shading a planet costs one loop over the light sources. In a
        // dense cluster that would be too slow, so planets stay flat there.
        let shade_planets = self.light_scratch.len() as u64 * n as u64 <= 4_000_000;

        // ---- Bodies ----
        self.instance_scratch.clear();
        if self.instance_scratch.capacity() < n {
            self.instance_scratch
                .reserve(n - self.instance_scratch.capacity());
        }
        for i in 0..n {
            if b.kind[i] == BodyKind::BlackHole {
                continue; // Handled by the dedicated pipeline
            }
            let (bx, by) = match override_body {
                Some((idx, [x, y])) if idx == i => (x, y),
                _ => (b.x[i], b.y[i]),
            };
            let color = body_color(b, i);
            self.color_scratch[i] = color;
            let mut inst = BodyInstance::new(
                b.kind[i],
                b.origin[i],
                color,
                bx - cam[0],
                by - cam[1],
                b.r[i],
                zoom,
            );
            // A planet is lit from the side of the star that gives it the
            // most light (luminosity / distance²).
            if shade_planets && matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
                let mut best_flux = 0.0f64;
                let mut best_angle = f64::NAN;
                for &(lx, ly, l) in &self.light_scratch {
                    let (dx, dy) = (lx - bx, ly - by);
                    let d2 = dx * dx + dy * dy;
                    if d2 <= 0.0 {
                        continue;
                    }
                    let flux = l / d2;
                    if flux > best_flux {
                        best_flux = flux;
                        best_angle = dy.atan2(dx);
                    }
                }
                inst = inst.with_light(best_angle);
            }
            self.instance_scratch.push(inst);
        }
        self.body_count = self.instance_scratch.len() as u32;

        // ---- Flashes ----
        let nf = b.flashes.len();
        self.flash_scratch.clear();
        self.flash_instance_count = nf as u32;
        if nf > 0 {
            if self.flash_scratch.capacity() < nf {
                self.flash_scratch
                    .reserve(nf - self.flash_scratch.capacity());
            }
            let min_px_world = 8.0 / zoom.max(1e-6) as f64;
            for f in &b.flashes {
                let r = f.r0.max(min_px_world);
                self.flash_scratch.push(FlashInstance {
                    pos: [(f.x - cam[0]) as f32, (f.y - cam[1]) as f32],
                    radius: r as f32,
                    progress: (f.age / f.life) as f32,
                });
            }
        }

        // ---- Star glow ----
        // One halo per luminous body. Radius and intensity are per-kind:
        // stars get a big soft halo, brown dwarfs a modest one, white
        // dwarfs a tight cool halo, neutron stars a pinprick of intense
        // blue-white. Black holes are skipped (they would read as a
        // spurious emission ring).
        self.glow_scratch.clear();
        for i in 0..n {
            // Stars, brown dwarfs and white dwarfs glow in proportion to their
            // luminosity. A neutron star keeps a fixed pinprick: most of its
            // light is X-rays.
            let (halo_mult, base_intensity, uses_lum) = match b.kind[i] {
                BodyKind::Star => (3.0, 0.55, true),
                BodyKind::BrownDwarf => (2.5, 0.30, true),
                BodyKind::WhiteDwarf => (2.0, 0.45, true),
                BodyKind::NeutronStar => (2.5, 0.90, false),
                _ => continue,
            };
            let lum_factor = if uses_lum {
                glow_luminosity_factor(self.lum_scratch[i])
            } else {
                1.0
            };
            let intensity = (base_intensity * lum_factor) as f32;
            let (bx, by) = match override_body {
                Some((idx, [x, y])) if idx == i => (x, y),
                _ => (b.x[i], b.y[i]),
            };
            // Halo radius in world units. The floor is 6 screen pixels for a
            // Sun-like body, smaller for a dim one and larger for a bright
            // one, so a luminous star at wide zoom has a bigger glare.
            let floor_px = 6.0 * lum_factor.sqrt();
            let r_world = (b.r[i] * halo_mult).max(floor_px / zoom.max(1e-6) as f64);
            let color = self.color_scratch[i];
            self.glow_scratch.push(GlowInstance {
                pos: [(bx - cam[0]) as f32, (by - cam[1]) as f32],
                radius: r_world as f32,
                intensity,
                color,
                _pad: 0.0,
            });
        }
        self.glow_count = self.glow_scratch.len() as u32;

        // ---- Black Holes ----
        self.blackhole_scratch.clear();
        for i in 0..n {
            if b.kind[i] != BodyKind::BlackHole {
                continue;
            }
            let (bx, by) = match override_body {
                Some((idx, [x, y])) if idx == i => (x, y),
                _ => (b.x[i], b.y[i]),
            };
            // `r` is the event horizon (Schwarzschild radius). The shader draws
            // the shadow at BH_SHADOW_IN_RS times this, and the disk beyond it.
            // The floor keeps the shadow at least 4 px in radius, so a hole
            // does not vanish at wide zoom.
            let r_world = b.r[i].max(4.0 / (BH_SHADOW_IN_RS * zoom.max(1e-6) as f64));
            self.blackhole_scratch.push(BlackHoleInstance {
                pos: [(bx - cam[0]) as f32, (by - cam[1]) as f32],
                radius: r_world as f32,
                accretion: physics::accretion_level(b, i) as f32,
            });
        }
        self.blackhole_count = self.blackhole_scratch.len() as u32;

        // ---- Dust ----
        // Visible fields: big enough on screen and overlapping the view.
        let z = zoom.max(1e-6) as f64;
        let (hw, hh) = (0.5 * vp[0] as f64 / z, 0.5 * vp[1] as f64 / z);
        let mut visible: Vec<usize> = Vec::new();
        for (i, f) in b.dust_fields.iter().enumerate() {
            if visible.len() >= MAX_DUST_FIELDS {
                break;
            }
            if f.primary >= n || f.grains.is_empty() {
                continue;
            }
            let (dx, dy) = (b.x[f.primary] - cam[0], b.y[f.primary] - cam[1]);
            let ext = f.extent_gm;
            if ext * z >= DUST_MIN_PX && dx.abs() - ext < hw && dy.abs() - ext < hh {
                visible.push(i);
            }
        }
        let sig: Vec<(usize, u64)> = visible
            .iter()
            .map(|&i| (i, b.dust_fields[i].version))
            .collect();
        let same_set = sig.len() == self.dust_sig.len()
            && sig.iter().zip(&self.dust_sig).all(|(a, c)| a.0 == c.0)
            && b.dust_fields.len() == self.dust_prev_field_count;
        let changed = sig != self.dust_sig || b.dust_fields.len() != self.dust_prev_field_count;
        // A different set of fields rebuilds at once; plain grain motion is
        // throttled to DUST_UPLOAD_MIN.
        if changed && (!same_set || self.dust_last_upload.elapsed() >= DUST_UPLOAD_MIN) {
            let total: usize = visible.iter().map(|&i| b.dust_fields[i].grains.len()).sum();
            let stride = total.div_ceil(MAX_DUST_POINTS).max(1);
            self.dust_scratch.clear();
            self.dust_scratch.reserve(total / stride + 1);
            for (slot, &i) in visible.iter().enumerate() {
                let f = &b.dust_fields[i];
                let (px, py) = (b.x[f.primary], b.y[f.primary]);
                self.dust_scratch
                    .extend(f.grains.iter().step_by(stride).map(|g| DustVertex {
                        pos: [(g.x - px) as f32, (g.y - py) as f32],
                        weight: g.mass as f32,
                        field: slot as u32,
                    }));
            }
            self.dust_slots = visible;
            self.dust_sig = sig;
            self.dust_prev_field_count = b.dust_fields.len();
            self.dust_last_upload = Instant::now();
            self.dust_upload_pending = true;
        }
        // Centres follow the primaries every frame (cheap, one small uniform).
        for c in self.dust_centres.iter_mut() {
            *c = [1e12, 1e12, 0.0, 0.0];
        }
        for (slot, &i) in self.dust_slots.iter().enumerate() {
            if let Some(f) = b.dust_fields.get(i) {
                if f.primary < n {
                    self.dust_centres[slot] = [
                        (b.x[f.primary] - cam[0]) as f32,
                        (b.y[f.primary] - cam[1]) as f32,
                        0.0,
                        0.0,
                    ];
                }
            }
        }
    }

    /// Upload the scratch buffers to the GPU. Does not read `Bodies`.
    pub fn upload_instances(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // ---- Bodies ----
        let n = self.body_count as u64;
        if n > self.instance_capacity {
            self.instance_capacity = n.next_power_of_two();
            self.instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instance"),
                size: self.instance_capacity * std::mem::size_of::<BodyInstance>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if n > 0 {
            queue.write_buffer(
                &self.instance_buf,
                0,
                bytemuck::cast_slice(&self.instance_scratch),
            );
        }

        // ---- Flashes ----
        let nf = self.flash_instance_count as u64;
        if nf > 0 {
            let byte_len = nf * std::mem::size_of::<FlashInstance>() as u64;
            if byte_len > self.flash_capacity {
                self.flash_capacity = byte_len.next_power_of_two();
                self.flash_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("flash"),
                    size: self.flash_capacity,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(
                &self.flash_buf,
                0,
                bytemuck::cast_slice(&self.flash_scratch),
            );
        }

        // ---- Glow ----
        let ng = self.glow_count as u64;
        if ng > 0 {
            let byte_len = ng * std::mem::size_of::<GlowInstance>() as u64;
            if byte_len > self.glow_capacity {
                self.glow_capacity = byte_len.next_power_of_two();
                self.glow_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("glow"),
                    size: self.glow_capacity,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(&self.glow_buf, 0, bytemuck::cast_slice(&self.glow_scratch));
        }

        // ---- Black Holes ----
        let nb = self.blackhole_count as u64;
        if nb > 0 {
            let byte_len = nb * std::mem::size_of::<BlackHoleInstance>() as u64;
            if byte_len > self.blackhole_capacity {
                self.blackhole_capacity = byte_len.next_power_of_two();
                self.blackhole_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("blackhole"),
                    size: self.blackhole_capacity,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(
                &self.blackhole_buf,
                0,
                bytemuck::cast_slice(&self.blackhole_scratch),
            );
        }

        // ---- Dust ----
        if self.dust_upload_pending {
            self.dust_upload_pending = false;
            let nd = self.dust_scratch.len() as u64;
            self.dust_count = nd as u32;
            if nd > 0 {
                let bytes = nd * std::mem::size_of::<DustVertex>() as u64;
                if bytes > self.dust_capacity {
                    self.dust_capacity = bytes.next_power_of_two();
                    self.dust_buf = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("dust"),
                        size: self.dust_capacity,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                }
                queue.write_buffer(&self.dust_buf, 0, bytemuck::cast_slice(&self.dust_scratch));
            }
        }
        queue.write_buffer(
            &self.dust_fields_buf,
            0,
            bytemuck::cast_slice(&self.dust_centres),
        );
    }

    /// Upload a pre-built orbit-vertex buffer. Empty input clears the orbits.
    pub fn set_orbit_vertices(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        verts: &[OrbitVertex],
    ) {
        if verts.is_empty() {
            self.orbit_vert_count = 0;
            return;
        }
        let byte_len = std::mem::size_of_val(verts) as u64;
        if byte_len > self.orbit_capacity {
            self.orbit_capacity = byte_len.next_power_of_two();
            self.orbit_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("orbit"),
                size: self.orbit_capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.orbit_buf, 0, bytemuck::cast_slice(verts));
        self.orbit_vert_count = verts.len() as u32;
    }

    /// Upload a pre-built trail-vertex buffer (same pipeline as orbits, own
    /// buffer so the two update cadences do not interfere). Positions must be
    /// camera-relative: trails are drawn with the relative camera.
    pub fn set_trail_vertices(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        verts: &[OrbitVertex],
    ) {
        if verts.is_empty() {
            self.trail_vert_count = 0;
            return;
        }
        let byte_len = std::mem::size_of_val(verts) as u64;
        if byte_len > self.trail_capacity {
            self.trail_capacity = byte_len.next_power_of_two();
            self.trail_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("trail"),
                size: self.trail_capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.trail_buf, 0, bytemuck::cast_slice(verts));
        self.trail_vert_count = verts.len() as u32;
    }

    /// Upload comet tail and coma vertices. World-space coordinates, same
    /// vertex format and pipeline as orbits/trails, so alpha blends the
    /// same way.
    pub fn set_comet_tail_vertices(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        verts: &[OrbitVertex],
    ) {
        if verts.is_empty() {
            self.comet_tail_vert_count = 0;
            return;
        }
        let byte_len = std::mem::size_of_val(verts) as u64;
        if byte_len > self.comet_tail_capacity {
            self.comet_tail_capacity = byte_len.next_power_of_two();
            self.comet_tail_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("comet-tail"),
                size: self.comet_tail_capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.comet_tail_buf, 0, bytemuck::cast_slice(verts));
        self.comet_tail_vert_count = verts.len() as u32;
    }
    /// Number of vertices in the trail buffer (`main.rs` uses it to detect
    /// "trails were just turned off").
    pub fn trail_count(&self) -> u32 {
        self.trail_vert_count
    }

    /// Record the draw calls. Order, bottom to top: glow, dust, trails,
    /// comet tails, orbits, black holes, flashes, bodies. `pass` is `'static`
    /// because the caller erased the encoder borrow with `forget_lifetime()`.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'static>) {
        // Glow first: halos read as background scatter, so bodies, trails
        // and flashes all draw on top of them. Additive blending means
        // order among glow-quads is irrelevant, but this placement keeps
        // trails and orbits crisp.
        if self.glow_count > 0 {
            pass.set_pipeline(&self.glow_pipeline);
            pass.set_bind_group(0, &self.camera_bg, &[]);
            pass.set_vertex_buffer(0, self.glow_buf.slice(..));
            pass.draw(0..4, 0..self.glow_count);
        }
        if self.dust_count > 0 {
            pass.set_pipeline(&self.dust_pipeline);
            pass.set_bind_group(0, &self.camera_bg, &[]);
            pass.set_bind_group(1, &self.dust_fields_bg, &[]);
            pass.set_vertex_buffer(0, self.dust_buf.slice(..));
            pass.draw(0..4, 0..self.dust_count);
        }
        if self.trail_vert_count > 0 {
            // Trail vertices are camera-relative (see `TrailState::build_vertices`).
            pass.set_pipeline(&self.orbit_pipeline);
            pass.set_bind_group(0, &self.camera_bg, &[]);
            pass.set_vertex_buffer(0, self.trail_buf.slice(..));
            pass.draw(0..self.trail_vert_count, 0..1);
        }
        if self.comet_tail_vert_count > 0 {
            pass.set_pipeline(&self.comet_pipeline);
            pass.set_bind_group(0, &self.orbit_camera_bg, &[]);
            pass.set_vertex_buffer(0, self.comet_tail_buf.slice(..));
            pass.draw(0..self.comet_tail_vert_count, 0..1);
        }
        if self.orbit_vert_count > 0 {
            pass.set_pipeline(&self.orbit_pipeline);
            pass.set_bind_group(0, &self.orbit_camera_bg, &[]);
            pass.set_vertex_buffer(0, self.orbit_buf.slice(..));
            pass.draw(0..self.orbit_vert_count, 0..1);
        }
        if self.blackhole_count > 0 {
            pass.set_pipeline(&self.blackhole_pipeline);
            pass.set_bind_group(0, &self.camera_bg, &[]);
            pass.set_vertex_buffer(0, self.blackhole_buf.slice(..));
            pass.draw(0..4, 0..self.blackhole_count);
        }
        if self.flash_instance_count > 0 {
            pass.set_pipeline(&self.flash_pipeline);
            pass.set_bind_group(0, &self.camera_bg, &[]);
            pass.set_vertex_buffer(0, self.flash_buf.slice(..));
            pass.draw(0..4, 0..self.flash_instance_count);
        }
        if self.body_count > 0 {
            pass.set_pipeline(&self.body_pipeline);
            pass.set_bind_group(0, &self.camera_bg, &[]);
            pass.set_vertex_buffer(0, self.instance_buf.slice(..));
            pass.draw(0..4, 0..self.body_count);
        }
    }
}
