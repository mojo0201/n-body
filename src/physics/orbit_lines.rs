//! Orbit lines: adaptive point count driven by apparent on-screen size.
//!
//! Each body's osculating ellipse is sampled at a resolution proportional to
//! its circumference in pixels, so orbits stay smooth when zoomed in and cheap
//! when zoomed out. Trig values come from a shared 512-entry lookup table;
//! sampling always uses a power-of-two stride into that table.

use super::*;

pub(super) const ORBIT_TABLE_SIZE: usize = 512;

pub(super) const ORBIT_PIXELS_PER_SEG: f64 = 20.0;

pub(super) const ORBIT_MAX_BODIES: usize = 4000;

/// Lazily-built sine/cosine tables over `[0, 2π)`, uniformly sampled.
///
/// Sampling an orbit at stride `ORBIT_TABLE_SIZE / n` visits `n` evenly spaced
/// angles without ever calling `sin`/`cos` in the hot loop.
pub(super) fn orbit_trig() -> &'static (Vec<f64>, Vec<f64>) {
    static T: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    T.get_or_init(|| {
        let mut s = Vec::with_capacity(ORBIT_TABLE_SIZE);
        let mut c = Vec::with_capacity(ORBIT_TABLE_SIZE);
        for k in 0..ORBIT_TABLE_SIZE {
            let th = (k as f64 / ORBIT_TABLE_SIZE as f64) * std::f64::consts::TAU;
            s.push(th.sin());
            c.push(th.cos());
        }
        (s, c)
    })
}

/// A fully resolved orbit ellipse, expressed in world-space coordinates.
///
/// The center already includes the reference-body offset, so emitting a point
/// is a single rotation about `(cx, cy)` with no further translation.
struct OrbitEllipse {
    /// Centre of the ellipse.
    c: [f64; 3],
    /// Semi-major axis (a) times the unit vector toward periapsis.
    pa: [f64; 3],
    /// Semi-minor axis (b) times the in-plane unit vector 90° ahead of it.
    qb: [f64; 3],
    a: f64,
    n_pts: usize,
    stride: usize,
}

impl OrbitEllipse {
    /// Point at table angle index `idx`.
    #[inline]
    fn point(&self, cos_t: f64, sin_t: f64) -> [f64; 3] {
        [
            self.c[0] + cos_t * self.pa[0] + sin_t * self.qb[0],
            self.c[1] + cos_t * self.pa[1] + sin_t * self.qb[1],
            self.c[2] + cos_t * self.pa[2] + sin_t * self.qb[2],
        ]
    }
}

/// Solve the two-body problem for body `i` relative to a reference state and
/// return the resulting ellipse, or `None` when the orbit is degenerate,
/// hyperbolic, implausibly large, or nearly parabolic.
///
/// All inputs are the reference body's state (`s*`) and the body's own state
/// from `Bodies`; the returned geometry is independent of any render context,
/// which lets both the public `osculating_ellipse` and the batched segment
/// builder share exactly the same math.
#[inline]
fn compute_orbit_ellipse(
    b: &Bodies,
    i: usize,
    sx: f64,
    sy: f64,
    sz: f64,
    svx: f64,
    svy: f64,
    svz: f64,
    sm: f64,
    zoom: f64,
) -> Option<OrbitEllipse> {
    // Relative state vector and squared distance.
    let dx = b.x[i] - sx;
    let dy = b.y[i] - sy;
    let dz = b.z[i] - sz;
    let dvx = b.vx[i] - svx;
    let dvy = b.vy[i] - svy;
    let dvz = b.vz[i] - svz;
    let r2 = dx * dx + dy * dy + dz * dz;
    if r2 < 1e-18 {
        return None;
    }
    let r = r2.sqrt();
    let v2 = dvx * dvx + dvy * dvy + dvz * dvz;
    let mu = G * (sm + b.m[i]);

    // Reciprocal semi-major axis from the vis-viva equation.
    // Non-positive means the body is on a parabolic or hyperbolic path.
    let inv_a = 2.0 / r - v2 / mu;
    if inv_a <= 1e-9 {
        return None;
    }
    let a = 1.0 / inv_a;
    if a > 100_000.0 {
        return None;
    }

    // Eccentricity vector points at periapsis; its magnitude is `e`.
    let rv = dx * dvx + dy * dvy + dz * dvz;
    let km = v2 - mu / r;
    let ex = (km * dx - rv * dvx) / mu;
    let ey = (km * dy - rv * dvy) / mu;
    let ez = (km * dz - rv * dvz) / mu;
    let e2 = ex * ex + ey * ey + ez * ez;
    // Reject e >= 0.9999 (near-parabolic) using e² to skip a sqrt.
    if e2 >= 0.9999 * 0.9999 {
        return None;
    }

    // Semi-minor axis, and the in-plane frame: p̂ toward periapsis (or the
    // current position for a circle), q̂ = ĥ × p̂ with h = r × v.
    let b_ax = a * (1.0 - e2).sqrt();
    let e_len = e2.sqrt();
    let (px, py, pz) = if e_len > 1e-9 {
        (ex / e_len, ey / e_len, ez / e_len)
    } else {
        (dx / r, dy / r, dz / r)
    };
    let (hx, hy, hz) = (dy * dvz - dz * dvy, dz * dvx - dx * dvz, dx * dvy - dy * dvx);
    let hn = (hx * hx + hy * hy + hz * hz).sqrt();
    let (qx, qy, qz) = if hn > 1e-30 {
        let (hx, hy, hz) = (hx / hn, hy / hn, hz / hn);
        (hy * pz - hz * py, hz * px - hx * pz, hx * py - hy * px)
    } else {
        (-py, px, 0.0) // radial plunge: any perpendicular will do
    };

    // The ellipse is centred at -a·p̂·e from the focus (reference body).
    let c = [-a * ex + sx, -a * ey + sy, -a * ez + sz];

    // Resolution: one sample per ~20 screen pixels of arc length, clamped to
    // [32, ORBIT_TABLE_SIZE] and rounded to a power of two so the stride into
    // the trig table divides evenly.
    let circum_px = a * zoom.max(0.001) * std::f64::consts::TAU;
    let n_pts_raw = (circum_px / ORBIT_PIXELS_PER_SEG).round() as usize;
    let n_pts = n_pts_raw.clamp(32, ORBIT_TABLE_SIZE).next_power_of_two();
    let stride = ORBIT_TABLE_SIZE / n_pts;

    Some(OrbitEllipse {
        c,
        pa: [px * a, py * a, pz * a],
        qb: [qx * b_ax, qy * b_ax, qz * b_ax],
        a,
        n_pts,
        stride,
    })
}

/// Sample the osculating ellipse of body `i` about the reference state and
/// return the closed polyline as world-space points.
#[allow(clippy::too_many_arguments)]
pub fn osculating_ellipse(
    b: &Bodies,
    i: usize,
    sx: f64,
    sy: f64,
    sz: f64,
    svx: f64,
    svy: f64,
    svz: f64,
    sm: f64,
    zoom: f64,
) -> Option<Vec<[f32; 3]>> {
    let el = compute_orbit_ellipse(b, i, sx, sy, sz, svx, svy, svz, sm, zoom)?;
    let (sin_t, cos_t) = orbit_trig();
    let mut pts = Vec::with_capacity(el.n_pts);
    for k in 0..el.n_pts {
        let idx = k * el.stride;
        let p = el.point(cos_t[idx], sin_t[idx]);
        pts.push([p[0] as f32, p[1] as f32, p[2] as f32]);
    }
    Some(pts)
}

/// Build drawable line segments for every visible orbit.
///
/// Planets are sub-sampled only if they exceed `ORBIT_MAX_BODIES`; debris is
/// capped at 256 visible rings to keep the segment buffer bounded. Ellipses
/// are emitted directly into the output vector — no intermediate allocation
/// per body — since this function is on the per-frame hot path.
pub fn compute_orbit_segments(
    b: &Bodies,
    zoom: f64,
    show_planets: bool,
    show_debris: bool,
) -> Vec<([f32; 3], [f32; 3], [u8; 4])> {
    let mut segs = compute_orbit_segments_inner(b, zoom, show_planets, show_debris);
    if hz_rings() {
        habitable_zone_segments(b, zoom, &mut segs);
    }
    segs
}

static HZ_RINGS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

static COMET_ORBITS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Draw comet orbits (on/off). Comets have their own toggle, separate from
/// the planet and debris orbits, and it starts off.
pub fn set_comet_orbits(on: bool) {
    COMET_ORBITS.store(on, std::sync::atomic::Ordering::Relaxed);
}
pub fn comet_orbits() -> bool {
    COMET_ORBITS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Draw each star's habitable zone as a green band (on/off).
pub fn set_hz_rings(on: bool) {
    HZ_RINGS.store(on, std::sync::atomic::Ordering::Relaxed);
}
pub fn hz_rings() -> bool {
    HZ_RINGS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Band of circles between the inner and outer habitable-zone radius of every
/// star (the limits come from `habitable_zone_of`: luminosity and colour).
fn habitable_zone_segments(b: &Bodies, zoom: f64, segs: &mut Vec<([f32; 3], [f32; 3], [u8; 4])>) {
    const RINGS: usize = 7;
    const SEGS: usize = 120;
    let au = au_in_gm();
    for i in 0..b.len() {
        if b.dead[i] || b.kind[i] != BodyKind::Star {
            continue;
        }
        let (r0, r1) = habitable_zone_of(b, i);
        if !(r0.is_finite() && r1.is_finite()) || r1 <= r0 {
            continue;
        }
        let (r0, r1) = (r0 * au, r1 * au);
        if r1 * zoom < 3.0 {
            continue; // smaller than a few pixels
        }
        for k in 0..RINGS {
            let f = k as f64 / (RINGS - 1) as f64;
            let r = r0 + (r1 - r0) * f;
            let edge = k == 0 || k == RINGS - 1;
            let col = [60u8, 210, 90, if edge { 150 } else { 45 }];
            let pt = |a: f64| {
                [(b.x[i] + r * a.cos()) as f32, (b.y[i] + r * a.sin()) as f32, b.z[i] as f32]
            };
            let mut prev = pt(0.0);
            for s in 1..=SEGS {
                let cur = pt(std::f64::consts::TAU * s as f64 / SEGS as f64);
                segs.push((prev, cur, col));
                prev = cur;
            }
        }
    }
}

fn compute_orbit_segments_inner(
    b: &Bodies,
    zoom: f64,
    show_planets: bool,
    show_debris: bool,
) -> Vec<([f32; 3], [f32; 3], [u8; 4])> {
    let show_comets = comet_orbits();
    if !show_planets && !show_debris && !show_comets {
        return Vec::new();
    }
    let n = b.len();
    if n < 2 {
        return Vec::new();
    }
    let Some(si) = primary_index(b) else {
        return Vec::new();
    };
    let sx = b.x[si];
    let sy = b.y[si];
    let sz = b.z[si];
    let svx = b.vx[si];
    let svy = b.vy[si];
    let svz = b.vz[si];
    let sm = b.m[si];
    let ctx = build_parent_context(b);
    let hosts = &ctx.hosts;

    // First pass: count visible bodies so the stride and capacity below are
    // exact. Cheap — one branch and one counter per body.
    let mut n_planets = 0usize;
    let mut n_debris = 0usize;
    for i in 0..n {
        if i == si {
            continue;
        }
        if b.origin[i] == BodyOrigin::Comet {
            continue;
        }
        if b.origin[i].is_belt() || b.origin[i].is_rubble() {
            if show_debris {
                n_debris += 1;
            }
        } else if show_planets {
            n_planets += 1;
        }
    }
    let planet_stride = n_planets.div_ceil(ORBIT_MAX_BODIES).max(1);
    let debris_stride = n_debris.div_ceil(256).max(1);

    // Pre-size assuming the worst case of 64 segments per visible body; the
    // adaptive count can exceed that for very large on-screen ellipses but
    // rarely does, so this avoids most reallocations without overshooting.
    let mut segs = Vec::with_capacity((n_planets.min(ORBIT_MAX_BODIES) + 256).saturating_mul(64));
    let mut seen_planet = 0usize;
    let mut seen_debris = 0usize;

    // Invariants that don't depend on the body: hoist them out of the loop.
    let (sin_t, cos_t) = orbit_trig();
    let debris_color = [85u8, 85u8, 90u8, 110u8];
    let planet_green = (0.55 * 255.0) as u8;

    for i in 0..n {
        if i == si {
            continue;
        }

        // Comets draw only with their own toggle, and are never decimated.
        let is_comet = b.origin[i] == BodyOrigin::Comet;
        if is_comet && !show_comets {
            continue;
        }
        // Stride-based decimation for planets and debris, respectively.
        let is_debris = b.origin[i].is_belt() || b.origin[i].is_rubble();
        if is_comet {
            // fall through: drawn undecimated
        } else if is_debris {
            if !show_debris {
                continue;
            }
            let skip = !seen_debris.is_multiple_of(debris_stride);
            seen_debris += 1;
            if skip {
                continue;
            }
        } else {
            if !show_planets {
                continue;
            }
            let skip = !seen_planet.is_multiple_of(planet_stride);
            seen_planet += 1;
            if skip {
                continue;
            }
        }

        // Use the nearest host (or the primary) as the two-body focus.
        let (rx, ry, rz, rvx, rvy, rvz, rm) = orbit_reference_cached(
            b,
            hosts,
            &ctx.partners,
            i,
            (sx, sy, sz, svx, svy, svz, sm),
        );

        let Some(el) = compute_orbit_ellipse(b, i, rx, ry, rz, rvx, rvy, rvz, rm, zoom) else {
            continue;
        };

        // Debris is flat gray; planet hue shifts toward white with distance
        // from the primary (log-ish ramp over 0–800 units).
        let color = if is_comet {
            [150u8, 200u8, 230u8, 110u8]
        } else if is_debris {
            debris_color
        } else {
            let d = host_distance(b, i) as f32;
            let t = (d / 800.0).clamp(0.0, 1.0);
            [
                ((0.4 + 0.4 * t) * 255.0) as u8,
                planet_green,
                ((1.0 - 0.5 * t) * 255.0) as u8,
                30u8,
            ]
        };

        // Emit the closed loop as N segments. Start from the sample at
        // idx = 0 (angle 0) so the wrap-around edge reuses the first point.
        let f = el.point(1.0, 0.0);
        let first = [f[0] as f32, f[1] as f32, f[2] as f32];
        let mut prev = first;
        for k in 1..el.n_pts {
            let idx = k * el.stride;
            let p = el.point(cos_t[idx], sin_t[idx]);
            let cur = [p[0] as f32, p[1] as f32, p[2] as f32];
            segs.push((prev, cur, color));
            prev = cur;
        }
        segs.push((prev, first, color));
    }
    segs
}
