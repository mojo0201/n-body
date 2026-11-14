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
    cx: f64,
    cy: f64,
    a: f64,
    b_ax: f64,
    cr: f64,
    sr: f64,
    n_pts: usize,
    stride: usize,
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
    svx: f64,
    svy: f64,
    sm: f64,
    zoom: f64,
) -> Option<OrbitEllipse> {
    // Relative state vector and squared distance.
    let dx = b.x[i] - sx;
    let dy = b.y[i] - sy;
    let dvx = b.vx[i] - svx;
    let dvy = b.vy[i] - svy;
    let r2 = dx * dx + dy * dy;
    if r2 < 1e-18 {
        return None;
    }
    let r = r2.sqrt();
    let v2 = dvx * dvx + dvy * dvy;
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
    let rv = dx * dvx + dy * dvy;
    let km = v2 - mu / r;
    let ex = (km * dx - rv * dvx) / mu;
    let ey = (km * dy - rv * dvy) / mu;
    let e2 = ex * ex + ey * ey;
    // Reject e >= 0.9999 (near-parabolic) using e² to skip a sqrt.
    if e2 >= 0.9999 * 0.9999 {
        return None;
    }

    // Semi-minor axis and the orientation of the major axis.
    let b_ax = a * (1.0 - e2).sqrt();
    let rot = ey.atan2(ex);
    let cr = rot.cos();
    let sr = rot.sin();

    // The ellipse is centered at -a·ê from the focus (reference body);
    // folding in (sx, sy) makes this the final world-space center.
    let cx = -a * ex + sx;
    let cy = -a * ey + sy;

    // Resolution: one sample per ~20 screen pixels of arc length, clamped to
    // [32, ORBIT_TABLE_SIZE] and rounded to a power of two so the stride into
    // the trig table divides evenly.
    let circum_px = a * zoom.max(0.001) * std::f64::consts::TAU;
    let n_pts_raw = (circum_px / ORBIT_PIXELS_PER_SEG).round() as usize;
    let n_pts = n_pts_raw.clamp(32, ORBIT_TABLE_SIZE).next_power_of_two();
    let stride = ORBIT_TABLE_SIZE / n_pts;

    Some(OrbitEllipse {
        cx,
        cy,
        a,
        b_ax,
        cr,
        sr,
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
    svx: f64,
    svy: f64,
    sm: f64,
    zoom: f64,
) -> Option<Vec<[f32; 2]>> {
    let el = compute_orbit_ellipse(b, i, sx, sy, svx, svy, sm, zoom)?;
    let (sin_t, cos_t) = orbit_trig();
    let mut pts = Vec::with_capacity(el.n_pts);
    for k in 0..el.n_pts {
        // Rotate the unit-circle sample by the major-axis orientation and
        // scale by (a, b); the center already accounts for the offset.
        let idx = k * el.stride;
        let ct = cos_t[idx] * el.a;
        let st = sin_t[idx] * el.b_ax;
        pts.push([
            (el.cx + ct * el.cr - st * el.sr) as f32,
            (el.cy + ct * el.sr + st * el.cr) as f32,
        ]);
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
) -> Vec<([f32; 2], [f32; 2], [u8; 4])> {
    if !show_planets && !show_debris {
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
    let svx = b.vx[si];
    let svy = b.vy[si];
    let sm = b.m[si];
    let hosts = host_list(b);

    // First pass: count visible bodies so the stride and capacity below are
    // exact. Cheap — one branch and one counter per body.
    let mut n_planets = 0usize;
    let mut n_debris = 0usize;
    for i in 0..n {
        if i == si {
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

        // Stride-based decimation for planets and debris, respectively.
        let is_debris = b.origin[i].is_belt() || b.origin[i].is_rubble();
        if is_debris {
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
        let (rx, ry, rvx, rvy, rm) = orbit_reference(b, &hosts, i, (sx, sy, svx, svy, sm));

        let Some(el) = compute_orbit_ellipse(b, i, rx, ry, rvx, rvy, rm, zoom) else {
            continue;
        };

        // Debris is flat gray; planet hue shifts toward white with distance
        // from the primary (log-ish ramp over 0–800 units).
        let color = if is_debris {
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
        // idx = 0 (angle 0 ⇒ ct = a, st = 0) so the wrap-around edge reuses
        // the first point without recomputing it.
        let first_px = (el.cx + el.a * el.cr) as f32;
        let first_py = (el.cy + el.a * el.sr) as f32;
        let mut px_prev = first_px;
        let mut py_prev = first_py;

        for k in 1..el.n_pts {
            let idx = k * el.stride;
            let ct = cos_t[idx] * el.a;
            let st = sin_t[idx] * el.b_ax;
            let px = (el.cx + ct * el.cr - st * el.sr) as f32;
            let py = (el.cy + ct * el.sr + st * el.cr) as f32;
            segs.push(([px_prev, py_prev], [px, py], color));
            px_prev = px;
            py_prev = py;
        }
        segs.push(([px_prev, py_prev], [first_px, first_py], color));
    }
    segs
}
