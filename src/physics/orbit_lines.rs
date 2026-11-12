//! Orbit lines: adaptive point count from apparent screen size.

use super::*;

pub(super) const ORBIT_TABLE_SIZE: usize = 512;

pub(super) const ORBIT_PIXELS_PER_SEG: f64 = 20.0;

pub(super) const ORBIT_MAX_BODIES: usize = 4000;

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
    let dx = b.x[i] - sx;
    let dy = b.y[i] - sy;
    let dvx = b.vx[i] - svx;
    let dvy = b.vy[i] - svy;
    let r = (dx * dx + dy * dy).sqrt();
    if r < 1e-9 {
        return None;
    }
    let v2 = dvx * dvx + dvy * dvy;
    let mu = G * (sm + b.m[i]);
    let inv_a = 2.0 / r - v2 / mu;
    if inv_a <= 1e-9 {
        return None;
    }
    let a = 1.0 / inv_a;
    if a > 100_000.0 {
        return None;
    }
    let rv = dx * dvx + dy * dvy;
    let ex = ((v2 - mu / r) * dx - rv * dvx) / mu;
    let ey = ((v2 - mu / r) * dy - rv * dvy) / mu;
    let e = (ex * ex + ey * ey).sqrt();
    if e >= 0.9999 {
        return None;
    }
    let b_ax = a * (1.0 - e * e).sqrt();
    let rot = ey.atan2(ex);
    let cx = -a * ex;
    let cy = -a * ey;
    let cr = rot.cos();
    let sr = rot.sin();

    let circum_px = a * zoom.max(0.001) * std::f64::consts::TAU;
    let n_pts_raw = (circum_px / ORBIT_PIXELS_PER_SEG).round() as usize;
    let n_pts = n_pts_raw.clamp(32, ORBIT_TABLE_SIZE).next_power_of_two();
    let stride = ORBIT_TABLE_SIZE / n_pts;

    let (sin_t, cos_t) = orbit_trig();
    let mut pts = Vec::with_capacity(n_pts);
    for k in 0..n_pts {
        let idx = k * stride;
        let ct = cos_t[idx] * a;
        let st = sin_t[idx] * b_ax;
        let px = cx + ct * cr - st * sr + sx;
        let py = cy + ct * sr + st * cr + sy;
        pts.push([px as f32, py as f32]);
    }
    Some(pts)
}

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

    let n_planets = if show_planets {
        (0..n)
            .filter(|&i| i != si && !(b.origin[i].is_belt() || b.origin[i].is_rubble()))
            .count()
    } else {
        0
    };
    let n_debris = if show_debris {
        (0..n)
            .filter(|&i| i != si && (b.origin[i].is_belt() || b.origin[i].is_rubble()))
            .count()
    } else {
        0
    };
    let planet_stride = n_planets.div_ceil(ORBIT_MAX_BODIES).max(1);
    let debris_stride = if n_debris <= 256 { 1 } else { n_debris / 256 };

    let mut segs = Vec::with_capacity(n_planets.min(ORBIT_MAX_BODIES) * 64 + 256 * 64);
    let mut seen_planet = 0usize;
    let mut seen_debris = 0usize;

    for i in 0..n {
        if i == si {
            continue;
        }
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

        let (rx, ry, rvx, rvy, rm) = orbit_reference(b, &hosts, i, (sx, sy, svx, svy, sm));
        if let Some(pts) = osculating_ellipse(b, i, rx, ry, rvx, rvy, rm, zoom) {
            let color = if is_debris {
                [85u8, 85u8, 90u8, 110u8]
            } else {
                let d = host_distance(b, i) as f32;
                let t = (d / 800.0).clamp(0.0, 1.0);
                [
                    ((0.4 + 0.4 * t) * 255.0) as u8,
                    (0.55 * 255.0) as u8,
                    ((1.0 - 0.5 * t) * 255.0) as u8,
                    30u8,
                ]
            };
            let m = pts.len();
            for k in 0..m {
                segs.push((pts[k], pts[(k + 1) % m], color));
            }
        }
    }
    segs
}
