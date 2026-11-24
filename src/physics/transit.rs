//! Eclipses and transits: the fraction of a star's light that reaches an
//! observer, from other bodies passing in front of it.
//!
//! The observer is far away along the unit vector `obs` (from the system towards
//! the observer). Every other body that is nearer to the observer than the star
//! and whose projected disc overlaps the star's blocks light in proportion to the
//! overlap area, weighted by linear limb darkening (u = 0.6) at the occulter's
//! position. Overlaps between occulters are not corrected for.

use super::*;

/// Linear limb-darkening coefficient.
const LIMB_U: f64 = 0.6;

/// Observer direction from azimuth and elevation above the reference plane, in
/// degrees. Elevation 0 looks along the plane, where orbits are edge-on.
pub fn observer_dir(azimuth_deg: f64, elevation_deg: f64) -> [f64; 3] {
    let (az, el) = (azimuth_deg.to_radians(), elevation_deg.to_radians());
    [el.cos() * az.cos(), el.cos() * az.sin(), el.sin()]
}

/// Overlap area of discs of radii `r1`, `r2` whose centres are `d` apart.
fn lens_area(r1: f64, r2: f64, d: f64) -> f64 {
    if d >= r1 + r2 {
        return 0.0;
    }
    if d <= (r1 - r2).abs() {
        let r = r1.min(r2);
        return std::f64::consts::PI * r * r;
    }
    let a1 = ((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)).clamp(-1.0, 1.0).acos();
    let a2 = ((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)).clamp(-1.0, 1.0).acos();
    let k = ((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)).max(0.0).sqrt();
    r1 * r1 * a1 + r2 * r2 * a2 - 0.5 * k
}

/// Fraction (0..1) of star `s`'s light visible along `obs`.
pub fn visible_fraction(b: &Bodies, s: usize, obs: [f64; 3]) -> f64 {
    let rs = b.r[s];
    if !(rs > 0.0) {
        return 1.0;
    }
    let mean_i = 1.0 - LIMB_U / 3.0;
    let disc = std::f64::consts::PI * rs * rs;
    let mut blocked = 0.0;
    for j in 0..b.len() {
        if j == s || b.dead[j] || !(b.r[j] > 0.0) {
            continue;
        }
        let rel = [b.x[j] - b.x[s], b.y[j] - b.y[s], b.z[j] - b.z[s]];
        let along = rel[0] * obs[0] + rel[1] * obs[1] + rel[2] * obs[2];
        if along <= 0.0 {
            continue;
        }
        let d2 = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2] - along * along).max(0.0);
        let d = d2.sqrt();
        if d >= rs + b.r[j] {
            continue;
        }
        let area = lens_area(rs, b.r[j], d);
        let rho = (d / rs).min(1.0);
        let intensity = 1.0 - LIMB_U * (1.0 - (1.0 - rho * rho).sqrt());
        blocked += area / disc * intensity / mean_i;
    }
    (1.0 - blocked).clamp(0.0, 1.0)
}
