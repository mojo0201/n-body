//! Yarkovsky drift for small bodies.
//!
//! A rotating rock absorbs sunlight and re-emits the heat after a delay, so
//! the thermal recoil has a component along its orbital velocity. Prograde
//! rotators gain energy and drift outward; retrograde rotators drift inward.
//! The transverse acceleration is modelled as
//!
//!     a_t = f · (π R² S / c) / m,   f = 0.1 / (1 + (R / 40 km)²)
//!
//! with S the stellar flux at the body, giving ~10⁻⁴ AU/Myr for a 1 km
//! asteroid at 2.5 AU. `f` falls off for large bodies (their thermal skin is
//! thin compared with their size) and for rocks too small to hold a thermal
//! gradient is not modelled. The spin sense is not tracked per body; it is a
//! deterministic ± drawn from the body id, so there is no RNG use.
//! The physical rate is far too slow to see, so `yarkovsky_boost` scales it
//! up (1 in physical mode).

use super::*;

/// Boost on the Yarkovsky acceleration (1 in physical mode).
pub const YARKOVSKY_BOOST: f64 = 1e4;

pub fn yarkovsky_boost() -> f64 {
    if physical_mode() {
        1.0
    } else {
        YARKOVSKY_BOOST
    }
}

/// Largest radius (Gm) that is treated: 100 km.
const YARKOVSKY_R_MAX: f64 = 1e-4;
const L_SUN_W: f64 = 3.828e26;
const C_SI: f64 = 2.997_924_58e8;

#[inline]
fn spin_sign(id: u64) -> f64 {
    if (id.wrapping_mul(2_654_435_761) >> 13) & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

fn is_yarkovsky_origin(o: BodyOrigin) -> bool {
    matches!(
        o,
        BodyOrigin::AsteroidBelt
            | BodyOrigin::Hilda
            | BodyOrigin::Trojan
            | BodyOrigin::BackgroundDebris
            | BodyOrigin::ImpactDebris
            | BodyOrigin::TidalDebris
    )
}

/// Apply the Yarkovsky transverse acceleration to every small rocky body
/// that has a luminous host. Non-conservative: the kinetic-energy change is
/// folded into `energy_offset`.
pub(super) fn yarkovsky(b: &mut Bodies, dt: f64) {
    let n = b.len();
    if n < 2 || !b.parent_cache_valid {
        return;
    }
    let boost = yarkovsky_boost();
    let tu = time_unit_s();
    let len_m = LEN_KM * 1e3;
    // Luminosity of each host, computed once per call.
    let mut lum: Vec<Option<f64>> = vec![None; n];

    for j in 0..n {
        if b.dead[j]
            || !is_yarkovsky_origin(b.origin[j])
            || !(b.r[j] > 0.0)
            || b.r[j] > YARKOVSKY_R_MAX
            || !(b.m[j] > 0.0)
        {
            continue;
        }
        let Some(h) = b.parent_cache.get(j).copied().flatten() else {
            continue;
        };
        if h >= n || !is_host_kind(b.kind[h]) {
            continue;
        }
        let l = *lum[h].get_or_insert_with(|| luminosity_of(b, h));
        if !(l > 0.0) {
            continue;
        }
        let dx = b.x[j] - b.x[h];
        let dy = b.y[j] - b.y[h];
        let dz = b.z[j] - b.z[h];
        let dvx = b.vx[j] - b.vx[h];
        let dvy = b.vy[j] - b.vy[h];
        let dvz = b.vz[j] - b.vz[h];
        let v = (dvx * dvx + dvy * dvy + dvz * dvz).sqrt();
        let d2 = dx * dx + dy * dy + dz * dz;
        if !(v > 0.0) || !(d2 > 0.0) {
            continue;
        }
        let d_m2 = d2 * len_m * len_m;
        let s_flux = L_SUN_W * l / (4.0 * std::f64::consts::PI * d_m2);
        let r_m = b.r[j] * len_m;
        let r_km = b.r[j] * LEN_KM;
        let f = 0.1 / (1.0 + (r_km / 40.0) * (r_km / 40.0));
        let force = f * std::f64::consts::PI * r_m * r_m * s_flux / C_SI;
        let a_si = force / (b.m[j] * MASS_KG);
        // SI → simulation units, with the sign of the (hidden) spin.
        let a_sim = spin_sign(b.id[j]) * a_si * boost * tu * tu / len_m;
        let dv = a_sim * dt;
        let ke0 = 0.5 * b.m[j] * (b.vx[j] * b.vx[j] + b.vy[j] * b.vy[j] + b.vz[j] * b.vz[j]);
        b.vx[j] += dv * dvx / v;
        b.vy[j] += dv * dvy / v;
        b.vz[j] += dv * dvz / v;
        if b.e_pre.is_none() {
            let ke1 = 0.5 * b.m[j] * (b.vx[j] * b.vx[j] + b.vy[j] * b.vy[j] + b.vz[j] * b.vz[j]);
            b.stats.energy_offset += ke1 - ke0;
        }
    }
}
