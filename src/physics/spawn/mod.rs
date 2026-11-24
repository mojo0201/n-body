//! Random system spawn: samplers, Kepler helpers, primary and system entry points.
//!
//! Large parts live in the `binary`, `systems`, `cluster` and `belts`
//! sub-modules. Everything is re-exported from here.

use super::*;
use std::f64::consts::TAU;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

pub mod belts;
pub mod binary;
pub mod cluster;
pub mod systems;

// Only `cluster` has public items; the rest are visible inside `physics`.
pub(super) use belts::*;
pub(super) use binary::*;
pub use cluster::*;
pub(super) use systems::*;

/// Add a moon in a circular orbit around the planet at index `i`.
pub fn spawn_moon(b: &mut Bodies, i: usize, rng: &mut XorShift) -> Option<usize> {
    if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
        return None;
    }
    let host = find_parent(b, i).or_else(|| primary_index(b))?;
    if host == i {
        return None;
    }
    let d_star = (b.x[i] - b.x[host]).hypot(b.y[i] - b.y[host]);
    let r_hill = d_star * (b.m[i] / (3.0 * b.m[host])).cbrt();
    let a_lo = 5.0 * b.r[i];
    let a_hi = MOON_HILL_FRAC * r_hill;
    if a_lo >= a_hi {
        return None;
    }
    let a = a_lo * (a_hi / a_lo).powf(rng.next_f64());
    let angle = rng.next_f64() * TAU;
    let m_moon = b.m[i] * 10f64.powf(-4.0 + rng.next_f64() * 2.0);
    // Moons orbit near their planet's own orbital plane.
    let n = orbit_normal(
        b,
        i,
        [b.x[host], b.y[host], b.z[host], b.vx[host], b.vy[host], b.vz[host]],
    );
    let tilt = tilt_rayleigh(tilt_key(&[angle, a]), 3.0);
    let moon = push_circular_about(b, i, n, tilt, a, angle, m_moon, BodyKind::Planet);
    // The moon inherits the parent planet's class: the Moon is Rocky,
    // Ganymede is Icy. Same rule everywhere moons are added.
    b.class[moon] = b.class[i];
    b.bulk_density[moon] = draw_density(rng, b.class[moon]);
    b.r[moon] = b.radius_now(moon);
    b.parent_cache_valid = false;
    let (x, y) = (b.x[moon], b.y[moon]);
    let text = format!("{} spawned as a moon of {}", tag(b, moon), tag(b, i));
    b.log(EventKind::User, text, x, y);
    Some(moon)
}

/// Common inbound trajectory shared by `spawn_interstellar_object` and
/// `spawn_rogue_planet`.
///
/// Draws from `b.rng`: position angle, periapsis, direction, then v∞.
struct Inbound {
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    b_impact_au: f64,
}

/// Set up an unbound inbound trajectory from `0.9 × CULL_DIST`, aimed
/// at a random impact parameter of 0.5–50 AU (log-uniform) from the
/// primary, arriving 10 % above escape velocity. Returns `None` if
/// there is no primary or `MAX_BODIES` is already reached.
fn setup_inbound(b: &mut Bodies) -> Option<Inbound> {
    let prim = primary_index(b)?;
    if b.len() + 1 > MAX_BODIES {
        return None;
    }
    let m_prim = b.m[prim];

    let r0 = 0.9 * CULL_DIST;
    let theta = b.rng.next_f64() * TAU;
    let (sx, sy) = (r0 * theta.cos(), r0 * theta.sin());

    // Periapsis, log-uniform 0.5-50 AU; direction of motion is random.
    let q_au = 0.5 * 10f64.powf(2.0 * b.rng.next_f64());
    let q = (q_au * au_in_gm()).min(0.5 * r0);
    let dir = if b.rng.next_f64() < 0.5 { 1.0 } else { -1.0 };

    // Hyperbolic excess speed (km/s -> Gm per code time unit).
    let v_inf = sample_normal(&mut b.rng, 26.0, 8.0).clamp(5.0, 60.0) * 1.0e-6 * time_unit_s();
    let mu = G * m_prim;
    // h² = q²·v_q² with v_q² = v∞² + 2μ/q  =>  periapsis is exactly q.
    let h = (q * q * v_inf * v_inf + 2.0 * mu * q).sqrt();
    let v_t = dir * h / r0;
    let v_r = -(v_inf * v_inf + 2.0 * mu / r0 - v_t * v_t).max(0.0).sqrt(); // inbound
    let (st, ct) = theta.sin_cos();
    let (vx, vy) = (v_r * ct - v_t * st, v_r * st + v_t * ct);

    // Interlopers arrive from any direction: spin the whole in-plane
    // trajectory about the x axis by a random angle (cos i uniform), which
    // keeps the periapsis and angular momentum magnitude exactly as drawn.
    let (beta, _) = tilt_isotropic(tilt_key(&[theta, q, v_inf]));
    let (sb, cb) = beta.sin_cos();
    let (sy, sz, vy, vz) = if beta == 0.0 {
        (sy, 0.0, vy, 0.0)
    } else {
        (sy * cb, sy * sb, vy * cb, vy * sb)
    };

    Some(Inbound {
        x: sx,
        y: sy,
        z: sz,
        vx,
        vy,
        vz,
        b_impact_au: q / au_in_gm(), // now the true periapsis
    })
}

/// Spawn an interstellar object inbound from `0.9 × CULL_DIST`.
///
/// The ISO is a small body (1e-9 to 1e-6 M⊕ ≈ 6 km to 65 km radius,
/// the range that includes 1I/ʻOumuamua and 2I/Borisov) aimed at an
/// impact parameter of 0.5–50 AU from the primary (log-uniform, so
/// wide passes are more common than deep sun-grazers). It arrives with
/// a speed 10 % above the primary's escape velocity at its start
/// radius, so it is unbound — the trajectory is hyperbolic and the
/// object will leave.
///
/// Logs `EventKind::Interstellar` "entered the system inbound" at the
/// spawn position; `cull_escaped` logs "left the system outbound" when
/// it eventually crosses `CULL_DIST` on the way out.
///
/// Returns `None` if `b` has no primary host (nothing to aim at) or
/// `MAX_BODIES` is reached.
///
/// User-triggered — from a UI button or the `I` hotkey — so it draws
/// from `b.rng` freely. Every subsequent user-triggered action sees a
/// shifted stream, but `advance` never calls this, so no automatic
/// test's RNG stream is affected.
pub fn spawn_interstellar_object(b: &mut Bodies) -> Option<usize> {
    let t = setup_inbound(b)?;

    // Mass: log-uniform 1e-9 to 1e-6 M⊕ (≈ 6 km to 65 km radius).
    let m = 1e-9 * 10f64.powf(3.0 * b.rng.next_f64());
    // Class: most ISOs are carbonaceous or silicaceous.
    let class = match b.rng.next_f64() {
        u if u < 0.50 => BodyClass::C,
        u if u < 0.85 => BodyClass::S,
        u if u < 0.95 => BodyClass::M,
        _ => BodyClass::Icy,
    };

    let i = push3(b, [t.x, t.y, t.z], [t.vx, t.vy, t.vz], m, BodyKind::Planet);
    b.origin[i] = BodyOrigin::Interstellar;
    b.class[i] = class;
    b.bulk_density[i] = draw_density(&mut b.rng, class);
    b.r[i] = b.radius_now(i);
    b.name[i] = Some(format!("ISO-{}", b.id[i]));

    let text = format!(
        "{} entered the system inbound ({:.1} AU closest approach)",
        tag(b, i),
        t.b_impact_au
    );
    b.log(EventKind::Interstellar, text, t.x, t.y);
    Some(i)
}

/// Add a 25-particle planetary ring between 1.4 and 2.6 planet radii.
pub fn add_ring(b: &mut Bodies, i: usize, n: usize, rng: &mut XorShift) -> Vec<usize> {
    let mut out = Vec::new();
    if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
        return out;
    }
    let n = n.clamp(1, 30);
    let r0 = b.r[i];
    let m_total = 1e-4 * b.m[i];
    // 2.6 / 1.4 — hoisted so the ratio isn't recomputed per particle.
    const RING_RATIO: f64 = 2.6 / 1.4;
    let radii: Vec<f64> = (0..n)
        .map(|_| r0 * 1.4 * RING_RATIO.powf(rng.next_f64()))
        .collect();
    // The `w = sqrt(r / r0)` weights are cheap to recompute; storing
    // them in a second `Vec` only bought an allocation.
    let wsum: f64 = radii.iter().map(|&r| (r / r0).sqrt()).sum();
    // Rings lie in the planet's orbital plane.
    let ring_n = match find_parent(b, i).or_else(|| primary_index(b)) {
        Some(h) if h != i => orbit_normal(
            b,
            i,
            [b.x[h], b.y[h], b.z[h], b.vx[h], b.vy[h], b.vz[h]],
        ),
        _ => [0.0, 0.0, 1.0],
    };
    for &r in &radii {
        let w = (r / r0).sqrt();
        let m = m_total * w / wsum;
        let angle = rng.next_f64() * TAU;
        let k = push_circular_about(b, i, ring_n, (0.0, 0.0), r, angle, m, BodyKind::Planet);
        b.origin[k] = BodyOrigin::ImpactDebris;
        // Rings are icy — Saturn's is > 95 % water ice by mass.
        b.class[k] = BodyClass::Icy;
        b.bulk_density[k] = draw_density(rng, b.class[k]);
        b.r[k] = b.radius_now(k);
        out.push(k);
    }
    if !out.is_empty() {
        b.parent_cache_valid = false;
        let (x, y) = (b.x[i], b.y[i]);
        let text = format!("ring added around {} ({} particles)", tag(b, i), out.len());
        b.log(EventKind::User, text, x, y);
    }
    out
}

pub(super) fn sample_rayleigh(rng: &mut XorShift, sigma: f64) -> f64 {
    let u = rng.next_f64().clamp(1e-12, 1.0 - 1e-12);
    sigma * (-2.0 * (1.0 - u).ln()).sqrt()
}

pub(super) fn sample_mass_power_law(rng: &mut XorShift, m_min: f64, m_max: f64, p: f64) -> f64 {
    let u = rng.next_f64();
    let inv = 1.0 - p;
    let a = m_min.powf(inv);
    let b = m_max.powf(inv);
    ((1.0 - u) * a + u * b).powf(1.0 / inv)
}

/// Draw `n` masses for one belt population. `n_dominant` of them are
/// "dominant" bodies (a single Pluto- or Ceres-mass body rather than a
/// tracer); the rest are tracers that together carry the remainder of
/// the population.
///
/// The number of dominant bodies is *deterministic* — it depends only on
/// the population mass — so this function consumes exactly the same
/// `next_f64()` draws as the pre-dominant version and the RNG stream is
/// unchanged. Each dominant is 30 % of the population's mass in total,
/// divided among 1–3 bodies in proportion to their power-law draws (so
/// their individual masses have a real spread, e.g. Pluto vs. a smaller
/// scattered KBO). Tracers share the other 70 %, capped as before.
///
/// `total` is preserved: the sum of the returned masses equals `total`
/// unless a tracer hits its cap (which is the same loss the pre-dominant
/// code had).
pub(super) fn draw_masses_to_total(
    rng: &mut XorShift,
    n: usize,
    m_min: f64,
    m_max: f64,
    plaw: f64,
    total: f64,
) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if total <= 0.0 {
        return vec![0.0; n];
    }

    // Deterministic dominant count, from the population's mass alone. No
    // `next_f64()` draw, so the RNG stream is preserved exactly.
    let n_dominant = ((total / BELT_REF_MASS).round() as usize)
        .clamp(1, 3)
        .min(n);

    // Same number of `next_f64()` draws as before, just split at the
    // dominant/tracer boundary below.
    let mut samples: Vec<f64> = (0..n)
        .map(|_| sample_mass_power_law(rng, m_min, m_max, plaw))
        .collect();

    // 30 % of the population's mass in the dominant bodies. 30 is a
    // compromise: massive enough that one body is visibly larger than
    // the tracers, small enough that a dominants-only belt doesn't
    // collapse the tracer population's total mass below the `got /
    // expected` tolerance of `spawn_outer_belt_mass_realistic`.
    const DOMINANT_BELT_FRAC: f64 = 0.30;
    let total_dom = total * DOMINANT_BELT_FRAC;
    let total_tr = total - total_dom;

    let (dom_slice, tr_slice) = samples.split_at_mut(n_dominant);

    let dom_sum: f64 = dom_slice.iter().sum();
    if dom_sum > 0.0 {
        let scale_d = total_dom / dom_sum;
        for m in dom_slice.iter_mut() {
            *m *= scale_d;
        }
    }

    let tr_sum: f64 = tr_slice.iter().sum();
    if tr_sum > 0.0 {
        let scale_t = total_tr / tr_sum;
        // Same per-tracer cap as before — dominants are exempt because
        // they are *supposed* to be individually significant.
        let cap = BELT_MAX_TRACER_MASS.min(total / 5.0);
        for m in tr_slice.iter_mut() {
            *m = (*m * scale_t).min(cap);
        }
    }

    samples
}

pub(super) fn spacing_range(
    n: usize,
    a_min: f64,
    a_max: f64,
    base_lo: f64,
    base_hi: f64,
) -> (f64, f64) {
    if n <= 1 {
        return (base_lo, base_hi);
    }
    let ratio_fit = (a_max / a_min).powf(1.0 / (n - 1) as f64);
    if ratio_fit >= base_hi {
        (base_lo, base_hi)
    } else {
        let scale = ratio_fit / base_hi;
        ((base_lo * scale).max(1.02), ratio_fit.max(1.02))
    }
}

/// Cartesian state of a body on a Kepler orbit about a fixed centre with
/// gravitational parameter `mu`, counter-clockwise, periapsis along `omega`.
pub(super) fn kepler_state(mu: f64, a: f64, e: f64, nu: f64, omega: f64) -> (f64, f64, f64, f64) {
    let (sn, cn) = nu.sin_cos();
    // `1 - e²` was computed twice; hoist it. The `a * one_minus_e2`
    // grouping matches the original `a * (1 - e*e)` exactly, so `r`
    // and `h` are bit-identical to the pre-refactor values.
    let one_minus_e2 = 1.0 - e * e;
    let r = a * one_minus_e2 / (1.0 + e * cn);
    let h = (mu * a * one_minus_e2).sqrt();
    let v_t = h / r;
    let v_r = mu / h * e * sn;
    let (x_orb, y_orb) = (r * cn, r * sn);
    let vx_orb = v_r * cn - v_t * sn;
    let vy_orb = v_r * sn + v_t * cn;
    let (sw, cw) = omega.sin_cos();
    (
        cw * x_orb - sw * y_orb,
        sw * x_orb + cw * y_orb,
        cw * vx_orb - sw * vy_orb,
        sw * vx_orb + cw * vy_orb,
    )
}

/// `kepler_state` for an inclined orbit: `omega` is the argument of periapsis
/// (from the ascending node), `inc` the inclination and `node` the longitude of
/// the ascending node. With `inc == 0 && node == 0` this is exactly the planar
/// state, so z = 0 systems are bit-identical.
pub(super) fn kepler_state_incl(
    mu: f64,
    a: f64,
    e: f64,
    nu: f64,
    omega: f64,
    inc: f64,
    node: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let (x, y, vx, vy) = kepler_state(mu, a, e, nu, omega);
    if inc == 0.0 && node == 0.0 {
        return (x, y, 0.0, vx, vy, 0.0);
    }
    // Tilt about the node line (x axis), then rotate the node to `node`.
    let (si, ci) = inc.sin_cos();
    let (sn, cn) = node.sin_cos();
    let (y1, z1) = (y * ci, y * si);
    let (vy1, vz1) = (vy * ci, vy * si);
    (
        cn * x - sn * y1,
        sn * x + cn * y1,
        z1,
        cn * vx - sn * vy1,
        sn * vx + cn * vy1,
        vz1,
    )
}

// ---------------------------------------------------------------------------
// Orbital inclination
// ---------------------------------------------------------------------------
//
// The reference plane (z = 0) is the system's mid-plane. Every tilt is drawn
// from a *private* RNG keyed on values the main stream already produced, so
// switching inclination on or off never shifts the main spawn stream, and a
// scale of 0 reproduces the old planar systems bit for bit.

/// Global multiplier on all random inclination spreads (1.0 = realistic,
/// 0.0 = every orbit coplanar). Stored as f64 bits.
static INCLINATION_SCALE: AtomicU64 = AtomicU64::new(0x3FF0_0000_0000_0000);

pub fn inclination_scale() -> f64 {
    f64::from_bits(INCLINATION_SCALE.load(Ordering::Relaxed))
}

pub fn set_inclination_scale(s: f64) {
    let s = if s.is_finite() { s.max(0.0) } else { 1.0 };
    INCLINATION_SCALE.store(s.to_bits(), Ordering::Relaxed);
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Hash a few already-drawn values into a key for the private tilt RNG.
pub(super) fn tilt_key(vals: &[f64]) -> u64 {
    vals.iter()
        .fold(0x1C1_1AB1_E5, |h, v| splitmix(h ^ v.to_bits()))
}

fn tilt_rng(key: u64) -> XorShift {
    let mut r = XorShift::new(splitmix(key) as u128);
    r.next_f64();
    r
}

/// `(inclination, node)` in radians: Rayleigh-distributed inclination with
/// scale `sigma_deg` (times the global scale) and a uniform node. Exactly
/// `(0, 0)` when the scale is zero.
pub(super) fn tilt_rayleigh(key: u64, sigma_deg: f64) -> (f64, f64) {
    let scale = inclination_scale();
    if scale <= 0.0 || sigma_deg <= 0.0 {
        return (0.0, 0.0);
    }
    let mut r = tilt_rng(key);
    let inc = sample_rayleigh(&mut r, sigma_deg.to_radians() * scale).min(std::f64::consts::FRAC_PI_2);
    (inc, r.next_f64() * TAU)
}

/// Isotropic orientation (cos i uniform in [-1, 1], so retrograde orbits are
/// as common as prograde). `(0, 0)` when the global scale is zero.
pub(super) fn tilt_isotropic(key: u64) -> (f64, f64) {
    if inclination_scale() <= 0.0 {
        return (0.0, 0.0);
    }
    let mut r = tilt_rng(key);
    let inc = (1.0 - 2.0 * r.next_f64()).clamp(-1.0, 1.0).acos();
    (inc, r.next_f64() * TAU)
}

/// Rotation matrix taking a system's own frame to a random orientation in
/// space (isotropic pole, uniform spin). `None` when inclinations are off, so
/// the system stays in the reference plane.
pub(super) fn random_orientation(key: u64) -> Option<[[f64; 3]; 3]> {
    if inclination_scale() <= 0.0 {
        return None;
    }
    let (inc, node) = tilt_isotropic(key);
    let spin = tilt_rng(key ^ 0x5B1D_0F5E).next_f64() * TAU;
    let rz = |a: f64| {
        let (s, c) = a.sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    };
    let (si, ci) = inc.sin_cos();
    let rx = [[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]];
    let mul = |a: [[f64; 3]; 3], b: [[f64; 3]; 3]| {
        let mut o = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                o[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
            }
        }
        o
    };
    Some(mul(rz(node), mul(rx, rz(spin))))
}

/// Rotate a whole system (positions and velocities about the origin) into a
/// random orientation. No-op when inclinations are off.
pub(in crate::physics) fn orient_system(temp: &mut Bodies, key: u64) {
    let Some(r) = random_orientation(key) else {
        return;
    };
    let rot = |v: [f64; 3]| [0, 1, 2].map(|i| r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2]);
    for i in 0..temp.len() {
        let p = rot([temp.x[i], temp.y[i], temp.z[i]]);
        let v = rot([temp.vx[i], temp.vy[i], temp.vz[i]]);
        (temp.x[i], temp.y[i], temp.z[i]) = (p[0], p[1], p[2]);
        (temp._px[i], temp._py[i], temp._pz[i]) = (p[0], p[1], p[2]);
        (temp.vx[i], temp.vy[i], temp.vz[i]) = (v[0], v[1], v[2]);
    }
}

/// Push a body with a full 3D state.
pub(super) fn push3(
    b: &mut Bodies,
    p: [f64; 3],
    v: [f64; 3],
    m: f64,
    kind: BodyKind,
) -> usize {
    let i = b.push(p[0], p[1], v[0], v[1], m, kind);
    b.z[i] = p[2];
    b._pz[i] = p[2];
    b.vz[i] = v[2];
    i
}

/// Unit normal of body `i`'s orbit about a centre `c = [x, y, z, vx, vy, vz]`
/// (+z for a planar prograde orbit).
pub(super) fn orbit_normal(b: &Bodies, i: usize, c: [f64; 6]) -> [f64; 3] {
    let r = [b.x[i] - c[0], b.y[i] - c[1], b.z[i] - c[2]];
    let v = [b.vx[i] - c[3], b.vy[i] - c[4], b.vz[i] - c[5]];
    let h = [
        r[1] * v[2] - r[2] * v[1],
        r[2] * v[0] - r[0] * v[2],
        r[0] * v[1] - r[1] * v[0],
    ];
    let hm = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
    if hm > 0.0 {
        [h[0] / hm, h[1] / hm, h[2] / hm]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Two unit vectors spanning the plane with unit normal `n`; for `n = +z`
/// they are exactly +x and +y.
pub(super) fn plane_basis_pub(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    plane_basis(n)
}

fn plane_basis(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let l = n[0].hypot(n[1]);
    let e1 = if l < 1e-12 {
        [1.0, 0.0, 0.0]
    } else {
        [-n[1] / l, n[0] / l, 0.0]
    };
    let e2 = [
        n[1] * e1[2] - n[2] * e1[1],
        n[2] * e1[0] - n[0] * e1[2],
        n[0] * e1[1] - n[1] * e1[0],
    ];
    (e1, e2)
}

/// `n` tilted by `inc` towards the direction at azimuth `node` about it.
pub(super) fn tilted_normal(n: [f64; 3], tilt: (f64, f64)) -> [f64; 3] {
    let (inc, node) = tilt;
    if inc == 0.0 {
        return n;
    }
    let (u, w) = plane_basis(n);
    let (si, ci) = inc.sin_cos();
    let (sn, cn) = node.sin_cos();
    [
        ci * n[0] + si * (cn * u[0] + sn * w[0]),
        ci * n[1] + si * (cn * u[1] + sn * w[1]),
        ci * n[2] + si * (cn * u[2] + sn * w[2]),
    ]
}

/// Offset and velocity (relative to the host) of a circular orbit of radius
/// `a` and speed `v` at phase `phi`, in the plane normal to `n` tilted by
/// `tilt`. For `n = +z` and no tilt this is the old planar formula exactly.
pub(super) fn circular_state(
    n: [f64; 3],
    tilt: (f64, f64),
    a: f64,
    v: f64,
    phi: f64,
) -> ([f64; 3], [f64; 3]) {
    let (sp, cp) = phi.sin_cos();
    if tilt.0 == 0.0 && n[0] == 0.0 && n[1] == 0.0 && n[2] == 1.0 {
        return ([a * cp, a * sp, 0.0], [-v * sp, v * cp, 0.0]);
    }
    let (e1, e2) = plane_basis(tilted_normal(n, tilt));
    let pos = [0, 1, 2].map(|k| a * (cp * e1[k] + sp * e2[k]));
    let vel = [0, 1, 2].map(|k| v * (cp * e2[k] - sp * e1[k]));
    (pos, vel)
}

/// Push a body on a circular orbit about `host` (moons, ring particles).
#[allow(clippy::too_many_arguments)]
pub(super) fn push_circular_about(
    b: &mut Bodies,
    host: usize,
    n: [f64; 3],
    tilt: (f64, f64),
    a: f64,
    phi: f64,
    m: f64,
    kind: BodyKind,
) -> usize {
    let v = (G * b.m[host] / a).sqrt();
    let (o, w) = circular_state(n, tilt, a, v, phi);
    let p = [b.x[host] + o[0], b.y[host] + o[1], b.z[host] + o[2]];
    let vel = [b.vx[host] + w[0], b.vy[host] + w[1], b.vz[host] + w[2]];
    push3(b, p, vel, m, kind)
}

/// Typical inclination spread (degrees, Rayleigh scale) of a small-body
/// population about the mid-plane. `None` means isotropic (Oort-cloud comets).
pub(super) fn population_sigma_deg(origin: BodyOrigin, a_au: f64) -> Option<f64> {
    Some(match origin {
        BodyOrigin::AsteroidBelt => 7.0,
        BodyOrigin::Hilda => 8.0,
        BodyOrigin::Trojan => 12.0,
        BodyOrigin::KuiperCold => 2.0,
        BodyOrigin::KuiperHot => 12.0,
        BodyOrigin::KuiperResonant => 10.0,
        BodyOrigin::KuiperScattered => 15.0,
        BodyOrigin::Comet if a_au > 200.0 => return None,
        BodyOrigin::Comet => 12.0,
        _ => 5.0,
    })
}

/// Solve Kepler's equation M = E - e sin E.
pub(super) fn solve_kepler(mean_anom: f64, e: f64) -> f64 {
    let mut ea = if e < 0.8 {
        mean_anom
    } else {
        std::f64::consts::PI
    };
    for _ in 0..60 {
        let d = (ea - e * ea.sin() - mean_anom) / (1.0 - e * ea.cos());
        ea -= d;
        if d.abs() < 1e-13 {
            break;
        }
    }
    ea
}

pub(super) fn true_anomaly(ecc_anom: f64, e: f64) -> f64 {
    // `ecc_anom * 0.5` was computed twice; hoist it.
    let half = ecc_anom * 0.5;
    2.0 * ((1.0 + e).sqrt() * half.sin()).atan2((1.0 - e).sqrt() * half.cos())
}

/// Push a test body on a Kepler orbit about an arbitrary centre (position,
/// velocity, central mass). Phase is uniform in mean anomaly (time).
pub(super) fn push_kepler_about(
    b: &mut Bodies,
    centre: (f64, f64, f64, f64, f64),
    a: f64,
    e: f64,
    m: f64,
    rng: &mut XorShift,
    kind: BodyKind,
) -> usize {
    let (cx, cy, cvx, cvy, mc) = centre;
    let omega = rng.next_f64() * TAU;
    let mean_anom = rng.next_f64() * TAU;
    let nu = true_anomaly(solve_kepler(mean_anom, e), e);
    // Planetary systems are nearly coplanar, but how flat varies from star to
    // star: Kepler multis sit within ~1-2 degrees, the Solar System ~3, and
    // dynamically hotter systems reach several degrees. Each system draws its
    // own spread, log-uniform from 0.7 to 4.2 degrees, keyed on its star's mass.
    let sigma = 0.7 * 6f64.powf(tilt_rng(tilt_key(&[mc])).next_f64());
    let (inc, node) = tilt_rayleigh(tilt_key(&[omega, mean_anom, a]), sigma);
    let (x, y, z, vx, vy, vz) = kepler_state_incl(G * (mc + m), a, e, nu, omega, inc, node);
    push3(b, [cx + x, cy + y, z], [cvx + vx, cvy + vy, vz], m, kind)
}

/// Push a planet and give it a bulk density: rocky inside the snow line,
/// icy outside, light for giants. The radius follows the density.
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_planet(
    b: &mut Bodies,
    centre: (f64, f64, f64, f64, f64),
    a_gm: f64,
    e: f64,
    m: f64,
    kind: BodyKind,
    snow_line_au: f64,
    rng: &mut XorShift,
) -> usize {
    let i = push_kepler_about(b, centre, a_gm, e, m, rng, kind);
    if m >= GIANT_DENSITY_M_MIN {
        // Gas giants are Icy for colour purposes; `render.rs` checks
        // mass and origin to give them the tan gas-giant look.
        b.class[i] = BodyClass::Icy;
        b.bulk_density[i] = giant_density(m, rng);
        // Hot-Jupiter radius inflation (Demory & Seager 2011): above
        // ~150 S⊕ the radius grows by up to ~25 %. Uses the snow-line scaling
        // for flux (flux at the snow line ≈ 0.137 S⊕), so it needs no RNG.
        let flux = flux_from_snow_line(a_gm / au_in_gm(), snow_line_au);
        let infl = 1.0 + 0.25 * ((flux / 150.0).max(1.0).log10() / 1.5).min(1.0);
        b.bulk_density[i] /= infl * infl * infl;
    } else {
        let a_au = a_gm / au_in_gm();
        b.class[i] = class_for(BodyOrigin::Spawned, a_au, snow_line_au);
        let flux = flux_from_snow_line(a_au, snow_line_au);
        b.bulk_density[i] = draw_planet_density(rng, b.class[i], m, flux);
    }
    b.r[i] = b.radius_now(i);
    i
}

/// Approximate stellar flux in S⊕ at `a_au`, inferred from the snow-line
/// distance (which scales as √L): flux ≈ 0.137 (snow / a)².
fn flux_from_snow_line(a_au: f64, snow_line_au: f64) -> f64 {
    if !(a_au > 0.0 && snow_line_au.is_finite() && snow_line_au > 0.0) {
        return 1.0;
    }
    0.137 * (snow_line_au / a_au) * (snow_line_au / a_au)
}

/// Bulk density (g/cm³) for a spawned planet of mass `m` (M⊕) and flux `flux`
/// (S⊕), with the same single RNG draw as `draw_density`.
///
/// Rocky planets get Earth-like compositions whose density rises with mass
/// through compression (∝ M^0.19). Above ~2 M⊕ a fraction of them keep a
/// volatile envelope (sub-Neptunes, ρ ≈ 1.5–3 g/cm³); that fraction rises with
/// mass and is stripped by photoevaporation at high flux. That reproduces the
/// radius valley: few planets between ~1.7 and ~2 R⊕ at high flux.
/// Icy planets keep the plain draw.
fn draw_planet_density(rng: &mut XorShift, class: BodyClass, m: f64, flux: f64) -> f64 {
    if class != BodyClass::Rocky {
        return draw_density(rng, class);
    }
    let u = rng.next_f64();
    let p_env = ((m - 2.0) / 8.0).clamp(0.0, 0.9) / (1.0 + flux / (30.0 * m.max(1.0)));
    if u < p_env {
        // Envelope-bearing sub-Neptune: R ≈ 2.5 R⊕ (m/6)^0.55.
        let r = 2.5 * (m / 6.0).powf(0.55);
        5.514 * m / (r * r * r)
    } else {
        let f = (u - p_env) / (1.0 - p_env).max(1e-9);
        (4.4 + 1.4 * f) * m.clamp(1.0, 20.0).powf(0.19)
    }
}

/// Cold-giant bulk density from an M-R table anchored on Neptune, Saturn and
/// Jupiter; radii saturate near 1 R_J from ~1 to 10 M_J (Fortney 2007).
/// Uses exactly one RNG draw, like the code it replaces.
fn giant_density(m: f64, rng: &mut XorShift) -> f64 {
    const T: [(f64, f64); 5] = [
        (17.0, 3.9),
        (95.0, 9.45),
        (318.0, 11.2),
        (1000.0, 11.4),
        (3180.0, 11.2),
    ];
    let r = if m <= T[0].0 {
        T[0].1
    } else if m >= T[4].0 {
        T[4].1
    } else {
        let k = T.windows(2).position(|w| m < w[1].0).unwrap_or(3);
        let f = (m / T[k].0).ln() / (T[k + 1].0 / T[k].0).ln();
        (T[k].1.ln() + f * (T[k + 1].1.ln() - T[k].1.ln())).exp()
    };
    5.514 * m / (r * r * r) * (0.85 + 0.30 * rng.next_f64())
}

/// Planets at or above this mass (Earth masses) get gas-giant densities.
pub(super) const GIANT_DENSITY_M_MIN: f64 = 50.0;

/// Pick a taxonomic class for a new body. Deterministic — takes no RNG,
/// so the RNG stream (and therefore every belt-body position, velocity
/// and the whole simulation) is unchanged from before this feature.
///
/// Real asteroid belts show S-type in the inner main belt and C-type in
/// the outer; Kuiper and Trojan bodies are ice-dominated. Using the
/// body's own orbital distance is more physically faithful than a random
/// draw anyway.
pub(in crate::physics) fn class_for(origin: BodyOrigin, a_au: f64, snow_line_au: f64) -> BodyClass {
    match origin {
        BodyOrigin::AsteroidBelt => {
            if a_au < 2.5 {
                BodyClass::S
            } else {
                BodyClass::C
            }
        }
        BodyOrigin::KuiperCold
        | BodyOrigin::KuiperHot
        | BodyOrigin::KuiperResonant
        | BodyOrigin::KuiperScattered
        | BodyOrigin::Trojan
        | BodyOrigin::Comet => BodyClass::Icy,
        BodyOrigin::Hilda => BodyClass::C,
        BodyOrigin::ImpactDebris
        | BodyOrigin::TidalDebris
        | BodyOrigin::BackgroundDebris
        | BodyOrigin::Spawned => {
            if a_au < snow_line_au {
                BodyClass::Rocky
            } else {
                BodyClass::Icy
            }
        }
        // Not reached in practice — `spawn_interstellar_object` and
        // `spawn_rogue_planet` draw a class directly so different
        // visitors can be different compositions. Listed here only so
        // this match stays exhaustive.
        BodyOrigin::Interstellar => BodyClass::Rocky,
        BodyOrigin::RoguePlanet => BodyClass::Icy,
    }
}

/// Bulk density in g/cm³ for a body of the given class.
pub(super) fn draw_density(rng: &mut XorShift, class: BodyClass) -> f64 {
    match class {
        BodyClass::C => 1.2 + 0.8 * rng.next_f64(),
        BodyClass::S => 2.4 + 1.0 * rng.next_f64(),
        BodyClass::M => 5.0 + 3.0 * rng.next_f64(),
        BodyClass::V => 2.9 + 0.5 * rng.next_f64(),
        BodyClass::Icy => 0.5 + 2.0 * rng.next_f64(),
        BodyClass::Rocky => 2.5 + 3.0 * rng.next_f64(),
    }
}

/// Push a test body on a Kepler orbit about body `primary`.
pub(super) fn push_kepler(
    b: &mut Bodies,
    primary: usize,
    a: f64,
    e: f64,
    m: f64,
    rng: &mut XorShift,
    kind: BodyKind,
) -> usize {
    let centre = (
        b.x[primary],
        b.y[primary],
        b.vx[primary],
        b.vy[primary],
        b.m[primary],
    );
    push_kepler_about(b, centre, a, e, m, rng, kind)
}

/// Kroupa (2001) IMF, two-piece: dN/dm ∝ m^-1.3 below 0.5 M☉, m^-2.3 above.
pub(super) fn sample_imf_mass(rng: &mut XorShift) -> f64 {
    // The four endpoint powers were recomputed on every call via
    // `powf`; hoist them into a `OnceLock`. The expressions are
    // unchanged, so the constants are bit-identical to the originals.
    static IMF: OnceLock<(f64, f64, f64, f64)> = OnceLock::new();
    let &(a1, c1, a2, c2) = IMF.get_or_init(|| {
        (
            0.08_f64.powf(-0.3),
            0.50_f64.powf(-0.3),
            0.50_f64.powf(-1.3),
            50.0_f64.powf(-1.3),
        )
    });

    let u = rng.next_f64();
    if u < 0.76 {
        let u2 = u / 0.76;
        (a1 + u2 * (c1 - a1)).powf(1.0 / -0.3)
    } else {
        let u2 = (u - 0.76) / 0.24;
        (a2 + u2 * (c2 - a2)).powf(1.0 / -1.3)
    }
}

/// Draw the primary of a random system.
///
/// The 85 / 15 split between stellar-mass and brown-dwarf primaries is a
/// coarse population-synthesis approximation, not a direct IMF integral.
/// Kroupa (2001) with a lower cutoff of 0.08 M_sun gives ~76 % of draws
/// below 0.5 M_sun, most of which are M dwarfs. Doing the IMF integral
/// properly (integrating the mass function over [0.08, STAR_M_MIN] vs
/// [STAR_M_MIN, 50]) gives ~86 / 14, close enough that the constant is
/// retained for clarity. If `sample_imf_mass`'s range ever changes, this
/// split must be recomputed - it is not automatic.
///
/// Remnants (WD/NS/BH) are not drawn here. `spawn_aged_primary` converts
/// stars that outlive their lifetime into remnants. Drawing them here too
/// would double-count the field remnant fraction.
pub(super) fn sample_primary(rng: &mut XorShift) -> (f64, BodyKind) {
    if rng.next_f64() < 0.85 {
        (sample_imf_mass(rng) * M_SUN_ME, BodyKind::Star)
    } else {
        // Substellar mass function dN/dm ∝ m^-0.3 (Kroupa 2001), sampled by
        // inverting its CDF; still one draw. It is flatter than the old
        // log-uniform (dN/dm ∝ m^-1), so more brown dwarfs sit near the
        // hydrogen-burning limit.
        let (lo, hi) = (BD_M_MIN.powf(0.7), BD_M_MAX.powf(0.7));
        let m = (lo + (hi - lo) * rng.next_f64()).powf(1.0 / 0.7);
        (m, BodyKind::BrownDwarf)
    }
}

pub(super) fn sample_normal(rng: &mut XorShift, mu: f64, sigma: f64) -> f64 {
    let u1 = rng.next_f64().max(1e-12);
    let u2 = rng.next_f64();
    mu + sigma * (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

pub(super) fn poisson(rng: &mut XorShift, mean: f64) -> usize {
    if mean > 30.0 {
        return (mean + mean.sqrt() * sample_normal(rng, 0.0, 1.0)).max(0.0) as usize;
    }
    let l = (-mean).exp();
    let mut k = 0usize;
    let mut p = 1.0f64;
    loop {
        k += 1;
        p *= rng.next_f64();
        if p <= l {
            break;
        }
    }
    k - 1
}

/// Planetary architecture of a generated binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryArch {
    /// Planets orbit the primary, inside the Holman-Wiegert limit.
    STypePrimary,
    /// Planets orbit the companion, inside the Holman-Wiegert limit.
    STypeSecondary,
    /// Circumbinary: planets orbit the barycentre, outside the limit.
    PType,
}

#[derive(Clone, Copy, Debug)]
pub struct BinaryInfo {
    /// Body index of the companion (the primary is the heavier star, index 0).
    pub companion: usize,
    pub period_days: f64,
    /// Semi-major axis of the relative orbit, AU.
    pub a_au: f64,
    pub ecc: f64,
    /// m2 / m1, at most 1.
    pub mass_ratio: f64,
    pub arch: BinaryArch,
    /// Holman-Wiegert critical semi-major axis in AU: the outer edge of the
    /// stable zone for S-type (about the host star), the inner edge for P-type
    /// (about the barycentre). The planets respect it with `HW_MARGIN_S` /
    /// `HW_MARGIN_P`.
    pub a_crit_au: f64,
}

/// What `spawn_system` generated.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemInfo {
    pub binary: Option<BinaryInfo>,
}

/// Probability that a main-sequence star of mass `m_sun` (solar masses) has a
/// stellar companion (Raghavan 2010, Duchene & Kraus 2013), piecewise.
pub fn binary_fraction(m_sun: f64) -> f64 {
    if m_sun < 0.5 {
        0.30
    } else if m_sun < 0.8 {
        0.40
    } else if m_sun < 1.2 {
        0.46
    } else if m_sun < 1.5 {
        0.50
    } else if m_sun < 2.5 {
        0.65
    } else {
        0.80
    }
}

/// Holman & Wiegert (1999) critical semi-major axis for a planet orbiting one
/// star of a binary (S-type). `mu` is the perturber's mass fraction
/// m_pert / (m_host + m_pert). Result in the units of `a_b`.
pub fn hw_critical_s(a_b: f64, e: f64, mu: f64) -> f64 {
    let f = 0.464 - 0.380 * mu - 0.631 * e + 0.586 * mu * e + 0.150 * e * e - 0.198 * mu * e * e;
    (a_b * f).max(0.01 * a_b)
}

/// Holman & Wiegert (1999) critical semi-major axis for a circumbinary
/// planet (P-type). `mu` is m2 / (m1 + m2).
pub fn hw_critical_p(a_b: f64, e: f64, mu: f64) -> f64 {
    a_b * (1.60 + 5.10 * e - 2.22 * e * e + 4.12 * mu - 4.27 * e * mu - 5.09 * mu * mu
        + 4.61 * e * e * mu * mu)
}

pub(super) fn orbital_period_days(a_gm: f64, m_tot: f64) -> f64 {
    // Explicit cube: LLVM lowers `powi(3)` to exactly this association,
    // but writing it out makes the cost obvious.
    TAU * ((a_gm * a_gm * a_gm) / (G * m_tot)).sqrt() * time_unit_s() / 86_400.0
}

pub fn spawn_system(b: &mut Bodies, body_count: usize, seed: u128) {
    spawn_system_with_info(b, body_count, seed);
}

/// Like `spawn_system`, but also reports what was generated (binary
/// parameters and architecture).
pub fn spawn_system_with_info(b: &mut Bodies, body_count: usize, seed: u128) -> SystemInfo {
    *b = Bodies::with_capacity(body_count);
    b.rng = XorShift::new(seed ^ 0xA5A5_5A5A);
    let mut rng = XorShift::new(seed);

    // Host metallicity [Fe/H] in dex (log-normal in Z): mean 0, sigma 0.20.
    b.metallicity = sample_normal(&mut rng, 0.0, 0.20);
    let (m_init, kind0) = sample_primary(&mut rng);
    let prim = spawn_aged_primary(b, m_init, kind0, &mut rng);
    let kind = b.kind[prim];
    // Older disc stars are on average more metal-poor (weak trend).
    if kind == BodyKind::Star {
        b.metallicity += (-0.03 * (b.age_gyr[prim] - 4.0)).clamp(-0.25, 0.05);
    }

    let mut remaining = body_count.saturating_sub(1);
    // Main-sequence hosts get a stellar companion with `binary_fraction`;
    // brown-dwarf hosts get a brown-dwarf companion in ~20 % of cases.
    // Remnant hosts keep their single-body architectures.
    let mut binary: Option<BinaryInfo> = None;
    let wants_companion = if kind == BodyKind::Star {
        rng.next_f64() < binary_fraction(m_init / M_SUN_ME)
    } else if kind == BodyKind::BrownDwarf {
        rng.next_f64() < 0.20
    } else {
        false
    };
    if wants_companion && remaining >= BIN_MIN_BODIES {
        let prim_age = b.age_gyr[prim];
        binary = spawn_companion(b, prim, &mut rng, prim_age);
        if binary.is_some() {
            remaining -= 1;
        }
    }
    // Hierarchical third (and fourth) star: planned now, so that its slot is
    // taken out of the budget and the planets stay inside its stability
    // limit; placed after the planets, around the whole inner system.
    let mut outer_stars: Vec<OuterStar> = Vec::new();
    if kind == BodyKind::Star {
        if let Some(bi) = &binary {
            let m1 = b.m[prim];
            let mut m_in = b.m[prim] + b.m[bi.companion];
            let mut a_in = bi.a_au;
            while outer_stars.len() < 2 && remaining > BIN_MIN_BODIES {
                let p_next = if outer_stars.is_empty() {
                    // Tertiaries are far commoner around tight binaries.
                    let lp = bi.period_days.max(1.0).log10();
                    (P_TRIPLE_GIVEN_BINARY * (1.0 + 1.5 * ((3.0 - lp) / 2.0).clamp(0.0, 1.0)))
                        .min(0.9)
                } else {
                    P_QUAD_GIVEN_TRIPLE
                };
                if rng.next_f64() >= p_next {
                    break;
                }
                let Some(s) = plan_outer_star(m1, m_in, a_in, &mut rng) else {
                    break;
                };
                m_in += s.m;
                a_in = s.a_au;
                remaining -= 1;
                outer_stars.push(s);
            }
        }
    }
    // Planets must stay well inside the nearest outer star's periapsis.
    let a_outer_limit_au = outer_stars
        .first()
        .map_or(f64::INFINITY, |s| s.a_au * (1.0 - s.ecc) / 3.5);
    match binary.as_mut() {
        Some(bi) => spawn_binary_planets(b, prim, bi, remaining, a_outer_limit_au, &mut rng),
        None => spawn_architecture(b, prim, remaining, &mut rng),
    }
    let prim_age_now = b.age_gyr[prim];
    for s in &outer_stars {
        place_outer_star(b, s, prim_age_now, &mut rng);
    }

    recenter_com(b);
    if early_evolution() {
        evolve_early(b);
        recenter_com(b);
    }
    compute_accel(b);
    rebase_energy(b);
    dust::init_fields(b, None);
    // Names come last and draw from their own stream, so nothing else changes.
    names::name_system(b, seed);
    SystemInfo { binary }
}

/// Spawn a rogue planet inbound from `0.9 × CULL_DIST`.
///
/// Like `spawn_interstellar_object`, but at gas-giant mass (0.1 to 10
/// M_Jup) — the range of free-floating planets detected by microlensing
/// surveys. Same trajectory setup: starts inbound at `0.9 × CULL_DIST`,
/// arrives 10 % above escape velocity, and passes the primary at an
/// impact parameter of 0.5–50 AU (log-uniform).
///
/// Logs `EventKind::Interstellar` "entered the system inbound" (the
/// arrival is the same event kind as an ISO arrival). The departure
/// eventually logs `EventKind::Escape` at `HARD_CULL_DIST` (10× farther
/// than an ISO — rogues are tracked longer because they're heavy enough
/// to matter to outer-system dynamics).
///
/// Returns `None` if `b` has no primary host, or `MAX_BODIES` is reached.
///
/// User-triggered — from a UI button or hotkey — so it draws from
/// `b.rng` freely. Not called from `advance`.
pub fn spawn_rogue_planet(b: &mut Bodies) -> Option<usize> {
    let t = setup_inbound(b)?;

    // Mass: 0.1 to 10 M_Jup (log-uniform). Matches the free-floating
    // planet mass range from microlensing (Sumi 2011, Mróz 2017).
    let m = 0.1 * JUPITER_MASS * 100f64.powf(b.rng.next_f64());
    // Rogue planets are overwhelmingly gas giants — almost always
    // Icy class. Slight chance of a stripped super-Earth (Rocky).
    let class = match b.rng.next_f64() {
        u if u < 0.90 => BodyClass::Icy,
        _ => BodyClass::Rocky,
    };

    let i = push3(b, [t.x, t.y, t.z], [t.vx, t.vy, t.vz], m, BodyKind::Planet);
    b.origin[i] = BodyOrigin::RoguePlanet;
    b.class[i] = class;
    b.bulk_density[i] = 0.6 + 1.1 * b.rng.next_f64();
    b.r[i] = b.radius_now(i);
    b.name[i] = Some(format!("Rogue-{}", b.id[i]));

    let text = format!(
        "{} entered the system inbound ({:.1} AU closest approach)",
        tag(b, i),
        t.b_impact_au
    );
    b.log(EventKind::Interstellar, text, t.x, t.y);
    Some(i)
}
