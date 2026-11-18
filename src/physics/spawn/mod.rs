//! Random system spawn: samplers, Kepler helpers, primary and system entry points.
//!
//! Large parts live in the `binary`, `systems`, `cluster` and `belts`
//! sub-modules. Everything is re-exported from here.

use super::*;
use std::f64::consts::TAU;
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
    let (sa, ca) = angle.sin_cos();
    let v_circ = (G * b.m[i] / a).sqrt();
    let m_moon = b.m[i] * 10f64.powf(-4.0 + rng.next_f64() * 2.0);
    let moon = b.push(
        b.x[i] + a * ca,
        b.y[i] + a * sa,
        b.vx[i] - v_circ * sa,
        b.vy[i] + v_circ * ca,
        m_moon,
        BodyKind::Planet,
    );
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
    vx: f64,
    vy: f64,
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

    Some(Inbound {
        x: sx,
        y: sy,
        vx: v_r * ct - v_t * st,
        vy: v_r * st + v_t * ct,
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

    let i = b.push(t.x, t.y, t.vx, t.vy, m, BodyKind::Planet);
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
    for &r in &radii {
        let w = (r / r0).sqrt();
        let m = m_total * w / wsum;
        let angle = rng.next_f64() * TAU;
        let (sa, ca) = angle.sin_cos();
        let v_circ = (G * b.m[i] / r).sqrt();
        let k = b.push(
            b.x[i] + r * ca,
            b.y[i] + r * sa,
            b.vx[i] - v_circ * sa,
            b.vy[i] + v_circ * ca,
            m,
            BodyKind::Planet,
        );
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
    let (x, y, vx, vy) = kepler_state(G * (mc + m), a, e, nu, omega);
    b.push(cx + x, cy + y, cvx + vx, cvy + vy, m, kind)
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
        // Log-uniform: the brown-dwarf mass function declines toward the
        // hydrogen-burning limit.
        let m = BD_M_MIN * (BD_M_MAX / BD_M_MIN).powf(rng.next_f64());
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
    compute_accel(b);
    rebase_energy(b);
    dust::init_fields(b, None);
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

    let i = b.push(t.x, t.y, t.vx, t.vy, m, BodyKind::Planet);
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
