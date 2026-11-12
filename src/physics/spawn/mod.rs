//! Random system spawn: samplers, Kepler helpers, primary and system entry points.
//!
//! Large parts live in the `binary`, `systems`, `cluster` and `belts`
//! sub-modules. Everything is re-exported from here.

use super::*;

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
    let angle = rng.next_f64() * std::f64::consts::TAU;
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
    b.bulk_density[moon] = draw_density(rng, false);
    b.r[moon] = b.radius_now(moon);
    b.parent_cache_valid = false;
    let (x, y) = (b.x[moon], b.y[moon]);
    let text = format!("{} spawned as a moon of {}", tag(b, moon), tag(b, i));
    b.log(EventKind::User, text, x, y);
    Some(moon)
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
    let radii: Vec<f64> = (0..n)
        .map(|_| r0 * 1.4 * (2.6_f64 / 1.4).powf(rng.next_f64()))
        .collect();
    let weights: Vec<f64> = radii.iter().map(|&r| (r / r0).sqrt()).collect();
    let wsum: f64 = weights.iter().sum();
    for (r, w) in radii.iter().zip(weights.iter()) {
        let m = m_total * w / wsum;
        let angle = rng.next_f64() * std::f64::consts::TAU;
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
        b.bulk_density[k] = draw_density(rng, false);
        b.r[k] = b.radius_now(k);
        b.parent_cache_valid = false;
        out.push(k);
    }
    if !out.is_empty() {
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

/// Draw `n` masses from a truncated power law, then scale so their sum is
/// `total`. Caps each tracer at `BELT_MAX_TRACER_MASS` and at `total/5`
/// (no single tracer should exceed 20 % of its belt — such a body would
/// act as a single perturber, not as a population). The power-law *shape*
/// is preserved; only the absolute scale changes, so relative body sizes
/// stay physical.
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
    let mut v: Vec<f64> = (0..n)
        .map(|_| sample_mass_power_law(rng, m_min, m_max, plaw))
        .collect();
    let sum: f64 = v.iter().sum();
    if sum <= 0.0 {
        return v;
    }
    let scale = total / sum;
    let cap = BELT_MAX_TRACER_MASS.min(total / 5.0);
    for m in v.iter_mut() {
        *m = (*m * scale).min(cap);
    }
    v
}

pub(super) fn spacing_range(n: usize, a_min: f64, a_max: f64, base_lo: f64, base_hi: f64) -> (f64, f64) {
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
    let r = a * (1.0 - e * e) / (1.0 + e * cn);
    let h = (mu * a * (1.0 - e * e)).sqrt();
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
    2.0 * ((1.0 + e).sqrt() * (ecc_anom * 0.5).sin())
        .atan2((1.0 - e).sqrt() * (ecc_anom * 0.5).cos())
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
    let omega = rng.next_f64() * std::f64::consts::TAU;
    let mean_anom = rng.next_f64() * std::f64::consts::TAU;
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
    b.bulk_density[i] = if m >= GIANT_DENSITY_M_MIN {
        // Gas giants: 0.6 (Saturn) to 1.7 (Neptune) g/cm³, at any distance.
        0.6 + 1.1 * rng.next_f64()
    } else {
        draw_density(rng, (a_gm / au_in_gm()) < snow_line_au)
    };
    b.r[i] = b.radius_now(i);
    i
}

/// Planets at or above this mass (Earth masses) get gas-giant densities.
pub(super) const GIANT_DENSITY_M_MIN: f64 = 50.0;

/// Bulk density in g/cm³: rocky inside the snow line, icy outside.
pub(super) fn draw_density(rng: &mut XorShift, inside_snow: bool) -> f64 {
    if inside_snow {
        2.5 + 3.0 * rng.next_f64()
    } else {
        0.5 + 2.0 * rng.next_f64()
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
    let u = rng.next_f64();
    if u < 0.76 {
        let u2 = u / 0.76;
        let a = 0.08_f64.powf(-0.3);
        let c = 0.50_f64.powf(-0.3);
        (a + u2 * (c - a)).powf(1.0 / -0.3)
    } else {
        let u2 = (u - 0.76) / 0.24;
        let a = 0.50_f64.powf(-1.3);
        let c = 50.0_f64.powf(-1.3);
        (a + u2 * (c - a)).powf(1.0 / -1.3)
    }
}

/// Draw the primary of a random system.
///
/// The 85 / 15 split between stellar-mass and brown-dwarf primaries is a
/// coarse population-synthesis approximation, not a direct IMF integral.
/// Kroupa (2001) with a lower cutoff of 0.08 M_sun gives ~85 % of draws
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
    mu + sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
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
    std::f64::consts::TAU * (a_gm.powi(3) / (G * m_tot)).sqrt() * time_unit_s() / 86_400.0
}

pub fn spawn_system(b: &mut Bodies, body_count: usize, seed: u32) {
    spawn_system_with_info(b, body_count, seed);
}

/// Like `spawn_system`, but also reports what was generated (binary
/// parameters and architecture).
pub fn spawn_system_with_info(b: &mut Bodies, body_count: usize, seed: u32) -> SystemInfo {
    *b = Bodies::with_capacity(body_count);
    b.rng = XorShift::new(seed ^ 0xA5A5_5A5A);
    let mut rng = XorShift::new(seed);

    // Host metallicity [Fe/H] in dex (log-normal in Z): mean 0, sigma 0.20.
    b.metallicity = sample_normal(&mut rng, 0.0, 0.20);
    let (m_init, kind0) = sample_primary(&mut rng);
    let prim = spawn_aged_primary(b, m_init, kind0, &mut rng);
    let kind = b.kind[prim];

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
                    P_TRIPLE_GIVEN_BINARY
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
