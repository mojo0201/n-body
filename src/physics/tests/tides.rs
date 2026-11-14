//! Unit tests for the equilibrium-tide module.
//!
//! Coverage map:
//!   * Physics:  decay + spin-up, mass scaling, cutoff, outward migration,
//!               synchronous lock, eccentricity damping, Hut 1981 rate,
//!               L conservation, integrator linearity, per-step clamp
//!               (positive and negative).
//!   * Guards:   n < 2, dead planet, non-Planet body, missing parent,
//!               out-of-bounds parent, dead host, non-host parent kind,
//!               zero/NaN host radius, inside host, sub-host a, unbound
//!               orbit, zero-mass planet.
//!   * Misc:     DwarfPlanet is *not* acted on (loop filters on Planet);
//!               two-planet independence; unrelated bodies untouched.

use crate::physics::tides::{tidal_dissipation, TIDAL_BOOST, TIDAL_K2_STAR, TIDAL_Q_STAR};
use crate::physics::*;

// ─────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────

/// Sun + a planet on a circular orbit at `a_au`.
///
/// μ = G·(M_* + m_p) is used for the orbital velocity so the orbit is
/// truly circular: a_sma = r exactly, which makes the Hut-rate and
/// synchronous-lock tests well-posed.
fn setup_hot_jupiter(planet_mass: f64, a_au: f64) -> (Bodies, usize, usize) {
    let mut b = Bodies::with_capacity(2);
    let star = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.r[star] = R_SUN_GM;
    let a0 = a_au * au_in_gm();
    let v = (G * (b.m[star] + planet_mass) / a0).sqrt();
    let planet = b.push(a0, 0.0, 0.0, v, planet_mass, BodyKind::Planet);
    b.parent_cache = vec![None, Some(star)];
    b.parent_cache_valid = true;
    (b, star, planet)
}

/// Generic host-of-arbitrary-kind with a planet on a circular orbit.
fn setup_host_planet(
    host_kind: BodyKind,
    host_mass: f64,
    host_r: f64,
    planet_mass: f64,
    a_au: f64,
) -> (Bodies, usize, usize) {
    let mut b = Bodies::with_capacity(2);
    let host = b.push(0.0, 0.0, 0.0, 0.0, host_mass, host_kind);
    b.r[host] = host_r;
    let a0 = a_au * au_in_gm();
    let v = (G * (host_mass + planet_mass) / a0).sqrt();
    let planet = b.push(a0, 0.0, 0.0, v, planet_mass, BodyKind::Planet);
    b.parent_cache = vec![None, Some(host)];
    b.parent_cache_valid = true;
    (b, host, planet)
}

/// Specific-angular-momentum magnitude `|r × v|` from the host to `planet`,
/// weighted by reduced mass, plus the host spin (angular velocity × inertia
/// in code units). Conserved exactly by `tidal_dissipation`.
fn total_angular_momentum(b: &Bodies, host: usize, planet: usize) -> f64 {
    let dx = b.x[planet] - b.x[host];
    let dy = b.y[planet] - b.y[host];
    let dvx = b.vx[planet] - b.vx[host];
    let dvy = b.vy[planet] - b.vy[host];
    let m_h = b.m[host];
    let m_p = b.m[planet];
    let mu_red = m_h * m_p / (m_h + m_p);
    let l_orb = mu_red * (dx * dvy - dy * dvx);
    l_orb + b.spin[host]
}

/// Semi-major axis (in code units) from vis-viva, relative to `host`.
fn semimajor_axis(b: &Bodies, host: usize, planet: usize) -> f64 {
    let dx = b.x[planet] - b.x[host];
    let dy = b.y[planet] - b.y[host];
    let dvx = b.vx[planet] - b.vx[host];
    let dvy = b.vy[planet] - b.vy[host];
    let r = (dx * dx + dy * dy).sqrt();
    let v2 = dvx * dvx + dvy * dvy;
    let mu = G * (b.m[host] + b.m[planet]);
    -mu / (v2 - 2.0 * mu / r)
}

/// Eccentricity from the eccentricity vector.
fn eccentricity(b: &Bodies, host: usize, planet: usize) -> f64 {
    let dx = b.x[planet] - b.x[host];
    let dy = b.y[planet] - b.y[host];
    let dvx = b.vx[planet] - b.vx[host];
    let dvy = b.vy[planet] - b.vy[host];
    let r = (dx * dx + dy * dy).sqrt();
    let v2 = dvx * dvx + dvy * dvy;
    let mu = G * (b.m[host] + b.m[planet]);
    let energy = 0.5 * v2 - mu / r;
    let h = dx * dvy - dy * dvx;
    (1.0 + 2.0 * energy * h * h / (mu * mu)).max(0.0).sqrt()
}

/// Compact read of the phase-space state, used to assert "no change."
fn snapshot(b: &Bodies, i: usize) -> (f64, f64, f64, f64, f64) {
    (b.x[i], b.y[i], b.vx[i], b.vy[i], b.spin[i])
}

// ─────────────────────────────────────────────────────────────────────────
// Baseline physics
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn hot_jupiter_decays_and_spins_up_host() {
    let (mut b, star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    let v0 = b.vx[planet].hypot(b.vy[planet]);
    let spin0 = b.spin[star];

    for _ in 0..20_000 {
        tidal_dissipation(&mut b, DT);
    }

    let v1 = b.vx[planet].hypot(b.vy[planet]);
    assert!(v1 < v0, "planet velocity should shrink: {v0} -> {v1}");
    assert!(b.spin[star] > spin0, "host should spin up");
}

#[test]
fn earth_mass_barely_feels_tides() {
    // Rate ∝ q = m_p/m_*, so 1 M⊕ vs 1 M_J is a ~10⁻³ suppression.
    let (mut b, _star, planet) = setup_hot_jupiter(1.0, 0.02);
    let v0 = b.vx[planet].hypot(b.vy[planet]);
    for _ in 0..20_000 {
        tidal_dissipation(&mut b, DT);
    }
    let v1 = b.vx[planet].hypot(b.vy[planet]);
    let rel = (v1 - v0).abs() / v0;
    assert!(rel < 1e-5, "Earth-mass planet should barely change: {rel}");
}

#[test]
fn distant_planet_unchanged() {
    // 1 AU is far outside 50 R∗; the squared-distance early-out fires.
    let (mut b, _star, planet) = setup_hot_jupiter(JUPITER_MASS, 1.0);
    let v0 = b.vy[planet];
    for _ in 0..1_000 {
        tidal_dissipation(&mut b, DT);
    }
    assert_eq!(b.vy[planet], v0, "distant planet must be untouched");
}

// ─────────────────────────────────────────────────────────────────────────
// Structural guards — every early-`continue`
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn empty_and_single_body_do_nothing() {
    let mut b = Bodies::with_capacity(1);
    tidal_dissipation(&mut b, DT); // n = 0
    let s = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let v = b.vy[s];
    tidal_dissipation(&mut b, DT); // n = 1
    assert_eq!(b.vy[s], v);
}

#[test]
fn dead_planet_skipped() {
    let (mut b, _star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    b.dead[planet] = true;
    let v0 = b.vy[planet];
    tidal_dissipation(&mut b, DT);
    assert_eq!(b.vy[planet], v0);
}

#[test]
fn non_planet_body_skipped() {
    // A star with a parent cached to another star must be skipped: the
    // outer loop only iterates BodyKind::Planet.
    let mut b = Bodies::with_capacity(2);
    let s0 = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let s1 = b.push(au_in_gm(), 0.0, 0.0, 1.0, M_SUN_ME, BodyKind::Star);
    b.r[s0] = R_SUN_GM;
    b.r[s1] = R_SUN_GM;
    b.parent_cache = vec![None, Some(s0)];
    b.parent_cache_valid = true;
    let v0 = b.vy[s1];
    tidal_dissipation(&mut b, DT);
    assert_eq!(b.vy[s1], v0);
}

#[test]
fn dwarf_planet_not_acted_on() {
    // Loop filters on `BodyKind::Planet`; DwarfPlanet is intentionally
    // excluded (Pluto-class bodies raise negligible tides).
    let mut b = Bodies::with_capacity(2);
    let star = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.r[star] = R_SUN_GM;
    let a0 = 0.02 * au_in_gm();
    let v = (G * b.m[star] / a0).sqrt();
    let dp = b.push(a0, 0.0, 0.0, v, JUPITER_MASS, BodyKind::DwarfPlanet);
    b.parent_cache = vec![None, Some(star)];
    b.parent_cache_valid = true;
    let v0 = b.vy[dp];
    for _ in 0..100 {
        tidal_dissipation(&mut b, DT);
    }
    assert_eq!(b.vy[dp], v0, "DwarfPlanet must be skipped");
}

#[test]
fn missing_parent_skipped() {
    let (mut b, _star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    b.parent_cache[planet] = None;
    let v0 = b.vy[planet];
    tidal_dissipation(&mut b, DT);
    assert_eq!(b.vy[planet], v0);
}

#[test]
fn out_of_bounds_parent_skipped() {
    let (mut b, _star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    b.parent_cache[planet] = Some(9999);
    let v0 = b.vy[planet];
    tidal_dissipation(&mut b, DT);
    assert_eq!(b.vy[planet], v0);
}

#[test]
fn dead_host_skipped() {
    let (mut b, star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    b.dead[star] = true;
    let v0 = b.vy[planet];
    tidal_dissipation(&mut b, DT);
    assert_eq!(b.vy[planet], v0);
}

#[test]
fn non_host_kind_parent_skipped() {
    let mut b = Bodies::with_capacity(2);
    let p0 = b.push(0.0, 0.0, 0.0, 0.0, JUPITER_MASS, BodyKind::Planet);
    b.r[p0] = R_SUN_GM;
    let a0 = 0.02 * au_in_gm();
    let v = (G * b.m[p0] / a0).sqrt();
    let p1 = b.push(a0, 0.0, 0.0, v, JUPITER_MASS, BodyKind::Planet);
    b.parent_cache = vec![None, Some(p0)];
    b.parent_cache_valid = true;
    let v0 = b.vy[p1];
    tidal_dissipation(&mut b, DT);
    assert_eq!(b.vy[p1], v0);
}

#[test]
fn zero_host_radius_skipped() {
    let (mut b, star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    b.r[star] = 0.0;
    let v0 = b.vy[planet];
    tidal_dissipation(&mut b, DT);
    assert_eq!(b.vy[planet], v0);
}

#[test]
fn non_finite_host_radius_skipped() {
    let (mut b, star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    b.r[star] = f64::NAN;
    let v0 = b.vy[planet];
    tidal_dissipation(&mut b, DT);
    assert_eq!(b.vy[planet], v0);
}

#[test]
fn planet_inside_host_skipped() {
    // r ≤ r_host: collision logic owns this, tides must not run.
    let (mut b, star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    b.x[planet] = b.r[star];
    b.y[planet] = 0.0;
    let v0 = b.vy[planet];
    tidal_dissipation(&mut b, DT);
    assert_eq!(b.vy[planet], v0);
}

#[test]
fn sub_host_semimajor_axis_skipped() {
    // r > r_host but the bound orbit itself has a < r_host (plunging
    // orbit). The code must defer to collision handling.
    let (mut b, star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    b.x[planet] = 1.5 * b.r[star];
    b.y[planet] = 0.0;
    let v_circ = (G * b.m[star] / (1.5 * b.r[star])).sqrt();
    b.vx[planet] = 0.0;
    b.vy[planet] = 0.5 * v_circ;

    let a = semimajor_axis(&b, star, planet);
    assert!(
        a < b.r[star],
        "setup error: a = {a} should be < r_host = {}",
        b.r[star]
    );

    let v0 = b.vy[planet];
    tidal_dissipation(&mut b, DT);
    assert_eq!(b.vy[planet], v0, "orbit with a < r_host must be skipped");
}

#[test]
fn hyperbolic_orbit_skipped() {
    // Energy ≥ 0 → no bound orbit → no tide.
    let (mut b, star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    let r = b.x[planet].hypot(b.y[planet]);
    let v_esc = (2.0 * G * b.m[star] / r).sqrt();
    b.vx[planet] = 0.0;
    b.vy[planet] = 1.5 * v_esc;
    let v0 = b.vy[planet];
    tidal_dissipation(&mut b, DT);
    assert_eq!(b.vy[planet], v0);
}

#[test]
fn zero_mass_planet_skipped() {
    // q = 0 → rate = 0 → no update.
    let (mut b, _star, planet) = setup_hot_jupiter(0.0, 0.02);
    let v0 = b.vy[planet];
    tidal_dissipation(&mut b, DT);
    assert_eq!(b.vy[planet], v0);
}

// ─────────────────────────────────────────────────────────────────────────
// Sign flip: fast rotator, synchronous lock
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn fast_rotator_migrates_outward() {
    // Ω = 4 n → decay factor = −3.  Planet gains v_t, host loses spin.
    let (mut b, star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    let inertia = 0.1 * b.m[star] * b.r[star] * b.r[star]; // k = 0.1 for a star
    let r = b.x[planet].hypot(b.y[planet]);
    let n = (G * (b.m[star] + b.m[planet]) / r.powi(3)).sqrt();
    b.spin[star] = inertia * 4.0 * n;

    let v0 = b.vx[planet].hypot(b.vy[planet]);
    let spin0 = b.spin[star];
    for _ in 0..5_000 {
        tidal_dissipation(&mut b, DT);
    }
    let v1 = b.vx[planet].hypot(b.vy[planet]);
    assert!(
        v1 > v0,
        "fast rotator must drive outward migration: {v0} -> {v1}"
    );
    assert!(
        b.spin[star] < spin0,
        "host spin must decrease to feed the orbit"
    );
}

#[test]
fn synchronous_orbit_unchanged() {
    // Ω = n → decay factor = 0 → no evolution.
    let (mut b, star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    let inertia = 0.1 * b.m[star] * b.r[star] * b.r[star];
    let r = b.x[planet].hypot(b.y[planet]);
    let n = (G * (b.m[star] + b.m[planet]) / r.powi(3)).sqrt();
    b.spin[star] = inertia * n;

    let v0 = b.vy[planet];
    let s0 = b.spin[star];
    for _ in 0..1_000 {
        tidal_dissipation(&mut b, DT);
    }
    let rel_v = (b.vy[planet] - v0).abs() / v0.abs();
    let rel_s = (b.spin[star] - s0).abs() / s0.abs();
    assert!(
        rel_v < 1e-12,
        "synchronous orbit must not evolve: rel_v = {rel_v}"
    );
    assert!(
        rel_s < 1e-12,
        "synchronous spin must not evolve: rel_s = {rel_s}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Conservation, Hut 1981 rate, clamp, linearity
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn total_angular_momentum_conserved() {
    let (mut b, star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    let l0 = total_angular_momentum(&b, star, planet);
    for _ in 0..10_000 {
        tidal_dissipation(&mut b, DT);
    }
    let l1 = total_angular_momentum(&b, star, planet);
    let rel = (l1 - l0).abs() / l0.abs();
    // Per-step cancellation is exact; only fp rounding accumulates.
    assert!(rel < 1e-10, "L must be conserved to fp precision: {rel}");
}

#[test]
fn one_step_matches_hut_rate() {
    let (mut b, star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);

    let dx = b.x[planet] - b.x[star];
    let dy = b.y[planet] - b.y[star];
    let r = (dx * dx + dy * dy).sqrt();
    let dvx = b.vx[planet] - b.vx[star];
    let dvy = b.vy[planet] - b.vy[star];
    let m_h = b.m[star];
    let m_p = b.m[planet];
    let mu = G * (m_h + m_p);
    let v2 = dvx * dvx + dvy * dvy;
    let energy = 0.5 * v2 - mu / r;
    let a_sma = -mu / (2.0 * energy);

    let k2q_45 = 4.5 * TIDAL_K2_STAR * TIDAL_BOOST / TIDAL_Q_STAR;
    let q = m_p / m_h;
    let ratio = b.r[star] / a_sma;
    let ratio5 = ratio * ratio * ratio * ratio * ratio;
    let n_mean = (mu / (a_sma * a_sma * a_sma)).sqrt();
    let inv_tf = k2q_45 * q * (1.0 + q) * ratio5 * n_mean;

    let frac = (inv_tf * DT).clamp(-0.05, 0.05);
    let v_t = dvy;
    let new_vt_pred = v_t * (1.0 - 0.5 * frac);

    tidal_dissipation(&mut b, DT);
    let vt_obs = b.vy[planet];

    let rel_err = (vt_obs - new_vt_pred).abs() / new_vt_pred.abs();
    assert!(
        rel_err < 1e-13,
        "single-step v_t mismatch: pred {new_vt_pred}, obs {vt_obs}, rel_err {rel_err:e}"
    );
}

#[test]
fn per_step_change_capped_for_huge_dt() {
    // Absurd dt must still respect the ±5% per-step clamp (TIDAL_MAX_FRAC).
    let (mut b, _star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    let v0 = b.vx[planet].hypot(b.vy[planet]);
    tidal_dissipation(&mut b, DT * 1e6);
    let v1 = b.vx[planet].hypot(b.vy[planet]);
    let rel = (v0 - v1) / v0;
    assert!(rel < 0.06, "positive clamp should cap change: rel = {rel}");
}

#[test]
fn negative_frac_clamped_for_huge_spin() {
    // Fast rotator with absurd spin + huge dt: the negative clamp must
    // fire, and the velocity must not blow up.
    let (mut b, star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    let inertia = 0.1 * b.m[star] * b.r[star] * b.r[star];
    let r = b.x[planet].hypot(b.y[planet]);
    let n = (G * (b.m[star] + b.m[planet]) / r.powi(3)).sqrt();
    b.spin[star] = inertia * 1e6 * n; // Ω = 10⁶ n

    let v0 = b.vx[planet].hypot(b.vy[planet]);
    tidal_dissipation(&mut b, DT * 1e6);
    let v1 = b.vx[planet].hypot(b.vy[planet]);
    let rel = (v1 - v0).abs() / v0;
    assert!(rel < 0.10, "negative clamp must cap change: rel = {rel}");
}

#[test]
fn accumulation_is_linear_in_steps() {
    // Two integrations with equal total step should agree to fp precision
    // (the per-step update is O(frac); a single vs. two half-steps differ
    // only at O(frac²), well below the tolerance).
    let (mut a, _sa, pa) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    let (mut b, _sb, pb) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    for _ in 0..1_000 {
        tidal_dissipation(&mut a, DT);
    }
    for _ in 0..2_000 {
        tidal_dissipation(&mut b, DT * 0.5);
    }
    let va = a.vx[pa].hypot(a.vy[pa]);
    let vb = b.vx[pb].hypot(b.vy[pb]);
    let rel = (va - vb).abs() / va;
    assert!(rel < 1e-6, "integrator should be ≈linear in dt: {rel}");
}

// ─────────────────────────────────────────────────────────────────────────
// Eccentricity damping
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn eccentric_orbit_circularizes() {
    // Circular setup then boost v_t by 15% → e ≈ 0.32. Tides should reduce e.
    let (mut b, star, planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    let v_circ = b.vy[planet];
    b.vy[planet] = 1.15 * v_circ;

    let e0 = eccentricity(&b, star, planet);
    assert!(
        e0 > 0.2,
        "setup did not yield suitable eccentricity: e0 = {e0}"
    );

    for _ in 0..20_000 {
        tidal_dissipation(&mut b, DT);
    }
    let e1 = eccentricity(&b, star, planet);
    assert!(e1 < e0, "eccentricity must decrease: {e0} -> {e1}");
}

// ─────────────────────────────────────────────────────────────────────────
// Host inertia, multiple planets, unrelated bodies
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn host_inertia_affects_first_step_decay() {
    // Same L_spin on hosts of different k gives different Ω/n, hence a
    // different decay factor. For k_star = 0.1, k_bd = 0.15 and
    // L_spin = I_star · 0.5 n:
    //   Ω_star = 0.5 n   → decay = 0.5
    //   Ω_bd   = n / 3   → decay = 2/3
    // So Δv_star / Δv_bd = 0.75 on the first step.
    let (mut bs, ss, ps) =
        setup_host_planet(BodyKind::Star, M_SUN_ME, R_SUN_GM, JUPITER_MASS, 0.02);
    let (mut bb, sb, pb) =
        setup_host_planet(BodyKind::BrownDwarf, M_SUN_ME, R_SUN_GM, JUPITER_MASS, 0.02);

    let inertia_star = 0.1 * bs.m[ss] * bs.r[ss] * bs.r[ss];
    let r = bs.x[ps].hypot(bs.y[ps]);
    let n = (G * (bs.m[ss] + bs.m[ps]) / r.powi(3)).sqrt();
    let l_spin = inertia_star * 0.5 * n;
    bs.spin[ss] = l_spin;
    bb.spin[sb] = l_spin;

    let vs0 = bs.vy[ps];
    let vb0 = bb.vy[pb];
    tidal_dissipation(&mut bs, DT);
    tidal_dissipation(&mut bb, DT);
    let dv_s = vs0 - bs.vy[ps];
    let dv_b = vb0 - bb.vy[pb];

    let ratio = dv_s / dv_b;
    assert!(
        (ratio - 0.75).abs() < 0.02,
        "first-step decay ratio should be ≈ 0.75, got {ratio}"
    );
}

#[test]
fn two_planets_decay_independently() {
    let mut b = Bodies::with_capacity(3);
    let star = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.r[star] = R_SUN_GM;
    let a0 = 0.02 * au_in_gm();
    let v0 = (G * b.m[star] / a0).sqrt();
    // Two Jupiters at slightly different a, so their v_t differ slightly.
    let p1 = b.push(a0, 0.0, 0.0, v0, JUPITER_MASS, BodyKind::Planet);
    let p2 = b.push(0.0, a0 * 1.1, -v0, 0.0, JUPITER_MASS, BodyKind::Planet);
    b.parent_cache = vec![None, Some(star), Some(star)];
    b.parent_cache_valid = true;

    let v1_0 = b.vx[p1].hypot(b.vy[p1]);
    let v2_0 = b.vx[p2].hypot(b.vy[p2]);
    let spin0 = b.spin[star];
    for _ in 0..1_000 {
        tidal_dissipation(&mut b, DT);
    }
    let v1_1 = b.vx[p1].hypot(b.vy[p1]);
    let v2_1 = b.vx[p2].hypot(b.vy[p2]);
    assert!(v1_1 < v1_0);
    assert!(v2_1 < v2_0);
    assert!(b.spin[star] > spin0, "both planets must feed the same spin");
}

#[test]
fn unrelated_bodies_untouched() {
    // A body with no parent in the cache must come through unchanged.
    let (mut b, _star, _planet) = setup_hot_jupiter(JUPITER_MASS, 0.02);
    let ast = b.push(10.0 * au_in_gm(), 0.0, 0.0, 0.0, 0.0, BodyKind::DwarfPlanet);
    b.parent_cache.push(None);
    let before = snapshot(&b, ast);
    for _ in 0..100 {
        tidal_dissipation(&mut b, DT);
    }
    assert_eq!(snapshot(&b, ast), before);
}
