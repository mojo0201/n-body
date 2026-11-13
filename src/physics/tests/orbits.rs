//! Tests: binaries, orbit_info, escape culling, energy diagnostics, kind transitions, user edits.

use super::super::*;
use super::shared::*;

// ===========================================================================
// k — binaries and circumbinary stability
// ===========================================================================
#[test]
fn k_hw_formulas_match_published_values() {
    // Equal-mass circular binary: S-type ~0.27 a_b, P-type ~2.4 a_b.
    let s = hw_critical_s(1.0, 0.0, 0.5);
    let p = hw_critical_p(1.0, 0.0, 0.5);
    assert!((s - 0.274).abs() < 0.01, "S {}", s);
    assert!((p - 2.39).abs() < 0.05, "P {}", p);
    assert!(hw_critical_s(1.0, 0.5, 0.5) < s);
    assert!(hw_critical_p(1.0, 0.5, 0.5) > p);
}

#[test]
fn k_binary_ic_respects_stability_limits() {
    let au = au_in_gm();
    let (mut stars, mut bins, mut n_s, mut n_p) = (0, 0, 0, 0);
    for seed in 0..600u32 {
        let mut b = Bodies::with_capacity(60);
        let info = spawn_system_with_info(&mut b, 60, seed);
        assert!(b.len() <= 60, "seed {}: {} bodies", seed, b.len());
        // Companion statistics are measured for MS-star primaries only.
        // Brown dwarfs get a BD companion with 20% probability (Luhman 16
        // and similar systems) — skip them; they don't count toward the
        // MS binary fraction.
        if b.kind[0] == BodyKind::BrownDwarf {
            continue;
        }
        // Remnants (WD/NS/BH) never get a companion in `spawn_system_with_info`:
        // `wants_companion` is false for every kind other than Star and
        // BrownDwarf. Assert that invariant explicitly.
        if b.kind[0] != BodyKind::Star {
            assert!(
                info.binary.is_none(),
                "remnant primaries must not have companions"
            );
            continue;
        }
        stars += 1;
        let Some(bi) = info.binary else { continue };
        bins += 1;
        assert_eq!(bi.companion, 1);
        assert_eq!(b.kind[1], BodyKind::Star);
        assert!(
            b.m[1] <= b.m[0] && b.m[1] >= STAR_M_MIN * 0.999,
            "seed {}",
            seed
        );
        assert!(
            bi.period_days >= BIN_P_MIN_D * 0.999,
            "seed {} P {}",
            seed,
            bi.period_days
        );
        assert!(
            bi.a_au <= BIN_A_MAX_AU * 1.001,
            "seed {} a {}",
            seed,
            bi.a_au
        );
        let peri = bi.a_au * au * (1.0 - bi.ecc);
        assert!(
            peri >= BIN_PERI_CLEAR * (b.r[0] + b.r[1]) * 0.999,
            "seed {}",
            seed
        );
        if bi.period_days < BIN_CIRC_P_D {
            assert_eq!(bi.ecc, 0.0);
        }
        let bary = star_pair_centre(&b);
        let rel = (b.x[0], b.y[0], b.vx[0], b.vy[0], b.m[0] + b.m[1]);
        let a_rel = osc_a(&b, 1, rel) / au;
        assert!(
            (a_rel / bi.a_au - 1.0).abs() < 1e-6,
            "seed {} {} vs {}",
            seed,
            a_rel,
            bi.a_au
        );
        // Moons orbit their host *planet*, not the star (S-type) or the
        // binary barycentre (P-type). Measured against either reference,
        // a moon's osculating `a` is not the stability quantity: its
        // velocity includes v_planet, so it can land well outside the
        // stability limit or even on a hyperbolic orbit (a < 0) while
        // remaining perfectly bound to its parent planet. Skip them.
        // `find_parent` returns a Planet exactly for moons; hosts
        // (including a binary's lead star for circumbinary planets)
        // return a Star.
        let is_moon = |bb: &Bodies, i: usize| -> bool {
            matches!(
                find_parent(bb, i),
                Some(p) if matches!(bb.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet)
            )
        };
        match bi.arch {
            BinaryArch::PType => {
                n_p += 1;
                let lim = HW_MARGIN_P * bi.a_crit_au * au;
                for i in 2..b.len() {
                    // Skip non-planets (e.g. the third star in a triple). A
                    // third star orbits the whole inner system (m_in + m_third),
                    // not just the inner pair, so `osc_a` about `star_pair_centre`
                    // (m0+m1) is not a stability quantity for it — it can be
                    // negative even when the star is bound.
                    if b.origin[i] != BodyOrigin::Spawned
                        || !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
                        || is_moon(&b, i)
                    {
                        continue;
                    }
                    let a = osc_a(&b, i, bary);
                    assert!(
                        a >= lim * 0.999,
                        "seed {} body {} a {} < {}",
                        seed,
                        i,
                        a / au,
                        lim / au
                    );
                }
            }
            BinaryArch::STypePrimary | BinaryArch::STypeSecondary => {
                n_s += 1;
                let h = if bi.arch == BinaryArch::STypePrimary {
                    0
                } else {
                    1
                };
                let c = (b.x[h], b.y[h], b.vx[h], b.vy[h], b.m[h]);
                let lim = HW_MARGIN_S * bi.a_crit_au * au;
                for i in 2..b.len() {
                    // Same reasoning as the P-type branch: exclude third
                    // (and fourth) stars, which are not planetary stability
                    // tests.
                    if b.origin[i] != BodyOrigin::Spawned
                        || !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
                        || is_moon(&b, i)
                    {
                        continue;
                    }
                    let a = osc_a(&b, i, c);
                    assert!(
                        a > 0.0 && a <= lim * 1.001,
                        "seed {} body {} a {} > {}",
                        seed,
                        i,
                        a / au,
                        lim / au
                    );
                }
            }
        }
    }
    let frac = bins as f64 / stars as f64;
    eprintln!(
        "binaries {}/{} = {:.2}  S {} P {}",
        bins, stars, frac, n_s, n_p
    );
    assert!(frac > 0.2 && frac < 0.6, "binary fraction {}", frac);
    assert!(n_s > 0 && n_p > 0, "both architectures must occur");
}

#[test]
fn k_parent_follows_binary() {
    let mut b = Bodies::with_capacity(8);
    let (m1, m2) = (M_SUN_ME, 0.5 * M_SUN_ME);
    let mt = m1 + m2;
    let a_b = 0.1 * au_in_gm();
    let v = (G * mt / a_b).sqrt();
    b.push(-m2 / mt * a_b, 0.0, 0.0, -m2 / mt * v, m1, BodyKind::Star);
    b.push(m1 / mt * a_b, 0.0, 0.0, m1 / mt * v, m2, BodyKind::Star);
    let mut rng = XorShift::new(7);
    let a_p = 3.0 * au_in_gm();
    let circ = push_kepler_about(
        &mut b,
        (0.0, 0.0, 0.0, 0.0, mt),
        a_p,
        0.0,
        1.0,
        &mut rng,
        BodyKind::Planet,
    );
    let near2 = push_kepler(
        &mut b,
        1,
        0.01 * au_in_gm(),
        0.0,
        1.0,
        &mut rng,
        BodyKind::Planet,
    );
    assert_eq!(binary_partner(&b, 0), Some(1));
    assert_eq!(binary_partner(&b, 1), Some(0));
    assert_eq!(
        find_parent(&b, circ),
        Some(0),
        "circumbinary planet -> heavier star"
    );
    assert_eq!(
        find_parent(&b, near2),
        Some(1),
        "S-type planet -> its own star"
    );
    b.refresh_parent_cache();
    let c = orbit_reference(&b, &host_list(&b), circ, (0.0, 0.0, 0.0, 0.0, 0.0));
    assert!(
        (c.4 - mt).abs() < 1e-9,
        "circumbinary orbit is drawn about the pair"
    );
    let s = orbit_reference(&b, &host_list(&b), near2, (0.0, 0.0, 0.0, 0.0, 0.0));
    assert!(
        (s.4 - m2).abs() < 1e-9,
        "S-type orbit is drawn about its own star"
    );
    assert!(!compute_orbit_segments(&b, 1.0, true, true).is_empty());
}

#[test]
fn k_circumbinary_planet_stays_bound() {
    let au = au_in_gm();
    let (m1, m2) = (M_SUN_ME, 0.6 * M_SUN_ME);
    let mt = m1 + m2;
    let (a_b_au, e_b) = (0.2, 0.1);
    let a_b = a_b_au * au;
    let mut b = Bodies::with_capacity(8);
    let (x, y, vx, vy) = kepler_state(G * mt, a_b, e_b, 0.7, 0.3);
    b.push(
        -m2 / mt * x,
        -m2 / mt * y,
        -m2 / mt * vx,
        -m2 / mt * vy,
        m1,
        BodyKind::Star,
    );
    b.push(
        m1 / mt * x,
        m1 / mt * y,
        m1 / mt * vx,
        m1 / mt * vy,
        m2,
        BodyKind::Star,
    );
    let a_p = HW_MARGIN_P * hw_critical_p(a_b_au, e_b, m2 / mt) * au;
    let mut rng = XorShift::new(11);
    push_kepler_about(
        &mut b,
        (0.0, 0.0, 0.0, 0.0, mt),
        a_p,
        0.0,
        1.0,
        &mut rng,
        BodyKind::Planet,
    );
    recenter_com(&mut b);
    compute_accel(&mut b);
    rebase_energy(&mut b);
    for _ in 0..4000 {
        advance(&mut b, DT);
    }
    assert_eq!(b.len(), 3, "nothing merged, escaped or broke up");
    let bary = star_pair_centre(&b);
    let a_end = osc_a(&b, 2, bary);
    let drift = energy_drift_pct(&b).unwrap().abs();
    eprintln!(
        "planet a {:.3} -> {:.3} AU, binary sep {:.4} AU, energy drift {:.4}%, max rung {}",
        a_p / au,
        a_end / au,
        (b.x[1] - b.x[0]).hypot(b.y[1] - b.y[0]) / au,
        drift,
        b.stats.max_rung
    );
    assert!(
        a_end > 0.8 * a_p && a_end < 1.25 * a_p,
        "planet a {} -> {}",
        a_p / au,
        a_end / au
    );
    assert!(drift < 0.5, "energy drift {}%", drift);
    assert_eq!(b.stats.merges + b.stats.tidal, 0);
}

// ===========================================================================
// o — orbit info, circularize, host_distance
// ===========================================================================
/// `orbit_info` must recover (a, e, r, v) exactly for a circular orbit.
#[test]
fn o_orbit_info_circular_round_trip() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0;
    let mu = G * (M_SUN_ME + 1.0);
    let v = (mu / a).sqrt();
    b.push(a, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    let o = orbit_info(&b, 1, 0);
    assert!(o.bound);
    assert!(o.ecc.abs() < 1e-9, "ecc {}", o.ecc);
    assert!((o.semi_major - a).abs() < 1e-9, "a {}", o.semi_major);
    assert!((o.dist - a).abs() < 1e-9);
    assert!((o.speed_rel - v).abs() < 1e-9);
    let expected_period = std::f64::consts::TAU * (a * a * a / mu).sqrt();
    assert!(
        (o.period - expected_period).abs() / expected_period < 1e-9,
        "period {} vs {}",
        o.period,
        expected_period
    );
}

/// `orbit_info` must recover (a, e) for an elliptical orbit generated by
/// `kepler_state`.
#[test]
fn o_orbit_info_elliptical_round_trip() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 10.0;
    let e = 0.3;
    let mu = G * (M_SUN_ME + 1.0);
    let (x, y, vx, vy) = kepler_state(mu, a, e, 1.1, 0.4);
    b.push(x, y, vx, vy, 1.0, BodyKind::Planet);
    let o = orbit_info(&b, 1, 0);
    assert!(o.bound);
    assert!((o.ecc - e).abs() < 1e-8, "ecc {} vs {}", o.ecc, e);
    assert!(
        (o.semi_major - a).abs() < 1e-8,
        "a {} vs {}",
        o.semi_major,
        a
    );
    assert!((o.peri - a * (1.0 - e)).abs() < 1e-8);
    assert!((o.apo - a * (1.0 + e)).abs() < 1e-8);
}

/// A body above escape speed must be reported as unbound with infinite
/// `semi_major` / `period` and `ecc > 1`.
#[test]
fn o_orbit_info_hyperbolic_unbound() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let r = 5.0;
    let v_esc = (2.0 * G * M_SUN_ME / r).sqrt();
    b.push(r, 0.0, 0.0, 2.0 * v_esc, 1.0, BodyKind::Planet);
    let o = orbit_info(&b, 1, 0);
    assert!(!o.bound);
    assert_eq!(o.semi_major, f64::INFINITY);
    assert_eq!(o.period, f64::INFINITY);
    assert!(o.ecc > 1.0, "ecc {}", o.ecc);
}

/// `circularize` must zero the eccentricity and preserve the orbital radius.
#[test]
fn o_circularize_zeroes_eccentricity() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0;
    let mu = G * (M_SUN_ME + 1.0);
    let (x, y, vx, vy) = kepler_state(mu, a, 0.4, 1.2, 0.5);
    b.push(x, y, vx, vy, 1.0, BodyKind::Planet);
    let r_before = (b.x[1] * b.x[1] + b.y[1] * b.y[1]).sqrt();
    assert!(circularize(&mut b, 1));
    let o = orbit_info(&b, 1, 0);
    assert!(o.ecc < 1e-10, "ecc after circularize: {}", o.ecc);
    let r_after = (b.x[1] * b.x[1] + b.y[1] * b.y[1]).sqrt();
    assert!((r_after - r_before).abs() < 1e-10);
}

/// `host_distance` for a moon must return the distance to the *host star*,
/// not to its parent planet — that's the value the renderer uses for the
/// planet colour gradient, and getting it wrong would collapse a Jupiter
/// moon onto Jupiter's distance.
#[test]
fn o_host_distance_from_moon() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0 * au_in_gm();
    let v_jup = (G * M_SUN_ME / a).sqrt();
    b.push(a, 0.0, 0.0, v_jup, 318.0, BodyKind::Planet);
    // Moon 0.1 Gm from Jupiter, at Jupiter's velocity plus the moon's own
    // circular velocity around Jupiter.
    let v_moon = v_jup + (G * 318.0 / 0.1).sqrt();
    b.push(a + 0.1, 0.0, 0.0, v_moon, 0.01, BodyKind::Planet);
    b.refresh_parent_cache();
    assert_eq!(find_parent(&b, 2), Some(1), "moon must orbit Jupiter");
    let d = host_distance(&b, 2);
    let expected = a + 0.1;
    assert!(
        (d - expected).abs() < 0.01,
        "host_distance = {} (expected ~{})",
        d,
        expected
    );
}

// ===========================================================================
// p — escape culling
// ===========================================================================
/// A planet beyond CULL_DIST moving radially outward at > v_esc must be
/// culled and logged.
#[test]
fn p_escape_cull_removes_outward_unbound() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let d = CULL_DIST * 2.0;
    let v_esc = (2.0 * G * M_SUN_ME / d).sqrt();
    b.push(d, 0.0, 2.0 * v_esc, 0.0, 1.0, BodyKind::Planet);
    cull_escaped(&mut b);
    assert_eq!(b.len(), 1);
    assert_eq!(b.stats.escaped, 1);
}

/// A planet beyond CULL_DIST moving radially inward must be kept (it may
/// still return) even if its speed exceeds v_esc.
#[test]
fn p_escape_cull_keeps_inward() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let d = CULL_DIST * 2.0;
    let v_esc = (2.0 * G * M_SUN_ME / d).sqrt();
    b.push(d, 0.0, -2.0 * v_esc, 0.0, 1.0, BodyKind::Planet);
    cull_escaped(&mut b);
    assert_eq!(b.len(), 2, "inward body must not be culled");
    assert_eq!(b.stats.escaped, 0);
}

/// Beyond HARD_CULL_DIST the body is removed regardless of velocity.
#[test]
fn p_escape_cull_hard_removes_regardless() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let d = HARD_CULL_DIST * 1.5;
    b.push(d, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    cull_escaped(&mut b);
    assert_eq!(b.len(), 1);
}

// ===========================================================================
// q — energy diagnostics
// ===========================================================================
/// On a stable solar system the raw integrator drift stays under 0.5 %.
#[test]
fn q_energy_drift_small_on_solar() {
    let mut b = Bodies::with_capacity(16);
    spawn_solar_system(&mut b);
    let mut max_drift = 0.0_f64;
    for _ in 0..500 {
        advance(&mut b, DT);
        if let Some(d) = energy_drift_pct(&b) {
            max_drift = max_drift.max(d.abs());
        }
    }
    assert!(max_drift < 0.5, "max drift {:.4}%", max_drift);
}

/// Above `ENERGY_MAX_N` the diagnostic helper must return `None` rather
/// than perform the O(n²) sum.
#[test]
fn q_energy_drift_returns_none_above_limit() {
    let mut b = Bodies::with_capacity(ENERGY_MAX_N + 10);
    for k in 0..ENERGY_MAX_N + 10 {
        b.push(k as f64, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    }
    rebase_energy(&mut b);
    assert_eq!(energy_drift_pct(&b), None);
    assert_eq!(energy_dissipated_pct(&b), None);
}

/// `settle_energy` must accumulate `energy_offset` by exactly
/// (post-collision energy - pre-collision energy).
#[test]
fn q_settle_energy_computes_offset() {
    let mut b = pair(1.0, 0.3, 0.4, 0.6);
    shift(&mut b, 40.0, -25.0, 3.0, 2.0);
    rebase_energy(&mut b);
    let e_pre = total_energy(&b).unwrap();
    collision_pass(&mut b, DT, None);
    settle_energy(&mut b);
    let e_post = total_energy(&b).unwrap();
    let expected = e_post - e_pre;
    assert!(
        (b.stats.energy_offset - expected).abs() < 1e-9 * (1.0 + expected.abs()),
        "offset {} vs expected {}",
        b.stats.energy_offset,
        expected
    );
}

// ===========================================================================
// r — kind transitions via merge
// ===========================================================================
/// A sub-`STAR_M_MIN` "star" that gains a tiny bit of mass must be
/// demoted to a brown dwarf.
#[test]
fn r_star_to_bd_transition() {
    let mut b = Bodies::with_capacity(2);
    let tiny_star_mass = 0.5 * STAR_M_MIN;
    b.push(0.0, 0.0, 0.0, 0.0, tiny_star_mass, BodyKind::Star);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1);
    assert_eq!(b.kind[0], BodyKind::BrownDwarf, "demoted to BD");
    assert!(b.m[0] >= BD_M_MIN);
}

/// Two brown dwarfs whose summed mass exceeds `BD_M_MAX` must ignite.
#[test]
fn r_bd_to_star_transition() {
    let mut b = Bodies::with_capacity(2);
    // 45 M_jup each = 14310 M_earth; sum 28620 > BD_M_MAX = 25440.
    let m_bd = 45.0 * JUPITER_MASS;
    b.push(0.0, 0.0, 0.0, 0.0, m_bd, BodyKind::BrownDwarf);
    b.push(0.0, 0.0, 0.0, 0.0, m_bd, BodyKind::BrownDwarf);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1);
    assert_eq!(b.kind[0], BodyKind::Star, "ignited as a star");
    assert!(b.m[0] >= STAR_M_MIN);
}

// ===========================================================================
// s — user edits
// ===========================================================================
/// `scale_mass(i, 2)` then `scale_mass(i, 0.5)` must return to the original
/// mass and radius.
#[test]
fn s_scale_mass_round_trip() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let (m0, r0) = (b.m[0], b.r[0]);
    scale_mass(&mut b, 0, 2.0);
    assert!((b.m[0] - 2.0 * m0).abs() < 1e-12);
    scale_mass(&mut b, 0, 0.5);
    assert!((b.m[0] - m0).abs() < 1e-12);
    assert!((b.r[0] - r0).abs() < 1e-12, "radius round-trips too");
}

/// `set_kind(BlackHole)` on a low-mass body must clamp its mass to the
/// black-hole minimum.
#[test]
fn s_set_kind_clamps_mass() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    set_kind(&mut b, 0, BodyKind::BlackHole);
    assert_eq!(b.kind[0], BodyKind::BlackHole);
    assert!(b.m[0] >= BH_M_MIN, "mass {} < BH_M_MIN", b.m[0]);
}

/// Deleting a body must compact the arrays and shift surviving ids down.
#[test]
fn s_delete_body_compacts() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tail_id = b.push(2.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tail = b.id[tail_id];
    delete_body(&mut b, 1);
    assert_eq!(b.len(), 2);
    assert_eq!(b.id[1], tail, "surviving body shifts down");
}

#[test]
fn s_rng_differs_across_seeds() {
    let mut a = Bodies::with_capacity(0);
    spawn_random_systems(&mut a, 100, 42, 1);
    let r1 = a.rng.next_u32();
    let mut b = Bodies::with_capacity(0);
    spawn_random_systems(&mut b, 100, 43, 1);
    let r2 = b.rng.next_u32();
    assert_ne!(r1, r2);
}

#[test]
fn s_seed_base62_roundtrip() {
    for seed in [
        0u32,
        1,
        61,
        62,
        63,
        1000,
        u32::MAX / 2,
        u32::MAX - 1,
        u32::MAX,
    ] {
        let s = seed_to_base62(seed);
        assert_eq!(s.len(), 6);
        assert_eq!(seed_from_base62(&s), Some(seed));
    }
    assert_eq!(seed_from_base62(""), None);
    assert_eq!(seed_from_base62("0000001"), None); // too long
    assert_eq!(seed_from_base62("!@#$%^"), None); // bad chars
    assert_eq!(seed_from_base62("zzzzzz"), None); // above u32::MAX
}

#[test]
fn s_preset_dwarf_planets_are_dwarfs() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    // The 5 dwarfs, by mass.
    for m in [0.000159, 0.00218, 0.00066, 0.00052, 0.00278] {
        let i = (0..b.len())
            .find(|&i| (b.m[i] - m).abs() < 1e-6 && b.origin[i] == BodyOrigin::Spawned)
            .unwrap_or_else(|| panic!("no body of mass {m}"));
        assert_eq!(b.kind[i], BodyKind::DwarfPlanet, "mass {}", m);
    }
}

#[test]
fn s_preset_phases_are_not_all_periapsis() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    b.refresh_parent_cache();
    let mut n_planets = 0usize;
    let mut n_at_peri = 0usize;
    for i in 1..b.len() {
        if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
            continue;
        }
        if b.origin[i] != BodyOrigin::Spawned {
            continue;
        }
        if matches!(
            find_parent(&b, i),
            Some(p) if matches!(b.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet)
        ) {
            continue; // a moon — skip
        }
        n_planets += 1;
        let o = orbit_info(&b, i, 0);
        if (o.dist / o.peri - 1.0).abs() < 1e-3 {
            n_at_peri += 1;
        }
    }
    // Random phases: most planets must NOT be at periapsis.
    assert!(
        n_at_peri * 2 < n_planets,
        "too many planets at periapsis: {}/{}",
        n_at_peri,
        n_planets
    );
}

#[test]
fn s_remnant_fraction_is_realistic() {
    let (mut wd, mut ns, mut bh, mut n) = (0, 0, 0, 0);
    for seed in 0..2000u32 {
        let mut b = Bodies::with_capacity(3);
        spawn_system(&mut b, 3, seed);
        n += 1;
        match b.kind[0] {
            BodyKind::WhiteDwarf => wd += 1,
            BodyKind::NeutronStar => ns += 1,
            BodyKind::BlackHole => bh += 1,
            _ => {}
        }
    }
    let bhbns = (ns + bh) as f64 / n as f64;
    assert!(bhbns < 0.10, "NS+BH fraction {}", bhbns);
    assert!((wd as f64 / n as f64) < 0.35, "WD fraction too high");
}

#[test]
fn s_moons_spawn_around_giants() {
    let mut total_moons = 0;
    for seed in 0..100u32 {
        let mut b = Bodies::with_capacity(80);
        spawn_system(&mut b, 80, seed);
        b.refresh_parent_cache();
        for i in 0..b.len() {
            if b.origin[i] != BodyOrigin::Spawned || b.kind[i] != BodyKind::Planet {
                continue;
            }
            if let Some(p) = find_parent(&b, i) {
                if b.kind[p] == BodyKind::Planet && b.m[p] >= 50.0 {
                    total_moons += 1;
                }
            }
        }
    }
    assert!(total_moons > 10, "expected moons, got {}", total_moons);
}

#[test]
fn s_body_count_with_moons_never_exceeds_ceiling() {
    for seed in 0..100u32 {
        let mut b = Bodies::with_capacity(100);
        spawn_system(&mut b, 100, seed);
        assert!(b.len() <= 100, "seed {}: {} bodies", seed, b.len());
    }
}

#[test]
fn s_warm_jupiters_exist() {
    let mut n = 0;
    for seed in 0..500u32 {
        let mut b = Bodies::with_capacity(50);
        spawn_system(&mut b, 50, seed);
        for i in 0..b.len() {
            if b.origin[i] != BodyOrigin::Spawned || b.m[i] < 50.0 || b.m[i] > 318.0 {
                continue;
            }
            let a_au = ((b.x[i] - b.x[0]).powi(2) + (b.y[i] - b.y[0]).powi(2)).sqrt() / au_in_gm();
            if a_au > 0.1 && a_au < 4.0 {
                n += 1;
                break;
            }
        }
    }
    assert!(n > 10, "warm Jupiters found: {}", n);
}

#[test]
fn o_moon_orbit_reference_is_the_planet() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0 * au_in_gm();
    let v_jup = (G * M_SUN_ME / a).sqrt();
    b.push(a, 0.0, 0.0, v_jup, 318.0, BodyKind::Planet);
    let v_moon = v_jup + (G * 318.0 / 0.1).sqrt();
    b.push(a + 0.1, 0.0, 0.0, v_moon, 0.01, BodyKind::Planet);
    b.refresh_parent_cache();
    assert_eq!(find_parent(&b, 2), Some(1));
    let hosts = host_list(&b);
    let r = orbit_reference(&b, &hosts, 2, (0.0, 0.0, 0.0, 0.0, 0.0));
    assert!((r.0 - b.x[1]).abs() < 1e-12);
    assert!((r.1 - b.y[1]).abs() < 1e-12);
    assert!((r.4 - b.m[1]).abs() < 1e-12);
}

#[test]
fn s_spawn_moon_creates_a_bound_moon() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    b.refresh_parent_cache();
    let earth = (0..b.len())
        .find(|&i| b.name[i].as_deref() == Some("Earth"))
        .unwrap_or_else(|| (0..b.len()).find(|&i| (b.m[i] - 1.0).abs() < 1e-9).unwrap());
    let n0 = b.len();
    let mut rng = XorShift::new(7);
    let m = spawn_moon(&mut b, earth, &mut rng).expect("moon should fit");
    assert_eq!(b.len(), n0 + 1);
    b.refresh_parent_cache();
    assert_eq!(find_parent(&b, m), Some(earth));
}

#[test]
fn s_add_ring_creates_small_particles() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let jup = (0..b.len())
        .find(|&i| (b.m[i] - 317.80).abs() < 1e-2)
        .expect("Jupiter");
    let n0 = b.len();
    let mut rng = XorShift::new(11);
    let parts = add_ring(&mut b, jup, 25, &mut rng);
    assert_eq!(parts.len(), 25);
    assert_eq!(b.len(), n0 + 25);
    let total: f64 = parts.iter().map(|&k| b.m[k]).sum();
    assert!(total < 1e-3 * b.m[jup]);
}

#[test]
fn s_spawn_moon_rejects_star() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mut rng = XorShift::new(1);
    assert!(spawn_moon(&mut b, 0, &mut rng).is_none());
}

/// An interstellar object crossing `CULL_DIST` outbound must be
/// removed and logged with `EventKind::Interstellar` (not `Escape`),
/// so the UI can distinguish it from a routine planet ejection.
#[test]
fn p_escape_cull_logs_interstellar_departure() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let d = CULL_DIST * 2.0;
    let v_esc = (2.0 * G * M_SUN_ME / d).sqrt();
    let i = b.push(d, 0.0, 2.0 * v_esc, 0.0, 1e-8, BodyKind::Planet);
    b.origin[i] = BodyOrigin::Interstellar;
    cull_escaped(&mut b);
    assert_eq!(b.len(), 1, "ISO must be culled");
    assert!(
        b.events
            .iter()
            .any(|e| matches!(e.kind, EventKind::Interstellar) && e.text.contains("left")),
        "ISO departure must be logged with Interstellar kind"
    );
    // A regular escapee still logs as Escape.
    let mut b2 = Bodies::with_capacity(2);
    b2.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let j = b2.push(d, 0.0, 2.0 * v_esc, 0.0, 1.0, BodyKind::Planet);
    b2.origin[j] = BodyOrigin::Spawned;
    cull_escaped(&mut b2);
    assert!(
        b2.events
            .iter()
            .any(|e| matches!(e.kind, EventKind::Escape) && e.text.contains("escaped")),
        "regular planet departure must still log with Escape kind"
    );
}

/// A rogue inside the CULL..HARD_CULL band must not be culled. This
/// is the "CULL_DIST exception" the feature adds.
#[test]
fn p_escape_cull_keeps_rogue_inside_hard_cull() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let d = CULL_DIST * 2.0;
    let v_esc = (2.0 * G * M_SUN_ME / d).sqrt();
    let i = b.push(d, 0.0, 2.0 * v_esc, 0.0, 300.0, BodyKind::Planet);
    b.origin[i] = BodyOrigin::RoguePlanet;
    cull_escaped(&mut b);
    assert_eq!(b.len(), 2, "rogue must survive inside HARD_CULL_DIST");
    assert_eq!(b.origin[i], BodyOrigin::RoguePlanet);
}

/// A rogue past HARD_CULL_DIST must be culled, regardless of origin.
#[test]
fn p_escape_cull_removes_rogue_past_hard_cull() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let d = HARD_CULL_DIST * 1.5;
    let v_esc = (2.0 * G * M_SUN_ME / d).sqrt();
    let i = b.push(d, 0.0, 2.0 * v_esc, 0.0, 300.0, BodyKind::Planet);
    b.origin[i] = BodyOrigin::RoguePlanet;
    cull_escaped(&mut b);
    assert_eq!(b.len(), 1, "rogue past HARD_CULL_DIST must be culled");
    assert!(b.events.iter().any(|e| matches!(e.kind, EventKind::Escape)));
}

/// A heavy ejected planet (>= ROGUE_MIN_MASS) crossing CULL_DIST
/// outbound is *promoted* to a rogue, not culled.
#[test]
fn p_escape_cull_promotes_heavy_ejectee() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let d = CULL_DIST * 2.0;
    let v_esc = (2.0 * G * M_SUN_ME / d).sqrt();
    let i = b.push(d, 0.0, 2.0 * v_esc, 0.0, 300.0, BodyKind::Planet);
    b.origin[i] = BodyOrigin::Spawned;
    cull_escaped(&mut b);
    assert_eq!(b.len(), 2, "heavy ejectee must be promoted, not culled");
    assert_eq!(b.origin[i], BodyOrigin::RoguePlanet);
    assert!(
        b.events.iter().any(|e| matches!(e.kind, EventKind::Eject)),
        "promotion must log an Eject event"
    );
}

/// A light ejected body (< ROGUE_MIN_MASS) crossing CULL_DIST is
/// culled — the pre-feature behaviour, unchanged. Regression guard.
#[test]
fn p_escape_cull_culls_light_ejectee() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let d = CULL_DIST * 2.0;
    let v_esc = (2.0 * G * M_SUN_ME / d).sqrt();
    let i = b.push(d, 0.0, 2.0 * v_esc, 0.0, 1.0, BodyKind::Planet);
    b.origin[i] = BodyOrigin::Spawned;
    cull_escaped(&mut b);
    assert_eq!(b.len(), 1, "light ejectee must still be culled");
    assert!(b.events.iter().any(|e| matches!(e.kind, EventKind::Escape)));
}

// ===========================================================================
// q — energy / angular-momentum bookkeeping: unit-level correctness
// ===========================================================================
//
// The existing `q_*` tests only prove that drift stays *small* over
// integration. These tests pin the helpers' contract instead: closed-form
// values, edge cases, None-return paths, idempotency, and the serial-vs-
// parallel equivalence inside `potential_energy`. A future change that
// preserved "small drift" while returning a consistently wrong energy
// would slip past the existing suite but fail every test below.

// ---------------------------------------------------------------------------
// total_energy — closed-form / edge cases
// ---------------------------------------------------------------------------

/// Empty system: no kinetic terms, no pairs → energy is exactly zero.
#[test]
fn q_total_energy_empty_is_zero() {
    let b = Bodies::with_capacity(0);
    assert_eq!(total_energy(&b), Some(0.0));
}

/// Single stationary body: no pairs (n < 2) and v = 0 → energy is zero.
#[test]
fn q_total_energy_single_stationary_is_zero() {
    let mut b = Bodies::with_capacity(1);
    b.push(1.0, 2.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    assert_eq!(total_energy(&b), Some(0.0));
}

/// Single moving body: energy reduces to exactly ½·m·v².
#[test]
fn q_total_energy_single_moving_is_kinetic() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 3.0, 4.0, 2.0, BodyKind::Planet); // v² = 25
    let e = total_energy(&b).unwrap();
    assert!((e - 25.0).abs() < 1e-12, "e = {e}");
}

/// Two stationary bodies: energy is the softened potential — negative,
/// with magnitude G·m₁·m₂/√(r² + SOFT2). Pins both the sign convention
/// and the `+ SOFT2` term.
#[test]
fn q_total_energy_two_body_at_rest_is_negative_pe() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(10.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let e = total_energy(&b).unwrap();
    let expected = -G / (100.0 + SOFT2).sqrt();
    assert!(
        (e - expected).abs() < 1e-12 * expected.abs(),
        "e = {e}, expected {expected}"
    );
    assert!(e < 0.0);
}

/// Coincident bodies: the `SOFT2` term keeps the 1/r² PE from blowing up.
/// The result must be finite and match -G·m₁·m₂/√SOFT2.
#[test]
fn q_total_energy_coincident_finite() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let e = total_energy(&b).unwrap();
    let expected = -G / SOFT2.sqrt();
    assert!(e.is_finite());
    assert!((e - expected).abs() < 1e-12 * expected.abs().max(1e-30));
}

/// Virial theorem on a circular two-body orbit at r ≫ √SOFT2:
/// PE ≈ -2·KE and E ≈ -KE. `a = 100` makes the softening correction
/// well below the tolerance regardless of SOFT2's exact magnitude.
#[test]
fn q_total_energy_circular_orbit_virial() {
    let (m1, m2) = (M_SUN_ME, 1.0);
    let mt = m1 + m2;
    let mu = m1 * m2 / mt;
    let a = 100.0;
    let v_rel = (G * mt / a).sqrt();
    let mut b = Bodies::with_capacity(2);
    b.push(-m2 / mt * a, 0.0, 0.0, -m2 / mt * v_rel, m1, BodyKind::Star);
    b.push(m1 / mt * a, 0.0, 0.0, m1 / mt * v_rel, m2, BodyKind::Planet);
    let e = total_energy(&b).unwrap();
    let ke = 0.5 * mu * v_rel * v_rel;
    assert!(e < 0.0, "circular orbit is bound: {e}");
    // E ≈ -KE (virial), with tolerance absorbing SOFT2 / a².
    let neg_ke = -ke;
    assert!((e + ke).abs() < 1e-3 * ke, "E {e} vs -KE {neg_ke} (virial)");
}

/// Above `ENERGY_MAX_N`, `total_energy` returns None directly — not only
/// via `energy_drift_pct`'s early return.
#[test]
fn q_total_energy_above_limit_none_direct() {
    let mut b = Bodies::with_capacity(ENERGY_MAX_N + 1);
    for k in 0..=ENERGY_MAX_N {
        b.push(k as f64, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    }
    assert_eq!(total_energy(&b), None);
}

/// The optimised paired-row `potential_energy` must agree with a naïve
/// inline O(n²) reference at every size, including the PE_PAR_THRESHOLD
/// boundary where the code flips serial ↔ parallel. Any missed pair, a
/// wrong `mid` split, or an off-by-one in the pairing surfaces here.
#[test]
fn q_total_energy_matches_reference() {
    let reference = |b: &Bodies| -> f64 {
        let n = b.len();
        let mut e = 0.0;
        for i in 0..n {
            let v2 = b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i];
            e += 0.5 * b.m[i] * v2;
            for j in (i + 1)..n {
                let dx = b.x[j] - b.x[i];
                let dy = b.y[j] - b.y[i];
                e -= G * b.m[i] * b.m[j] / (dx * dx + dy * dy + SOFT2).sqrt();
            }
        }
        e
    };
    // 127 = below threshold (serial), 128 = at threshold, 129 = above
    // (parallel), plus a couple of larger sizes to stress the pairing.
    for &n in &[2usize, 10, 127, 128, 129, 200] {
        let mut b = Bodies::with_capacity(n);
        let mut rng = XorShift::new(n as u32);
        for _ in 0..n {
            b.push(
                100.0 * (rng.next_f64() - 0.5),
                100.0 * (rng.next_f64() - 0.5),
                10.0 * (rng.next_f64() - 0.5),
                10.0 * (rng.next_f64() - 0.5),
                0.1 + 10.0 * rng.next_f64(),
                BodyKind::Planet,
            );
        }
        let fast = total_energy(&b).unwrap();
        let slow = reference(&b);
        let rel = (fast - slow).abs() / slow.abs().max(1e-30);
        assert!(rel < 1e-10, "n = {n}: {fast} vs {slow}, rel {rel}");
    }
}

// ---------------------------------------------------------------------------
// total_angular_momentum — closed-form / edge cases
// ---------------------------------------------------------------------------

/// Empty system: no bodies → L = 0.
#[test]
fn q_l_empty_is_zero() {
    let b = Bodies::with_capacity(0);
    assert_eq!(total_angular_momentum(&b), 0.0);
}

/// Single body: L about its own CoM is zero regardless of velocity.
/// Without spin the function must return exactly zero for every
/// direction of motion, since the "orbit" is degenerate.
#[test]
fn q_l_single_body_is_zero_without_spin() {
    for (vx, vy) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (3.0, -4.0)] {
        let mut b = Bodies::with_capacity(1);
        b.push(5.0, -3.0, vx, vy, 1.0, BodyKind::Planet);
        assert_eq!(total_angular_momentum(&b), 0.0, "v = ({vx}, {vy})");
    }
}

/// Single body with spin: the orbital term is still zero, so L = spin.
#[test]
fn q_l_single_body_returns_spin() {
    let mut b = Bodies::with_capacity(1);
    b.push(1.0, 2.0, 3.0, 4.0, 1.0, BodyKind::Planet);
    b.spin[0] = 0.75;
    assert!((total_angular_momentum(&b) - 0.75).abs() < 1e-15);
}

/// Two bodies separating purely radially: velocity is parallel to the
/// separation vector, so r × v = 0 per body. L must vanish.
#[test]
fn q_l_two_body_radial_is_zero() {
    let mut b = Bodies::with_capacity(2);
    b.push(-1.0, 0.0, -1.0, 0.0, 1.0, BodyKind::Planet);
    b.push(1.0, 0.0, 1.0, 0.0, 1.0, BodyKind::Planet);
    assert!(total_angular_momentum(&b).abs() < 1e-15);
}

/// Circular two-body orbit: closed-form L = μ·√(G·M·a) with μ the
/// reduced mass. Sign must flip when the orbit reverses (prograde ↔
/// retrograde), which the second half verifies.
#[test]
fn q_l_two_body_circular_closed_form() {
    let (m1, m2) = (M_SUN_ME, 1.0);
    let mt = m1 + m2;
    let mu = m1 * m2 / mt;
    let a = 5.0;
    let v_rel = (G * mt / a).sqrt();
    let mut b = Bodies::with_capacity(2);
    b.push(-m2 / mt * a, 0.0, 0.0, -m2 / mt * v_rel, m1, BodyKind::Star);
    b.push(m1 / mt * a, 0.0, 0.0, m1 / mt * v_rel, m2, BodyKind::Planet);
    let expected = mu * (G * mt * a).sqrt();
    let got = total_angular_momentum(&b);
    assert!(
        (got / expected - 1.0).abs() < 1e-9,
        "prograde L = {got} vs {expected}"
    );
    assert!(got > 0.0, "prograde must be positive");
    // Retrograde: flip every velocity. L flips sign.
    for i in 0..b.len() {
        b.vy[i] = -b.vy[i];
    }
    let got_retro = total_angular_momentum(&b);
    assert!(
        (got_retro + expected).abs() / expected < 1e-9,
        "retrograde L = {got_retro} vs -{expected}"
    );
    assert!(got_retro < 0.0);
}

/// Translation invariance: L about the CoM is invariant under a global
/// coordinate shift. The two-pass subtraction form of
/// `total_angular_momentum` is what makes this hold to fp precision;
/// the algebraically-equivalent one-pass form Σm·r×v − M·R×V would
/// diverge as the offset grows.
#[test]
fn q_l_translation_invariance() {
    let make = |cx: f64, cy: f64| -> Bodies {
        let (m1, m2) = (10.0, 1.0);
        let mt = m1 + m2;
        let a = 5.0;
        let v_rel = (G * mt / a).sqrt();
        let mut b = Bodies::with_capacity(2);
        b.push(
            cx - m2 / mt * a,
            cy,
            0.0,
            -m2 / mt * v_rel,
            m1,
            BodyKind::Planet,
        );
        b.push(
            cx + m1 / mt * a,
            cy,
            0.0,
            m1 / mt * v_rel,
            m2,
            BodyKind::Planet,
        );
        b
    };
    let l0 = total_angular_momentum(&make(0.0, 0.0));
    let l1 = total_angular_momentum(&make(1000.0, 500.0));
    assert!(
        (l0 - l1).abs() < 1e-9 * (1.0 + l0.abs()),
        "L shifted: {l0} -> {l1}"
    );
}

/// Spin adds linearly to the orbital L. Setting each spin to 0.5 must
/// raise the total by exactly 1.0.
#[test]
fn q_l_spin_adds() {
    let mut b = Bodies::with_capacity(2);
    b.push(-1.0, 0.0, 0.0, 1.0, 1.0, BodyKind::Planet);
    b.push(1.0, 0.0, 0.0, -1.0, 1.0, BodyKind::Planet);
    let l_orb = total_angular_momentum(&b);
    b.spin[0] = 0.5;
    b.spin[1] = 0.5;
    let l_total = total_angular_momentum(&b);
    assert!((l_total - (l_orb + 1.0)).abs() < 1e-15);
}

// ---------------------------------------------------------------------------
// mark_energy / settle_energy / rebase_energy — contracts
// ---------------------------------------------------------------------------

/// `mark_energy` populates `e_pre` with the current energy.
#[test]
fn q_mark_energy_stores_e_pre() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(5.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    assert!(b.e_pre.is_none());
    mark_energy(&mut b);
    assert!(b.e_pre.is_some());
    assert_eq!(b.e_pre, total_energy(&b));
}

/// `mark_energy` is idempotent — a retry mid-step must not overwrite
/// the original snapshot, or drift would be under-counted.
#[test]
fn q_mark_energy_idempotent() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(5.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    mark_energy(&mut b);
    let first = b.e_pre;
    b.vx[0] = 1.0; // mutate so a second snapshot would differ
    mark_energy(&mut b);
    assert_eq!(b.e_pre, first, "second mark must be a no-op");
}

/// `settle_energy` without a prior mark is a no-op — e_pre stays None,
/// offset stays put.
#[test]
fn q_settle_energy_noop_without_mark() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(5.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let offset_before = b.stats.energy_offset;
    settle_energy(&mut b);
    assert_eq!(b.stats.energy_offset, offset_before);
    assert!(b.e_pre.is_none());
}

/// Two mark/settle cycles accumulate their deltas into `energy_offset`.
#[test]
fn q_settle_energy_accumulates() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(5.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let mut total_expected = 0.0;
    for k in 0..3 {
        let e_before = total_energy(&b).unwrap();
        mark_energy(&mut b);
        b.vx[0] += 0.1;
        let e_after = total_energy(&b).unwrap();
        settle_energy(&mut b);
        total_expected += e_after - e_before;
        assert!(
            (b.stats.energy_offset - total_expected).abs() < 1e-10 * (1.0 + total_expected.abs()),
            "k = {k}: offset {} vs {total_expected}",
            b.stats.energy_offset
        );
    }
}

/// `rebase_energy` resets all three state variables together.
#[test]
fn q_rebase_energy_resets_state() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(5.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.stats.energy_offset = 42.0;
    b.e_pre = Some(-100.0);
    rebase_energy(&mut b);
    assert_eq!(b.stats.energy0, total_energy(&b).unwrap());
    assert_eq!(b.stats.energy_offset, 0.0);
    assert!(b.e_pre.is_none());
}

/// Above `ENERGY_MAX_N`, `rebase_energy` still succeeds — `energy0`
/// becomes 0 via `unwrap_or(0.0)`. Downstream drift helpers must then
/// return None without panicking.
#[test]
fn q_rebase_energy_above_limit_sets_zero() {
    let mut b = Bodies::with_capacity(ENERGY_MAX_N + 5);
    for k in 0..=ENERGY_MAX_N + 4 {
        b.push(k as f64, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    }
    rebase_energy(&mut b);
    assert_eq!(b.stats.energy0, 0.0);
    assert_eq!(b.stats.energy_offset, 0.0);
    assert!(b.e_pre.is_none());
}

/// Immediately after rebase, drift is exactly zero (well, ε).
#[test]
fn q_rebase_then_drift_is_zero() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let v = (G * M_SUN_ME / 5.0).sqrt();
    b.push(5.0, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    rebase_energy(&mut b);
    let d = energy_drift_pct(&b).unwrap();
    assert!(d.abs() < 1e-15, "drift right after rebase: {d}");
}

// ---------------------------------------------------------------------------
// energy_drift_pct / energy_dissipated_pct — None paths and formula
// ---------------------------------------------------------------------------

/// `e0 = 0` → `energy_drift_pct` returns None (undefined ratio). A
/// single stationary body is the smallest such system.
#[test]
fn q_drift_zero_e0_none() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    rebase_energy(&mut b);
    assert_eq!(b.stats.energy0, 0.0);
    assert_eq!(energy_drift_pct(&b), None);
}

/// Bound system (e0 < 0): the `|e0|` denominator preserves the sign of
/// the drift as a *change*, not the sign of energy itself.
#[test]
fn q_drift_uses_abs_e0_for_bound_system() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let r = 5.0;
    let v = (G * M_SUN_ME / r).sqrt();
    b.push(r, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    rebase_energy(&mut b);
    let e0 = b.stats.energy0;
    assert!(e0 < 0.0, "circular orbit is bound: e0 = {e0}");
    // Add energy: bump the tangential speed. E rises (less negative).
    b.vx[1] += 0.005 * v;
    let e1 = total_energy(&b).unwrap();
    let d = energy_drift_pct(&b).unwrap();
    let expected = (e1 - e0) / e0.abs() * 100.0;
    assert!((d - expected).abs() < 1e-12, "formula: {d} vs {expected}");
    assert!(d > 0.0, "adding energy must give positive drift: {d}");
}

/// `energy_dissipated_pct` with zero offset is exactly 0.0, not None.
#[test]
fn q_dissipated_zero_offset_is_zero() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let v = (G * M_SUN_ME / 5.0).sqrt();
    b.push(5.0, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    rebase_energy(&mut b);
    assert_eq!(energy_dissipated_pct(&b), Some(0.0));
}

/// `energy_dissipated_pct` with `energy0 = 0` (single body) returns None,
/// **independent** of the above-limit path — a distinct None branch
/// from the one tested in `q_energy_drift_returns_none_above_limit`.
#[test]
fn q_dissipated_zero_e0_none() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    rebase_energy(&mut b);
    assert_eq!(b.stats.energy0, 0.0);
    assert_eq!(energy_dissipated_pct(&b), None);
}

/// End-to-end purpose of `energy_offset`: after an inelastic collision,
/// raw energy changes substantially, but the **offset-adjusted** drift
/// stays near zero. If `settle_energy` stopped being called after
/// collisions, this test would fail even though the existing
/// drift-smallness tests pass.
#[test]
fn q_offset_excludes_collision_from_drift() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(0.001, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet); // overlapping, at rest
    rebase_energy(&mut b);
    let e_raw_before = total_energy(&b).unwrap();
    assert!(e_raw_before < 0.0, "overlapping pair is bound");
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1, "inelastic merge happened");
    // Raw drift would be large (energy lost to the merge).
    let e_raw_after = total_energy(&b).unwrap();
    let raw_drift = (e_raw_after - e_raw_before) / e_raw_before.abs() * 100.0;
    assert!(raw_drift.abs() > 1.0, "raw drift after merge: {raw_drift}%");
    // But offset-accounted drift must be ~0.
    let drift = energy_drift_pct(&b).unwrap();
    assert!(
        drift.abs() < 1e-6,
        "offset-adjusted drift {drift}% should exclude the merge"
    );
    // And the offset itself carries the (negative) collision delta.
    assert!(b.stats.energy_offset != 0.0);
}
