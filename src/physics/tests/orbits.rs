//! Tests for orbit/host/binary physics and user-edit operations.
//!
//! Test-name prefixes (k_/o_/p_/q_/r_/s_) are stable: `cargo test --lib
//! physics::tests::orbits::k_` filters by category. Do not rename.
//!
//!   1.  Binary systems and circumbinary stability      (k_*)
//!   2.  Parent and host resolution                     (k_*)
//!   3.  Orbit info                                     (o_*)
//!   4.  Circularize                                    (o_*)
//!   5.  Host distance                                  (o_*)
//!   6.  Escape culling                                 (p_*)
//!   7.  Energy / angular-momentum diagnostics          (q_*)
//!       7a. total_energy
//!       7b. total_angular_momentum
//!       7c. mark / settle / rebase
//!       7d. drift / dissipated
//!   8.  Kind transitions via collision                 (r_*)
//!   9.  User edits                                     (s_*)
//!       9a. scale_mass
//!       9b. set_kind
//!       9c. delete_body
//!   10. System generation                              (s_*)

use super::super::*;
use super::shared::*;

// ===========================================================================
// 1. Binary systems and circumbinary stability
// ===========================================================================

/// Holman–Wiegert limits match published values for an equal-mass circular
/// binary, and move the expected direction (S shrinks, P grows) with
/// rising companion mass.
#[test]
fn k_hw_formulas_match_published_values() {
    let s = hw_critical_s(1.0, 0.0, 0.5);
    let p = hw_critical_p(1.0, 0.0, 0.5);
    assert!((s - 0.274).abs() < 0.01, "S {}", s);
    assert!((p - 2.39).abs() < 0.05, "P {}", p);
    assert!(hw_critical_s(1.0, 0.5, 0.5) < s);
    assert!(hw_critical_p(1.0, 0.5, 0.5) > p);
}

/// 600 seeded systems: every spawned binary respects HW stability limits
/// (period, semi-major, periapsis clearance, circumbinary exclusion),
/// and the population shows a realistic MS binary fraction with both
/// S- and P-type architectures.
///
/// Two intentional skips: moons orbit a planet (their `osc_a` about the
/// star is not a stability quantity), and third stars orbit the whole
/// inner system (not just the binary barycentre). Both would produce
/// false negatives if not excluded.
#[test]
fn k_binary_ic_respects_stability_limits() {
    let au = au_in_gm();
    let (mut stars, mut bins, mut n_s, mut n_p) = (0, 0, 0, 0);
    for seed in 0..600u128 {
        let mut b = Bodies::with_capacity(60);
        let info = spawn_system_with_info(&mut b, 60, seed);
        assert!(b.len() <= 60, "seed {}: {} bodies", seed, b.len());

        // Only MS-star primaries count toward the binary fraction.
        if b.kind[0] == BodyKind::BrownDwarf {
            continue;
        }
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

        // `osc_a3` about the pair recovers the requested semi-major axis.
        // 3D on purpose: the spawned system is tilted, and a 2D vis-viva
        // would report a different `a` for every inclined orbit.
        let bary = star_pair_centre3(&b);
        // The relative orbit of the companion about the primary, with the
        // pair's total mass as the central mass.
        let mut rel = body_ref3(&b, 0);
        rel.6 = b.m[0] + b.m[1];
        let a_rel = osc_a3(&b, 1, rel) / au;
        // 0.1 %, not fp-exact: the companion is placed by a Kepler solve about
        // the primary, body by body, so the recovered osculating `a` matches
        // the requested value to ~1e-4 rather than to the last bit.
        assert!(
            (a_rel / bi.a_au - 1.0).abs() < 1e-3,
            "seed {} {} vs {}",
            seed,
            a_rel,
            bi.a_au
        );

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
                    if b.origin[i] != BodyOrigin::Spawned
                        || !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
                        || is_moon(&b, i)
                    {
                        continue;
                    }
                    let a = osc_a3(&b, i, bary);
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
                let c = (
                    b.x[h],
                    b.y[h],
                    b.z[h],
                    b.vx[h],
                    b.vy[h],
                    b.vz[h],
                    b.m[h],
                );
                let lim = HW_MARGIN_S * bi.a_crit_au * au;
                for i in 2..b.len() {
                    if b.origin[i] != BodyOrigin::Spawned
                        || !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
                        || is_moon(&b, i)
                    {
                        continue;
                    }
                    let a = osc_a3(&b, i, c);
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

/// A circumbinary body reports the heavier star as its `find_parent` (so
/// every body in the disk agrees on a host), but its `orbit_reference`
/// mass is the pair total — the quantity that actually governs the orbit.
/// S-type bodies use their own star for both.
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
    let c = orbit_reference(&b, &host_list(&b), circ, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    assert!(
        (c.6 - mt).abs() < 1e-9,
        "circumbinary orbit is drawn about the pair"
    );
    let s = orbit_reference(&b, &host_list(&b), near2, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    assert!(
        (s.6 - m2).abs() < 1e-9,
        "S-type orbit is drawn about its own star"
    );
    assert!(!compute_orbit_segments(&b, 1.0, true, true).is_empty());
}

/// A circumbinary planet placed just outside the HW P-type limit stays
/// bound and roughly circular over 4000 steps: no merges, no escapes,
/// energy drift < 0.5 %, semi-major axis within ±25 %.
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

/// A lone host has no bound companion.
#[test]
fn k_binary_partner_single_host_is_none() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    assert_eq!(binary_partner(&b, 0), None);
}

/// A pair separating at 4× escape speed is rejected from both sides —
/// `binary_partner_in` is symmetric.
#[test]
fn k_binary_partner_unbound_pair_is_none() {
    let m = M_SUN_ME;
    let d = 5.0 * au_in_gm();
    let v_esc = (2.0 * G * 2.0 * m / d).sqrt();
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, -2.0 * v_esc, 0.0, m, BodyKind::Star);
    b.push(d, 0.0, 2.0 * v_esc, 0.0, m, BodyKind::Star);
    assert_eq!(binary_partner(&b, 0), None);
    assert_eq!(binary_partner(&b, 1), None);
}

// ===========================================================================
// 2. Parent and host resolution
// ===========================================================================

#[test]
fn k_primary_index_empty_is_none() {
    let b = Bodies::with_capacity(0);
    assert_eq!(primary_index(&b), None);
}

#[test]
fn k_primary_index_single_host() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    assert_eq!(primary_index(&b), Some(0));
}

#[test]
fn k_primary_index_picks_heaviest_host() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, 0.5 * M_SUN_ME, BodyKind::Star);
    b.push(10.0, 0.0, 0.0, 0.0, 2.0 * M_SUN_ME, BodyKind::Star);
    b.push(20.0, 0.0, 0.0, 0.0, 1.0 * M_SUN_ME, BodyKind::Star);
    assert_eq!(primary_index(&b), Some(1));
}

/// Tie-break uses strict `>`: equal masses keep the lower index.
#[test]
fn k_primary_index_tie_keeps_lowest_index() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, 2.0 * M_SUN_ME, BodyKind::Star);
    b.push(10.0, 0.0, 0.0, 0.0, 2.0 * M_SUN_ME, BodyKind::Star);
    b.push(20.0, 0.0, 0.0, 0.0, 1.0 * M_SUN_ME, BodyKind::Star);
    assert_eq!(primary_index(&b), Some(0));
}

/// Planets — however massive — are never primaries.
#[test]
fn k_primary_index_skips_non_hosts() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0 * M_SUN_ME, BodyKind::Star);
    b.push(10.0, 0.0, 0.0, 0.0, 100.0 * M_SUN_ME, BodyKind::Planet);
    assert_eq!(primary_index(&b), Some(0), "planets must not be primaries");
}

#[test]
fn k_primary_index_no_hosts_is_none() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(10.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    assert_eq!(primary_index(&b), None);
}

#[test]
fn k_host_list_filters_and_preserves_order() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(10.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(20.0, 0.0, 0.0, 0.0, 0.5 * M_SUN_ME, BodyKind::Star);
    b.push(30.0, 0.0, 0.0, 0.0, 0.01, BodyKind::Planet);
    assert_eq!(host_list(&b), vec![0, 2]);
}

#[test]
fn k_host_list_empty() {
    let b = Bodies::with_capacity(0);
    assert!(host_list(&b).is_empty());
}

/// Only `Spawned` planets get a parent. Interstellar objects and rogue
/// planets are orphans by design — reporting the primary would make the
/// UI draw them as orbiting the star.
#[test]
fn k_find_parent_origin_exclusions() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let i = b.push(5.0 * au_in_gm(), 0.0, 0.0, 1.0, 1.0, BodyKind::Planet);

    assert_eq!(find_parent(&b, i), Some(0));

    for origin in [BodyOrigin::Interstellar, BodyOrigin::RoguePlanet] {
        b.origin[i] = origin;
        assert_eq!(find_parent(&b, i), None, "origin {:?}", origin);
    }
}

/// Non-planet kinds have no parent.
#[test]
fn k_find_parent_non_planet_kinds_none() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(10.0, 0.0, 0.0, 0.0, 0.5 * M_SUN_ME, BodyKind::BrownDwarf);
    b.push(20.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Star);
    assert_eq!(find_parent(&b, 0), None, "star has no parent");
    assert_eq!(find_parent(&b, 1), None, "brown dwarf has no parent");
    assert_eq!(find_parent(&b, 2), None, "star has no parent");
}

/// No hosts → host scan stays None and the moon scan is skipped.
#[test]
fn k_find_parent_no_hosts_none() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(5.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    assert_eq!(find_parent(&b, 0), None);
    assert_eq!(find_parent(&b, 1), None);
}

/// `refresh_parent_cache` must be bit-identical to `find_parent` across
/// many seeds — any divergence in candidate set, iteration order, or
/// circumbinary lead selection surfaces here.
#[test]
fn k_parent_cache_matches_find_parent() {
    for seed in 0..100u128 {
        let mut b = Bodies::with_capacity(80);
        spawn_system(&mut b, 80, seed);
        b.refresh_parent_cache();
        assert_eq!(b.parent_cache.len(), b.len(), "seed {}", seed);
        for i in 0..b.len() {
            assert_eq!(
                b.parent_cache[i],
                find_parent(&b, i),
                "seed {} body {}: cached vs uncached parent differ",
                seed,
                i
            );
        }
    }
}

/// Solo star: `circumbinary_frame` returns None → effective mass = m[p].
#[test]
fn k_effective_host_mass_single_star() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(5.0 * au_in_gm(), 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let hosts = host_list(&b);
    let m = effective_host_mass(&b, &hosts, 1, 0);
    assert!((m - M_SUN_ME).abs() < 1e-12);
}

/// For a circumbinary planet the Hill sphere is governed by m1 + m2, not
/// by the lead star's mass alone.
#[test]
fn k_effective_host_mass_circumbinary_is_pair_total() {
    let (m1, m2) = (M_SUN_ME, 0.5 * M_SUN_ME);
    let mt = m1 + m2;
    let sep = 0.1 * au_in_gm();
    let mut b = Bodies::with_capacity(3);
    b.push(-m2 / mt * sep, 0.0, 0.0, 0.0, m1, BodyKind::Star);
    b.push(m1 / mt * sep, 0.0, 0.0, 0.0, m2, BodyKind::Star);
    b.push(100.0 * sep, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let hosts = host_list(&b);
    let m = effective_host_mass(&b, &hosts, 2, 0);
    assert!(
        (m - mt).abs() < 1e-9,
        "circumbinary effective mass should be pair total: {} vs {}",
        m,
        mt
    );
}

/// Stale parent cache → `orbit_reference` returns the caller's default
/// (normally the primary). Prevents drawing orbits around a stale parent.
#[test]
fn k_orbit_reference_stale_cache_returns_default() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(5.0 * au_in_gm(), 0.0, 0.0, 1.0, 1.0, BodyKind::Planet);
    b.parent_cache_valid = false;
    let default = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
    assert_eq!(orbit_reference(&b, &host_list(&b), 1, default), default);
}

/// Orphan body (interstellar here) → default reference.
#[test]
fn k_orbit_reference_no_parent_returns_default() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let i = b.push(5.0 * au_in_gm(), 0.0, 0.0, 1.0, 1.0, BodyKind::Planet);
    b.origin[i] = BodyOrigin::Interstellar;
    b.refresh_parent_cache();
    let default = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
    assert_eq!(orbit_reference(&b, &host_list(&b), i, default), default);
}

// ===========================================================================
// 3. Orbit info
// ===========================================================================

/// `orbit_info` recovers (a, e, r, v, period) exactly for a circular orbit.
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

/// `orbit_info` recovers (a, e, peri, apo) from a `kepler_state` ellipse.
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

/// Above escape speed: unbound, `semi_major = period = +∞`, `ecc > 1`.
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

/// A moon's `orbit_reference` returns its parent planet's state, not the
/// star's — the wrong reference would collapse the moon onto the star.
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
    let r = orbit_reference(&b, &hosts, 2, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    assert!((r.0 - b.x[1]).abs() < 1e-12);
    assert!((r.1 - b.y[1]).abs() < 1e-12);
    assert!((r.6 - b.m[1]).abs() < 1e-12);
}

// ===========================================================================
// 4. Circularize
// ===========================================================================

/// `circularize` zeroes eccentricity at fixed orbital radius.
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

/// No parent → false, no mutation, no log, no commit. The
/// `energy_offset` sentinel proves `user_modified` never ran.
#[test]
fn o_circularize_no_parent_returns_false() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.parent_cache_valid = true;
    b.stats.energy_offset = 42.0;
    let (vx0, vy0) = (b.vx[0], b.vy[0]);
    let ev0 = b.events.len();

    assert!(!circularize(&mut b, 0));

    assert_eq!((b.vx[0], b.vy[0]), (vx0, vy0));
    assert_eq!(b.events.len(), ev0);
    assert_eq!(b.stats.energy_offset, 42.0, "user_modified must not run");
}

/// Coincident with parent (`r < 1e-9`) → false, no side effects. A
/// distinct early-return path from the no-parent case.
#[test]
fn o_circularize_coincident_returns_false() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    // Force cached parent so we hit the r-guard, not the no-parent return.
    b.parent_cache = vec![None, Some(0)];
    b.parent_cache_valid = true;
    b.stats.energy_offset = 42.0;
    let (vx0, vy0) = (b.vx[1], b.vy[1]);
    let ev0 = b.events.len();

    assert!(!circularize(&mut b, 1));

    assert_eq!((b.vx[1], b.vy[1]), (vx0, vy0));
    assert_eq!(b.events.len(), ev0);
    assert_eq!(b.stats.energy_offset, 42.0);
}

/// Prograde stays prograde. Paired with the retrograde test to catch any
/// sign-handling regression (a hard-coded `+1` would pass this one).
#[test]
fn o_circularize_preserves_prograde_sign() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(5.0, 0.0, 0.0, 1.0, 1.0, BodyKind::Planet);
    b.refresh_parent_cache();

    assert!(circularize(&mut b, 1));

    assert!(b.vy[1] > 0.0, "prograde vy = {}", b.vy[1]);
    let l = b.x[1] * b.vy[1] - b.y[1] * b.vx[1];
    assert!(l > 0.0, "L should be positive, got {}", l);
}

/// Retrograde stays retrograde.
#[test]
fn o_circularize_preserves_retrograde_sign() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(5.0, 0.0, 0.0, -1.0, 1.0, BodyKind::Planet);
    b.refresh_parent_cache();

    assert!(circularize(&mut b, 1));

    assert!(b.vy[1] < 0.0, "retrograde vy = {}", b.vy[1]);
    let l = b.x[1] * b.vy[1] - b.y[1] * b.vx[1];
    assert!(l < 0.0, "L should be negative, got {}", l);
}

/// `vc = sqrt(G·(m_p + m_i)/r)`, not `sqrt(G·m_p/r)`. The equal-mass
/// child makes the two formulas differ by √2; using only parent mass
/// would leave e ≈ 0.5 after circularize.
#[test]
fn o_circularize_uses_total_mass() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let vc_true = (G * 2.0 * M_SUN_ME / 5.0).sqrt();
    b.push(5.0, 0.0, 0.0, 0.5 * vc_true, M_SUN_ME, BodyKind::Planet);
    b.refresh_parent_cache();
    assert_eq!(find_parent(&b, 1), Some(0), "setup: parent not found");

    assert!(circularize(&mut b, 1));

    assert!(
        (b.vy[1] - vc_true).abs() < 1e-12,
        "vy {} vs vc {}",
        b.vy[1],
        vc_true
    );
    assert_eq!(b.vx[1], 0.0);
    let o = orbit_info(&b, 1, 0);
    assert!(o.ecc < 1e-10, "ecc {} (should be 0 with total mass)", o.ecc);
}

/// Success path runs `user_modified`: cache cleared, energy rebased.
#[test]
fn o_circularize_commits_on_success() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(5.0, 0.0, 0.0, 1.0, 1.0, BodyKind::Planet);
    b.refresh_parent_cache();
    b.parent_cache_valid = true;
    b.stats.energy_offset = 42.0;
    b.e_pre = Some(999.0);

    assert!(circularize(&mut b, 1));

    assert!(!b.parent_cache_valid);
    assert_eq!(b.stats.energy_offset, 0.0);
    assert!(b.e_pre.is_none());
}

#[test]
fn o_circularize_logs_event() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(5.0, 0.0, 0.0, 1.0, 1.0, BodyKind::Planet);
    b.refresh_parent_cache();
    let ev0 = b.events.len();

    assert!(circularize(&mut b, 1));

    assert_eq!(b.events.len(), ev0 + 1);
    let e = b.events.back().unwrap();
    assert_eq!(e.kind, EventKind::User);
    assert!(e.text.contains("circularized"), "text = {:?}", e.text);
}

/// Failure path must *not* commit. Distinct code path from the
/// no-parent return; both are needed because removing either
/// early-return would still pass the other test.
#[test]
fn o_circularize_no_commit_on_failure() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.parent_cache = vec![None, Some(0)];
    b.parent_cache_valid = true;
    b.stats.energy_offset = 42.0;
    b.e_pre = Some(999.0);

    assert!(!circularize(&mut b, 1));

    assert_eq!(b.stats.energy_offset, 42.0, "must not rebase");
    assert_eq!(b.e_pre, Some(999.0), "must not mark_energy");
}

// ===========================================================================
// 5. Host distance
// ===========================================================================

/// A moon's `host_distance` is the distance to the host *star*, not to
/// the parent planet — the value the renderer uses for the planet colour
/// gradient. Getting this wrong collapses a Jupiter moon onto Jupiter.
#[test]
fn o_host_distance_from_moon() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0 * au_in_gm();
    let v_jup = (G * M_SUN_ME / a).sqrt();
    b.push(a, 0.0, 0.0, v_jup, 318.0, BodyKind::Planet);
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

/// Regular planet: distance to its host star, no moon-hop.
#[test]
fn o_host_distance_regular_planet() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let r = 5.0 * au_in_gm();
    b.push(r, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.refresh_parent_cache();
    let d = host_distance(&b, 1);
    assert!((d - r).abs() < 1e-9, "d = {} vs {}", d, r);
}

/// No host (belt/rubble bodies): fall back to |x|. Returning 0 or NaN
/// would collapse the UI colour gradient.
#[test]
fn o_host_distance_no_parent_falls_back_to_origin() {
    let mut b = Bodies::with_capacity(1);
    b.push(3.0, 4.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.refresh_parent_cache();
    assert!((host_distance(&b, 0) - 5.0).abs() < 1e-9);
}

/// Stale cache → same fallback as no-parent.
#[test]
fn o_host_distance_stale_cache_falls_back_to_origin() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(3.0, 4.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.parent_cache_valid = false;
    assert!((host_distance(&b, 1) - 5.0).abs() < 1e-9);
}

// ===========================================================================
// 6. Escape culling
// ===========================================================================

/// Outbound + beyond CULL_DIST + above escape → culled and logged.
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

/// Inbound at the same speed and distance is kept — it may still return.
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

/// Past HARD_CULL_DIST: removed regardless of velocity.
#[test]
fn p_escape_cull_hard_removes_regardless() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let d = HARD_CULL_DIST * 1.5;
    b.push(d, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    cull_escaped(&mut b);
    assert_eq!(b.len(), 1);
}

/// ISO departure logs `EventKind::Interstellar`; a routine ejection
/// still logs `EventKind::Escape`. The UI distinguishes the two.
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

/// Rogues inside HARD_CULL_DIST are protected regardless of speed.
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

/// Past HARD_CULL_DIST, rogues are culled like anything else.
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

/// Heavy ejectee (>= ROGUE_MIN_MASS) is promoted to rogue, not culled,
/// so the UI can show it wandering forever.
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

/// Light ejectee (< ROGUE_MIN_MASS) is culled — pre-feature behaviour,
/// regression guard.
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
// 7. Energy / angular-momentum diagnostics
// ===========================================================================

// ---------------------------------------------------------------------------
// 7a. total_energy
// ---------------------------------------------------------------------------

#[test]
fn q_total_energy_empty_is_zero() {
    let b = Bodies::with_capacity(0);
    assert_eq!(total_energy(&b), Some(0.0));
}

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

/// Two stationary bodies: softened PE = −G·m1·m2/√(r² + SOFT2). Pins
/// both the sign convention and the `+ SOFT2` term.
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

/// Coincident bodies: SOFT2 keeps PE finite.
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
/// PE ≈ −2·KE, E ≈ −KE.
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
    let neg_ke = -ke;
    assert!((e + ke).abs() < 1e-3 * ke, "E {e} vs -KE {neg_ke} (virial)");
}

/// Above ENERGY_MAX_N, `total_energy` returns None directly (not only
/// via `energy_drift_pct`'s early return).
#[test]
fn q_total_energy_above_limit_none_direct() {
    let mut b = Bodies::with_capacity(ENERGY_MAX_N + 1);
    for k in 0..=ENERGY_MAX_N {
        b.push(k as f64, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    }
    assert_eq!(total_energy(&b), None);
}

/// The optimised paired-row `potential_energy` must agree with a naïve
/// O(n²) reference at every size, including the PE_PAR_THRESHOLD
/// boundary where the code flips serial ↔ parallel. Catches a missed
/// pair, a bad mid split, or an off-by-one in the pairing.
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
    // 127 = below threshold (serial), 128 = threshold, 129 = parallel.
    for &n in &[2usize, 10, 127, 128, 129, 200] {
        let mut b = Bodies::with_capacity(n);
        let mut rng = XorShift::new(n as u128);
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
// 7b. total_angular_momentum
// ---------------------------------------------------------------------------

#[test]
fn q_l_empty_is_zero() {
    let b = Bodies::with_capacity(0);
    assert_eq!(total_angular_momentum(&b), 0.0);
}

/// Degenerate one-body "orbit": L about the CoM is zero for every
/// velocity direction.
#[test]
fn q_l_single_body_is_zero_without_spin() {
    for (vx, vy) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (3.0, -4.0)] {
        let mut b = Bodies::with_capacity(1);
        b.push(5.0, -3.0, vx, vy, 1.0, BodyKind::Planet);
        assert_eq!(total_angular_momentum(&b), 0.0, "v = ({vx}, {vy})");
    }
}

#[test]
fn q_l_single_body_returns_spin() {
    let mut b = Bodies::with_capacity(1);
    b.push(1.0, 2.0, 3.0, 4.0, 1.0, BodyKind::Planet);
    b.spin[0] = 0.75;
    assert!((total_angular_momentum(&b) - 0.75).abs() < 1e-15);
}

/// Radial motion: r × v = 0 per body → L vanishes.
#[test]
fn q_l_two_body_radial_is_zero() {
    let mut b = Bodies::with_capacity(2);
    b.push(-1.0, 0.0, -1.0, 0.0, 1.0, BodyKind::Planet);
    b.push(1.0, 0.0, 1.0, 0.0, 1.0, BodyKind::Planet);
    assert!(total_angular_momentum(&b).abs() < 1e-15);
}

/// Circular two-body: L = μ·√(G·M·a). Sign flips on retrograde.
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
/// the algebraically-equivalent one-pass form Σm·r×v − M·R×V diverges
/// as the offset grows.
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

/// Spin adds linearly to the orbital L.
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
// 7c. mark / settle / rebase
// ---------------------------------------------------------------------------

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

/// Idempotent: a retry mid-step must not overwrite the original
/// snapshot, or drift would be under-counted.
#[test]
fn q_mark_energy_idempotent() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(5.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    mark_energy(&mut b);
    let first = b.e_pre;
    b.vx[0] = 1.0;
    mark_energy(&mut b);
    assert_eq!(b.e_pre, first, "second mark must be a no-op");
}

/// `settle_energy` without a prior mark is a no-op.
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

/// Multiple mark/settle cycles accumulate into `energy_offset`.
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

/// `settle_energy` accumulates exactly (post-collision − pre-collision)
/// energy into `energy_offset`.
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

/// Above ENERGY_MAX_N: `rebase_energy` sets energy0 = 0 via
/// `unwrap_or`; drift helpers must return None without panicking.
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

/// End-to-end purpose of `energy_offset`: after an inelastic collision,
/// raw energy changes substantially but offset-adjusted drift stays ~0.
/// If `settle_energy` stopped being called after collisions, the
/// drift-smallness tests would still pass but this would fail.
#[test]
fn q_offset_excludes_collision_from_drift() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(0.001, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    rebase_energy(&mut b);
    let e_raw_before = total_energy(&b).unwrap();
    assert!(e_raw_before < 0.0, "overlapping pair is bound");

    mark_energy(&mut b);
    collision_pass(&mut b, DT, None);
    settle_energy(&mut b);

    assert_eq!(b.len(), 1, "inelastic merge happened");
    let e_raw_after = total_energy(&b).unwrap();
    let raw_drift = (e_raw_after - e_raw_before) / e_raw_before.abs() * 100.0;
    assert!(raw_drift.abs() > 1.0, "raw drift after merge: {raw_drift}%");
    let drift = energy_drift_pct(&b).unwrap();
    assert!(
        drift.abs() < 1e-6,
        "offset-adjusted drift {drift}% should exclude the merge"
    );
    assert!(b.stats.energy_offset != 0.0);
}

// ---------------------------------------------------------------------------
// 7d. drift / dissipated
// ---------------------------------------------------------------------------

/// `e0 = 0` → drift returns None (undefined ratio).
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
    b.vx[1] += 0.005 * v;
    let e1 = total_energy(&b).unwrap();
    let d = energy_drift_pct(&b).unwrap();
    let expected = (e1 - e0) / e0.abs() * 100.0;
    assert!((d - expected).abs() < 1e-12, "formula: {d} vs {expected}");
    assert!(d > 0.0, "adding energy must give positive drift: {d}");
}

/// Zero offset → 0.0, not None.
#[test]
fn q_dissipated_zero_offset_is_zero() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let v = (G * M_SUN_ME / 5.0).sqrt();
    b.push(5.0, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    rebase_energy(&mut b);
    assert_eq!(energy_dissipated_pct(&b), Some(0.0));
}

/// `energy0 = 0` → None. Distinct None branch from the above-limit path
/// tested elsewhere.
#[test]
fn q_dissipated_zero_e0_none() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    rebase_energy(&mut b);
    assert_eq!(b.stats.energy0, 0.0);
    assert_eq!(energy_dissipated_pct(&b), None);
}

/// Stable solar system: integrator drift stays under 0.5 % over 500 steps.
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

/// Above ENERGY_MAX_N the diagnostic helper returns None rather than
/// performing the O(n²) sum.
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

// ===========================================================================
// 8. Kind transitions via collision
// ===========================================================================

/// A sub-STAR_M_MIN "star" that gains mass is demoted to brown dwarf.
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

/// Two brown dwarfs whose combined mass exceeds BD_M_MAX ignite.
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
// 9. User edits
// ===========================================================================

// ---------------------------------------------------------------------------
// 9a. scale_mass
// ---------------------------------------------------------------------------

/// `scale_mass(i, 2)` then `scale_mass(i, 0.5)` restores mass and radius.
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

/// Non-finite or sub-1e-4 scaled mass is rejected: no mutation, no log,
/// no commit. Three sub-cases in one test so a regression to
/// `!m.is_finite()` alone or to `<= 1e-4` is caught.
#[test]
fn s_scale_mass_rejects_invalid() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(5.0, 0.0, 0.0, 2.0, 1.0, BodyKind::Planet);
    let (m0, mi0, r0) = (b.m[1], b.m_init[1], b.r[1]);
    let ev0 = b.events.len();
    b.parent_cache_valid = true;

    scale_mass(&mut b, 1, f64::INFINITY);
    scale_mass(&mut b, 1, f64::NAN);
    scale_mass(&mut b, 1, f64::NEG_INFINITY);
    scale_mass(&mut b, 1, 1e-10);

    assert_eq!(b.m[1], m0);
    assert_eq!(b.m_init[1], mi0);
    assert_eq!(b.r[1], r0);
    assert_eq!(b.events.len(), ev0, "no log on rejection");
    assert!(b.parent_cache_valid, "no commit on rejection");
}

/// Exact boundary `m == 1e-4` is accepted (guard is `< 1e-4`, not `<=`).
#[test]
fn s_scale_mass_accepts_exact_floor() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    scale_mass(&mut b, 0, 1e-4);
    assert_eq!(b.m[0], 1e-4);
    assert_eq!(b.m_init[0], 1e-4);
}

/// Success path runs `user_modified` (cache cleared, energy rebased).
#[test]
fn s_scale_mass_commits_state() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(5.0, 0.0, 0.0, 2.0, 1.0, BodyKind::Planet);
    b.parent_cache_valid = true;
    b.stats.energy_offset = 42.0;
    b.e_pre = Some(999.0);

    scale_mass(&mut b, 1, 2.0);

    assert!(!b.parent_cache_valid, "parent cache not invalidated");
    assert_eq!(b.stats.energy_offset, 0.0, "rebase_energy not called");
    assert!(b.e_pre.is_none(), "e_pre not cleared");
}

#[test]
fn s_scale_mass_logs_event() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let ev0 = b.events.len();
    scale_mass(&mut b, 0, 2.0);
    assert_eq!(b.events.len(), ev0 + 1);
    let e = b.events.back().unwrap();
    assert_eq!(e.kind, EventKind::User);
    assert!(e.text.contains("mass scaled by"), "text = {:?}", e.text);
}

/// Star branch: lifetime fraction `age / t_lifetime(m_init)` is
/// preserved exactly — a star keeps its evolutionary stage when the
/// user rescales it.
#[test]
fn s_scale_mass_star_preserves_lifetime_fraction() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.age_gyr[0] = 2.5;
    let t_old = star_lifetime_gyr(BodyKind::Star, b.m_init[0]);
    let frac_before = b.age_gyr[0] / t_old;

    scale_mass(&mut b, 0, 2.0);

    let t_new = star_lifetime_gyr(BodyKind::Star, b.m_init[0]);
    let frac_after = b.age_gyr[0] / t_new;
    assert!(
        (frac_after - frac_before).abs() < 1e-12,
        "fraction {frac_after} != {frac_before}"
    );
    assert!((b.m_init[0] - 2.0 * M_SUN_ME).abs() < 1e-12);
    assert!((b.m[0] - 2.0 * M_SUN_ME).abs() < 1e-12);
}

/// Star branch, age 0: fraction is 0/t = 0, so age stays exactly zero
/// regardless of lifetime change.
#[test]
fn s_scale_mass_star_at_zero_age() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.age_gyr[0] = 0.0;
    scale_mass(&mut b, 0, 2.0);
    assert_eq!(b.age_gyr[0], 0.0);
    assert!((b.m_init[0] - 2.0 * M_SUN_ME).abs() < 1e-12);
}

/// Star branch, near death: age = 0.99·t_old, halve the mass (lifetime
/// roughly quintuples for M^-2.5). New age must be 0.99·t_new — catches
/// a regression that rescales mass but forgets the age.
#[test]
fn s_scale_mass_star_near_death() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let t_old = star_lifetime_gyr(BodyKind::Star, M_SUN_ME);
    b.age_gyr[0] = 0.99 * t_old;

    scale_mass(&mut b, 0, 0.5);

    let t_new = star_lifetime_gyr(BodyKind::Star, 0.5 * M_SUN_ME);
    let expected = 0.99 * t_new;
    assert!(
        (b.age_gyr[0] - expected).abs() < 1e-9 * t_new,
        "age {} vs expected {}",
        b.age_gyr[0],
        expected
    );
}

/// Non-star branch: `m_init = m`, not `m_init *= factor`. Distinguishes
/// the two paths — dropping the star `if` would make every non-star
/// behave like the star path.
#[test]
fn s_scale_mass_nonstar_sets_m_init() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    scale_mass(&mut b, 0, 3.0);
    assert!((b.m_init[0] - 3.0).abs() < 1e-12);
    assert!((b.m[0] - 3.0).abs() < 1e-12);
}

/// `factor = 1.0` still logs and commits — the UI relies on
/// `scale_mass(i, 1.0)` re-deriving accelerations after a slider is
/// "committed" without being dragged.
#[test]
fn s_scale_mass_factor_one_still_commits_and_logs() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(5.0, 0.0, 0.0, 2.0, 1.0, BodyKind::Planet);
    b.parent_cache_valid = true;
    let m0 = b.m[1];
    let ev0 = b.events.len();

    scale_mass(&mut b, 1, 1.0);

    assert_eq!(b.m[1], m0, "mass must not change");
    assert_eq!(b.events.len(), ev0 + 1, "must still log");
    assert!(!b.parent_cache_valid, "must still commit");
}

// ---------------------------------------------------------------------------
// 9b. set_kind
// ---------------------------------------------------------------------------

/// Loose low-side clamp: a tiny planet promoted to black hole ends up
/// above `BH_M_MIN`. The exact snap target (10·M_SUN_ME) is pinned by
/// the more precise `s_set_kind_bh_undersized_and_interior` below.
#[test]
fn s_set_kind_clamps_mass() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    set_kind(&mut b, 0, BodyKind::BlackHole);
    assert_eq!(b.kind[0], BodyKind::BlackHole);
    assert!(b.m[0] >= BH_M_MIN, "mass {} < BH_M_MIN", b.m[0]);
}

/// Star branch: undersized and oversized both snap to M_SUN_ME;
/// in-range mass is preserved. Both sides of the range guard in one test.
#[test]
fn s_set_kind_star_guard_covers_both_sides_and_interior() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(50.0, 0.0, 0.0, 0.0, 50_000_000.0, BodyKind::Planet);
    b.push(100.0, 0.0, 0.0, 0.0, 50_000.0, BodyKind::Planet);

    set_kind(&mut b, 0, BodyKind::Star);
    assert_eq!(b.m[0], M_SUN_ME, "undersized star must snap to 1 M☉");
    set_kind(&mut b, 1, BodyKind::Star);
    assert_eq!(b.m[1], M_SUN_ME, "oversized star must snap to 1 M☉");
    set_kind(&mut b, 2, BodyKind::Star);
    assert_eq!(b.m[2], 50_000.0, "in-range mass must be preserved");
}

#[test]
fn s_set_kind_bd_clamps_both_directions() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet); // << BD_M_MIN
    b.push(50.0, 0.0, 0.0, 0.0, 1e6, BodyKind::Planet); // >> BD_M_MAX
    set_kind(&mut b, 0, BodyKind::BrownDwarf);
    assert_eq!(b.m[0], BD_M_MIN);
    set_kind(&mut b, 1, BodyKind::BrownDwarf);
    assert_eq!(b.m[1], BD_M_MAX);
}

#[test]
fn s_set_kind_wd_clamps_both_directions() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(50.0, 0.0, 0.0, 0.0, 1e6, BodyKind::Planet);
    set_kind(&mut b, 0, BodyKind::WhiteDwarf);
    assert_eq!(b.m[0], WD_M_MIN);
    set_kind(&mut b, 1, BodyKind::WhiteDwarf);
    assert_eq!(b.m[1], WD_M_MAX);
}

#[test]
fn s_set_kind_ns_clamps_both_directions() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(50.0, 0.0, 0.0, 0.0, 1e7, BodyKind::Planet);
    set_kind(&mut b, 0, BodyKind::NeutronStar);
    assert_eq!(b.m[0], NS_M_MIN);
    set_kind(&mut b, 1, BodyKind::NeutronStar);
    assert_eq!(b.m[1], NS_M_MAX);
}

/// Black hole low-side snaps to *exactly* 10·M_SUN_ME (not BH_M_MIN);
/// in-range preserved.
#[test]
fn s_set_kind_bh_undersized_and_interior() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(50.0, 0.0, 0.0, 0.0, 1e7, BodyKind::Planet);
    set_kind(&mut b, 0, BodyKind::BlackHole);
    assert_eq!(b.m[0], 10.0 * M_SUN_ME, "undersized BH snaps to 10 M☉");
    set_kind(&mut b, 1, BodyKind::BlackHole);
    assert_eq!(b.m[1], 1e7, "in-range BH mass preserved");
}

/// Dwarf-planet: clamps high side only; below DWARF_M_MAX stays.
#[test]
fn s_set_kind_dwarf_clamps_high_only() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 10.0, BodyKind::Planet); // >> DWARF_M_MAX
    b.push(50.0, 0.0, 0.0, 0.0, 0.001, BodyKind::Planet); // << DWARF_M_MAX
    set_kind(&mut b, 0, BodyKind::DwarfPlanet);
    assert_eq!(b.m[0], DWARF_M_MAX);
    set_kind(&mut b, 1, BodyKind::DwarfPlanet);
    assert_eq!(b.m[1], 0.001, "below-max mass untouched");
}

/// Planet: mass ≥ 1000 snaps to JUPITER_MASS; below stays.
#[test]
fn s_set_kind_planet_guard() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 2000.0, BodyKind::Planet);
    b.push(50.0, 0.0, 0.0, 0.0, 500.0, BodyKind::Planet);
    set_kind(&mut b, 0, BodyKind::Planet);
    assert_eq!(b.m[0], JUPITER_MASS);
    set_kind(&mut b, 1, BodyKind::Planet);
    assert_eq!(b.m[1], 500.0);
}

/// Non-guarded path resets all derived fields: class → Rocky, age →
/// default_age_gyr, bulk_density recomputed, origin → Spawned,
/// m_init = new_m.
#[test]
fn s_set_kind_resets_all_derived_fields() {
    let mut b = Bodies::with_capacity(1);
    let i = b.push(0.0, 0.0, 0.0, 0.0, 0.5 * M_SUN_ME, BodyKind::Planet);
    b.class[i] = BodyClass::Icy;
    b.age_gyr[i] = 5.0;
    b.bulk_density[i] = 999.0;
    b.origin[i] = BodyOrigin::ImpactDebris;

    set_kind(&mut b, 0, BodyKind::WhiteDwarf);

    assert_eq!(b.kind[0], BodyKind::WhiteDwarf);
    assert_eq!(b.m[0], 0.5 * M_SUN_ME, "in-range WD mass preserved");
    assert_eq!(b.class[0], BodyClass::Rocky);
    assert_eq!(b.age_gyr[0], default_age_gyr(BodyKind::WhiteDwarf));
    assert_ne!(b.bulk_density[0], 999.0);
    assert!(b.bulk_density[0] > 0.0);
    assert_eq!(b.origin[0], BodyOrigin::Spawned);
    assert_eq!(b.m_init[0], 0.5 * M_SUN_ME);
}

/// Same kind is not a no-op: age and class reset anyway. The UI relies
/// on this so "re-classify" resets a half-spent star to ZAMS.
#[test]
fn s_set_kind_same_kind_still_resets() {
    let mut b = Bodies::with_capacity(1);
    let i = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.age_gyr[i] = 3.0;
    b.class[i] = BodyClass::Icy;

    set_kind(&mut b, 0, BodyKind::Star);

    assert_eq!(b.m[0], M_SUN_ME, "in-range mass unchanged");
    assert_eq!(b.age_gyr[0], 0.0, "age reset to ZAMS");
    assert_eq!(b.class[0], BodyClass::Rocky, "class reset");
}

#[test]
fn s_set_kind_commits_and_logs() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(5.0, 0.0, 0.0, 2.0, 1.0, BodyKind::Planet);
    b.parent_cache_valid = true;
    b.stats.energy_offset = 42.0;
    let ev0 = b.events.len();

    set_kind(&mut b, 1, BodyKind::WhiteDwarf);

    assert!(!b.parent_cache_valid);
    assert_eq!(b.stats.energy_offset, 0.0);
    assert_eq!(b.events.len(), ev0 + 1);
    assert_eq!(b.events.back().unwrap().kind, EventKind::User);
    assert!(b.events.back().unwrap().text.contains("changed type"));
}

// ---------------------------------------------------------------------------
// 9c. delete_body
// ---------------------------------------------------------------------------

/// `compact()` is a retain, not swap-remove — array order is preserved.
/// Deleting index 0 shifts every later body down; ids follow.
#[test]
fn s_delete_body_first_shifts_all() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mid = b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let id_mid = b.id[mid];
    let tail = b.push(2.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let id_tail = b.id[tail];

    delete_body(&mut b, 0);

    assert_eq!(b.len(), 2);
    assert_eq!(b.id[0], id_mid, "mid body shifts to 0");
    assert_eq!(b.id[1], id_tail, "tail body shifts to 1");
}

#[test]
fn s_delete_body_last_no_shift() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mid = b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let id_mid = b.id[mid];
    b.push(2.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);

    delete_body(&mut b, 2);

    assert_eq!(b.len(), 2);
    assert_eq!(b.id[1], id_mid);
}

/// Only body: leaves empty system. `compute_accel` / `rebase_energy`
/// on zero bodies must not panic.
#[test]
fn s_delete_body_only_body() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    delete_body(&mut b, 0);
    assert_eq!(b.len(), 0);
}

/// The event log must use `tag(b, i)` captured *before* `compact()`.
/// Moving the tag call after compaction would surface here as the wrong
/// body's name in the log.
#[test]
fn s_delete_body_logs_pre_compact_tag() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let t = b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(2.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.name[t] = Some("Target".to_string());

    delete_body(&mut b, t);

    let e = b.events.back().unwrap();
    assert_eq!(e.kind, EventKind::User);
    assert!(
        e.text.starts_with("Target"),
        "log must use the deleted body's tag, got {:?}",
        e.text
    );
    assert!(e.text.contains("deleted"));
}

#[test]
fn s_delete_body_commits_state() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.parent_cache_valid = true;
    b.stats.energy_offset = 42.0;
    b.e_pre = Some(999.0);

    delete_body(&mut b, 1);

    assert!(!b.parent_cache_valid);
    assert_eq!(b.stats.energy_offset, 0.0);
    assert!(b.e_pre.is_none());
}

// ===========================================================================
// 10. System generation
// ===========================================================================

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
    let max16 = 62u128.pow(16) - 1; // largest value expressible in 16 chars
    for seed in [
        0u128,
        1,
        61,
        62,
        63,
        1000,
        u32::MAX as u128 / 2,
        u32::MAX as u128 - 1,
        u32::MAX as u128,
        // Values only representable in the 16-char format:
        u32::MAX as u128 + 1,
        u64::MAX as u128,
        max16 / 2,
        max16,
    ] {
        let s = seed_to_base62(seed);
        assert_eq!(s.len(), 16, "seed {seed} formatted as {s:?}");
        assert_eq!(seed_from_base62(&s), Some(seed), "seed {seed}");
    }

    // Rejections that still apply in the 16-char format.
    assert_eq!(seed_from_base62(""), None); // empty
    assert_eq!(seed_from_base62("00000000000000000"), None); // 17 chars
    assert_eq!(seed_from_base62("!@#$%^"), None); // bad chars

    // A 6-char string is now a *short* seed (leading zeros implied), not
    // an error. `"zzzzzz"` decodes to 62^6 - 1 with no truncation.
    assert_eq!(seed_from_base62("ZZZZZZ"), Some(62u128.pow(6) - 1));
}

#[test]
fn s_preset_dwarf_planets_are_dwarfs() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    for m in [0.000159, 0.00218, 0.00066, 0.00052, 0.00278] {
        let i = (0..b.len())
            .find(|&i| (b.m[i] - m).abs() < 1e-6 && b.origin[i] == BodyOrigin::Spawned)
            .unwrap_or_else(|| panic!("no body of mass {m}"));
        assert_eq!(b.kind[i], BodyKind::DwarfPlanet, "mass {}", m);
    }
}

/// Random phases: most spawned planets must NOT be at periapsis.
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
        // Skip moons.
        if matches!(
            find_parent(&b, i),
            Some(p) if matches!(b.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet)
        ) {
            continue;
        }
        n_planets += 1;
        let o = orbit_info(&b, i, 0);
        if (o.dist / o.peri - 1.0).abs() < 1e-3 {
            n_at_peri += 1;
        }
    }
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
    for seed in 0..2000u128 {
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

/// Giant planets are given 1–4 regular satellites, but the moon budget is
/// `(remaining − base_placed) / 10`, so an 80-body system only has room for
/// a handful of moons across the whole run. This test therefore guards the
/// *channel* (giant moons appear, and are bound to their giant), not a rate.
#[test]
fn s_moons_spawn_around_giants() {
    let (mut total_moons, mut unbound) = (0, 0);
    for seed in 0..100u128 {
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
                    if !orbit_info(&b, i, p).bound {
                        unbound += 1;
                    }
                }
            }
        }
    }
    assert!(total_moons > 0, "no giant moons spawned in 100 systems");
    assert_eq!(unbound, 0, "{unbound} of {total_moons} moons unbound");
}

#[test]
fn s_body_count_with_moons_never_exceeds_ceiling() {
    for seed in 0..100u128 {
        let mut b = Bodies::with_capacity(100);
        spawn_system(&mut b, 100, seed);
        assert!(b.len() <= 100, "seed {}: {} bodies", seed, b.len());
    }
}

/// Giants are spawned at 4–15 AU (`GIANT_A_MIN`), so a 50–318 M⊕ planet
/// inside 4 AU can only come from the hot/warm-Jupiter channel. That channel
/// is a few percent of systems by design; this test guards that it exists and
/// that its rate stays in a plausible band, using the full 3D separation.
#[test]
fn s_warm_jupiters_exist() {
    let mut n = 0;
    for seed in 0..500u128 {
        let mut b = Bodies::with_capacity(50);
        spawn_system(&mut b, 50, seed);
        for i in 0..b.len() {
            if b.origin[i] != BodyOrigin::Spawned || b.m[i] < 50.0 || b.m[i] > 318.0 {
                continue;
            }
            let d = (b.x[i] - b.x[0]).hypot(b.y[i] - b.y[0]).hypot(b.z[i] - b.z[0]);
            let a_au = d / au_in_gm();
            if a_au > 0.1 && a_au < 4.0 {
                n += 1;
                break;
            }
        }
    }
    eprintln!("warm Jupiters: {n}/500 systems");
    assert!(n >= 3, "warm Jupiters found: {n} (channel disappeared?)");
    assert!(n < 100, "warm Jupiters found: {n} (channel too common?)");
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

/// `refresh_parent_cache`'s moon scan must be bit-identical regardless of
/// whether the spatial index (`MoonIndex`) is in use. The test builds a
/// population well past `MOON_TREE_MIN` — a Plummer cluster of systems plus
/// hundreds of dense planet-kind bodies in far corners — so the index prunes
/// most candidates per query, then checks *every* body through all three
/// paths: tree-pruned cached, plain-scan cached, and the true uncached
/// `find_parent`.
#[test]
fn parent_cache_index_matches_plain_scan_at_scale() {
    let mut b = Bodies::with_capacity(14_000);
    spawn_random_systems(&mut b, 40, 0xB0BAC, 6);
    let mut rng = XorShift::new(0xC0FFEE);
    let centres = [
        (1e4f64, 0.0f64, 0.0f64),
        (-1e4, 1e4, 0.0),
        (1e4, -1e4, 1e4),
        (-1e4, -1e4, -1e4),
    ];
    for &(cx, cy, cz) in &centres {
        for _ in 0..800usize {
            let m = 1e-6 + rng.next_f64() * 1e-5;
            let (px, py, pz) = (
                cx + (rng.next_f64() - 0.5) * 4.0,
                cy + (rng.next_f64() - 0.5) * 4.0,
                cz + (rng.next_f64() - 0.5) * 4.0,
            );
            let i = b.push(px, py, 0.0, 0.0, m, BodyKind::DwarfPlanet);
            b.z[i] = pz;
            b.r[i] = 2e-4;
        }
    }
    let n = b.len();
    assert!(n > 3_400, "population {n} too small to stress the index");

    let ctx = build_parent_context(&b);
    assert!(ctx.moon_index.is_some(), "index must be built at this size");
    let plain = ParentContext {
        hosts: ctx.hosts.clone(),
        partners: ctx.partners.clone(),
        moons: ctx.moons.clone(),
        moon_index: None,
    };

    for i in 0..n {
        let cached = find_parent_cached(&b, i, &ctx);
        assert_eq!(
            cached,
            find_parent_cached(&b, i, &plain),
            "forest vs plain scan diverged for body {i}"
        );
        assert_eq!(
            cached,
            find_parent(&b, i),
            "pruned moon scan diverged from find_parent for body {i}"
        );
    }
}

/// A moon inside its host's Hill reach must be classified as a moon, and
/// one outside must not — through both the spatial-index and plain-scan
/// paths, agreeing with the uncached `find_parent`. The inside/outside
/// placement uses 0.4 × and 0.6 × `r_hill` so the float ratio is clearly on
/// one side of `MOON_HILL_FRAC` (the exact-boundary case is float-ambiguous
/// even for the plain scan).
#[test]
fn parent_cache_moon_at_hill_boundary_selects_planet() {
    let mut b = Bodies::with_capacity(700);
    let au = au_in_gm();
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let v = (G * M_SUN_ME / au).sqrt();
    let planet = b.push(au, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.vy[planet] = v;
    let rhill = au * (1.0 / (3.0 * M_SUN_ME)).cbrt();
    let inside = b.push(
        au + 0.4 * rhill,
        0.0,
        0.0,
        (G * M_SUN_ME / (au + 0.4 * rhill)).sqrt(),
        0.0123,
        BodyKind::Planet,
    );
    let outside = b.push(
        au + 0.6 * rhill,
        0.0,
        0.0,
        (G * M_SUN_ME / (au + 0.6 * rhill)).sqrt(),
        0.0123,
        BodyKind::Planet,
    );
    // Distant clutter forces the index to be built (well past
    // `MOON_TREE_MIN` candidates) while staying far outside every reach.
    let mut rng = XorShift::new(7);
    for _ in 0..400usize {
        let i = b.push(
            3.0 * au + (rng.next_f64() - 0.5) * au,
            0.0,
            0.0,
            0.0,
            1e-5,
            BodyKind::DwarfPlanet,
        );
        b.r[i] = 1e-4;
    }

    let ctx = build_parent_context(&b);
    assert!(ctx.moon_index.is_some(), "index must be built");
    let plain = ParentContext {
        hosts: ctx.hosts.clone(),
        partners: ctx.partners.clone(),
        moons: ctx.moons.clone(),
        moon_index: None,
    };

    assert_eq!(find_parent_cached(&b, inside, &ctx), Some(planet));
    assert_eq!(find_parent_cached(&b, inside, &plain), Some(planet));
    assert_eq!(find_parent(&b, inside), Some(planet));
    assert_eq!(find_parent_cached(&b, outside, &ctx), Some(0));
    assert_eq!(find_parent_cached(&b, outside, &plain), Some(0));
    assert_eq!(find_parent(&b, outside), Some(0));
    // And the two cached paths agree system-wide.
    for i in 0..b.len() {
        assert_eq!(
            find_parent_cached(&b, i, &ctx),
            find_parent_cached(&b, i, &plain),
            "body {i}"
        );
    }
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
