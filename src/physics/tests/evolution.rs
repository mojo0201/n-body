//! Tests: stellar evolution, remnants, mass transfer, aging.

use super::super::*;
use super::shared::*;

#[test]
fn al_evolution_is_off_by_default() {
    let mut b = sun_system(&[1.0]);
    let (m, r) = (b.m[0], b.r[0]);
    for _ in 0..200 {
        advance(&mut b, DT);
    }
    assert_eq!(b.age_gyr[0], 0.0, "age should not advance");
    assert_eq!(b.m[0], m, "mass unchanged");
    assert_eq!(b.r[0], r, "radius unchanged");
    assert_eq!(b.kind[0], BodyKind::Star);
}

#[test]
fn al_age_zero_matches_the_old_relations() {
    // At age 0 the evolution model reproduces the pre-evolution relations
    // exactly, so presets look identical with evolution on and off.
    for s in [0.1_f64, 0.5, 1.0, 2.0, 7.0, 30.0] {
        let m = s * M_SUN_ME;
        let st = star_state(m, m, 0.0);
        assert_eq!(
            st.radius_gm,
            physical_radius(BodyKind::Star, m),
            "star_state at age 0 must match physical_radius"
        );
        assert_eq!(
            st.luminosity,
            star_luminosity(BodyKind::Star, m),
            "star_state at age 0 must match star_luminosity"
        );
        let t = star_temperature(BodyKind::Star, m);
        assert!((st.temperature - t).abs() < 1e-9 * t);
        assert_eq!(st.phase, StellarPhase::MainSequence);
    }
}

#[test]
fn al_giant_phase_is_bright_large_and_cool() {
    let m = M_SUN_ME;
    let t_ms = star_lifetime_gyr(BodyKind::Star, m);
    let late = star_state(m, m * 0.6, t_ms * (1.0 + EVO_POST_MS_FRAC * 0.95));
    assert_eq!(late.phase, StellarPhase::RedGiant);
    assert!(late.radius_gm > 100.0 * R_SUN_GM);
    assert!(late.luminosity > 1000.0);
    assert!(late.temperature < 4500.0);
    // Radius and luminosity are continuous across the MS→giant transition.
    let before = star_state(m, m, t_ms * 0.999_999);
    let after = star_state(m, m, t_ms * 1.000_001);
    assert!((after.radius_gm / before.radius_gm - 1.0).abs() < 1e-3);
    assert!((after.luminosity / before.luminosity - 1.0).abs() < 1e-2);
}

#[test]
fn al_sunlike_star_becomes_a_white_dwarf() {
    let mut b = sun_system(&[30.0]);
    set_evolution_rate(&mut b, 4.0);
    run_to_remnant(&mut b, 5000);
    assert_eq!(b.kind[0], BodyKind::WhiteDwarf);
    // Kalirai relation for 1 M_SUN_ME ≈ 0.503 M_SUN_ME.
    let ms = b.m[0] / M_SUN_ME;
    assert!((ms - 0.503).abs() < 0.02, "WD mass {ms}");
    assert!(b.r[0] < 0.05);
    assert!(b.events.iter().any(|e| e.kind == EventKind::Evolve));
}

#[test]
fn al_massive_star_becomes_a_neutron_star() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, 12.0 * M_SUN_ME, BodyKind::Star);
    let host = b.len() - 1;
    push_orbit(&mut b, host, 50.0, 0.0, 1.0, 0.0);
    b.sn_kicks = false;
    compute_accel(&mut b);
    rebase_energy(&mut b);
    set_evolution_rate(&mut b, 10.0);
    run_to_remnant(&mut b, 1000);
    assert_eq!(b.kind[0], BodyKind::NeutronStar);
    let ms = b.m[0] / M_SUN_ME;
    // Section-B model: NS mass = 1.4 + 0.6 * (12-8)/(20-8) = 1.6.
    assert!((ms - 1.6).abs() < 0.05, "NS mass {ms}");
}

#[test]
fn al_very_massive_star_becomes_a_black_hole() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, 25.0 * M_SUN_ME, BodyKind::Star);
    let host = b.len() - 1;
    push_orbit(&mut b, host, 50.0, 0.0, 1.0, 0.0);
    b.sn_kicks = false;
    compute_accel(&mut b);
    rebase_energy(&mut b);
    set_evolution_rate(&mut b, 10.0);
    run_to_remnant(&mut b, 1000);
    assert_eq!(b.kind[0], BodyKind::BlackHole);
    let ms = b.m[0] / M_SUN_ME;
    // Section-B model: BH mass = max(0.25 * 25, BH_M_MIN) = 6.25.
    assert!((ms - 6.25).abs() < 0.05, "BH mass {ms}");
}

#[test]
fn al_supernova_kick_is_optional() {
    for kicks in [false, true] {
        let mut b = Bodies::with_capacity(2);
        b.push(0.0, 0.0, 0.0, 0.0, 25.0 * M_SUN_ME, BodyKind::Star);
        b.sn_kicks = kicks;
        become_remnant(&mut b, 0);
        let v = b.vx[0].hypot(b.vy[0]);
        if kicks {
            assert!(v > 0.0, "BH kick should give velocity");
        } else {
            assert_eq!(v, 0.0, "kicks off → no velocity");
        }
    }
}

#[test]
fn al_roche_lobe_matches_eggleton() {
    let rl = roche_lobe_radius(1.0, 1.0, 1.0);
    // Eggleton 1983: q = 1 gives ≈ 0.379.
    assert!((rl - 0.379).abs() < 0.002, "equal-mass RL = {rl}");
    assert!(
        roche_lobe_radius(1.0, 0.2, 1.0) < rl,
        "lighter donor has smaller lobe"
    );
}

#[test]
fn al_transfer_conserves_momentum_and_angular_momentum() {
    let mut b = Bodies::with_capacity(2);
    let d = b.push(10.0, -20.0, 1.0, 2.0, M_SUN_ME, BodyKind::Star);
    let a = b.push(100.0, 50.0, -3.0, 0.5, 0.6 * M_SUN_ME, BodyKind::WhiteDwarf);
    let totals = |b: &Bodies| {
        let mut t = (0.0, 0.0, 0.0);
        for i in 0..b.len() {
            t.0 += b.m[i] * b.vx[i];
            t.1 += b.m[i] * b.vy[i];
            t.2 += b.m[i] * (b.x[i] * b.vy[i] - b.y[i] * b.vx[i]);
        }
        t
    };
    let before = totals(&b);
    let dm = 0.01 * M_SUN_ME;
    assert!(!transfer_mass(&mut b, d, a, dm), "under Chandrasekhar");
    let after = totals(&b);
    let lost = dm * (1.0 - MT_EFF_COMPACT);
    let (px, py) = (after.0 + lost * 1.0, after.1 + lost * 2.0);
    let l = after.2 + lost * (10.0 * 2.0 - (-20.0) * 1.0);
    let scale = before.0.abs().max(before.1.abs()).max(before.2.abs());
    assert!(
        (px - before.0).abs() < 1e-9 * scale,
        "px {px} vs {}",
        before.0
    );
    assert!(
        (py - before.1).abs() < 1e-9 * scale,
        "py {py} vs {}",
        before.1
    );
    assert!((l - before.2).abs() < 1e-9 * scale, "L {l} vs {}", before.2);
}

#[test]
fn al_white_dwarf_crossing_the_limit_explodes() {
    let mut b = Bodies::with_capacity(2);
    let d = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = b.push(100.0, 0.0, 0.0, 0.0, 1.39 * M_SUN_ME, BodyKind::WhiteDwarf);
    assert!(
        transfer_mass(&mut b, d, a, 0.05 * M_SUN_ME),
        "should trigger Type Ia"
    );
    assert!(b.dead[a]);
    assert!(b.events.iter().any(|e| e.text.contains("Type Ia")));
}

#[test]
fn al_white_dwarf_cooling_decreases_luminosity() {
    let m = 0.6 * M_SUN_ME;
    let l1 = wd_luminosity(m, 1.0);
    let l10 = wd_luminosity(m, 10.0);
    let ratio = l1 / l10;
    let want = 10f64.powf(WD_COOL_EXP);
    assert!(
        (ratio / want - 1.0).abs() < 0.01,
        "10 Gyr fainter by {ratio} (want {want})"
    );
    assert!(wd_temperature(m, 0.0) > 50_000.0);
    assert!(wd_temperature(m, 10.0) < 6_000.0);
}

#[test]
fn al_brown_dwarf_cools_and_changes_class() {
    let m = 60.0 * JUPITER_MASS;
    let young = bd_state(m, 0.1);
    let old = bd_state(m, 10.0);
    assert!(young.luminosity > 100.0 * old.luminosity);
    assert!(young.radius_gm > old.radius_gm);
    let cy = bd_class_for_temp(young.temperature);
    let co = bd_class_for_temp(old.temperature);
    assert_ne!(cy, co, "young {cy}, old {co} should differ");
    assert!(young.temperature > old.temperature);
}

#[test]
fn al_default_ages_for_dwarfs_are_set() {
    let mut b = Bodies::with_capacity(2);
    let w = b.push(0.0, 0.0, 0.0, 0.0, 0.6 * M_SUN_ME, BodyKind::WhiteDwarf);
    let d = b.push(
        1e4,
        0.0,
        0.0,
        0.0,
        40.0 * JUPITER_MASS,
        BodyKind::BrownDwarf,
    );
    assert_eq!(b.age_gyr[w], WD_DEFAULT_AGE_GYR);
    assert_eq!(b.age_gyr[d], BD_DEFAULT_AGE_GYR);
    assert_eq!(default_age_gyr(BodyKind::WhiteDwarf), WD_DEFAULT_AGE_GYR);
    assert_eq!(default_age_gyr(BodyKind::BrownDwarf), BD_DEFAULT_AGE_GYR);
    assert_eq!(default_age_gyr(BodyKind::Star), 0.0);
}

#[test]
fn al_compact_moves_evolution_fields_with_their_bodies() {
    let mut b = Bodies::with_capacity(4);
    for k in 0..3 {
        b.push(
            k as f64,
            0.0,
            0.0,
            0.0,
            (k as f64 + 1.0) * M_SUN_ME,
            BodyKind::Star,
        );
        b.age_gyr[k] = k as f64;
        b.rlof[k] = k == 2;
    }
    b.dead[0] = true;
    b.compact();
    assert_eq!(b.len(), 2);
    assert_eq!(b.age_gyr, vec![1.0, 2.0]);
    assert_eq!(b.m_init, vec![2.0 * M_SUN_ME, 3.0 * M_SUN_ME]);
    assert_eq!(b.rlof, vec![false, true]);
}

#[test]
fn al_merging_stars_rejuvenates_the_survivor() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(0.5, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.age_gyr[0] = 5.0; // half MS lifetime
    let keep = merge(&mut b, 0, 1);
    assert_eq!(b.m_init[keep], 2.0 * M_SUN_ME, "m_init = sum");
    // The survivor keeps the mass-weighted fraction of MS life.
    let t_ms = star_lifetime_gyr(BodyKind::Star, 2.0 * M_SUN_ME);
    let expected = 0.25 * t_ms; // (0.5*1 + 0.0*1) / 2 = 0.25 fraction used
    assert!(
        (b.age_gyr[keep] - expected).abs() < 1e-6,
        "age {} vs expected {}",
        b.age_gyr[keep],
        expected
    );
}

#[test]
fn al_class_name_reports_red_giant_after_aging() {
    let mut b = sun_system(&[1.0]);
    let t_ms = star_lifetime_gyr(BodyKind::Star, M_SUN_ME);
    set_star_age(&mut b, 0, t_ms * 1.12);
    assert_eq!(class_name(&b, 0), "red giant");
}

#[test]
fn al_giant_luminosity_reaches_nearby_planets() {
    let mut b = sun_system(&[1.0]);
    let (x, y) = (b.x[1], b.y[1]);
    let f0 = stellar_flux(&b, x, y, 1);
    let t_ms = star_lifetime_gyr(BodyKind::Star, M_SUN_ME);
    set_star_age(&mut b, 0, t_ms * 1.12);
    assert_eq!(phase_of(&b, 0), Some(StellarPhase::RedGiant));
    let f1 = stellar_flux(&b, x, y, 1);
    assert!(f1 > 50.0 * f0, "flux {f0} → {f1}");
}

#[test]
fn al_set_star_age_clamps_to_total_life() {
    let mut b = sun_system(&[]);
    let life = star_total_life_gyr(M_SUN_ME);
    set_star_age(&mut b, 0, life * 2.0);
    // Setter clamps to end * 0.999999, so still a star, just barely.
    assert!(b.age_gyr[0] < life);
    assert!(b.age_gyr[0] > life * 0.999_990);
    assert_eq!(b.kind[0], BodyKind::Star);
}

#[test]
fn al_age_advances_only_when_rate_is_positive() {
    let mut b = sun_system(&[]);
    // Rate 0: age frozen.
    advance(&mut b, DT);
    assert_eq!(b.age_gyr[0], 0.0);
    // Rate 1 Gyr per tu: after one advance, age = 1 * DT Gyr.
    set_evolution_rate(&mut b, 1.0);
    advance(&mut b, DT);
    assert!((b.age_gyr[0] - DT).abs() < 1e-12);
    // Negative rate (invalid) → 0, no advance.
    set_evolution_rate(&mut b, -5.0);
    advance(&mut b, DT);
    assert!(
        (b.age_gyr[0] - DT).abs() < 1e-12,
        "negative rate should freeze"
    );
}

// ---- Prompt 1: random ages, remnant spawns, adiabatic vs SN mass loss ----
#[test]
fn al_wd_formation_expands_orbits() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a0 = 50.0;
    let v = (G * M_SUN_ME / a0).sqrt();
    b.push(a0, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    compute_accel(&mut b);
    let o0 = orbit_info(&b, 1, 0);
    become_remnant(&mut b, 0);
    assert_eq!(b.kind[0], BodyKind::WhiteDwarf);
    let o1 = orbit_info(&b, 1, 0);
    // m_old = 1 M_sun, m_new ≈ 0.5 M_sun; a should roughly double.
    let expected = o0.semi_major * (M_SUN_ME / b.m[0]);
    assert!(
        (o1.semi_major / expected - 1.0).abs() < 0.05,
        "a {} vs expected {}",
        o1.semi_major,
        expected
    );
    // The orbit stays bound and keeps its shape.
    assert!(o1.bound);
    assert!((o1.ecc - o0.ecc).abs() < 0.05, "e {} -> {}", o0.ecc, o1.ecc);
}

#[test]
fn al_wd_formation_moves_satellites_with_their_parent() {
    // A planet with a moon around a star that becomes a white dwarf: the
    // planet's orbit expands, the moon keeps its orbit about the planet.
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a0 = 50.0;
    let v = (G * M_SUN_ME / a0).sqrt();
    let p = b.push(a0, 0.0, 0.0, v, 300.0, BodyKind::Planet);
    let d = 0.2;
    let vm = (G * 300.0 / d).sqrt();
    let moon = b.push(a0 + d, 0.0, 0.0, v + vm, 0.01, BodyKind::Planet);
    compute_accel(&mut b);
    let before = orbit_info(&b, moon, p);
    become_remnant(&mut b, 0);
    let after = orbit_info(&b, moon, p);
    assert!(
        (after.semi_major / before.semi_major - 1.0).abs() < 1e-6,
        "moon a {} -> {}",
        before.semi_major,
        after.semi_major
    );
}

#[test]
fn al_wd_formation_leaves_other_systems_alone() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    // A bound neighbour 200 AU away, as on the ring of systems.
    b.push(30_000.0, 0.0, 0.0, 5.0, M_SUN_ME, BodyKind::Star);
    let (x, y, vx, vy) = (b.x[1], b.y[1], b.vx[1], b.vy[1]);
    become_remnant(&mut b, 0);
    assert_eq!((b.x[1], b.y[1], b.vx[1], b.vy[1]), (x, y, vx, vy));
}

#[test]
fn al_wd_formation_expands_close_companion() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let d = 1000.0;
    let v = (G * 2.0 * M_SUN_ME / d).sqrt();
    b.push(d, 0.0, 0.0, v, 0.3 * M_SUN_ME, BodyKind::Star);
    become_remnant(&mut b, 0);
    let o = orbit_info(&b, 1, 0);
    assert!(o.bound);
    assert!(o.semi_major > 1.5 * d, "a {}", o.semi_major);
}

#[test]
fn al_sn_unbinds_wide_planet() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 12.0 * M_SUN_ME, BodyKind::Star);
    let a0 = 50.0;
    let v = (G * 12.0 * M_SUN_ME / a0).sqrt();
    b.push(a0, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    compute_accel(&mut b);
    b.sn_kicks = false;
    become_remnant(&mut b, 0);
    let o = orbit_info(&b, 1, 0);
    // v_circ(12) > v_esc(1.6): planet becomes unbound.
    assert!(!o.bound, "wide planet should be unbound by SN");
}

#[test]
fn al_random_age_produces_some_remnants() {
    let mut n_rem = 0;
    for seed in 0..200u32 {
        let mut b = Bodies::with_capacity(10);
        spawn_system(&mut b, 10, seed);
        if !matches!(b.kind[0], BodyKind::Star | BodyKind::BrownDwarf) {
            n_rem += 1;
        }
    }
    assert!(
        n_rem >= 5,
        "expected at least 5 remnant primaries, got {}",
        n_rem
    );
}

#[test]
fn al_random_ages_are_consistent() {
    for seed in 0..300u32 {
        let mut b = Bodies::with_capacity(12);
        spawn_system(&mut b, 12, seed);
        for i in 0..b.len() {
            assert!(b.age_gyr[i].is_finite() && b.age_gyr[i] >= 0.0);
            if b.kind[i] == BodyKind::Star {
                // A star is never older than its life, and its radius and
                // mass match its evolutionary state.
                assert!(b.age_gyr[i] <= star_total_life_gyr(b.m_init[i]) + 1e-9);
                assert!(b.m[i] <= b.m_init[i] + 1e-9);
                assert!((b.r[i] - b.radius_now(i)).abs() <= 1e-12 * b.r[i].max(1.0));
            }
        }
    }
}

#[test]
fn al_random_systems_keep_ages_on_the_ring() {
    let mut b = Bodies::with_capacity(0);
    spawn_random_systems(&mut b, 10, 7, 4);
    let mut aged = 0;
    for i in 0..b.len() {
        if b.kind[i] == BodyKind::Star && b.age_gyr[i] > 0.0 {
            aged += 1;
        }
        assert!((b.r[i] - b.radius_now(i)).abs() <= 1e-12 * b.r[i].max(1.0));
    }
    // Four systems of random ages: at least one aged star or a remnant host.
    let remnants = (0..b.len())
        .filter(|&i| {
            matches!(
                b.kind[i],
                BodyKind::WhiteDwarf | BodyKind::NeutronStar | BodyKind::BlackHole
            )
        })
        .count();
    assert!(aged + remnants > 0);
}

#[test]
fn smoke_evolution_runs_without_panic() {
    for seed in 0..40u32 {
        let mut b = Bodies::with_capacity(0);
        spawn_random_systems(&mut b, 30, seed, 1);
        set_evolution_rate(&mut b, 1.0);
        for _ in 0..600 {
            advance(&mut b, DT);
        }
        assert!((0..b.len()).all(|i| b.x[i].is_finite() && b.vx[i].is_finite()));
    }
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    set_evolution_rate(&mut b, 100.0);
    for _ in 0..3000 {
        advance(&mut b, DT);
    }
    assert_eq!(b.kind[0], BodyKind::WhiteDwarf);
}

// ===========================================================================
// al2 — fixes for CRITICAL.txt
// ===========================================================================
/// The giant-phase envelope is lost over the whole post-MS phase, so
/// `evolve_stars` must call `adiabatic_expand` on every mass decrease.
/// The old code only expanded once, in `become_remnant`, after the mass had
/// already dropped to `m_rem`.
#[test]
fn evolve_sun_to_wd_expands_planet_orbit() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    // Start well outside the Sun's eventual giant radius (peak ≈ 139 Gm);
    // otherwise the planet is swallowed mid-giant and there is nothing
    // left to measure. With a0 = 500 Gm, a_peak ≈ 994 Gm and the planet
    // stays bound and outside the envelope.
    let a0 = 500.0;
    let v = (G * M_SUN_ME / a0).sqrt();
    b.push(a0, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    compute_accel(&mut b);
    rebase_energy(&mut b);
    set_evolution_rate(&mut b, 4.0);
    let o0 = orbit_info(&b, 1, 0);
    for _ in 0..5000 {
        advance(&mut b, DT);
        if b.kind[0] == BodyKind::WhiteDwarf {
            break;
        }
    }
    assert_eq!(b.kind[0], BodyKind::WhiteDwarf, "sun should become WD");
    assert_eq!(b.len(), 2, "planet must not have been swallowed");
    let o1 = orbit_info(&b, 1, 0);
    let expected = M_SUN_ME / b.m[0];
    let ratio = o1.semi_major / o0.semi_major;
    assert!(
        (ratio / expected - 1.0).abs() < 0.15,
        "a ratio {:.3} vs expected {:.3} (m_old {:.0} m_new {:.0})",
        ratio,
        expected,
        M_SUN_ME,
        b.m[0]
    );
    assert!(o1.bound);
}

#[test]
fn al_spawned_ages_within_universe_age() {
    for seed in 0..500u32 {
        let mut b = Bodies::with_capacity(30);
        spawn_system(&mut b, 30, seed);
        for i in 0..b.len() {
            if b.kind[i] == BodyKind::Star {
                assert!(
                    b.age_gyr[i] <= AGE_UNIVERSE_GYR,
                    "seed {} star {} age {}",
                    seed,
                    i,
                    b.age_gyr[i]
                );
            }
        }
    }
}
