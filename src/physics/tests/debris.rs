//! Tests: tidal disruption, rubble coalescence, belts and bulk density.

use super::super::*;

// ===========================================================================
// e — tidal break-up timing
// ===========================================================================
#[test]
fn e_tidal_break_up_waits_for_pericentre() {
    let mut b = Bodies::with_capacity(32);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let j = b.push(1.0, 0.0, 0.0, 0.0, 318.0, BodyKind::Planet);
    let r0 = 0.95 * roche_limit(&b, j, 0);
    let vc = (G * M_SUN_ME / r0).sqrt();
    b.x[j] = r0;
    b.vx[j] = -0.15 * vc;
    b.vy[j] = 0.9 * vc;
    compute_accel(&mut b);
    advance(&mut b, DT);
    assert_eq!(
        b.stats.tidal, 0,
        "must not break up on entering the Roche limit"
    );
    assert!(b.disrupting[1], "must be marked");
    let mut steps = 1;
    while b.stats.tidal == 0 && steps < 200 {
        advance(&mut b, DT);
        steps += 1;
    }
    assert_eq!(b.stats.tidal, 1, "must break up after pericentre");
    let rm = (1..b.len()).map(|k| b.x[k].hypot(b.y[k])).sum::<f64>() / (b.len() - 1) as f64;
    eprintln!(
        "broke up after {} steps at mean r {:.3} (start {:.3})",
        steps, rm, r0
    );
    assert!(
        rm < 0.9 * r0,
        "break-up must happen deeper than where the body entered"
    );
}

// ===========================================================================
// h — rubble coalescence
// ===========================================================================
#[test]
fn h_coalesce_counter_accumulates_and_merges() {
    let mut b = Bodies::with_capacity(8);
    let _star = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);

    let au = au_in_gm();
    let vc_star = (G * M_SUN_ME / au).sqrt();
    let p = b.push(au, 0.0, 0.0, vc_star, 1.0, BodyKind::Planet);

    let m_frag = 0.0005;
    let r_orb = 0.2;
    let vc_p = (G * 1.0 / r_orb).sqrt();
    let ang = 0.001_f64;

    let i = b.push(
        au + r_orb,
        0.0,
        0.0,
        vc_star + vc_p,
        m_frag,
        BodyKind::Planet,
    );
    let j = b.push(
        au + r_orb * ang.cos(),
        r_orb * ang.sin(),
        -vc_p * ang.sin(),
        vc_star + vc_p * ang.cos(),
        m_frag,
        BodyKind::Planet,
    );
    b.origin[i] = BodyOrigin::ImpactDebris;
    b.origin[j] = BodyOrigin::ImpactDebris;
    b.imm[i] = 0.0;
    b.imm[j] = 0.0;

    assert_eq!(find_parent(&b, i), Some(p));
    assert_eq!(find_parent(&b, j), Some(p));
    b.refresh_parent_cache();

    for _ in 0..DEBRIS_CAPTURE_STEPS {
        coalesce_debris(&mut b);
    }

    assert_eq!(b.len(), 3, "the two fragments should have coalesced");
    assert!(
        (b.m[2] - 2.0 * m_frag).abs() < 1e-9,
        "survivor has both masses"
    );
    assert_eq!(b.stats.coalesces, 1);
    assert_eq!(b.stats.merges, 1);
}

// ---- Prompt 2: belts, bulk density, mean-anomaly sampling ----
#[test]
fn belt_spec_returns_belt_for_most_systems() {
    let mut with_belt = 0;
    for seed in 0..300u32 {
        let mut b = Bodies::with_capacity(50);
        spawn_system(&mut b, 50, seed);
        if (0..b.len()).any(|i| b.origin[i].is_belt()) {
            with_belt += 1;
        }
    }
    // Belt occurrence is ~85 % for MS primaries, ~30 % for remnants, and
    // ~15 % of primaries are remnants. Expected ~77 %.
    assert!(with_belt > 180, "belt systems: {}/300", with_belt);
}

#[test]
fn kirkwood_gaps_at_jupiter_resonances() {
    let gaps = kirkwood_gaps(5.2026);
    assert!(gaps.iter().any(|g| (g.a_au - 2.50).abs() < 0.05)); // 3:1
    assert!(gaps.iter().any(|g| (g.a_au - 2.82).abs() < 0.05)); // 5:2
    assert!(gaps.iter().any(|g| (g.a_au - 2.96).abs() < 0.05)); // 7:3
    assert!(gaps.iter().any(|g| (g.a_au - 3.28).abs() < 0.05)); // 2:1
}

#[test]
fn belt_bodies_have_physical_bulk_density() {
    for seed in 0..40u32 {
        let mut b = Bodies::with_capacity(50);
        spawn_system(&mut b, 50, seed);
        for i in 0..b.len() {
            if b.origin[i].is_belt() {
                assert!(
                    b.bulk_density[i] > 0.1 && b.bulk_density[i] < 10.0,
                    "seed {} body {} density {}",
                    seed,
                    i,
                    b.bulk_density[i]
                );
            }
        }
    }
}

#[test]
fn belt_bodies_are_never_moons() {
    for seed in 0..30u32 {
        let mut b = Bodies::with_capacity(50);
        spawn_system(&mut b, 50, seed);
        b.refresh_parent_cache();
        for i in 0..b.len() {
            if b.origin[i].is_belt() {
                assert!(
                    find_parent(&b, i).is_none(),
                    "seed {} body {} classified as moon",
                    seed,
                    i
                );
                assert_ne!(body_class(&b, i), "moon");
            }
        }
    }
}

#[test]
fn trojans_sit_sixty_degrees_from_jupiter() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let jup = (0..b.len())
        .find(|&i| (b.m[i] - 317.80).abs() < 1e-2)
        .expect("Jupiter");
    let ang = |i: usize| (b.y[i] - b.y[0]).atan2(b.x[i] - b.x[0]);
    let aj = ang(jup);
    let mut n = 0;
    for i in 0..b.len() {
        if b.origin[i] == BodyOrigin::Trojan {
            let mut d = (ang(i) - aj).abs();
            if d > std::f64::consts::PI {
                d = std::f64::consts::TAU - d;
            }
            assert!(
                (d - std::f64::consts::FRAC_PI_3).abs() < 0.25,
                "Trojan {} is {:.2} rad from Jupiter",
                i,
                d
            );
            n += 1;
        }
    }
    assert!(n > 0);
}

#[test]
fn belt_mass_scatter_is_log_normal() {
    let mut rng = XorShift::new(42);
    let masses: Vec<f64> = (0..100)
        .map(|_| belt_mass_earth(1.0, 0.0, 1.0, &mut rng))
        .collect();
    let min = masses.iter().copied().fold(f64::INFINITY, f64::min);
    let max = masses.iter().copied().fold(0.0f64, f64::max);
    assert!(max / min > 20.0, "spread {}x", max / min);
}

#[test]
fn ices_are_lighter_than_rocks() {
    let r_rock = radius_from_density(1.0, 5.0);
    let r_ice = radius_from_density(1.0, 1.0);
    assert!(r_ice > r_rock, "rock {} ice {}", r_rock, r_ice);
}

#[test]
fn belt_dwarf_planets_use_dwarf_kind() {
    for seed in 0..200u32 {
        let mut b = Bodies::with_capacity(100);
        spawn_system(&mut b, 100, seed);
        for i in 0..b.len() {
            if b.origin[i].is_belt() {
                // Dwarfs have an upper mass limit too, so a heavy belt body
                // stays a planet.
                assert_eq!(
                    b.kind[i] == BodyKind::DwarfPlanet,
                    (BELT_DWARF_M_MIN..=DWARF_M_MAX).contains(&b.m[i])
                );
            }
        }
    }
}

#[test]
fn belt_body_distance_matches_primary() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    b.refresh_parent_cache();
    let prim = primary_index(&b).unwrap();
    let belt = (0..b.len()).find(|&i| b.origin[i].is_belt()).unwrap();
    assert!(find_parent(&b, belt).is_none());
    let o = orbit_info(&b, belt, prim);
    assert!(o.dist > 0.0);
    assert!(o.period.is_finite());
}

#[test]
fn belt_classification_uses_thresholds() {
    const { assert!(BELT_DWARF_M_MIN < DWARF_M_MAX) };
    assert_eq!(
        classify_small_body(BELT_DWARF_M_MIN * 0.999),
        BodyKind::Planet
    );
    assert_eq!(classify_small_body(BELT_DWARF_M_MIN), BodyKind::DwarfPlanet);
    assert_eq!(classify_small_body(DWARF_M_MAX), BodyKind::DwarfPlanet);
    assert_eq!(classify_small_body(DWARF_M_MAX * 1.001), BodyKind::Planet);
}

#[test]
fn classify_small_body_matches_belt_spawn() {
    let mut checked = 0usize;
    let mut check = |b: &Bodies, ctx: &str| {
        for i in 0..b.len() {
            if b.origin[i].is_belt() {
                checked += 1;
                assert_eq!(
                    b.kind[i],
                    classify_small_body(b.m[i]),
                    "{ctx} body {i} mass {}",
                    b.m[i]
                );
            }
        }
    };
    for seed in 0..100u32 {
        let mut b = Bodies::with_capacity(100);
        spawn_system(&mut b, 100, seed);
        check(&b, &format!("seed {seed}"));
    }
    // The Solar System preset has the heavy belt and the Trojans.
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    check(&b, "solar_system");
    assert!(checked > 0, "no belt bodies were checked");
}
