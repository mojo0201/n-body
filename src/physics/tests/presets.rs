//! Tests: preset library, named presets, names on bodies.

use super::super::*;

#[test]
fn preset_library_loads_all() {
    // The count and the full slug list are checked by
    // `preset_library_has_all_sixteen`; this test just proves the
    // library loads and contains the six originals.
    let presets = load_presets();
    assert!(presets.len() >= 6, "library shrunk: {}", presets.len());
    let slugs: Vec<&str> = presets.iter().map(|(s, _)| s.as_str()).collect();
    for s in [
        "solar_system",
        "trappist_1",
        "hot_jupiter",
        "vega",
        "hr_8799",
        "hd_40307",
    ] {
        assert!(slugs.contains(&s), "missing preset {}", s);
    }
}

#[test]
fn every_preset_spawns_and_runs() {
    for (slug, preset) in load_presets() {
        let mut b = Bodies::with_capacity(0);
        spawn_from_preset(&mut b, preset);
        assert!(!b.is_empty(), "preset {} spawned empty", slug);
        assert!(b.m[0] > 0.0);
        let n0 = b.len();
        for _ in 0..100 {
            advance(&mut b, DT);
        }
        for i in 0..b.len() {
            assert!(
                b.x[i].is_finite() && b.y[i].is_finite(),
                "preset {} body {} NaN",
                slug,
                i
            );
            assert!(b.m[i] > 0.0, "preset {} body {} dead", slug, i);
        }
        println!("{slug}: {n0} bodies, {} after 100 steps", b.len());
    }
}

#[test]
fn preset_by_slug_is_idempotent() {
    let p1 = load_preset_by_slug("solar_system").expect("solar_system");
    let p2 = load_preset_by_slug("solar_system").expect("solar_system");
    assert_eq!(p1.name, p2.name);
    assert_eq!(p1.planets.len(), p2.planets.len());
    assert!(load_preset_by_slug("nope").is_none());
}

#[test]
fn vega_has_no_planets_but_has_a_belt() {
    let p = load_preset_by_slug("vega").expect("vega");
    assert_eq!(p.planets.len(), 0);
    assert!(p.belts.is_some());
}

#[test]
fn trappist_1_has_seven_planets() {
    let p = load_preset_by_slug("trappist_1").expect("trappist_1");
    assert_eq!(p.planets.len(), 7);
    // The preset ships a small outer belt; assert it stays under 1e-3 M⊕.
    if let Some(b) = &p.belts {
        assert!(b.total_mass_earth < 1e-3, "TRAPPIST-1 belt too heavy");
    }
}

#[test]
fn hot_jupiter_has_outer_belt() {
    let p = load_preset_by_slug("hot_jupiter").expect("hot_jupiter");
    let b = p.belts.as_ref().expect("51 Peg belt");
    assert!(b.outer.is_some(), "51 Peg has no outer belt");
    // The inner belt exists but is a minor component (mass_frac ≤ 0.05).
    if let Some(inner) = &b.inner {
        assert!(
            inner.mass_frac <= 0.05,
            "inner mass_frac {}",
            inner.mass_frac
        );
    }
}

#[test]
fn preset_names_are_attached_to_bodies() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let earth = (0..b.len())
        .find(|&i| (b.m[i] - 1.0).abs() < 1e-9)
        .expect("Earth");
    assert_eq!(b.name[earth].as_deref(), Some("Earth"));
    assert_eq!(b.name[0].as_deref(), Some("Sun"));
    let belt = (0..b.len())
        .find(|&i| b.origin[i].is_belt())
        .expect("belt body");
    assert_eq!(b.name[belt], None);
}

#[test]
fn named_body_tag_uses_name() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let i = b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.name[i] = Some("Earth".to_string());
    let t = tag(&b, i);
    assert!(t.starts_with("Earth"), "tag was {:?}", t);
    assert!(t.contains("M⊕"), "tag was {:?}", t);
    let t0 = tag(&b, 0);
    assert!(t0.starts_with("#1 "), "tag was {:?}", t0);
}

#[test]
fn compact_moves_names_with_bodies() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(2.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.name[1] = Some("A".to_string());
    b.name[2] = Some("B".to_string());
    b.dead[1] = true;
    b.compact();
    assert_eq!(b.len(), 2);
    assert_eq!(b.name[1].as_deref(), Some("B"));
}

#[test]
fn merge_keeps_survivor_name() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(0.001, 0.0, 0.0, 0.0, 0.01, BodyKind::Planet);
    b.name[0] = Some("Earth".to_string());
    b.name[1] = Some("Moon".to_string());
    merge(&mut b, 0, 1);
    b.compact();
    assert_eq!(b.len(), 1);
    assert_eq!(b.name[0].as_deref(), Some("Earth"));
}

#[test]
fn event_log_uses_names() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(0.001, 0.0, 0.0, 0.0, 0.01, BodyKind::Planet);
    b.name[0] = Some("Earth".to_string());
    b.name[1] = Some("Moon".to_string());
    collision_pass(&mut b, DT, None);
    let txt = &b.events.back().unwrap().text;
    assert!(txt.contains("Earth"), "event {:?}", txt);
    assert!(txt.contains("Moon"), "event {:?}", txt);
}

#[test]
fn solar_preset_moons_spawn_with_parents() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    b.refresh_parent_cache();
    let moon_count = (1..b.len())
        .filter(|&i| {
            matches!(
                find_parent(&b, i),
                Some(p) if matches!(b.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet)
            )
        })
        .count();
    assert!(
        moon_count >= 8,
        "expected at least 8 Solar-System moons, got {}",
        moon_count
    );
}

#[test]
fn solar_moon_periods_match_real() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    b.refresh_parent_cache();
    let expected = [
        (0.0123_f64, 27.32), // Moon
        (0.015, 1.769),      // Io
        (0.008, 3.551),      // Europa
        (0.025, 7.155),      // Ganymede
        (0.018, 16.689),     // Callisto
        (0.0225, 15.945),    // Titan
        (0.00359, 5.877),    // Triton
        (0.000254, 6.387),   // Charon
    ];
    for &(m_target, real) in &expected {
        // Tolerance 1e-5 (not 1e-4): Ceres m=0.000159 is within 1e-4 of
        // Charon m=0.000254, so 1e-4 would pick Ceres (which orbits the
        // Sun) and fail the "parent is a planet" check below.
        let i = (1..b.len())
            .find(|&i| b.origin[i] == BodyOrigin::Spawned && (b.m[i] - m_target).abs() < 1e-5)
            .unwrap_or_else(|| panic!("no spawned body of mass {} M⊕", m_target));
        let p =
            find_parent(&b, i).unwrap_or_else(|| panic!("moon of mass {} has no parent", m_target));
        assert!(matches!(
            b.kind[p],
            BodyKind::Planet | BodyKind::DwarfPlanet
        ));
        let o = orbit_info(&b, i, p);
        assert!(o.bound);
        let p_days = o.period * time_unit_s() / 86_400.0;
        assert!(
            (p_days / real - 1.0).abs() < 0.05,
            "moon of mass {}: {:.3} d vs {:.3} d",
            m_target,
            p_days,
            real
        );
    }
}

#[test]
fn legacy_presets_without_moons_still_load() {
    for (slug, preset) in load_presets() {
        if slug != "solar_system" {
            for p in &preset.planets {
                assert!(
                    p.moons.is_empty(),
                    "{}/{} unexpectedly has moons",
                    slug,
                    p.name
                );
            }
        }
    }
}

#[test]
fn moon_schema_round_trips() {
    let json = r#"{
        "name": "Fixture",
        "star": { "mass_earth": 332946.0, "age_gyr": 4.6, "metallicity": 0.0 },
        "planets": [
            { "name": "p", "a_au": 1.0, "e": 0.0, "m_earth": 1.0,
              "moons": [
                  { "name": "m1", "a_au": 0.00257, "e": 0.0549, "m_earth": 0.0123,
                    "density_gcc": 3.34 }
              ] }
        ]
    }"#;
    let preset: SystemPreset = serde_json::from_str(json).expect("parse");
    assert_eq!(preset.planets[0].moons.len(), 1);
    assert_eq!(preset.planets[0].moons[0].name, "m1");
}

#[test]
fn preset_library_has_all_sixteen() {
    let presets = load_presets();
    assert_eq!(presets.len(), 16, "library length");
    let slugs: Vec<&str> = presets.iter().map(|(s, _)| s.as_str()).collect();
    for s in [
        "solar_system",
        "trappist_1",
        "hot_jupiter",
        "vega",
        "hr_8799",
        "hd_40307",
        "gamma_cephei",
        "kepler_16",
        "algol",
        "luhman_16",
        "psr_b1257_plus_12",
        "wd_1145_plus_017",
        "wasp_12b",
        "hd_80606",
        "cygnus_x1",
        "kepler_90",
    ] {
        assert!(slugs.contains(&s), "missing preset {s}");
    }
}

#[test]
fn gamma_cephei_is_s_type() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("gamma_cephei").unwrap());
    assert_eq!(b.kind[0], BodyKind::Star);
    assert_eq!(b.kind[1], BodyKind::Star);
    let planet = (0..b.len())
        .find(|&i| b.name[i].as_deref() == Some("γ Cep b"))
        .unwrap();
    assert_eq!(
        find_parent(&b, planet),
        Some(0),
        "S-type planet must orbit the primary"
    );
}

#[test]
fn kepler_16_is_p_type() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("kepler_16").unwrap());
    let planet = (0..b.len())
        .find(|&i| b.name[i].as_deref() == Some("Kepler-16 b"))
        .unwrap();
    let bary_x = (b.m[0] * b.x[0] + b.m[1] * b.x[1]) / (b.m[0] + b.m[1]);
    let bary_y = (b.m[0] * b.y[0] + b.m[1] * b.y[1]) / (b.m[0] + b.m[1]);
    let sep = (b.x[1] - b.x[0]).hypot(b.y[1] - b.y[0]);
    let d = (b.x[planet] - bary_x).hypot(b.y[planet] - bary_y);
    assert!(d > 3.0 * sep, "P-type: d {} vs 3×sep {}", d, 3.0 * sep);
}

#[test]
fn algol_is_a_triple() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("algol").unwrap());
    let n_stars = (0..b.len())
        .filter(|&i| b.kind[i] == BodyKind::Star)
        .count();
    assert_eq!(n_stars, 3, "Algol is a triple");
}

#[test]
fn psr_b1257_host_is_a_neutron_star() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("psr_b1257_plus_12").unwrap());
    assert_eq!(b.kind[0], BodyKind::NeutronStar);
    let n = (1..b.len())
        .filter(|&i| b.kind[i] == BodyKind::Planet)
        .count();
    assert_eq!(n, 3);
}

#[test]
fn wd_1145_host_is_a_white_dwarf() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("wd_1145_plus_017").unwrap());
    assert_eq!(b.kind[0], BodyKind::WhiteDwarf);
    let n_belt = (0..b.len()).filter(|&i| b.origin[i].is_belt()).count();
    assert!(n_belt > 0, "no belt bodies in WD 1145+017");
    let belt_mass: f64 = (0..b.len())
        .filter(|&i| b.origin[i].is_belt())
        .map(|i| b.m[i])
        .sum();
    assert!(belt_mass > 0.0, "belt mass {} is zero", belt_mass);
}

#[test]
fn luhman_16_is_a_bd_binary() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("luhman_16").unwrap());
    assert_eq!(b.kind[0], BodyKind::BrownDwarf);
    assert_eq!(b.kind[1], BodyKind::BrownDwarf);
}

#[test]
fn cygnus_x1_host_is_a_black_hole() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("cygnus_x1").unwrap());
    assert_eq!(b.kind[0], BodyKind::Star);
    assert_eq!(b.kind[1], BodyKind::BlackHole);
}

#[test]
fn hd_80606_b_has_extreme_eccentricity() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("hd_80606").unwrap());
    let planet = (0..b.len())
        .find(|&i| b.name[i].as_deref() == Some("HD 80606 b"))
        .expect("HD 80606 b should be in the preset");
    assert_ne!(planet, 1, "reading the companion star HD 80607");

    let o = orbit_info(&b, planet, 0);
    assert!(o.bound, "HD 80606 b should be bound");
    assert!(o.ecc > 0.9, "HD 80606 b e = {}", o.ecc);
    let peri_au = o.peri / au_in_gm();
    assert!(
        (peri_au - 0.031).abs() < 0.01,
        "peri = {} AU (expected ~0.031)",
        peri_au
    );
}

#[test]
fn kepler_90_has_eight_planets() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("kepler_90").unwrap());
    let n = (1..b.len())
        .filter(|&i| b.kind[i] == BodyKind::Planet && b.origin[i] == BodyOrigin::Spawned)
        .count();
    assert_eq!(n, 8);
}

#[test]
fn legacy_presets_parse_without_companions() {
    for (slug, preset) in load_presets() {
        if matches!(
            slug.as_str(),
            "solar_system" | "trappist_1" | "hot_jupiter" | "vega" | "hr_8799" | "hd_40307"
        ) {
            assert!(preset.companions.is_empty(), "{slug} grew companions");
            assert!(preset.star.kind.is_none(), "{slug} grew a kind");
        }
    }
}

// Append to src/physics/tests/presets.rs (uses `use super::super::*;` already there).

/// Index of the body with this preset name. Names survive `compact()`,
/// indices do not, so tests that run many steps look the body up each time.
fn find_named(b: &Bodies, name: &str) -> Option<usize> {
    (0..b.len()).find(|&i| b.name[i].as_deref() == Some(name))
}

#[test]
fn moon_stays_bound_to_earth() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let a0 = 0.0025695 * au_in_gm();
    for step in 0..10_000 {
        advance(&mut b, DT);
        let e = find_named(&b, "Earth").expect("Earth vanished");
        let m = find_named(&b, "Moon").unwrap_or_else(|| panic!("Moon vanished at step {step}"));
        let d = (b.x[m] - b.x[e]).hypot(b.y[m] - b.y[e]);
        assert!(
            d >= 0.5 * a0 && d <= 2.0 * a0,
            "step {step}: Moon-Earth distance {d:.4} Gm, expected [{:.4}, {:.4}]",
            0.5 * a0,
            2.0 * a0
        );
    }
}

#[test]
fn moon_parent_stable() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    for step in 0..10_000 {
        advance(&mut b, DT);
        let e = find_named(&b, "Earth").expect("Earth vanished");
        let m = find_named(&b, "Moon").unwrap_or_else(|| panic!("Moon vanished at step {step}"));
        assert_eq!(
            find_parent(&b, m),
            Some(e),
            "step {step}: Moon parent changed"
        );
    }
}

/// Diagnostic only. Run with:
///   cargo test --release --lib moon_diagnose -- --ignored --nocapture
/// Delete after Task 1 is closed.
#[test]
#[ignore]
fn moon_diagnose() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let a0 = 0.0025695 * au_in_gm();
    for step in 0..10_000 {
        if step % 250 == 0 {
            let e = find_named(&b, "Earth").unwrap();
            let m = find_named(&b, "Moon").unwrap();
            let d = (b.x[m] - b.x[e]).hypot(b.y[m] - b.y[e]);
            let o = orbit_info(&b, m, e);
            println!(
                "step {step:5} n {} d/a0 {:.3} rung {} dtl/DT {:.3} bound {} ecc {:.3} kmax {}",
                b.len(),
                d / a0,
                b.rung[m],
                b.dtl[m] / DT,
                o.bound,
                o.ecc,
                b.stats.max_rung
            );
        }
        advance(&mut b, DT);
    }
}
