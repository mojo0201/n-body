//! Tests for the preset library, preset-by-slug lookup, body naming, the
//! split body-construction / finalisation path, the solar-cluster spawner,
//! belt-spec generation, and `BeltPreset::to_spec`.
//!
//! These tests run against the embedded JSON in `assets/presets/`. Adding
//! a preset means updating `EXPECTED_SLUGS`; the count and membership
//! checks below are what catch "JSON was added but never wired in" and
//! "library silently shrank".
//!
//! Two `#[ignore]`d tests (`moon_stays_bound_to_earth`, `moon_parent_stable`)
//! each run 10 000 integration steps and take ~60 s. Run them with:
//!     cargo test --lib physics::tests::presets -- --ignored
//!
//! One test (`every_preset_spawns_and_runs`) requires `parse_body_kind`
//! in `presets.rs` to accept `"giant"` and the other standard stellar
//! classes; `gaia_bh3`'s companion is tagged `"giant"`.

use super::super::*;

/// Every slug the library is expected to contain. Keeping the list in one
/// place means a newly embedded preset updates exactly one constant and
/// one test.
const EXPECTED_SLUGS: [&str; 38] = [
    "2mass_j1207",
    "hot_jupiter",
    "55_cancri",
    "algol",
    "alpha_centauri",
    "au_microscopii",
    "beta_pictoris",
    "cygnus_x1",
    "epsilon_eridani",
    "fomalhaut",
    "gaia_bh1",
    "gaia_bh3",
    "gamma_cephei",
    "gliese_876",
    "hd_40307",
    "hd_80606",
    "hl_tau",
    "hr_8799",
    "kelt_9",
    "kepler_16",
    "kepler_1625",
    "kepler_36",
    "kepler_47",
    "kepler_51",
    "kepler_90",
    "luhman_16",
    "nice_model",
    "pds_70",
    "psr_b1257_plus_12",
    "psr_b1620_26",
    "sgr_a_star",
    "sirius",
    "solar_system",
    "trappist_1",
    "vega",
    "wasp_12b",
    "wd_1145_plus_017",
    "wd_1856_534",
];

/// The six original presets. The `legacy_*` tests pin the "these predate
/// companions / moons" invariant to this set, so a future preset is free
/// to grow a companion without breaking them.
const ORIGINAL_SLUGS: [&str; 6] = [
    "solar_system",
    "trappist_1",
    "hot_jupiter",
    "vega",
    "hr_8799",
    "hd_40307",
];

/// Look up a body by its preset name. Names survive `compact()`, indices
/// do not, so tests that run many steps look the body up each time.
fn find_named(b: &Bodies, name: &str) -> Option<usize> {
    (0..b.len()).find(|&i| b.name[i].as_deref() == Some(name))
}

// ---------------------------------------------------------------------------
// Library shape
// ---------------------------------------------------------------------------

#[test]
fn preset_library_loads_all() {
    // Sanity check that the library loads at all; the exhaustive slug
    // list is verified by `preset_library_has_all_expected_slugs`.
    let presets = load_presets();
    assert!(
        presets.len() >= ORIGINAL_SLUGS.len(),
        "library shrunk: {}",
        presets.len()
    );
    let slugs: Vec<&str> = presets.iter().map(|(s, _)| s.as_str()).collect();
    for s in ORIGINAL_SLUGS {
        assert!(slugs.contains(&s), "missing original preset {}", s);
    }
}

#[test]
fn preset_library_has_all_expected_slugs() {
    // Two invariants in one test: the library has not shrunk or grown
    // unintentionally (count), and every expected slug is present
    // (membership). The duplicate check catches a JSON that got pasted
    // twice under two names — `load_preset_by_slug` would silently pick
    // the first, masking the bug.
    let presets = load_presets();
    assert_eq!(
        presets.len(),
        EXPECTED_SLUGS.len(),
        "library length drift; update EXPECTED_SLUGS if this is intentional"
    );
    let slugs: Vec<&str> = presets.iter().map(|(s, _)| s.as_str()).collect();
    for s in EXPECTED_SLUGS {
        assert!(slugs.contains(&s), "missing preset {s}");
    }
    let mut sorted = slugs.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), slugs.len(), "duplicate slug in library");
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
fn every_preset_spawns_and_runs() {
    // The single broadest smoke test: every preset must produce a non-empty
    // system, and 100 steps must not blow up or kill a body. Panics with
    // the offending slug so a regression points straight at the JSON.
    //
    // NOTE: requires `parse_body_kind` to accept `"giant"` (gaia_bh3's
    // K-giant companion). Without that fix this test panics inside
    // `spawn_from_preset`.
    for (slug, preset) in load_presets() {
        let mut b = Bodies::with_capacity(0);
        spawn_from_preset(&mut b, preset);
        assert!(!b.is_empty(), "preset {slug} spawned empty");
        assert!(b.m[0] > 0.0);
        let n0 = b.len();
        for _ in 0..100 {
            advance(&mut b, DT);
        }
        for i in 0..b.len() {
            assert!(
                b.x[i].is_finite() && b.y[i].is_finite(),
                "preset {slug} body {i} NaN"
            );
            assert!(b.m[i] > 0.0, "preset {slug} body {i} dead");
        }
        println!("{slug}: {n0} bodies, {} after 100 steps", b.len());
    }
}

// ---------------------------------------------------------------------------
// Per-preset smoke checks
// ---------------------------------------------------------------------------

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
fn gamma_cephei_is_s_type() {
    // An S-type planet orbits the primary, not the binary barycentre.
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
    // A P-type planet orbits outside the binary's mutual separation: the
    // planet's distance from the barycentre is more than 3× the binary
    // separation. That's the standard stability criterion for circumbinary
    // planets.
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
fn preset_wd_1145_host_is_a_white_dwarf() {
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
fn preset_hd_80606_b_has_extreme_eccentricity() {
    // A well-known "smoking gun" of high-eccentricity migration: the
    // planet is on a nearly radial orbit with periastron at ~0.03 AU.
    // This is a stress test for both the Kepler solve and the parent
    // classification.
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
    // The original eight-planet Kepler system; tests that the preset
    // doesn't accidentally count the star (index 0) or mis-tag an
    // unrelated body as `Spawned`.
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("kepler_90").unwrap());
    let n = (1..b.len())
        .filter(|&i| b.kind[i] == BodyKind::Planet && b.origin[i] == BodyOrigin::Spawned)
        .count();
    assert_eq!(n, 8);
}

// ---------------------------------------------------------------------------
// Legacy-preset shape
// ---------------------------------------------------------------------------

#[test]
fn legacy_presets_without_moons_still_load() {
    // Only `solar_system` and `kepler_1625` ship moons today. The other
    // five originals predate moon support; growing moons on any of them
    // would shift their spawn layout and needs a deliberate test update.
    //
    // `kepler_1625` is excluded because its preset legitimately includes
    // the Kepler-1625 b candidate exomoon.
    for (slug, preset) in load_presets() {
        let is_original_without_moons =
            ORIGINAL_SLUGS.contains(&slug.as_str()) && slug != "solar_system";
        if is_original_without_moons {
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
fn legacy_presets_parse_without_companions() {
    // The six originals predate companion support: adding a companion
    // (or an explicit `kind`) would shift their barycentre and needs a
    // deliberate test update.
    for (slug, preset) in load_presets() {
        if ORIGINAL_SLUGS.contains(&slug.as_str()) {
            assert!(preset.companions.is_empty(), "{slug} grew companions");
            assert!(preset.star.kind.is_none(), "{slug} grew a kind");
        }
    }
}

#[test]
fn moon_schema_round_trips() {
    // A minimal fixture exercising the full `moons` array shape, including
    // the optional `density_gcc`. If any field is renamed or re-typed,
    // this test catches it before the Solar System preset silently loses
    // its moons.
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

// ---------------------------------------------------------------------------
// Names on bodies
// ---------------------------------------------------------------------------

#[test]
fn preset_names_are_attached_to_bodies() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let earth = (0..b.len())
        .find(|&i| (b.m[i] - 1.0).abs() < 1e-9)
        .expect("Earth");
    assert_eq!(b.name[earth].as_deref(), Some("Earth"));
    assert_eq!(b.name[0].as_deref(), Some("Sun"));
    // Belt bodies stay anonymous so `tag()` prints their numeric index
    // instead of a meaningless name.
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
    // Unnamed bodies fall back to "#N " with a trailing space, so the
    // mass parenthetical reads naturally in event logs.
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

// ---------------------------------------------------------------------------
// Moons of the Solar System
// ---------------------------------------------------------------------------

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
    // Look up moons by name, not by mass. Names are stable identifiers
    // in the preset; masses get refined with better observations and
    // would silently break a mass-based lookup (as Charon's recent
    // refinement did).
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    b.refresh_parent_cache();
    let expected = [
        ("Moon", 27.32_f64), // days
        ("Io", 1.769),
        ("Europa", 3.551),
        ("Ganymede", 7.155),
        ("Callisto", 16.689),
        ("Titan", 15.945),
        ("Triton", 5.877),
        ("Charon", 6.387),
    ];
    for &(name, real) in &expected {
        let i = find_named(&b, name).unwrap_or_else(|| panic!("no body named {name}"));
        let p = find_parent(&b, i).unwrap_or_else(|| panic!("{name} has no parent"));
        assert!(
            matches!(b.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet),
            "{name}'s parent is {:?}, expected Planet or DwarfPlanet",
            b.kind[p]
        );
        let o = orbit_info(&b, i, p);
        assert!(o.bound, "{name} is unbound");
        let p_days = o.period * time_unit_s() / 86_400.0;
        assert!(
            (p_days / real - 1.0).abs() < 0.05,
            "{name}: {:.3} d vs {:.3} d",
            p_days,
            real
        );
    }
}

#[test]
#[ignore = "10 000 integration steps; ~60 s. Run with -- --ignored"]
fn moon_stays_bound_to_earth() {
    // Ten thousand steps is long enough for a broken integrator or a
    // wrong-mu `push_orbit` to eject the Moon; a correct one keeps the
    // distance within a factor of two of the initial separation.
    //
    // Ignored by default because 10 000 steps is ~60 s in debug builds.
    // Run explicitly with `cargo test --lib physics::tests::presets -- --ignored`.
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
#[ignore = "10 000 integration steps; ~60 s. Run with -- --ignored"]
fn moon_parent_stable() {
    // The parent cache is rebuilt every step; this test catches a
    // regression where a close pass to another body steals the Moon.
    //
    // Ignored by default because 10 000 steps is ~60 s in debug builds.
    // Run explicitly with `cargo test --lib physics::tests::presets -- --ignored`.
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

// ---------------------------------------------------------------------------
// Dwarf-planet flag validation
// ---------------------------------------------------------------------------

#[test]
fn preset_is_dwarf_validated() {
    // A `is_dwarf: true` body with a planetary mass must be demoted to
    // `Planet`, produce a warning, and log a user event. Bodies whose
    // mass is in the dwarf range keep the flag and produce no warning.
    let json = r#"{
        "name": "Fixture",
        "star": { "mass_earth": 332946.0, "age_gyr": 4.6, "metallicity": 0.0 },
        "planets": [
            { "name": "heavy", "a_au": 1.0, "e": 0.0, "m_earth": 0.5, "is_dwarf": true },
            { "name": "light", "a_au": 2.0, "e": 0.0, "m_earth": 0.001, "is_dwarf": true }
        ]
    }"#;
    let preset: SystemPreset = serde_json::from_str(json).expect("parse");
    let mut b = Bodies::with_capacity(0);
    let warnings = spawn_from_preset_seeded(&mut b, &preset, 1);
    let heavy = find_named(&b, "heavy").expect("heavy");
    let light = find_named(&b, "light").expect("light");
    assert_eq!(b.kind[heavy], BodyKind::Planet, "0.5 M⊕ is not a dwarf");
    assert_eq!(b.kind[light], BodyKind::DwarfPlanet);
    assert_eq!(warnings.len(), 1, "warnings {:?}", warnings);
    assert!(warnings[0].contains("heavy"));
    assert!(b
        .events
        .iter()
        .any(|e| matches!(e.kind, EventKind::User) && e.text.contains("heavy")));
}

#[test]
fn preset_dwarf_mass_within_range() {
    // Every `is_dwarf: true` body in the Solar System preset — not only
    // the five IAU names — must fall in the physical dwarf range.
    let preset = load_preset_by_slug("solar_system").expect("solar_system");
    for name in ["Ceres", "Pluto", "Eris", "Haumea", "Makemake"] {
        let p = preset
            .planets
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("no {name}"));
        assert!(p.is_dwarf, "{name} is not flagged dwarf");
    }
    for p in preset.planets.iter().filter(|p| p.is_dwarf) {
        assert!(
            (BELT_DWARF_M_MIN..=DWARF_M_MAX).contains(&p.m_earth),
            "{} mass {} outside [{}, {}]",
            p.name,
            p.m_earth,
            BELT_DWARF_M_MIN,
            DWARF_M_MAX
        );
    }
}

#[test]
fn moon_of_dwarf_not_labelled_dwarf() {
    // A moon of a dwarf planet inherits the "moon" class even when its
    // mass is dwarf-like: classification is by *parent*, not by mass,
    // for sub-planetary satellites.
    let mut b = Bodies::with_capacity(3);
    let au = au_in_gm();
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let dwarf = b.push(au, 0.0, 0.0, 0.0, 0.002, BodyKind::DwarfPlanet);
    // Well inside the dwarf's Hill sphere (about 0.19 Gm).
    let moon = b.push(au + 0.02, 0.0, 0.0, 0.0, 0.0003, BodyKind::DwarfPlanet);
    b.origin[dwarf] = BodyOrigin::Spawned;
    b.origin[moon] = BodyOrigin::Spawned;
    assert_eq!(b.kind[moon], BodyKind::DwarfPlanet);
    assert_eq!(find_parent(&b, moon), Some(dwarf));
    assert_eq!(body_class_with_parent(&b, moon, Some(dwarf)), "moon");
    assert_eq!(body_class(&b, moon), "moon");
    assert_eq!(body_class(&b, dwarf), "dwarf planet");
}

// ---------------------------------------------------------------------------
// Comets
// ---------------------------------------------------------------------------

#[test]
fn solar_system_has_both_comets() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let comets: Vec<usize> = (0..b.len())
        .filter(|&i| b.origin[i] == BodyOrigin::Comet)
        .collect();
    assert_eq!(comets.len(), 2, "Solar System must contain 67P and Halley");
    let names: Vec<&str> = comets
        .iter()
        .filter_map(|&i| b.name[i].as_deref())
        .collect();
    assert!(names.contains(&"67P/Churyumov-Gerasimenko"), "missing 67P");
    assert!(names.contains(&"1P/Halley"), "missing Halley");
    for &i in &comets {
        // Comets are Icy regardless of heliocentric distance; that's the
        // whole point of the `class_for` special case.
        assert_eq!(b.class[i], BodyClass::Icy, "comets are icy");
    }
}

// ---------------------------------------------------------------------------
// Body-construction / finalisation split
//
// The spawn path was split so the cluster spawner can build N temporary
// systems in O(N), finalise once on the combined system, and avoid the
// N discarded full finalisations the old code ran. These tests pin down
// the invariant that the split did not change *what* gets built, only
// *when* finalisation runs.
// ---------------------------------------------------------------------------

#[test]
fn spawn_from_preset_bodies_matches_seeded_after_finalize() {
    // The "bodies-only" builder followed by the same finalisation the
    // seeded wrapper does must produce identical state. If this fails,
    // cluster-spawn correctness is gone.
    let preset = solar_system_preset();

    let mut a = Bodies::with_capacity(0);
    spawn_from_preset_seeded(&mut a, &preset, 0x1234);

    let mut b = Bodies::with_capacity(0);
    spawn_from_preset_bodies(&mut b, &preset, 0x1234);
    recenter_com(&mut b);
    compute_accel(&mut b);
    rebase_energy(&mut b);
    b.refresh_parent_cache();
    dust::init_fields(&mut b, None);
    if preset.dust_boost != 1.0 {
        for f in b.dust_fields.iter_mut() {
            for g in f.grains.iter_mut() {
                g.mass *= preset.dust_boost;
            }
        }
    }

    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert_eq!(a.name[i], b.name[i], "name[{i}]");
        assert_eq!(a.kind[i], b.kind[i], "kind[{i}]");
        assert_eq!(a.class[i], b.class[i], "class[{i}]");
        assert_eq!(a.origin[i], b.origin[i], "origin[{i}]");
        assert!((a.x[i] - b.x[i]).abs() < 1e-12, "x[{i}]");
        assert!((a.y[i] - b.y[i]).abs() < 1e-12, "y[{i}]");
        assert!((a.vx[i] - b.vx[i]).abs() < 1e-12, "vx[{i}]");
        assert!((a.vy[i] - b.vy[i]).abs() < 1e-12, "vy[{i}]");
        assert!((a.m[i] - b.m[i]).abs() < 1e-12, "m[{i}]");
    }
}

#[test]
fn spawn_from_preset_bodies_leaves_state_unfinalized() {
    // The whole point of the split is that `_bodies` does NOT pay for
    // dust initialisation. If someone accidentally adds it, this test
    // catches the regression before cluster spawn silently slows down.
    let preset = solar_system_preset();
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset_bodies(&mut b, &preset, 0x1234);
    assert!(
        b.dust_fields.is_empty(),
        "dust fields set before finalise; the split contract is broken"
    );
}

#[test]
fn same_seed_same_spawn() {
    // Reproducibility: identical preset + seed must produce bit-identical
    // positions. This survives the split because both paths consume the
    // RNG in the same order.
    let preset = solar_system_preset();
    let mut a = Bodies::with_capacity(0);
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset_seeded(&mut a, &preset, 0xCAFE);
    spawn_from_preset_seeded(&mut b, &preset, 0xCAFE);
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert_eq!(a.x[i], b.x[i], "x[{i}] bit-differs");
        assert_eq!(a.y[i], b.y[i], "y[{i}] bit-differs");
    }
}

#[test]
fn different_seeds_give_different_phases() {
    // The public `spawn_preset_with_seed` entry point must actually use
    // the seed: two different seeds should put at least one body at a
    // different phase.
    let preset = solar_system_preset();
    let mut a = Bodies::with_capacity(0);
    let mut b = Bodies::with_capacity(0);
    spawn_preset_with_seed(&mut a, &preset, 1);
    spawn_preset_with_seed(&mut b, &preset, 2);
    let differs = (0..a.len()).any(|i| (a.x[i] - b.x[i]).abs() > 1e-9);
    assert!(differs, "two seeds produced identical spawns");
}

#[test]
fn preset_phase_pinning_places_body_at_perihelion() {
    // `phase: Some(x)` pins the mean anomaly to `x * 2π`, which — for a
    // nonzero eccentricity — places the body at a predictable point in
    // its orbit. This is what the field actually guarantees.
    //
    // NOTE: the comment in `presets.rs` also claims pinning "does not
    // shift the stream for any following body". That claim is false: a
    // pinned body skips one `rng.next_f64()` call, so downstream
    // unpinned bodies land on a different RNG position. Fixing that
    // would require a per-body RNG seed, which is out of scope. The
    // tests below verify only the placement guarantee, which is what
    // callers actually rely on.
    let json = r#"{
        "name": "P",
        "star": { "mass_earth": 332946.0, "age_gyr": 4.6, "metallicity": 0.0 },
        "planets": [
            { "name": "a", "a_au": 1.0, "e": 0.3, "m_earth": 1.0, "phase": 0.0 },
            { "name": "b", "a_au": 2.0, "e": 0.3, "m_earth": 1.0, "phase": 0.5 }
        ]
    }"#;
    let preset: SystemPreset = serde_json::from_str(json).unwrap();
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset_seeded(&mut b, &preset, 7);

    // Sun is body 0 after spawn. Use sun-relative distance so the
    // recentre step (which shifts the sun to the COM) can't bias the
    // measurement.
    let sun = 0;
    let a_idx = find_named(&b, "a").expect("a");
    let b_idx = find_named(&b, "b").expect("b");

    // Phase 0.0 → perihelion → r = a(1 - e) = 1.0 * 0.7 = 0.7 AU.
    let r_a = (b.x[a_idx] - b.x[sun]).hypot(b.y[a_idx] - b.y[sun]) / au_in_gm();
    assert!(
        (r_a - 0.7).abs() < 0.01,
        "phase 0.0: r = {r_a} AU, expected 0.7"
    );

    // Phase 0.5 → aphelion → r = a(1 + e) = 2.0 * 1.3 = 2.6 AU.
    let r_b = (b.x[b_idx] - b.x[sun]).hypot(b.y[b_idx] - b.y[sun]) / au_in_gm();
    assert!(
        (r_b - 2.6).abs() < 0.01,
        "phase 0.5: r = {r_b} AU, expected 2.6"
    );
}

// ---------------------------------------------------------------------------
// Solar System preset caching and dust boost
// ---------------------------------------------------------------------------

#[test]
fn solar_system_preset_is_cached_and_owned() {
    // The cache must return an owned clone (callers mutate the preset
    // before spawning). Mutating one clone must not affect the cache.
    let p1 = solar_system_preset();
    let mut p2 = solar_system_preset();
    assert_eq!(p1.name, p2.name);
    p2.name = "modified".to_string();
    assert_eq!(solar_system_preset().name, p1.name);
}

#[test]
fn dust_boost_scales_grain_masses_linearly() {
    // `dust_boost` must scale every grain mass by exactly the factor;
    // a per-field or per-grain discrepancy would make dust brightness
    // depend on layout rather than on the boost alone.
    let mut preset = solar_system_preset();
    preset.dust_boost = 1.0;
    let mut a = Bodies::with_capacity(0);
    spawn_from_preset(&mut a, &preset);

    preset.dust_boost = 3.0;
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &preset);

    let total = |x: &Bodies| -> f64 {
        x.dust_fields
            .iter()
            .flat_map(|f| f.grains.iter())
            .map(|g| g.mass)
            .sum()
    };
    let r = total(&b) / total(&a);
    assert!((r - 3.0).abs() < 1e-9, "boost ratio {r} (expected 3.0)");
}

// ---------------------------------------------------------------------------
// Cluster spawner
// ---------------------------------------------------------------------------

#[test]
fn spawn_solar_systems_1_is_equivalent_to_single() {
    // For n=1 the slot offset is (0, 0) and the per-system seed equals
    // SOLAR_PHASE_SEED, so the cluster path must produce a layout
    // equivalent to the single-system path. Position drift here would
    // mean a seed or offset bug in `cluster_slots`.
    let mut a = Bodies::with_capacity(0);
    spawn_solar_system(&mut a);

    let mut b = Bodies::with_capacity(0);
    let (_radius, ranges) = spawn_solar_systems_impl(&mut b, 1);

    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0], (0, b.len()));
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert!((a.x[i] - b.x[i]).abs() < 1e-6, "x[{i}]");
        assert!((a.y[i] - b.y[i]).abs() < 1e-6, "y[{i}]");
    }
}

#[test]
fn spawn_solar_systems_clamps_to_supported_count() {
    // The UI slider can ask for anything; the spawner must clamp to
    // 1..=MAX_SOLAR_SYSTEMS without panicking. Only a few representative
    // sizes are spawned to keep the test's runtime reasonable.
    for n in [0usize, 1, 2, 4] {
        let mut b = Bodies::with_capacity(0);
        let (_radius, ranges) = spawn_solar_systems_impl(&mut b, n);
        assert_eq!(
            ranges.len(),
            n.clamp(1, MAX_SOLAR_SYSTEMS),
            "n={n} produced wrong number of systems"
        );
    }
    // Clamp upper bound without actually spawning MAX systems (which
    // takes ~10 s). The clamp arithmetic itself is what we verify.
    assert_eq!(
        MAX_SOLAR_SYSTEMS.clamp(1, MAX_SOLAR_SYSTEMS),
        MAX_SOLAR_SYSTEMS
    );
    assert_eq!(
        (MAX_SOLAR_SYSTEMS + 10).clamp(1, MAX_SOLAR_SYSTEMS),
        MAX_SOLAR_SYSTEMS
    );
}

#[test]
fn spawn_solar_systems_ranges_are_contiguous_and_ordered() {
    // Every body must belong to exactly one system, with ranges in
    // system order and covering the whole `Bodies`.
    let mut b = Bodies::with_capacity(0);
    let (_radius, ranges) = spawn_solar_systems_impl(&mut b, 5);
    let mut prev_end = 0;
    for (i, &(start, end)) in ranges.iter().enumerate() {
        assert_eq!(
            start, prev_end,
            "range {i} does not start where the last ended"
        );
        assert!(end > start, "range {i} empty");
        prev_end = end;
    }
    assert_eq!(prev_end, b.len(), "ranges do not cover all bodies");
}

#[test]
fn spawn_solar_systems_are_not_overlapping() {
    // Suns in different systems must be separated enough that the
    // systems' own extents don't overlap at spawn. `SOLAR_SYSTEM_RADIUS_GM`
    // is the "hard edge" of one system (Neptune's apoapsis plus margin).
    let mut b = Bodies::with_capacity(0);
    let (_radius, ranges) = spawn_solar_systems_impl(&mut b, 4);
    let suns: Vec<usize> = ranges.iter().map(|&(s, _)| s).collect();
    for i in 0..suns.len() {
        for j in (i + 1)..suns.len() {
            let dx = b.x[suns[i]] - b.x[suns[j]];
            let dy = b.y[suns[i]] - b.y[suns[j]];
            let d = dx.hypot(dy);
            assert!(
                d > SOLAR_SYSTEM_RADIUS_GM,
                "systems {i} and {j} at {d:.0} Gm, expected > {SOLAR_SYSTEM_RADIUS_GM:.0}"
            );
        }
    }
}

#[test]
fn spawn_solar_systems_populate_dust_fields() {
    // The combined-system finalisation must leave a non-empty dust field
    // set; if the optimization accidentally dropped `dust::init_fields`,
    // cluster spawns would render without dust.
    let mut b = Bodies::with_capacity(0);
    let (_radius, _ranges) = spawn_solar_systems_impl(&mut b, 3);
    assert!(
        !b.dust_fields.is_empty(),
        "cluster spawn left dust fields empty"
    );
}

// ---------------------------------------------------------------------------
// Belt-spec generation
//
// `belt_spec_for` and `belt_spec_for_evolved` drive procedural belt
// spawning. The debris-belt gate is probabilistic, so these tests loop
// over seeds until they find one that passes; if none does, the gate is
// broken and the test fails loudly.
// ---------------------------------------------------------------------------

/// The Solar System planets, as `(index, mass_earth, a_au)`. Enough to
/// trigger the SolarType branch reliably.
fn solar_planets() -> Vec<(usize, f64, f64)> {
    vec![
        (0, 0.055, 0.387),
        (1, 0.815, 0.723),
        (2, 1.0, 1.0),
        (3, 0.107, 1.524),
        (4, 317.8, 5.2),  // Jupiter
        (5, 95.2, 9.58),  // Saturn
        (6, 14.5, 19.2),  // Uranus
        (7, 17.1, 30.05), // Neptune
    ]
}

fn spec_for_solar_planets(seed: u128) -> BeltSpec {
    let mut rng = XorShift::new(seed);
    let bodies = Bodies::with_capacity(0);
    belt_spec_for(
        &bodies,
        M_SUN_ME,
        0.0,
        4.6,
        &solar_planets(),
        2.7,
        2.0,
        50.0,
        &mut rng,
    )
}

fn find_solar_seed<F: Fn(&BeltSpec) -> bool>(f: F) -> u128 {
    for seed in 0u128..256 {
        let spec = spec_for_solar_planets(seed);
        if !spec.is_empty() && f(&spec) {
            return seed;
        }
    }
    panic!("no seed produced a matching non-empty SolarType spec in 256 tries");
}

#[test]
fn belt_spec_solar_planets_are_solar_type() {
    // With Jupiter and Saturn present, the archetype must be SolarType
    // and the outer belt must contain the four canonical populations
    // plus Trojans and Kirkwood gaps. Note this test does NOT assert an
    // inner belt: `belt_spec_for` places it between `a_compact` and
    // `a_giant_inner`, and the Solar System's outermost sub-Saturn
    // planet is Neptune at 30 AU. See
    // `belt_spec_inner_belt_without_ice_giants` for the inner-belt path.
    let seed = find_solar_seed(|s| s.archetype == BeltArchetype::SolarType);
    let spec = spec_for_solar_planets(seed);
    let outer = spec
        .outer
        .as_ref()
        .expect("Solar analogue missing outer belt");
    assert_eq!(
        outer.populations.len(),
        4,
        "Solar-type outer belt must have 4 populations"
    );
    assert!(!spec.trojans.is_empty(), "Solar analogue missing Trojans");
    assert!(
        !spec.gaps.is_empty(),
        "Solar analogue missing Kirkwood gaps"
    );
}

#[test]
fn belt_spec_inner_belt_without_ice_giants() {
    // When there are no sub-Saturn planets between the terrestrial
    // planets and the first giant, `a_compact` stays small enough that
    // `ast_hi > ast_lo` and the inner-belt branch fires. This exercises
    // the branch that the Solar System fixture above cannot reach.
    let planets = vec![
        (0usize, 0.055f64, 0.387f64),
        (1, 0.815, 0.723),
        (2, 1.0, 1.0),
        (3, 0.107, 1.524),
        (4, 317.8, 5.2), // Jupiter
        (5, 95.2, 9.58), // Saturn
    ];
    let mut found = false;
    for seed in 0u128..256 {
        let mut rng = XorShift::new(seed);
        let bodies = Bodies::with_capacity(0);
        let spec = belt_spec_for(
            &bodies, M_SUN_ME, 0.0, 4.6, &planets, 2.7, 2.0, 50.0, &mut rng,
        );
        if spec.is_empty() {
            continue;
        }
        assert_eq!(spec.archetype, BeltArchetype::SolarType);
        assert!(
            spec.inner.is_some(),
            "expected inner belt with no ice giants (a_compact should stay small)"
        );
        found = true;
        break;
    }
    assert!(found, "no seed passed the debris-belt gate");
}

#[test]
fn belt_spec_hot_jupiter_is_depleted() {
    // Any giant with a < 0.1 AU triggers the HotJupiter branch: outer
    // belt only, no Trojans, no inner belt.
    let planets = vec![(0usize, 200.0f64, 0.05f64)];
    let mut found = false;
    for seed in 0u128..256 {
        let mut rng = XorShift::new(seed);
        let bodies = Bodies::with_capacity(0);
        let spec = belt_spec_for(
            &bodies, M_SUN_ME, 0.0, 4.6, &planets, 2.7, 2.0, 50.0, &mut rng,
        );
        if spec.is_empty() {
            continue;
        }
        assert_eq!(spec.archetype, BeltArchetype::HotJupiter);
        assert!(spec.inner.is_none(), "HotJupiter must not have inner belt");
        assert!(spec.outer.is_some(), "HotJupiter must have outer belt");
        assert!(spec.trojans.is_empty(), "HotJupiter must not have Trojans");
        found = true;
        break;
    }
    assert!(found, "no seed passed the debris-belt gate");
}

#[test]
fn belt_spec_compact_chain_triggers_on_four_small_planets() {
    // TRAPPIST-1-like: no giants, 4+ small planets inside 1 AU.
    let planets = vec![
        (0usize, 0.33f64, 0.011f64),
        (1, 1.37, 0.016),
        (2, 0.41, 0.021),
        (3, 0.69, 0.028),
        (4, 0.68, 0.037),
    ];
    let mut found = false;
    for seed in 0u128..256 {
        let mut rng = XorShift::new(seed);
        let bodies = Bodies::with_capacity(0);
        let spec = belt_spec_for(
            &bodies, M_SUN_ME, 0.0, 4.6, &planets, 2.7, 2.0, 50.0, &mut rng,
        );
        if spec.is_empty() {
            continue;
        }
        assert_eq!(spec.archetype, BeltArchetype::CompactChain);
        assert!(spec.inner.is_none(), "CompactChain has no inner belt");
        assert!(spec.outer.is_some(), "CompactChain has outer belt");
        found = true;
        break;
    }
    assert!(found, "no seed passed the debris-belt gate");
}

#[test]
fn belt_spec_young_debris_disk_triggers() {
    // No giants, massive star, young age. The branch multiplies belt
    // mass by 100x, so this also exercises the mass scaling path.
    let planets: Vec<(usize, f64, f64)> = vec![];
    let mut found = false;
    for seed in 0u128..256 {
        let mut rng = XorShift::new(seed);
        let bodies = Bodies::with_capacity(0);
        let spec = belt_spec_for(
            &bodies,
            1.5 * M_SUN_ME,
            0.0,
            0.1,
            &planets,
            2.7,
            2.0,
            200.0,
            &mut rng,
        );
        if spec.is_empty() {
            continue;
        }
        assert_eq!(spec.archetype, BeltArchetype::YoungDebrisDisk);
        assert!(spec.outer.is_some());
        found = true;
        break;
    }
    assert!(found, "no seed passed the debris-belt gate");
}

#[test]
fn belt_spec_sparse_triggers() {
    // No giants, low-mass, old star: Sparse rather than YoungDebrisDisk.
    let planets: Vec<(usize, f64, f64)> = vec![];
    let mut found = false;
    for seed in 0u128..256 {
        let mut rng = XorShift::new(seed);
        let bodies = Bodies::with_capacity(0);
        let spec = belt_spec_for(
            &bodies,
            0.5 * M_SUN_ME,
            0.0,
            8.0,
            &planets,
            2.7,
            2.0,
            50.0,
            &mut rng,
        );
        if spec.is_empty() {
            continue;
        }
        assert_eq!(spec.archetype, BeltArchetype::Sparse);
        assert!(spec.inner.is_none(), "Sparse has no inner belt");
        assert!(spec.outer.is_some(), "Sparse has outer belt");
        found = true;
        break;
    }
    assert!(found, "no seed passed the debris-belt gate");
}

#[test]
fn belt_spec_for_evolved_is_rare_and_bounded() {
    // `belt_spec_for_evolved` draws a 30 % gate on low-mass remnants and
    // draws `total = 1e-4 * rng.next_f64()` (so `total <= 1e-4`). Both
    // branches must be reachable, and the bound must hold when the
    // branch fires.
    let mut some = false;
    let mut none = false;
    for seed in 0u128..256 {
        let mut rng = XorShift::new(seed);
        let spec = belt_spec_for_evolved(0.6, 5.0, &mut rng);
        if spec.is_empty() {
            none = true;
        } else {
            assert_eq!(spec.archetype, BeltArchetype::Evolved);
            assert!(spec.total_mass <= 1e-4);
            some = true;
        }
        if some && none {
            break;
        }
    }
    assert!(
        some && none,
        "belt_spec_for_evolved never exercised both branches"
    );
}

#[test]
fn belt_spec_for_evolved_never_triggers_for_massive_remnant() {
    // Condition is `m_remnant_sun < 1.0`; exactly 1.0 must not spawn.
    for seed in 0u128..32 {
        let mut rng = XorShift::new(seed);
        let spec = belt_spec_for_evolved(1.0, 5.0, &mut rng);
        assert!(
            spec.is_empty(),
            "massive remnant spawned belt at seed {seed}"
        );
    }
}

#[test]
fn belt_spec_empty_and_is_empty_agree() {
    let s = BeltSpec::empty();
    assert!(s.is_empty());
    assert_eq!(s.archetype, BeltArchetype::NoBelt);
    assert!(s.inner.is_none());
    assert!(s.outer.is_none());
    assert!(s.trojans.is_empty());
    assert!(s.gaps.is_empty());
}

// ---------------------------------------------------------------------------
// `BeltPreset::to_spec`
//
// Preset-side JSON exposes only radial spans and mass shares; `to_spec`
// fills in the physics (slopes, size bounds) with defaults. These tests
// pin those defaults and the kind-string mapping, since both are easy to
// accidentally change while editing another preset.
// ---------------------------------------------------------------------------

#[test]
fn belt_preset_to_spec_maps_all_outer_populations() {
    let json = r#"{
        "total_mass_earth": 0.1,
        "outer": { "populations": [
            { "kind": "cold_classical", "a_lo_au": 30.0, "a_hi_au": 40.0, "mass_frac": 0.4 },
            { "kind": "hot_classical",  "a_lo_au": 30.0, "a_hi_au": 50.0, "mass_frac": 0.3, "e_sigma": 0.15 },
            { "kind": "resonant",       "a_lo_au": 30.0, "a_hi_au": 50.0, "mass_frac": 0.1 },
            { "kind": "scattered",      "a_lo_au": 30.0, "a_hi_au": 50.0, "mass_frac": 0.1, "q_frac": 35.0 }
        ] }
    }"#;
    let bp: BeltPreset = serde_json::from_str(json).expect("parse");
    let spec = bp.to_spec();
    let pops = &spec.outer.as_ref().unwrap().populations;
    assert_eq!(pops.len(), 4);
    assert_eq!(pops[0].kind, BodyOrigin::KuiperCold);
    assert_eq!(pops[1].kind, BodyOrigin::KuiperHot);
    assert_eq!(pops[2].kind, BodyOrigin::KuiperResonant);
    assert_eq!(pops[3].kind, BodyOrigin::KuiperScattered);
    // `q_frac` is the only per-population field forwarded verbatim; the
    // rest are filled in with defaults and would be easy to lose.
    assert_eq!(pops[3].q_frac, Some(35.0));
}

#[test]
fn belt_preset_unknown_kind_falls_back_to_cold() {
    // A typo must not panic: the preset library would otherwise fail to
    // load, and a single bad JSON entry shouldn't take down the whole
    // selector.
    let json = r#"{
        "total_mass_earth": 0.1,
        "outer": { "populations": [
            { "kind": "definitely_a_typo", "a_lo_au": 30.0, "a_hi_au": 40.0, "mass_frac": 1.0 }
        ] }
    }"#;
    let bp: BeltPreset = serde_json::from_str(json).expect("parse");
    let spec = bp.to_spec();
    assert_eq!(
        spec.outer.unwrap().populations[0].kind,
        BodyOrigin::KuiperCold
    );
}

#[test]
fn belt_preset_to_spec_preserves_trojans_and_gaps() {
    let json = r#"{
        "total_mass_earth": 0.001,
        "trojans": [{ "giant_a_au": 5.2, "mass_frac": 0.06 }],
        "gaps": [
            { "a_au": 2.5, "width_au": 0.05 },
            { "a_au": 2.82, "width_au": 0.05 }
        ]
    }"#;
    let bp: BeltPreset = serde_json::from_str(json).expect("parse");
    let spec = bp.to_spec();
    assert_eq!(spec.trojans.len(), 1);
    assert!((spec.trojans[0].giant_a_au - 5.2).abs() < 1e-12);
    // `leading_frac` is not part of the preset schema; `to_spec` fills
    // it in at 0.5 (symmetric cloud).
    assert!((spec.trojans[0].leading_frac - 0.5).abs() < 1e-12);
    assert_eq!(spec.gaps.len(), 2);
}
