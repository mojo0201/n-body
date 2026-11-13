//! Tests for the stellar-evolution module: MS/giant structure, remnant
//! formation, dwarf cooling, Roche-lobe mass transfer, and orbit response
//! to adiabatic vs impulsive mass loss.
//!
//! Tests fall into three tiers:
//!
//!  1. **Unit checks** on the pure formulas (`star_luminosity`,
//!     `remnant_of`, `roche_lobe_radius`, `bd_state`, …). These pin down
//!     the physics models against textbook reference values.
//!
//!  2. **State-machine checks** on the `Bodies` integration: what happens
//!     when a star evolves, when a WD crosses the Chandrasekhar mass,
//!     when a NS crosses the TOV limit, when a BD ignites, when a wide
//!     planet is unbound by a supernova. These pin down the *branching*
//!     of the module's `match` arms.
//!
//!  3. **Invariant checks** on the surrounding n-body machinery: momentum
//!     and angular momentum conservation during mass transfer, orbital
//!     response to adiabatic vs impulsive mass loss, and cross-field
//!     consistency after `compact()` / `merge()`.

use super::super::*;
use super::shared::*;

// ---------------------------------------------------------------------------
// Static defaults and no-op behaviour
// ---------------------------------------------------------------------------

/// With `evolution_rate == 0` the whole module is a no-op. Nothing about
/// a star — age, mass, radius, kind — should drift from the preset value.
/// This is the "presets are unchanged" guarantee that the rest of the app
/// relies on until the user explicitly enables evolution.
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

/// At age 0 every evolutionary track must collapse exactly onto the
/// static `physical_radius` / `star_luminosity` / `star_temperature`
/// relations. This is what makes presets look identical whether or not
/// evolution is on. The tolerance is exact equality (not epsilon) because
/// the code multiplies by `(1 + growth * 0.0) == 1.0` on the age-0 branch.
#[test]
fn al_age_zero_matches_the_old_relations() {
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

// ---------------------------------------------------------------------------
// Star structure across the MS → giant transition
// ---------------------------------------------------------------------------

/// A late-stage solar-mass star must be physically a red giant:
/// hundreds of R☉, thousands of L☉, cooler than 4500 K. Additionally,
/// radius and luminosity must be *continuous* across the MS→giant
/// boundary — a visible jump would show as a star snapping in the UI.
#[test]
fn al_giant_phase_is_bright_large_and_cool() {
    let m = M_SUN_ME;
    let t_ms = star_lifetime_gyr(BodyKind::Star, m);
    let late = star_state(m, m * 0.6, t_ms * (1.0 + EVO_POST_MS_FRAC * 0.95));
    assert_eq!(late.phase, StellarPhase::RedGiant);
    assert!(late.radius_gm > 100.0 * R_SUN_GM);
    assert!(late.luminosity > 1000.0);
    assert!(late.temperature < 4500.0);
    // Continuity check with sub-microscopic steps across the boundary.
    let before = star_state(m, m, t_ms * 0.999_999);
    let after = star_state(m, m, t_ms * 1.000_001);
    assert!((after.radius_gm / before.radius_gm - 1.0).abs() < 1e-3);
    assert!((after.luminosity / before.luminosity - 1.0).abs() < 1e-2);
}

// ---------------------------------------------------------------------------
// Remnant formation — one test per mass regime
// ---------------------------------------------------------------------------

/// A solar-mass star should end as a WD on the Kalirai 2008 IFMR
/// (≈0.503 M☉ for m_init = 1 M☉), and the transition must be logged
/// as an `Evolve` event. Radius must collapse to a WD-scale value.
#[test]
fn al_sunlike_star_becomes_a_white_dwarf() {
    let mut b = sun_system(&[30.0]);
    set_evolution_rate(&mut b, 4.0);
    run_to_remnant(&mut b, 5000);
    assert_eq!(b.kind[0], BodyKind::WhiteDwarf);
    let ms = b.m[0] / M_SUN_ME;
    assert!((ms - 0.503).abs() < 0.02, "WD mass {ms}");
    assert!(b.r[0] < 0.05);
    assert!(b.events.iter().any(|e| e.kind == EventKind::Evolve));
}

/// A 12 M☉ progenitor lands in the NS branch of `remnant_of`. The
/// section-B model gives 1.4 + 0.6·(12−8)/(20−8) = 1.6 M☉.
/// `sn_kicks` is disabled so the mass test isn't perturbed by a
/// random velocity draw.
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
    // NS mass = 1.4 + 0.6 * (12-8)/(20-8) = 1.6.
    assert!((ms - 1.6).abs() < 0.05, "NS mass {ms}");
}

/// A 25 M☉ progenitor lands in the BH branch: 25% retention, floored
/// at `BH_M_MIN`. The test asserts 6.25 M☉ exactly.
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
    // BH mass = max(0.25 * 25, BH_M_MIN) = 6.25.
    assert!((ms - 6.25).abs() < 0.05, "BH mass {ms}");
}

/// `sn_kicks` gates the stochastic velocity impulse applied at
/// `become_remnant`. Off → exactly zero. On → nonzero. This is the
/// only place the RNG enters remnant formation, so the test pins
/// both ends of the flag.
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

// ---------------------------------------------------------------------------
// Roche geometry
// ---------------------------------------------------------------------------

/// Eggleton 1983 at q = 1 gives ≈0.379. A lighter donor must have a
/// smaller lobe. Together these pin both the normalization and the
/// monotonic direction of `roche_lobe_radius`.
#[test]
fn al_roche_lobe_matches_eggleton() {
    let rl = roche_lobe_radius(1.0, 1.0, 1.0);
    assert!((rl - 0.379).abs() < 0.002, "equal-mass RL = {rl}");
    assert!(
        roche_lobe_radius(1.0, 0.2, 1.0) < rl,
        "lighter donor has smaller lobe"
    );
}

// ---------------------------------------------------------------------------
// Momentum and angular momentum during transfer
// ---------------------------------------------------------------------------

/// `transfer_mass` must conserve linear and angular momentum *of the
/// whole system*, accounting for the fraction of the transferred mass
/// that is carried away by the donor's wind (fraction `1 − MT_EFF_COMPACT`,
/// travelling at the donor's velocity). The test builds a non-trivial
/// two-body configuration (non-zero positions and velocities, off-axis)
/// so that a bug that forgets one of the two impulses will show up.
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
    // The mass lost from the system moves with the donor's velocity,
    // so add its (would-be) contribution back before comparing.
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

/// A WD that crosses the Chandrasekhar mass must explode as Type Ia:
/// `transfer_mass` returns `true`, the body is flagged dead, and an
/// event is logged. The 0.05 M☉ transfer × `MT_EFF_COMPACT` is enough
/// to push 1.39 M☉ over the 1.4 limit.
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

// ---------------------------------------------------------------------------
// Dwarf cooling tracks
// ---------------------------------------------------------------------------

/// Mestel law: L ∝ t^-WD_COOL_EXP for cooling time t ≫ t₀. Ten
/// Gyr versus 1 Gyr should show a factor of ≈10^WD_COOL_EXP. Also
/// confirms WD temperature crosses the ≥ 50 000 K hot start and the
/// < 6 000 K old-WD regime.
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

/// Young brown dwarfs are larger, brighter, and hotter than old ones.
/// Their temperature must cross a spectral-class boundary within
/// 0.1 → 10 Gyr, which is what the UI's "class" field depends on.
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

/// New WDs and BDs must not start at age 0 (they'd appear hotter /
/// puffier than the "today" preset look). `default_age_gyr` is the
/// single source of truth that spawn code consults.
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

// ---------------------------------------------------------------------------
// Data structure invariants — compaction and merging
// ---------------------------------------------------------------------------

/// `compact()` removes dead bodies. Every per-body array, including the
/// evolution fields `age_gyr`, `m_init`, and `rlof`, must move in lockstep
/// with the standard fields, or the surviving bodies end up with a
/// mismatched evolutionary state.
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

/// Merging a half-spent star with a fresh one gives a 2 M☉ star at the
/// mass-weighted fraction of its (shorter) new MS lifetime. The point
/// is that the survivor's clock is re-normalized by mass, not by
/// absolute age.
#[test]
fn al_merging_stars_rejuvenates_the_survivor() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(0.5, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.age_gyr[0] = 5.0; // half of a 10 Gyr MS
    let keep = merge(&mut b, 0, 1);
    assert_eq!(b.m_init[keep], 2.0 * M_SUN_ME, "m_init = sum");
    let t_ms = star_lifetime_gyr(BodyKind::Star, 2.0 * M_SUN_ME);
    let expected = 0.25 * t_ms; // (0.5 + 0) / 2 = 0.25 mass-weighted fraction
    assert!(
        (b.age_gyr[keep] - expected).abs() < 1e-6,
        "age {} vs expected {}",
        b.age_gyr[keep],
        expected
    );
}

// ---------------------------------------------------------------------------
// User-facing behaviour (UI strings, HZ, remaining life)
// ---------------------------------------------------------------------------

/// Once a star is aged past t_MS by `set_star_age`, `class_name`
/// (which the UI displays) must report "red giant".
#[test]
fn al_class_name_reports_red_giant_after_aging() {
    let mut b = sun_system(&[1.0]);
    let t_ms = star_lifetime_gyr(BodyKind::Star, M_SUN_ME);
    set_star_age(&mut b, 0, t_ms * 1.12);
    assert_eq!(class_name(&b, 0), "red giant");
}

/// A planet sitting at the current position of an Earth-like orbit
/// must see a >50× flux increase when the star becomes a red giant.
/// This is the observable consequence of the giant branch for
/// habitable-zone display.
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

/// `set_star_age` clamps user input to `0.999999 × total_life`, so a
/// star can never be advanced into the remnant state by the setter.
/// The test verifies the clamp lands just below the death boundary.
#[test]
fn al_set_star_age_clamps_to_total_life() {
    let mut b = sun_system(&[]);
    let life = star_total_life_gyr(M_SUN_ME);
    set_star_age(&mut b, 0, life * 2.0);
    assert!(b.age_gyr[0] < life);
    assert!(b.age_gyr[0] > life * 0.999_990);
    assert_eq!(b.kind[0], BodyKind::Star);
}

/// `evolution_rate <= 0` freezes age. `> 0` advances by `rate × dt`
/// per step. Negative values are rejected to 0 rather than inverted.
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

// ---------------------------------------------------------------------------
// Orbit response to mass loss: adiabatic (WD) vs impulsive (SN)
// ---------------------------------------------------------------------------

/// Direct call to `become_remnant`: a solar-mass WD formation is
/// adiabatic, so a planet at 50 Gm expands to a_new ≈ a_old · M_old/M_new.
/// Eccentricity is preserved (that's the "adiabatic" part — the shape
/// doesn't change, only the scale).
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
    // m_old = 1 M☉, m_new ≈ 0.503 M☉ → a_new ≈ 2 a_old.
    let expected = o0.semi_major * (M_SUN_ME / b.m[0]);
    assert!(
        (o1.semi_major / expected - 1.0).abs() < 0.05,
        "a {} vs expected {}",
        o1.semi_major,
        expected
    );
    assert!(o1.bound);
    assert!((o1.ecc - o0.ecc).abs() < 0.05, "e {} -> {}", o0.ecc, o1.ecc);
}

/// A moon orbiting a planet must ride along with its parent during
/// adiabatic expansion, so the *internal* orbit (moon around planet)
/// is unchanged even while the planet's orbit around the WD grows.
/// `adiabatic_expand` has a separate "ride" pass for this case; without
/// it the moon would be left behind and the moon-planet pair would
/// visibly break apart on screen.
#[test]
fn al_wd_formation_moves_satellites_with_their_parent() {
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

/// `adiabatic_expand` must not touch systems beyond `ADIABATIC_COMPANION_MAX_GM`.
/// On the ring of systems, a wide bound neighbour is indistinguishable
/// from an unrelated system; moving it would scramble the ring layout.
///
/// The neighbour is a genuine bound companion (~200 AU, v = 0.5 × v_circ)
/// so the test also excludes the "unbound so obviously untouched" case.
#[test]
fn al_wd_formation_leaves_other_systems_alone() {
    let mut b = Bodies::with_capacity(2);
    // Host star at the origin.
    let h = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    // Wide bound neighbour at ~200 AU, well outside
    // ADIABATIC_COMPANION_MAX_GM (15 000 Gm).
    let d = 30_000.0;
    let v_circ = (G * 2.0 * M_SUN_ME / d).sqrt();
    let n = b.push(d, 0.0, 0.0, 0.5 * v_circ, M_SUN_ME, BodyKind::Star);
    let before = (b.x[n], b.y[n], b.vx[n], b.vy[n]);
    become_remnant(&mut b, h);
    assert_eq!(b.kind[h], BodyKind::WhiteDwarf);
    assert_eq!(
        (b.x[n], b.y[n], b.vx[n], b.vy[n]),
        before,
        "distant neighbour must not move"
    );
}

/// A close bound stellar companion inside `ADIABATIC_COMPANION_MAX_GM`
/// must expand with the host's mass loss. The companion is placed on
/// a *circular* orbit (v = sqrt(G·(M₁+M₂)/d)) so the test's
/// adiabatic-invariant check (a ∝ 1/M_total) is exact.
#[test]
fn al_wd_formation_expands_close_companion() {
    let mut b = Bodies::with_capacity(2);
    let m0 = M_SUN_ME;
    let m1 = 0.3 * M_SUN_ME;
    b.push(0.0, 0.0, 0.0, 0.0, m0, BodyKind::Star);
    let d = 1000.0;
    // Circular relative velocity for the actual two-body total mass
    // — the previous draft used 2.0·M☉ by mistake, giving an eccentric
    // start whose semi-major was large enough to pass the threshold
    // even with `adiabatic_expand` disabled.
    let v = (G * (m0 + m1) / d).sqrt();
    b.push(d, 0.0, 0.0, v, m1, BodyKind::Star);
    let o0 = orbit_info(&b, 1, 0);
    become_remnant(&mut b, 0);
    let o1 = orbit_info(&b, 1, 0);
    assert!(o1.bound);
    // Adiabatic invariant: a · (m_host + m_companion) is conserved.
    let expected = o0.semi_major * (m0 + m1) / (b.m[0] + m1);
    assert!(
        (o1.semi_major / expected - 1.0).abs() < 0.05,
        "a {} vs expected {} (o0 {})",
        o1.semi_major,
        expected,
        o0.semi_major
    );
}

/// A supernova is *impulsive*: the envelope leaves long before a wide
/// planet can respond, so the orbital velocity is no longer enough to
/// stay bound to the lighter remnant. v_circ(12 M☉ at 50 Gm) exceeds
/// v_esc(1.6 M☉ at 50 Gm), so the planet must be ejected.
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
    assert!(!o.bound, "wide planet should be unbound by SN");
}

// ---------------------------------------------------------------------------
// Random spawn sanity — the spawner must produce self-consistent states
// ---------------------------------------------------------------------------

/// Over 200 seeds the spawner must occasionally produce a system whose
/// primary is already a remnant. Without this, remnant-handling code
/// paths would be unreachable via the random spawner.
#[test]
fn al_random_age_produces_some_remnants() {
    let mut n_rem = 0;
    for seed in 0..200u128 {
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

/// Every spawned star must satisfy the core invariants of the
/// evolution model: finite non-negative age, age ≤ total lifetime,
/// m ≤ m_init, and `r == radius_now(i)`. This catches spawner bugs
/// that set, say, an age past the end-of-life boundary without
/// converting to a remnant.
#[test]
fn al_random_ages_are_consistent() {
    for seed in 0..300u128 {
        let mut b = Bodies::with_capacity(12);
        spawn_system(&mut b, 12, seed);
        for i in 0..b.len() {
            assert!(b.age_gyr[i].is_finite() && b.age_gyr[i] >= 0.0);
            if b.kind[i] == BodyKind::Star {
                assert!(b.age_gyr[i] <= star_total_life_gyr(b.m_init[i]) + 1e-9);
                assert!(b.m[i] <= b.m_init[i] + 1e-9);
                assert!((b.r[i] - b.radius_now(i)).abs() <= 1e-12 * b.r[i].max(1.0));
            }
        }
    }
}

/// Same invariant, but across a full ring of randomly-seeded systems.
/// Any body — aged star, remnant, brown dwarf, planet — must have
/// radius consistent with its current evolutionary state.
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

/// Smoke test: 40 random 30-body universes simulated with evolution on
/// must not panic or produce NaN / infinity in positions or velocities.
/// Also: a solar system with a fast evolution rate must reach the WD
/// stage within 3000 steps.
/// Smoke test: 40 random 30-body universes simulated with evolution on
/// must not panic or produce NaN / infinity in positions or velocities.
/// Also: a solar system with a fast evolution rate must reach the WD
/// stage within 3000 steps.
#[test]
#[ignore = "40x30-body + solar runs (27 000 advance calls); ~60 s in debug. Run with -- --ignored"]
fn smoke_evolution_runs_without_panic() {
    for seed in 0..40u128 {
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

// ---------------------------------------------------------------------------
// Regression: giant-phase mass loss must expand orbits continuously
// ---------------------------------------------------------------------------

/// **Regression test for CRITICAL.txt.**
///
/// The giant-phase envelope is shed *gradually* across the entire
/// post-MS phase, so `evolve_stars` must call `adiabatic_expand` on
/// every step's mass decrease, not once at `become_remnant` after the
/// mass has already collapsed to `m_rem`. The old code only expanded
/// once — producing a final orbit that was too small — and this test
/// caught that.
///
/// The planet starts at 500 Gm, well outside the Sun's ~139 Gm giant
/// radius, so it survives the whole giant phase and its final semi-major
/// can be measured cleanly.
#[test]
fn evolve_sun_to_wd_expands_planet_orbit() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    // Start outside the giant envelope. With a0 = 500 Gm the orbit
    // peaks around ~994 Gm and the planet stays bound and outside.
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

/// The spawner must not assign an age older than the age of the
/// universe. Otherwise, WDs and BDs in the preset would have cooled
/// past the model's valid range and the resulting temperature would
/// be unphysical.
#[test]
fn al_spawned_ages_within_universe_age() {
    for seed in 0..500u128 {
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

// ---------------------------------------------------------------------------
// transfer_mass: exotic outcomes
// ---------------------------------------------------------------------------

/// A brown dwarf that accretes past the hydrogen-burning limit must
/// ignite: kind becomes Star, initial mass and age are reset to the
/// post-ignition state, and the event is logged. This is the "BD
/// ignites" arm of `transfer_mass`; without it the accretor would
/// stay a BD forever despite exceeding the mass limit.
#[test]
fn al_bd_ignites_when_crossing_the_hydrogen_limit() {
    let mut b = Bodies::with_capacity(2);
    let d = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let m_bd = (STAR_M_MIN - 0.001 * M_SUN_ME).max(1.0);
    let a = b.push(100.0, 0.0, 0.0, 0.0, m_bd, BodyKind::BrownDwarf);
    let dm = 0.01 * M_SUN_ME; // pushes it above STAR_M_MIN
    assert!(!transfer_mass(&mut b, d, a, dm));
    assert_eq!(b.kind[a], BodyKind::Star);
    assert_eq!(b.age_gyr[a], 0.0);
    assert!(b
        .events
        .iter()
        .any(|e| e.text.contains("ignited as a star")));
}

/// A neutron star that accretes past `NS_M_MAX` must collapse to a BH.
/// The transfer amount is computed from the gap-to-limit divided by
/// `MT_EFF_COMPACT` plus a margin, so the test is insensitive to the
/// exact efficiency constant.
#[test]
fn al_ns_accreting_too_much_collapses_to_bh() {
    let mut b = Bodies::with_capacity(2);
    let d = b.push(0.0, 0.0, 0.0, 0.0, 5.0 * M_SUN_ME, BodyKind::Star);
    let a = b.push(
        100.0,
        0.0,
        0.0,
        0.0,
        NS_M_MAX * 0.999,
        BodyKind::NeutronStar,
    );
    // Enough mass (accounting for retention) to exceed the limit.
    let dm = (NS_M_MAX - b.m[a]) / MT_EFF_COMPACT + 0.1 * M_SUN_ME;
    assert!(!transfer_mass(&mut b, d, a, dm));
    assert_eq!(b.kind[a], BodyKind::BlackHole);
    assert!(b
        .events
        .iter()
        .any(|e| e.text.contains("collapsed to a black hole")));
}

// ---------------------------------------------------------------------------
// mass_transfer_step: no-op paths and RLOF begin/end logging
// ---------------------------------------------------------------------------

/// Three configurations where mass transfer must not fire:
/// no hosts at all, a single host (no partner to transfer to), and
/// two hosts with radii far below their mutual Roche lobes. Each must
/// return `false` (nothing changed) so the caller skips `compute_accel`.
#[test]
fn al_mass_transfer_step_returns_false_without_rlof() {
    // No hosts at all.
    let mut b = Bodies::with_capacity(0);
    assert!(!mass_transfer_step(&mut b, DT));

    // A single host.
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    assert!(!mass_transfer_step(&mut b, DT));

    // Two hosts but well outside each other's Roche lobes.
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(10_000.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.r[0] = 0.001;
    b.r[1] = 0.001;
    b.mass_transfer = true;
    assert!(!mass_transfer_step(&mut b, DT));
}

/// When RLOF begins the `rlof` flag flips on and an event is logged;
/// when the donor shrinks back inside its lobe the flag flips off
/// and a "stopped transferring" event fires. This is the state
/// machine the UI uses to display a binary-mass-transfer indicator.
#[test]
fn al_mass_transfer_step_logs_begin_and_end() {
    let mut b = Bodies::with_capacity(2);
    let m1 = M_SUN_ME;
    let m2 = M_SUN_ME;
    let sep = 20.0;
    b.push(0.0, 0.0, 0.0, 0.0, m1, BodyKind::Star);
    b.push(sep, 0.0, 0.0, 0.0, m2, BodyKind::Star);
    b.mass_transfer = true;
    b.r[0] = roche_lobe_radius(sep, m1, m2) * 1.01;
    b.r[1] = 0.001;
    compute_accel(&mut b);
    rebase_energy(&mut b);

    assert!(mass_transfer_step(&mut b, DT));
    assert!(b.rlof[0]);
    assert!(b
        .events
        .iter()
        .any(|e| e.text.contains("began transferring")));

    // Shrink the donor well inside its lobe: RLOF ends.
    b.r[0] = 0.001;
    let _ = mass_transfer_step(&mut b, DT);
    assert!(!b.rlof[0]);
    assert!(b
        .events
        .iter()
        .any(|e| e.text.contains("stopped transferring")));
}

/// End-to-end Type Ia: a `0.999 × WD_M_MAX` white dwarf is continuously
/// fed by a donor star, and `mass_transfer_step` alone (not a direct
/// `transfer_mass` call) must drive it over the limit and destroy it.
///
/// `transfer_mass` calls `set_mass`, which re-derives a *main-sequence*
/// star's radius from its evolutionary state — so the donor's manual
/// inflation would be wiped out after the first step and RLOF would
/// stop. Real giants are continuously re-inflated by `evolve_stars`;
/// this test mimics that by re-inflating before each step.
#[test]
fn al_type_ia_via_mass_transfer_step() {
    let mut b = Bodies::with_capacity(2);
    let m1 = M_SUN_ME;
    let m2_old = WD_M_MAX * 0.999;
    let sep = 100.0;
    b.push(0.0, 0.0, 0.0, 0.0, m1, BodyKind::Star);
    b.push(sep, 0.0, 0.0, 0.0, m2_old, BodyKind::WhiteDwarf);
    b.mass_transfer = true;
    compute_accel(&mut b);
    rebase_energy(&mut b);

    // Factor 1.5 puts x ≈ 0.33 → x³ ≈ 0.037, roughly 50× the transfer
    // rate of a marginal 1.1 inflation, so the ~10⁻³ M☉ needed to
    // reach Chandrasekhar completes in a handful of steps.
    for _ in 0..5000 {
        b.r[0] = roche_lobe_radius(sep, b.m[0], b.m[1]) * 1.5;
        mass_transfer_step(&mut b, DT);
        if b.events.iter().any(|e| e.text.contains("Type Ia")) {
            break;
        }
    }
    assert!(
        b.events.iter().any(|e| e.text.contains("Type Ia")),
        "expected Type Ia; events = {:?}",
        b.events.iter().map(|e| e.text.as_str()).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// adiabatic_expand guards
// ---------------------------------------------------------------------------

/// The early-return branches of `adiabatic_expand` must be no-ops:
/// mass *increases* (m_new ≥ m_old) and degenerate systems (n < 2).
/// A regression that moved bodies on these paths would corrupt
/// orbits whenever the caller re-runs the function defensively.
#[test]
fn al_adiabatic_expand_no_op_cases() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(100.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let before = (b.x[1], b.vx[1]);
    adiabatic_expand(&mut b, 0, M_SUN_ME, M_SUN_ME);
    assert_eq!((b.x[1], b.vx[1]), before);
    adiabatic_expand(&mut b, 0, M_SUN_ME, 2.0 * M_SUN_ME);
    assert_eq!((b.x[1], b.vx[1]), before);
    // n < 2: must not panic on an empty-of-neighbours universe.
    let mut single = Bodies::with_capacity(1);
    single.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    adiabatic_expand(&mut single, 0, M_SUN_ME, 0.5 * M_SUN_ME);
}

// ---------------------------------------------------------------------------
// evolve_stars: dwarf aging and the MS → giant log
// ---------------------------------------------------------------------------

/// White and brown dwarfs age but don't transform. The test pins both
/// behaviours: age must advance by `rate × dt`, and the BD's radius
/// must be refreshed from `radius_now` (which consults the cooling
/// track), not left at the pre-step value.
#[test]
fn al_evolve_stars_ages_wd_and_bd() {
    let mut b = Bodies::with_capacity(2);
    let w = b.push(0.0, 0.0, 0.0, 0.0, 0.6 * M_SUN_ME, BodyKind::WhiteDwarf);
    let d = b.push(
        1.0,
        0.0,
        0.0,
        0.0,
        40.0 * JUPITER_MASS,
        BodyKind::BrownDwarf,
    );
    b.age_gyr[w] = 1.0;
    b.age_gyr[d] = 1.0;
    set_evolution_rate(&mut b, 2.0);
    evolve_stars(&mut b, DT);
    assert!((b.age_gyr[w] - (1.0 + 2.0 * DT)).abs() < 1e-12);
    assert!((b.age_gyr[d] - (1.0 + 2.0 * DT)).abs() < 1e-12);
    assert!((b.r[d] - b.radius_now(d)).abs() <= 1e-12 * b.r[d].max(1.0));
}

/// Crossing t_MS in a single step must produce exactly one
/// "left the main sequence" event. `evolve_stars` detects the crossing
/// by comparing `age` before and after the step.
#[test]
fn al_evolve_stars_logs_ms_giant_transition() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let t_ms = star_lifetime_gyr(BodyKind::Star, M_SUN_ME);
    // Just below the boundary; one step will push it across.
    b.age_gyr[0] = t_ms - 0.5 * DT;
    set_evolution_rate(&mut b, 1.0);
    evolve_stars(&mut b, DT);
    assert!(b
        .events
        .iter()
        .any(|e| e.text.contains("left the main sequence")));
}

/// The habitable zone must scale as √L. A red giant's L ~ 1000× solar
/// means the HZ moves out by ~30×. The test uses a conservative 10×
/// floor so it stays robust to small changes in the giant-luminosity
/// model, while still verifying the direction and magnitude.
#[test]
fn al_hz_expands_as_star_brightens() {
    let mut b = sun_system(&[]);
    let (i0, o0) = habitable_zone_of(&b, 0);
    let t_ms = star_lifetime_gyr(BodyKind::Star, M_SUN_ME);
    set_star_age(&mut b, 0, t_ms * 1.12);
    let (i1, o1) = habitable_zone_of(&b, 0);
    assert!(i1 > 10.0 * i0);
    assert!(o1 > 10.0 * o0);
}
