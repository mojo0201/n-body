//! Tests: orbit line rendering — `compute_orbit_segments`.

use super::super::*;

// ===========================================================================
// an — compute_orbit_segments: early returns
// ===========================================================================

#[test]
fn an_empty_returns_empty() {
    let b = Bodies::with_capacity(0);
    assert!(compute_orbit_segments(&b, 1.0, true, true).is_empty());
}

#[test]
fn an_single_body_returns_empty() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    assert!(compute_orbit_segments(&b, 1.0, true, true).is_empty());
}

#[test]
fn an_no_flags_returns_empty() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let v = (G * M_SUN_ME / 5.0).sqrt();
    b.push(5.0, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    let j = b.push(5.0, 0.1, 0.0, v, 1e-6, BodyKind::Planet);
    b.origin[j] = BodyOrigin::AsteroidBelt;
    assert!(compute_orbit_segments(&b, 1.0, false, false).is_empty());
}

// ===========================================================================
// an — gating (planets vs debris)
// ===========================================================================

fn planet_and_debris() -> Bodies {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let v = (G * M_SUN_ME / 5.0).sqrt();
    b.push(5.0, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    let j = b.push(5.0, 0.1, 0.0, v, 1e-6, BodyKind::Planet);
    b.origin[j] = BodyOrigin::AsteroidBelt;
    b
}

const DEBRIS_COLOR: [u8; 4] = [85, 85, 90, 110];

#[test]
fn an_planets_only_skips_debris() {
    let b = planet_and_debris();
    let segs = compute_orbit_segments(&b, 1.0, true, false);
    assert!(!segs.is_empty());
    for (_, _, c) in &segs {
        assert_ne!(*c, DEBRIS_COLOR, "debris color leaked through");
    }
}

#[test]
fn an_debris_only_skips_planets() {
    let b = planet_and_debris();
    let segs = compute_orbit_segments(&b, 1.0, false, true);
    assert!(!segs.is_empty());
    for (_, _, c) in &segs {
        assert_eq!(*c, DEBRIS_COLOR, "planet color leaked through");
    }
}

#[test]
fn an_planet_and_debris_combined() {
    let b = planet_and_debris();
    let segs = compute_orbit_segments(&b, 1.0, true, true);
    let n_debris = segs.iter().filter(|(_, _, c)| *c == DEBRIS_COLOR).count();
    let n_planet = segs.iter().filter(|(_, _, c)| *c != DEBRIS_COLOR).count();
    assert!(n_debris > 0);
    assert!(n_planet > 0);
}

// ===========================================================================
// an — orbit counts and segment closure
// ===========================================================================

#[test]
fn an_small_system_draws_all_orbits() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    for k in 0..3 {
        let r = 5.0 + k as f64;
        let v = (G * M_SUN_ME / r).sqrt();
        b.push(r, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    }
    let segs = compute_orbit_segments(&b, 0.01, true, false);
    assert_eq!(segs.len(), 3 * 32);
}

#[test]
fn an_small_debris_system_draws_all_orbits() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    for k in 0..3 {
        let r = 5.0 + k as f64;
        let v = (G * M_SUN_ME / r).sqrt();
        let j = b.push(r, 0.0, 0.0, v, 1e-6, BodyKind::Planet);
        b.origin[j] = BodyOrigin::AsteroidBelt;
    }
    let segs = compute_orbit_segments(&b, 0.01, false, true);
    assert_eq!(segs.len(), 3 * 32);
}

#[test]
fn an_segments_form_closed_loops() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let v = (G * M_SUN_ME / 5.0).sqrt();
    b.push(5.0, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    let segs = compute_orbit_segments(&b, 1.0, true, false);
    assert!(!segs.is_empty());
    for (p0, p1, _) in &segs {
        assert!(
            segs.iter().any(|(q0, _, _)| q0 == p1),
            "segment ({p0:?}, {p1:?}) has no successor",
        );
    }
    let first = segs[0].0;
    assert!(
        segs.iter().any(|(_, p1, _)| *p1 == first),
        "first point {first:?} is never reached — loop not closed",
    );
}

// ===========================================================================
// an — colour policy
// ===========================================================================

#[test]
fn an_debris_color_is_fixed() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let v = (G * M_SUN_ME / 5.0).sqrt();
    let j = b.push(5.0, 0.0, 0.0, v, 1e-6, BodyKind::Planet);
    b.origin[j] = BodyOrigin::AsteroidBelt;
    for (_, _, c) in compute_orbit_segments(&b, 1.0, false, true) {
        assert_eq!(c, DEBRIS_COLOR);
    }
}

#[test]
fn an_planet_color_varies_with_distance() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    for r in [5.0, 400.0] {
        let v = (G * M_SUN_ME / r).sqrt();
        b.push(r, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    }
    let segs = compute_orbit_segments(&b, 1.0, true, false);
    let colors: std::collections::HashSet<[u8; 4]> = segs
        .iter()
        .filter(|(_, _, c)| *c != DEBRIS_COLOR)
        .map(|(_, _, c)| *c)
        .collect();
    assert!(colors.len() >= 2, "planet colours all identical");
}

#[test]
fn an_planet_color_saturates_at_far_distance() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    for r in [1000.0, 2000.0] {
        let v = (G * M_SUN_ME / r).sqrt();
        b.push(r, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    }
    let segs = compute_orbit_segments(&b, 1.0, true, false);
    let colors: std::collections::HashSet<[u8; 4]> = segs
        .iter()
        .filter(|(_, _, c)| *c != DEBRIS_COLOR)
        .map(|(_, _, c)| *c)
        .collect();
    assert_eq!(colors.len(), 1);
    assert_eq!(colors.into_iter().next().unwrap(), [204, 140, 127, 30]);
}

#[test]
fn an_planet_color_near_endpoint() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let r = 1e-2;
    let v = (G * M_SUN_ME / r).sqrt();
    b.push(r, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    let segs = compute_orbit_segments(&b, 1.0, true, false);
    let c = segs[0].2;
    assert!((c[0] as i32 - 102).abs() <= 1, "r={}", c[0]);
    assert!((c[1] as i32 - 140).abs() <= 1, "g={}", c[1]);
    assert!((c[2] as i32 - 255).abs() <= 1, "b={}", c[2]);
    assert_eq!(c[3], 30);
}

// ===========================================================================
// an — stride policies
// ===========================================================================

#[test]
fn an_debris_stride_caps_at_256() {
    let build = |n_debris: usize| {
        let mut b = Bodies::with_capacity(n_debris + 1);
        b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
        for k in 0..n_debris {
            let r = 10.0 + (k as f64) * 0.001;
            let v = (G * M_SUN_ME / r).sqrt();
            let j = b.push(r, 0.0, 0.0, v, 1e-6, BodyKind::Planet);
            b.origin[j] = BodyOrigin::AsteroidBelt;
        }
        compute_orbit_segments(&b, 0.01, false, true)
    };
    let segs_500 = build(500);
    let segs_600 = build(600);
    assert!(
        segs_600.len() < segs_500.len(),
        "600-debris ({}) not strided below 500-debris ({})",
        segs_600.len(),
        segs_500.len(),
    );
    let ratio = segs_600.len() as f64 / segs_500.len() as f64;
    assert!(
        (0.4..=0.7).contains(&ratio),
        "expected ~0.5, got {ratio:.3}"
    );
}

#[test]
fn an_planet_stride_kicks_in_above_max() {
    let build = |n: usize| {
        let mut b = Bodies::with_capacity(n + 1);
        b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
        for k in 0..n {
            let r = 5.0 + (k % 100) as f64 * 0.1;
            let v = (G * M_SUN_ME / r).sqrt();
            b.push(r, 0.0, 0.0, v, 1.0, BodyKind::Planet);
        }
        compute_orbit_segments(&b, 0.01, true, false)
    };
    let segs_4000 = build(ORBIT_MAX_BODIES);
    let segs_4001 = build(ORBIT_MAX_BODIES + 1);
    assert_eq!(segs_4000.len(), ORBIT_MAX_BODIES * 32);
    assert_eq!(segs_4001.len(), 2001 * 32);
}
