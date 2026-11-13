//! Tests: radius laws; Barnes-Hut tree structure, force branches, accuracy,
//! and serial/parallel determinism; integrator internals (rungs, dt limits,
//! energy conservation); Kepler element helpers; and Solar-System realism.

use super::super::*;

// ===========================================================================
// f — radius laws
// ===========================================================================
/// Radius-law continuity and known anchor points: Jupiter, Mars, and the
/// Moon land within tolerance of their real radii; radius grows with mass
/// below 90 M⊕ (before degeneracy pressure kicks in); and there is no
/// discontinuity at SMALL_BODY_M.
#[test]
fn f_radius_law() {
    let rj = planet_true_r(JUPITER_MASS) / R_EARTH_GM;
    assert!((rj - 11.21).abs() < 0.2, "Jupiter {}", rj);
    let mars = planet_true_r(0.107) / R_EARTH_GM * 6371.0;
    assert!((mars - 3390.0).abs() / 3390.0 < 0.1, "Mars {}", mars);
    let moon = planet_true_r(0.0123) / R_EARTH_GM * 6371.0;
    assert!((moon - 1737.0).abs() / 1737.0 < 0.1, "Moon {}", moon);
    let mut prev = 0.0;
    let mut m = 1e-4;
    while m < 3000.0 {
        let r = planet_true_r(m);
        if m < 90.0 {
            assert!(r > prev, "radius must grow with mass below 90 M_E");
        }
        prev = r;
        m *= 1.07;
    }
    let eps = 1e-9;
    let (lo, hi) = (
        radius_earth(SMALL_BODY_M - eps),
        radius_earth(SMALL_BODY_M + eps),
    );
    assert!((lo - hi).abs() / hi < 1e-6, "jump at SMALL_BODY_M");
}

// ===========================================================================
// i — radius / survivor / primary / Chandrasekhar
// ===========================================================================
/// Per-kind radius anchors: brown dwarfs are Jupiter-sized at BD_M_MIN,
/// white dwarfs are sub-Earth at 0.6 M☉, neutron stars sit at the 10 km
/// floor, and DwarfPlanet returns the same radius as Planet at equal mass.
#[test]
fn i_radius_laws() {
    let r_bd = physical_radius(BodyKind::BrownDwarf, BD_M_MIN) / R_EARTH_GM;
    assert!((r_bd - 11.0).abs() < 1.0, "BD radius: {}", r_bd);

    let r_wd = physical_radius(BodyKind::WhiteDwarf, 0.6 * M_SUN_ME) / R_EARTH_GM;
    assert!(r_wd > 0.7 && r_wd < 2.0, "WD radius: {}", r_wd);

    let r_ns = physical_radius(BodyKind::NeutronStar, 1.4 * M_SUN_ME);
    assert!((r_ns - 1e-5).abs() < 1e-6, "NS radius: {}", r_ns);

    let r_dp = physical_radius(BodyKind::DwarfPlanet, 0.002);
    let r_p = physical_radius(BodyKind::Planet, 0.002);
    assert!((r_dp - r_p).abs() < 1e-12);
}

/// `pick_survivor` ranks by kind first, then mass. A neutron star outranks
/// a star regardless of argument order; a planet outranks a dwarf planet.
#[test]
fn i_survivor_ranking() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0 * M_SUN_ME, BodyKind::Star);
    b.push(0.001, 0.0, 0.0, 0.0, 1.4 * M_SUN_ME, BodyKind::NeutronStar);
    let (keep, _) = pick_survivor(&b, 0, 1);
    assert_eq!(b.kind[keep], BodyKind::NeutronStar, "NS outranks star");

    let mut b2 = Bodies::with_capacity(4);
    b2.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b2.push(0.001, 0.0, 0.0, 0.0, 0.5, BodyKind::DwarfPlanet);
    let (keep, _) = pick_survivor(&b2, 0, 1);
    assert_eq!(
        b2.kind[keep],
        BodyKind::Planet,
        "planet outranks dwarf planet"
    );
}

/// A brown dwarf is a valid primary: `primary_index` returns it, and
/// `find_parent` recognises planets orbiting it.
#[test]
fn i_primary_eligibility() {
    let mut b = Bodies::with_capacity(4);
    b.push(
        0.0,
        0.0,
        0.0,
        0.0,
        50.0 * JUPITER_MASS,
        BodyKind::BrownDwarf,
    );
    b.push(10.0, 0.0, 0.0, 0.3, 1.0, BodyKind::Planet);
    assert_eq!(primary_index(&b), Some(0), "BD can be primary");
    assert_eq!(find_parent(&b, 1), Some(0), "BD can host planets");
}

/// Two white dwarfs whose sum exceeds WD_M_MAX collapse to a neutron star
/// (the Type-Ia channel). The result has kind == NeutronStar and the body
/// count drops to one.
#[test]
fn i_wd_chandrasekhar_collapse() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, WD_M_MAX * 0.9, BodyKind::WhiteDwarf);
    b.push(0.005, 0.0, 0.0, 0.0, 0.6 * M_SUN_ME, BodyKind::WhiteDwarf);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1);
    assert_eq!(b.kind[0], BodyKind::NeutronStar, "should become NS");
}

// ===========================================================================
// l — Barnes-Hut: force accuracy vs direct N-body
// ===========================================================================
/// Correctness gate for the whole BH path: `accel_for` must agree with
/// `force_direct_on` to within the documented per-body error on a
/// realistic 500-body disc. A broken opening criterion, a stale node, or
/// a wrong softening constant would surface here as > 10% drift.
#[test]
fn l_bh_vs_direct_random() {
    let mut b = Bodies::with_capacity(500);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mut rng = XorShift::new(123);
    for _ in 0..499 {
        let a = 5.0 + 500.0 * rng.next_f64();
        let th = rng.next_f64() * std::f64::consts::TAU;
        let vc = (G * M_SUN_ME / a).sqrt();
        let m = 0.1 + 100.0 * rng.next_f64();
        b.push(
            a * th.cos(),
            a * th.sin(),
            -vc * th.sin(),
            vc * th.cos(),
            m,
            BodyKind::Planet,
        );
    }
    let tree = Tree::build(&b);
    let mut worst = 0.0_f64;
    let mut worst_i = 0usize;
    for i in 0..b.len() {
        let (rx, ry, _) = force_direct_on(&b, i);
        let (ax, ay, _) = tree.accel_for(&b, i);
        let rmag = (rx * rx + ry * ry).sqrt().max(1e-30);
        let dmag = ((ax - rx).powi(2) + (ay - ry).powi(2)).sqrt();
        let rel = dmag / rmag;
        if rel > worst {
            worst = rel;
            worst_i = i;
        }
    }
    assert!(
        worst < 0.10,
        "worst BH relative error {:.4} at body {} exceeds 10 %",
        worst,
        worst_i
    );
}

/// A body's own leaf must be skipped: an isolated body feels zero force
/// from itself. Guards against the walk including the query body's node.
#[test]
fn l_bh_zero_self_force() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let tree = Tree::build(&b);
    let (ax, ay, _) = tree.accel_for(&b, 0);
    assert_eq!((ax, ay), (0.0, 0.0));
}

/// Newton's third law at body level: two equal masses at ±5 must feel
/// equal-and-opposite forces. Any asymmetry (a sign bug in the split, a
/// biased acceptance criterion) would break this.
#[test]
fn l_bh_third_law() {
    let mut b = Bodies::with_capacity(2);
    b.push(-5.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(5.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    let (a0x, a0y, _) = tree.accel_for(&b, 0);
    let (a1x, a1y, _) = tree.accel_for(&b, 1);
    assert!((a0x + a1x).abs() < 1e-12, "third law x: {} {}", a0x, a1x);
    assert!((a0y + a1y).abs() < 1e-12, "third law y: {} {}", a0y, a1y);
    assert!(a0x > 0.0, "body 0 is pulled toward +x");
    assert!(a1x < 0.0, "body 1 is pulled toward -x");
}

/// Two coincident bodies exercise the collapse path (`body == -2` node).
/// Neither body may receive a NaN or infinite force, regardless of which
/// subtree the walk resolves first.
#[test]
fn l_bh_handles_coincident() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    let (ax, ay, _) = tree.accel_for(&b, 0);
    assert!(ax.is_finite() && ay.is_finite(), "body 0: {} {}", ax, ay);
    let (ax, ay, _) = tree.accel_for(&b, 1);
    assert!(ax.is_finite() && ay.is_finite(), "body 1: {} {}", ax, ay);
}

// ===========================================================================
// n — integrator (rungs, dt limits, energy)
// ===========================================================================
/// `rung_for` must land exactly on the power-of-two boundary and clamp to
/// MAX_RUNG. Infinity → 0, zero → MAX_RUNG, and just below a boundary
/// (0.99 × DT/8) must round up to the next rung.
#[test]
fn n_rung_for_boundaries() {
    assert_eq!(rung_for(f64::INFINITY, DT), 0);
    assert_eq!(rung_for(0.0, DT), MAX_RUNG);
    assert_eq!(rung_for(DT, DT), 0);
    assert_eq!(rung_for(DT / 8.0, DT), 3);
    assert_eq!(rung_for(DT / 16.0, DT), 4);
    assert_eq!(rung_for(DT / 32.0, DT), 5);
    assert_eq!(rung_for(DT / 8.0 * 0.99, DT), 4);
}

/// Both inputs of `dt_limit` are monotone: more gravity → smaller dt,
/// faster relative motion → smaller dt.
#[test]
fn n_dt_limit_monotonic() {
    let strong = dt_limit(1.0, 0.0);
    let weak = dt_limit(0.01, 0.0);
    assert!(weak > strong, "weaker gravity: {} > {}", weak, strong);
    let slow = dt_limit(0.0, 0.001);
    let fast = dt_limit(0.0, 0.1);
    assert!(slow > fast, "slower motion: {} > {}", slow, fast);
}

/// The block-leapfrog path must conserve energy to ~0.5% over 500 steps on
/// a circular orbit, and `stats.max_rung > 0` proves the block path
/// actually fired (rung-0-only runs would bypass it).
#[test]
fn n_block_leapfrog_conserves_energy() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0;
    let mu = G * M_SUN_ME;
    let v = (mu / a).sqrt();
    b.push(a, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    compute_accel(&mut b);
    rebase_energy(&mut b);
    for _ in 0..500 {
        advance(&mut b, DT);
    }
    let drift = energy_drift_pct(&b).unwrap().abs();
    assert!(drift < 0.5, "drift {}%", drift);
    assert!(
        b.stats.max_rung > 0,
        "test should exercise the block path (rung > 0)"
    );
}

// ===========================================================================
// ae — integrator internals
// ===========================================================================
/// `assign_rungs` maps `dtl[]` through `rung_for`: infinity gets rung 0,
/// `DT/8` gets rung 3. The return value is the maximum rung in the system.
#[test]
fn ae_assign_rungs_uses_dtl() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.dtl[0] = f64::INFINITY;
    b.dtl[1] = DT / 8.0;
    assert_eq!(assign_rungs(&mut b, DT), 3);
    assert_eq!(b.rung[0], 0);
    assert_eq!(b.rung[1], 3);
}

/// Flashes are aged by `FLASH_LIFE` and pruned from the buffer. A flash
/// aged by half its life survives; a full life removes it.
#[test]
fn ae_flash_lifecycle() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    flash_at(&mut b, 1.0, 2.0, 0.1);
    age_flashes(&mut b, FLASH_LIFE * 0.5);
    assert_eq!(b.flashes.len(), 1);
    age_flashes(&mut b, FLASH_LIFE);
    assert_eq!(b.flashes.len(), 0);
}

/// 4th-order Yoshida on a circular orbit: energy drift must stay under
/// 0.05% over 200 steps.
#[test]
fn ae_yoshida_energy_conservation() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0;
    let v = (G * M_SUN_ME / a).sqrt();
    b.push(a, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    compute_accel(&mut b);
    rebase_energy(&mut b);
    for _ in 0..200 {
        yoshida_step(&mut b, DT);
    }
    let drift = energy_drift_pct(&b).unwrap().abs();
    assert!(drift < 0.05, "yoshida drift {:.4}%", drift);
}

/// At equal force-evaluation cost (100 Yoshida calls = 300 leapfrog
/// steps), 4th-order integration must beat 2nd-order on energy drift.
#[test]
fn ae_yoshida_beats_leapfrog() {
    let run = |step: fn(&mut Bodies, f64), n: usize| -> f64 {
        let mut b = Bodies::with_capacity(2);
        b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
        let a = 5.0;
        let v = (G * M_SUN_ME / a).sqrt();
        b.push(a, 0.0, 0.0, v, 1.0, BodyKind::Planet);
        compute_accel(&mut b);
        rebase_energy(&mut b);
        for _ in 0..n {
            step(&mut b, DT);
        }
        energy_drift_pct(&b).unwrap().abs()
    };
    let dl = run(leapfrog_step, 300);
    let dy = run(yoshida_step, 100);
    eprintln!("leapfrog drift {:.4e}, yoshida drift {:.4e}", dl, dy);
    assert!(dy < dl, "4th-order should beat 2nd at equal cost");
}

// ===========================================================================
// ag — orbit / anomaly helpers
// ===========================================================================
/// `push_kepler(a, e)` must produce a body whose osculating elements match
/// the request: circular orbit at exactly `a`, zero eccentricity.
#[test]
fn ag_push_kepler_yields_requested_elements() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mut rng = XorShift::new(5);
    let a = 5.0;
    push_kepler(&mut b, 0, a, 0.0, 1.0, &mut rng, BodyKind::Planet);
    let o = orbit_info(&b, 1, 0);
    assert!(o.bound);
    assert!(
        (o.semi_major / a - 1.0).abs() < 1e-9,
        "a {:.8} vs {}",
        o.semi_major,
        a
    );
    assert!(o.ecc < 1e-4, "ecc {}", o.ecc);
}

/// `true_anomaly` agrees with `E` at `e = 0`, and reduces to the two fixed
/// points (`ν(0) = 0`, `ν(π) = π`) for any eccentricity.
#[test]
fn ag_true_anomaly_endpoints() {
    for k in 0..20 {
        let e_anom = k as f64 / 20.0 * std::f64::consts::TAU;
        let nu = true_anomaly(e_anom, 0.0);
        let d = (nu - e_anom).rem_euclid(std::f64::consts::TAU);
        assert!(d < 1e-9 || (std::f64::consts::TAU - d) < 1e-9);
    }
    assert!(true_anomaly(0.0, 0.3).abs() < 1e-9);
    assert!((true_anomaly(std::f64::consts::PI, 0.3) - std::f64::consts::PI).abs() < 1e-9);
}

/// Kepler's third law: 1 AU around 1 M☉ is 365.25 days.
#[test]
fn ag_orbital_period_earth_one_year() {
    let p = orbital_period_days(au_in_gm(), M_SUN_ME);
    assert!((p - 365.25).abs() < 1.0, "Earth year {}", p);
}

/// `binary_fraction` is monotone in primary mass and bounded in [0, 1].
#[test]
fn ag_binary_fraction_monotone() {
    assert!(binary_fraction(0.3) < binary_fraction(1.0));
    assert!(binary_fraction(1.0) < binary_fraction(3.0));
    for m in [0.1, 0.5, 1.0, 2.0, 5.0] {
        let f = binary_fraction(m);
        assert!((0.0..=1.0).contains(&f));
    }
}

// ===========================================================================
// ah — osculating ellipse geometry
// ===========================================================================
/// A circular orbit's osculating "ellipse" is a circle of radius `a`:
/// every returned point must sit at exactly `a` from the focus.
#[test]
fn ah_osculating_ellipse_circular() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0;
    let v = (G * M_SUN_ME / a).sqrt();
    b.push(a, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    let pts = osculating_ellipse(&b, 1, 0.0, 0.0, 0.0, 0.0, M_SUN_ME, 1.0).unwrap();
    assert!(pts.len() >= 32, "expected at least 32 points");
    for p in &pts {
        let r = ((p[0] as f64).powi(2) + (p[1] as f64).powi(2)).sqrt();
        assert!((r - a).abs() < 1e-3, "point r={} vs {}", r, a);
    }
}

/// A body above escape speed has no closed ellipse — `osculating_ellipse`
/// must return `None`.
#[test]
fn ah_osculating_ellipse_hyperbolic_none() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0;
    let v_esc = (2.0 * G * M_SUN_ME / a).sqrt();
    b.push(a, 0.0, 0.0, 2.0 * v_esc, 1.0, BodyKind::Planet);
    assert!(osculating_ellipse(&b, 1, 0.0, 0.0, 0.0, 0.0, M_SUN_ME, 1.0).is_none());
}

// ===========================================================================
// ai — accuracy (tighter than the baseline l/n/q tests)
// ===========================================================================
/// Tighter than `l_bh_vs_direct_random`: 5 random seeds, worst per-body
/// relative error must stay below 15%. The module doc's "~1%" is the mean;
/// the worst-body error at θ = 0.5 can reach ~θ² (~25% in theory) and
/// empirically ~10% here. 15% catches regressions without pretending
/// θ = 0.5 is more accurate than it is.
#[test]
fn ai_bh_error_below_documented_bound() {
    for seed in 0..5u32 {
        let mut b = Bodies::with_capacity(400);
        b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
        let mut rng = XorShift::new(seed);
        for _ in 0..399 {
            let a = 5.0 + 200.0 * rng.next_f64();
            let th = rng.next_f64() * std::f64::consts::TAU;
            let vc = (G * M_SUN_ME / a).sqrt();
            let m = 0.1 + 50.0 * rng.next_f64();
            b.push(
                a * th.cos(),
                a * th.sin(),
                -vc * th.sin(),
                vc * th.cos(),
                m,
                BodyKind::Planet,
            );
        }
        let tree = Tree::build(&b);
        let mut worst = 0.0_f64;
        for i in 0..b.len() {
            let (rx, ry, _) = force_direct_on(&b, i);
            let (ax, ay, _) = tree.accel_for(&b, i);
            let rmag = (rx * rx + ry * ry).sqrt().max(1e-30);
            let dmag = ((ax - rx).powi(2) + (ay - ry).powi(2)).sqrt();
            worst = worst.max(dmag / rmag);
        }
        assert!(
            worst < 0.15,
            "seed {}: worst relative BH error {:.4}",
            seed,
            worst
        );
    }
}

/// Solar System drift over 500 steps must stay under 0.05% — the expected
/// behaviour of the 4th-order integrator, tighter than `q_*`'s 0.5%.
#[test]
fn ai_solar_energy_drift_tight() {
    let mut b = Bodies::with_capacity(16);
    spawn_solar_system(&mut b);
    let mut worst = 0.0_f64;
    for _ in 0..500 {
        advance(&mut b, DT);
        if let Some(d) = energy_drift_pct(&b) {
            worst = worst.max(d.abs());
        }
    }
    assert!(worst < 0.05, "solar drift {:.4}%", worst);
}

/// A planet on an `e = 0.3` ellipse must return to the same `(a, e)` after
/// one full orbit to within 2% — the accuracy contract for element
/// preservation over a single period.
#[test]
fn ai_orbital_elements_preserved_over_one_orbit() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let (x, y, vx, vy) = kepler_state(G * M_SUN_ME, 5.0, 0.3, 1.0, 0.0);
    b.push(x, y, vx, vy, 1.0, BodyKind::Planet);
    let o0 = orbit_info(&b, 1, 0);
    compute_accel(&mut b);
    rebase_energy(&mut b);
    let n = (o0.period / DT) as usize;
    for _ in 0..n {
        advance(&mut b, DT);
    }
    let o1 = orbit_info(&b, 1, 0);
    assert!(
        (o1.semi_major / o0.semi_major - 1.0).abs() < 0.02,
        "a: {} -> {}",
        o0.semi_major,
        o1.semi_major
    );
    assert!(
        (o1.ecc - o0.ecc).abs() < 0.02,
        "e: {} -> {}",
        o0.ecc,
        o1.ecc
    );
}

// ===========================================================================
// aj — realism: Solar System matches known periods
// ===========================================================================
/// Preset realism: the 8 planets of `spawn_solar_system` must reproduce
/// their real sidereal periods to 5%. Catches a wrong AU constant, a
/// wrong central mass, or a swapped orbit.
#[test]
fn aj_solar_system_reproduces_real_periods() {
    let mut b = Bodies::with_capacity(16);
    spawn_solar_system(&mut b);
    assert!(b.len() >= 9, "expected at least 9 bodies, got {}", b.len());
    let expected = [
        (0.0553_f64, 87.97_f64),
        (0.8150, 224.70),
        (1.0000, 365.25),
        (0.1070, 686.98),
        (317.80, 4332.59),
        (95.160, 10759.22),
        (14.540, 30688.5),
        (17.150, 60182.0),
    ];
    for &(m_target, real) in &expected {
        let body = (1..b.len())
            .find(|&i| (b.m[i] - m_target).abs() < 1e-3)
            .unwrap_or_else(|| panic!("no body with mass ~{} M⊕", m_target));
        let o = orbit_info(&b, body, 0);
        let p_days = o.period * time_unit_s() / 86_400.0;
        assert!(
            (p_days / real - 1.0).abs() < 0.05,
            "body {} (m={} M⊕): {:.1} d vs {:.1} d",
            body,
            m_target,
            p_days,
            real
        );
    }
}

// ===========================================================================
// l — Barnes-Hut: tree structure (node kinds, partitioning, collapse)
// ===========================================================================

/// Empty input → empty node array. No phantom root, no panic.
#[test]
fn l_bh_build_empty() {
    let b = Bodies::with_capacity(0);
    let tree = Tree::build(&b);
    assert!(tree.nodes.is_empty());
}

/// One body → one leaf. No internal root, no phantoms, COM = position.
#[test]
fn l_bh_build_single_body() {
    let mut b = Bodies::with_capacity(1);
    b.push(1.0, 2.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    assert_eq!(tree.nodes.len(), 1);
    let n = &tree.nodes[0];
    assert_eq!(n.body, 0, "single body becomes leaf");
    assert_eq!(n.m, 1.0);
    assert_eq!((n.com_x, n.com_y), (1.0, 2.0));
    assert!(n.children.iter().all(|&c| c < 0));
}

/// Two coincident bodies must collapse to exactly one `body == -2` node
/// carrying the aggregate mass and COM. This is the canonical
/// MIN_NODE_SIZE path — exact coincidence always collapses.
#[test]
fn l_bh_build_coincident_collapses() {
    let mut b = Bodies::with_capacity(2);
    b.push(1.0, 1.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(1.0, 1.0, 0.0, 0.0, 2.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    let collapsed: Vec<&Node> = tree.nodes.iter().filter(|n| n.body == -2).collect();
    assert_eq!(collapsed.len(), 1, "exactly one coincident-collapse node");
    assert!(
        (collapsed[0].m - 3.0).abs() < 1e-15,
        "mass {}",
        collapsed[0].m
    );
    assert!((collapsed[0].com_x - 1.0).abs() < 1e-15);
    assert!((collapsed[0].com_y - 1.0).abs() < 1e-15);
    assert!(collapsed[0].children.iter().all(|&c| c < 0));
}

/// One body per quadrant → internal root with all four children filled.
/// Empty quadrants would receive zero-mass phantom nodes instead.
#[test]
fn l_bh_build_all_quadrants() {
    let mut b = Bodies::with_capacity(4);
    for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        b.push(x, y, 0.0, 0.0, 1.0, BodyKind::Planet);
    }
    let tree = Tree::build(&b);
    assert_eq!(tree.nodes[0].body, -1, "root is internal");
    assert!(
        tree.nodes[0].children.iter().all(|&c| c >= 0),
        "all four quadrants must be filled"
    );
    assert!((tree.nodes[0].m - 4.0).abs() < 1e-15);
    assert!(tree.nodes[0].com_x.abs() < 1e-15);
    assert!(tree.nodes[0].com_y.abs() < 1e-15);
}

/// Bodies at (-1, 0) and (1, 0) put the root cell at (0, 0), so both
/// bodies have `y == cy`. The `< cy` split is strict, so `0 < 0` is false
/// and both go high-y (q2, q3). Empty quadrants receive zero-mass
/// phantom nodes (`m == 0`, `body == -1`), not `-1` child indices — so
/// the discriminator between "empty" and "filled" is the child's mass,
/// not its index sign.
#[test]
fn l_bh_build_boundary_uses_strict_less_than() {
    let mut b = Bodies::with_capacity(2);
    b.push(-1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    assert_eq!(tree.nodes[0].body, -1);

    let child_m = |q: usize| -> f64 {
        let ci = tree.nodes[0].children[q];
        assert!(ci >= 0, "q{q} must have a child node (phantom or real)");
        tree.nodes[ci as usize].m
    };
    assert_eq!(
        child_m(0),
        0.0,
        "q0 (low-x low-y) must be a zero-mass phantom"
    );
    assert_eq!(
        child_m(1),
        0.0,
        "q1 (high-x low-y) must be a zero-mass phantom"
    );
    assert!(child_m(2) > 0.0, "q2 (low-x high-y) must hold a body");
    assert!(child_m(3) > 0.0, "q3 (high-x high-y) must hold a body");
}

/// Sub-MIN_NODE_SIZE clusters must not cause unbounded recursion. The
/// tree either collapses them or splits them into leaves — both are
/// valid terminations. What MIN_NODE_SIZE guarantees is boundedness: a
/// small node count and finite forces on every body.
#[test]
fn l_bh_build_min_node_size_terminates() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(1e-13, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(2e-13, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(100.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    assert!(
        tree.nodes.len() < 500,
        "tree blew up to {} nodes — MIN_NODE_SIZE not honoured",
        tree.nodes.len()
    );
    for i in 0..b.len() {
        let (ax, ay, _) = tree.accel_for(&b, i);
        assert!(ax.is_finite() && ay.is_finite(), "body {i}: {ax} {ay}");
    }
}

/// Bodies in a line force subdivision at every level: the tree must
/// actually reflect the point layout, not flatten a skewed input.
#[test]
fn l_bh_build_skewed_subdivides() {
    let mut b = Bodies::with_capacity(4);
    for k in 0..4 {
        b.push(k as f64, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    }
    let tree = Tree::build(&b);
    let internal = tree.nodes.iter().filter(|n| n.body == -1).count();
    assert!(internal >= 2, "expected >=2 internal nodes, got {internal}");
}

// ===========================================================================
// l — Barnes-Hut: accel_for branches (opening, edges, timestep)
// ===========================================================================

/// Far-body / small-cluster: the whole tree is accepted at the root and
/// the result matches the direct COM force well below the θ = 0.5 bound.
#[test]
fn l_bh_accel_accepts_far_cluster() {
    let mut b = Bodies::with_capacity(5);
    for k in 0..4 {
        let th = std::f64::consts::TAU * k as f64 / 4.0;
        b.push(th.cos(), th.sin(), 0.0, 0.0, 1.0, BodyKind::Planet);
    }
    b.push(100.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    let (ax, ay, _) = tree.accel_for(&b, 4);
    let (dx, dy, _) = force_direct_on(&b, 4);
    let err = (ax - dx).hypot(ay - dy) / dx.hypot(dy);
    assert!(err < 1e-3, "far-cluster relative error {err:e}");
}

/// Body at the cluster centre: the root fails θ, and the walk must open
/// all four leaves, matching the direct sum to numerical precision.
#[test]
fn l_bh_accel_rejects_close_cluster() {
    let mut b = Bodies::with_capacity(5);
    for k in 0..4 {
        let th = std::f64::consts::TAU * k as f64 / 4.0;
        b.push(th.cos(), th.sin(), 0.0, 0.0, 1.0, BodyKind::Planet);
    }
    b.push(0.01, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    let (ax, ay, _) = tree.accel_for(&b, 4);
    let (dx, dy, _) = force_direct_on(&b, 4);
    let err = (ax - dx).hypot(ay - dy) / dx.hypot(dy);
    assert!(err < 1e-10, "close-cluster relative error {err:e}");
}

/// A probe at `0.5·SOFT` from a coincident pair must receive a **finite**
/// force. Whether the pair collapses to a `body == -2` node (which would
/// trigger the `4·SOFT2` skip) or splits into leaves depends on the
/// surrounding tree layout; in the split case each body contributes as a
/// normal leaf. The finite-force contract is what the integrator needs.
#[test]
fn l_bh_accel_coincident_near_probe_finite() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(0.5 * SOFT2.sqrt(), 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    let (ax, ay, _) = tree.accel_for(&b, 2);
    assert!(ax.is_finite() && ay.is_finite(), "ax {ax}, ay {ay}");
}

/// Far from a coincident cluster: accepted as a single point mass at the
/// cluster COM. Since the cluster's COM equals every body's position
/// here, this is exact — must match direct sum to ~1e-12.
#[test]
fn l_bh_accel_coincident_cluster_far() {
    let mut b = Bodies::with_capacity(4);
    for _ in 0..3 {
        b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    }
    b.push(10.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    let (ax, ay, _) = tree.accel_for(&b, 3);
    let (dx, dy, _) = force_direct_on(&b, 3);
    let err = (ax - dx).hypot(ay - dy) / dx.hypot(dy);
    assert!(err < 1e-12, "coincident-cluster error {err:e}");
}

/// Zero-mass phantom nodes in empty quadrants must contribute nothing to
/// the force: the diagonal pair equals the direct sum to ~1e-12.
#[test]
fn l_bh_accel_zero_mass_phantom_ignored() {
    let mut b = Bodies::with_capacity(2);
    b.push(-1.0, -1.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(1.0, 1.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    let (ax, ay, _) = tree.accel_for(&b, 0);
    let (dx, dy, _) = force_direct_on(&b, 0);
    let err = (ax - dx).hypot(ay - dy) / dx.hypot(dy);
    assert!(err < 1e-12, "phantom leaked into force: {err:e}");
}

/// A body at 1e6 Gm from a tight pair is far enough that the entire tree
/// collapses to the root COM. Must equal the direct sum to ~1e-10.
#[test]
fn l_bh_accel_root_accepted_at_infinity() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(0.1, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(1e6, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    let (ax, ay, _) = tree.accel_for(&b, 2);
    let (dx, dy, _) = force_direct_on(&b, 2);
    let err = (ax - dx).hypot(ay - dy) / dx.hypot(dy);
    assert!(err < 1e-10, "COM collapse error {err:e}");
}

/// A two-body, unequal-mass tree must reproduce Newton's law exactly:
/// same softening constant, same formula, same sign.
#[test]
fn l_bh_accel_matches_newton_2body() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(5.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    let (ax, ay, _) = tree.accel_for(&b, 1);
    let d2 = 25.0 + SOFT2;
    let d3 = d2 * d2.sqrt();
    let expected_ax = -G * M_SUN_ME * 5.0 / d3;
    assert!(
        (ax - expected_ax).abs() / expected_ax.abs() < 1e-12,
        "ax {ax} expected {expected_ax}"
    );
    assert!(ay.abs() < 1e-15);
}

/// `accel_for`'s third return (the timestep limit) must equal
/// `dt_limit(s_max, 0)` on a static 2-body pair: `w_max = 0` because the
/// relative velocity is zero, so only the gravity term contributes.
#[test]
fn l_bh_accel_dt_limit_2body() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(5.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    let (_, _, dt) = tree.accel_for(&b, 0);
    let d2 = 25.0 + SOFT2;
    let d3 = d2 * d2.sqrt();
    let s_max = 2.0 / d3;
    let expected = dt_limit(s_max, 0.0);
    assert!((dt - expected).abs() / expected < 1e-12);
}

/// A single body has no interactions: `s_max == w_max == 0`, so `dt_limit`
/// returns `+∞`. This is what lets the integrator skip isolated bodies.
#[test]
fn l_bh_accel_single_body_dt_infinite() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    let (ax, ay, dt) = tree.accel_for(&b, 0);
    assert_eq!((ax, ay), (0.0, 0.0));
    assert!(dt.is_infinite(), "expected inf, got {dt}");
}

/// Sweep 20 randomised configurations: `accel_for` must never return NaN
/// or infinite for finite input, and `dt` must be non-negative (or `+∞`).
#[test]
fn l_bh_accel_never_nan_across_configs() {
    for seed in 0..20u32 {
        let mut b = Bodies::with_capacity(50);
        b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
        let mut rng = XorShift::new(seed);
        for _ in 0..49 {
            let a = 0.5 + 50.0 * rng.next_f64();
            let th = rng.next_f64() * std::f64::consts::TAU;
            let vc = (G * M_SUN_ME / a).sqrt();
            let m = 0.01 + 10.0 * rng.next_f64();
            b.push(
                a * th.cos(),
                a * th.sin(),
                -vc * th.sin(),
                vc * th.cos(),
                m,
                BodyKind::Planet,
            );
        }
        let tree = Tree::build(&b);
        for i in 0..b.len() {
            let (ax, ay, dt) = tree.accel_for(&b, i);
            assert!(ax.is_finite(), "seed {seed} body {i} ax = {ax}");
            assert!(ay.is_finite(), "seed {seed} body {i} ay = {ay}");
            assert!(
                dt >= 0.0 || dt.is_infinite(),
                "seed {seed} body {i} dt = {dt}"
            );
        }
    }
}

// ===========================================================================
// l — Barnes-Hut: determinism (serial vs parallel build)
// ===========================================================================

/// The recursive builder claims bit-identical output regardless of rayon
/// thread count. Force the parallel path with 600 bodies and compare a
/// default-threaded build against a single-threaded one — every field,
/// bit for bit. A nondeterministic reduction in the COM aggregation or a
/// race in the quadrant partition would surface here.
#[test]
fn l_bh_serial_parallel_identical() {
    let mut b = Bodies::with_capacity(600);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mut rng = XorShift::new(123);
    for _ in 0..599 {
        let a = 5.0 + 500.0 * rng.next_f64();
        let th = rng.next_f64() * std::f64::consts::TAU;
        let vc = (G * M_SUN_ME / a).sqrt();
        b.push(
            a * th.cos(),
            a * th.sin(),
            -vc * th.sin(),
            vc * th.cos(),
            1.0,
            BodyKind::Planet,
        );
    }
    let multi = Tree::build(&b);
    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| Tree::build(&b));

    assert_eq!(multi.nodes.len(), single.nodes.len(), "node counts differ");
    for (i, (p, s)) in multi.nodes.iter().zip(single.nodes.iter()).enumerate() {
        assert_eq!(p.cx.to_bits(), s.cx.to_bits(), "node {i} cx");
        assert_eq!(p.cy.to_bits(), s.cy.to_bits(), "node {i} cy");
        assert_eq!(p.size.to_bits(), s.size.to_bits(), "node {i} size");
        assert_eq!(p.m.to_bits(), s.m.to_bits(), "node {i} m");
        assert_eq!(p.com_x.to_bits(), s.com_x.to_bits(), "node {i} com_x");
        assert_eq!(p.com_y.to_bits(), s.com_y.to_bits(), "node {i} com_y");
        assert_eq!(p.body, s.body, "node {i} body");
        assert_eq!(p.children, s.children, "node {i} children");
    }
}

// ===========================================================================
// n — dt_limit: every branch of min(t_dyn, t_vel)
// ===========================================================================

/// Pure gravity: `w_max = 0` → only the dynamical-time term survives.
#[test]
fn n_dt_limit_pure_gravity() {
    let dt = dt_limit(1.0, 0.0);
    let expected = ETA_GRAV * (1.0 / G).sqrt();
    assert!((dt - expected).abs() < 1e-12);
}

/// Pure velocity: `s_max = 0` → only the crossing-time term survives.
#[test]
fn n_dt_limit_pure_velocity() {
    let dt = dt_limit(0.0, 4.0);
    let expected = ETA_VEL * 0.5;
    assert!((dt - expected).abs() < 1e-12);
}

/// Both terms finite → the result is exactly `min(t_dyn, t_vel)`.
#[test]
fn n_dt_limit_takes_min() {
    let (s_max, w_max) = (1e6, 1e-6);
    let dt = dt_limit(s_max, w_max);
    let t_dyn = ETA_GRAV * (1.0 / (G * s_max)).sqrt();
    let t_vel = ETA_VEL * (1.0 / w_max).sqrt();
    assert!((dt - t_dyn.min(t_vel)).abs() < 1e-15);
}

/// Both zero → both terms infinite → result infinite.
#[test]
fn n_dt_limit_zero_both() {
    assert!(dt_limit(0.0, 0.0).is_infinite());
}
