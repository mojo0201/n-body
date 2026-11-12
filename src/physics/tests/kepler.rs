//! Tests: radius laws, Barnes-Hut vs direct, integrator internals, accuracy, Kepler helpers.

use super::super::*;

// ===========================================================================
// f — radius laws
// ===========================================================================
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
// l — Barnes-Hut vs direct
// ===========================================================================
/// `Tree::accel_for` must agree with `force_direct_on` to within the
/// documented BH error (~1 % for θ = 0.5) on a realistic configuration.
/// This is the correctness gate for the entire BH path — if it drifts, the
/// physics still runs but with the wrong answer.
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

/// A body's own tree leaf must be skipped; a body alone in the tree must
/// feel zero force from itself.
#[test]
fn l_bh_zero_self_force() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let tree = Tree::build(&b);
    let (ax, ay, _) = tree.accel_for(&b, 0);
    assert_eq!((ax, ay), (0.0, 0.0));
}

/// BH must respect Newton's third law at the body level for a 2-body system:
/// a_0 + a_1 = 0 and the force points from heavy → light.
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

/// Two coincident bodies stress the tree's `coincident` node path. Neither
/// body should get a NaN/inf force.
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
// n — integrator
// ===========================================================================
/// `rung_for(lim, dt)` must land exactly on the documented boundary
/// (dt/2^k for integer k) and clamp to MAX_RUNG.
#[test]
fn n_rung_for_boundaries() {
    assert_eq!(rung_for(f64::INFINITY, DT), 0);
    assert_eq!(rung_for(0.0, DT), MAX_RUNG);
    assert_eq!(rung_for(DT, DT), 0);
    assert_eq!(rung_for(DT / 8.0, DT), 3);
    assert_eq!(rung_for(DT / 16.0, DT), 4);
    assert_eq!(rung_for(DT / 32.0, DT), 5);
    // Just below a power-of-two boundary pushes to the next rung.
    assert_eq!(rung_for(DT / 8.0 * 0.99, DT), 4);
}

/// `dt_limit` is monotonically decreasing in both the gravitational term
/// and the velocity term.
#[test]
fn n_dt_limit_monotonic() {
    // Stronger gravity -> smaller dt.
    let strong = dt_limit(1.0, 0.0);
    let weak = dt_limit(0.01, 0.0);
    assert!(weak > strong, "weaker gravity: {} > {}", weak, strong);
    // Faster relative motion -> smaller dt.
    let slow = dt_limit(0.0, 0.001);
    let fast = dt_limit(0.0, 0.1);
    assert!(slow > fast, "slower motion: {} > {}", slow, fast);
}

/// The block-leapfrog path (`kmax > 0`) must conserve energy to ~0.5 %
/// over hundreds of steps on a circular orbit.
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

#[test]
fn ae_yoshida_beats_leapfrog() {
    // Equal force-evaluation cost: 100 Yoshida calls = 300 leapfrog steps.
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
#[test]
fn ag_push_kepler_yields_requested_elements() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mut rng = XorShift::new(5);
    let a = 5.0;
    push_kepler(&mut b, 0, a, 0.0, 1.0, &mut rng, BodyKind::Planet);
    let o = orbit_info(&b, 1, 0);
    assert!(o.bound);
    // push_kepler and orbit_info both use G * (m_star + m_planet).
    assert!(
        (o.semi_major / a - 1.0).abs() < 1e-9,
        "a {:.8} vs {}",
        o.semi_major,
        a
    );
    assert!(o.ecc < 1e-4, "ecc {}", o.ecc);
}

#[test]
fn ag_true_anomaly_endpoints() {
    // Circular: ν = E
    for k in 0..20 {
        let e_anom = k as f64 / 20.0 * std::f64::consts::TAU;
        let nu = true_anomaly(e_anom, 0.0);
        let d = (nu - e_anom).rem_euclid(std::f64::consts::TAU);
        assert!(d < 1e-9 || (std::f64::consts::TAU - d) < 1e-9);
    }
    assert!(true_anomaly(0.0, 0.3).abs() < 1e-9);
    assert!((true_anomaly(std::f64::consts::PI, 0.3) - std::f64::consts::PI).abs() < 1e-9);
}

#[test]
fn ag_orbital_period_earth_one_year() {
    let p = orbital_period_days(au_in_gm(), M_SUN_ME);
    assert!((p - 365.25).abs() < 1.0, "Earth year {}", p);
}

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
#[test]
fn ai_bh_error_below_documented_bound() {
    // Worst *per-body* error at θ = 0.5. The module doc's "~1 %" is the
    // mean error; the worst-case body can reach ~θ² (~25 %), and empirically
    // ~10 % for this configuration. 15 % catches regressions without
    // pretending θ = 0.5 is more accurate than it is.
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

#[test]
fn ai_solar_energy_drift_tight() {
    // 4th-order integrator should keep the Solar System within 0.05 %
    // energy over hundreds of steps.
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
#[test]
fn aj_solar_system_reproduces_real_periods() {
    let mut b = Bodies::with_capacity(16);
    spawn_solar_system(&mut b);
    assert!(b.len() >= 9, "expected at least 9 bodies, got {}", b.len());
    // (mass in M_E as it appears in solar_system.json, real sidereal period in days)
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
