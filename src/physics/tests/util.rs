//! Tests: RNG, samplers, geometry, naming, star properties, small utilities.

use super::super::*;
use super::shared::*;

// ===========================================================================
// u — RNG / samplers
// ===========================================================================
/// Same seed -> same sequence; different seeds differ within 100 draws.
#[test]
fn u_xor_shift_deterministic() {
    let mut a = XorShift::new(42);
    let mut b = XorShift::new(42);
    for k in 0..100 {
        assert_eq!(a.next_u32(), b.next_u32(), "same seed, draw {}", k);
    }
    let mut a = XorShift::new(42);
    let mut c = XorShift::new(43);
    let mut differ = false;
    for _ in 0..100 {
        if a.next_u32() != c.next_u32() {
            differ = true;
            break;
        }
    }
    assert!(differ, "different seeds should diverge");
}

/// The Kroupa IMF sampler must actually span its declared range
/// [0.08, 50] M_sun and never produce a value outside it.
#[test]
fn u_imf_spans_range() {
    let mut rng = XorShift::new(7);
    let mut min_m = f64::INFINITY;
    let mut max_m = f64::NEG_INFINITY;
    for _ in 0..10_000 {
        let m = sample_imf_mass(&mut rng);
        assert!(
            (0.08 - 1e-9..=50.0 + 1e-9).contains(&m),
            "out of range {}",
            m
        );
        min_m = min_m.min(m);
        max_m = max_m.max(m);
    }
    assert!(min_m < 0.1, "min {} never near the low end", min_m);
    assert!(max_m > 5.0, "max {} never near the high end", max_m);
}

/// The Poisson sampler's empirical mean matches its rate parameter.
#[test]
fn u_poisson_mean() {
    let mut rng = XorShift::new(11);
    let lambda = 3.0;
    let n = 5000u64;
    let sum: u64 = (0..n).map(|_| poisson(&mut rng, lambda) as u64).sum();
    let mean = sum as f64 / n as f64;
    // stddev of mean ≈ sqrt(3 / 5000) ≈ 0.0245; tolerance 0.2 is ~8 sigma.
    assert!(
        (mean - lambda).abs() < 0.2,
        "mean {} vs lambda {}",
        mean,
        lambda
    );
}

/// `solve_kepler(M, e)` must satisfy Kepler's equation M = E - e sin E to
/// machine precision across a uniform sweep of M.
#[test]
fn u_kepler_equation_roundtrip() {
    let e = 0.3;
    for k in 0..20 {
        let m = k as f64 / 20.0 * std::f64::consts::TAU;
        let ea = solve_kepler(m, e);
        let m_verify = ea - e * ea.sin();
        assert!((m_verify - m).abs() < 1e-10, "M={} vs {}", m, m_verify);
    }
}

// ===========================================================================
// v — geometry helpers
// ===========================================================================
/// `interact_frac` is bounded by [0, 1] and decreases monotonically with
/// impact parameter.
#[test]
fn v_interact_frac_bounds() {
    for rp in [0.001, 0.01, 0.1] {
        let rt = rp * 2.0;
        let mut prev = f64::INFINITY;
        for k in 0..=100 {
            let bb = k as f64 / 100.0;
            let f = interact_frac(rt, rp, bb);
            assert!((0.0..=1.0).contains(&f), "bb {} -> {}", bb, f);
            assert!(f <= prev + 1e-9, "bb {} -> {} (was {})", bb, f, prev);
            prev = f;
        }
        // Head-on: full interaction.
        assert!((interact_frac(rt, rp, 0.0) - 1.0).abs() < 1e-9);
        // Grazing: minimal interaction.
        assert!(interact_frac(rt, rp, 1.0) < 0.05);
    }
}

/// A perfectly head-on encounter has impact parameter zero.
#[test]
fn v_impact_offset_head_on_zero() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(-1.0, 0.0, 1.0, 0.0, 1.0, BodyKind::Planet);
    // dx = -1, dy = 0, vx = 1, vy = 0 -> offset = |(-1)*0 - 0*1| / 1 = 0.
    assert!((impact_offset(&b, 0, 1) - 0.0).abs() < 1e-10);
}

/// A purely tangential encounter has impact parameter equal to the
/// separation.
#[test]
fn v_impact_offset_grazing_max() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(-1.0, 0.0, 0.0, 1.0, 1.0, BodyKind::Planet);
    // dx = -1, dy = 0, vx = 0, vy = 1 -> offset = |(-1)*1 - 0*0| / 1 = 1.
    assert!((impact_offset(&b, 0, 1) - 1.0).abs() < 1e-10);
}

/// Earth's Hill radius around the Sun must come out to ~0.01 AU.
#[test]
fn v_hill_radius_earth_sun() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    b.push(au, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let r = hill_radius(&b, 1, 0);
    let expected = au * (1.0 / (3.0 * M_SUN_ME)).cbrt();
    assert!(
        (r - expected).abs() < 1e-9 * au,
        "hill {} vs {}",
        r,
        expected
    );
    // Sanity: about 0.01 AU.
    assert!((r / au - 0.01).abs() < 1e-4);
}

/// The Roche limit of an Earth-density body around the Sun must lie
/// between the rigid (1.26 × R) and fluid (2.44 × R) coefficient bounds,
/// scaled by (M/m)^(1/3).
#[test]
fn v_roche_limit_earth_sun() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    b.push(au, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let roche = roche_limit(&b, 1, 0);
    let r_e = planet_true_r(1.0);
    let scale = (M_SUN_ME / 1.0).cbrt();
    let lo = ROCHE_RIGID * r_e * scale;
    let hi = ROCHE_FLUID * r_e * scale;
    assert!(
        roche >= lo - 1e-9 && roche <= hi + 1e-9,
        "roche {} not in [{}, {}]",
        roche,
        lo,
        hi
    );
    // Earth has rho ≈ 5.5 g/cm³, above ROCHE_RHO_RIGID = 4, so it should
    // be fully rigid: roche ≈ ROCHE_RIGID * r_e * scale.
    let expected_rigid = ROCHE_RIGID * r_e * scale;
    assert!(
        (roche - expected_rigid).abs() / expected_rigid < 0.01,
        "Earth should be fully rigid: {} vs {}",
        roche,
        expected_rigid
    );
}

// ===========================================================================
// w — naming helpers
// ===========================================================================
#[test]
fn w_planet_class_thresholds() {
    assert_eq!(planet_class(0.5), "rocky");
    assert_eq!(planet_class(4.999), "rocky");
    assert_eq!(planet_class(5.0), "super-earth");
    assert_eq!(planet_class(9.999), "super-earth");
    assert_eq!(planet_class(10.0), "ice giant");
    assert_eq!(planet_class(49.999), "ice giant");
    assert_eq!(planet_class(50.0), "gas giant");
    assert_eq!(planet_class(317.999), "gas giant");
    assert_eq!(planet_class(318.0), "super-jupiter");
}

#[test]
fn w_star_class_thresholds() {
    assert_eq!(star_class(0.1 * M_SUN_ME), 'M');
    assert_eq!(star_class(0.6 * M_SUN_ME), 'K');
    assert_eq!(star_class(1.0 * M_SUN_ME), 'G');
    assert_eq!(star_class(1.3 * M_SUN_ME), 'F');
    assert_eq!(star_class(2.0 * M_SUN_ME), 'A');
    assert_eq!(star_class(5.0 * M_SUN_ME), 'B');
}

#[test]
fn w_bd_class_thresholds() {
    assert_eq!(bd_class(60.0 * JUPITER_MASS), 'M');
    assert_eq!(bd_class(40.0 * JUPITER_MASS), 'L');
    assert_eq!(bd_class(25.0 * JUPITER_MASS), 'T');
    assert_eq!(bd_class(15.0 * JUPITER_MASS), 'Y');
}

#[test]
fn w_mass_str_formats() {
    let s = mass_str(M_SUN_ME);
    assert!(s.contains("M☉") && s.contains("1.00"), "{}", s);
    let s = mass_str(1.0);
    assert!(s.contains("M⊕") && s.contains("1.00"), "{}", s);
    let s = mass_str(300.0);
    assert!(s.contains("M⊕") && s.contains("300"), "{}", s);
}

#[test]
fn w_body_class_moon_vs_planet() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0 * au_in_gm();
    let v = (G * M_SUN_ME / a).sqrt();
    // 200 M_E: comfortably a gas giant (50 <= m < 318).
    b.push(a, 0.0, 0.0, v, 200.0, BodyKind::Planet);
    let v_moon = v + (G * 200.0 / 0.1).sqrt();
    b.push(a + 0.1, 0.0, 0.0, v_moon, 0.01, BodyKind::Planet);
    b.refresh_parent_cache();
    assert_eq!(body_class(&b, 1), "gas giant");
    assert_eq!(body_class(&b, 2), "moon");
}

// ===========================================================================
// x — star & habitability
// ===========================================================================
#[test]
fn x_luminosity_mass_law() {
    assert!((star_luminosity(BodyKind::Star, M_SUN_ME) - 1.0).abs() < 1e-6);
    // 10 M☉ → 10^3.5 = 3162
    let l10 = star_luminosity(BodyKind::Star, 10.0 * M_SUN_ME);
    assert!((l10 / 3162.28 - 1.0).abs() < 0.01, "10 M☉ L = {}", l10);
    assert!(star_luminosity(BodyKind::Planet, 1.0) == 0.0);
}

#[test]
fn x_effective_temperature() {
    let t = star_temperature(BodyKind::Star, M_SUN_ME);
    assert!((t - 5772.0).abs() < 10.0, "sun T = {}", t);
    // M dwarf: T ∝ m^0.475, so 0.2 M☉ → ~2,900 K.
    let t02 = star_temperature(BodyKind::Star, 0.2 * M_SUN_ME);
    assert!(t02 < 4000.0, "0.2 M☉ T = {}", t02);
    // 5 M☉: 5^0.475 = 2.15 → ~12,400 K.
    let t5 = star_temperature(BodyKind::Star, 5.0 * M_SUN_ME);
    assert!(t5 > 12_000.0, "5 M☉ T = {}", t5);
    // 10 M☉: 10^0.475 = 2.99 → ~17,200 K.
    let t10 = star_temperature(BodyKind::Star, 10.0 * M_SUN_ME);
    assert!(t10 > 16_000.0, "10 M☉ T = {}", t10);
}

#[test]
fn x_main_sequence_lifetime() {
    assert!((star_lifetime_gyr(BodyKind::Star, M_SUN_ME) - 10.0).abs() < 0.01);
    let l10 = star_lifetime_gyr(BodyKind::Star, 10.0 * M_SUN_ME);
    assert!((l10 - 0.0316).abs() < 1e-3, "10 M☉ lifetime {}", l10);
    assert!(star_lifetime_gyr(BodyKind::BrownDwarf, BD_M_MIN).is_infinite());
}

#[test]
fn x_habitable_zone_and_flux() {
    let (inner, outer) = habitable_zone_au(BodyKind::Star, M_SUN_ME);
    assert!(inner < 1.0 && outer > 1.0, "HZ {:?}..{:?}", inner, outer);
    let (i, o) = habitable_zone_au(BodyKind::Planet, 1.0);
    assert!(i.is_nan() && o.is_nan());

    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    assert!((stellar_flux(&b, au, 0.0, 1) - 1.0).abs() < 1e-6);
    assert!((stellar_flux(&b, 2.0 * au, 0.0, 1) - 0.25).abs() < 1e-6);
    assert_eq!(stellar_flux(&b, 0.0, 0.0, 0), 0.0);
    // Earth's equilibrium temperature should be ~255 K.
    assert!((equilibrium_temp(1.0) - 255.0).abs() < 1.0);
}

// ===========================================================================
// y — samplers
// ===========================================================================
#[test]
fn y_rayleigh_mean_matches_theory() {
    let mut rng = XorShift::new(3);
    let sigma = 0.1_f64;
    let n = 20_000;
    let mean: f64 = (0..n)
        .map(|_| sample_rayleigh(&mut rng, sigma))
        .sum::<f64>()
        / n as f64;
    let expected = sigma * (std::f64::consts::PI / 2.0).sqrt();
    assert!(
        (mean - expected).abs() < 0.01,
        "mean {} vs {}",
        mean,
        expected
    );
}

#[test]
fn y_power_law_in_range_and_biased_low() {
    let mut rng = XorShift::new(5);
    let (lo, hi, p) = (0.1, 5.0, 1.5);
    let mut below_median = 0;
    for _ in 0..2000 {
        let m = sample_mass_power_law(&mut rng, lo, hi, p);
        assert!(m >= lo - 1e-9 && m <= hi + 1e-9);
        if m < (lo + hi) * 0.5 {
            below_median += 1;
        }
    }
    // p = 1.5 puts most samples below the midpoint.
    assert!(
        below_median > 1200,
        "distribution should skew low: {}",
        below_median
    );
}

#[test]
fn y_normal_mean_var() {
    let mut rng = XorShift::new(7);
    let n = 20_000;
    let mut s = 0.0;
    let mut ss = 0.0;
    for _ in 0..n {
        let x = sample_normal(&mut rng, 2.0, 0.5);
        s += x;
        ss += x * x;
    }
    let m = s / n as f64;
    let v = ss / n as f64 - m * m;
    assert!((m - 2.0).abs() < 0.05, "mean {}", m);
    assert!((v - 0.25).abs() < 0.02, "var {}", v);
}

#[test]
fn y_imf_median_around_02_msun() {
    // Kroupa (2001) → median ~0.2 M☉ for the 0.08..50 range.
    let mut rng = XorShift::new(42);
    let mut ms: Vec<f64> = (0..20_000).map(|_| sample_imf_mass(&mut rng)).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ms[ms.len() / 2];
    assert!(med > 0.15 && med < 0.35, "IMF median {}", med);
}

// ===========================================================================
// z — small utilities
// ===========================================================================
#[test]
fn z_time_unit_is_about_fifty_days() {
    let days = time_unit_s() / 86_400.0;
    assert!(days > 40.0 && days < 60.0, "{} days", days);
}

#[test]
fn z_index_of_id() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let idx = b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let id = b.id[idx];
    assert_eq!(b.index_of_id(id), Some(1));
    assert_eq!(b.index_of_id(9999), None);
}

#[test]
fn z_set_mass_updates_radius() {
    // `set_mass` recomputes the radius from the body's bulk density, not
    // the mass-radius law (which is only used to seed `bulk_density`).
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let r1 = b.r[0];
    b.set_mass(0, 4.0);
    assert!(b.r[0] > r1);
    assert!((b.r[0] - radius_from_density(4.0, b.bulk_density[0])).abs() < 1e-15);
}

#[test]
fn z_log_ring_buffer_wraps() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    for k in 0..EVENT_LOG_MAX + 50 {
        b.log(EventKind::User, format!("e{}", k), 0.0, 0.0);
    }
    assert_eq!(b.events.len(), EVENT_LOG_MAX);
    assert!(b
        .events
        .back()
        .unwrap()
        .text
        .contains(&format!("{}", EVENT_LOG_MAX + 49)));
}

#[test]
fn z_contact_r_black_hole() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 10.0 * M_SUN_ME, BodyKind::BlackHole);
    let expected = (BH_CAPTURE_HORIZONS * b.r[0]).max(BH_MIN_CAPTURE);
    assert!((b.contact_r(0) - expected).abs() < 1e-15);
    let mut p = Bodies::with_capacity(1);
    p.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    assert_eq!(p.contact_r(0), p.r[0]);
}

#[test]
fn z_recenter_com_zeroes_both() {
    let mut b = Bodies::with_capacity(2);
    b.push(1.0, 2.0, 3.0, 4.0, M_SUN_ME, BodyKind::Star);
    b.push(-1.0, -2.0, -3.0, -4.0, 1.0, BodyKind::Planet);
    recenter_com(&mut b);
    let (px, py) = momentum(&b);
    assert!(px.abs() < 1e-9 && py.abs() < 1e-9);
    let mt: f64 = b.m.iter().sum();
    let cx: f64 = (0..b.len()).map(|i| b.m[i] * b.x[i]).sum::<f64>() / mt;
    let cy: f64 = (0..b.len()).map(|i| b.m[i] * b.y[i]).sum::<f64>() / mt;
    assert!(cx.abs() < 1e-9 && cy.abs() < 1e-9);
}

#[test]
fn z_cell_key_distinct_nearby() {
    let k00 = cell_key(0, 0);
    assert_eq!(k00, cell_key(0, 0));
    assert_ne!(k00, cell_key(1, 0));
    assert_ne!(k00, cell_key(0, 1));
}

#[test]
fn z_au_in_gm() {
    assert!((au_in_gm() - 149.6).abs() < 0.1);
}

#[test]
fn z_solar_extent_monotone() {
    assert!((solar_extent_gm(1) - 5000.0).abs() < 1.0);
    assert!(solar_extent_gm(3) > solar_extent_gm(1));
    assert!(solar_extent_gm(6) > solar_extent_gm(3));
}
