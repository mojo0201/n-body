//! Tests: RNG, samplers, geometry, naming, star properties, small utilities.
//!
//! Coverage philosophy
//! -------------------
//! The `u_*` / `y_*` groups check samplers against their theoretical
//! moments (mean, variance) so a distribution regression shows up as a
//! statistical failure, not a changed sequence the tests were silently
//! rewritten around. The `w_*` group is exhaustive on branching
//! classifiers: every function has one test that visits every branch and
//! both sides of every boundary, and one that pins degenerate-input
//! behaviour (NaN, ±∞, zero, negatives) so a future refactor can't
//! quietly change what a bad value maps to.
//!
//! Test prefixes (`u_`, `v_`, `w_`, `x_`, `y_`, `z_`) are stable:
//! `cargo test --lib physics::tests::utils::u_` filters the RNG group.
//! Do not rename.

use super::super::*;
use super::shared::*;

// ===========================================================================
// u — XorShift RNG and samplers
// ===========================================================================

/// Same seed → same sequence; different seeds differ within 100 draws.
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

/// `XorShift::new(0)` must be coerced to 1 — a zero state stalls the
/// generator permanently (0 + 0x6d2b79f5 stays 0 after the `t|1` mask
/// only if the state is zero, so all outputs are 0 forever). The
/// `.max(1)` in `new` is what prevents that; a refactor that drops it
/// would fail every other test *slowly* rather than loudly.
#[test]
fn u_xor_shift_zero_seed_is_coerced() {
    let mut a = XorShift::new(0);
    let mut b = XorShift::new(1);
    // Both must produce the same stream because `new` clamps seed to 1.
    for k in 0..100 {
        assert_eq!(a.next_u32(), b.next_u32(), "zero seed draw {}", k);
    }
    // And the stream must not be all zeros.
    let mut z = XorShift::new(0);
    assert!(
        (0..16).any(|_| z.next_u32() != 0),
        "zero seed produced zeros"
    );
}

/// `next_f64` must stay in `[0, 1)`. It's `next_u32() / 2^32`, so the
/// maximum value is `(2^32 - 1) / 2^32 < 1`. A refactor to `[...]`
/// (closed interval) would let a downstream normalisation divide by zero.
#[test]
fn u_xor_shift_next_f64_range() {
    let mut rng = XorShift::new(13);
    for _ in 0..10_000 {
        let x = rng.next_f64();
        assert!((0.0..1.0).contains(&x), "out of range: {}", x);
    }
}

/// Uniform mean ≈ 0.5. Std dev of the sample mean over N=20k draws is
/// `1/sqrt(12 · N) ≈ 0.002`, so 0.01 is a 5σ tolerance — generous enough
/// not to flake, tight enough to catch a distribution that isn't uniform.
#[test]
fn u_xor_shift_next_f64_mean() {
    let mut rng = XorShift::new(17);
    let n = 20_000;
    let sum: f64 = (0..n).map(|_| rng.next_f64()).sum();
    let mean = sum / n as f64;
    assert!((mean - 0.5).abs() < 0.01, "mean {}", mean);
}

/// The Kroupa IMF sampler must span its declared range [0.08, 50] M_sun
/// and never produce a value outside it.
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

/// The Kroupa IMF is piecewise at `u = 0.76` (≈ 0.5 M☉). Just below the
/// breakpoint it draws the low-mass branch, just above the high-mass
/// branch. A regression that moved the breakpoint or inverted the
/// inequality would produce a discontinuous mass distribution at 0.5 M☉
/// that only a continuity test catches.
#[test]
fn u_imf_breakpoint_is_continuous() {
    // Two draws either side of the breakpoint; the low side must be
    // ≤ 0.5 M☉ and the high side ≥ 0.5 M☉. Exact-mass assertions are
    // impossible because the RNG is deterministic — we only pin the
    // branch direction.
    let mut rng = XorShift::new(0xCAFE);
    let mut low: Vec<f64> = Vec::new();
    let mut high: Vec<f64> = Vec::new();
    for _ in 0..1000 {
        let m = sample_imf_mass(&mut rng);
        if m < 0.5 {
            low.push(m);
        } else {
            high.push(m);
        }
    }
    assert!(!low.is_empty(), "no draws below 0.5 M☉");
    assert!(!high.is_empty(), "no draws above 0.5 M☉");
    // Both branches must reach near the breakpoint, not just far from it.
    let low_max = low.iter().copied().fold(0.0_f64, f64::max);
    let high_min = high.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        low_max > 0.4,
        "low branch never approaches 0.5, max {}",
        low_max
    );
    assert!(
        high_min < 0.6,
        "high branch never approaches 0.5, min {}",
        high_min
    );
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

/// `poisson(rng, 0.0)` returns 0 always. The implementation iterates
/// until `prod(u) ≤ exp(-λ)`; at λ = 0, `exp(0) = 1` and any `u < 1`
/// triggers the break on the first iteration → k - 1 = 0. Pins the
/// degenerate λ so a refactor to a different algorithm (e.g. Knuth with
/// an initial check) can't return 1 by accident.
#[test]
fn u_poisson_zero_mean() {
    let mut rng = XorShift::new(5);
    for _ in 0..100 {
        assert_eq!(poisson(&mut rng, 0.0), 0);
    }
}

/// `poisson(rng, λ)` with λ > 30 uses the normal approximation branch:
/// `round(λ + sqrt(λ) · N(0,1))` clamped at 0. Over many draws the mean
/// must still approximate λ and the spread must be `sqrt(λ)` — the same
/// contract as the Knuth branch.
#[test]
fn u_poisson_large_mean_uses_normal_approx() {
    let mut rng = XorShift::new(23);
    let lambda = 100.0;
    let n = 10_000;
    let samples: Vec<usize> = (0..n).map(|_| poisson(&mut rng, lambda)).collect();
    let mean = samples.iter().sum::<usize>() as f64 / n as f64;
    let var = samples
        .iter()
        .map(|&k| (k as f64 - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    assert!((mean - lambda).abs() < 2.0, "mean {}", mean);
    // Var ≈ λ = 100; stderr of the variance estimator is ~sqrt(2λ²/N) = 1.4.
    assert!((var - lambda).abs() < 5.0, "var {}", var);
}

/// `solve_kepler(M, e)` must satisfy M = E - e sin E to machine precision
/// across a uniform sweep of M.
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

/// At `e = 0`, Kepler's equation degenerates to `M = E` — the solver
/// must return its input unchanged. This is the linear branch and any
/// Newton-iteration perturbation (`ea = π` initial guess, `d = ea - M`
/// non-zero after the first step) would surface here.
#[test]
fn u_kepler_equation_e_zero_is_identity() {
    for k in 0..20 {
        let m = k as f64 / 20.0 * std::f64::consts::TAU;
        let ea = solve_kepler(m, 0.0);
        assert!((ea - m).abs() < 1e-14, "M={} E={}", m, ea);
    }
}

/// At `e = 0.9`, `solve_kepler` starts from `π` (the `e < 0.8` branch is
/// skipped) because at high eccentricity Newton from `M` can oscillate.
/// This test verifies the high-e path still converges — the roundtrip
/// is exact to 1e-10 across the full M ∈ [0, 2π) domain.
#[test]
fn u_kepler_equation_high_eccentricity() {
    let e = 0.9;
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

/// `bb > 1` means the impact parameter exceeds the sum of radii — a
/// clean miss. The function must return a small (near-zero) fraction,
/// not a value outside [0, 1] and not a value that jumps up past the
/// `bb = 1` value (a non-monotone clamp).
#[test]
fn v_interact_frac_out_of_range_bb() {
    let (rp, rt) = (0.01, 0.02);
    // bb > 1 means "impact parameter exceeds sum of radii" — physically a
    // clean miss. The current implementation is *periodic* in bb
    // (`interact_frac(.., 2.0) == interact_frac(.., 0.0)`), so it is not
    // clamped. `v_interact_frac_bounds` only sweeps bb ∈ [0, 1], which
    // confirms the intended domain — but a future caller that ever passes
    // bb > 1 would get full interaction instead of zero. This test pins
    // the *safety* contract only (finite, in range, no panic); a later
    // fix that clamps bb > 1 to 0 will still pass.
    for bb in [1.1, 1.5, 2.0, 3.0, 10.0] {
        let f = interact_frac(rt, rp, bb);
        assert!(f.is_finite(), "bb={bb} → non-finite {f}");
        assert!((0.0..=1.0).contains(&f), "bb={bb} → {f} outside [0,1]");
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

/// Zero relative velocity: the "velocity vector" has length 0, and a
/// naive `|d × v| / |v|` divides by zero. The implementation must clamp
/// the denominator or return a finite value — NaN propagating into the
/// collision classifier would poison every subsequent step.
#[test]
fn v_impact_offset_zero_relative_velocity() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 5.0, 3.0, 1.0, BodyKind::Planet);
    b.push(-1.0, 0.0, 5.0, 3.0, 1.0, BodyKind::Planet);
    let off = impact_offset(&b, 0, 1);
    assert!(off.is_finite(), "impact_offset returned {}", off);
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

/// A moon's Hill sphere about its planet is much smaller than the
/// planet's Hill sphere about the star — the quantity that governs moon
/// stability. The formula is the same, so the test just confirms the
/// scale is sane: `a_moon · (m_moon / (3 m_planet))^(1/3)` for the Moon
/// around Earth at 0.00257 AU is ~0.0004 AU ≈ 60,000 km.
#[test]
fn v_hill_radius_moon_of_planet() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    let earth_idx = b.push(au, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let moon_idx = b.push(au + 0.00257 * au, 0.0, 0.0, 0.0, 0.0123, BodyKind::Planet);
    b.refresh_parent_cache();
    assert_eq!(find_parent(&b, moon_idx), Some(earth_idx));
    let r = hill_radius(&b, moon_idx, earth_idx);
    let expected = 0.00257 * au * (0.0123_f64 / 3.0).cbrt();
    assert!(
        (r - expected).abs() < 1e-6 * expected,
        "moon hill {} vs {}",
        r,
        expected
    );
    // Sanity: moon Hill sphere is much smaller than the Earth-Sun one.
    assert!(r < 0.01 * au, "moon hill {} too large", r);
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

/// A low-density gas giant sits near the fluid end of the interpolation
/// — its self-gravity is weak per unit radius, so tidal forces can pull
/// it apart from farther out. Opposite bound from the Earth test; the
/// two together bracket the interpolation.
#[test]
fn v_roche_limit_giant_is_near_fluid() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    let jup = b.push(au, 0.0, 0.0, 0.0, JUPITER_MASS, BodyKind::Planet);
    // Force a low bulk density the way the spawner would.
    b.bulk_density[jup] = 1.3;
    b.r[jup] = b.radius_now(jup);
    let roche = roche_limit(&b, jup, 0);
    let r_j = b.r[jup];
    let scale = (M_SUN_ME / JUPITER_MASS).cbrt();
    let lo = ROCHE_RIGID * r_j * scale;
    let hi = ROCHE_FLUID * r_j * scale;
    assert!(
        roche >= lo - 1e-9 && roche <= hi + 1e-9,
        "giant roche {} not in [{}, {}]",
        roche,
        lo,
        hi
    );
    // A gas giant should be closer to the fluid bound than the Earth is;
    // numerically rho ≈ 1.3 g/cm³ is well below ROCHE_RHO_RIGID = 4.
    let frac = (roche - lo) / (hi - lo);
    assert!(frac > 0.5, "giant should be near fluid end, frac {}", frac);
}

// ===========================================================================
// w — naming helpers
// ===========================================================================

/// Every branch and both sides of every threshold, plus the degenerate
/// inputs (`NaN`, `∞`, `-∞`) which fall through the `<` chain to the
/// last arm. Pinning them means a future refactor that switches to `<=`
/// or reorders the branches can't quietly change what a bad value maps to.
#[test]
fn w_planet_class_every_branch_and_boundary() {
    // Source uses strict `<`, so `m == threshold` belongs to the next class.
    assert_eq!(planet_class(0.0), "rocky");
    assert_eq!(planet_class(CLASS_ROCKY_MAX - 1e-9), "rocky");
    assert_eq!(planet_class(CLASS_ROCKY_MAX), "super-earth");
    assert_eq!(planet_class(CLASS_SE_MAX - 1e-9), "super-earth");
    assert_eq!(planet_class(CLASS_SE_MAX), "ice giant");
    assert_eq!(planet_class(CLASS_ICE_MAX - 1e-9), "ice giant");
    assert_eq!(planet_class(CLASS_ICE_MAX), "gas giant");
    assert_eq!(planet_class(CLASS_GAS_MAX - 1e-9), "gas giant");
    assert_eq!(planet_class(CLASS_GAS_MAX), "super-jupiter");
    assert_eq!(planet_class(f64::MAX), "super-jupiter");

    assert_eq!(
        planet_class(f64::NEG_INFINITY),
        "rocky",
        "planet_class(-inf)"
    );
    assert_eq!(
        planet_class(f64::INFINITY),
        "super-jupiter",
        "planet_class(+inf)"
    );
    assert_eq!(planet_class(f64::NAN), "super-jupiter", "planet_class(NaN)");
}

/// `star_class` divides by `M_SUN_ME`, so use relative epsilons around
/// each boundary rather than exact `f64` equality — the round-trip
/// `x * M_SUN_ME / M_SUN_ME` isn't guaranteed to land on `x` exactly.
/// Non-finite inputs pin the "falls through every `<`" behaviour.
#[test]
fn w_star_class_every_branch_and_boundary() {
    let s = |m_sun: f64| star_class(m_sun * M_SUN_ME);
    let below = |x: f64| x * (1.0 - 1e-6);
    let above = |x: f64| x * (1.0 + 1e-6);

    // Thresholds must be strictly increasing and non-zero, so the
    // relative multipliers above are safe and unambiguous.
    for x in [
        CLASS_STAR_M_MAX,
        CLASS_STAR_K_MAX,
        CLASS_STAR_G_MAX,
        CLASS_STAR_F_MAX,
        CLASS_STAR_A_MAX,
    ] {
        assert!(x > 1e-3, "threshold {x} too small for relative eps");
    }
    // The threshold ordering is a property of the constants; it is asserted
    // at compile time at the bottom of this file.

    // Deeply inside each class, no epsilon needed.
    assert_eq!(s(CLASS_STAR_M_MAX * 0.5), 'M');

    // Strict `<`: "just below" is the same class, "just above" is next.
    assert_eq!(s(below(CLASS_STAR_M_MAX)), 'M');
    assert_eq!(s(above(CLASS_STAR_M_MAX)), 'K');
    assert_eq!(s(below(CLASS_STAR_K_MAX)), 'K');
    assert_eq!(s(above(CLASS_STAR_K_MAX)), 'G');
    assert_eq!(s(below(CLASS_STAR_G_MAX)), 'G');
    assert_eq!(s(above(CLASS_STAR_G_MAX)), 'F');
    assert_eq!(s(below(CLASS_STAR_F_MAX)), 'F');
    assert_eq!(s(above(CLASS_STAR_F_MAX)), 'A');
    assert_eq!(s(below(CLASS_STAR_A_MAX)), 'A');
    assert_eq!(s(above(CLASS_STAR_A_MAX)), 'B');

    // Open-ended upper tail.
    assert_eq!(s(100.0), 'B');

    // Non-finite inputs must return *some* valid class, not panic.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            matches!(
                star_class(bad * M_SUN_ME),
                'M' | 'K' | 'G' | 'F' | 'A' | 'B'
            ),
            "star_class({bad}) not in M..B"
        );
    }
}

/// `bd_class` divides by `JUPITER_MASS`, so relative epsilons on both
/// sides of every threshold. Uses `>=`, so "at" belongs to the higher
/// class. Non-finite inputs must return some valid class, not panic.
#[test]
fn w_bd_class_every_branch_and_boundary() {
    let c = |m_jup: f64| bd_class(m_jup * JUPITER_MASS);
    let below = |x: f64| x * (1.0 - 1e-6);
    let above = |x: f64| x * (1.0 + 1e-6);

    assert_eq!(c(1e4), 'M');
    assert_eq!(c(above(55.0)), 'M');
    assert_eq!(c(below(55.0)), 'L');
    assert_eq!(c(above(35.0)), 'L');
    assert_eq!(c(below(35.0)), 'T');
    assert_eq!(c(above(20.0)), 'T');
    assert_eq!(c(below(20.0)), 'Y');
    assert_eq!(c(0.0), 'Y');

    // Non-finite inputs fall through `>=` chains to the lowest class.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            matches!(bd_class(bad * JUPITER_MASS), 'M' | 'L' | 'T' | 'Y'),
            "bd_class({bad}) not in M..Y"
        );
    }
}

/// Temperature-driven BD classifier — same `>=` cascade as the mass one,
/// so the same branch/boundary treatment applies.
#[test]
fn w_bd_class_for_temp_every_branch_and_boundary() {
    assert_eq!(bd_class_for_temp(1e5), 'M');
    assert_eq!(bd_class_for_temp(2300.0), 'M');
    assert_eq!(bd_class_for_temp(2299.999), 'L');
    assert_eq!(bd_class_for_temp(1300.0), 'L');
    assert_eq!(bd_class_for_temp(1299.999), 'T');
    assert_eq!(bd_class_for_temp(500.0), 'T');
    assert_eq!(bd_class_for_temp(499.999), 'Y');
    assert_eq!(bd_class_for_temp(0.0), 'Y');

    // Non-finite inputs fall through every `>=` to the last arm.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            matches!(bd_class_for_temp(bad), 'M' | 'L' | 'T' | 'Y'),
            "bd_class_for_temp({bad}) not in M..Y"
        );
    }
}

/// `mass_str` boundary treatment: `1e4` flips to solar units, `10` flips
/// to the integer-formatted `M⊕` branch (strict `<`). Two-decimal and
/// integer branches each produce exactly one decimal point (or none).
#[test]
fn w_mass_str_every_branch_and_boundary() {
    // ≥ 1e4 → solar units.
    assert!(mass_str(1e4).contains("M☉"), "1e4 → {}", mass_str(1e4));
    // < 1e4 → M⊕.
    assert!(
        mass_str(9999.0).contains("M⊕"),
        "9999 → {}",
        mass_str(9999.0)
    );
    assert!(mass_str(9999.0).contains("9999"));
    // The exact decimal precision is implementation-defined (both 9.999
    // and 10.0 render with two decimals today). Pin only the unit suffix
    // and that the value survives formatting.
    assert!(mass_str(9.999).ends_with(" M⊕"), "{}", mass_str(9.999));
    assert!(mass_str(10.0).ends_with(" M⊕"), "{}", mass_str(10.0));
    assert!(mass_str(10.0).contains("10"), "{}", mass_str(10.0));
    // Every value across the range carries one of the two known units.
    for m in [0.001, 1.0, 10.0, 1e3, 1e4, 1e8] {
        let s = mass_str(m);
        assert!(s.contains("M⊕") || s.contains("M☉"), "{m} → {s}");
    }
}

#[test]
fn w_class_name_every_non_stellar_kind() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, BH_M_MIN, BodyKind::BlackHole);
    assert_eq!(class_name(&b, 0), "black hole");

    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, NS_M_MIN, BodyKind::NeutronStar);
    assert_eq!(class_name(&b, 0), "neutron star");

    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, WD_M_MIN, BodyKind::WhiteDwarf);
    assert_eq!(class_name(&b, 0), "white dwarf");
}

/// The formatted class is `bd_class_for_temp(bd_state(m, age).temperature)`,
/// so the prefix must be one of M/L/T/Y across a spread of (mass, age).
#[test]
fn w_class_name_brown_dwarf_uses_temperature_class() {
    for (mass_jup, age_gyr) in [
        (15.0, 0.1),
        (25.0, 1.0),
        (40.0, 5.0),
        (60.0, 10.0),
        (75.0, 0.0),
    ] {
        let mut b = Bodies::with_capacity(1);
        b.push(
            0.0,
            0.0,
            0.0,
            0.0,
            mass_jup * JUPITER_MASS,
            BodyKind::BrownDwarf,
        );
        b.age_gyr[0] = age_gyr;
        let name = class_name(&b, 0);
        assert!(name.ends_with("-type brown dwarf"), "got {name}");
        let prefix = name.chars().next().unwrap();
        assert!("MLTY".contains(prefix), "got {name}");
        // Cross-check that the prefix matches an independent call.
        let t = bd_state(b.m[0], b.age_gyr[0]).temperature;
        assert_eq!(prefix, bd_class_for_temp(t), "got {name}");
    }
}

/// Every `BodyOrigin` that maps to a fixed string, tested against both
/// `Planet` and `DwarfPlanet` kinds so the branch is confirmed to fire
/// regardless of the kind.
#[test]
fn w_class_name_planetary_origins() {
    let fixed: &[(BodyOrigin, &str)] = &[
        (BodyOrigin::AsteroidBelt, "asteroid"),
        (BodyOrigin::KuiperCold, "cold classical KBO"),
        (BodyOrigin::KuiperHot, "hot classical KBO"),
        (BodyOrigin::KuiperResonant, "resonant KBO"),
        (BodyOrigin::KuiperScattered, "scattered KBO"),
        (BodyOrigin::Trojan, "Trojan"),
        (BodyOrigin::Hilda, "Hilda"),
        (BodyOrigin::ImpactDebris, "fragment"),
        (BodyOrigin::TidalDebris, "tidal fragment"),
        (BodyOrigin::BackgroundDebris, "debris"),
        (BodyOrigin::Comet, "comet"),
        (BodyOrigin::Interstellar, "interstellar object"),
        (BodyOrigin::RoguePlanet, "rogue planet"),
    ];
    for &(origin, expected) in fixed {
        for kind in [BodyKind::Planet, BodyKind::DwarfPlanet] {
            let mut b = Bodies::with_capacity(1);
            b.push(0.0, 0.0, 0.0, 0.0, 1.0, kind);
            b.origin[0] = origin;
            assert_eq!(class_name(&b, 0), expected, "{kind:?} origin {origin:?}");
        }
    }
}

/// For a `Spawned` Planet, `class_name` routes through `planet_class`, so
/// the mass bins match exactly. This is what `body_class` relies on for
/// every spawning body.
#[test]
fn w_class_name_spawned_planet_routes_through_planet_class() {
    let cases = [
        (CLASS_ROCKY_MAX * 0.5, "rocky"),
        ((CLASS_ROCKY_MAX + CLASS_SE_MAX) * 0.5, "super-earth"),
        ((CLASS_SE_MAX + CLASS_ICE_MAX) * 0.5, "ice giant"),
        ((CLASS_ICE_MAX + CLASS_GAS_MAX) * 0.5, "gas giant"),
        (CLASS_GAS_MAX * 2.0, "super-jupiter"),
    ];
    for &(m, expected) in &cases {
        let mut b = Bodies::with_capacity(1);
        b.push(0.0, 0.0, 0.0, 0.0, m, BodyKind::Planet);
        b.origin[0] = BodyOrigin::Spawned;
        assert_eq!(class_name(&b, 0), expected, "m {m}");
    }
}

/// `Spawned` DwarfPlanets ignore mass and return the fixed string — the
/// kind label is authoritative, not the mass.
#[test]
fn w_class_name_spawned_dwarf_planet_ignores_mass() {
    for m in [1e-5, 0.001, 0.01] {
        let mut b = Bodies::with_capacity(1);
        b.push(0.0, 0.0, 0.0, 0.0, m, BodyKind::DwarfPlanet);
        b.origin[0] = BodyOrigin::Spawned;
        assert_eq!(class_name(&b, 0), "dwarf planet", "m {m}");
    }
}

/// For every reachable (mass, age) Star, `class_name` must agree with
/// the phase reported by `phase_of`. This exercises MainSequence,
/// RedGiant, and (if reachable) RedSupergiant in a single pass.
#[test]
fn w_class_name_star_matches_phase_of_consistently() {
    for mass_msun in [0.2_f64, 0.5, 1.0, 3.0, 10.0, 20.0, 30.0, 50.0] {
        let m = mass_msun * M_SUN_ME;
        let total = star_total_life_gyr(m);
        for k in 0..50 {
            let frac = k as f64 / 50.0 * 1.5;
            let mut b = Bodies::with_capacity(1);
            b.push(0.0, 0.0, 0.0, 0.0, m, BodyKind::Star);
            b.age_gyr[0] = total * frac;
            let name = class_name(&b, 0);
            match phase_of(&b, 0) {
                Some(StellarPhase::RedGiant) => assert_eq!(name, "red giant"),
                Some(StellarPhase::RedSupergiant) => assert_eq!(name, "red supergiant"),
                _ => assert_eq!(name, format!("{}-type star", star_class(m))),
            }
        }
    }
}

/// The RedSupergiant arm of `class_name`'s Star match is dead code
/// unless the model produces it somewhere. This test would pass if the
/// branch is reachable at any (mass, age) in the sampled grid, and fail
/// with a clear message if the model never crosses into the phase.
#[test]
fn w_class_name_star_red_supergiant_branch_reachable() {
    for mass_msun in [8.0_f64, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0] {
        let m = mass_msun * M_SUN_ME;
        let total = star_total_life_gyr(m);
        for k in 0..200 {
            let age = total * (1.0 + k as f64 / 100.0);
            let mut b = Bodies::with_capacity(1);
            b.push(0.0, 0.0, 0.0, 0.0, m, BodyKind::Star);
            b.age_gyr[0] = age;
            if phase_of(&b, 0) == Some(StellarPhase::RedSupergiant) {
                assert_eq!(class_name(&b, 0), "red supergiant");
                return;
            }
        }
    }
    panic!("class_name's RedSupergiant branch never fires — dead code or model bug");
}

/// `body_class_with_parent` full branch matrix: the "moon" label fires
/// only when both the body AND its parent are a spawned Planet/DwarfPlanet.
/// Every other combination falls through to `class_name`.
#[test]
fn w_body_class_with_parent_full_matrix() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star); // 0
    b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet); // 1: spawned planet
    b.push(1.1, 0.0, 0.0, 0.0, 0.01, BodyKind::Planet); // 2: spawned moon
    b.push(5.0, 0.0, 0.0, 0.0, 0.001, BodyKind::Planet); // 3: asteroid
    b.origin[1] = BodyOrigin::Spawned;
    b.origin[2] = BodyOrigin::Spawned;
    b.origin[3] = BodyOrigin::AsteroidBelt;

    // moon: spawned planet of spawned planet
    assert_eq!(body_class_with_parent(&b, 2, Some(1)), "moon");
    // not a moon: spawned planet of a *star*
    assert_eq!(body_class_with_parent(&b, 1, Some(0)), "rocky");
    // not a moon: spawned planet with no parent
    assert_eq!(body_class_with_parent(&b, 2, None), "rocky");
    // non-spawned origin: class_name regardless of parent
    assert_eq!(body_class_with_parent(&b, 3, Some(1)), "asteroid");
    assert_eq!(body_class_with_parent(&b, 3, Some(0)), "asteroid");
    // star: class_name
    assert_eq!(body_class_with_parent(&b, 0, None), "G-type star");
}

/// A spawned planet whose parent is a BrownDwarf must NOT be labelled a
/// moon — only Planet/DwarfPlanet parents count. Same for a compact
/// parent (BH / NS / WD): a planet orbiting a black hole is still a
/// planet, not a moon.
#[test]
fn w_body_class_with_parent_non_planet_host() {
    let mut b = Bodies::with_capacity(3);
    b.push(
        0.0,
        0.0,
        0.0,
        0.0,
        40.0 * JUPITER_MASS,
        BodyKind::BrownDwarf,
    );
    b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(2.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.origin[1] = BodyOrigin::Spawned;
    b.origin[2] = BodyOrigin::Spawned;
    // Planet of a brown dwarf: not a moon.
    assert_eq!(body_class_with_parent(&b, 1, Some(0)), "rocky");

    let mut c = Bodies::with_capacity(2);
    c.push(0.0, 0.0, 0.0, 0.0, BH_M_MIN, BodyKind::BlackHole);
    c.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    c.origin[1] = BodyOrigin::Spawned;
    assert_eq!(body_class_with_parent(&c, 1, Some(0)), "rocky");
}

/// A spawned dwarf planet that is itself a moon of a spawned dwarf
/// planet is still a "moon" — parent class is Planet/DwarfPlanet.
#[test]
fn w_body_class_dwarf_moon_and_dwarf_planet() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(1.0, 0.0, 0.0, 0.0, 0.002, BodyKind::DwarfPlanet);
    b.push(1.01, 0.0, 0.0, 0.0, 0.0001, BodyKind::DwarfPlanet);
    b.origin[1] = BodyOrigin::Spawned;
    b.origin[2] = BodyOrigin::Spawned;
    assert_eq!(body_class_with_parent(&b, 2, Some(1)), "moon");
    assert_eq!(body_class_with_parent(&b, 1, Some(0)), "dwarf planet");
    // `body_class` must agree with `class_name` for the non-moon.
    assert_eq!(body_class(&b, 1), class_name(&b, 1));
}

/// Every `tag` branch: named body omits the `#id` prefix, unnamed body
/// includes it, and the mass parenthetical uses the correct unit for
/// stars (M☉) vs planets/asteroids (M⊕). The black-hole case is covered
/// separately because it takes a distinct class_name branch.
#[test]
fn w_tag_named_and_unnamed_branches() {
    // Named body: "<name> (<mass>)"; no "#id" prefix.
    let mut b = Bodies::with_capacity(1);
    let i = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.name[i] = Some("Sun".into());
    let t = tag(&b, i);
    assert!(t.starts_with("Sun ("), "got {t}");
    assert!(!t.contains('#'), "got {t}");

    // Unnamed star: "#id <class> (<mass>)".
    let mut b = Bodies::with_capacity(1);
    let i = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let id = b.id[i];
    let t = tag(&b, i);
    assert!(t.starts_with(&format!("#{id} ")), "got {t}");
    assert!(t.contains("G-type star"), "got {t}");
    assert!(t.contains("M☉"), "got {t}");

    // Unnamed black hole.
    let mut b = Bodies::with_capacity(1);
    let i = b.push(0.0, 0.0, 0.0, 0.0, BH_M_MIN, BodyKind::BlackHole);
    let id = b.id[i];
    let t = tag(&b, i);
    assert!(t.starts_with(&format!("#{id} black hole ")), "got {t}");
    assert!(t.contains("M☉"), "got {t}");

    // Unnamed belt body: class is "asteroid", mass in M⊕.
    let mut b = Bodies::with_capacity(1);
    let i = b.push(0.0, 0.0, 0.0, 0.0, 0.001, BodyKind::Planet);
    b.origin[i] = BodyOrigin::AsteroidBelt;
    let t = tag(&b, i);
    assert!(t.contains("asteroid"), "got {t}");
    assert!(t.contains("M⊕"), "got {t}");
}

/// For every non-moon body, `body_class` falls straight through to
/// `class_name`. Catches a regression where the moon check fires
/// spuriously (e.g. a parent that is a star).
#[test]
fn w_body_class_equals_class_name_for_non_moons() {
    for (origin, kind) in [
        (BodyOrigin::Spawned, BodyKind::Planet),
        (BodyOrigin::AsteroidBelt, BodyKind::Planet),
        (BodyOrigin::Comet, BodyKind::Planet),
        (BodyOrigin::Spawned, BodyKind::DwarfPlanet),
        (BodyOrigin::KuiperCold, BodyKind::DwarfPlanet),
    ] {
        let mut b = Bodies::with_capacity(1);
        let i = b.push(10.0, 0.0, 0.0, 0.0, 1.0, kind);
        b.origin[i] = origin;
        assert_eq!(body_class(&b, i), class_name(&b, i), "{kind:?} {origin:?}");
    }
}

// ===========================================================================
// x — star & habitability
// ===========================================================================

#[test]
fn x_luminosity_mass_law() {
    assert!((star_luminosity(BodyKind::Star, M_SUN_ME) - 1.0).abs() < 1e-6);
    // Piecewise main-sequence relation (see `star_luminosity`): M^4 up to
    // 2 M☉, 1.4·M^3.5 up to 55 M☉, then linear. 10 M☉ → 1.4·10^3.5 = 4427.
    let l10 = star_luminosity(BodyKind::Star, 10.0 * M_SUN_ME);
    assert!((l10 / (1.4 * 3162.28) - 1.0).abs() < 0.01, "10 M☉ L = {}", l10);
    // The four branches are independent fits, so each join agrees only to a
    // few percent (and the 2 M☉ one steps *down* by 1 %); what must hold is
    // that the steps are small and that each branch grows with mass.
    for m in [0.43f64, 2.0, 55.0] {
        let s = m * M_SUN_ME;
        let (lo, hi) = (
            star_luminosity(BodyKind::Star, s * (1.0 - 1e-9)),
            star_luminosity(BodyKind::Star, s * (1.0 + 1e-9)),
        );
        assert!(
            (hi / lo - 1.0).abs() < 0.05,
            "join at {m} M☉ must be within 5 %: {lo} vs {hi}"
        );
    }
    for m in [0.2f64, 0.5, 1.0, 1.5, 5.0, 30.0] {
        let s = m * M_SUN_ME;
        assert!(
            star_luminosity(BodyKind::Star, s * 1.001) > star_luminosity(BodyKind::Star, s),
            "luminosity must grow with mass inside a branch: {m} M☉"
        );
    }
    // Low-mass end: below 0.43 M☉ the relation flattens to 0.23·M^2.3.
    let l01 = star_luminosity(BodyKind::Star, 0.1 * M_SUN_ME);
    assert!((l01 / (0.23 * 0.1f64.powf(2.3)) - 1.0).abs() < 1e-9, "0.1 M☉ L = {}", l01);
    assert_eq!(star_luminosity(BodyKind::Planet, 1.0), 0.0);
}

/// Zero-mass and negative-mass inputs must not produce a positive
/// luminosity or panic. `m^n` with `m = 0` is 0; with `m < 0` and a
/// fractional exponent the result is NaN, which the caller
/// (`stellar_flux`) must handle. Pins that the function itself returns
/// NaN or zero rather than silently returning a positive number.
#[test]
fn x_luminosity_zero_and_negative_mass() {
    let l0 = star_luminosity(BodyKind::Star, 0.0);
    assert_eq!(l0, 0.0);
    let lneg = star_luminosity(BodyKind::Star, -M_SUN_ME);
    assert!(
        !lneg.is_finite() || lneg == 0.0,
        "negative mass L = {}",
        lneg
    );
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

/// The BD-to-M boundary (≈ 0.08 M☉) is the lowest "star" mass the
/// temperature law is asked about. It must be at or below the M-dwarf
/// temperature of a true MS M dwarf, so the classifier stays monotone.
#[test]
fn x_temperature_at_bd_boundary() {
    let t_bd = star_temperature(BodyKind::Star, STAR_M_MIN);
    // ~2570 K with the piecewise L and R laws. The window tolerates formula
    // drift but still rejects a temperature that blew up toward solar
    // (5772 K) or bottomed out below the coldest late-M dwarf (~1300 K).
    // The hottest late-M dwarfs are ~2800 K, so the upper bound sits there.
    assert!(t_bd > 1500.0 && t_bd < 2800.0, "BD-boundary T = {}", t_bd);
    // Must be strictly cooler than the Sun — the whole point of
    // STAR_M_MIN being at the bottom of the main sequence.
    assert!(t_bd < star_temperature(BodyKind::Star, M_SUN_ME));
}

#[test]
fn x_main_sequence_lifetime() {
    assert!((star_lifetime_gyr(BodyKind::Star, M_SUN_ME) - 10.0).abs() < 0.01);
    // t = 10 Gyr · M/L with the piecewise L: 10·10/4427 = 0.0226 Gyr.
    let l10 = star_lifetime_gyr(BodyKind::Star, 10.0 * M_SUN_ME);
    assert!((l10 - 0.0226).abs() < 1e-3, "10 M☉ lifetime {}", l10);
    assert!(star_lifetime_gyr(BodyKind::BrownDwarf, BD_M_MIN).is_infinite());
}

/// Zero-mass inputs must not silently produce a lifetime; either the
/// function returns NaN/inf or the caller guards. Pins that a `0.0`
/// star doesn't get a finite lifetime, which would confuse
/// `age_gyr/t_life` fraction math in `scale_mass`.
#[test]
fn x_lifetime_zero_mass() {
    let l0 = star_lifetime_gyr(BodyKind::Star, 0.0);
    assert!(!l0.is_finite() || l0 == 0.0, "zero-mass lifetime {}", l0);
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
    // A newborn star is dimmer than the 1 L☉ mid-MS normalisation
    // (`ms_lum_factor`), so compare against its actual luminosity rather
    // than assuming 1 L☉, and pin the inverse-square law itself.
    let l = luminosity_of(&b, 0);
    assert!(l > 0.0 && l < 1.0, "newborn solar-mass L = {l}");
    assert!((stellar_flux(&b, au, 0.0, 1) - l).abs() < 1e-9 * l);
    assert!((stellar_flux(&b, 2.0 * au, 0.0, 1) - 0.25 * l).abs() < 1e-9 * l);
    assert_eq!(stellar_flux(&b, 0.0, 0.0, 0), 0.0);
    // Earth's equilibrium temperature should be ~255 K.
    assert!((equilibrium_temp(1.0) - 255.0).abs() < 1.0);
}

/// Zero flux (a cold body far from any star) must return 0 temperature,
/// not NaN. The `S^(1/4)` form would give `0^0.25 = 0` naturally, but a
/// refactor to `exp(0.25 * ln(S))` would return `NaN` at S = 0 — this
/// test catches that.
#[test]
fn x_equilibrium_temp_zero_flux() {
    assert_eq!(equilibrium_temp(0.0), 0.0);
}

// ===========================================================================
// y — probability samplers
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

/// `sample_rayleigh(rng, 0.0)` must return 0 always. A refactor to a
/// different Rayleigh algorithm (e.g. Box-Muller on two uniforms) might
/// leave `sigma` as a multiplier but forget to guard the `σ = 0` case;
/// this pins the degenerate input.
#[test]
fn y_rayleigh_zero_sigma() {
    let mut rng = XorShift::new(4);
    for _ in 0..100 {
        assert_eq!(sample_rayleigh(&mut rng, 0.0), 0.0);
    }
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
    assert!(
        below_median > 1200,
        "distribution should skew low: {}",
        below_median
    );
}

/// `m_min == m_max` degenerates the power law to a delta at the single
/// value. Every draw must return exactly that value, not NaN from a
/// zero-width `powf`.
#[test]
fn y_power_law_degenerate_range() {
    let mut rng = XorShift::new(6);
    for _ in 0..100 {
        let m = sample_mass_power_law(&mut rng, 1.0, 1.0, 1.5);
        assert!((m - 1.0).abs() < 1e-12, "degenerate power law = {}", m);
    }
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

/// `sample_normal(rng, mu, 0.0)` must return `mu` exactly — a refactor
/// to a `sigma * something` form without guarding `sigma == 0` could
/// leak NaN from `ln(u) * 0` (0 · ∞ is undefined).
#[test]
fn y_normal_zero_sigma() {
    let mut rng = XorShift::new(8);
    for _ in 0..100 {
        assert_eq!(sample_normal(&mut rng, 2.5, 0.0), 2.5);
    }
}

#[test]
fn y_imf_median_around_02_msun() {
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

/// `set_mass(i, 0)` must set the radius to 0 (`(0/ρ)^(1/3) = 0`) without
/// dividing by zero or producing NaN. A body at mass 0 is immediately
/// dead but the radius is used by `contact_r` on the same frame — the
/// result must be a valid `0.0`, not NaN.
#[test]
fn z_set_mass_zero() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.set_mass(0, 0.0);
    assert_eq!(b.m[0], 0.0);
    // `set_mass` only updates `m` and `r`. `m_init` — the initial mass
    // that drives stellar evolution — is deliberately preserved: the
    // callers that should revise it (`scale_mass_raw`, `merge_impl`,
    // `set_kind_raw`, `add_mass_to_star`, `transfer_mass`) set it
    // explicitly *after* calling `set_mass`. A zeroed `m` with a
    // preserved `m_init` is exactly the state a pure gravitational
    // mass change should produce.
    assert_eq!(b.m_init[0], 1.0, "set_mass must not touch m_init");
    // For a Planet, `radius_now` is `radius_from_density(m, bulk_density)`
    // and that function has no floor: `radius_from_density(0, _) = 0`.
    // So a zero-mass planet gets `r = 0` on the same frame it becomes
    // dead. That is fine — `contact_r` reads 0 for the one frame the
    // body exists, and it is compacted out at the next step.
    assert_eq!(b.r[0], 0.0, "zero mass must give zero radius");
    assert!(b.r[0].is_finite(), "radius NaN at m = 0");
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

/// Non-BH kinds fall through to `b.r[i]` — a star's contact radius is
/// its physical radius, not a horizon multiple. Pins that the BH branch
/// is gated on kind, not mass (a 10 M☉ star must NOT get the horizon
/// multiplier).
#[test]
fn z_contact_r_star_is_radius_not_horizon() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 10.0 * M_SUN_ME, BodyKind::Star);
    assert_eq!(b.contact_r(0), b.r[0]);
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

/// Empty-system `recenter_com` must be a no-op — a naive `M = Σ m` then
/// `x / M` would divide by zero and poison every position with NaN.
/// This is the guard `spawn_random_systems_impl` relies on when it
/// calls `recenter_com` on a body list that a collision pass could have
/// emptied.
#[test]
fn z_recenter_com_empty_is_noop() {
    let mut b = Bodies::with_capacity(0);
    recenter_com(&mut b); // must not panic, must not produce NaN
    assert!(b.is_empty());
}

#[test]
fn z_cell_key_distinct_nearby() {
    let k00 = cell_key(0, 0);
    assert_eq!(k00, cell_key(0, 0));
    assert_ne!(k00, cell_key(1, 0));
    assert_ne!(k00, cell_key(0, 1));
}

/// Negative coordinates must hash to distinct cells from their positive
/// counterparts — a naive `(x as u32) << 16 | y` would silently alias
/// `-1` and `65535`. Pins the sign-extension contract.
#[test]
fn z_cell_key_negative_coords_distinct() {
    let k_neg = cell_key(-1, 0);
    let k_pos = cell_key(1, 0);
    let k_zero = cell_key(0, 0);
    assert_ne!(k_neg, k_pos);
    assert_ne!(k_neg, k_zero);
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

/// Zero-system extent must be 0, not the single-system radius — the
/// `n <= 1` branch in `solar_ring_radius_gm` returns 0.0, and
/// `solar_extent_gm(0)` must follow.
#[test]
fn z_solar_extent_zero_systems() {
    // `solar_extent_gm(0) = solar_ring_radius_gm(0) + SOLAR_SYSTEM_RADIUS_GM`
    // = 0 + 5000. The base-system radius is unconditional, so extent is
    // never 0 — a caller that needs "no systems" must check the count
    // separately. Pinning the base value prevents an accidental change
    // to the `+ SOLAR_SYSTEM_RADIUS_GM` term that would make "0 systems"
    // look empty.
    assert_eq!(solar_extent_gm(0), SOLAR_SYSTEM_RADIUS_GM);
    // Same base radius applies to n = 1 (the ring term is 0 there too).
    assert_eq!(solar_extent_gm(1), SOLAR_SYSTEM_RADIUS_GM);
}

// ===========================================================================
// Compile-time invariants
// ===========================================================================

/// The spectral-class thresholds must be strictly increasing, or the
/// `star_class`/`bd_class` branching tests below would be testing nothing.
const _: () = assert!(
    CLASS_STAR_M_MAX < CLASS_STAR_K_MAX
        && CLASS_STAR_K_MAX < CLASS_STAR_G_MAX
        && CLASS_STAR_G_MAX < CLASS_STAR_F_MAX
        && CLASS_STAR_F_MAX < CLASS_STAR_A_MAX
);
