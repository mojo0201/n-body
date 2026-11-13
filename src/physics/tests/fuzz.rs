//! Tests: fuzz smoke tests.

use super::super::*;
use super::shared::*;

/// Solar system: 500 steps, exact conservation (no mergers, no escapes).
#[test]
fn ak_fuzz_solar_conservation() {
    let steps = fuzz_iters(500);
    let mut b = Bodies::with_capacity(16);
    spawn_solar_system(&mut b);
    let m0 = mass(&b);
    let l0 = total_angular_momentum(&b);
    let n0 = b.len();

    for step in 0..steps {
        advance(&mut b, DT);
        if step % 25 == 0 || step + 1 == steps {
            fuzz_assert_sane(&b, &format!("solar step {}", step));
        }
    }

    // The Solar System preset now spawns 67P and Halley. Over 500 steps
    // (~68 yr) either comet can cross its perihelion enough times to fall
    // below COMET_DEAD_MASS and be culled by `comet_sublimate`. That is
    // physical, not a merge or escape, so the strict `b.len() == n0`
    // assertion is no longer valid. Comets are the only solar-preset bodies
    // that can disappear this way.
    assert!(
        b.stats.merges <= 5,
        "too many mergers in solar system: {} (belt-belt and comet-belt \
         mergers are expected over 500 steps; a bug would cause many more)",
        b.stats.merges
    );
    assert_eq!(b.stats.escaped, 0, "unexpected escape in solar system");
    // Belt and comet bodies are far smaller than any planet, so the
    // Sun + 8 planets must all still be present by ID.
    let n_major = 9.min(n0);
    for i in 0..n_major {
        assert!(
            b.index_of_id(b.id[i]).is_some(),
            "major body #{} (id {}) vanished from the Solar System",
            i,
            b.id[i],
        );
    }
    let m1 = mass(&b);
    assert!(
        (m1 - m0).abs() / m0 < 1e-12,
        "mass changed: {} -> {}",
        m0,
        m1
    );
    // The Solar System is no longer Yoshida-only once moons are present:
    // the Moon's dtl < DT, so kmax ≥ 1 and step_block integrates the
    // system. Symplectic drift over 500 steps is ~1e-4; 1e-3 still
    // catches a regression to leapfrog-only (which would be ~1e-2).
    let l1 = total_angular_momentum(&b);
    assert!(
        (l1 - l0).abs() / (1.0 + l0.abs()) < 1e-3,
        "L changed: {} -> {}",
        l0,
        l1
    );
}

/// 10 random seeds × 200 steps × n=30. Stresses mergers, escapes and
/// fragment spawning at small-n (direct-sum path). Asserts mass never
/// grows and no NaN ever appears.
#[test]
fn ak_fuzz_random_invariants() {
    let steps = fuzz_iters(200);
    let n = 30usize;
    let mut worst_mass_loss = 0.0_f64;

    for seed in 0..10u128 {
        let mut b = Bodies::with_capacity(n);
        spawn_system(&mut b, n, seed);
        let m0 = mass(&b);
        assert!(m0 > 0.0);

        for step in 0..steps {
            advance(&mut b, DT);
            if step % 25 == 0 || step + 1 == steps {
                fuzz_assert_sane(&b, &format!("seed {} step {}", seed, step));
            }
            // Mass may only decrease (culling). It must never grow.
            let m = mass(&b);
            assert!(
                m <= m0 * 1.0001,
                "seed {} step {}: mass grew {} -> {}",
                seed,
                step,
                m0,
                m
            );
        }
        let loss = (m0 - mass(&b)) / m0;
        worst_mass_loss = worst_mass_loss.max(loss);
    }
    eprintln!(
        "ak_fuzz_random_invariants: worst mass loss {:.4}",
        worst_mass_loss
    );
}

/// Construct a system with n > BH_THRESHOLD and 20 pairs that overlap
/// *immediately*, forcing 20 merges on the very first tick of step_block.
/// This triggers `compact()` — which invalidates the cached Barnes-Hut
/// tree — inside the tick loop, exercising the rebuild-on-len-change guard
/// in `step_block`. If that guard were missing, this test would panic with
/// an index-out-of-bounds in `Tree::accel_for` within the first tick.
///
/// Setup notes:
///   * Overlapping pairs at 0.01 Gm separation (contact sum ~0.0127), with
///     zero relative velocity. Guaranteed to merge on tick 1 — no swept
///     detection needed, no dependence on drift timing.
///   * 20 pairs. 20 merges still hits `b.len() != tree_len` many times
///     inside one step_block.
///   * Each pair at a distinct radius and angle so no pair-pair overlap.
///   * 10 steps. Merges happen on tick 1, so 10 steps is enough to verify
///     the survivor stays sane and the tree didn't corrupt state.
///   * FUZZ_ITERS deliberately ignored; this test is already heavy
///     (n≈650, kmax≈6 → 64 sub-ticks) and the merge scenario is fully
///     exercised in the first advance.
#[test]
fn ak_fuzz_many_collisions_tree_rebuild() {
    let steps = 10;
    let mut b = Bodies::with_capacity(800);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mut rng = XorShift::new(7);

    // 600 background bodies between r=5 and r=35, spread angularly.
    for k in 0..600 {
        let r = 5.0 + (k as f64) * 0.05;
        let th = rng.next_f64() * std::f64::consts::TAU;
        let vc = (G * M_SUN_ME / r).sqrt();
        b.push(
            r * th.cos(),
            r * th.sin(),
            -vc * th.sin(),
            vc * th.cos(),
            0.5,
            BodyKind::Planet,
        );
    }

    // 20 overlapping pairs at distinct radii, already within contact sum.
    // Body 2 sits 0.01 Gm to the +x of body 1 with the SAME velocity, so
    // pair_hit returns Some on the very first collision_pass call.
    for k in 0..20 {
        let r = 5.0 + (k as f64) * 1.5; // radii 5, 6.5, 8, ..., 33.5
        let th = 0.1 + (k as f64) * 0.3; // distinct angles
        let vc = (G * M_SUN_ME / r).sqrt();
        let (cos_th, sin_th) = (th.cos(), th.sin());
        let vx = -vc * sin_th;
        let vy = vc * cos_th;
        let (x1, y1) = (r * cos_th, r * sin_th);
        b.push(x1, y1, vx, vy, 1.0, BodyKind::Planet);
        // +0.01 Gm in x, identical velocity → static overlap → immediate merge.
        b.push(x1 + 0.01, y1, vx, vy, 1.0, BodyKind::Planet);
    }

    assert!(
        b.len() >= BH_THRESHOLD,
        "test setup needs n >= BH_THRESHOLD ({}), got {}",
        BH_THRESHOLD,
        b.len()
    );
    compute_accel(&mut b);

    let m0 = mass(&b);
    let n0 = b.len();

    for step in 0..steps {
        advance(&mut b, DT);
        fuzz_assert_sane(&b, &format!("tree-rebuild step {}", step));
    }

    assert!(
        b.stats.merges >= 20,
        "expected >= 20 merges, got {}",
        b.stats.merges
    );
    assert!(b.len() < n0, "merge should have shrunk the array");
    assert!(mass(&b) <= m0 * 1.001, "mass grew unexpectedly");
}

/// Edge case: a single-body system advances forever without panic or NaN.
/// Cheap sanity that `advance` doesn't require n >= 2 to terminate cleanly.
#[test]
fn ak_fuzz_degenerate_n1() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    compute_accel(&mut b);
    rebase_energy(&mut b);
    for step in 0..500 {
        advance(&mut b, DT);
        if step % 100 == 0 {
            fuzz_assert_sane(&b, &format!("n1 step {}", step));
        }
    }
    assert_eq!(b.len(), 1);
    assert!(b.x[0].abs() < 1e-9 && b.y[0].abs() < 1e-9);
}
