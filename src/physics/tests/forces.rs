//! Tests: force evaluation paths.
//!
//! Every path in `physics::forces` computes the same physical quantities
//! (per-body acceleration and timestep) via different algorithms and
//! summation orders. `kepler.rs`'s `l_*` and `ai_*` groups already pin the
//! Barnes-Hut tree against the direct sum. The tests below close the gaps
//! on the direct-sum side:
//!
//!   - `force_direct_on` vs closed-form Newton (2-body, 3-body, boundaries).
//!   - `compute_accel_direct_sym` vs `force_direct_on` (same answer, different
//!     summation order; `dtl` must be bit-identical because it is a max).
//!   - `force_direct_range` partition vs `force_direct_on`.
//!   - `force_direct_on_parallel` vs serial, plus bit-identical output across
//!     thread counts (the entire reason `DIRECT_CHUNK` is a constant).
//!   - `eval_forces_with` dispatch: subset `active`, empty `active`, and
//!     caller-supplied `Some(&tree)`.
//!   - `compute_accel` COM correction and end-to-end consistency.

use super::super::*;

// ===========================================================================
// Fixtures
// ===========================================================================

/// Central star plus `n - 1` planets on circular orbits at 5–505 Gm. That
/// is well outside `SOFT2.sqrt()`, so the softening term has negligible
/// effect and closed-form Newton comparisons are valid.
fn solar_disc(n: usize, seed: u128) -> Bodies {
    let mut b = Bodies::with_capacity(n);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mut rng = XorShift::new(seed);
    for _ in 1..n {
        let r = 5.0 + 500.0 * rng.next_f64();
        let th = rng.next_f64() * std::f64::consts::TAU;
        let vc = (G * M_SUN_ME / r).sqrt();
        b.push(
            r * th.cos(),
            r * th.sin(),
            -vc * th.sin(),
            vc * th.cos(),
            1.0 + 100.0 * rng.next_f64(),
            BodyKind::Planet,
        );
    }
    b
}

fn assert_close(a: f64, b: f64, rel_tol: f64, ctx: &str) {
    let denom = a.abs().max(b.abs()).max(1e-300);
    assert!(
        (a - b).abs() <= rel_tol * denom,
        "{ctx}: {a:.18e} vs {b:.18e} (rel {:.3e} > {rel_tol:.0e})",
        (a - b).abs() / denom,
    );
}

// ===========================================================================
// am — force_direct_on: closed-form correctness
// ===========================================================================

/// The only test in the whole suite that pins `force_direct_on`'s
/// **magnitude** to an independent formula. Every BH-vs-direct check in
/// `kepler.rs` uses `force_direct_on` as its own reference, so a bug in the
/// direct sum would pass all of them silently.
///
/// The expected value is computed by mirroring the source's operation
/// order exactly (`f = G·m·inv`, then `f·dx`), so the comparison is
/// bit-identical, not merely approximate.
#[test]
fn am_force_direct_on_matches_newton_2body() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let (rx, ry) = (5.0, -3.0);
    b.push(rx, ry, 0.0, 0.0, 1.0, BodyKind::Planet);

    // Mirror force_direct_on's inner loop for body 1 exactly.
    let dx = 0.0 - rx;
    let dy = 0.0 - ry;
    let d2 = dx * dx + dy * dy + SOFT2;
    let d = d2.sqrt();
    let inv = 1.0 / (d2 * d);
    let f = G * M_SUN_ME * inv;
    let (ax_expected, ay_expected) = (f * dx, f * dy);

    let (ax, ay, _, _) = force_direct_on(&b, 1);
    assert_eq!(ax.to_bits(), ax_expected.to_bits(), "planet ax");
    assert_eq!(ay.to_bits(), ay_expected.to_bits(), "planet ay");

    // Newton's third law on the star side.
    let (sx, sy, _, _) = force_direct_on(&b, 0);
    assert_close(sx, -ax / M_SUN_ME, 1e-14, "star ax");
    assert_close(sy, -ay / M_SUN_ME, 1e-14, "star ay");
}

/// Three-body non-collinear configuration: `force_direct_on` equals the
/// closed-form pairwise sum. The reference accumulates in ascending `j`
/// order with `i` skipped, matching the chained iterator in the source.
#[test]
fn am_force_direct_on_matches_newton_3body() {
    let masses = [M_SUN_ME, 1.0, 2.0];
    let pos = [(0.0, 0.0), (5.0, 0.0), (0.0, 7.0)];
    let mut b = Bodies::with_capacity(3);
    for k in 0..3 {
        b.push(pos[k].0, pos[k].1, 0.0, 0.0, masses[k], BodyKind::Planet);
    }
    for i in 0..3 {
        let (mut ax, mut ay) = (0.0, 0.0);
        for j in 0..3 {
            if i == j {
                continue;
            }
            let dx = pos[j].0 - pos[i].0;
            let dy = pos[j].1 - pos[i].1;
            let d2 = dx * dx + dy * dy + SOFT2;
            let d = d2.sqrt();
            let inv = 1.0 / (d2 * d);
            let f = G * masses[j] * inv;
            ax += f * dx;
            ay += f * dy;
        }
        let (got_x, got_y, _, _) = force_direct_on(&b, i);
        assert_close(got_x, ax, 1e-14, &format!("body {i} ax"));
        assert_close(got_y, ay, 1e-14, &format!("body {i} ay"));
    }
}

/// Single body: no `j != i` term contributes, so acceleration is exactly
/// zero and `dt_limit(0, 0) = +∞`.
#[test]
fn am_force_direct_on_single_body_is_zero() {
    let mut b = Bodies::with_capacity(1);
    b.push(3.0, -4.0, 1.0, 2.0, 1.0, BodyKind::Planet);
    let (ax, ay, _, dt) = force_direct_on(&b, 0);
    assert_eq!((ax, ay), (0.0, 0.0));
    assert!(dt.is_infinite());
}

/// Boundary indices `i = 0` (empty head) and `i = n − 1` (empty tail) must
/// traverse the correct sub-range. Matched against an inline reference.
#[test]
fn am_force_direct_on_boundary_indices() {
    let b = solar_disc(5, 7);
    let n = b.len();
    for &i in &[0usize, n / 2, n - 1] {
        let (got_x, got_y, _, _) = force_direct_on(&b, i);
        let (mut ax, mut ay) = (0.0, 0.0);
        for j in 0..n {
            if j == i {
                continue;
            }
            let dx = b.x[j] - b.x[i];
            let dy = b.y[j] - b.y[i];
            let d2 = dx * dx + dy * dy + SOFT2;
            let d = d2.sqrt();
            let inv = 1.0 / (d2 * d);
            let f = G * b.m[j] * inv;
            ax += f * dx;
            ay += f * dy;
        }
        assert_close(got_x, ax, 1e-13, &format!("i={i} ax"));
        assert_close(got_y, ay, 1e-13, &format!("i={i} ay"));
    }
}

/// Coincident bodies: `dx = dy = 0` so every pair term is exactly zero,
/// and softening keeps `s_max` finite. This is the only softening test on
/// the direct-sum side; the corresponding tree test lives in `kepler.rs`.
#[test]
fn am_force_direct_on_softens_coincident() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let (ax, ay, _, dt) = force_direct_on(&b, 0);
    assert_eq!((ax, ay), (0.0, 0.0), "coincident → no preferred direction");
    assert!(dt.is_finite() && dt > 0.0);
}

/// Close-range force must match the softened closed form and stay bounded.
/// If someone dropped `+ SOFT2` from `d2` this test would catch it via the
/// magnitude check even though coincident bodies still give zero.
#[test]
fn am_force_direct_on_close_range_bounded() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let r = 0.5 * SOFT2.sqrt();
    b.push(r, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let (ax, _, _, _) = force_direct_on(&b, 1);
    let dx = 0.0 - r;
    let d2 = dx * dx + SOFT2;
    let d = d2.sqrt();
    let inv = 1.0 / (d2 * d);
    let f = G * M_SUN_ME * inv;
    let expected = f * dx;
    assert_close(ax, expected, 1e-14, "close range");
    // As r → 0 the softened force magnitude → 0 (rather than diverging).
    assert!(
        ax.abs() < G * M_SUN_ME / SOFT2,
        "softened force {} exceeds G·M/SOFT2 = {}",
        ax.abs(),
        G * M_SUN_ME / SOFT2,
    );
}

// ===========================================================================
// am — force_direct_range: partition
// ===========================================================================

/// Empty range: identity for the accumulators, so all four values are 0.
#[test]
fn am_range_empty_is_zero() {
    let b = solar_disc(5, 1);
    let (sx, sy, sz, sm, wm) = force_direct_range(&b, 0, 0..0);
    assert_eq!((sx, sy, sz, sm, wm), (0.0, 0.0, 0.0, 0.0, 0.0));
}

/// The two ranges `[0, i)` and `(i, n)` must together reproduce
/// `force_direct_on`. The `sx`/`sy` sums differ by fp roundoff (each range
/// accumulates independently), but `s_max`/`w_max` are maxes, hence
/// order-independent, so `dt_limit` is bit-identical.
#[test]
fn am_range_partition_matches_direct() {
    for seed in 0..3u128 {
        let b = solar_disc(15, seed);
        let n = b.len();
        for i in 0..n {
            let (ref_x, ref_y, _, ref_dt) = force_direct_on(&b, i);
            let (hx, hy, _, hs, hw) = force_direct_range(&b, i, 0..i);
            let (tx, ty, _, ts, tw) = force_direct_range(&b, i, (i + 1)..n);
            assert_close(hx + tx, ref_x, 1e-13, &format!("seed {seed} i {i} x"));
            assert_close(hy + ty, ref_y, 1e-13, &format!("seed {seed} i {i} y"));
            let dt = dt_limit(hs.max(ts), hw.max(tw));
            assert_eq!(dt.to_bits(), ref_dt.to_bits(), "seed {seed} i {i} dt");
        }
    }
}

// ===========================================================================
// am — compute_accel_direct_sym: cross-check
// ===========================================================================

/// The symmetric pairwise path computes the same physical acceleration as
/// `force_direct_on`, but with a different fp summation order (it defers
/// the `j > i` partial and adds it at the end). So the accelerations agree
/// to ~machine precision, not bit-exactly. `dtl` comes from `max()`, which
/// is order-independent, so it must match bit-for-bit.
#[test]
fn am_sym_matches_direct() {
    for seed in 0..5u128 {
        let mut b = solar_disc(20, seed);
        let n = b.len();
        let ref_acc: Vec<(f64, f64, f64, f64)> = (0..n).map(|i| force_direct_on(&b, i)).collect();
        compute_accel_direct_sym(&mut b);
        for i in 0..n {
            assert_close(
                b.ax[i],
                ref_acc[i].0,
                1e-12,
                &format!("seed {seed} i {i} ax"),
            );
            assert_close(
                b.ay[i],
                ref_acc[i].1,
                1e-12,
                &format!("seed {seed} i {i} ay"),
            );
            assert_eq!(
                b.dtl[i].to_bits(),
                ref_acc[i].3.to_bits(),
                "seed {seed} i {i} dtl should be bit-identical",
            );
        }
    }
}

/// Newton's third law in accumulated form: `Σ m·a = 0`. Fails if the pair
/// weights `fi` and `fj` in `compute_accel_direct_sym` are asymmetric.
#[test]
fn am_sym_conserves_momentum() {
    let mut b = solar_disc(30, 42);
    compute_accel_direct_sym(&mut b);
    let (mut px, mut py) = (0.0, 0.0);
    let scale: f64 = (0..b.len()).map(|i| b.m[i] * b.ax[i].hypot(b.ay[i])).sum();
    for i in 0..b.len() {
        px += b.m[i] * b.ax[i];
        py += b.m[i] * b.ay[i];
    }
    assert!(px.abs() < 1e-10 * scale, "Σm·ax = {px} vs scale {scale}");
    assert!(py.abs() < 1e-10 * scale, "Σm·ay = {py} vs scale {scale}");
}

/// Zero- and one-body systems exercise the loop's zero-iteration path.
#[test]
fn am_sym_empty_and_single() {
    let mut b0 = Bodies::with_capacity(0);
    compute_accel_direct_sym(&mut b0); // must not panic
    let mut b1 = Bodies::with_capacity(1);
    b1.push(1.0, 2.0, 3.0, 4.0, 1.0, BodyKind::Planet);
    compute_accel_direct_sym(&mut b1);
    assert_eq!((b1.ax[0], b1.ay[0]), (0.0, 0.0));
    assert!(b1.dtl[0].is_infinite());
}

/// Coincident pair through the symmetric path: forces are exactly zero,
/// `dtl` is finite (softening bound at `SOFT2`).
#[test]
fn am_sym_handles_coincident() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    compute_accel_direct_sym(&mut b);
    for i in 0..2 {
        assert_eq!((b.ax[i], b.ay[i]), (0.0, 0.0));
        assert!(b.dtl[i].is_finite() && b.dtl[i] > 0.0);
    }
}

// ===========================================================================
// am — force_direct_on_parallel: correctness and determinism
// ===========================================================================

/// The chunked parallel path must agree with the serial path to fp
/// precision (chunk partial sums change the summation tree). The tested
/// bodies straddle chunk boundaries at `DIRECT_CHUNK` so that both
/// sub-ranges of the j-loop are exercised.
#[test]
fn am_parallel_matches_serial() {
    let n = 2 * DIRECT_CHUNK + 300;
    let b = solar_disc(n, 1);
    let probes = [
        0usize,
        1,
        DIRECT_CHUNK - 1,
        DIRECT_CHUNK,
        DIRECT_CHUNK + 1,
        n / 2,
        n - 2,
        n - 1,
    ];
    for &i in &probes {
        let (ref_x, ref_y, _, ref_dt) = force_direct_on(&b, i);
        let (par_x, par_y, _, par_dt) = force_direct_on_parallel(&b, i);
        assert_close(par_x, ref_x, 1e-11, &format!("i {i} x"));
        assert_close(par_y, ref_y, 1e-11, &format!("i {i} y"));
        // dt is a max → order-independent → bit-identical.
        assert_eq!(par_dt.to_bits(), ref_dt.to_bits(), "i {i} dt");
    }
}

/// The chunked path must be **bit-identical** across thread counts. This is
/// the contract that `DIRECT_CHUNK`'s constant-ness exists to guarantee; a
/// change that made chunk boundaries depend on `rayon::current_num_threads`
/// would fail here even though the result stayed approximately correct.
#[test]
fn am_parallel_is_deterministic_across_threads() {
    let n = 2 * DIRECT_CHUNK + 300;
    let b = solar_disc(n, 42);
    let i = n / 2;
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| force_direct_on_parallel(&b, i))
    };
    let one = run(1);
    let many = run(8);
    assert_eq!(one.0.to_bits(), many.0.to_bits(), "sx across thread counts");
    assert_eq!(one.1.to_bits(), many.1.to_bits(), "sy across thread counts");
    assert_eq!(one.2.to_bits(), many.2.to_bits(), "sz across thread counts");
    assert_eq!(one.3.to_bits(), many.3.to_bits(), "dt across thread counts");
}

/// Boundary `i` values must still partition the j-loop correctly in the
/// chunk emitter (head empty for `i = 0`, tail empty for `i = n − 1`).
#[test]
fn am_parallel_boundary_indices() {
    let n = 2 * DIRECT_CHUNK + 300;
    let b = solar_disc(n, 3);
    for &i in &[0usize, n - 1] {
        let (par_x, par_y, _, par_dt) = force_direct_on_parallel(&b, i);
        let (ref_x, ref_y, _, ref_dt) = force_direct_on(&b, i);
        assert_close(par_x, ref_x, 1e-11, &format!("i {i} x"));
        assert_close(par_y, ref_y, 1e-11, &format!("i {i} y"));
        assert_eq!(par_dt.to_bits(), ref_dt.to_bits(), "i {i} dt");
    }
}

// ===========================================================================
// am — eval_forces_with: dispatch and subset semantics
// ===========================================================================

/// Empty `active` must return without touching `ax`, `ay`, or `dtl` on
/// any body. Poisoned sentinels catch even a write of a computed value.
#[test]
fn am_eval_forces_empty_active_is_noop() {
    let mut b = solar_disc(10, 1);
    for i in 0..b.len() {
        b.ax[i] = 999.0;
        b.ay[i] = -999.0;
        b.dtl[i] = -1.0;
    }
    eval_forces(&mut b, &[]);
    for i in 0..b.len() {
        assert_eq!(
            (b.ax[i], b.ay[i], b.dtl[i]),
            (999.0, -999.0, -1.0),
            "body {i}"
        );
    }
}

/// Only bodies listed in `active` are updated. `step_block` depends on
/// this to keep non-active accelerations from the previous tick.
#[test]
fn am_eval_forces_only_updates_active() {
    let mut b = solar_disc(15, 3);
    let sentinel: Vec<(f64, f64, f64)> = (0..b.len())
        .map(|i| {
            let s = (500.0 + i as f64, 600.0 + i as f64, -1.0 - i as f64);
            b.ax[i] = s.0;
            b.ay[i] = s.1;
            b.dtl[i] = s.2;
            s
        })
        .collect();
    let active = [1u32, 5, 10];
    eval_forces(&mut b, &active);
    for i in 0..b.len() {
        if active.contains(&(i as u32)) {
            assert_ne!(b.ax[i], sentinel[i].0, "active body {i} not written");
        } else {
            assert_eq!(
                (b.ax[i], b.ay[i], b.dtl[i]),
                sentinel[i],
                "inactive body {i} touched"
            );
        }
    }
}

/// Single-body `eval_forces` goes through the serial path and yields zero
/// acceleration and infinite `dtl` (no pair term exists).
#[test]
fn am_eval_forces_single_body() {
    let mut b = Bodies::with_capacity(1);
    b.push(1.0, 2.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    eval_forces(&mut b, &[0]);
    assert_eq!((b.ax[0], b.ay[0]), (0.0, 0.0));
    assert!(b.dtl[0].is_infinite());
}

/// When the caller passes `Some(&tree)` and dispatch selects the tree path,
/// that tree is used verbatim — not rebuilt. A stale tree (built before
/// bodies moved) yields `accel_for` on the stale tree, which differs from a
/// fresh build.
#[test]
fn am_eval_forces_uses_provided_tree() {
    let n = 2000usize;
    if n < BH_THRESHOLD || n < DIRECT_ACTIVE_MAX {
        return; // tree path can't be selected; nothing to test here
    }
    let mut b = solar_disc(n, 4);
    let tree = Tree::build(&b);
    // Advance positions so the tree is stale.
    for i in 0..b.len() {
        b.x[i] += 1e-3;
    }
    let active: Vec<u32> = (0..n as u32).collect();
    eval_forces_with(&mut b, &active, Some(&tree));
    let stale_first = b.ax[0];
    // Must exactly match `accel_for` on the *stale* tree.
    let (exact_x, _, _, _) = tree.accel_for(&b, 0);
    assert_eq!(
        stale_first.to_bits(),
        exact_x.to_bits(),
        "stale tree not used"
    );
    // Fresh build should differ.
    let (fresh_x, _, _, _) = Tree::build(&b).accel_for(&b, 0);
    assert_ne!(
        stale_first.to_bits(),
        fresh_x.to_bits(),
        "stale and fresh trees must differ on shifted input",
    );
}

/// With a tiny `active` set the tree path is gated off (`DIRECT_ACTIVE_MAX`)
/// even though a tree is provided — the direct serial path fires, and its
/// result matches `force_direct_on` bit-for-bit.
#[test]
fn am_eval_forces_ignores_tree_on_small_active() {
    if 1 >= DIRECT_ACTIVE_MAX {
        return; // tree would fire at n = 1 active; not our test case
    }
    let n = 500usize;
    let mut b = solar_disc(n, 5);
    let tree = Tree::build(&b);
    eval_forces_with(&mut b, &[0u32], Some(&tree));
    let (ref_x, ref_y, _, _) = force_direct_on(&b, 0);
    assert_eq!(b.ax[0].to_bits(), ref_x.to_bits(), "serial ax");
    assert_eq!(b.ay[0].to_bits(), ref_y.to_bits(), "serial ay");
}

// ===========================================================================
// am — compute_accel: end-to-end
// ===========================================================================

/// Empty system: no-op.
#[test]
fn am_compute_accel_empty_is_noop() {
    let mut b = Bodies::with_capacity(0);
    compute_accel(&mut b);
    assert_eq!(b.len(), 0);
}

/// Single body: force is zero, COM correction is zero, `dtl` is +∞.
#[test]
fn am_compute_accel_single_body() {
    let mut b = Bodies::with_capacity(1);
    b.push(3.0, -4.0, 5.0, 6.0, 1.0, BodyKind::Planet);
    compute_accel(&mut b);
    assert_eq!((b.ax[0], b.ay[0]), (0.0, 0.0));
    assert!(b.dtl[0].is_infinite());
}

/// `compute_accel` subtracts the mass-weighted mean acceleration, so
/// `Σ m·a = 0` after the call. Tested across both internal paths: `n = 5`
/// stays on `compute_accel_direct_sym`, `n = 500` goes through
/// `eval_forces`.
#[test]
fn am_compute_accel_zeroes_com_accel() {
    for n in [5usize, 500] {
        let mut b = solar_disc(n, n as u128);
        compute_accel(&mut b);
        let mt: f64 = b.m.iter().sum();
        let (mut px, mut py) = (0.0, 0.0);
        for i in 0..n {
            px += b.m[i] * b.ax[i];
            py += b.m[i] * b.ay[i];
        }
        let scale = (0..n).map(|i| b.m[i] * b.ax[i].hypot(b.ay[i])).sum::<f64>() / mt;
        let cx = px / mt;
        let cy = py / mt;
        assert!(cx.abs() < 1e-12 * scale, "n {n}: cx {cx} vs scale {scale}");
        assert!(cy.abs() < 1e-12 * scale, "n {n}: cy {cy} vs scale {scale}");
    }
}

/// `compute_accel` output equals `force_direct_on`'s per-body acceleration
/// plus a single constant offset (the COM correction). Removing that offset
/// (weighted mean) recovers the raw acceleration to fp precision on both
/// the small-`n` symmetric path and the medium-`n` dispatch path.
#[test]
fn am_compute_accel_matches_force_direct_on() {
    for n in [5usize, 200] {
        let mut b = solar_disc(n, n as u128);
        // Park the disc: `compute_accel` adds the 1PN/2.5PN terms whenever a
        // body moves faster than ~3 km/s (`PN_V_OVER_C_MIN`), and those are a
        // relative O(v²/c²) ≈ 1e-8 correction that `force_direct_on` — the
        // Newtonian reference — does not contain. With zero velocities the
        // comparison is purely Newtonian and the 1e-10 tolerance is meaningful.
        b.vx.iter_mut().for_each(|v| *v = 0.0);
        b.vy.iter_mut().for_each(|v| *v = 0.0);
        b.vz.iter_mut().for_each(|v| *v = 0.0);
        let ref_acc: Vec<(f64, f64, f64, f64)> = (0..n).map(|i| force_direct_on(&b, i)).collect();
        compute_accel(&mut b);
        let mt: f64 = b.m.iter().sum();
        let (mut cx, mut cy) = (0.0, 0.0);
        for i in 0..n {
            cx += b.m[i] * (ref_acc[i].0 - b.ax[i]);
            cy += b.m[i] * (ref_acc[i].1 - b.ay[i]);
        }
        cx /= mt;
        cy /= mt;
        for i in 0..n {
            assert_close(
                b.ax[i] + cx,
                ref_acc[i].0,
                1e-10,
                &format!("n {n} i {i} ax"),
            );
            assert_close(
                b.ay[i] + cy,
                ref_acc[i].1,
                1e-10,
                &format!("n {n} i {i} ay"),
            );
        }
    }
}

/// Every force path produces finite `ax`, `ay`, `dtl` on well-separated
/// bodies. Closes the "no NaN reaches state vectors" invariant at the
/// force-evaluation boundary rather than after integration.
#[test]
fn am_compute_accel_finite_across_sizes() {
    for n in [2usize, 3, 10, 50, 200] {
        let mut b = solar_disc(n, n as u128);
        compute_accel(&mut b);
        for i in 0..n {
            assert!(b.ax[i].is_finite(), "n {n} body {i} ax");
            assert!(b.ay[i].is_finite(), "n {n} body {i} ay");
            assert!(b.dtl[i] > 0.0, "n {n} body {i} dtl");
            assert!(!b.dtl[i].is_nan(), "n {n} body {i} dtl NaN");
        }
    }
}
