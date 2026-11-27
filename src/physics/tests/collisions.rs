//! Tests: collisions — hit-and-run, debris, grazing, hot path, rubble,
//! fragmentation, silent merge, and detection coverage.
//!
//! Notation: `b` (italic `b`) is the dimensionless impact parameter
//! (`impact_offset / rr`) in `[0, 1]`; ratios of `√(2GM/rr)` are used as
//! "fraction of escape speed". Both are computed inside `collide` from the
//! supplied `pair(...)` fixture.

use super::super::*;
use super::shared::*;

// ===========================================================================
// b — hit-and-run
// ===========================================================================
/// A grazing, fast encounter must deflect and strip mass instead of
/// merging: `bb > HIT_RUN_B` and `ratio > HIT_RUN_VEL_RATIO`. The
/// second clause of the test confirms that the *same* encounter at
/// `bb = 0.1` (nearly head-on) does not hit-and-run — it merges or
/// fragments, so the `bb` gate is genuinely selecting the branch.
#[test]
fn b_hit_and_run_at_high_b() {
    let mut b = pair(1.0, 0.5, 1.5, 0.9);
    shift(&mut b, 10.0, -4.0, 0.5, 0.3);
    let (mb0, ms0) = (b.m[0], b.m[1]);
    let v0 = (b.vx[1] - b.vx[0], b.vy[1] - b.vy[0]);
    let s0 = snapshot(&b);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 2, "both bodies must survive");
    assert_eq!(b.stats.merges, 0);
    assert_eq!(b.stats.impacts, 1);
    assert!(
        b.m[0] > mb0 && b.m[1] < ms0,
        "light body loses mass to the heavy body"
    );
    // Deflection: relative-velocity direction must change and speed must
    // drop (inelastic, `HIT_RUN_RESTITUTION < 1`).
    let v1 = (b.vx[1] - b.vx[0], b.vy[1] - b.vy[0]);
    let (n0, n1) = (v0.0.hypot(v0.1), v1.0.hypot(v1.1));
    let cos = (v0.0 * v1.0 + v0.1 * v1.1) / (n0 * n1);
    assert!(cos < 0.9999, "path must be deflected");
    assert!(n1 < n0, "bounce is inelastic");
    assert_conserved(s0, &b);

    // Negative control: same speed, nearly head-on ⇒ not a hit-and-run.
    let mut h = pair(1.0, 0.5, 1.5, 0.1);
    collision_pass(&mut h, DT, None);
    assert_ne!(h.len(), 2, "head-on impact must not be a hit-and-run");
}

// ===========================================================================
// c — moon-forming impacts
// ===========================================================================
/// A canonical Moon-forming impact: a Mars-mass impactor strikes a
/// 1 M⊕ target at ~0.7–1.0 × escape speed with `b = 0.7`. The result
/// must be a surviving target, a bound prograde fragment population,
/// and an ejected mass in the lunar range — i.e. the disk-forming
/// branch of `collide`, not a merge or a hit-and-run.
#[test]
fn c_moon_forming_impact_makes_a_disk() {
    for &ratio in &[0.7, 0.85, 1.0] {
        let mut b = pair(1.0, 0.107, ratio, 0.7);
        let sense = -1.0;
        collision_pass(&mut b, DT, None);
        assert_eq!(b.stats.impacts, 1, "ratio {}", ratio);
        let ejected = mass(&b) - b.m[0];
        let bound = bound_prograde(&b, sense);
        eprintln!(
            "ratio {:.2}: bodies {} ejected {:.4} M_E ({:.2} lunar) bound prograde {} spin {:.4}",
            ratio,
            b.len(),
            ejected,
            ejected / 0.0123,
            bound,
            b.spin[0]
        );
        // At least the remnant + a handful of fragments (n >= FRAG_COUNT_BASE).
        assert!(b.len() >= 4);
        // Ejecta mass must be lunar-ish, not zero and not the whole impactor.
        assert!(ejected > 0.005 && ejected < 0.1, "ejected {}", ejected);
        // Remnant must keep (indeed exceed) the target mass — it also
        // absorbs some impactor material.
        assert!(b.m[0] > 1.0, "remnant keeps the target mass");
        // Bulk of the fragments must be bound and prograde; the disk
        // branch puts them on near-circular orbits in the impact sense.
        assert!(
            bound * 2 >= b.len() - 1,
            "most fragments must be bound and prograde"
        );
    }
}

// ===========================================================================
// d — grazing vs head-on
// ===========================================================================
/// The same impact energy delivered obliquely must produce much more
/// spin on the survivor and put more material into orbit than a head-on
/// delivery — the physical reason the Moon-forming impact works at all.
#[test]
fn d_grazing_differs_from_head_on() {
    let sense = -1.0;
    let mut head = pair(1.0, 0.107, 1.2, 0.0);
    let mut graze = pair(1.0, 0.107, 1.2, 0.7);
    collision_pass(&mut head, DT, None);
    collision_pass(&mut graze, DT, None);
    let (bh, bg) = (bound_prograde(&head, sense), bound_prograde(&graze, sense));
    eprintln!(
        "head-on: bodies {} spin {:.4} bound-prograde {} | grazing: bodies {} spin {:.4} bound-prograde {}",
        head.len(), head.spin[0], bh, graze.len(), graze.spin[0], bg
    );
    // `L` handed to the remnant scales roughly linearly with `b`, so a
    // 5× margin is generous but still catches a missing `b` dependence.
    assert!(
        graze.spin[0].abs() > 5.0 * head.spin[0].abs() + 1e-12,
        "grazing spin must be larger"
    );
    assert!(bg > bh, "grazing impact must put more fragments into orbit");
}

// ===========================================================================
// m — collision detection
// ===========================================================================
/// Small population ⇒ brute-force path (`n < HASH_THRESHOLD`). Two
/// overlapping bodies must be reported as a pair.
#[test]
fn m_find_collisions_detects_overlap() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let r = b.contact_r(0);
    b.push(r - 0.001, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let pairs = find_collisions(&b, false, DT);
    assert_eq!(pairs, vec![(0, 1)]);
}

/// Large population ⇒ hash-grid path. The grid must find *exactly* the
/// same pairs as an O(n²) scan of the same configuration, in the same
/// order after sorting. This is the ground-truth test for the broad
/// phase's correctness — it would catch a missing 3×3 neighbour probe,
/// a wrong cell size, or a hash-key mix-up.
#[test]
fn m_find_collisions_hash_matches_bruteforce() {
    // 80 bodies on a 0.01-spaced line. Each has r ≈ 0.0064 (1 M_earth),
    // so adjacent pairs overlap (sep 0.01 < sum 0.0127) and pairs two
    // or more steps apart do not.
    let mut b = Bodies::with_capacity(80);
    for k in 0..80 {
        b.push(k as f64 * 0.01, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    }
    assert!(b.len() >= HASH_THRESHOLD, "hash path is what we're testing");

    let mut reference: Vec<(usize, usize)> = Vec::new();
    for i in 0..b.len() {
        for j in (i + 1)..b.len() {
            if pair_hit(&b, i, j, false) {
                reference.push((i, j));
            }
        }
    }
    reference.sort_unstable();

    let pairs = find_collisions(&b, false, DT);
    assert_eq!(pairs, reference);
    assert!(!pairs.is_empty(), "test should produce collisions");
}

/// Immunity is the mechanism that prevents a resolved pair from being
/// re-resolved on the same step. Both sides must be able to trigger it.
#[test]
fn m_find_collisions_respects_immunity() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let r = b.contact_r(0);
    b.push(r - 0.001, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    // Sanity: without immunity, they collide.
    assert_eq!(find_collisions(&b, false, DT), vec![(0, 1)]);
    // Now make body 0 immune.
    b.imm[0] = 10.0;
    assert!(find_collisions(&b, false, DT).is_empty());
    // And immune on the other body.
    b.imm[0] = 0.0;
    b.imm[1] = 10.0;
    assert!(find_collisions(&b, false, DT).is_empty());
}

// ===========================================================================
// ab — fast (hot) collision path
// ===========================================================================
/// Two-hot-body case: `find_collisions_hot` falls through to its O(n²)
/// fallback (below `HOT_GRID_MIN`). Must agree with the full search.
#[test]
fn ab_hot_small_list_matches_full() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let r = b.contact_r(0);
    b.push(r - 0.001, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let hot = vec![0usize, 1usize];
    assert_eq!(find_collisions_hot(&b, &hot, false, DT), vec![(0, 1)]);
}

/// Thirty-hot-body case: crosses `HOT_GRID_MIN`, so the counting-sort
/// hash-grid branch of `find_collisions_hot` runs. Its output must match
/// `find_collisions` exactly — this is the ground-truth test for the
/// hot-path broad phase.
#[test]
fn ab_hot_grid_matches_full_path() {
    // 30 hot bodies on a 0.01 grid → exercises the grid branch.
    let mut b = Bodies::with_capacity(30);
    for k in 0..30 {
        b.push(k as f64 * 0.01, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    }
    const { assert!(30 >= HOT_GRID_MIN) };
    let hot: Vec<usize> = (0..30).collect();
    let mut full = find_collisions(&b, false, DT);
    let mut fast = find_collisions_hot(&b, &hot, false, DT);
    full.sort_unstable();
    fast.sort_unstable();
    assert_eq!(full, fast);
    assert!(!full.is_empty());
}

// ===========================================================================
// ab (extension) — hot-vs-cold complement
// ===========================================================================
/// A fast belt impactor must be caught even when the hot set is small and
/// the total body count is high. `collision_pass` used to gate hot-vs-cold
/// on `h·n ≤ 2 000 000` and skip the sweep entirely above that, so these
/// projectiles tunnelled straight through the cold belt: their *only* pair
/// was hot-vs-cold, and the gate dropped it. The belt below is sized so
/// `1e2 · 3.7e4 > 2e6` (which trips the old gate) while h² = 1e4 stays
/// below `HOT_FULL_SCAN_MIN`, so the dense-population full sweep does not
/// rescue the case. Every projectile must merge with the belt body on its
/// own column.
#[test]
fn ab_hot_vs_cold_resolves_belt_impactors_at_high_n() {
    let mut b = Bodies::with_capacity(40_000);
    // Cold belt: a 220 x 170 sheet of touching bodies at z = 0. Spacing 2.0
    // with contact radius 0.99 leaves neighbours 2e-2 apart — close, but
    // never overlapping, so the only possible pairs are the ones we place.
    for k in 0..220usize {
        for l in 0..170usize {
            let i = b.push(
                k as f64 * 2.0,
                l as f64 * 2.0,
                0.0,
                0.0,
                1e-4,
                BodyKind::Planet,
            );
            b.r[i] = 0.99;
        }
    }
    // A hundred projectiles, each parked 0.9 units above its own belt column.
    let mut hot = Vec::new();
    for k in 0..100usize {
        let i = b.push(k as f64 * 4.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
        b.z[i] = 0.9;
        b.r[i] = 0.5;
        hot.push(i);
    }
    let n = b.len();
    assert!(
        hot.len().saturating_mul(n) > 2_000_000,
        "must trip the old h·n gate"
    );
    assert!(
        hot.len().saturating_mul(hot.len()) <= HOT_FULL_SCAN_MIN,
        "must not be the dense-population fallback"
    );

    collision_pass(&mut b, DT, Some(&hot));

    assert_eq!(
        b.stats.merges, 100,
        "every projectile must merge with its belt body"
    );
    assert_eq!(b.stats.impacts, 0, "parked impactors merge, they don't hit-and-run");
    assert_eq!(b.len(), n - 100, "one belt body per projectile is compacted away");
}

/// The swept narrow phase must catch a fast projectile whose whole crossing
/// happens inside one sub-tick — neither endpoint is near the belt body, so
/// only the segment test sees the hit. This pins the anti-tunnelling reach
/// of the hot-vs-cold path specifically (the hot-hot swept case is covered
/// by the `n_swept_*` tests).
#[test]
fn ab_hot_vs_cold_swept_catches_fast_impactor() {
    let mut b = Bodies::with_capacity(300);
    // A small cold belt ringed around the target column; the projectile's
    // path misses every other body.
    for k in 0..15usize {
        for l in 0..15usize {
            let i = b.push(
                k as f64 * 2.0,
                l as f64 * 2.0,
                0.0,
                0.0,
                1e-4,
                BodyKind::Planet,
            );
            b.r[i] = 0.99;
        }
    }
    // Projectile: starts 20 units above the belt and ends 20 below it within
    // one sub-tick. Endpoints are far outside contact range.
    let target = 0usize; // belt body at (0, 0, 0)
    let p = b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.r[p] = 0.5;
    b._pz[p] = 20.0;
    b.z[p] = -20.0;
    assert!(!pair_hit(&b, p, target, false), "no endpoint overlap");
    collision_pass(&mut b, DT, Some(&[p]));
    assert_eq!(b.stats.merges, 1, "swept hot-vs-cold must merge the crossing");
}

/// Ground truth for the always-on hot-vs-cold complement: for a static
/// population (no `_p -> x` motion, so the swept and static tests agree),
/// the union of `find_collisions_hot` and `hot_vs_cold` must equal the full
/// sweep restricted to pairs that touch the hot set. Both sizes run so each
/// branch is exercised: the small one takes the brute-force full path, the
/// hot O(n²) loop and the hot-vs-cold x-scan; the large one takes the hash
/// grids in all three.
#[test]
fn ab_hot_vs_cold_matches_full_sweep() {
    for &(n, hot_n, r) in &[(40usize, 20usize, 0.09), (520, 120, 0.05)] {
        let mut b = Bodies::with_capacity(n);
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (s >> 33) as f64 / (1u64 << 31) as f64
        };
        for _ in 0..n {
            let i = b.push(
                (rnd() - 0.5) * 0.4,
                (rnd() - 0.5) * 0.4,
                0.0,
                0.0,
                1.0,
                BodyKind::Planet,
            );
            b.r[i] = r;
        }
        let hot: Vec<usize> = (0..hot_n).collect();

        // `hot` is the index prefix, so "touches the hot set" == `i < hot_n`.
        let mut expected: Vec<(usize, usize)> = find_collisions(&b, false, DT)
            .into_iter()
            .filter(|&(i, j)| i < hot_n || j < hot_n)
            .collect();
        expected.sort_unstable();
        expected.dedup();
        assert!(!expected.is_empty(), "n={n}: scenario must produce pairs");

        let mut got = find_collisions_hot(&b, &hot, false, DT);
        got.extend(hot_vs_cold(&b, &hot, &mut ColdScan::default()));
        got.sort_unstable();
        got.dedup();

        assert_eq!(
            got, expected,
            "n={n}: hot ∪ hot-cold must equal the full sweep"
        );
    }
}

// ===========================================================================
// ac — rubble bounce
// ===========================================================================
/// `rubble_bounce`'s guard is `if vn >= 0.0 { return; }`. In the *graze*
/// limit the perpendicular offset `d ≥ rr`, so `s = sqrt(rr² − d²) = 0`
/// and `vn = (d·hux − s·ux)/rr · dvx + ...` reduces to `vn = 0`, which
/// hits the guard. The test constructs this limit directly to exercise
/// the no-op path; the *firing* path is covered by `r_*` below.
#[test]
fn ac_rubble_bounce_noop_on_graze() {
    // `rubble_bounce` returns early when `vn >= 0`. In a head-on
    // configuration `vn = -s·v/rr` is always < 0 (s = sqrt(rr² - d²), v > 0),
    // so head-on always fires regardless of approach vs separation sign.
    // The guard only fires for a *graze*: perpendicular offset d >= rr,
    // which makes s = 0 -> vn = 0.
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 0.001, BodyKind::Planet);
    b.push(0.0, 0.0, 1.0, 0.0, 0.001, BodyKind::Planet);
    b.origin[0] = BodyOrigin::ImpactDebris;
    b.origin[1] = BodyOrigin::ImpactDebris;
    let rr = b.contact_r(0) + b.contact_r(1);
    // Move body 1 perpendicular to its velocity by more than rr.
    b.y[1] = rr + 0.001;
    let v0x = b.vx[1];
    let v0y = b.vy[1];
    let (px0, py0) = momentum(&b);
    rubble_bounce(&mut b, 0, 1, rr);
    assert_eq!(b.vx[1], v0x);
    assert_eq!(b.vy[1], v0y);
    assert_eq!(b.stats.bounces, 0);
    let (px1, py1) = momentum(&b);
    assert_eq!(px1, px0);
    assert_eq!(py1, py0);
}

// ===========================================================================
// ad — fragmentation & disruption
// ===========================================================================
/// Fragments must (a) sum exactly to the requested total, (b) all be
/// positive, and (c) have a heavy-tailed size distribution — the Pareto
/// slope `FRAG_SLOPE/3 = 0.95 < 1` has an infinite mean per draw, so the
/// variance must be measured across many draws rather than relied on
/// within one.
#[test]
fn ad_frag_masses_sum_and_skew() {
    let mut rng = XorShift::new(11);
    let md = 0.1;

    // Per-draw invariants hold for every n.
    for n in [2, 5, 20] {
        for _ in 0..50 {
            let ms = frag_masses(&mut rng, n, md);
            assert_eq!(ms.len(), n);
            let sum: f64 = ms.iter().sum();
            assert!((sum - md).abs() < 1e-9, "n={} sum {} vs {}", n, sum, md);
            assert!(ms.iter().all(|&m| m > 0.0), "n={} non-positive mass", n);
        }
    }

    // The Pareto tail (FRAG_SLOPE = 2.85 → beta = 0.95 < 1) has infinite
    // mean, so a *single* draw's CV is not reliable. Average it.
    for n in [5, 20] {
        let draws = 100;
        let mut cv_sum = 0.0;
        for _ in 0..draws {
            let ms = frag_masses(&mut rng, n, md);
            let mean = md / n as f64;
            let var: f64 = ms.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            cv_sum += var.sqrt() / mean;
        }
        let cv_mean = cv_sum / draws as f64;
        assert!(
            cv_mean > 0.20,
            "n={} mean CV over {} draws = {:.3} (should be heavy-tailed)",
            n,
            draws,
            cv_mean
        );
    }
}

/// Physical constants: 1 M⊕ must come out at Earth's bulk density, and
/// `rigidity` must interpolate 0 → 1 across the fluid/rigid Roche range.
#[test]
fn ad_density_and_rigidity() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let d = density_gcc(&b, 0);
    assert!((d - 5.51).abs() < 0.5, "Earth density {}", d);
    assert!((rigidity(6.0) - 1.0).abs() < 1e-9);
    assert!((rigidity(1.0) - 0.0).abs() < 1e-9);
    let mid = (ROCHE_RHO_FLUID + ROCHE_RHO_RIGID) * 0.5;
    assert!((rigidity(mid) - 0.5).abs() < 1e-9);
}

/// Tidal disruption: a planet inside the Roche limit of a star must be
/// shredded into a small number of fragments, with momentum and angular
/// momentum conserved to round-off.
#[test]
fn ad_disrupt_fragment_count_and_conservation() {
    let mut b = Bodies::with_capacity(20);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let p = b.push(1.0, 0.0, 0.0, 0.0, 10.0, BodyKind::Planet);
    let before = b.len();
    let l0 = total_angular_momentum(&b);
    let (px0, py0) = momentum(&b);
    assert!(disrupt(&mut b, p, 0));
    // compact() removes the dead planet; new_len = before - 1 + n_frag.
    b.compact();
    let fragments = b.len() - before + 1;
    assert!((4..=12).contains(&fragments), "fragments {}", fragments);
    let (px1, py1) = momentum(&b);
    assert!((px1 - px0).abs() < 1e-8 * (1.0 + px0.abs()));
    assert!((py1 - py0).abs() < 1e-8 * (1.0 + py0.abs()));
    assert!((total_angular_momentum(&b) - l0).abs() < 1e-8 * (1.0 + l0.abs()));
}

// ===========================================================================
// af — silent merge
// ===========================================================================
/// `silent = true` must suppress both the flash and the event-log entry
/// while still performing the physical merge and updating `stats.merges`.
/// This path is used by `collide` to fold the pair before spawning
/// fragments, so any spurious side effect here would double-count.
#[test]
fn af_silent_merge_does_not_log_or_flash() {
    let mut b = pair(1.0, 0.1, 0.3, 0.5);
    let l0 = total_angular_momentum(&b);
    let (px0, py0) = momentum(&b);
    let m0 = mass(&b);
    let ev0 = b.events.len();
    let fl0 = b.flashes.len();
    merge_impl(&mut b, 0, 1, /*silent=*/ true);
    // merge_impl marks the victim dead but leaves compaction to the caller.
    assert!(b.dead.iter().any(|&d| d), "victim should be dead");
    b.compact();
    assert_eq!(b.len(), 1);
    assert_eq!(b.events.len(), ev0);
    assert_eq!(b.flashes.len(), fl0);
    assert_eq!(b.stats.merges, 1);
    assert!((mass(&b) - m0).abs() < 1e-12);
    let (px1, py1) = momentum(&b);
    assert!((px1 - px0).abs() < 1e-9);
    assert!((py1 - py0).abs() < 1e-9);
    assert!((total_angular_momentum(&b) - l0).abs() < 1e-9);
}

/// The `[BD_M_MAX, STAR_M_MIN)` mass gap must be reached *without* mass
/// creation: two brown dwarfs whose sum lands in the gap must stay a
/// brown dwarf (below `STAR_M_MIN`) with total mass preserved. Guards
/// against an ignition rule that rounds up across the gap.
#[test]
fn merge_never_creates_mass_across_bd_star_gap() {
    // Just above BD_M_MAX (25 440 M⊕), below STAR_M_MIN (26 636 M⊕):
    // midpoint ≈ 26 037; use 13 018 per body.
    let m_each = 13_018.0;
    assert!(2.0 * m_each > BD_M_MAX);
    assert!(2.0 * m_each < STAR_M_MIN);
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, m_each, BodyKind::BrownDwarf);
    b.push(0.001, 0.0, 0.0, 0.0, m_each, BodyKind::BrownDwarf);
    let m0: f64 = b.m.iter().sum();
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1);
    assert_eq!(b.kind[0], BodyKind::BrownDwarf, "still below STAR_M_MIN");
    assert!(
        (b.m[0] - m0).abs() < 1e-9 * m0,
        "mass created: {} -> {}",
        m0,
        b.m[0]
    );
}

// ===========================================================================
// n — swept (anti-tunnelling) collision detection
// ===========================================================================
/// A pair moving fast enough to cross within one `DT` has no overlap at
/// either endpoint. Only `sweep = true` can catch it — this is the exact
/// case that motivated the swept segment test in `pair_hit_rr`.
#[test]
fn n_swept_detects_tunnel() {
    let mut b = Bodies::with_capacity(2);
    let m = 1.0;
    // Start 1 unit apart, closing at ±500. With DT = 1/480 a step moves
    // each by ~1.04 units, so they cross. Sum of contact radii is ~0.0127,
    // so neither endpoint overlaps.
    b.push(-0.5, 0.0, 500.0, 0.0, m, BodyKind::Planet);
    b.push(0.5, 0.0, -500.0, 0.0, m, BodyKind::Planet);
    // _p = start of the step; x = end of the step.
    b._px[0] = -0.5;
    b._py[0] = 0.0;
    b._px[1] = 0.5;
    b._py[1] = 0.0;
    b.x[0] = -0.5 + 500.0 * DT;
    b.x[1] = 0.5 - 500.0 * DT;

    assert!(
        find_collisions(&b, false, DT).is_empty(),
        "non-swept must miss the crossing"
    );
    assert_eq!(
        find_collisions(&b, true, DT),
        vec![(0, 1)],
        "swept must catch it"
    );
}

/// A pair that starts overlapping and moves *apart* must not be flagged —
/// the swept test's `t ≤ 0` guard is what makes this work. Without it a
/// resolved pair could be re-resolved on the very next step.
#[test]
fn n_swept_ignores_separating_pair() {
    let mut b = Bodies::with_capacity(2);
    let m = 1.0;
    b.push(0.0, 0.0, -500.0, 0.0, m, BodyKind::Planet);
    b.push(0.005, 0.0, 500.0, 0.0, m, BodyKind::Planet);
    b._px[0] = 0.0;
    b._py[0] = 0.0;
    b._px[1] = 0.005;
    b._py[1] = 0.0;
    // After one step they're >1 unit apart.
    b.x[0] = -1.0;
    b.x[1] = 1.0;
    assert!(!pair_hit(&b, 0, 1, true), "separating pair must not hit");
}

/// A currently-overlapping pair must be reported regardless of `sweep` —
/// the sweep is an *additional* test, not a replacement for the static
/// overlap check.
#[test]
fn n_sweep_still_fires_on_current_overlap() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let r = b.contact_r(0);
    b.push(r * 0.5, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    assert!(pair_hit(&b, 0, 1, false));
    assert!(pair_hit(&b, 0, 1, true));
}

// ===========================================================================
// o — collision_pass early return
// ===========================================================================
/// `collision_pass` bails when fewer than two bodies exist. Guarded so a
/// regression that changed the condition (e.g. `<= 2`) would show up as
/// a panic in the hot loop of the simulation, not here.
#[test]
fn o_collision_pass_noop_on_empty_and_single() {
    let mut b = Bodies::with_capacity(1);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 0);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1);
    assert_eq!(b.stats.impacts, 0);
    assert_eq!(b.stats.merges, 0);
}

// ===========================================================================
// p — collision_pass with the `hot` set
// ===========================================================================
/// End-to-end run through the `hot = Some(...)` path. Both bodies are
/// hot, so the branch calls `find_collisions_hot`; the hit-and-run must
/// still fire exactly once and leave two bodies alive.
#[test]
fn p_collision_pass_hot_path() {
    let mut b = pair(1.0, 0.5, 1.5, 0.9);
    shift(&mut b, 10.0, -4.0, 0.5, 0.3);
    let hot = [0usize, 1usize];
    collision_pass(&mut b, DT, Some(&hot));
    assert_eq!(b.stats.impacts, 1, "hot path must resolve the hit-and-run");
    assert_eq!(b.len(), 2);
}

// ===========================================================================
// q — rubble coalesce branch
// ===========================================================================
/// Rubble pairs with `ratio < RUBBLE_COALESCE_VRATIO` must merge rather
/// than bounce. This is the `if ratio < RUBBLE_COALESCE_VRATIO` branch
/// inside `collide`; at `v = 0` the velocity ratio is exactly 0, so the
/// branch is guaranteed to fire. Also verifies that `collide` routes
/// coalesce through `merge` and bumps `stats.coalesces`.
#[test]
fn q_rubble_coalesces_at_zero_vrel() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 0.001, BodyKind::Planet);
    let r = b.contact_r(0);
    b.push(r * 0.5, 0.0, 0.0, 0.0, 0.001, BodyKind::Planet);
    b.origin[0] = BodyOrigin::ImpactDebris;
    b.origin[1] = BodyOrigin::ImpactDebris;
    let (px0, py0) = momentum(&b);
    let l0 = total_angular_momentum(&b);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1, "rubble coalesces at zero vrel");
    assert_eq!(b.stats.coalesces, 1);
    assert_eq!(b.stats.merges, 1); // coalesce routes through merge
    let (px1, py1) = momentum(&b);
    assert!((px1 - px0).abs() < 1e-12);
    assert!((py1 - py0).abs() < 1e-12);
    assert!((total_angular_momentum(&b) - l0).abs() < 1e-12 * (1.0 + l0.abs()));
}

// ===========================================================================
// r — rubble bounce (firing path)
// ===========================================================================
/// The *firing* half of `rubble_bounce`'s guard (the no-op half is `ac_*`).
/// Geometry must be a genuine overlap: separation `< rr` so `d = 0` gives
/// `s = rr`, and the velocity must be *approaching* so `vn < 0`. Since
/// `rubble_bounce` is called directly (bypassing `collide`), we have to
/// arrange the overlap by hand rather than relying on detection.
#[test]
fn r_rubble_bounce_fires_and_conserves() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 0.001, BodyKind::Planet);
    // Separation 0.9·rr ⇒ overlapping; velocity along −x ⇒ approaching.
    // d = 0, so s = rr and vn = −|dvx| < 0 — the impulse branch runs.
    let r0 = b.contact_r(0);
    b.push(r0 * 0.9, 0.0, -1.0, 0.0, 0.001, BodyKind::Planet);
    b.origin[0] = BodyOrigin::ImpactDebris;
    b.origin[1] = BodyOrigin::ImpactDebris;
    let rr = b.contact_r(0) + b.contact_r(1);

    let (px0, py0) = momentum(&b);
    let l0 = total_angular_momentum(&b);
    let t0 = b.time;

    rubble_bounce(&mut b, 0, 1, rr);

    assert_eq!(b.stats.bounces, 1, "bounce must fire");
    let (px1, py1) = momentum(&b);
    assert!((px1 - px0).abs() < 1e-15, "px not conserved");
    assert!((py1 - py0).abs() < 1e-15, "py not conserved");
    let dl = (total_angular_momentum(&b) - l0).abs();
    assert!(
        dl < 1e-12 * (1.0 + l0.abs()),
        "L not conserved: dL = {}",
        dl
    );
    // Approach reverses after the impulse.
    let rvx = b.vx[1] - b.vx[0];
    assert!(rvx > 0.0, "relative velocity must reverse, got {}", rvx);
    // Both bodies receive a fresh immunity window.
    assert!(b.imm[0] > t0, "body 0 not immune");
    assert!(b.imm[1] > t0, "body 1 not immune");
}

// ===========================================================================
// s — slow head-on merge (the merge branch of collide)
// ===========================================================================
/// Head-on at 0.3× escape speed is well below `MERGE_VEL_RATIO`, so the
/// merge branch (`ratio < MERGE_VEL_RATIO · (1 − MERGE_B_WEIGHT·b²)`)
/// must fire — not hit-and-run and not fragmentation. This pins the
/// branch ordering inside `collide`.
#[test]
fn s_headon_slow_impact_merges() {
    // Well below escape speed, dead-on: must merge, not hit-and-run or
    // fragment.
    let mut b = pair(1.0, 0.5, 0.3, 0.0);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1, "slow head-on must merge");
    assert_eq!(b.stats.merges, 1);
    assert_eq!(b.stats.impacts, 0);
}

// ===========================================================================
// t — merge-driven state transitions
// ===========================================================================
/// Two white dwarfs whose combined mass crosses `WD_M_MAX` must collapse
/// to a neutron star (Type Ia-like channel). `merge_impl` also resets
/// mass to the NS floor, which the second assertion pins down.
#[test]
fn t_white_dwarf_collapses_to_neutron_star() {
    let m_each = WD_M_MAX * 0.55; // sum = 1.10 * WD_M_MAX > WD_M_MAX
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, m_each, BodyKind::WhiteDwarf);
    b.push(0.01, 0.0, 0.0, 0.0, m_each, BodyKind::WhiteDwarf);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1);
    assert_eq!(b.kind[0], BodyKind::NeutronStar);
    // Collapse resets mass to the NS floor per merge_impl.
    assert!((b.m[0] - NS_M_MIN * 1.07).abs() < 1e-9);
}

/// Two brown dwarfs whose combined mass crosses `STAR_M_MIN` must ignite
/// as a zero-age main-sequence star (fresh `m_init`, `age_gyr = 0`).
#[test]
fn t_brown_dwarf_ignites_to_star() {
    // Two half-of-(STAR_M_MIN + BD_M_MAX) bodies sum just above STAR_M_MIN,
    // so the merge crosses the H-burning threshold.
    let m_each = (STAR_M_MIN + BD_M_MAX) * 0.5;
    assert!(2.0 * m_each > STAR_M_MIN);
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, m_each, BodyKind::BrownDwarf);
    b.push(0.01, 0.0, 0.0, 0.0, m_each, BodyKind::BrownDwarf);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1);
    assert_eq!(b.kind[0], BodyKind::Star, "must ignite");
    // Ignited bodies become zero-age stars.
    assert_eq!(b.age_gyr[0], 0.0);
    assert!((b.m_init[0] - b.m[0]).abs() < 1e-9);
}

// ===========================================================================
// u — pick_survivor
// ===========================================================================
/// `pick_survivor` has three tie-break levels: kind rank (higher survives),
/// mass within the same rank (heavier survives), and argument order on
/// exact ties (first argument wins). Each is asserted in both orderings
/// so the function cannot pass by accident.
#[test]
fn u_pick_survivor_rank_and_mass() {
    // Different kind ranks: higher-ranked body survives, regardless of
    // argument order.
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Star);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::BlackHole);
    assert_eq!(pick_survivor(&b, 0, 1), (1, 0));
    assert_eq!(pick_survivor(&b, 1, 0), (1, 0));
    assert_eq!(pick_survivor(&b, 0, 2), (2, 0));
    assert_eq!(pick_survivor(&b, 2, 1), (2, 1));

    // Same rank: heavier wins.
    let mut c = Bodies::with_capacity(2);
    c.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    c.push(0.0, 0.0, 0.0, 0.0, 2.0, BodyKind::Planet);
    assert_eq!(pick_survivor(&c, 0, 1), (1, 0));
    assert_eq!(pick_survivor(&c, 1, 0), (1, 0));

    // Tie goes to `i`.
    let mut d = Bodies::with_capacity(2);
    d.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    d.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    assert_eq!(pick_survivor(&d, 0, 1), (0, 1));
    assert_eq!(pick_survivor(&d, 1, 0), (1, 0));
}

// ===========================================================================
// v — combine_origin
// ===========================================================================
/// `combine_origin` preserves the classification when both sides are the
/// same kind (so debris discs and belts survive gentle merging intact)
/// and promotes to `Spawned` when kinds are mixed.
#[test]
fn v_combine_origin() {
    use BodyOrigin::*;
    // Same origin class: preserved.
    assert_eq!(combine_origin(ImpactDebris, ImpactDebris), ImpactDebris);
    // Mixed with a regular body: promoted to regular.
    assert_eq!(combine_origin(ImpactDebris, Spawned), Spawned);
    assert_eq!(combine_origin(Spawned, ImpactDebris), Spawned);
    assert_eq!(combine_origin(Spawned, Spawned), Spawned);
}

// ===========================================================================
// w — interact_frac
// ===========================================================================
/// `interact_frac` has two branches: the polynomial (partial overlap) and
/// the `1.0` clamp (full engulfment). This test pins the endpoints, the
/// boundary between the two branches (`l = 2·rp`), and the monotonicity
/// the physics relies on (a more grazing impact must erode less).
#[test]
fn w_interact_frac_edges_and_monotone() {
    let rt = 1.0;
    let rp = 0.1;
    // Head-on: full engulfment.
    assert!((interact_frac(rt, rp, 0.0) - 1.0).abs() < 1e-12);
    // Grazing: zero overlap.
    assert_eq!(interact_frac(rt, rp, 1.0), 0.0);
    // Boundary case where l = 2·rp → exact 1.0.
    let bb_edge = 1.0 - 2.0 * rp / (rt + rp);
    assert!((interact_frac(rt, rp, bb_edge) - 1.0).abs() < 1e-12);

    // Monotone decreasing in bb.
    let mut prev = 1.0;
    for &bb in &[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
        let cur = interact_frac(rt, rp, bb);
        assert!(cur <= prev + 1e-15, "not monotone at bb={}", bb);
        prev = cur;
    }
}

// ===========================================================================
// x — post-collision immunity
// ===========================================================================
/// Every collision branch must leave both participants under
/// `IMMUNITY_TIME`. Without it the survivor could be re-detected as
/// overlapping (since it isn't moved out of contact by the impulse) and
/// the pair would be resolved again on the next step.
#[test]
fn x_hit_and_run_grants_immunity() {
    let mut b = pair(1.0, 0.5, 1.5, 0.9);
    shift(&mut b, 10.0, -4.0, 0.5, 0.3);
    let t0 = b.time;
    collision_pass(&mut b, DT, None);
    assert_eq!(b.stats.impacts, 1);
    assert!(b.imm[0] > t0, "body 0 must be immune for a while");
    assert!(b.imm[1] > t0, "body 1 must be immune for a while");
}

// ===========================================================================
// y — impact_offset edge cases
// ===========================================================================
/// `impact_offset` is the perpendicular distance from body `i` to the
/// relative-velocity line through body `j`. It has a `v ≤ 0 → 0.0`
/// degenerate guard (a stationary pair is treated as head-on) and is
/// used by every `bb`-based branch in `collide`. A sign or axis error
/// here would silently mis-route every impact.
#[test]
fn y_impact_offset_handles_degenerate_cases() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);

    // Zero relative velocity → defined as 0 (head-on).
    assert_eq!(impact_offset(&b, 0, 1), 0.0);

    // On-axis relative velocity, on-axis offset → 0.
    b.vx[1] = 1.0;
    assert!((impact_offset(&b, 0, 1) - 0.0).abs() < 1e-15);

    // Off-axis offset → the perpendicular distance, exactly.
    b.y[1] = 2.0;
    assert!((impact_offset(&b, 0, 1) - 2.0).abs() < 1e-15);
}

// ===========================================================================
// m (extension) — dead-body skip in find_collisions
// ===========================================================================
/// A body flagged `dead[]` must be excluded from broad-phase output on
/// either side. Dead bodies persist between a `merge` and the following
/// `compact`, so the detection pass has to skip them explicitly.
#[test]
fn m_find_collisions_skips_dead() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let r = b.contact_r(0);
    b.push(r * 0.5, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    assert_eq!(find_collisions(&b, false, DT), vec![(0, 1)]);
    b.dead[0] = true;
    assert!(find_collisions(&b, false, DT).is_empty());
    b.dead[0] = false;
    b.dead[1] = true;
    assert!(find_collisions(&b, false, DT).is_empty());
}
