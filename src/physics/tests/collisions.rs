//! Tests: collisions: hit-and-run, debris, grazing, hot path, rubble, fragmentation, silent merge.

use super::super::*;
use super::shared::*;

// ===========================================================================
// b — hit-and-run
// ===========================================================================
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
    let v1 = (b.vx[1] - b.vx[0], b.vy[1] - b.vy[0]);
    let (n0, n1) = (v0.0.hypot(v0.1), v1.0.hypot(v1.1));
    let cos = (v0.0 * v1.0 + v0.1 * v1.1) / (n0 * n1);
    assert!(cos < 0.9999, "path must be deflected");
    assert!(n1 < n0, "bounce is inelastic");
    assert_conserved(s0, &b);
    let mut h = pair(1.0, 0.5, 1.5, 0.1);
    collision_pass(&mut h, DT, None);
    assert_ne!(h.len(), 2, "head-on impact must not be a hit-and-run");
}

// ===========================================================================
// c — moon-forming impacts
// ===========================================================================
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
        assert!(b.len() >= 4);
        assert!(ejected > 0.005 && ejected < 0.1, "ejected {}", ejected);
        assert!(b.m[0] > 1.0, "remnant keeps the target mass");
        assert!(
            bound * 2 >= b.len() - 1,
            "most fragments must be bound and prograde"
        );
    }
}

// ===========================================================================
// d — grazing vs head-on
// ===========================================================================
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
    assert!(
        graze.spin[0].abs() > 5.0 * head.spin[0].abs() + 1e-12,
        "grazing spin must be larger"
    );
    assert!(bg > bh, "grazing impact must put more fragments into orbit");
}

// ===========================================================================
// m — collision detection
// ===========================================================================
/// Two overlapping bodies on a tiny `n` use the brute-force path; verify
/// it detects the pair.
#[test]
fn m_find_collisions_detects_overlap() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let r = b.contact_r(0);
    b.push(r - 0.001, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let pairs = find_collisions(&b, false, DT);
    assert_eq!(pairs, vec![(0, 1)]);
}

/// The hash-grid path (n >= `HASH_THRESHOLD`) must find exactly the same
/// pairs as an exhaustive pairwise scan of the same configuration.
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

/// A body under `IMMUNITY_TIME` must not be reported as colliding.
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
#[test]
fn ab_hot_small_list_matches_full() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let r = b.contact_r(0);
    b.push(r - 0.001, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let hot = vec![0usize, 1usize];
    assert_eq!(find_collisions_hot(&b, &hot, false, DT), vec![(0, 1)]);
}

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
// ac — rubble bounce
// ===========================================================================
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

/// Two brown dwarfs merging to a mass in `[BD_M_MAX, STAR_M_MIN)` must not
/// ignite (they're below the H-burning limit), and must conserve mass.
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
