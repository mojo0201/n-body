//! Tests: conservation invariants (L, p, mass) across merges.

use super::super::*;
use super::shared::*;

// ===========================================================================
// a — merge conservation
// ===========================================================================
#[test]
fn a_merge_conserves_angular_momentum() {
    let mut b = pair(1.0, 0.3, 0.4, 0.6);
    shift(&mut b, 40.0, -25.0, 3.0, 2.0);
    b.spin[0] = 0.01;
    b.spin[1] = -0.004;
    let s0 = snapshot(&b);
    assert!(s0.0.abs() > 1e-3, "test needs a non-zero L");
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1);
    assert_eq!(b.stats.merges, 1);
    assert_conserved(s0, &b);
}

#[test]
fn a_star_merge_conserves_angular_momentum() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(0.5, 0.2, -3.0, 40.0, 1.0, BodyKind::Planet);
    let s0 = snapshot(&b);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1);
    assert_conserved(s0, &b);
}

#[test]
fn a_debris_conserves_everything() {
    let mut b = pair(1.0, 0.107, 1.0, 0.7);
    shift(&mut b, -30.0, 12.0, 1.5, -0.7);
    let s0 = snapshot(&b);
    collision_pass(&mut b, DT, None);
    assert!(b.len() > 3, "expected fragments, got {} bodies", b.len());
    assert_conserved(s0, &b);
}

// Append to src/physics/tests/conservation.rs or a Barnes-Hut test file (group `l_`).

/// A tree built at the start of a block must still give the right force on a
/// close pair after the bodies have moved. Earth moves about 0.27 Gm per step,
/// which is most of the Earth-Moon distance (0.38 Gm).
#[test]
fn l_stale_tree_close_pair_matches_direct() {
    let mut b = Bodies::with_capacity(3);
    let au = au_in_gm();
    let vc = (G * M_SUN_ME / au).sqrt();
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(au, 0.0, 0.0, vc, 1.0, BodyKind::Planet);
    let a_moon = 0.384;
    let vm = (G * 1.0 / a_moon).sqrt();
    let moon = b.push(au + a_moon, 0.0, 0.0, vc + vm, 0.0123, BodyKind::Planet);

    let tree = Tree::build(&b);
    // Drift every body by one full step, as `step_block` does.
    for i in 0..b.len() {
        b.x[i] += b.vx[i] * DT;
        b.y[i] += b.vy[i] * DT;
    }
    let (tx, ty, _) = tree.accel_for(&b, moon);
    let (dx, dy, _) = force_direct_on(&b, moon);
    let err = (tx - dx).hypot(ty - dy) / dx.hypot(dy);
    assert!(err < 1e-9, "stale-tree error {err:e} on the Moon");
}
