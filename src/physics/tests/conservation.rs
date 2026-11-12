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
