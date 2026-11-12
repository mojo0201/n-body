//! Tests: dust field.

use super::super::*;

// ===========================================================================
// dust — statistical dust field
// ===========================================================================
#[test]
fn dust_field_initialized_for_solar() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    assert_eq!(b.dust_fields.len(), 1);
    assert!(
        b.dust_fields[0].extent_gm > 4000.0,
        "extent {}",
        b.dust_fields[0].extent_gm
    );
    assert!(!b.dust_fields[0].grains.is_empty());
}

#[test]
fn dust_field_initialized_for_multi_system() {
    let mut b = Bodies::with_capacity(0);
    let _ = spawn_random_systems(&mut b, 30, 42, 3);
    assert_eq!(b.dust_fields.len(), 3);
}

#[test]
fn dust_field_survives_compact() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let primary = b.dust_fields[0].primary;
    b.dead[primary] = true;
    b.compact();
    assert!(
        b.dust_fields.is_empty(),
        "field with dead primary should be dropped"
    );
}

#[test]
fn dust_field_primary_is_remapped_by_compact() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    // Remove the last body: the field must stay, with its primary index kept.
    let last = b.len() - 1;
    b.dead[last] = true;
    let primary = b.dust_fields[0].primary;
    b.compact();
    assert_eq!(b.dust_fields.len(), 1);
    assert_eq!(b.dust_fields[0].primary, primary);
}

#[test]
fn dust_field_survives_advance() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    for _ in 0..100 {
        advance(&mut b, DT);
    }
    let f = &b.dust_fields[0];
    assert!(f
        .grains
        .iter()
        .all(|g| g.x.is_finite() && g.y.is_finite() && g.mass > 0.0));
}
