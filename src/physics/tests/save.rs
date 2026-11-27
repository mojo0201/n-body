//! Save-file validation.
//!
//! The loader used to accept 2 000 000 bodies while the engine supports
//! 256 000 (`MAX_BODIES`), so an oversized file passed validation and then
//! tried to fill a `Bodies` past its limit. `SAVE_MAX_BODIES` is now defined
//! from `MAX_BODIES`, so the two cannot drift apart again, and the boundary is
//! checked with a message that names both numbers.

use super::super::*;

/// The save cap *is* the engine cap — by definition, not by coincidence.
#[test]
fn save_cap_is_the_engine_cap() {
    assert_eq!(SAVE_MAX_BODIES, MAX_BODIES);
}

/// A save holding exactly the cap loads; one body more is refused, and the
/// error says what the file has and what this build supports.
#[test]
fn body_count_boundary_is_enforced() {
    assert!(check_body_count(0).is_ok());
    assert!(check_body_count(1).is_ok());
    assert!(
        check_body_count(MAX_BODIES).is_ok(),
        "a save at exactly the cap must load"
    );

    let err = check_body_count(MAX_BODIES + 1).expect_err("over the cap must be refused");
    assert!(
        err.contains(&(MAX_BODIES + 1).to_string()),
        "error must name the file's count: {err}"
    );
    assert!(
        err.contains(&MAX_BODIES.to_string()),
        "error must name the supported count: {err}"
    );
}
