//! User-edit operations on `Bodies`: scale mass, change kind, delete,
//! circularize orbit.
//!
//! ## Structure
//!
//! Every operation is split into two layers:
//!
//! 1. A `*_raw` helper that mutates the body row and appends a log entry,
//!    but does **not** touch global physics state.
//! 2. A public single-body wrapper that calls the raw helper and then
//!    runs `commit_user_edits` exactly once.
//!
//! Batched entry points (`scale_masses`, `set_kinds`, `delete_bodies`,
//! `circularize_many`) call the raw helpers in a loop and commit once at
//! the end. This matters because `commit_user_edits` invokes
//! `compute_accel` (O(N²)) and `rebase_energy`; the original code paid
//! that cost on *every* individual edit, so a K-body batch did K full
//! physics recomputes.
//!
//! ## Why per-body arithmetic is byte-identical to the original
//!
//! The optimization is purely structural — the same number of `f64`
//! operations in the same order on the same inputs. No algebraic
//! rearrangement (e.g. `1.0 / r` in place of `x / r`) is applied, so
//! results round-trip bit-for-bit and physics cannot drift.

use super::*;

// ---------------------------------------------------------------------------
// Commit
// ---------------------------------------------------------------------------

/// Recompute everything that depends on masses, positions, or velocities
/// after a set of user edits.
///
/// Must be called at most once per logical edit — not per body. Callers
/// that touch several bodies should use the batched API below rather
/// than calling the single-body wrappers in a loop.
#[inline]
pub fn commit_user_edits(b: &mut Bodies) {
    // The parent cache is derived from positions and masses; any edit
    // invalidates it. `compute_accel` will rebuild lazily.
    b.parent_cache_valid = false;
    compute_accel(b);
    // Re-anchor the reference total so user edits don't register as
    // spurious energy drift in diagnostics.
    rebase_energy(b);
}

/// Legacy name kept for call sites that referenced the old entry point.
#[inline]
pub fn user_modified(b: &mut Bodies) {
    commit_user_edits(b);
}

// ---------------------------------------------------------------------------
// Raw per-body mutations (no commit)
// ---------------------------------------------------------------------------

/// Scale body `i`'s mass by `factor`, preserving a star's fractional
/// position through its lifetime. Returns `false` (and makes no change)
/// if the result is non-finite or below the floor.
///
/// The star path re-evaluates `star_lifetime_gyr` at the old and new
/// initial mass and rescales `age_gyr` by the ratio of those lifetimes.
/// This assumes the lifetime law is monotone in mass; it deliberately
/// does not simplify to `factor.powf(alpha)` because the law is not
/// guaranteed to be a pure power law.
fn scale_mass_raw(b: &mut Bodies, i: usize, factor: f64) -> bool {
    let m = b.m[i] * factor;
    if !m.is_finite() || m < 1e-4 {
        return false;
    }
    if b.kind[i] == BodyKind::Star {
        let old_init = b.m_init[i];
        let f = b.age_gyr[i] / star_lifetime_gyr(BodyKind::Star, old_init);
        let new_init = old_init * factor;
        b.m_init[i] = new_init;
        b.age_gyr[i] = f * star_lifetime_gyr(BodyKind::Star, new_init);
    } else {
        b.m_init[i] = m;
    }
    b.set_mass(i, m);
    b.origin[i] = BodyOrigin::Spawned;
    let text = format!("{} mass scaled by {}", tag(b, i), factor);
    b.log(EventKind::User, text, b.x[i], b.y[i]);
    true
}

/// Reclassify body `i` as `kind`, snapping its mass into the new class's
/// valid range if necessary, then resetting density, initial mass, and
/// age to that class's defaults.
///
/// The `match` arms fire only when the current mass is *outside* the
/// target class's range; otherwise the mass is left untouched. This is
/// what makes `set_kind` idempotent for a body already in a valid mass
/// regime for the new kind.
///
/// `class` is reset to `Rocky` unconditionally rather than sampled,
/// because this function has no RNG access — the result must be
/// deterministic for a given `(body, kind)` pair.
fn set_kind_raw(b: &mut Bodies, i: usize, kind: BodyKind) {
    let m = b.m[i];
    let new_m = match kind {
        BodyKind::Star if m < STAR_M_MIN || m > 100.0 * M_SUN_ME => M_SUN_ME,
        BodyKind::BrownDwarf if m < BD_M_MIN || m > BD_M_MAX => m.clamp(BD_M_MIN, BD_M_MAX),
        BodyKind::WhiteDwarf if m < WD_M_MIN || m > WD_M_MAX => m.clamp(WD_M_MIN, WD_M_MAX),
        BodyKind::NeutronStar if m < NS_M_MIN || m > NS_M_MAX => m.clamp(NS_M_MIN, NS_M_MAX),
        BodyKind::BlackHole if m < BH_M_MIN => 10.0 * M_SUN_ME,
        BodyKind::DwarfPlanet if m > DWARF_M_MAX => DWARF_M_MAX,
        BodyKind::Planet if m >= 1000.0 => JUPITER_MASS,
        _ => m,
    };
    b.kind[i] = kind;
    b.origin[i] = BodyOrigin::Spawned;
    b.class[i] = BodyClass::Rocky;
    b.bulk_density[i] = default_bulk_density(kind, new_m);
    b.m_init[i] = new_m;
    b.age_gyr[i] = default_age_gyr(kind);
    b.set_mass(i, new_m);
    let text = format!("{} changed type", tag(b, i));
    b.log(EventKind::User, text, b.x[i], b.y[i]);
}

/// Rewrite body `i`'s velocity relative to its parent to a circular
/// orbit at the current separation, preserving the sense of rotation.
///
/// Returns `false` if the body has no parent, if the two bodies are
/// coincident (separation < 1e-9), or if no parent could be found.
///
/// The rotation sign is taken from the sign of the 2D cross product
/// `dx·dvy − dy·dvx`, so a body orbiting one way keeps that direction.
fn circularize_raw(b: &mut Bodies, i: usize) -> bool {
    let Some(p) = find_parent(b, i) else {
        return false;
    };
    let dx = b.x[i] - b.x[p];
    let dy = b.y[i] - b.y[p];
    let r = (dx * dx + dy * dy).sqrt();
    if r < 1e-9 {
        return false;
    }
    let dvx = b.vx[i] - b.vx[p];
    let dvy = b.vy[i] - b.vy[p];
    let sign = if dx * dvy - dy * dvx >= 0.0 {
        1.0
    } else {
        -1.0
    };
    // Circular speed from the two-body energy relation, using the total
    // mass of the pair.
    let vc = (G * (b.m[p] + b.m[i]) / r).sqrt();
    // Tangential unit vector times `sign`, added to the parent's velocity.
    // Kept as `dy / r * vc` rather than `dy * (vc / r)` so the result is
    // bit-identical to the pre-optimization code.
    b.vx[i] = b.vx[p] - sign * dy / r * vc;
    b.vy[i] = b.vy[p] + sign * dx / r * vc;
    let text = format!("{} circularized", tag(b, i));
    b.log(EventKind::User, text, b.x[i], b.y[i]);
    true
}

// ---------------------------------------------------------------------------
// Single-body public API
//
// Each of these commits exactly once. They are the correct choice for a
// one-off edit; use the batched functions for multi-body operations.
// ---------------------------------------------------------------------------

/// Scale body `i`'s mass by `factor`. No-op if the result is invalid.
pub fn scale_mass(b: &mut Bodies, i: usize, factor: f64) {
    if scale_mass_raw(b, i, factor) {
        commit_user_edits(b);
    }
}

/// Change body `i`'s kind and re-derive its mass/density/age defaults.
pub fn set_kind(b: &mut Bodies, i: usize, kind: BodyKind) {
    set_kind_raw(b, i, kind);
    commit_user_edits(b);
}

/// Mark body `i` as dead and compact it out of the arrays.
///
/// The log entry's `(x, y)` are captured *before* `compact()`, because
/// compaction may reorder the arrays and `i` no longer refers to the
/// deleted body afterwards. The tag is likewise resolved before the
/// move for the same reason.
pub fn delete_body(b: &mut Bodies, i: usize) {
    let text = format!("{} deleted", tag(b, i));
    let (x, y) = (b.x[i], b.y[i]);
    b.dead[i] = true;
    b.compact();
    b.log(EventKind::User, text, x, y);
    commit_user_edits(b);
}

/// Circularize body `i`'s orbit around its parent. Returns `true` on
/// success, `false` if the body has no parent or the orbit degenerates.
pub fn circularize(b: &mut Bodies, i: usize) -> bool {
    if circularize_raw(b, i) {
        commit_user_edits(b);
        true
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// Batched public API
//
// Each function commits once for the whole batch. Prefer these whenever
// the UI produces multiple edits from a single user action (multi-select
// delete, slider drag with N selected bodies, etc.).
// ---------------------------------------------------------------------------

/// Scale several masses in one commit. Invalid `(i, factor)` pairs are
/// silently skipped, matching the single-body behaviour.
pub fn scale_masses(b: &mut Bodies, edits: &[(usize, f64)]) {
    let mut any = false;
    for &(i, f) in edits {
        any |= scale_mass_raw(b, i, f);
    }
    if any {
        commit_user_edits(b);
    }
}

/// Reclassify several bodies in one commit.
pub fn set_kinds(b: &mut Bodies, edits: &[(usize, BodyKind)]) {
    for &(i, k) in edits {
        set_kind_raw(b, i, k);
    }
    if !edits.is_empty() {
        commit_user_edits(b);
    }
}

/// Delete several bodies in one compaction pass.
///
/// `indices` refer to the *pre-call* array layout. Duplicates are
/// collapsed and the remaining indices are processed in a single
/// `compact()` call, giving O(N) total instead of O(N·K) for K
/// sequential `delete_body` calls.
///
/// Note: compact() is a stable shift, so survivors keep their relative order.
pub fn delete_bodies(b: &mut Bodies, mut indices: Vec<usize>) {
    if indices.is_empty() {
        return;
    }
    indices.sort_unstable();
    indices.dedup();

    // Snapshot tags and positions up front; compaction invalidates both.
    let mut logs: Vec<(String, f64, f64)> = Vec::with_capacity(indices.len());
    for &i in &indices {
        logs.push((format!("{} deleted", tag(b, i)), b.x[i], b.y[i]));
    }
    for &i in &indices {
        b.dead[i] = true;
    }
    b.compact();
    for (text, x, y) in logs {
        b.log(EventKind::User, text, x, y);
    }
    commit_user_edits(b);
}

/// Circularize several orbits in one commit.
///
/// Safe to batch: `circularize_raw` only changes velocities, never
/// positions or masses, so the parent cache (which depends only on the
/// latter) is valid across the entire batch.
/// Note: each body still calls the uncached `find_parent`, so the cost is O(N·K).
///
/// Returns the number of bodies successfully circularized.
pub fn circularize_many(b: &mut Bodies, indices: &[usize]) -> usize {
    let mut changed = 0;
    for &i in indices {
        if circularize_raw(b, i) {
            changed += 1;
        }
    }
    if changed > 0 {
        commit_user_edits(b);
    }
    changed
}
