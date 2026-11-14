//! Tests: dust field.
//!
//! File path is `src/physics/tests/dust.rs` ⇒ module
//! `crate::physics::tests::dust`. `super` resolves to `tests`, not `physics`,
//! so paths into the physics module are written `crate::physics::…` (or,
//! equivalently, `super::super::…`).
//!
//! Source-side requirements — three `pub(crate)` items in `physics/dust.rs`:
//!   * `SEQUENTIAL_GRAIN_THRESHOLD` — cutoff between the serial
//!     `retain_mut` and parallel `par_iter_mut` integrators.
//!   * `BELT_GRIND_MIN_M` — floor below which a belt body is not ground.
//!   * `grind_belts` — belt-collision routine that the tests drive directly.
//!
//! `MAX_COMETS`, `COMET_DUST_LIFE_TU`, `NONCOMET_DUST_LIFE_TU` and
//! `DUST_COMET_FIELD_CAP` are `pub` in `physics/dust.rs`; `Grain::ttl` is a
//! public field, so every `Grain { .. }` literal here sets it (use
//! `f32::INFINITY` for "never expires").
//!
//! Everything else arrives through `physics/mod.rs`'s `pub use` re-exports,
//! so the internal integrator (`step_grain`, `accel`, `emit_dir`) stays
//! private and the tests exercise only the public surface.

use crate::physics::dust::{grind_belts, BELT_GRIND_MIN_M, SEQUENTIAL_GRAIN_THRESHOLD};
use crate::physics::*;

// ===========================================================================
// Fixtures
// ===========================================================================

/// Sun + one grain at radius `r`.
///
/// A grain with `β < 1` starts on a circular orbit of the *reduced* gravity
/// `(1 − β)·GM`. A grain with `β ≥ 1` starts at rest, because the net force
/// is repulsive and no circular orbit exists.
fn sun_field(size_um: f64, r: f64) -> (Bodies, DustField) {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let beta = beta_of(luminosity_of(&b, 0) / (b.m[0] / M_SUN_ME), size_um);
    let v = if beta < 1.0 {
        (G * M_SUN_ME * (1.0 - beta) / r).sqrt()
    } else {
        0.0
    };
    let f = DustField {
        primary: 0,
        extent_gm: 500.0,
        grains: vec![Grain {
            x: r,
            y: 0.0,
            vx: 0.0,
            vy: v,
            mass: 1.0,
            size_um,
            ttl: f32::INFINITY,
        }],
        version: 0,
    };
    (b, f)
}

/// Effective semi-major axis of a grain with radiation pressure `β`.
///
/// Vis-viva with `GM → (1 − β)·GM`. Reduces to the ordinary Keplerian `a`
/// when `β = 0` and diverges as `β → 1` (parabolic escape). Used as a cheap
/// scalar proxy for orbit size, without re-deriving the full elements.
fn a_eff(g: &Grain, beta: f64) -> f64 {
    let r = g.x.hypot(g.y);
    1.0 / (2.0 / r - (g.vx * g.vx + g.vy * g.vy) / (G * M_SUN_ME * (1.0 - beta)))
}

/// Sun + a single empty field, ready for a `deposit` call.
fn deposit_rig(extent_gm: f64) -> Bodies {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.dust_fields.push(DustField {
        primary: 0,
        extent_gm,
        grains: Vec::new(),
        version: 0,
    });
    b
}

/// Sun + one belt-origin planet at `r` with mass `m_body`.
///
/// On a circular Keplerian orbit about the Sun. `origin = AsteroidBelt` is
/// what `grind_belts` uses to select members.
fn belt_body(r: f64, m_body: f64) -> (Bodies, usize) {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let v = (G * M_SUN_ME / r).sqrt();
    let i = b.push(r, 0.0, 0.0, v, m_body, BodyKind::Planet);
    b.origin[i] = BodyOrigin::AsteroidBelt;
    (b, i)
}

/// Empty field around a Sun-like primary; grains are inserted by hand.
fn empty_field(extent_gm: f64) -> DustField {
    DustField {
        primary: 0,
        extent_gm,
        grains: Vec::new(),
        version: 0,
    }
}

// ===========================================================================
// Field lifecycle: init, evolve gating, primary handling
// ===========================================================================

/// The solar preset must produce exactly one dust field, extending at least
/// out to Neptune (~30 AU ≈ 4500 Gm), populated by belt/rubble grains.
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

/// Three separate systems → three separate fields; `init_fields` must not
/// merge them into a single global field.
#[test]
fn dust_field_initialized_for_multi_system() {
    let mut b = Bodies::with_capacity(0);
    let _ = spawn_random_systems(&mut b, 30, 42, 3);
    assert_eq!(b.dust_fields.len(), 3);
}

/// A field whose primary dies must be dropped. `compact` filters it out
/// rather than leaving a dangling `primary` index that would crash the
/// next `step`.
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

/// When a body *after* the primary dies, the primary's index is unchanged
/// (bodies before it keep their slot). The complementary reindexing case —
/// dead body *before* the primary — is covered by the remap loop in
/// `compact` that this test exercises end-to-end.
#[test]
fn dust_field_primary_is_remapped_by_compact() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let last = b.len() - 1;
    b.dead[last] = true;
    let primary = b.dust_fields[0].primary;
    b.compact();
    assert_eq!(b.dust_fields.len(), 1);
    assert_eq!(b.dust_fields[0].primary, primary);
}

/// A hundred full physics steps must not corrupt grain state: positions
/// stay finite and masses stay positive. Catches NaN/inf leakage from the
/// PR-drag or radiation-pressure terms into the grain population.
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

/// `init_fields(…, None)` uses a single default range covering all bodies.
/// With no explicit ranges, it must reproduce the solar preset's one field.
#[test]
fn init_fields_default_range_covers_all() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    b.dust_fields.clear();
    init_fields(&mut b, None);
    assert_eq!(b.dust_fields.len(), 1);
    assert!(!b.dust_fields[0].grains.is_empty());
}

/// A range with no host kind (planet-only system) must produce no field,
/// rather than panicking or creating one with an invalid primary.
#[test]
fn init_fields_skips_system_without_host() {
    let mut b = Bodies::with_capacity(0);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    init_fields(&mut b, None);
    assert!(b.dust_fields.is_empty());
}

/// `evolve` is a no-op when there are no fields to advance. It must not
/// panic and must not create a field.
#[test]
fn evolve_noop_when_no_fields() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    b.dust_fields.clear();
    b.stats.steps = DUST_STEP_INTERVAL;
    evolve(&mut b, DT);
    assert!(b.dust_fields.is_empty());
}

/// `evolve` is gated on `stats.steps % DUST_STEP_INTERVAL == 0`. Off the
/// interval it must leave every field's `version` untouched — otherwise
/// the renderer would redraw unchanged data.
#[test]
fn evolve_noop_when_step_not_multiple() {
    if DUST_STEP_INTERVAL <= 1 {
        return; // every step is a multiple, no gating to test
    }
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    b.stats.steps = 1;
    let before: Vec<u64> = b.dust_fields.iter().map(|f| f.version).collect();
    evolve(&mut b, DT);
    let after: Vec<u64> = b.dust_fields.iter().map(|f| f.version).collect();
    assert_eq!(before, after, "evolve must be a no-op off the interval");
}

/// A field with an out-of-range primary must early-return from `step`,
/// not index out of bounds. The grain must be untouched.
#[test]
fn step_skips_invalid_primary() {
    let mut b = Bodies::with_capacity(0);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mut f = DustField {
        primary: 99,
        extent_gm: 500.0,
        grains: vec![Grain {
            x: 100.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
            mass: 1.0,
            size_um: 100.0,
            ttl: f32::INFINITY,
        }],
        version: 0,
    };
    f.step(&b, DT);
    assert_eq!(f.grains.len(), 1);
    assert_eq!(f.grains[0].x, 100.0);
}

// ===========================================================================
// Grain integrator: PR drag, blow-out, culling
// ===========================================================================

/// PR drag must strictly remove orbital energy (shrink `a`). A 5 % decay
/// would need ~500 orbits under `PR_DRAG_BOOST = 100`, so this asserts
/// monotone non-increase over a moderate run — enough to catch a sign flip,
/// a disabled drag term, or the wrong `c` in `accel`. All three would
/// leave `a` constant or growing.
#[test]
fn dust_pr_drag_shrinks_orbit() {
    let (b, mut f) = sun_field(50.0, 150.0);
    let beta = beta_of(1.0, 50.0);
    let a0 = a_eff(&f.grains[0], beta);
    for _ in 0..2000 {
        f.step(&b, DT * DUST_STEP_INTERVAL as f64);
    }
    assert_eq!(f.grains.len(), 1);
    let a1 = a_eff(&f.grains[0], beta);
    assert!(
        a1 <= a0 * (1.0 + 1e-9),
        "PR drag must not add orbital a: {:.6} -> {:.6}",
        a0,
        a1
    );
}

/// `β = 2.28 > 1` for `s = 0.1 μm`: the net force is purely radial outward,
/// so a grain released at rest is unbound. It must escape the field entirely
/// (culled at `4·extent_gm = 2000 Gm`).
///
/// Analytic escape time under `a = (β−1)·GM/r²`:
///
///   `t(R) = r₀·√(r₀ / 2(β−1)GM) · [S√(1+S²) + asinh S]`,  `S = √(R/r₀ − 1)`
///
/// which evaluates to ≈ 10.8 tu ≈ 2600 calls to `step` for `r₀ = 150 Gm`,
/// `R = 2000 Gm`. 5000 is a ~2× safety cap.
///
/// The monotone-radial check is a physics invariant: with β > 1 and zero
/// initial velocity, `ṙ > 0` for all t > 0. Any decrease means sign-flipped
/// radiation pressure, a mis-signed PR drag term, or a broken substep.
#[test]
fn small_grains_are_blown_out() {
    let (b, mut f) = sun_field(0.1, 150.0);
    let mut last_r = 150.0;
    for k in 0..5000 {
        f.step(&b, DT * DUST_STEP_INTERVAL as f64);
        if f.grains.is_empty() {
            return;
        }
        let r = f.grains[0].x.hypot(f.grains[0].y);
        assert!(
            r >= last_r * (1.0 - 1e-9),
            "step {k}: radius decreased from {last_r} to {r}"
        );
        last_r = r;
    }
    panic!(
        "grain never escaped: still at r = {last_r} Gm after 5000 steps \
         (expected cull at 2000 Gm)"
    );
}

/// A grain spawned inside the star's contact radius is culled immediately,
/// even with finite velocity — no tunnelling through the surface.
#[test]
fn grain_inside_star_is_removed() {
    let (b, mut f) = sun_field(100.0, 150.0);
    f.grains[0].x = 0.5 * R_SUN_GM;
    f.grains[0].y = 0.0;
    f.grains[0].vy = 0.0;
    f.step(&b, DT);
    assert!(f.grains.is_empty());
}

/// A grain placed beyond the cull radius (`4·extent_gm`) is removed on the
/// first step, before the drift moves it. Pins the cull loop separately
/// from the surface-collision loop.
#[test]
fn grain_beyond_cull_radius_is_removed() {
    let (b, mut f) = sun_field(100.0, 150.0);
    f.grains[0].x = 3000.0;
    f.grains[0].y = 0.0;
    f.grains[0].vy = 0.0;
    f.step(&b, DT);
    assert!(f.grains.is_empty(), "grain past cull radius survived");
}

/// A grain that starts inside a *companion* star's contact radius is culled
/// on the first sub-step. Exercises the perturber-contact loop, not just
/// the primary-collision check.
#[test]
fn grain_near_companion_star_is_removed() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(100.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.dust_fields.push(DustField {
        primary: 0,
        extent_gm: 500.0,
        grains: vec![Grain {
            x: 100.0 + 0.5 * R_SUN_GM,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
            mass: 1.0,
            size_um: 100.0,
            ttl: f32::INFINITY,
        }],
        version: 0,
    });
    // `step(&b)` needs `&b` while mutating the field; take the field out.
    let mut fields = std::mem::take(&mut b.dust_fields);
    fields[0].step(&b, DT);
    b.dust_fields = fields;
    assert!(b.dust_fields[0].grains.is_empty());
}

/// The serial `retain_mut` and parallel `par_iter_mut` paths call the same
/// `step_grain`, so they must produce **bit-identical** trajectories on
/// the same input. This is a stronger claim than approximate agreement,
/// and the one that catches accidental float reordering in `accel` or the
/// sub-step loop.
#[test]
fn sequential_and_parallel_agree() {
    let make = |n: usize, seed: u128| -> (Bodies, DustField) {
        let mut b = Bodies::with_capacity(2);
        b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
        let mut rng = XorShift::new(seed);
        let mut grains = Vec::with_capacity(n);
        for _ in 0..n {
            let r = 150.0 + rng.next_f64() * 10.0;
            let v = (G * M_SUN_ME / r).sqrt();
            grains.push(Grain {
                x: r,
                y: 0.0,
                vx: 0.0,
                vy: v,
                mass: 1.0,
                size_um: 100.0,
                ttl: f32::INFINITY,
            });
        }
        (
            b,
            DustField {
                primary: 0,
                extent_gm: 500.0,
                grains,
                version: 0,
            },
        )
    };
    // 100 grains → serial path; threshold + 100 → parallel path.
    let (b_seq, mut f_seq) = make(100, 42);
    let (b_par, mut f_par) = make(SEQUENTIAL_GRAIN_THRESHOLD + 100, 42);
    assert!(f_seq.grains.len() < SEQUENTIAL_GRAIN_THRESHOLD);
    assert!(f_par.grains.len() >= SEQUENTIAL_GRAIN_THRESHOLD);

    f_seq.step(&b_seq, DT * DUST_STEP_INTERVAL as f64);
    f_par.step(&b_par, DT * DUST_STEP_INTERVAL as f64);

    assert_eq!(f_seq.grains.len(), 100);
    assert!(f_par.grains.len() >= 100);
    // The parallel field's first 100 grains are the serial run's 100 grains.
    for i in 0..100 {
        let a = &f_seq.grains[i];
        let b = &f_par.grains[i];
        assert_eq!(a.x.to_bits(), b.x.to_bits(), "grain {} x differs", i);
        assert_eq!(a.y.to_bits(), b.y.to_bits(), "grain {} y differs", i);
        assert_eq!(a.vx.to_bits(), b.vx.to_bits(), "grain {} vx differs", i);
        assert_eq!(a.vy.to_bits(), b.vy.to_bits(), "grain {} vy differs", i);
    }
}

// ===========================================================================
// Emission and deposit
// ===========================================================================

/// Pin `β(s) = 0.57·(L/M)/(ρ·s)` at three reference sizes and confirm the
/// blow-out boundary `β = 1` sits at `s ≈ 0.228 μm` for a Sun-like star.
#[test]
fn beta_of_known_values() {
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    assert!(close(beta_of(1.0, 0.5), 0.57 / (2.5 * 0.5)));
    assert!(close(beta_of(1.0, 1.0), 0.57 / (2.5 * 1.0)));
    assert!(close(beta_of(1.0, 100.0), 0.57 / (2.5 * 100.0)));
    // Half-luminosity star halves beta.
    assert!(close(beta_of(0.5, 1.0), 0.5 * beta_of(1.0, 1.0)));
    // Blow-out boundary sits near 0.228 μm.
    assert!(beta_of(1.0, 0.2) > 1.0);
    assert!(beta_of(1.0, 0.3) < 1.0);
}

/// `deposit` must preserve total mass across the 40 emission grains and
/// weight each grain by `√s` (Dohnanyi cascade, log-uniform in size). Sizes
/// must stay inside `DUST_SIZE_UM`.
#[test]
fn emitted_mass_is_conserved_and_follows_cascade() {
    let mut b = deposit_rig(500.0);
    deposit(&mut b, 100.0, 0.0, 0.0, 0.0, 2.0, None, 0.0);
    let grains = &b.dust_fields[0].grains;
    assert_eq!(grains.len(), DUST_EMIT_GRAINS);
    let total: f64 = grains.iter().map(|g| g.mass).sum();
    assert!((total - 2.0).abs() < 1e-9, "total {}", total);
    // Cascade: mass / sqrt(size) is constant.
    let k0 = grains[0].mass / grains[0].size_um.sqrt();
    for g in grains {
        assert!((g.mass / g.size_um.sqrt() - k0).abs() < 1e-9 * k0.abs().max(1e-30));
        assert!(g.size_um >= DUST_SIZE_UM.0 && g.size_um <= DUST_SIZE_UM.1);
    }
}

/// Happy path: a deposit inside a field's extent appends `DUST_EMIT_GRAINS`
/// grains.
#[test]
fn deposit_adds_grains_to_matching_field() {
    let mut b = deposit_rig(500.0);
    assert!(b.dust_fields[0].grains.is_empty());
    deposit(&mut b, 100.0, 0.0, 0.0, 0.0, 1.0, None, 0.0);
    assert_eq!(b.dust_fields[0].grains.len(), DUST_EMIT_GRAINS);
}

/// A deposit outside every field's extent emits nothing (200 > extent 100).
#[test]
fn deposit_skips_out_of_extent() {
    let mut b = deposit_rig(100.0);
    deposit(&mut b, 200.0, 0.0, 0.0, 0.0, 1.0, None, 0.0);
    assert!(
        b.dust_fields[0].grains.is_empty(),
        "deposit outside extent must not emit"
    );
}

/// With no fields present, `deposit` is a no-op and must not panic.
#[test]
fn deposit_skips_when_fields_empty() {
    let mut b = Bodies::with_capacity(0);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    deposit(&mut b, 100.0, 0.0, 0.0, 0.0, 1.0, None, 0.0);
    assert!(b.dust_fields.is_empty());
}

/// With `Some(dir)`, initial positions are offset along `dir` by up to 0.05
/// units, producing a leading wedge rather than a flat wall at the source.
/// The velocity cone is tilted by the same direction, but the position
/// jitter is a separate code path and is what this test pins.
#[test]
fn deposit_with_direction_offsets_along_tail() {
    let mut b = deposit_rig(500.0);
    deposit(&mut b, 100.0, 0.0, 0.0, 10.0, 1.0, Some((1.0, 0.0)), 0.0);
    let grains = &b.dust_fields[0].grains;
    assert_eq!(grains.len(), DUST_EMIT_GRAINS);
    let mut xs: Vec<f64> = grains.iter().map(|g| g.x).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(xs[0] >= 100.0 - 1e-12);
    assert!(xs[xs.len() - 1] <= 100.05 + 1e-12);
    // Non-degenerate: some grains must be strictly past the source.
    assert!(
        xs[xs.len() - 1] > 100.0 + 1e-12,
        "no tail spread: all grains at 100.0"
    );
    // All emission is along +x in this configuration.
    for g in grains {
        assert!(g.y.abs() < 1e-12);
    }
}

// ===========================================================================
// Belt grinding
// ===========================================================================

/// Each grinding event moves exactly `BELT_GRIND_CHUNK·m` from a belt body
/// into the dust field, so `m_body + m_dust` is invariant. Also checks that
/// grinding actually fires: a 0.01 M⊕ body must lose >10 % of its mass
/// under `1e4·period` steps.
#[test]
fn belt_grinding_conserves_mass_and_feeds_dust() {
    let (mut b, i) = belt_body(150.0, 0.01);
    let mut f = empty_field(500.0);
    let mut rng = XorShift::new(9);
    let m0 = b.m[i];
    let period = std::f64::consts::TAU * (150.0f64.powi(3) / (G * M_SUN_ME)).sqrt();
    for _ in 0..50 {
        grind_belts(&mut b, &mut f, 1e4 * period, &mut rng);
    }
    let dust: f64 = f.grains.iter().map(|g| g.mass).sum();
    assert!(b.m[i] < 0.9 * m0, "belt body kept {} of {}", b.m[i], m0);
    assert!((m0 - b.m[i] - dust).abs() < 1e-9 * m0, "mass not conserved");
}

/// The grinding rate scales with belt mass (`rate ∝ belt_mass`, clamped to
/// 0.01–100×). A 3 M⊕ belt must lose a strictly larger *fraction* of its
/// mass than a 0.003 M⊕ belt over the same number of steps.
#[test]
fn massive_belts_grind_faster() {
    let run = |m_body: f64| {
        let (mut b, i) = belt_body(150.0, m_body);
        let mut f = empty_field(500.0);
        let mut rng = XorShift::new(4);
        let period = std::f64::consts::TAU * (150.0f64.powi(3) / (G * M_SUN_ME)).sqrt();
        for _ in 0..40 {
            grind_belts(&mut b, &mut f, 100.0 * period, &mut rng);
        }
        (m_body - b.m[i]) / m_body
    };
    assert!(
        run(3.0) > run(0.003),
        "heavy belt must lose a larger fraction"
    );
}

/// Bodies below `BELT_GRIND_MIN_M` are frozen: no mass loss, no dust.
#[test]
fn grinding_skips_tiny_bodies() {
    let tiny = BELT_GRIND_MIN_M * 0.5;
    let (mut b, i) = belt_body(150.0, tiny);
    let mut f = empty_field(500.0);
    let mut rng = XorShift::new(3);
    let period = std::f64::consts::TAU * (150.0f64.powi(3) / (G * M_SUN_ME)).sqrt();
    for _ in 0..100 {
        grind_belts(&mut b, &mut f, 1e4 * period, &mut rng);
    }
    assert_eq!(b.m[i], tiny, "tiny belt body should not be ground");
    assert!(f.grains.is_empty(), "no dust should be emitted");
}

/// When the field is at `DUST_MAX_GRAINS`, `grind_belts` must skip emission
/// entirely — the belt body keeps its mass and no grain is added. Pins the
/// `grains.len() + DUST_EMIT_GRAINS > DUST_MAX_GRAINS` guard.
#[test]
fn grinding_respects_grain_cap() {
    let (mut b, _) = belt_body(150.0, 1.0);
    let mut f = empty_field(500.0);
    f.grains.resize(
        DUST_MAX_GRAINS,
        Grain {
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
            mass: 0.0,
            size_um: 1.0,
            ttl: f32::INFINITY,
        },
    );
    let mut rng = XorShift::new(2);
    let period = std::f64::consts::TAU * (150.0f64.powi(3) / (G * M_SUN_ME)).sqrt();
    grind_belts(&mut b, &mut f, 1e6 * period, &mut rng);
    assert_eq!(
        f.grains.len(),
        DUST_MAX_GRAINS,
        "grind_belts must not push past DUST_MAX_GRAINS"
    );
}

// ===========================================================================
// Grain lifetime, comet dust cap, comet limit
// ===========================================================================

/// The comet cap must sit below the hard cap with room for one emission, or
/// the comet-specific guard would be unreachable.
const _: () = assert!(DUST_COMET_FIELD_CAP + DUST_EMIT_GRAINS <= DUST_MAX_GRAINS);

/// A grain that never leaves the field and carries no mass, for padding a
/// field up to a given population.
fn filler_grain() -> Grain {
    Grain {
        x: 0.0,
        y: 0.0,
        vx: 0.0,
        vy: 0.0,
        mass: 0.0,
        size_um: 1.0,
        ttl: f32::INFINITY,
    }
}

/// Sun + `n` comets on circular orbits, 100 Gm apart radially by 10 Gm each.
/// Comet `k` sits at index `k + 1`, so a higher index means farther out.
fn comet_rig(n: usize) -> Bodies {
    let mut b = Bodies::with_capacity(n + 1);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    for k in 0..n {
        let r = 100.0 + 10.0 * k as f64;
        let v = (G * M_SUN_ME / r).sqrt();
        let i = b.push(r, 0.0, 0.0, v, 1e-6, BodyKind::Planet);
        b.origin[i] = BodyOrigin::Comet;
        b.class[i] = BodyClass::Icy;
    }
    b.refresh_parent_cache();
    b
}

/// A grain is removed once its `ttl` runs out, even on a safe bound orbit.
/// With `ttl = 1.0` and `dt = 0.4` it survives two steps (ttl 0.2 left) and
/// expires on the third.
#[test]
fn grain_with_expired_ttl_is_removed() {
    let (b, mut f) = sun_field(100.0, 150.0);
    f.grains[0].ttl = 1.0;
    f.step(&b, 0.4);
    f.step(&b, 0.4);
    assert_eq!(
        f.grains.len(),
        1,
        "grain should survive 0.8 tu of a 1.0 tu ttl"
    );
    f.step(&b, 0.4);
    assert!(f.grains.is_empty(), "grain should expire once ttl <= 0");
}

/// An infinite `ttl` is never consumed: the grain stays alive and its `ttl`
/// stays infinite, so belt dust is unaffected by the lifetime logic.
#[test]
fn infinite_ttl_never_expires() {
    let (b, mut f) = sun_field(100.0, 150.0);
    assert!(f.grains[0].ttl.is_infinite());
    for _ in 0..100 {
        f.step(&b, 0.05);
    }
    assert_eq!(f.grains.len(), 1);
    assert!(f.grains[0].ttl.is_infinite());
}

/// Comet tracers (`deposit` with a direction) get a finite lifetime drawn from
/// 0.5-1.5 x `COMET_DUST_LIFE_TU`, and the draws differ from grain to grain.
#[test]
fn comet_dust_gets_finite_lifetime() {
    let mut b = deposit_rig(500.0);
    deposit(&mut b, 100.0, 0.0, 0.0, 10.0, 1.0, Some((1.0, 0.0)), 0.0);
    let grains = &b.dust_fields[0].grains;
    assert_eq!(grains.len(), DUST_EMIT_GRAINS);
    let (lo, hi) = (0.5 * COMET_DUST_LIFE_TU, 1.5 * COMET_DUST_LIFE_TU);
    for g in grains {
        let t = g.ttl as f64;
        assert!(
            t.is_finite() && t >= lo - 1e-4 && t <= hi + 1e-4,
            "comet grain ttl {t} outside [{lo}, {hi}]"
        );
    }
    let first = grains[0].ttl;
    assert!(
        grains.iter().any(|g| g.ttl != first),
        "all comet grains drew the same lifetime"
    );
}

/// Collision and belt dust (`deposit` without a direction) take
/// `NONCOMET_DUST_LIFE_TU`, which is infinite by default.
#[test]
fn collision_dust_uses_noncomet_lifetime() {
    let mut b = deposit_rig(500.0);
    deposit(&mut b, 100.0, 0.0, 0.0, 0.0, 1.0, None, 0.0);
    let grains = &b.dust_fields[0].grains;
    assert_eq!(grains.len(), DUST_EMIT_GRAINS);
    for g in grains {
        assert_eq!(g.ttl, NONCOMET_DUST_LIFE_TU as f32);
    }
}

/// Comets may fill a field up to exactly `DUST_COMET_FIELD_CAP` and no
/// further. One emission that lands on the cap is allowed; the next is dropped.
#[test]
fn comet_dust_stops_at_field_cap() {
    let mut b = deposit_rig(500.0);
    b.dust_fields[0]
        .grains
        .resize(DUST_COMET_FIELD_CAP - DUST_EMIT_GRAINS, filler_grain());
    deposit(&mut b, 100.0, 0.0, 0.0, 10.0, 1.0, Some((1.0, 0.0)), 0.0);
    assert_eq!(b.dust_fields[0].grains.len(), DUST_COMET_FIELD_CAP);
    deposit(&mut b, 100.0, 0.0, 0.0, 10.0, 1.0, Some((1.0, 0.0)), 0.0);
    assert_eq!(
        b.dust_fields[0].grains.len(),
        DUST_COMET_FIELD_CAP,
        "comet emission must not push past DUST_COMET_FIELD_CAP"
    );
}

/// The lower comet cap must not throttle collision dust: with the field at
/// the comet cap, an isotropic deposit still goes through.
#[test]
fn collision_dust_ignores_comet_cap() {
    let mut b = deposit_rig(500.0);
    b.dust_fields[0]
        .grains
        .resize(DUST_COMET_FIELD_CAP, filler_grain());
    deposit(&mut b, 100.0, 0.0, 0.0, 0.0, 1.0, None, 0.0);
    assert_eq!(
        b.dust_fields[0].grains.len(),
        DUST_COMET_FIELD_CAP + DUST_EMIT_GRAINS
    );
}
