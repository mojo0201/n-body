//! Helpers shared by the test modules.

use super::super::*;

/// Two overlapping planets, with a given relative speed ratio and impact
/// parameter. Used by every collision-conservation test.
pub(super) fn pair(m_big: f64, m_small: f64, ratio: f64, bb: f64) -> Bodies {
    let mut b = Bodies::with_capacity(8);
    let big = b.push(0.0, 0.0, 0.0, 0.0, m_big, BodyKind::Planet);
    let small = b.push(0.0, 0.0, 0.0, 0.0, m_small, BodyKind::Planet);
    let rr = b.contact_r(big) + b.contact_r(small);
    let v_esc = (2.0 * G * (m_big + m_small) / rr).sqrt();
    let d = bb * rr;
    let s = (rr * rr - d * d).sqrt();
    b.x[small] = -s * 0.999;
    b.y[small] = d * 0.999;
    b.vx[small] = ratio * v_esc;
    b._px[small] = b.x[small];
    b._py[small] = b.y[small];
    b
}

pub(super) fn shift(b: &mut Bodies, x: f64, y: f64, vx: f64, vy: f64) {
    for k in 0..b.len() {
        b.x[k] += x;
        b.y[k] += y;
        b._px[k] = b.x[k];
        b._py[k] = b.y[k];
        b.vx[k] += vx;
        b.vy[k] += vy;
    }
}

pub(super) fn momentum(b: &Bodies) -> (f64, f64) {
    (
        (0..b.len()).map(|k| b.m[k] * b.vx[k]).sum(),
        (0..b.len()).map(|k| b.m[k] * b.vy[k]).sum(),
    )
}

pub(super) fn mass(b: &Bodies) -> f64 {
    b.m.iter().sum()
}

pub(super) fn bound_prograde(b: &Bodies, sense: f64) -> usize {
    (1..b.len())
        .filter(|&k| {
            let (dx, dy) = (b.x[k] - b.x[0], b.y[k] - b.y[0]);
            let (dvx, dvy) = (b.vx[k] - b.vx[0], b.vy[k] - b.vy[0]);
            let e = 0.5 * (dvx * dvx + dvy * dvy) - G * b.m[0] / (dx * dx + dy * dy).sqrt();
            e < 0.0 && sense * (dx * dvy - dy * dvx) > 0.0
        })
        .count()
}

pub(super) fn assert_conserved(b0: (f64, f64, f64, f64), b: &Bodies) {
    let (l0, px0, py0, m0) = b0;
    let l1 = total_angular_momentum(b);
    let (px1, py1) = momentum(b);
    assert!(
        (l1 - l0).abs() < 1e-9 * (1.0 + l0.abs()),
        "L {} -> {}",
        l0,
        l1
    );
    assert!((px1 - px0).abs() < 1e-9 * (1.0 + px0.abs()), "px");
    assert!((py1 - py0).abs() < 1e-9 * (1.0 + py0.abs()), "py");
    assert!((mass(b) - m0).abs() < 1e-12 * m0, "mass");
}

pub(super) fn snapshot(b: &Bodies) -> (f64, f64, f64, f64) {
    let (px, py) = momentum(b);
    (total_angular_momentum(b), px, py, mass(b))
}

/// Osculating semi-major axis of body `i` about a centre
/// `(x, y, vx, vy, mass)`.
pub(super) fn osc_a(b: &Bodies, i: usize, c: (f64, f64, f64, f64, f64)) -> f64 {
    let (dx, dy) = (b.x[i] - c.0, b.y[i] - c.1);
    let (dvx, dvy) = (b.vx[i] - c.2, b.vy[i] - c.3);
    let r = dx.hypot(dy);
    let v2 = dvx * dvx + dvy * dvy;
    1.0 / (2.0 / r - v2 / (G * c.4))
}

pub(super) fn star_pair_centre(b: &Bodies) -> (f64, f64, f64, f64, f64) {
    let m = b.m[0] + b.m[1];
    (
        (b.m[0] * b.x[0] + b.m[1] * b.x[1]) / m,
        (b.m[0] * b.y[0] + b.m[1] * b.y[1]) / m,
        (b.m[0] * b.vx[0] + b.m[1] * b.vx[1]) / m,
        (b.m[0] * b.vy[0] + b.m[1] * b.vy[1]) / m,
        m,
    )
}

/// True iff body `i` is a moon of a planet (its parent is a Planet or
/// DwarfPlanet, not a star or remnant). Use this instead of
/// `find_parent(b, i).is_some()` when you mean "is a moon": the latter
/// is true for *every* primary planet, since planets orbit a star.
pub(super) fn is_moon(b: &Bodies, i: usize) -> bool {
    matches!(
        find_parent(b, i),
        Some(p) if matches!(b.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet)
    )
}

// ===========================================================================
// ak — fuzz smoke tests
// ===========================================================================
//
// These tests intentionally do *not* check physics outcomes — those are
// covered by the specific groups. The point here is:
//
//   1. No NaN/Inf ever reaches the state vectors.
//   2. Mass and radius stay strictly positive per body.
//   3. Mass never *increases* across a run (all operations conserve exactly
//      or lose mass via culling; nothing gains).
//   4. n >= BH_THRESHOLD exercises the block-leapfrog path with a live
//      Barnes-Hut tree, including compact()-triggered tree rebuilds inside
//      step_block — the exact class of bug that used to crash at scale.
//
// `FUZZ_ITERS` env var (opt-in) multiplies the run length on the three
// cheap tests. The tree-rebuild test ignores it because it's already heavy.
pub(super) fn fuzz_iters(default: usize) -> usize {
    std::env::var("FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Assert every state vector field is finite and every physical quantity
/// is in its allowed range.
pub(super) fn fuzz_assert_sane(b: &Bodies, ctx: &str) {
    for i in 0..b.len() {
        assert!(b.x[i].is_finite(), "{}: x[{}] = {}", ctx, i, b.x[i]);
        assert!(b.y[i].is_finite(), "{}: y[{}] = {}", ctx, i, b.y[i]);
        assert!(b.vx[i].is_finite(), "{}: vx[{}] = {}", ctx, i, b.vx[i]);
        assert!(b.vy[i].is_finite(), "{}: vy[{}] = {}", ctx, i, b.vy[i]);
        assert!(b.ax[i].is_finite(), "{}: ax[{}] = {}", ctx, i, b.ax[i]);
        assert!(b.ay[i].is_finite(), "{}: ay[{}] = {}", ctx, i, b.ay[i]);
        assert!(
            b.m[i].is_finite() && b.m[i] > 0.0,
            "{}: m[{}] = {}",
            ctx,
            i,
            b.m[i]
        );
        assert!(
            b.r[i].is_finite() && b.r[i] >= 0.0,
            "{}: r[{}] = {}",
            ctx,
            i,
            b.r[i]
        );
        assert!(
            b.bulk_density[i].is_finite() && b.bulk_density[i] > 0.0,
            "{}: bulk_density[{}] = {}",
            ctx,
            i,
            b.bulk_density[i]
        );
        assert!(
            b.spin[i].is_finite(),
            "{}: spin[{}] = {}",
            ctx,
            i,
            b.spin[i]
        );
        assert!(!b.dead[i], "{}: body {} dead post-step", ctx, i);
    }
    assert!(b.time.is_finite(), "{}: time = {}", ctx, b.time);
    // Flashes are aged and pruned every step; the buffer should never
    // grow without bound even during sustained tidal disruption.
    assert!(
        b.flashes.len() < 10_000,
        "{}: {} live flashes (leak)",
        ctx,
        b.flashes.len()
    );
    // Event log is a ring buffer of EVENT_LOG_MAX; sanity check the wrap.
    assert!(
        b.events.len() <= EVENT_LOG_MAX,
        "{}: event log overflow",
        ctx
    );
}

// ===========================================================================
// al — stellar evolution
// ===========================================================================
pub(super) fn sun_system(planet_au: &[f64]) -> Bodies {
    let mut b = Bodies::with_capacity(8);
    let sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    for &a in planet_au {
        push_orbit(&mut b, sun, a, 0.0, 1.0, 0.0);
    }
    recenter_com(&mut b);
    compute_accel(&mut b);
    rebase_energy(&mut b);
    b.refresh_parent_cache();
    b
}

pub(super) fn run_to_remnant(b: &mut Bodies, max_steps: usize) -> usize {
    for k in 0..max_steps {
        advance(b, DT);
        if !b.kind.contains(&BodyKind::Star) {
            return k;
        }
    }
    panic!("star did not die in {max_steps} steps");
}

// ===========================================================================
// m — multi-system ring layout
// ===========================================================================
pub(super) fn range_com(b: &Bodies, (s, e): (usize, usize)) -> (f64, f64) {
    let m: f64 = (s..e).map(|i| b.m[i]).sum();
    (
        (s..e).map(|i| b.m[i] * b.x[i]).sum::<f64>() / m,
        (s..e).map(|i| b.m[i] * b.y[i]).sum::<f64>() / m,
    )
}

// ===========================================================================
// Spawning realism and belt occurrence
// ===========================================================================
/// A solar-mass, 4.6-Gyr, solar-metallicity host with the main-sequence
/// architecture of `spawn_architecture` (no binary, so the compact-chain
/// statistics are isolated).
pub(super) fn solar_mass_system(seed: u128, n: usize) -> Bodies {
    let mut b = Bodies::with_capacity(n);
    b.rng = XorShift::new(seed ^ 0xA5A5_5A5A);
    b.metallicity = 0.0;
    let mut rng = XorShift::new(seed);
    let i = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.age_gyr[i] = 4.6;
    b.set_mass(i, M_SUN_ME);
    spawn_architecture(&mut b, i, n - 1, &mut rng);
    b.refresh_parent_cache();
    b
}

/// (compact planets, giants) among the spawned planets of `b`. Moons are
/// excluded. Compact: <= 30 Me. Giant: > 50 Me.
pub(super) fn count_compact_and_giants(b: &Bodies) -> (usize, usize) {
    let (mut c, mut g) = (0, 0);
    for i in 1..b.len() {
        if b.origin[i] != BodyOrigin::Spawned || b.kind[i] != BodyKind::Planet || is_moon(b, i) {
            continue;
        }
        if b.m[i] > 50.0 {
            g += 1;
        } else if b.m[i] <= 30.0 {
            c += 1;
        }
    }
    (c, g)
}

/// Semi-major axis (Gm) of body `i` about body 0, from vis-viva.
pub(super) fn sma_about_primary(b: &Bodies, i: usize) -> f64 {
    let (dx, dy) = (b.x[i] - b.x[0], b.y[i] - b.y[0]);
    let (dvx, dvy) = (b.vx[i] - b.vx[0], b.vy[i] - b.vy[0]);
    let r = dx.hypot(dy);
    let mu = G * (b.m[0] + b.m[i]);
    1.0 / (2.0 / r - (dvx * dvx + dvy * dvy) / mu)
}
