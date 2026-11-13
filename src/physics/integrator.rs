//! Leapfrog and Yoshida steps, rung-adaptive block stepping, and the
//! top-level `advance` entry point.
//!
//! Two integrators live here:
//!
//!   * `leapfrog_step` — one kick-drift-kick velocity-Verlet step over every
//!     body, used as the building block of both schemes below.
//!   * `yoshida_step`  — a 3-stage symmetric composition of leapfrog with
//!     weights `Y_W1, Y_W0, Y_W1`, which cancels the leading third-order
//!     error and leaves 4th-order accuracy. Used when the whole population
//!     can share one step size.
//!
//! For larger or stiffer populations `step_block` runs `2^kmax` sub-ticks of
//! the same leapfrog, where each body takes a step of `dt / 2^rung` and the
//! rung may change between sub-ticks when the body's `dtl` (largest safe step)
//! shifts. A single Barnes-Hut tree is built per block and reused across
//! sub-ticks unless a collision changes the body set.

use super::*;

/// One kick-drift-kick leapfrog (velocity-Verlet) step of size `dt`.
///
/// The half-kicks at the ends compose with the half-kick from the next call,
/// so each body effectively uses the average of the accelerations at the
/// start and end of its drift — that is what makes the integrator symplectic
/// and time-reversible. `compute_accel` is the expensive part; the three
/// velocity/position loops are memory-bandwidth bound on their own.
#[inline]
pub(super) fn leapfrog_step(b: &mut Bodies, dt: f64) {
    let n = b.len();
    for i in 0..n {
        b.vx[i] += 0.5 * b.ax[i] * dt;
        b.vy[i] += 0.5 * b.ay[i] * dt;
        b.x[i] += b.vx[i] * dt;
        b.y[i] += b.vy[i] * dt;
    }
    compute_accel(b);
    for i in 0..n {
        b.vx[i] += 0.5 * b.ax[i] * dt;
        b.vy[i] += 0.5 * b.ay[i] * dt;
    }
}

/// Smallest `k in 0..=MAX_RUNG` such that `2^k >= dt / lim`, or `0` if `lim`
/// is not finite or already at least `dt` (meaning the body can take the
/// full block step unhalved).
///
/// Extracted from `log2(x).ceil()` so the result is exact at power-of-two
/// boundaries: `log2(4.0)` can return `2.0000004` on some platforms, which
/// would round up to rung 3 for what is genuinely rung 2. The bit path is
/// also libm-free, which matters because this is called once per body per
/// sub-tick from the rung-promotion loop in `step_block`.
#[inline]
pub(super) fn rung_for(lim: f64, dt: f64) -> u8 {
    if lim.is_finite() && lim < dt {
        let x = dt / lim;
        let bits = x.to_bits();
        let exp = (((bits >> 52) & 0x7FF) as i32) - 1023;
        let mantissa = bits & ((1u64 << 52) - 1);
        // x is an exact power of two iff the mantissa is zero; otherwise
        // the smallest k with 2^k >= x is exp + 1.
        let k = if mantissa == 0 { exp } else { exp + 1 };
        k.clamp(0, MAX_RUNG as i32) as u8
    } else {
        0
    }
}

/// Assign every body a rung from its current `dtl`, and return the maximum
/// rung used — the log2 of the number of sub-ticks `step_block` will take.
///
/// Called once per `advance`, before the block is entered. Within the block,
/// individual bodies may still change rungs if their `dtl` shifts; that path
/// goes through the `move_rung` bookkeeping in `step_block`, not through here.
pub(super) fn assign_rungs(b: &mut Bodies, dt: f64) -> u8 {
    let mut kmax = 0;
    for i in 0..b.len() {
        let k = rung_for(b.dtl[i], dt);
        b.rung[i] = k;
        kmax = kmax.max(k);
    }
    kmax
}

/// Drop flashes whose absolute expiry has passed, then refresh `age` on the
/// survivors against `b.time`.
///
/// Flashes are anchored to wall-clock sim time (`expires_at` is set at birth
/// to `b.time + FLASH_LIFE`), not to an accumulated age. That is what lets
/// the caller invoke this once per block rather than once per sub-tick:
/// nothing in the physics reads flashes, and the post-block survivor set is
/// exactly `{ f : f.expires_at > block_end }`. `age` is re-derived from
/// absolute time so the renderer's `progress = age / life` stays correct.
#[inline]
pub(super) fn age_flashes(b: &mut Bodies) {
    let t = b.time;
    b.flashes.retain(|f| f.expires_at > t);
    let t = b.time;
    for f in &mut b.flashes {
        f.age = t - (f.expires_at - f.life);
    }
}

/// Save each body's current position into `_px / _py`.
///
/// The swept-sphere collision test treats the segment `_p -> x` as the body's
/// motion since the last snapshot, so this must be called before every drift
/// — both here (once per `yoshida_step`, before the three sub-steps) and once
/// per sub-tick inside `step_block`.
pub(super) fn snapshot_positions(b: &mut Bodies) {
    for i in 0..b.len() {
        b._px[i] = b.x[i];
        b._py[i] = b.y[i];
    }
}

/// One fourth-order Yoshida step of size `dt` over the whole population.
///
/// Composes three leapfrog sub-steps with weights `Y_W1, Y_W0, Y_W1` — a
/// symmetric sequence that cancels the base scheme's third-order error and
/// leaves it 4th-order accurate. Used when the population is small and quiet
/// enough that every body can share the same step size (the `kmax == 0` path
/// in `advance`); otherwise the block stepper wins.
///
/// `age_flashes` runs *after* `b.time` has been advanced so its absolute-time
/// retention is exact.
pub(super) fn yoshida_step(b: &mut Bodies, dt: f64) {
    snapshot_positions(b);
    leapfrog_step(b, Y_W1 * dt);
    leapfrog_step(b, Y_W0 * dt);
    leapfrog_step(b, Y_W1 * dt);
    collision_pass(b, dt, None);
    settle_energy(b);
    b.time += dt;
    age_flashes(b);
}

// ---------------------------------------------------------------------------
// Per-rung index lists
// ---------------------------------------------------------------------------

/// Recompute `by_rung`, `bucket_pos`, and `hot` from the current `b.rung`.
///
/// `by_rung[r]` is the index list of bodies on rung `r`; `bucket_pos[i]` is
/// body `i`'s slot inside its bucket, so `move_rung` can swap-remove in O(1);
/// and `hot` is the union of rungs `>= 1`, kept sorted so `collision_pass`
/// gets an ascending index list (which is what `find_collisions_hot`'s fast
/// pair-ordering contract expects).
///
/// Called only when the *set* of bodies changes — a compact, or a collision
/// that spawned or killed bodies. Rung changes within a block go through
/// `move_rung`, which does not rebuild.
pub(super) fn rebuild_rung_lists(
    b: &Bodies,
    kmax: u8,
    by_rung: &mut [Vec<u32>],
    bucket_pos: &mut Vec<u32>,
    hot: &mut Vec<usize>,
) {
    debug_assert_eq!(by_rung.len(), kmax as usize + 1);
    for v in by_rung.iter_mut() {
        v.clear();
    }
    if bucket_pos.len() != b.len() {
        bucket_pos.resize(b.len(), 0);
    }
    for (i, pos) in bucket_pos.iter_mut().enumerate() {
        let r = b.rung[i].min(kmax) as usize;
        *pos = by_rung[r].len() as u32;
        by_rung[r].push(i as u32);
    }
    hot.clear();
    for list in by_rung.iter().take(kmax as usize + 1).skip(1) {
        hot.extend(list.iter().map(|&x| x as usize));
    }
    hot.sort_unstable();
}

/// Move body `i` from `by_rung[old_r]` to `by_rung[new_r]` in O(1).
///
/// Removes by swapping with the last element of the old bucket, updating the
/// swapped element's `bucket_pos` entry, then pushes onto the new bucket. The
/// caller is responsible for keeping the `hot` list in sync when either the
/// old or new rung is 0 (see the rung-change branch in `step_block`).
#[inline]
pub(super) fn move_rung(
    i: u32,
    old_r: usize,
    new_r: usize,
    by_rung: &mut [Vec<u32>],
    bucket_pos: &mut [u32],
) {
    let pos = bucket_pos[i as usize] as usize;
    let last = by_rung[old_r].pop().unwrap();
    if pos < by_rung[old_r].len() {
        by_rung[old_r][pos] = last;
        bucket_pos[last as usize] = pos as u32;
    }
    bucket_pos[i as usize] = by_rung[new_r].len() as u32;
    by_rung[new_r].push(i);
}

// ---------------------------------------------------------------------------
// Block stepper
// ---------------------------------------------------------------------------

/// Advance by `dt` using `2^kmax` sub-ticks of the leapfrog, with per-body
/// step sizes `dt / 2^rung`.
///
/// Sub-tick bookkeeping. Each body is active on sub-tick `tick` iff
/// `trailing_zeros(tick) >= kmax - rung`, which is equivalent to
/// `tick % 2^(kmax - rung) == 0`. Bodies on rung `r = kmax` therefore move
/// every sub-tick, bodies on `r = kmax - 1` every other sub-tick, and so on.
/// This is the standard "power-of-two block time step" scheduling: every
/// body drifts at exactly one sub-tick rate throughout the block, and the
/// active-set union over rungs `>= r_min` is what `active` holds each tick.
///
/// Tree reuse. One Barnes-Hut tree is built at the start of the block and
/// reused across all sub-ticks. Bodies drift at most `dtf` per tick — well
/// under 0.01 % of a typical BH cell — so the topology is valid for the whole
/// block and the node COMs drift by far less than the opening-angle error.
/// The one exception is `compact()`, which remaps every body index; when the
/// body set changes, the tree and rung lists are rebuilt before the next
/// sub-tick.
///
/// Half-kick ordering. Each body's sub-step is: half-kick with its *old* rung
/// size `h`, re-evaluate its `dtl`, half-kick with its *new* rung size `h2`.
/// Using different half-kicks across a rung change is what keeps the
/// symplectic structure across the transition — both halves correspond to the
/// same kick-drift-kick interval, just at different resolutions.
pub(super) fn step_block(b: &mut Bodies, dt: f64, kmax: u8) {
    let ticks = 1u32 << kmax;
    let dtf = dt / ticks as f64;
    // Precomputed sub-tick sizes: rung_dt[r] = dt / 2^r. One division per
    // rung per block, replacing two per active body per tick.
    let rung_dt: Vec<f64> = (0..=(kmax as usize))
        .map(|r| dt / (1u64 << r) as f64)
        .collect();

    // First half-kick of the block, at each body's assigned rung size.
    let n0 = b.len();
    for i in 0..n0 {
        let h = rung_dt[b.rung[i].min(kmax) as usize];
        b.vx[i] += 0.5 * b.ax[i] * h;
        b.vy[i] += 0.5 * b.ay[i] * h;
    }

    // Build the BH tree once for the whole block. See the doc comment above
    // for why reuse is safe and when it gets rebuilt.
    let mut tree = if b.len() >= BH_THRESHOLD {
        Some(Tree::build(&*b))
    } else {
        None
    };
    let mut tree_len = b.len();

    // Per-rung index lists and the sorted `hot` set (rung >= 1). Both are
    // rebuilt whenever the body set changes.
    let mut by_rung: Vec<Vec<u32>> = (0..=(kmax as usize)).map(|_| Vec::new()).collect();
    let mut bucket_pos: Vec<u32> = Vec::with_capacity(b.len());
    let mut hot: Vec<usize> = Vec::new();
    rebuild_rung_lists(&*b, kmax, &mut by_rung, &mut bucket_pos, &mut hot);

    // Sized for the worst case: every body active, plus up to
    // FRAG_COUNT_MAX spawned during this block. Reallocs if a bigger
    // disruption lands, but a quiet block never grows.
    let mut active: Vec<u32> = Vec::with_capacity(b.len() + FRAG_COUNT_MAX);

    for tick in 1..=ticks {
        // Drift everyone by the sub-tick dtf, snapshotting the old position
        // for the swept collision test in the same pass.
        let n = b.len();
        for i in 0..n {
            b._px[i] = b.x[i];
            b._py[i] = b.y[i];
            b.x[i] += b.vx[i] * dtf;
            b.y[i] += b.vy[i] * dtf;
        }

        // Active set for this sub-tick: bodies whose rung r has
        // `trailing_zeros(tick) >= kmax - r`, i.e. the union of the cached
        // buckets for rungs r >= r_min.
        let tz = tick.trailing_zeros() as usize;
        let r_min = (kmax as usize).saturating_sub(tz);
        active.clear();
        for list in by_rung.iter().take(kmax as usize + 1).skip(r_min) {
            active.extend_from_slice(list);
        }

        // A previous sub-tick's `collision_pass` may have compacted `b`,
        // which remaps every index. Rebuild the tree and rung lists so
        // they match the new indexing.
        if b.len() != tree_len {
            tree = if b.len() >= BH_THRESHOLD {
                Some(Tree::build(&*b))
            } else {
                None
            };
            tree_len = b.len();
            rebuild_rung_lists(&*b, kmax, &mut by_rung, &mut bucket_pos, &mut hot);
        }

        // Force evaluation for the active subset only. Cold bodies are
        // handled by the tree's node COMs, which are still valid.
        eval_forces_with(b, &active, tree.as_ref());

        // Second half-kick (with old rung), re-evaluate this body's dtl,
        // promote its rung if it now needs a finer step, then apply the
        // second half-kick with the (possibly different) new rung.
        for &iu in &active {
            let i = iu as usize;
            let r = b.rung[i].min(kmax);
            let h = rung_dt[r as usize];
            b.vx[i] += 0.5 * b.ax[i] * h;
            b.vy[i] += 0.5 * b.ay[i] * h;

            if tick < ticks {
                // Rung change: at most one step finer is taken per sub-tick
                // (so the two half-kicks above and below stay consistent).
                // `k` walks up to `r` until the body can no longer afford
                // to step every `2^(kmax-k)` sub-ticks.
                let mut k = rung_for(b.dtl[i], dt).min(kmax);
                while k < r && tz < (kmax - k) as usize {
                    k += 1;
                }
                let old_r = r as usize;
                let new_r = k as usize;
                if new_r != old_r {
                    move_rung(iu, old_r, new_r, &mut by_rung, &mut bucket_pos);
                    // Keep `hot` (sorted set of rung > 0 bodies) in sync
                    // when the body crosses the rung-0 boundary.
                    match (old_r == 0, new_r == 0) {
                        (true, false) => match hot.binary_search(&i) {
                            Ok(_) => {}
                            Err(pos) => hot.insert(pos, i),
                        },
                        (false, true) => {
                            if let Ok(pos) = hot.binary_search(&i) {
                                hot.remove(pos);
                            }
                        }
                        _ => {}
                    }
                }
                b.rung[i] = k;
                let h2 = rung_dt[k as usize];
                b.vx[i] += 0.5 * b.ax[i] * h2;
                b.vy[i] += 0.5 * b.ay[i] * h2;
            }
        }

        // Collision resolution and energy bookkeeping. Any change to the
        // body set (length or any collision statistic) means the cached tree
        // and rung lists must be rebuilt on the next sub-tick. When `hot`
        // is empty the collision pass is skipped entirely, so the stats
        // cannot change — that is why the snapshot lives inside the guard.
        let mut needs_rebuild = b.len() != tree_len;
        if !hot.is_empty() {
            let stats_before = (
                b.stats.merges,
                b.stats.impacts,
                b.stats.tidal,
                b.stats.coalesces,
                b.stats.bounces,
            );
            collision_pass(b, dtf, Some(&hot));
            settle_energy(b);
            let stats_after = (
                b.stats.merges,
                b.stats.impacts,
                b.stats.tidal,
                b.stats.coalesces,
                b.stats.bounces,
            );
            needs_rebuild = needs_rebuild || stats_before != stats_after;
        }
        if needs_rebuild {
            tree = if b.len() >= BH_THRESHOLD {
                Some(Tree::build(&*b))
            } else {
                None
            };
            tree_len = b.len();
            rebuild_rung_lists(&*b, kmax, &mut by_rung, &mut bucket_pos, &mut hot);
        }

        b.time += dtf;
    }

    // Flashes are cosmetic. One O(F) retain at block end gives the same
    // survivor set as per-sub-tick aging would, because age is linear in
    // time and nothing in the block reads flashes.
    age_flashes(b);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Advance the whole system by `dt`.
///
/// Dispatches to `yoshida_step` (uniform 4th-order, cheap) for small quiet
/// populations, or `step_block` (rung-adaptive leapfrog with a `2^kmax`
/// sub-tick block) otherwise. After integration, runs the once-per-step
/// housekeeping: stellar evolution, dust, debris coalescence, tidal break-up,
/// comet sublimation, escape culling, and the final energy snapshot.
///
/// The parent cache is refreshed at most once every 100 steps: each refresh
/// is O(n²) per body for `find_parent`, but parent relationships only change
/// when a body crosses into a different Hill sphere, so amortising over 100
/// steps is more than fast enough even at n = 1000.
pub fn advance(b: &mut Bodies, dt: f64) {
    let n = b.len();
    if n == 0 {
        return;
    }
    if !b.parent_cache_valid || b.stats.steps.saturating_sub(b.parent_cache_step) >= 100 {
        b.refresh_parent_cache();
    }
    let kmax = assign_rungs(b, dt);
    b.stats.max_rung = kmax;
    if kmax == 0 && n < YOSHIDA_MAX_N {
        yoshida_step(b, dt);
        b.stats.last_nsub = 1;
    } else {
        step_block(b, dt, kmax);
        b.stats.last_nsub = 1u32 << kmax;
    }
    evolve_stars(b, dt);
    dust::evolve(b, dt);
    coalesce_debris(b);
    tidal_breakup(b);
    comet_sublimate(b, dt);
    cull_escaped(b);
    settle_energy(b);
    b.stats.steps += 1;
}
