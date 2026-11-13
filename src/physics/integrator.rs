//! Leapfrog and Yoshida steps, block time steps and `advance`.

use super::*;

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

#[inline]
pub(super) fn rung_for(lim: f64, dt: f64) -> u8 {
    if lim.is_finite() && lim < dt {
        ((dt / lim).log2().ceil() as i32).clamp(0, MAX_RUNG as i32) as u8
    } else {
        0
    }
}

pub(super) fn assign_rungs(b: &mut Bodies, dt: f64) -> u8 {
    let mut kmax = 0;
    for i in 0..b.len() {
        let k = rung_for(b.dtl[i], dt);
        b.rung[i] = k;
        kmax = kmax.max(k);
    }
    kmax
}

pub(super) fn age_flashes(b: &mut Bodies, dt: f64) {
    for f in &mut b.flashes {
        f.age += dt;
    }
    b.flashes.retain(|f| f.age < f.life);
}

pub(super) fn snapshot_positions(b: &mut Bodies) {
    for i in 0..b.len() {
        b._px[i] = b.x[i];
        b._py[i] = b.y[i];
    }
}

pub(super) fn yoshida_step(b: &mut Bodies, dt: f64) {
    snapshot_positions(b);
    leapfrog_step(b, Y_W1 * dt);
    leapfrog_step(b, Y_W0 * dt);
    leapfrog_step(b, Y_W1 * dt);
    age_flashes(b, dt);
    collision_pass(b, dt, None);
    settle_energy(b);
    b.time += dt;
}

// ---- Fix 3: per-rung index lists. Rebuilt only when the body set
// ---- changes (compact or any collision), then mutated in O(1) per rung
// ---- change via `move_rung`. `hot` (rung > 0) is kept sorted.
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

#[inline]
pub(super) fn move_rung(i: u32, old_r: usize, new_r: usize, by_rung: &mut [Vec<u32>], bucket_pos: &mut [u32]) {
    let pos = bucket_pos[i as usize] as usize;
    let last = by_rung[old_r].pop().unwrap();
    if pos < by_rung[old_r].len() {
        by_rung[old_r][pos] = last;
        bucket_pos[last as usize] = pos as u32;
    }
    bucket_pos[i as usize] = by_rung[new_r].len() as u32;
    by_rung[new_r].push(i);
}

pub(super) fn step_block(b: &mut Bodies, dt: f64, kmax: u8) {
    let ticks = 1u32 << kmax;
    let dtf = dt / ticks as f64;
    let n0 = b.len();
    for i in 0..n0 {
        let h = dt / (1u32 << b.rung[i]) as f64;
        b.vx[i] += 0.5 * b.ax[i] * h;
        b.vy[i] += 0.5 * b.ay[i] * h;
    }
    // Build the Barnes-Hut tree ONCE for the whole block. Bodies move at
    // most dt/tick per tick (≈ 1/15000 sim units), which is < 0.01 % of a
    // tree cell, so the topology is valid for all `ticks` sub-steps. The
    // node COM values drift by an amount far below the BH opening-angle
    // error, so reuse is safe. This replaces 2^kmax tree builds with one.
    //
    // EXCEPTION: `collision_pass` inside the tick loop can call
    // `compact()`, which shrinks `b` and remaps every index. The cached
    // tree still holds the *old* body indices and would panic or silently
    // read the wrong body. Rebuild the tree (and the per-rung lists)
    // whenever the body set changes.
    let mut tree = if b.len() >= BH_THRESHOLD {
        Some(Tree::build(&*b))
    } else {
        None
    };
    let mut tree_len = b.len();

    // Per-rung index lists, rebuilt whenever the body set changes.
    let mut by_rung: Vec<Vec<u32>> = (0..=(kmax as usize)).map(|_| Vec::new()).collect();
    let mut bucket_pos: Vec<u32> = Vec::with_capacity(b.len());
    let mut hot: Vec<usize> = Vec::new();
    rebuild_rung_lists(&*b, kmax, &mut by_rung, &mut bucket_pos, &mut hot);

    let mut active: Vec<u32> = Vec::new();
    for tick in 1..=ticks {
        let n = b.len();
        for i in 0..n {
            b._px[i] = b.x[i];
            b._py[i] = b.y[i];
            b.x[i] += b.vx[i] * dtf;
            b.y[i] += b.vy[i] * dtf;
        }
        // Active set: bodies whose rung r satisfies trailing_zeros(tick)
        // >= kmax - r. Union of the cached buckets for r >= r_min.
        let tz = tick.trailing_zeros() as usize;
        let r_min = (kmax as usize).saturating_sub(tz);
        active.clear();
        for list in by_rung.iter().take(kmax as usize + 1).skip(r_min) {
            active.extend_from_slice(list);
        }
        if b.len() != tree_len {
            tree = if b.len() >= BH_THRESHOLD {
                Some(Tree::build(&*b))
            } else {
                None
            };
            tree_len = b.len();
            rebuild_rung_lists(&*b, kmax, &mut by_rung, &mut bucket_pos, &mut hot);
        }
        eval_forces_with(b, &active, tree.as_ref());
        for &iu in &active {
            let i = iu as usize;
            let r = b.rung[i].min(kmax);
            let h = dt / (1u32 << r) as f64;
            b.vx[i] += 0.5 * b.ax[i] * h;
            b.vy[i] += 0.5 * b.ay[i] * h;
            if tick < ticks {
                let mut k = rung_for(b.dtl[i], dt).min(kmax);
                while k < r && tick % (1u32 << (kmax - k)) != 0 {
                    k += 1;
                }
                let old_r = r as usize;
                let new_r = k as usize;
                if new_r != old_r {
                    move_rung(iu, old_r, new_r, &mut by_rung, &mut bucket_pos);
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
                let h2 = dt / (1u32 << k) as f64;
                b.vx[i] += 0.5 * b.ax[i] * h2;
                b.vy[i] += 0.5 * b.ay[i] * h2;
            }
        }
        age_flashes(b, dtf);
        let stats_before = (
            b.stats.merges,
            b.stats.impacts,
            b.stats.tidal,
            b.stats.coalesces,
            b.stats.bounces,
        );
        if !hot.is_empty() {
            collision_pass(b, dtf, Some(&hot));
            settle_energy(b);
        }
        let stats_after = (
            b.stats.merges,
            b.stats.impacts,
            b.stats.tidal,
            b.stats.coalesces,
            b.stats.bounces,
        );
        if b.len() != tree_len || stats_before != stats_after {
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
}

/// Advance by `dt`: collisions, debris-disk coalescence, tidal break-up and
/// escape culling included.
pub fn advance(b: &mut Bodies, dt: f64) {
    let n = b.len();
    if n == 0 {
        return;
    }
    // Parent relationships change slowly (only when a body moves to a
    // different Hill sphere). Refreshing every 100 steps amortizes the
    // O(n^2) cost to a few ms per second even at n=1000.
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
