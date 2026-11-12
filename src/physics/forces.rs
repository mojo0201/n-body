//! Gravity evaluation: direct sums, tree walks and dispatch.

use super::*;

// ---- Fix 2: fixed chunk size for `force_direct_on_parallel`. The value
// ---- must stay constant so the fp sum order is deterministic across
// ---- thread counts. 2048 pairs × ~5 ns ≈ 10 µs per chunk, roughly the
// ---- cost of a rayon dispatch.
pub(super) const DIRECT_CHUNK: usize = 2048;

// ---- Fix 2: serial `force_direct_on` kept byte-for-byte (same fp order).
// ---- `force_direct_range` is only used by the parallel path.
pub(super) fn force_direct_on(b: &Bodies, i: usize) -> (f64, f64, f64) {
    let n = b.len();
    let (xs, ys, ms) = (&b.x[..n], &b.y[..n], &b.m[..n]);
    let (vxs, vys) = (&b.vx[..n], &b.vy[..n]);
    let (xi, yi, mi) = (xs[i], ys[i], ms[i]);
    let (vxi, vyi) = (vxs[i], vys[i]);
    let (mut sx, mut sy) = (0.0, 0.0);
    let mut s_max = 0.0f64;
    let mut w_max = 0.0f64;
    // Two ranges instead of a `j == i` branch in the hot loop.
    for range in [0..i, (i + 1)..n] {
        for j in range {
            let dx = xs[j] - xi;
            let dy = ys[j] - yi;
            let d2 = dx * dx + dy * dy + SOFT2;
            let d = d2.sqrt();
            let inv = 1.0 / (d2 * d);
            let f = G * ms[j] * inv;
            sx += f * dx;
            sy += f * dy;
            let s = (mi + ms[j]) * inv;
            let dvx = vxs[j] - vxi;
            let dvy = vys[j] - vyi;
            let w = (dvx * dvx + dvy * dvy) * inv * d;
            if s > s_max {
                s_max = s;
            }
            if w > w_max {
                w_max = w;
            }
        }
    }
    (sx, sy, dt_limit(s_max, w_max))
}

/// One contiguous range of the j-loop. Used only by `force_direct_on_parallel`.
#[inline]
pub(super) fn force_direct_range(b: &Bodies, i: usize, range: std::ops::Range<usize>) -> (f64, f64, f64, f64) {
    let (xs, ys, ms) = (&b.x, &b.y, &b.m);
    let (vxs, vys) = (&b.vx, &b.vy);
    let (xi, yi, mi) = (xs[i], ys[i], ms[i]);
    let (vxi, vyi) = (vxs[i], vys[i]);
    let (mut sx, mut sy) = (0.0, 0.0);
    let mut s_max = 0.0f64;
    let mut w_max = 0.0f64;
    for j in range {
        let dx = xs[j] - xi;
        let dy = ys[j] - yi;
        let d2 = dx * dx + dy * dy + SOFT2;
        let d = d2.sqrt();
        let inv = 1.0 / (d2 * d);
        let f = G * ms[j] * inv;
        sx += f * dx;
        sy += f * dy;
        let s = (mi + ms[j]) * inv;
        let dvx = vxs[j] - vxi;
        let dvy = vys[j] - vyi;
        let w = (dvx * dvx + dvy * dvy) * inv * d;
        if s > s_max {
            s_max = s;
        }
        if w > w_max {
            w_max = w;
        }
    }
    (sx, sy, s_max, w_max)
}

/// Parallel variant of `force_direct_on` for the case where the active
/// set is smaller than the thread pool but each body's j-loop is long
/// enough to justify splitting. Fixed chunk boundaries keep the fp sum
/// order identical across thread counts.
pub(super) fn force_direct_on_parallel(b: &Bodies, i: usize) -> (f64, f64, f64) {
    let n = b.len();
    let mut bounds: Vec<(usize, usize)> = Vec::new();
    let mut lo = 0;
    while lo < i {
        let hi = (lo + DIRECT_CHUNK).min(i);
        bounds.push((lo, hi));
        lo = hi;
    }
    lo = i + 1;
    while lo < n {
        let hi = (lo + DIRECT_CHUNK).min(n);
        bounds.push((lo, hi));
        lo = hi;
    }
    let chunks: Vec<(f64, f64, f64, f64)> = bounds
        .par_iter()
        .map(|&(lo, hi)| force_direct_range(b, i, lo..hi))
        .collect();
    let (mut sx, mut sy, mut s_max, mut w_max) = (0.0, 0.0, 0.0f64, 0.0f64);
    for c in chunks {
        sx += c.0;
        sy += c.1;
        s_max = s_max.max(c.2);
        w_max = w_max.max(c.3);
    }
    (sx, sy, dt_limit(s_max, w_max))
}

pub(super) fn compute_accel_direct_sym(b: &mut Bodies) {
    let n = b.len();
    for i in 0..n {
        b.ax[i] = 0.0;
        b.ay[i] = 0.0;
    }
    let mut s_max = vec![0.0f64; n];
    let mut w_max = vec![0.0f64; n];
    for i in 0..n {
        let (xi, yi, mi, vxi, vyi) = (b.x[i], b.y[i], b.m[i], b.vx[i], b.vy[i]);
        let (mut axi, mut ayi) = (0.0, 0.0);
        let (mut si, mut wi) = (s_max[i], w_max[i]);
        for j in (i + 1)..n {
            let dx = b.x[j] - xi;
            let dy = b.y[j] - yi;
            let d2 = dx * dx + dy * dy + SOFT2;
            let d = d2.sqrt();
            let inv = 1.0 / (d2 * d);
            let fi = G * b.m[j] * inv;
            let fj = G * mi * inv;
            axi += fi * dx;
            ayi += fi * dy;
            b.ax[j] -= fj * dx;
            b.ay[j] -= fj * dy;
            let s = (mi + b.m[j]) * inv;
            let dvx = b.vx[j] - vxi;
            let dvy = b.vy[j] - vyi;
            let w = (dvx * dvx + dvy * dvy) * inv * d;
            if s > si {
                si = s;
            }
            if s > s_max[j] {
                s_max[j] = s;
            }
            if w > wi {
                wi = w;
            }
            if w > w_max[j] {
                w_max[j] = w;
            }
        }
        b.ax[i] += axi;
        b.ay[i] += ayi;
        s_max[i] = si;
        w_max[i] = wi;
    }
    for i in 0..n {
        b.dtl[i] = dt_limit(s_max[i], w_max[i]);
    }
}

pub(super) fn eval_forces(b: &mut Bodies, active: &[u32]) {
    eval_forces_with(b, active, None);
}

/// Like `eval_forces`, but accepts a pre-built tree. Callers inside
/// `step_block` build the tree once and pass `Some(&tree)` for every
/// tick; everything else passes `None` and pays a rebuild.
///
// ---- Fix 1 + Fix 2: dispatch order is
// ----   1. tree (n >= BH_THRESHOLD, active >= DIRECT_ACTIVE_MAX)
// ----   2. serial (active * n < PAR_WORK_MIN)
// ----   3. parallel-within-body (few active, n >= 2*DIRECT_CHUNK)
// ----   4. parallel-across-bodies (everything else)
pub(super) fn eval_forces_with(b: &mut Bodies, active: &[u32], tree: Option<&Tree>) {
    let n = b.len();
    if active.is_empty() {
        return;
    }
    let use_tree = n >= BH_THRESHOLD && active.len() >= DIRECT_ACTIVE_MAX;
    let threads = rayon::current_num_threads();
    let out: Vec<(f64, f64, f64)> = if use_tree {
        // Build only if the caller didn't hand us one.
        let owned;
        let tree_ref = if let Some(t) = tree {
            t
        } else {
            owned = Tree::build(b);
            &owned
        };
        let bb: &Bodies = b;
        active
            .par_iter()
            .map(|&i| tree_ref.accel_for(bb, i as usize))
            .collect()
    } else if active.len() * n < PAR_WORK_MIN {
        // Tiny: dispatch would cost more than the work.
        active
            .iter()
            .map(|&i| force_direct_on(b, i as usize))
            .collect()
    } else if active.len() < threads && n >= 2 * DIRECT_CHUNK {
        // Few active bodies, long j-loops: split each body's loop.
        let bb: &Bodies = b;
        active
            .iter()
            .map(|&i| force_direct_on_parallel(bb, i as usize))
            .collect()
    } else {
        // Plenty of active bodies: split over active.
        let bb: &Bodies = b;
        active
            .par_iter()
            .map(|&i| force_direct_on(bb, i as usize))
            .collect()
    };
    for (k, &i) in active.iter().enumerate() {
        let i = i as usize;
        b.ax[i] = out[k].0;
        b.ay[i] = out[k].1;
        b.dtl[i] = out[k].2;
    }
}

/// Recompute forces on every body.
pub fn compute_accel(b: &mut Bodies) {
    let n = b.len();
    if n == 0 {
        return;
    }
    if n < PAR_DIRECT_MIN {
        compute_accel_direct_sym(b);
    } else {
        let all: Vec<u32> = (0..n as u32).collect();
        eval_forces(b, &all);
    }
    let mut mt = 0.0;
    let mut px = 0.0;
    let mut py = 0.0;
    for i in 0..n {
        mt += b.m[i];
        px += b.m[i] * b.ax[i];
        py += b.m[i] * b.ay[i];
    }
    if mt > 0.0 {
        let cx = px / mt;
        let cy = py / mt;
        for i in 0..n {
            b.ax[i] -= cx;
            b.ay[i] -= cy;
        }
    }
}
