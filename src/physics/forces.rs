//! Gravity evaluation: direct sums, tree walks, and dispatch between them.
//!
//! Every path in this module is written so that the floating-point
//! summation order depends only on body indices — not on thread count,
//! chunk scheduling, or which dispatch branch was taken. That invariant
//! is the reason `DIRECT_CHUNK` is a constant and the parallel reducer
//! is a plain ordered fold rather than `rayon::reduce`.

use super::*;

/// Fixed chunk size for `force_direct_on_parallel`.
///
/// **Do not tune at runtime.** Every parallel j-loop is partitioned into
/// chunks of exactly `DIRECT_CHUNK` bodies starting from a fixed origin
/// (index 0 for the head, `i + 1` for the tail). Because the boundaries
/// are index-derived rather than thread-derived, the ordered fold over
/// chunk partial sums is bit-identical to the serial sum regardless of
/// how many threads ran. Roughly 10 µs of work per chunk on current
/// hardware, which is about the cost of one rayon dispatch.
pub(super) const DIRECT_CHUNK: usize = 2048;

/// PN corrections run when some body moves faster than this fraction of
/// `c`. 1e-5 (about 3 km/s) is below the speed of every planet, so GR
/// works for ordinary planetary systems.
pub(super) const PN_V_OVER_C_MIN: f64 = 1e-5;

/// The 2.5PN drag runs for a pair when `(v_rel / c)^2` is above this value
/// (v_rel above about 300 km/s). `add_pn_corrections` and `account_gw_loss`
/// both use it. Keep them in sync.
pub(super) const GW_REL_V2_MIN: f64 = 1e-6;

/// Serial direct-sum acceleration on body `i`.
///
/// Computes, over all `j != i`:
///   - `sx`, `sy`: the gravitational acceleration on `i`,
///   - `s_max`:  max over j of `(m_i + m_j) / d^3`   (softening strength),
///   - `w_max`:  max over j of `|Δv|^2 / d^3`       (velocity shear),
///
/// then collapses `(s_max, w_max)` through `dt_limit` to a timestep.
///
/// The j-loop is split into `[0, i)` and `(i, n)` rather than carrying a
/// `j == i` branch through the hot loop. This is the canonical serial
/// path: its fp order defines what the parallel variants must reproduce.
pub(super) fn force_direct_on(b: &Bodies, i: usize) -> (f64, f64, f64, f64) {
    let (mut sx, mut sy, mut sz, mut s_max, mut w_max) = (0.0, 0.0, 0.0, 0.0f64, 0.0f64);
    for (lo, hi) in direct_bounds(i, b.len()) {
        let c = force_direct_range(b, i, lo..hi);
        sx += c.0;
        sy += c.1;
        sz += c.2;
        s_max = s_max.max(c.3);
        w_max = w_max.max(c.4);
    }
    (sx, sy, sz, dt_limit(s_max, w_max))
}

/// Chunk bounds over `[0, n) \ {i}` in ascending order. The serial and
/// parallel paths both use this, so both add partial sums in the same order.
fn direct_bounds(i: usize, n: usize) -> impl Iterator<Item = (usize, usize)> {
    let head = (0..i)
        .step_by(DIRECT_CHUNK)
        .map(move |lo| (lo, (lo + DIRECT_CHUNK).min(i)));
    let tail = ((i + 1)..n)
        .step_by(DIRECT_CHUNK)
        .map(move |lo| (lo, (lo + DIRECT_CHUNK).min(n)));
    head.chain(tail)
}

/// One contiguous slab of the j-loop, for `force_direct_on_parallel`.
///
/// Returns the raw `(sx, sy, s_max, w_max)` rather than a finished dt,
/// so the caller can fold several slabs' partial maxima before applying
/// `dt_limit` once. The arithmetic inside is line-for-line identical to
/// `force_direct_on`'s loop body, which is what makes the chunked result
/// match the serial one.
#[inline]
pub(super) fn force_direct_range(
    b: &Bodies,
    i: usize,
    range: std::ops::Range<usize>,
) -> (f64, f64, f64, f64, f64) {
    let (xi, yi, zi, mi) = (b.x[i], b.y[i], b.z[i], b.m[i]);
    let (vxi, vyi, vzi) = (b.vx[i], b.vy[i], b.vz[i]);
    // Pre-slice every column to the range so the loop has no bounds checks.
    // The summation order is unchanged, so results stay bit-identical.
    let xs = &b.x[range.clone()];
    let ys = &b.y[range.clone()];
    let zs = &b.z[range.clone()];
    let ms = &b.m[range.clone()];
    let vxs = &b.vx[range.clone()];
    let vys = &b.vy[range.clone()];
    let vzs = &b.vz[range];
    let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
    let mut s_max = 0.0f64;
    let mut w_max = 0.0f64;
    for k in 0..xs.len().min(ys.len()).min(zs.len()).min(ms.len()).min(vxs.len()).min(vys.len()).min(vzs.len()) {
        let mj = ms[k];
        let dx = xs[k] - xi;
        let dy = ys[k] - yi;
        let dz = zs[k] - zi;
        let d2 = dx * dx + dy * dy + dz * dz + SOFT2;
        let d = d2.sqrt();
        let inv = 1.0 / (d2 * d);
        let f = G * mj * inv;
        sx += f * dx;
        sy += f * dy;
        sz += f * dz;
        let s = (mi + mj) * inv;
        let dvx = vxs[k] - vxi;
        let dvy = vys[k] - vyi;
        let dvz = vzs[k] - vzi;
        let w = (dvx * dvx + dvy * dvy + dvz * dvz) * inv * d;
        s_max = s_max.max(s);
        w_max = w_max.max(w);
    }
    (sx, sy, sz, s_max, w_max)
}

/// Parallel-within-body variant of `force_direct_on`.
///
/// Used when the active set is smaller than the thread pool but each
/// body's j-loop is long enough that intra-body parallelism pays. The
/// j-range `[0, n) \ {i}` is cut into `DIRECT_CHUNK`-sized slabs from
/// fixed origins, each slab is evaluated by `force_direct_range`, and
/// the partials are folded **in the order the slabs were emitted** —
/// not in completion order. That ordered fold is what keeps the sum
/// bit-identical to the serial path across every thread count.
pub(super) fn force_direct_on_parallel(b: &Bodies, i: usize) -> (f64, f64, f64, f64) {
    let n = b.len();

    let bounds: Vec<(usize, usize)> = direct_bounds(i, n).collect();

    let chunks: Vec<(f64, f64, f64, f64, f64)> = bounds
        .par_iter()
        .map(|&(lo, hi)| force_direct_range(b, i, lo..hi))
        .collect();

    // Fold in emission order. rayon::reduce would give a different
    // reduction tree and a different last-ULP result.
    let (mut sx, mut sy, mut sz, mut s_max, mut w_max) = (0.0, 0.0, 0.0, 0.0f64, 0.0f64);
    for c in chunks {
        sx += c.0;
        sy += c.1;
        sz += c.2;
        s_max = s_max.max(c.3);
        w_max = w_max.max(c.4);
    }
    (sx, sy, sz, dt_limit(s_max, w_max))
}

/// Serial pairwise (symmetric) direct sum, used below `PAR_DIRECT_MIN`.
///
/// Exploits Newton's third law: each pair `(i, j)` with `j > i` is
/// visited once and its impulse is applied to both bodies with
/// opposite signs. That halves the work relative to `force_direct_on`
/// but necessarily changes the fp order (body `j` accumulates its `-fj`
/// contributions in ascending `i` order here, versus ascending `j`
/// order in `force_direct_on`). This path is therefore used *only*
/// when no cross-check against the parallel path is possible — i.e.
/// small, fully serial runs.
///
/// Layout notes:
///   - `Bodies` is destructured so each `Vec` is a separate `&mut`,
///     giving LLVM `noalias` facts it cannot derive from indexed access
///     through a single `&mut Bodies`.
///   - `split_at_mut(i + 1)` proves to LLVM that the head write
///     (`ax[i] += axi`) and the tail updates (`ax[j] -= ...`) do not
///     alias, removing a runtime alias check per iteration.
///   - `s_max[i]` and `w_max[i]` are final at the end of iteration
///     `i`, so `dt_limit` is called in place and the separate `dtl`
///     pass is skipped.
pub(super) fn compute_accel_direct_sym(b: &mut Bodies) {
    compute_accel_direct_sym_impl::<true>(b);
}

/// `LIMITER = false` skips the per-pair step-size limiter (`dtl` is left as it
/// was); the accelerations are bit-identical either way.
pub(super) fn compute_accel_direct_sym_impl<const LIMITER: bool>(b: &mut Bodies) {
    let Bodies {
        x,
        y,
        z,
        m,
        vx,
        vy,
        vz,
        ax,
        ay,
        az,
        dtl,
        ..
    } = b;
    let n = x.len();

    ax[..n].fill(0.0);
    ay[..n].fill(0.0);
    az[..n].fill(0.0);

    // Per-body running maxima; index j is updated by every i < j and
    // finalised once the outer loop reaches j.
    let mut s_max = vec![0.0f64; if LIMITER { n } else { 0 }];
    let mut w_max = vec![0.0f64; if LIMITER { n } else { 0 }];

    let xs = &x[..n];
    let ys = &y[..n];
    let zs = &z[..n];
    let ms = &m[..n];
    let vxs = &vx[..n];
    let vys = &vy[..n];
    let vzs = &vz[..n];
    let axs = &mut ax[..n];
    let ays = &mut ay[..n];
    let azs = &mut az[..n];
    let dtls = &mut dtl[..n];

    for i in 0..n {
        let xi = xs[i];
        let yi = ys[i];
        let zi = zs[i];
        let mi = ms[i];
        let vxi = vxs[i];
        let vyi = vys[i];
        let vzi = vzs[i];
        let mut axi = 0.0;
        let mut ayi = 0.0;
        let mut azi = 0.0;

        // Head/tail split: ax_hi[i] is the accumulator we add to here,
        // ax_rest covers j > i. No fp-order change — j still runs
        // i+1..n in order.
        let (ax_hi, ax_rest) = axs.split_at_mut(i + 1);
        let (ay_hi, ay_rest) = ays.split_at_mut(i + 1);
        let (az_hi, az_rest) = azs.split_at_mut(i + 1);
        let (sm_hi, sm_rest) = s_max.split_at_mut(if LIMITER { i + 1 } else { 0 });
        let (wm_hi, wm_rest) = w_max.split_at_mut(if LIMITER { i + 1 } else { 0 });
        let mut si = if LIMITER { sm_hi[i] } else { 0.0 };
        let mut wi = if LIMITER { wm_hi[i] } else { 0.0 };

        for k in 0..(n - i - 1) {
            let j = k + i + 1;
            let mj = ms[j];
            let dx = xs[j] - xi;
            let dy = ys[j] - yi;
            let dz = zs[j] - zi;
            let d2 = dx * dx + dy * dy + dz * dz + SOFT2;
            let d = d2.sqrt();
            let inv = 1.0 / (d2 * d);

            // Pairwise: i gets +f_i * Δ, j gets -f_j * Δ. Physically
            // these are equal and opposite impulses; numerically they
            // are independent accumulations and hence need the split.
            let fi = G * mj * inv;
            let fj = G * mi * inv;
            axi += fi * dx;
            ayi += fi * dy;
            azi += fi * dz;
            ax_rest[k] -= fj * dx;
            ay_rest[k] -= fj * dy;
            az_rest[k] -= fj * dz;

            if LIMITER {
                let s = (mi + mj) * inv;
                let dvx = vxs[j] - vxi;
                let dvy = vys[j] - vyi;
                let dvz = vzs[j] - vzi;
                let w = (dvx * dvx + dvy * dvy + dvz * dvz) * inv * d;

                // `f64::max` == `if a > b` for non-NaN operands; d2 >= SOFT2
                // guarantees both s and w here are finite, so the branchless
                // form is safe and avoids a cmov / branch per element.
                si = si.max(s);
                sm_rest[k] = sm_rest[k].max(s);
                wi = wi.max(w);
                wm_rest[k] = wm_rest[k].max(w);
            }
        }

        ax_hi[i] += axi;
        ay_hi[i] += ayi;
        az_hi[i] += azi;
        if LIMITER {
            sm_hi[i] = si;
            wm_hi[i] = wi;
        }
        // s_max[i], w_max[i] are final for body i: nothing will ever
        // update them again (all j > i pairs have already been folded
        // into sm_rest/wm_rest above). Emit dtl[i] now and skip the
        // post-loop pass.
        if LIMITER {
            dtls[i] = dt_limit(si, wi);
        }
    }
}

/// Recompute forces on the given active set, building a tree if needed.
pub(super) fn eval_forces(b: &mut Bodies, active: &[u32]) {
    eval_forces_with(b, active, None);
}

/// Core force evaluator. Chooses between tree and direct paths and, for
/// direct, between serial / intra-body parallel / across-active
/// parallel based on problem shape.
///
/// `tree` may be supplied by callers (e.g. `step_block`) that hold a
/// built tree across multiple ticks; if `None`, a tree is built only
/// when the dispatch actually selects the tree path.
///
/// Dispatch priority (first match wins):
///   1. **tree**      n >= BH_THRESHOLD && active >= DIRECT_ACTIVE_MAX
///                    — both dimensions large; O(active · log n) beats
///                      O(active · n).
///   2. **serial**    active · n < PAR_WORK_MIN
///                    — the whole job costs less than a rayon dispatch;
///                      parallel overhead would dominate.
///   3. **intra-body parallel**  active < threads && n >= 2·DIRECT_CHUNK
///                    — too few active bodies to split across, but each
///                      j-loop is long enough to justify chunking.
///   4. **across-active parallel**  (everything else)
///                    — plenty of active bodies; split the outer loop.
pub(super) fn eval_forces_with(b: &mut Bodies, active: &[u32], tree: Option<&Tree>) {
    let n = b.len();
    if active.is_empty() {
        return;
    }
    let use_tree = n >= BH_THRESHOLD && active.len() >= DIRECT_ACTIVE_MAX;
    let threads = rayon::current_num_threads();
    let out: Vec<(f64, f64, f64, f64)> = if use_tree {
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

    // Scatter results back to the bodies they belong to.
    for (k, &i) in active.iter().enumerate() {
        let i = i as usize;
        b.ax[i] = out[k].0;
        b.ay[i] = out[k].1;
        b.az[i] = out[k].2;
        b.dtl[i] = out[k].3;
    }
}

/// Recompute forces on every body and remove net COM acceleration.
///
/// After evaluating accelerations, the mass-weighted mean acceleration
/// is subtracted from every body so the system's center of mass is
/// (numerically) inertial. This suppresses the slow drift that
/// accumulated roundoff would otherwise introduce into free-floating
/// configurations. Only the linear term is corrected; angular and
/// tidal components are left in the field.
pub fn compute_accel(b: &mut Bodies) {
    compute_accel_opt(b, true);
}

/// `compute_accel`, optionally without refreshing the step-size limiter `dtl`
/// (small systems only; larger ones always refresh it). Accelerations are
/// bit-identical either way.
pub(super) fn compute_accel_opt(b: &mut Bodies, limiter: bool) {
    let n = b.len();
    if n == 0 {
        return;
    }
    if n < PAR_DIRECT_MIN {
        if limiter {
            compute_accel_direct_sym_impl::<true>(b);
        } else {
            compute_accel_direct_sym_impl::<false>(b);
        }
    } else {
        let all: Vec<u32> = (0..n as u32).collect();
        eval_forces(b, &all);
    }

    // Post-Newtonian corrections. 1PN is the leading GR correction
    // (perihelion precession); 2.5PN is the leading gravitational-wave
    // radiation reaction (orbital decay). Only run when at least one
    // body is moving at v/c > 1e-5 (about 3 km/s). Only pairs that
    // include a host body are corrected, so the cost is O(hosts · n);
    // the sorted-x slab cache inside `add_pn_corrections` keeps both the
    // sort and the pair scans off the per-tick hot path.
    if n >= 2 && needs_pn(b) {
        add_pn_corrections(b, None);
    }

    // Mass-weighted mean acceleration → subtract to zero the COM drift.
    let Bodies { m, ax, ay, az, .. } = b;
    let mut mt = 0.0;
    let mut px = 0.0;
    let mut py = 0.0;
    let mut pz = 0.0;
    for i in 0..n {
        let mi = m[i];
        mt += mi;
        px += mi * ax[i];
        py += mi * ay[i];
        pz += mi * az[i];
    }
    if mt > 0.0 {
        let cx = px / mt;
        let cy = py / mt;
        let cz = pz / mt;
        for i in 0..n {
            ax[i] -= cx;
            ay[i] -= cy;
            az[i] -= cz;
        }
    }
}

/// True if any live body has `v/c > PN_V_OVER_C_MIN` (1e-5, about 3 km/s).
/// Below this speed the post-Newtonian terms are smaller than the
/// integrator error. O(n).
pub(super) fn needs_pn(b: &Bodies) -> bool {
    let c = c_sim();
    let thresh = PN_V_OVER_C_MIN * PN_V_OVER_C_MIN * c * c;
    for i in 0..b.len() {
        if b.dead[i] {
            continue;
        }
        let v2 = b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i] + b.vz[i] * b.vz[i];
        if v2 > thresh {
            return true;
        }
    }
    false
}

/// Total mass-weighted momentum `(px, py, pz, M)` of the live bodies.
pub(super) fn total_momentum(b: &Bodies) -> (f64, f64, f64, f64) {
    let (mut px, mut py, mut pz, mut mt) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..b.len() {
        if b.dead[i] {
            continue;
        }
        px += b.m[i] * b.vx[i];
        py += b.m[i] * b.vy[i];
        pz += b.m[i] * b.vz[i];
        mt += b.m[i];
    }
    (px, py, pz, mt)
}

/// Remove the momentum that the numerical scheme added since `p0` was
/// measured. Every body gets the same velocity shift, so the centre of
/// mass stays inertial. The block stepper uses this at the end of a
/// block, in place of the acceleration fix in `compute_accel`.
pub(super) fn remove_momentum_drift(b: &mut Bodies, p0: (f64, f64, f64)) {
    let (px, py, pz, mt) = total_momentum(b);
    if mt <= 0.0 {
        return;
    }
    let dvx = (px - p0.0) / mt;
    let dvy = (py - p0.1) / mt;
    let dvz = (pz - p0.2) / mt;
    for i in 0..b.len() {
        b.vx[i] -= dvx;
        b.vy[i] -= dvy;
        b.vz[i] -= dvz;
    }
}

/// Add the energy that the 2.5PN drag removed during a step of length `dt`
/// to `stats.energy_offset`, so `energy_drift_pct` does not show it as
/// integrator error. For one pair the power is `K · |v_rel|²` with
/// `K = (32/5) G³ m_i² m_j² / (c⁵ r⁴)`. Call it once per step, after the
/// integration and before `settle_energy`.
pub(super) fn account_gw_loss(b: &mut Bodies, dt: f64) {
    let n = b.len();
    if n < 2 || !needs_pn(b) {
        return;
    }
    let c = c_sim();
    let inv_c2 = 1.0 / (c * c);
    let inv_c5 = inv_c2 * inv_c2 / c;
    let mut power = 0.0;
    for i in 0..n {
        if b.dead[i] || !is_host_kind(b.kind[i]) {
            continue;
        }
        for j in 0..n {
            if j == i || b.dead[j] || (is_host_kind(b.kind[j]) && j < i) {
                continue;
            }
            let rvx = b.vx[i] - b.vx[j];
            let rvy = b.vy[i] - b.vy[j];
            let rvz = b.vz[i] - b.vz[j];
            let rel_v_sq = rvx * rvx + rvy * rvy + rvz * rvz;
            if rel_v_sq * inv_c2 <= GW_REL_V2_MIN {
                continue;
            }
            let dx = b.x[i] - b.x[j];
            let dy = b.y[i] - b.y[j];
            let dz = b.z[i] - b.z[j];
            let r2 = dx * dx + dy * dy + dz * dz + SOFT2;
            let mi = b.m[i];
            let mj = b.m[j];
            let k = (32.0 / 5.0) * G * G * G * mi * mi * mj * mj * inv_c5 / (r2 * r2);
            power += k * rel_v_sq;
        }
    }
    b.stats.energy_offset -= power * dt;
}

/// Add 1PN (Einstein–Infeld–Hoffmann) and 2.5PN (Peters-drag) corrections
/// to `b.ax` / `b.ay`. Called from `compute_accel` after the Newtonian
/// pass, and only when `needs_pn` is true.
///
/// The host–body pair walk uses a sorted-x slab cache on `Bodies` (`pn_order`
/// / `pn_x0`) that is rebuilt only when bodies have drifted appreciably (see
/// `PN_SORT_DELTA_FRAC`), so quiet ticks avoid the O(n log n) sort entirely;
/// buffers are all reused, so the function allocates nothing per call.
///
/// **1PN** produces the same perihelion precession as the full EIH
/// equations. Verified against the Mercury limit: for a test particle on
/// a circular orbit (m_i ≪ m_j, v_j ≈ 0), the bracket reduces to
/// `(v² − 4Gm_j/r) = −3Gm_j/r`, giving the standard
/// `−3 (Gm_j)² / (c² r³)` correction. Signs and mass terms follow
/// Will (1993) §6.2.
///
/// **2.5PN** uses the *Peters circular approximation*: a drag on the
/// relative velocity `v_i − v_j` with coefficient
/// `−(32/5) G³ m_i m_j² / (c⁵ r⁴)`. For circular binaries this exactly
/// reproduces the Peters (1964) energy-loss rate
/// `−(32/5) G⁴ m₁² m₂² (m₁+m₂) / (c⁵ r⁵)`. For eccentric orbits it
/// captures the enhanced periapsis loss (the `r⁴` denominator
/// dominates), and the drag is momentum-conserving per pair:
/// `m_i a_i + m_j a_j = 0`. It is not the full 2.5PN EIH form — the
/// radial-velocity terms that distinguish prograde from retrograde are
/// omitted — but for a top-down visual sim the difference is below
/// renderable resolution.
/// Pairs with G(m_i+m_j)/(c² r) below this get no post-Newtonian terms. About
/// 100 AU from a Sun-mass star; Mercury's coupling is ~2.5e-8, Neptune's ~3e-10.
const PN_COUPLING_MIN: f64 = 1.0e-10;

/// Post-Newtonian terms for the pair (host `i`, body `j`), applied to whichever
/// of the two is in `on_i` / `on_j`. Far pairs, and pairs whose relative speed is
/// far below `c`, add nothing.
fn pn_pair(b: &mut Bodies, i: usize, j: usize, on_i: bool, on_j: bool, inv_c2: f64, inv_c5: f64) {
    let (mi, xi, yi, zi) = (b.m[i], b.x[i], b.y[i], b.z[i]);
    let (vxi, vyi, vzi) = (b.vx[i], b.vy[i], b.vz[i]);
    let vi_sq = vxi * vxi + vyi * vyi + vzi * vzi;
    let mj = b.m[j];
    let dx = xi - b.x[j];
    let dy = yi - b.y[j];
    let dz = zi - b.z[j];
    let r2 = dx * dx + dy * dy + dz * dz + SOFT2;
    // Far pairs: the PN terms scale as G(mi+mj)/(c² r) relative to the
    // Newtonian force, so beyond a cutoff they sit below integrator
    // error. This also drops every cross-system pair in a cluster.
    let cut = G * (mi + mj) * inv_c2 * (1.0 / PN_COUPLING_MIN);
    if r2 > cut * cut {
        return;
    }
    let r = r2.sqrt();
    let r_inv = 1.0 / r;
    let nxi = dx * r_inv;
    let nyi = dy * r_inv;
    let nzi = dz * r_inv;

    let vxj = b.vx[j];
    let vyj = b.vy[j];
    let vzj = b.vz[j];
    let vj_sq = vxj * vxj + vyj * vyj + vzj * vzj;
    let v_dot = vxi * vxj + vyi * vyj + vzi * vzj;
    let n_dot_vi = nxi * vxi + nyi * vyi + nzi * vzi;
    let n_dot_vj = nxi * vxj + nyi * vyj + nzi * vzj;

    // ---- 1PN (EIH), evaluated separately for each body ----
    // n points from j to i. For body j the roles swap and n -> -n.
    let coeff_i = G * mj * inv_c2 * r_inv * r_inv;
    let bracket_i = -(vi_sq + 2.0 * vj_sq - 4.0 * v_dot - 1.5 * n_dot_vj * n_dot_vj)
        + 5.0 * G * mi * r_inv
        + 4.0 * G * mj * r_inv;
    let vcoef_i = 4.0 * n_dot_vi - 3.0 * n_dot_vj;
    if on_i {
        b.ax[i] += coeff_i * (bracket_i * nxi + vcoef_i * (vxi - vxj));
        b.ay[i] += coeff_i * (bracket_i * nyi + vcoef_i * (vyi - vyj));
        b.az[i] += coeff_i * (bracket_i * nzi + vcoef_i * (vzi - vzj));
    }

    let coeff_j = G * mi * inv_c2 * r_inv * r_inv;
    let bracket_j = -(vj_sq + 2.0 * vi_sq - 4.0 * v_dot - 1.5 * n_dot_vi * n_dot_vi)
        + 5.0 * G * mj * r_inv
        + 4.0 * G * mi * r_inv;
    let vcoef_j = -4.0 * n_dot_vj + 3.0 * n_dot_vi;
    if on_j {
        b.ax[j] += coeff_j * (bracket_j * (-nxi) + vcoef_j * (vxj - vxi));
        b.ay[j] += coeff_j * (bracket_j * (-nyi) + vcoef_j * (vyj - vyi));
        b.az[j] += coeff_j * (bracket_j * (-nzi) + vcoef_j * (vzj - vzi));
    }

    // ---- 2.5PN (Peters drag) ----
    // Skips the arithmetic for pairs where v_rel << c. The
    // threshold is (v_rel/c)² > GW_REL_V2_MIN (1e-6), that is
    // v_rel above about 300 km/s. Below that the drag is
    // far smaller than the integrator error.
    let rvx = vxi - vxj;
    let rvy = vyi - vyj;
    let rvz = vzi - vzj;
    let rel_v_sq = rvx * rvx + rvy * rvy + rvz * rvz;
    if rel_v_sq * inv_c2 > GW_REL_V2_MIN {
        let r4 = r2 * r2;
        let coeff_i = -(32.0 / 5.0) * G * G * G * mi * mj * mj * inv_c5 / r4;
        let coeff_j = (32.0 / 5.0) * G * G * G * mi * mi * mj * inv_c5 / r4;
        if on_i {
            b.ax[i] += coeff_i * rvx;
            b.ay[i] += coeff_i * rvy;
            b.az[i] += coeff_i * rvz;
        }
        if on_j {
            b.ax[j] += coeff_j * rvx;
            b.ay[j] += coeff_j * rvy;
            b.az[j] += coeff_j * rvz;
        }
    }
}

/// Fraction of the tightest host's PN coupling distance at which the sorted
/// x-order cache in `add_pn_corrections` is rebuilt. Between rebuilds the
/// slab window is widened by the measured drift, so correctness never depends
/// on this value — it only trades re-sort cost against window width.
const PN_SORT_DELTA_FRAC: f64 = 0.25;

/// (Re)build the x-order cache for `add_pn_corrections`: `order` receives the
/// live body indices sorted by (x, index) at the *current* positions, and
/// `x0` receives a per-body copy of those x values (`x0[k]` is body `k`'s x
/// at this build). The index tie-break makes the order a total order, so the
/// cache (and the slabs the binary searches below produce) is independent of
/// when the last build ran. Bodies that die afterwards stay in the order
/// until the next build; the slab walk skips them.
fn pn_rebuild_order(b: &Bodies, order: &mut Vec<u32>, x0: &mut Vec<f64>, n: usize) {
    order.clear();
    for k in 0..n {
        if !b.dead[k] {
            order.push(k as u32);
        }
    }
    let x = &b.x;
    order.sort_unstable_by(|&p, &q| {
        x[p as usize]
            .total_cmp(&x[q as usize])
            .then_with(|| p.cmp(&q))
    });
    x0.clear();
    x0.resize(n, 0.0);
    for &k in order.iter() {
        x0[k as usize] = x[k as usize];
    }
}

/// First index `m` in `order` such that `x0[order[m]] >= target`. `order`
/// must be sorted by these keys (see `pn_rebuild_order`). Analogous to
/// `slice::partition_point` applied through the permutation.
fn pn_partition_lo(order: &[u32], x0: &[f64], target: f64) -> usize {
    let mut lo = 0;
    let mut hi = order.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if x0[order[mid] as usize] < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// First index `m` in `order` such that `x0[order[m]] > target`.
fn pn_partition_hi(order: &[u32], x0: &[f64], target: f64) -> usize {
    let mut lo = 0;
    let mut hi = order.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if x0[order[mid] as usize] <= target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

pub(super) fn add_pn_corrections(b: &mut Bodies, active: Option<&[u32]>) {
    let n = b.len();
    if n < 2 {
        return;
    }
    let c = c_sim();
    let inv_c2 = 1.0 / (c * c);
    let inv_c5 = inv_c2 * inv_c2 / c;

    // When called from the block stepper, only update active bodies so
    // inactive bodies don't get PN added twice. `mask` is a reused scratch
    // buffer (`Vec<bool>` is a bit-vector), so a call allocates nothing.
    let uses_mask = active.is_some();
    let mut mask = std::mem::take(&mut b.pn_mask);
    if let Some(act) = active {
        mask.clear();
        mask.resize(n, false);
        for &k in act {
            mask[k as usize] = true;
        }
    }
    let on = |k: usize| !uses_mask || mask[k];

    // Scratch out of the way so `pn_pair` below can borrow `b` mutably; the
    // buffers themselves (and their capacity) are kept across calls.
    let mut order = std::mem::take(&mut b.pn_order);
    let mut x0 = std::mem::take(&mut b.pn_x0);
    let mut hosts = std::mem::take(&mut b.pn_hosts);
    // The x-order cache holds when the body set has not changed (`len` is
    // bumped by every push / compaction) and a build happened at some point.
    let cached = b.pn_valid && b.pn_len == n;

    // Every pair that gets PN terms contains a host body. The outer loop
    // visits hosts only. The `j` loop below skips a host `j` with `j < i`,
    // so a host-host pair is done once, and a host-planet pair is done
    // once. Planet-planet pairs get no PN terms (they are negligible).
    //
    // Host list, `m_max` and — when the cache is usable — the largest
    // displacement since the last build are collected in one pass.
    hosts.clear();
    let mut m_max = 0.0f64;
    let mut m_host_min = f64::INFINITY;
    let mut delta_max = 0.0f64;
    for k in 0..n {
        if b.dead[k] {
            continue;
        }
        if is_host_kind(b.kind[k]) {
            hosts.push(k);
            let mk = b.m[k];
            if mk < m_host_min {
                m_host_min = mk;
            }
        }
        let mk = b.m[k];
        if mk > m_max {
            m_max = mk;
        }
        if cached {
            let d = (b.x[k] - x0[k]).abs();
            if d > delta_max {
                delta_max = d;
            }
        }
    }
    // A host that is not active this tick only needs its pairs with active
    // bodies; those are the common case in a block step (a few fast bodies).
    // Each body still receives its terms from the hosts in the same order, so
    // the result is identical to visiting every pair.
    let any_active_host = hosts.iter().any(|&h| on(h));

    // Bodies sorted by (x, index), with x frozen at the build, so each active
    // host only scans the slab within its coupling cutoff instead of the whole
    // population (matters with many systems, where almost every pair is far
    // beyond the cutoff). The sort is the O(n log n) part and is *not* run
    // every call: it is rebuilt when missing, or when the bodies have drifted
    // a noticeable fraction of the tightest host's coupling radius. In
    // between, the slab window widens by `delta_max`, which keeps the
    // candidate set exact — a body within `cut` of a host today was within
    // `cut + delta_max` of it at the last build, so nothing is missed, and
    // `pn_pair` (which re-checks the current distance against the per-pair
    // cutoff) drops the few extra margin bodies.
    let margin = if any_active_host {
        if !cached {
            pn_rebuild_order(&*b, &mut order, &mut x0, n);
            b.pn_valid = true;
            b.pn_len = n;
            0.0
        } else {
            let cut_min = G * (m_host_min + m_max) * inv_c2 * (1.0 / PN_COUPLING_MIN);
            if delta_max > PN_SORT_DELTA_FRAC * cut_min {
                pn_rebuild_order(&*b, &mut order, &mut x0, n);
                0.0
            } else {
                delta_max
            }
        }
    } else {
        0.0
    };

    for &i in &hosts {
        if on(i) {
            let xi = b.x[i];
            let cut_max = G * (b.m[i] + m_max) * inv_c2 * (1.0 / PN_COUPLING_MIN);
            let span = cut_max + margin;
            let lo = pn_partition_lo(&order, &x0, xi - span);
            let hi = pn_partition_hi(&order, &x0, xi + span);
            for &ju in &order[lo..hi] {
                let j = ju as usize;
                if b.dead[j] || j == i || (is_host_kind(b.kind[j]) && j < i) {
                    continue;
                }
                pn_pair(b, i, j, true, on(j), inv_c2, inv_c5);
            }
        } else if let Some(act) = active {
            for &ju in act {
                let j = ju as usize;
                if b.dead[j] || j == i || (is_host_kind(b.kind[j]) && j < i) {
                    continue;
                }
                pn_pair(b, i, j, false, true, inv_c2, inv_c5);
            }
        }
    }

    // Put the scratch buffers back (capacity is preserved for the next call).
    b.pn_order = order;
    b.pn_x0 = x0;
    b.pn_hosts = hosts;
    b.pn_mask = mask;
}
