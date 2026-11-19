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
    let (xs, ys, zs, ms) = (&b.x, &b.y, &b.z, &b.m);
    let (vxs, vys, vzs) = (&b.vx, &b.vy, &b.vz);
    let (xi, yi, zi, mi) = (xs[i], ys[i], zs[i], ms[i]);
    let (vxi, vyi, vzi) = (vxs[i], vys[i], vzs[i]);
    let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
    let mut s_max = 0.0f64;
    let mut w_max = 0.0f64;
    for j in range {
        let mj = ms[j];
        let dx = xs[j] - xi;
        let dy = ys[j] - yi;
        let dz = zs[j] - zi;
        let d2 = dx * dx + dy * dy + dz * dz + SOFT2;
        let d = d2.sqrt();
        let inv = 1.0 / (d2 * d);
        let f = G * mj * inv;
        sx += f * dx;
        sy += f * dy;
        sz += f * dz;
        let s = (mi + mj) * inv;
        let dvx = vxs[j] - vxi;
        let dvy = vys[j] - vyi;
        let dvz = vzs[j] - vzi;
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
    let mut s_max = vec![0.0f64; n];
    let mut w_max = vec![0.0f64; n];

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
        let (sm_hi, sm_rest) = s_max.split_at_mut(i + 1);
        let (wm_hi, wm_rest) = w_max.split_at_mut(i + 1);
        let mut si = sm_hi[i];
        let mut wi = wm_hi[i];

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

        ax_hi[i] += axi;
        ay_hi[i] += ayi;
        az_hi[i] += azi;
        sm_hi[i] = si;
        wm_hi[i] = wi;
        // s_max[i], w_max[i] are final for body i: nothing will ever
        // update them again (all j > i pairs have already been folded
        // into sm_rest/wm_rest above). Emit dtl[i] now and skip the
        // post-loop pass.
        dtls[i] = dt_limit(si, wi);
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

    // Post-Newtonian corrections. 1PN is the leading GR correction
    // (perihelion precession); 2.5PN is the leading gravitational-wave
    // radiation reaction (orbital decay). Only run when at least one
    // body is moving at v/c > 1e-5 (about 3 km/s). Only pairs that
    // include a host body are corrected, so the cost is O(hosts · n).
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
pub(super) fn add_pn_corrections(b: &mut Bodies, active: Option<&[u32]>) {
    let n = b.len();
    if n < 2 {
        return;
    }
    let c = c_sim();
    let inv_c2 = 1.0 / (c * c);
    let inv_c5 = inv_c2 * inv_c2 / c;

    // When called from the block stepper, only update active bodies so
    // inactive bodies don't get PN added twice.
    let mask: Option<Vec<bool>> = active.map(|a| {
        let mut m = vec![false; n];
        for &k in a {
            m[k as usize] = true;
        }
        m
    });
    let on = |k: usize| mask.as_ref().map_or(true, |m| m[k]);

    // Every pair that gets PN terms contains a host body. The outer loop
    // visits hosts only. The `j` loop below skips a host `j` with `j < i`,
    // so a host-host pair is done once, and a host-planet pair is done
    // once. Planet-planet pairs get no PN terms (they are negligible).
    let hosts: Vec<usize> = (0..n)
        .filter(|&k| !b.dead[k] && is_host_kind(b.kind[k]))
        .collect();
    for &i in &hosts {
        let mi = b.m[i];
        let xi = b.x[i];
        let yi = b.y[i];
        let zi = b.z[i];
        let vxi = b.vx[i];
        let vyi = b.vy[i];
        let vzi = b.vz[i];
        let vi_sq = vxi * vxi + vyi * vyi + vzi * vzi;

        for j in 0..n {
            if j == i || b.dead[j] || (is_host_kind(b.kind[j]) && j < i) {
                continue;
            }
            let mj = b.m[j];
            let dx = xi - b.x[j];
            let dy = yi - b.y[j];
            let dz = zi - b.z[j];
            let r2 = dx * dx + dy * dy + dz * dz + SOFT2;
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
            if on(i) {
                b.ax[i] += coeff_i * (bracket_i * nxi + vcoef_i * (vxi - vxj));
                b.ay[i] += coeff_i * (bracket_i * nyi + vcoef_i * (vyi - vyj));
                b.az[i] += coeff_i * (bracket_i * nzi + vcoef_i * (vzi - vzj));
            }

            let coeff_j = G * mi * inv_c2 * r_inv * r_inv;
            let bracket_j = -(vj_sq + 2.0 * vi_sq - 4.0 * v_dot - 1.5 * n_dot_vi * n_dot_vi)
                + 5.0 * G * mj * r_inv
                + 4.0 * G * mi * r_inv;
            let vcoef_j = -4.0 * n_dot_vj + 3.0 * n_dot_vi;
            if on(j) {
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
                if on(i) {
                    b.ax[i] += coeff_i * rvx;
                    b.ay[i] += coeff_i * rvy;
                    b.az[i] += coeff_i * rvz;
                }
                if on(j) {
                    b.ax[j] += coeff_j * rvx;
                    b.ay[j] += coeff_j * rvy;
                    b.az[j] += coeff_j * rvz;
                }
            }
        }
    }
}
