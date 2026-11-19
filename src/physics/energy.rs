//! Energy and angular-momentum bookkeeping.
//!
//! These routines are diagnostics, not physics: they read the current
//! `Bodies` state and never mutate it.  They are called from the
//! simulation loop's statistics hooks, so their cost lands on the hot
//! path and the potential-energy sum in particular is worth making
//! cache- and thread-friendly.

use super::*;

/// Cut-over below which we skip rayon.  For small `n` the per-task
/// dispatch cost (work-stealing, channel ops) exceeds the work itself;
/// around 128 bodies the parallel path reliably wins on 4+ cores.
const PE_PAR_THRESHOLD: usize = 128;

/// Total mechanical energy `KE + PE`.
///
/// Returns `None` for systems above `ENERGY_MAX_N` so a caller can
/// cheaply refuse to compute an O(n²) diagnostic on huge systems
/// instead of stalling the frame.  The same cap is enforced (silently)
/// by `energy_dissipated_pct`, so the two stay consistent.
pub fn total_energy(b: &Bodies) -> Option<f64> {
    let n = b.len();
    if n > ENERGY_MAX_N {
        return None;
    }

    // Kinetic energy: independent per-body, so sum in index order for a
    // deterministic result.  Loading velocity into locals lets the
    // compiler keep them in registers across the square.
    let ke: f64 = (0..n)
        .map(|i| {
            let vx = b.vx[i];
            let vy = b.vy[i];
            let vz = b.vz[i];
            0.5 * b.m[i] * (vx * vx + vy * vy + vz * vz)
        })
        .sum();

    Some(ke + potential_energy(b, n))
}

/// Contribution of every pair `(i, m)` with `m > i` — i.e. the upper
/// triangle of the symmetric pair matrix.
///
/// `G*m[i]` is hoisted out of the loop: it's invariant per row and
/// turns "one multiply for G, one for m[i], one for m[m]" into a single
/// multiply of the pre-scaled mass.  Rounding is unchanged because
/// `(-G*m[i]) * m[m]` and `-(G*m[i]*m[m])` differ only by sign-bit
/// placement, and `-x + y == y - x` in IEEE 754.
#[inline]
fn row_pe(b: &Bodies, i: usize, n: usize) -> f64 {
    let xi = b.x[i];
    let yi = b.y[i];
    let zi = b.z[i];
    let gmi = -G * b.m[i];
    let mut s = 0.0;
    for m in (i + 1)..n {
        let dx = b.x[m] - xi;
        let dy = b.y[m] - yi;
        let dz = b.z[m] - zi;
        let r2 = dx * dx + dy * dy + dz * dz + SOFT2;
        // Softened Newtonian potential: -G m_i m_j / sqrt(r² + ε²).
        // `SOFT2` matches the softened force used by the integrator, so
        // this PE is the true conserved quantity of the simulated system,
        // not the idealized point-mass one.
        s += gmi * b.m[m] / r2.sqrt();
    }
    s
}

/// Sum of `row_pe(i, n)` for `i = 0..n`.
///
/// Naïve parallelization over `(0..n)` makes row 0 do `n-1` pair
/// evaluations and row `n-1` do zero.  Rayon work-stealing smooths this
/// out but cannot remove the tail: with `T` threads, the last ~`n/2`
/// rows still idle at the end.  We instead pair row `k` with row
/// `n-1-k` so **every** task evaluates exactly `n-1` pairs, giving
/// near-perfect load balance regardless of thread count.  For odd `n`
/// the middle row has no partner and is added serially.
fn potential_energy(b: &Bodies, n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let mid = n / 2;

    let mut pe: f64 = if n >= PE_PAR_THRESHOLD {
        let parts: Vec<f64> = (0..mid)
            .into_par_iter()
            .map(|k| {
                let i = k;
                let j = n - 1 - k;
                let mut s = row_pe(b, i, n);
                // `j == i` only when n is odd *and* k == mid, which cannot
                // happen inside 0..mid — but keep the guard for clarity.
                if j != i {
                    s += row_pe(b, j, n);
                }
                s
            })
            .collect();
        parts.iter().sum::<f64>()
    } else {
        // Serial path for small n: same pairing, no rayon overhead.
        let mut s = 0.0;
        for k in 0..mid {
            let i = k;
            let j = n - 1 - k;
            s += row_pe(b, i, n);
            if j != i {
                s += row_pe(b, j, n);
            }
        }
        s
    };

    // Odd n leaves the middle row unmatched by the loop above.
    if n % 2 == 1 {
        pe += row_pe(b, mid, n);
    }
    pe
}

/// Total angular momentum (z component only until the vector form lands in
/// the 3D work) about the center of mass, including intrinsic
/// spin.  `L = Σ m_i (r_i - R) × (v_i - V) + Σ spin_i`.
///
/// Two passes on purpose.  The algebraically equivalent one-pass form
/// `L = Σ m_i r_i × v_i − M · R × V` suffers catastrophic cancellation
/// when the CoM is far from the origin (common early in a run, e.g.
/// a cluster collapsing from a wide initial spread), losing most of the
/// significant digits of a quantity we want to conserve to ~1e-12.
/// Subtracting the CoM *before* multiplying keeps every term O(r−R),
/// which is the scale that actually carries the physics.
pub fn total_angular_momentum(b: &Bodies) -> f64 {
    let n = b.len();
    if n == 0 {
        return 0.0;
    }

    // Pass 1: mass-weighted sums → CoM position and velocity.
    let (mut mt, mut mx, mut my, mut mvx, mut mvy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        let mi = b.m[i];
        mt += mi;
        mx += mi * b.x[i];
        my += mi * b.y[i];
        mvx += mi * b.vx[i];
        mvy += mi * b.vy[i];
    }
    if mt <= 0.0 {
        // Degenerate system; angular momentum is undefined.
        return 0.0;
    }
    // Divide rather than multiply by `1/mt`: bit-identical to the
    // straightforward form, which makes regression diffs against the
    // previous implementation exact.
    let cx = mx / mt;
    let cy = my / mt;
    let vx = mvx / mt;
    let vy = mvy / mt;

    // Pass 2: cross products about the CoM.  The hoisted differences let
    // LLVM CSE the repeated loads and keep `dx,dy,dvx,dvy` in registers.
    let mut l = 0.0;
    for i in 0..n {
        let mi = b.m[i];
        let dx = b.x[i] - cx;
        let dy = b.y[i] - cy;
        let dvx = b.vx[i] - vx;
        let dvy = b.vy[i] - vy;
        l += mi * (dx * dvy - dy * dvx) + b.spin[i];
    }
    l
}

/// Snapshot the current energy as the "before" reference for a step.
/// Idempotent: if a snapshot is already pending (e.g. a step was
/// aborted and retried), the earlier one wins so drift accumulates
/// over the whole logical step rather than being reset mid-way.
pub(super) fn mark_energy(b: &mut Bodies) {
    if b.e_pre.is_none() {
        b.e_pre = total_energy(b);
    }
}

/// Compare against the snapshot taken by `mark_energy` and fold the
/// delta into `stats.energy_offset`.  `energy_offset` is a cumulative
/// record of energy injected/removed by non-conservative operations
/// (rebase, merger, resampling); it is subtracted from the raw drift
/// in `energy_drift_pct` so the reported drift reflects only the
/// integrator, not deliberate bookkeeping adjustments.
pub(super) fn settle_energy(b: &mut Bodies) {
    if let Some(e0) = b.e_pre.take() {
        if let Some(e1) = total_energy(b) {
            b.stats.energy_offset += e1 - e0;
        }
    }
}

/// Re-baseline the energy tracker: the current energy becomes the new
/// "expected" value and the accumulated offset is cleared.  Call this
/// after any operation that intentionally changes the system's energy
/// (adding/removing bodies, resizing the domain) so subsequent drift
/// measurements are meaningful again.
pub fn rebase_energy(b: &mut Bodies) {
    b.stats.energy0 = total_energy(b).unwrap_or(0.0);
    b.stats.energy_offset = 0.0;
    b.e_pre = None;
}

/// Integrator drift in percent of the baseline energy, *excluding*
/// changes already accounted for by `energy_offset`.
///
/// Uses `|e0|` in the denominator so the sign of the drift is meaningful
/// even for bound (negative-energy) systems.  Returns `None` if the
/// baseline is zero (undefined ratio) or the system exceeds the
/// `ENERGY_MAX_N` cap enforced by `total_energy`.
pub fn energy_drift_pct(b: &Bodies) -> Option<f64> {
    let e = total_energy(b)?;
    let e0 = b.stats.energy0;
    if e0 == 0.0 {
        return None;
    }
    Some(((e - b.stats.energy_offset - e0) / e0.abs()) * 100.0)
}

/// Fraction of the baseline energy that has been removed or injected by
/// non-conservative operations, in percent.  Complementary to
/// `energy_drift_pct`: drift + dissipated ≈ total change from `energy0`.
pub fn energy_dissipated_pct(b: &Bodies) -> Option<f64> {
    let e0 = b.stats.energy0;
    if e0 == 0.0 || b.len() > ENERGY_MAX_N {
        return None;
    }
    Some(b.stats.energy_offset / e0.abs() * 100.0)
}
