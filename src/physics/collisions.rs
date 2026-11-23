//! Collision detection and the impact model.
//!
//! Detection is a broad phase (uniform hash grid for larger populations,
//! plain pair loop for small ones) followed by a swept-sphere narrow phase,
//! so a fast pair cannot tunnel through each other within one `DT = 1/480`
//! step. Resolution routes every detected pair through `collide`, which
//! chooses merge / elastic bounce / hit-and-run / catastrophic fragmentation
//! from the dimensionless impact parameter `b` and the relative speed in
//! units of escape speed.

use super::*;

/// Below this many "hot" bodies the hash grid's fixed cost (bucket table,
/// counting sort, neighbour probe) exceeds a plain O(n²) pair scan, so
/// `find_collisions_hot` falls back to the loop.
pub(super) const HOT_GRID_MIN: usize = 24;

/// Queue a short-lived visual flash. Purely cosmetic; never affects physics.
pub(super) fn flash_at(b: &mut Bodies, x: f64, y: f64, z: f64, r0: f64) {
    b.flashes.push(Flash {
        x,
        y,
        z,
        r0,
        age: 0.0,
        life: FLASH_LIFE,
        expires_at: b.time + FLASH_LIFE,
    });
}

/// Impact parameter of the pair `(i, j)`: the perpendicular distance from
/// body `i` to the relative-velocity line through body `j`. Normalised by
/// the contact radius it becomes the dimensionless `b` in `[0, 1]` used to
/// distinguish head-on (`b → 0`) from grazing (`b → 1`) encounters.
///
/// Returns `0.0` for a stationary pair so a degenerate contact is treated
/// as head-on rather than NaN.
pub(super) fn impact_offset(b: &Bodies, i: usize, j: usize) -> f64 {
    let dx = b.x[j] - b.x[i];
    let dy = b.y[j] - b.y[i];
    let dz = b.z[j] - b.z[i];
    let vx = b.vx[j] - b.vx[i];
    let vy = b.vy[j] - b.vy[i];
    let vz = b.vz[j] - b.vz[i];
    let v = (vx * vx + vy * vy + vz * vz).sqrt();
    if v <= 0.0 {
        return 0.0;
    }
    let (cx, cy, cz) = (dy * vz - dz * vy, dz * vx - dx * vz, dx * vy - dy * vx);
    (cx * cx + cy * cy + cz * cz).sqrt() / v
}

/// Convenience wrapper that computes `rr = r_i + r_j` internally. Prefer
/// `pair_hit_rr` inside a broad-phase loop, where `rr` can be cached.
pub(super) fn pair_hit(b: &Bodies, i: usize, j: usize, sweep: bool) -> bool {
    let rr = b.contact_r(i) + b.contact_r(j);
    pair_hit_rr(b, i, j, rr, sweep)
}

/// Swept-sphere overlap test for a supplied contact radius `rr = r_i + r_j`.
///
/// First checks the current overlap, then — if `sweep` — sweeps the segment
/// `x0 -> x` of relative displacement, parameterised by `t in (0, 1)`, and
/// tests the distance from the origin to that segment. This is what stops
/// a fast pair from passing through each other between two sampled steps.
#[inline]
pub(super) fn pair_hit_rr(b: &Bodies, i: usize, j: usize, rr: f64, sweep: bool) -> bool {
    let rr2 = rr * rr;
    let dx = b.x[j] - b.x[i];
    let dy = b.y[j] - b.y[i];
    let dz = b.z[j] - b.z[i];
    if dx * dx + dy * dy + dz * dz < rr2 {
        return true;
    }
    if !sweep {
        return false;
    }
    let x0 = b._px[j] - b._px[i];
    let y0 = b._py[j] - b._py[i];
    let z0 = b._pz[j] - b._pz[i];
    let ex = dx - x0;
    let ey = dy - y0;
    let ez = dz - z0;
    let e2 = ex * ex + ey * ey + ez * ez;
    if e2 <= 0.0 || e2.is_nan() {
        return false;
    }
    let t = -(x0 * ex + y0 * ey + z0 * ez) / e2;
    if t <= 0.0 || t >= 1.0 {
        return false;
    }
    let cx = x0 + t * ex;
    let cy = y0 + t * ey;
    let cz = z0 + t * ez;
    cx * cx + cy * cy + cz * cz < rr2
}

/// 64-bit hash of the integer grid cell `(cx, cy)`. Two independent
/// odd-multiplier mixers XORed together; good enough to scatter adjacent
/// cells in a sort-based spatial hash without an actual hash table.
#[inline]
pub(super) fn cell_key(cx: i64, cy: i64) -> u64 {
    (cx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (cy as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
}

/// Broad + narrow phase over every live body.
///
/// Below `HASH_THRESHOLD` the O(n²) pair loop dominates the grid's fixed
/// cost, so it wins. Above it, a uniform grid keyed by `cell_key` is built,
/// sorted once, and probed with the 3×3 neighbourhood of each body.
///
/// The grid edge is sized to hold both static overlap (`2 · max contact
/// radius`) and one step of relative motion (`2 · vmax · dt · 1.05`), so a
/// pair that could overlap this step is guaranteed to be within one cell
/// neighbourhood. Pairs come out sorted and deduplicated so the caller's
/// merge order is deterministic.
pub(super) fn find_collisions(b: &Bodies, sweep: bool, dt: f64) -> Vec<(usize, usize)> {
    let n = b.len();
    let t_now = b.time;
    let mut pairs: Vec<(usize, usize)> = Vec::new();

    if n < HASH_THRESHOLD {
        let radii: Vec<f64> = (0..n).map(|i| b.contact_r(i)).collect();
        for i in 0..n {
            if b.dead[i] || t_now < b.imm[i] {
                continue;
            }
            let ri = radii[i];
            for j in (i + 1)..n {
                if b.dead[j] || t_now < b.imm[j] {
                    continue;
                }
                if pair_hit_rr(b, i, j, ri + radii[j], sweep) {
                    pairs.push((i, j));
                }
            }
        }
        return pairs;
    }

    let mut radii = vec![0.0f64; n];
    let mut max_cr: f64 = 0.0;
    let mut vmax2: f64 = 0.0;
    for i in 0..n {
        if b.dead[i] {
            continue;
        }
        let ri = b.contact_r(i);
        radii[i] = ri;
        if ri > max_cr {
            max_cr = ri;
        }
        if sweep {
            let v2 = b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i] + b.vz[i] * b.vz[i];
            if v2 > vmax2 {
                vmax2 = v2;
            }
        }
    }
    let motion = if sweep {
        2.0 * vmax2.sqrt() * dt * 1.05
    } else {
        0.0
    };
    let cell = (2.0 * max_cr + motion).max(1e-4);
    let inv_cell = 1.0 / cell;

    // Immune bodies never collide, so they are not added to the sorted key
    // list — every pair query already filters them out on either side.
    let mut keys: Vec<(u64, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        if b.dead[i] || t_now < b.imm[i] {
            continue;
        }
        let cx = (b.x[i] * inv_cell).floor() as i64;
        let cy = (b.y[i] * inv_cell).floor() as i64;
        keys.push((cell_key(cx, cy), i as u32));
    }
    if keys.len() > 8192 {
        keys.par_sort_unstable();
    } else {
        keys.sort_unstable();
    }

    for i in 0..n {
        if b.dead[i] || t_now < b.imm[i] {
            continue;
        }
        let cx = (b.x[i] * inv_cell).floor() as i64;
        let cy = (b.y[i] * inv_cell).floor() as i64;
        let ri = radii[i];
        for ddx in -1..=1i64 {
            for ddy in -1..=1i64 {
                let key = cell_key(cx + ddx, cy + ddy);
                let start = keys.partition_point(|e| e.0 < key);
                for e in &keys[start..] {
                    if e.0 != key {
                        break;
                    }
                    let j = e.1 as usize;
                    if j <= i || b.dead[j] || t_now < b.imm[j] {
                        continue;
                    }
                    if pair_hit_rr(b, i, j, ri + radii[j], sweep) {
                        pairs.push((i, j));
                    }
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Same contract as `find_collisions`, restricted to bodies in `hot`
/// (expected ascending — e.g. the set of bodies that moved this step).
///
/// Falls back to the full scan when `hot` is nearly the whole population,
/// because an almost-dense hot set is exactly what the O(n²) fallback is
/// best at. Pair order matches the brute-force loop (ascending i, then
/// ascending j).
pub(super) fn find_collisions_hot(
    b: &Bodies,
    hot: &[usize],
    sweep: bool,
    dt: f64,
) -> Vec<(usize, usize)> {
    let _ = dt;
    if hot.len() * hot.len() > 4_000_000 {
        return find_collisions(b, sweep, dt);
    }
    let t_now = b.time;
    let live: Vec<usize> = hot
        .iter()
        .copied()
        .filter(|&i| !b.dead[i] && t_now >= b.imm[i])
        .collect();
    let nl = live.len();
    let mut pairs = Vec::new();

    // Too few hot bodies to amortise a hash table.
    if nl < HOT_GRID_MIN {
        let radii: Vec<f64> = live.iter().map(|&i| b.contact_r(i)).collect();
        for (a, &i) in live.iter().enumerate() {
            let ri = radii[a];
            for (k, &j) in live[a + 1..].iter().enumerate() {
                if pair_hit_rr(b, i, j, ri + radii[a + 1 + k], sweep) {
                    pairs.push((i, j));
                }
            }
        }
        return pairs;
    }

    let mut radii = vec![0.0f64; nl];
    let mut max_cr = 0.0f64;
    let mut max_disp_sq = 0.0f64;
    for (a, &i) in live.iter().enumerate() {
        let ri = b.contact_r(i);
        radii[a] = ri;
        if ri > max_cr {
            max_cr = ri;
        }
        if sweep {
            let ddx = b.x[i] - b._px[i];
            let ddy = b.y[i] - b._py[i];
            let ddz = b.z[i] - b._pz[i];
            let d2 = ddx * ddx + ddy * ddy + ddz * ddz;
            if d2 > max_disp_sq {
                max_disp_sq = d2;
            }
        }
    }
    let max_disp = max_disp_sq.sqrt();
    let cell = ((2.0 * max_cr + 2.0 * max_disp) * 1.0001).max(1e-4);
    let inv = 1.0 / cell;

    // Counting-sort bucket table: `start[k]..start[k+1]` is the slice of
    // `items` whose `cell_key` truncated to `table` slots equals `k`. Hash
    // collisions simply cluster extra bodies into the same bucket, which
    // costs a few extra distance checks but never loses a pair.
    let table = (2 * nl).next_power_of_two().max(16);
    let shift = 64 - table.trailing_zeros();
    let bucket = |cx: i64, cy: i64| (cell_key(cx, cy) >> shift) as usize;

    let cells: Vec<(i64, i64)> = live
        .iter()
        .map(|&i| ((b.x[i] * inv).floor() as i64, (b.y[i] * inv).floor() as i64))
        .collect();
    let mut start = vec![0u32; table + 1];
    for &(cx, cy) in &cells {
        start[bucket(cx, cy) + 1] += 1;
    }
    for k in 0..table {
        start[k + 1] += start[k];
    }
    let mut fill: Vec<u32> = start[..table].to_vec();
    let mut items = vec![0u32; nl];
    for (p, &(cx, cy)) in cells.iter().enumerate() {
        let k = bucket(cx, cy);
        items[fill[k] as usize] = p as u32;
        fill[k] += 1;
    }

    for (p, &i) in live.iter().enumerate() {
        let (cx, cy) = cells[p];
        let ri = radii[p];
        // Multiple of the 9 neighbour cells can hash to the same bucket;
        // skip repeats so each bucket is scanned at most once per body.
        let mut seen = [usize::MAX; 9];
        let mut ns = 0;
        for ddx in -1..=1i64 {
            for ddy in -1..=1i64 {
                let k = bucket(cx + ddx, cy + ddy);
                if seen[..ns].contains(&k) {
                    continue;
                }
                seen[ns] = k;
                ns += 1;
                for &q in &items[start[k] as usize..start[k + 1] as usize] {
                    let q = q as usize;
                    if q <= p {
                        continue;
                    }
                    let j = live[q];
                    if pair_hit_rr(b, i, j, ri + radii[q], sweep) {
                        pairs.push((i, j));
                    }
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs
}

/// Hot bodies against cold ones. `find_collisions_hot` only pairs hot with
/// hot, but every body drifts each sub-tick, so hot-vs-cold hits must be
/// tested too.
fn hot_vs_cold(b: &Bodies, hot: &[usize]) -> Vec<(usize, usize)> {
    let t_now = b.time;
    let n = b.len();
    let mut in_hot = vec![false; n];
    for &i in hot {
        in_hot[i] = true;
    }
    let live_hot: Vec<usize> = hot
        .iter()
        .copied()
        .filter(|&i| !b.dead[i] && t_now >= b.imm[i])
        .collect();
    let cold: Vec<usize> = (0..n)
        .filter(|&j| !in_hot[j] && !b.dead[j] && t_now >= b.imm[j])
        .collect();
    if live_hot.is_empty() || cold.is_empty() {
        return Vec::new();
    }

    // Grid cell edge: static overlap (2 x largest contact radius) plus one
    // sub-step of motion for both bodies, the same sizing rule as
    // `find_collisions_hot`. Any pair that `pair_hit_rr` can accept lies
    // within one cell of each other, so a 3x3 probe never misses a hit.
    let mut max_cr = 0.0f64;
    let mut max_disp_sq = 0.0f64;
    for &i in live_hot.iter().chain(cold.iter()) {
        max_cr = max_cr.max(b.contact_r(i));
        let ddx = b.x[i] - b._px[i];
        let ddy = b.y[i] - b._py[i];
        let ddz = b.z[i] - b._pz[i];
        max_disp_sq = max_disp_sq.max(ddx * ddx + ddy * ddy + ddz * ddz);
    }
    let cell = ((2.0 * max_cr + 2.0 * max_disp_sq.sqrt()) * 1.0001).max(1e-4);
    let inv = 1.0 / cell;

    let nc = cold.len();
    let table = (2 * nc).next_power_of_two().max(16);
    let shift = 64 - table.trailing_zeros();
    let bucket = |cx: i64, cy: i64| (cell_key(cx, cy) >> shift) as usize;

    // Counting-sort bucket table over the cold bodies (same scheme as
    // `find_collisions_hot`): hash collisions only add a few extra distance
    // checks, they never lose a pair.
    let cells: Vec<(i64, i64)> = cold
        .iter()
        .map(|&j| ((b.x[j] * inv).floor() as i64, (b.y[j] * inv).floor() as i64))
        .collect();
    let radii: Vec<f64> = cold.iter().map(|&j| b.contact_r(j)).collect();
    let mut start = vec![0u32; table + 1];
    for &(cx, cy) in &cells {
        start[bucket(cx, cy) + 1] += 1;
    }
    for k in 0..table {
        start[k + 1] += start[k];
    }
    let mut fill: Vec<u32> = start[..table].to_vec();
    let mut items = vec![0u32; nc];
    for (p, &(cx, cy)) in cells.iter().enumerate() {
        let k = bucket(cx, cy);
        items[fill[k] as usize] = p as u32;
        fill[k] += 1;
    }

    let mut out = Vec::new();
    for &i in &live_hot {
        let ri = b.contact_r(i);
        let cx = (b.x[i] * inv).floor() as i64;
        let cy = (b.y[i] * inv).floor() as i64;
        // Several of the 9 neighbour cells can hash to one bucket: scan
        // each bucket once per body.
        let mut seen = [usize::MAX; 9];
        let mut ns = 0;
        for ddx in -1..=1i64 {
            for ddy in -1..=1i64 {
                let k = bucket(cx + ddx, cy + ddy);
                if seen[..ns].contains(&k) {
                    continue;
                }
                seen[ns] = k;
                ns += 1;
                for &q in &items[start[k] as usize..start[k + 1] as usize] {
                    let q = q as usize;
                    let j = cold[q];
                    if pair_hit_rr(b, i, j, ri + radii[q], true) {
                        out.push((i.min(j), i.max(j)));
                    }
                }
            }
        }
    }
    out
}

/// One detection pass and one resolution pass per call.
///
/// Every body involved in a resolved pair becomes immune for
/// `IMMUNITY_TIME` (0.4 sim units, much larger than `DT = 1/480`), so no
/// new pair involving it can survive the immunity filter within this same
/// call. That is why a single pass is sufficient — the cascade the old
/// multi-iteration loop guarded against cannot occur. Newly spawned
/// fragments inherit the same immunity so they cannot be re-hit on their
/// birth step.
pub(super) fn collision_pass(b: &mut Bodies, dt: f64, hot: Option<&[usize]>) {
    if b.len() < 2 {
        return;
    }
    let mut spawns: Vec<Spawn> = Vec::new();
    let pairs = match hot {
        Some(h) => {
            let mut p = find_collisions_hot(b, h, true, dt);
            // Brute force is fine while hot × n stays small; above that
            // it falls back to hot-vs-hot only.
            if h.len() * b.len() <= 2_000_000 {
                p.extend(hot_vs_cold(b, h));
                p.sort_unstable();
                p.dedup();
            }
            p
        }
        None => find_collisions(b, true, dt),
    };
    for (i, j) in pairs {
        if b.dead[i] || b.dead[j] {
            continue;
        }
        if !pair_hit(b, i, j, true) {
            continue;
        }
        collide(b, i, j, &mut spawns);
    }
    if b.any_dead() {
        b.compact();
    }
    let t_imm = b.time + IMMUNITY_TIME;
    for s in spawns {
        let k = b.push(s.x, s.y, s.vx, s.vy, s.m, s.kind);
        b.z[k] = s.z;
        b.vz[k] = s.vz;
        b._pz[k] = s.z;
        b.imm[k] = t_imm;
        b.origin[k] = s.origin;
        b.class[k] = s.class;
    }
}

/// Specific orbital angular momentum of body `i` about the origin:
/// `m · (r × v)`, taking the z-component only.
#[inline]
pub(super) fn l_orb(b: &Bodies, i: usize) -> f64 {
    b.m[i] * (b.x[i] * b.vy[i] - b.y[i] * b.vx[i])
}

/// Total angular momentum of a pair: both orbital terms plus both spin
/// terms. Callers snapshot this before an inelastic event and rebalance
/// the difference into spin afterward, so `L` is conserved to round-off
/// across merges, bounces, and hit-and-run stripping.
pub(super) fn pair_l(b: &Bodies, i: usize, j: usize) -> f64 {
    l_orb(b, i) + l_orb(b, j) + b.spin[i] + b.spin[j]
}

/// Reciprocal of Earth's mass in sim units — the reference density used
/// by `q_star_rd`. `R_EARTH_GM` is the Earth radius in units where
/// `G · M_earth = 1`, so `EARTH_RHO_GCC · (4/3)π · R_EARTH_GM³` is the
/// Earth's mass in sim units.
pub(super) fn rho1_sim() -> f64 {
    1.0 / (EARTH_RHO_GCC * 4.0 / 3.0 * std::f64::consts::PI * R_EARTH_GM.powi(3))
}

/// Fraction of the smaller body's volume that overlaps the larger one at
/// impact parameter `bb`. `1.0` once the smaller sphere is fully engulfed
/// (`l ≥ 2·rp`), smoothly decaying to `0` at grazing (`bb → 1`). The
/// analytic form is the spherical-cap volume normalised to `1` at full
/// overlap; using it (rather than a step) is what makes oblique impacts
/// erode less than head-on ones.
pub(super) fn interact_frac(rt: f64, rp: f64, bb: f64) -> f64 {
    let l = (rt + rp) * (1.0 - bb);
    if l <= 0.0 {
        return 0.0;
    }
    if l >= 2.0 * rp {
        return 1.0;
    }
    (3.0 * rp * l * l - l * l * l) / (4.0 * rp * rp * rp)
}

/// Specific disruption energy `Q*_RD` for the gravity regime, following
/// Stewart & Leinhardt: the energy per unit mass needed to disperse half
/// the target. Scales with the target's surface gravity × radius
/// (`∝ rc1²`) and with the impactor-to-target mass ratio `γ` via
/// `(1 + γ)² / 4γ`.
pub(super) fn q_star_rd(m_tot: f64, gamma: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let rho1 = rho1_sim();
    let rc1 = (3.0 * m_tot / (4.0 * pi * rho1)).cbrt();
    let q11 = LS_C_STAR * 0.8 * pi * rho1 * G * rc1 * rc1;
    // Leinhardt & Stewart (2012): the mass-ratio factor has the exponent
    // 2/(3·μ̄) − 1, which is about 0.80 and not 1.
    let ratio_factor = (gamma + 1.0).powi(2) / (4.0 * gamma);
    q11 * ratio_factor.powf(2.0 / (3.0 * LS_MU_BAR) - 1.0)
}

/// Draw `n` fragment masses from a power-law distribution with slope
/// `FRAG_SLOPE`, spanning `FRAG_DYN_RANGE` (largest / smallest mass),
/// then rescale so the total is exactly `md`. Inverse-CDF sampling; the
/// single rescale means the masses sum to `md` to round-off.
pub(super) fn frag_masses(rng: &mut XorShift, n: usize, md: f64) -> Vec<f64> {
    let beta = FRAG_SLOPE / 3.0;
    let hi = FRAG_DYN_RANGE.powf(-beta);
    let mut v: Vec<f64> = (0..n)
        .map(|_| (1.0 - rng.next_f64() * (1.0 - hi)).powf(-1.0 / beta))
        .collect();
    let sum: f64 = v.iter().sum();
    for m in v.iter_mut() {
        *m *= md / sum;
    }
    v
}

/// Unit contact normal of a grazing pair: from the relative position `d` and
/// relative velocity `v`, the point of closest approach lies at perpendicular
/// offset `h` from the velocity line, and the sphere of radius `rr` is touched
/// where the line enters it. Falls back to any perpendicular for a head-on hit.
pub(super) fn contact_normal(
    dx: f64,
    dy: f64,
    dz: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    rr: f64,
) -> (f64, f64, f64) {
    let v = (vx * vx + vy * vy + vz * vz).sqrt().max(1e-12);
    let (ux, uy, uz) = (vx / v, vy / v, vz / v);
    let along = dx * ux + dy * uy + dz * uz;
    let (hx, hy, hz) = (dx - along * ux, dy - along * uy, dz - along * uz);
    let d = (hx * hx + hy * hy + hz * hz).sqrt();
    let (hux, huy, huz) = if d > 1e-12 {
        (hx / d, hy / d, hz / d)
    } else {
        let pn = (ux * ux + uy * uy).sqrt();
        if uz == 0.0 {
            (-uy, ux, 0.0)
        } else if pn > 1e-12 {
            (-uy / pn, ux / pn, 0.0)
        } else {
            (1.0, 0.0, 0.0)
        }
    };
    let s = (rr * rr - d * d).max(0.0).sqrt();
    ((d * hux - s * ux) / rr, (d * huy - s * uy) / rr, (d * huz - s * uz) / rr)
}

/// Grazing, non-merging encounter: the small body keeps (most of) its mass
/// and continues roughly along its incoming direction, deflected by the
/// usual elastic impulse. `bb` sets the stripped fraction through
/// `interact_frac`, and `(nn)` is the contact normal at closest approach.
///
/// Angular momentum is conserved by dumping the post-event imbalance into
/// the larger body's spin; both bodies get fresh immunity so the pair does
/// not immediately re-collide next step.
pub(super) fn hit_and_run(b: &mut Bodies, big: usize, small: usize, rr: f64, bb: f64, ratio: f64) {
    mark_energy(b);
    let l_before = pair_l(b, big, small);
    let (dx, dy, dz) = (b.x[small] - b.x[big], b.y[small] - b.y[big], b.z[small] - b.z[big]);
    let (vx, vy, vz) = (
        b.vx[small] - b.vx[big],
        b.vy[small] - b.vy[big],
        b.vz[small] - b.vz[big],
    );
    let (nx, ny, nz) = contact_normal(dx, dy, dz, vx, vy, vz, rr);
    let vn = vx * nx + vy * ny + vz * nz;
    let (mb, ms) = (b.m[big], b.m[small]);
    let mtot = mb + ms;
    if vn < 0.0 {
        let dv = -(1.0 + HIT_RUN_RESTITUTION) * vn;
        b.vx[small] += mb / mtot * dv * nx;
        b.vy[small] += mb / mtot * dv * ny;
        b.vz[small] += mb / mtot * dv * nz;
        b.vx[big] -= ms / mtot * dv * nx;
        b.vy[big] -= ms / mtot * dv * ny;
        b.vz[big] -= ms / mtot * dv * nz;
    }
    let alpha = interact_frac(b.contact_r(big), b.contact_r(small), bb);
    let mt = (HIT_RUN_LOSS * alpha * ms).min(0.5 * ms);
    if mt > 0.0 {
        // Strip `mt` of the small body's mass and hand it to the big one,
        // shifting the pair's centre of mass so the surviving positions
        // stay consistent with the new mass ratio.
        let nb = mb + mt;
        let cx0 = (mb * b.x[big] + ms * b.x[small]) / mtot;
        let cy0 = (mb * b.y[big] + ms * b.y[small]) / mtot;
        let cx1 = (nb * b.x[big] + (ms - mt) * b.x[small]) / mtot;
        let cy1 = (nb * b.y[big] + (ms - mt) * b.y[small]) / mtot;
        let cz0 = (mb * b.z[big] + ms * b.z[small]) / mtot;
        let cz1 = (nb * b.z[big] + (ms - mt) * b.z[small]) / mtot;
        for k in [big, small] {
            b.x[k] += cx0 - cx1;
            b.y[k] += cy0 - cy1;
            b.z[k] += cz0 - cz1;
            b._px[k] = b.x[k];
            b._py[k] = b.y[k];
            b._pz[k] = b.z[k];
        }
        b.vx[big] = (mb * b.vx[big] + mt * b.vx[small]) / nb;
        b.vy[big] = (mb * b.vy[big] + mt * b.vy[small]) / nb;
        b.vz[big] = (mb * b.vz[big] + mt * b.vz[small]) / nb;
        b.set_mass(big, nb);
        b.set_mass(small, ms - mt);
    }
    let l_after = pair_l(b, big, small);
    b.spin[big] += l_before - l_after;
    let t_imm = b.time + IMMUNITY_TIME;
    b.imm[big] = t_imm;
    b.imm[small] = t_imm;
    let (cx, cy) = (
        b.x[big] + nx * b.contact_r(big),
        b.y[big] + ny * b.contact_r(big),
    );
    let cz = b.z[big] + nz * b.contact_r(big);
    flash_at(b, cx, cy, cz, (b.contact_r(big) * 2.0).max(0.05));
    b.stats.impacts += 1;
    let text = format!(
        "hit-and-run #{} × #{} at {:.1}× escape speed, b = {:.2}: {} moved to the heavy body",
        b.id[big],
        b.id[small],
        ratio,
        bb,
        mass_str(mt)
    );
    b.log(EventKind::Impact, text, cx, cy);
}

/// Inelastic bounce between two rubble-pile bodies at moderate relative
/// speed. Same contact geometry as `hit_and_run`, but no mass exchange —
/// both bodies get equal-and-opposite impulses scaled by mass, and the
/// resulting angular-momentum discrepancy is split into spin so `L` is
/// preserved. Brief immunity so the pair cannot ping-pong every step.
pub(super) fn rubble_bounce(b: &mut Bodies, i: usize, j: usize, rr: f64) {
    let dx = b.x[j] - b.x[i];
    let dy = b.y[j] - b.y[i];
    let dz = b.z[j] - b.z[i];
    let dvx = b.vx[j] - b.vx[i];
    let dvy = b.vy[j] - b.vy[i];
    let dvz = b.vz[j] - b.vz[i];
    let (nx, ny, nz) = contact_normal(dx, dy, dz, dvx, dvy, dvz, rr);
    let vn = dvx * nx + dvy * ny + dvz * nz;
    if vn >= 0.0 {
        return;
    }

    mark_energy(b);
    let l_before = pair_l(b, i, j);
    let (mi, mj) = (b.m[i], b.m[j]);
    let mtot = mi + mj;
    let dvn = -(1.0 + RUBBLE_BOUNCE_E) * vn;
    b.vx[i] -= mj / mtot * dvn * nx;
    b.vy[i] -= mj / mtot * dvn * ny;
    b.vz[i] -= mj / mtot * dvn * nz;
    b.vx[j] += mi / mtot * dvn * nx;
    b.vy[j] += mi / mtot * dvn * ny;
    b.vz[j] += mi / mtot * dvn * nz;

    let dl = l_before - pair_l(b, i, j);
    b.spin[i] += dl * mi / mtot;
    b.spin[j] += dl * mj / mtot;

    let t_imm = b.time + RUBBLE_BOUNCE_IMMUNITY;
    b.imm[i] = t_imm;
    b.imm[j] = t_imm;

    b.stats.bounces += 1;
}

/// Which of `(i, j)` survives a merge. Higher-ranked kinds always win — a
/// black hole never becomes a planet — and ties are broken by mass, so
/// identity (and evolution history) sticks to the more massive / more
/// evolved body.
pub(super) fn pick_survivor(b: &Bodies, i: usize, j: usize) -> (usize, usize) {
    let rank = |k: usize| match b.kind[k] {
        BodyKind::BlackHole => 6,
        BodyKind::NeutronStar => 5,
        BodyKind::WhiteDwarf => 4,
        BodyKind::Star => 3,
        BodyKind::BrownDwarf => 2,
        BodyKind::Planet => 1,
        BodyKind::DwarfPlanet => 0,
    };
    let (ri, rj) = (rank(i), rank(j));
    if ri != rj {
        return if ri > rj { (i, j) } else { (j, i) };
    }
    if b.m[i] >= b.m[j] {
        (i, j)
    } else {
        (j, i)
    }
}

/// Remaining spawn capacity, accounting for `pending` spawns already
/// queued but not yet pushed onto `b`.
pub(super) fn room(b: &Bodies, pending: usize) -> usize {
    MAX_BODIES.saturating_sub(b.len() + pending)
}

/// Origin tag of a merged body. Two debris stay debris, two belt bodies
/// stay in the belt, two comets stay a comet — so debris discs and belts
/// keep their classification across gentle merges. Mixed origins promote
/// to `Spawned`, i.e. a regular body.
pub(super) fn combine_origin(keep: BodyOrigin, kill: BodyOrigin) -> BodyOrigin {
    if (keep.is_rubble() && kill.is_rubble())
        || (keep.is_belt() && kill.is_belt())
        || (keep.is_comet() && kill.is_comet())
    {
        keep
    } else {
        BodyOrigin::Spawned
    }
}

/// Merge `kill` into `keep`, conserving mass, momentum, and angular
/// momentum. Walks the full stellar-evolution ladder (star ↔ brown dwarf
/// ↔ planet ↔ white dwarf ↔ neutron star) and preserves a mass-weighted
/// fraction of the survivor's main-sequence age across the merge.
///
/// Returns the (pre-compaction) index of the survivor; the killed body is
/// marked dead but left in place for the next `compact`. `silent` suppresses
/// the flash + log entry for intermediate merges inside a fragmentation
/// event — `collide` pops the spurious entry and re-emits its own.
pub(super) fn merge_impl(b: &mut Bodies, i: usize, j: usize, silent: bool) -> usize {
    mark_energy(b);
    let (keep, kill) = pick_survivor(b, i, j);
    let total = b.m[keep] + b.m[kill];
    let (cx, cy, cz) = merge_motion(b, keep, kill, total);

    let verb = if b.kind[keep] != BodyKind::Planet && b.kind[kill] == BodyKind::Planet {
        "fell into"
    } else {
        "merged into"
    };
    let text = format!("{} {} {}", tag(b, kill), verb, tag(b, keep));

    let keep_was_star = b.kind[keep] == BodyKind::Star;
    let keep_kind_before = b.kind[keep];
    let kill_kind = b.kind[kill];
    blend_age_and_density(b, keep, kill, total);
    b.set_mass(keep, total);
    b.origin[keep] = combine_origin(b.origin[keep], b.origin[kill]);
    promote_by_mass(b, keep, cx, cy);
    climb_stellar_ladder(b, keep, cx, cy);
    reset_clocks_after_merge(b, keep, keep_was_star, keep_kind_before, kill_kind);
    b.r[keep] = b.radius_now(keep);

    b.dead[kill] = true;
    b.coalesce_counter[keep] = 0;

    b.stats.merges += 1;
    if !silent {
        let flash_r = (b.contact_r(keep) * 5.0).max(0.05);
        flash_at(b, cx, cy, cz, flash_r);
        b.log(EventKind::Merge, text, cx, cy);
    }
    keep
}

/// Move `keep` to the pair's centre of mass and momentum, and add the relative
/// orbital angular momentum about it to the survivor's spin. Returns the centre.
fn merge_motion(b: &mut Bodies, keep: usize, kill: usize, total: f64) -> (f64, f64, f64) {
    let cx = (b.m[keep] * b.x[keep] + b.m[kill] * b.x[kill]) / total;
    let cy = (b.m[keep] * b.y[keep] + b.m[kill] * b.y[kill]) / total;
    let cz = (b.m[keep] * b.z[keep] + b.m[kill] * b.z[kill]) / total;
    let cvx = (b.m[keep] * b.vx[keep] + b.m[kill] * b.vx[kill]) / total;
    let cvy = (b.m[keep] * b.vy[keep] + b.m[kill] * b.vy[kill]) / total;
    let cvz = (b.m[keep] * b.vz[keep] + b.m[kill] * b.vz[kill]) / total;

    // Reduce to the two-body problem: relative L about the centre of mass
    // becomes the survivor's spin contribution.
    let mu = b.m[keep] * b.m[kill] / total;
    let rl = (b.x[kill] - b.x[keep]) * (b.vy[kill] - b.vy[keep])
        - (b.y[kill] - b.y[keep]) * (b.vx[kill] - b.vx[keep]);
    b.spin[keep] += b.spin[kill] + mu * rl;

    b.x[keep] = cx;
    b.y[keep] = cy;
    b.vx[keep] = cvx;
    b.vy[keep] = cvy;
    b.z[keep] = cz;
    b.vz[keep] = cvz;
    b._px[keep] = cx;
    b._py[keep] = cy;
    b._pz[keep] = cz;
    (cx, cy, cz)
}

/// Mass-weight the main-sequence age fraction and the bulk density.
///
/// The age fraction is clamped just below 1 so a nearly-departed star does not
/// immediately leave the MS after a merge. For dwarfs the same ratio is a valid
/// "fraction cooled" proxy; `reset_clocks_after_merge` overwrites it once the
/// survivor's final kind is known.
fn blend_age_and_density(b: &mut Bodies, keep: usize, kill: usize, total: f64) {
    let frac_age = |b: &Bodies, k: usize| -> f64 {
        let t = star_lifetime_gyr(BodyKind::Star, b.m_init[k]);
        if t.is_finite() && t > 0.0 {
            (b.age_gyr[k] / t).clamp(0.0, 0.99)
        } else {
            0.0
        }
    };
    let f = (frac_age(b, keep) * b.m[keep] + frac_age(b, kill) * b.m[kill]) / total;
    b.m_init[keep] += b.m_init[kill];
    b.age_gyr[keep] = f * star_lifetime_gyr(BodyKind::Star, b.m_init[keep]);
    if matches!(b.kind[keep], BodyKind::Planet | BodyKind::DwarfPlanet) {
        let rho = (b.m[keep] * b.bulk_density[keep] + b.m[kill] * b.bulk_density[kill]) / total;
        b.bulk_density[keep] = rho;
    }
}

/// Mass-driven promotions of small bodies: rubble to planet, belt body to dwarf planet.
fn promote_by_mass(b: &mut Bodies, keep: usize, cx: f64, cy: f64) {
    if b.origin[keep].is_rubble() && b.m[keep] >= RUBBLE_PROMOTE_MASS {
        b.origin[keep] = BodyOrigin::Spawned;
        let text = format!("{} grew to planet mass", tag(b, keep));
        b.log(EventKind::Merge, text, cx, cy);
    }
    // The threshold is the same one `spawn` uses, so the two paths agree.
    if b.origin[keep].is_belt()
        && b.kind[keep] == BodyKind::Planet
        && classify_small_body(b.m[keep]) == BodyKind::DwarfPlanet
    {
        b.kind[keep] = BodyKind::DwarfPlanet;
    }
}

/// Stellar-evolution ladder. Each transition is followed by a `set_mass`
/// so the radius, density, and radius caching are recomputed against the
/// new kind; leaving a body just above a threshold on the wrong side of
/// the boundary would break later classification.
fn climb_stellar_ladder(b: &mut Bodies, keep: usize, cx: f64, cy: f64) {
    if b.kind[keep] == BodyKind::WhiteDwarf && b.m[keep] > WD_M_MAX {
        b.kind[keep] = BodyKind::NeutronStar;
        b.set_mass(keep, NS_M_MIN * 1.07);
        let text = format!("{} collapsed to neutron star", tag(b, keep));
        b.log(EventKind::Merge, text, cx, cy);
    }
    if b.kind[keep] == BodyKind::Star && b.m[keep] < STAR_M_MIN {
        b.kind[keep] = BodyKind::BrownDwarf;
        b.set_mass(keep, b.m[keep].max(BD_M_MIN));
    }
    if b.kind[keep] == BodyKind::BrownDwarf && b.m[keep] < BD_M_MIN {
        b.kind[keep] = BodyKind::Planet;
        b.bulk_density[keep] = default_bulk_density(BodyKind::Planet, b.m[keep]);
    }
    if b.kind[keep] == BodyKind::BrownDwarf && b.m[keep] >= STAR_M_MIN {
        b.kind[keep] = BodyKind::Star;
        let text = format!("{} ignited as a star", tag(b, keep));
        b.log(EventKind::Merge, text, cx, cy);
    }
}

/// Set the survivor's initial mass and age once its final kind is known.
fn reset_clocks_after_merge(
    b: &mut Bodies,
    keep: usize,
    keep_was_star: bool,
    keep_kind_before: BodyKind,
    kill_kind: BodyKind,
) {
    let kind_after = b.kind[keep];
    if kind_after == BodyKind::Star {
        if !keep_was_star {
            // A brown dwarf that just ignited is a brand-new zero-age star.
            b.m_init[keep] = b.m[keep];
            b.age_gyr[keep] = 0.0;
        }
        return;
    }
    b.m_init[keep] = b.m[keep];
    let same_kind = keep_kind_before == kind_after;
    // A dwarf that swallows a planet or a star keeps its cooling clock.
    // Two dwarfs merging, or a kind transition, restarts it.
    let swallowed_other = same_kind
        && kill_kind != kind_after
        && matches!(kind_after, BodyKind::WhiteDwarf | BodyKind::BrownDwarf);
    if !swallowed_other {
        b.age_gyr[keep] = if same_kind { 0.0 } else { default_age_gyr(kind_after) };
    }
}

/// Logged merge: emits a flash and an event. See `merge_impl`.
pub(super) fn merge(b: &mut Bodies, i: usize, j: usize) -> usize {
    merge_impl(b, i, j, false)
}

/// Resolve one detected pair. The branching order matters:
///
/// 1. Non-planets always merge — stars, dwarfs, and compact remnants do
///    not fragment or bounce in this model.
/// 2. Rubble-pile pairs coalesce, bounce, or pass through depending on
///    relative speed.
/// 3. Fast, oblique impacts hit-and-run instead of merging.
/// 4. Slow, near head-on impacts merge.
/// 5. Anything faster than the merge threshold falls through to the
///    Q*_RD-based disruption model, spawning fragments plus (optionally)
///    a circumplanetary disk.
pub(super) fn collide(b: &mut Bodies, i: usize, j: usize, spawns: &mut Vec<Spawn>) {
    if b.kind[i] != BodyKind::Planet || b.kind[j] != BodyKind::Planet {
        merge(b, i, j);
        return;
    }
    let (rvx, rvy, rvz) = (b.vx[j] - b.vx[i], b.vy[j] - b.vy[i], b.vz[j] - b.vz[i]);
    let vrel = (rvx * rvx + rvy * rvy + rvz * rvz).sqrt();
    let rr = (b.contact_r(i) + b.contact_r(j)).max(1e-6);
    let mtot = b.m[i] + b.m[j];
    let ratio = vrel / (2.0 * G * mtot / rr).sqrt();
    let bb = (impact_offset(b, i, j) / rr).clamp(0.0, 1.0);
    let (big, small) = if b.m[i] >= b.m[j] { (i, j) } else { (j, i) };

    if b.origin[i].is_rubble() && b.origin[j].is_rubble() {
        if ratio < RUBBLE_COALESCE_VRATIO {
            merge(b, i, j);
            b.stats.coalesces += 1;
            return;
        } else if ratio < RUBBLE_BOUNCE_VRATIO {
            rubble_bounce(b, i, j, rr);
            return;
        }
    }

    if bb > HIT_RUN_B && ratio > HIT_RUN_VEL_RATIO {
        hit_and_run(b, big, small, rr, bb, ratio);
        return;
    }
    if ratio < MERGE_VEL_RATIO * (1.0 - MERGE_B_WEIGHT * bb * bb) {
        merge(b, i, j);
        return;
    }

    // Disruption regime.
    let Some(d) = disruption_budget(b, big, small, bb, vrel, mtot, spawns.len()) else {
        merge(b, i, j);
        return;
    };

    // Snapshot the pair's total L before the merge so the post-impact
    // configuration can be rebalanced to match.
    let l_before = pair_l(b, i, j);
    let rel_pos = (b.x[j] - b.x[i], b.y[j] - b.y[i], b.z[j] - b.z[i]);
    let plane = impact_plane(rel_pos, (rvx, rvy, rvz));
    let idg = (b.id[i], b.id[j]);
    // Perform the merge to get a single survivor, then immediately strip
    // it back to the residual mass. The spurious merge event and its
    // statistic are popped; this impact will log its own line.
    let rem = merge(b, i, j);
    b.events.pop_back();
    b.stats.merges -= 1;
    let m_rem = mtot - d.m_ejected;
    b.set_mass(rem, m_rem);
    b.imm[rem] = b.time + IMMUNITY_TIME;
    b.origin[rem] = BodyOrigin::Spawned;

    let (rx, ry) = (b.x[rem], b.y[rem]); // event position: before the recoil
    let s0 = spawns.len();
    let recoil = eject_debris(b, rem, &d, m_rem, &plane, spawns);
    apply_recoil(b, rem, m_rem, &recoil);

    // Spin absorbs whatever L the fragment population could not carry.
    let mut l_after = l_orb(b, rem);
    for s in &spawns[s0..] {
        l_after += s.m * (s.x * s.vy - s.y * s.vx);
    }
    b.spin[rem] = l_before - l_after;

    b.stats.impacts += 1;
    let text = format!(
        "impact #{} × #{} at {:.1}× escape speed, b = {:.2}: {} fragments, {} ejected{}",
        idg.0,
        idg.1,
        ratio,
        bb,
        d.n_fragments,
        mass_str(d.m_ejected),
        if d.m_disk > d.m_erode { " (disk)" } else { "" }
    );
    b.log(EventKind::Impact, text, rx, ry);
}

/// How much mass an impact throws off and into how many fragments.
struct Disruption {
    /// Mass stripped from the colliders, erosion plus disk material, capped.
    m_ejected: f64,
    /// Part of `m_ejected` due to erosion alone (before the cap).
    m_erode: f64,
    /// Mass put into a circumplanetary disk.
    m_disk: f64,
    n_fragments: usize,
}

/// Size the disruption of an impact, or `None` when too little would be
/// ejected for fragments to be worth spawning (the pair then simply merges).
///
/// `alpha` and `m_int` define the *interacting* part of the small body;
/// `mu_a` is the reduced mass of that sub-collision. `x` is the impact
/// energy in units of `Q*_RD`; `lr` is the surviving (largest-remnant) mass
/// fraction after the impact.
fn disruption_budget(
    b: &Bodies,
    big: usize,
    small: usize,
    bb: f64,
    vrel: f64,
    mtot: f64,
    pending_spawns: usize,
) -> Option<Disruption> {
    let alpha = interact_frac(b.contact_r(big), b.contact_r(small), bb).max(1e-3);
    let m_int = alpha * b.m[small];
    let mu_a = b.m[big] * m_int / (b.m[big] + m_int);
    let x = 0.5 * mu_a * vrel * vrel / mtot / q_star_rd(mtot, b.m[small] / b.m[big]);
    let lr = if x <= 1.8 {
        1.0 - 0.5 * x
    } else {
        0.1 * (x / 1.8).powf(-1.5)
    };
    let m_erode = mtot * (1.0 - lr);
    let m_disk = if b.len() <= DISK_MAX_N {
        DISK_MASS_FRAC * m_int * 4.0 * bb * bb * (1.0 - bb * bb)
    } else {
        0.0
    };
    let cap = (DEBRIS_MAX_FRAC_HEADON + (DEBRIS_MAX_FRAC - DEBRIS_MAX_FRAC_HEADON) * bb) * mtot;
    let m_ejected = (m_erode + m_disk).min(cap);
    let n_raw = FRAG_COUNT_BASE
        + FRAG_COUNT_SCALE * x.powf(FRAG_COUNT_EXP)
        + FRAG_COUNT_DISK_GAIN * (m_disk / DEBRIS_MIN_MASS).ln_1p();
    // Clamp fragment count to: model, per-event maximum, what the ejected
    // mass can actually pay for, and remaining body-slot budget (+1 for
    // the survivor).
    let n_fragments = (n_raw.round() as usize)
        .min(FRAG_COUNT_MAX)
        .min((m_ejected / DEBRIS_MIN_MASS) as usize)
        .min(room(b, pending_spawns) + 1);
    if m_ejected < DEBRIS_MIN_MASS || n_fragments < 2 {
        return None;
    }
    Some(Disruption { m_ejected, m_erode, m_disk, n_fragments })
}

/// Orthonormal frame of the impact plane plus the sense of rotation.
struct ImpactPlane {
    e1: (f64, f64, f64),
    e2: (f64, f64, f64),
    sense: f64,
}

/// The impact plane is spanned by the relative position and velocity. Planar
/// pairs use the xy axes and a ±1 sense (as before); otherwise the frame is
/// right-handed about the orbit normal, so the sense is always +1.
fn impact_plane(rel_pos: (f64, f64, f64), rel_vel: (f64, f64, f64)) -> ImpactPlane {
    let (px0, py0, pz0) = rel_pos;
    let (rvx, rvy, rvz) = rel_vel;
    let (hxv, hyv, hzv) = (py0 * rvz - pz0 * rvy, pz0 * rvx - px0 * rvz, px0 * rvy - py0 * rvx);
    if hxv == 0.0 && hyv == 0.0 {
        return ImpactPlane {
            e1: (1.0, 0.0, 0.0),
            e2: (0.0, 1.0, 0.0),
            sense: if hzv >= 0.0 { 1.0 } else { -1.0 },
        };
    }
    let hn = (hxv * hxv + hyv * hyv + hzv * hzv).sqrt();
    let (nhx, nhy, nhz) = (hxv / hn, hyv / hn, hzv / hn);
    let ln = (nhx * nhx + nhy * nhy).sqrt();
    let e1 = (-nhy / ln, nhx / ln, 0.0);
    let e2 = (nhy * e1.2 - nhz * e1.1, nhz * e1.0 - nhx * e1.2, nhx * e1.1 - nhy * e1.0);
    ImpactPlane { e1, e2, sense: 1.0 }
}

/// Total fragment momentum and mass-weighted offset, so the survivor can be
/// pushed the opposite way afterwards.
struct Recoil {
    px: f64,
    py: f64,
    pz: f64,
    sx: f64,
    sy: f64,
    sz: f64,
}

/// Throw the ejected mass off the survivor `rem` as fragments (pushed onto
/// `spawns`) plus a dust deposit. Draw order from `b.rng` is part of the
/// model's determinism: masses, then the base angle, then per fragment the
/// disk flag, angle jitter and its kick.
fn eject_debris(
    b: &mut Bodies,
    rem: usize,
    d: &Disruption,
    m_rem: f64,
    plane: &ImpactPlane,
    spawns: &mut Vec<Spawn>,
) -> Recoil {
    let (e1, e2, sense) = (plane.e1, plane.e2, plane.sense);
    let (n, md, m_disk) = (d.n_fragments, d.m_ejected, d.m_disk);
    let ms = frag_masses(&mut b.rng, n, md);
    // Dust source term: a bonus population, see `DUST_YIELD_FRAC`.
    let (dust_x, dust_y, dust_z) = (b.x[rem], b.y[rem], b.z[rem]);
    let (dust_vx, dust_vy, dust_vz) = (b.vx[rem], b.vy[rem], b.vz[rem]);
    dust::deposit(
        b,
        dust_x,
        dust_y,
        dust_z,
        dust_vx,
        dust_vy,
        dust_vz,
        md * DUST_YIELD_FRAC,
        None,
        1.0,
    );
    let r_rem = b.contact_r(rem);
    let v_esc = (2.0 * G * m_rem / r_rem.max(1e-6)).sqrt();
    let disk_share = (m_disk / md).clamp(0.0, 1.0);
    let base = b.rng.next_f64() * std::f64::consts::TAU;
    let (rx, ry, rz) = (b.x[rem], b.y[rem], b.z[rem]);
    let (rvx0, rvy0, rvz0) = (b.vx[rem], b.vy[rem], b.vz[rem]);
    let mut recoil = Recoil { px: 0.0, py: 0.0, pz: 0.0, sx: 0.0, sy: 0.0, sz: 0.0 };
    for (k, &mk) in ms.iter().enumerate() {
        // Disk fragments inherit the survivor's orbital frame; ejecta get
        // a radial kick scaled by `v_esc`. `disk_share` sets the fraction
        // of fragments in each population.
        let is_disk = b.rng.next_f64() < disk_share;
        let ang = base
            + (k as f64) * std::f64::consts::TAU / n as f64
            + (b.rng.next_f64() - 0.5) * DEBRIS_ANGLE_JIT;
        let (cs, sn) = (ang.cos(), ang.sin());
        let ux = cs * e1.0 + sn * e2.0;
        let uy = cs * e1.1 + sn * e2.1;
        let uz = cs * e1.2 + sn * e2.2;
        let tx = -sn * e1.0 + cs * e2.0;
        let ty = -sn * e1.1 + cs * e2.1;
        let tz = -sn * e1.2 + cs * e2.2;
        let (dist, wx, wy, wz);
        if is_disk {
            dist = r_rem * (DISK_R_MIN + DISK_R_SPAN * b.rng.next_f64()) + planet_true_r(mk);
            let vc = (G * m_rem / dist).sqrt() * (0.9 + 0.2 * b.rng.next_f64());
            wx = sense * tx * vc;
            wy = sense * ty * vc;
            wz = sense * tz * vc;
        } else {
            dist = (r_rem + planet_true_r(mk)) * (1.0 + DEBRIS_GAP_FRAC);
            let kick = v_esc * (DEBRIS_KICK_MIN + DEBRIS_KICK_SPAN * b.rng.next_f64());
            wx = ux * kick;
            wy = uy * kick;
            wz = uz * kick;
        }
        recoil.px += mk * wx;
        recoil.py += mk * wy;
        recoil.pz += mk * wz;
        recoil.sx += mk * ux * dist;
        recoil.sy += mk * uy * dist;
        recoil.sz += mk * uz * dist;
        spawns.push(Spawn {
            x: rx + ux * dist,
            y: ry + uy * dist,
            z: rz + uz * dist,
            vx: rvx0 + wx,
            vy: rvy0 + wy,
            vz: rvz0 + wz,
            m: mk,
            kind: BodyKind::Planet,
            origin: BodyOrigin::ImpactDebris,
            // Fragments inherit the survivor's surface class so a rocky
            // impact yields rocky debris and an icy impact yields icy
            // debris. `merge` does not touch `class`, so `b.class[rem]`
            // is still the pre-impact class of the dominant body.
            class: b.class[rem],
        });
    }
    recoil
}

/// Momentum- and centre-of-mass-conserving recoil of the survivor.
fn apply_recoil(b: &mut Bodies, rem: usize, m_rem: f64, r: &Recoil) {
    b.x[rem] -= r.sx / m_rem;
    b.y[rem] -= r.sy / m_rem;
    b.z[rem] -= r.sz / m_rem;
    b._px[rem] = b.x[rem];
    b._py[rem] = b.y[rem];
    b._pz[rem] = b.z[rem];
    b.vx[rem] -= r.px / m_rem;
    b.vy[rem] -= r.py / m_rem;
    b.vz[rem] -= r.pz / m_rem;
}
