//! Debris-disk coalescence, Roche limit, tidal break-up, escape culling.

use super::*;

/// Merge rubble fragments that are gravitationally bound to each other
/// and share the same parent host.
///
/// Each fragment must survive `DEBRIS_CAPTURE_STEPS` consecutive steps
/// inside its pair partner's Hill sphere with sub-escape relative
/// velocity before the pair is merged. Fragments are spatially binned
/// with a grid whose cell size is `DEBRIS_SEARCH_RADII × max(contact_radius)`,
/// so only the 3×3 neighbourhood of each cell needs scanning.
pub(super) fn coalesce_debris(b: &mut Bodies) {
    let n = b.len();
    if n < 2 {
        return;
    }
    // Quick pass to decide whether the expensive path is worth taking.
    // Simulations with no rubble, or exactly one fragment, exit here.
    let n_eligible = (0..n).filter(|&i| can_coalesce(b, i)).count();
    if n_eligible < 2 {
        // No pair can form, so no counters can advance — but stale
        // counters from a previous step must still be cleared so a
        // fragment that leaves the neighbourhood resets its streak.
        reset_rubble_counters(b, |_| true);
        return;
    }

    let hosts = host_list(b);
    let cands = Candidates::collect(b);

    // Rubble bodies that were not selected still need their counter
    // cleared, so a fragment that temporarily leaves the size window
    // doesn't resume a stale capture streak when it re-enters.
    reset_rubble_counters(b, |i| cands.position(i).is_none());

    let (pairs, in_capture) = find_capture_pairs(b, &hosts, &cands);
    advance_capture_streaks(b, &cands.eligible, &in_capture);
    if pairs.is_empty() {
        return;
    }
    if merge_captured_pairs(b, pairs) {
        b.compact();
    }
}

/// A live rubble fragment small enough and old enough to coalesce.
fn can_coalesce(b: &Bodies, i: usize) -> bool {
    !b.dead[i]
        && b.origin[i].is_rubble()
        && b.m[i] < DEBRIS_COALESCE_MAX_MASS
        && b.time >= b.imm[i]
}

/// Zero the capture streak of every live rubble body for which `select`
/// returns true.
fn reset_rubble_counters(b: &mut Bodies, select: impl Fn(usize) -> bool) {
    for i in 0..b.len() {
        if !b.dead[i] && b.origin[i].is_rubble() && select(i) {
            b.coalesce_counter[i] = 0;
        }
    }
}

const NOT_ELIGIBLE: u32 = u32::MAX;

/// The fragments that may coalesce this step, collected once so every later
/// pass iterates a compact list instead of scanning all bodies.
struct Candidates {
    /// Body indices of the eligible fragments.
    eligible: Vec<usize>,
    /// Reverse map: body index → position in `eligible`, or `NOT_ELIGIBLE`.
    body_to_eligible: Vec<u32>,
    /// Cached parent host of each eligible fragment.
    parents: Vec<Option<usize>>,
    /// Distance from each eligible fragment to its parent, computed once
    /// and reused by every pair evaluation the fragment takes part in.
    d_to_parent: Vec<f64>,
}

impl Candidates {
    fn collect(b: &Bodies) -> Self {
        let n = b.len();
        debug_assert!(n <= u32::MAX as usize);
        let eligible: Vec<usize> = (0..n).filter(|&i| can_coalesce(b, i)).collect();
        let mut body_to_eligible: Vec<u32> = vec![NOT_ELIGIBLE; n];
        for (pos, &i) in eligible.iter().enumerate() {
            body_to_eligible[i] = pos as u32;
        }
        let parents: Vec<Option<usize>> = eligible
            .iter()
            .map(|&i| b.parent_cache.get(i).copied().flatten())
            .collect();
        let d_to_parent: Vec<f64> = eligible
            .iter()
            .zip(&parents)
            .map(|(&i, parent)| match parent {
                Some(pi) => {
                    let pi = *pi;
                    let dx = b.x[i] - b.x[pi];
                    let dy = b.y[i] - b.y[pi];
                    let dz = b.z[i] - b.z[pi];
                    (dx * dx + dy * dy + dz * dz).sqrt()
                }
                None => 0.0,
            })
            .collect();
        Self { eligible, body_to_eligible, parents, d_to_parent }
    }

    /// Position of body `i` in `eligible`, if it is a candidate.
    fn position(&self, i: usize) -> Option<usize> {
        match self.body_to_eligible[i] {
            NOT_ELIGIBLE => None,
            pos => Some(pos as usize),
        }
    }
}

/// Find fragment pairs that are bound to each other right now. Returns the
/// pairs and, per body, whether it is in at least one.
///
/// Fragments are binned in a grid whose cell is sized to the largest
/// fragment's interaction radius, so the 3×3 neighbourhood of a cell always
/// covers the capture range. Fragments without a parent cannot be captured
/// and are left out.
fn find_capture_pairs(
    b: &Bodies,
    hosts: &[usize],
    cands: &Candidates,
) -> (Vec<(usize, usize)>, Vec<bool>) {
    let eligible = &cands.eligible;
    let max_cr = eligible.iter().map(|&i| b.contact_r(i)).fold(0.0f64, f64::max);
    let cell = (DEBRIS_SEARCH_RADII * max_cr).max(1e-4);

    let mut keys: Vec<(u64, u32)> = Vec::with_capacity(eligible.len());
    for (pos, &i) in eligible.iter().enumerate() {
        if cands.parents[pos].is_none() {
            continue;
        }
        let cx = (b.x[i] / cell).floor() as i64;
        let cy = (b.y[i] / cell).floor() as i64;
        keys.push((cell_key(cx, cy), i as u32));
    }
    keys.par_sort_unstable();

    let mut in_capture = vec![false; b.len()];
    let mut pairs: Vec<(usize, usize)> = Vec::new();

    for (pos_i, &i) in eligible.iter().enumerate() {
        let Some(pi) = cands.parents[pos_i] else { continue };

        // Quantities shared by every candidate partner j for this i.
        let m_p = effective_host_mass(b, hosts, i, pi);
        let d_i = cands.d_to_parent[pos_i];
        let cx = (b.x[i] / cell).floor() as i64;
        let cy = (b.y[i] / cell).floor() as i64;

        for ddx in -1..=1i64 {
            for ddy in -1..=1i64 {
                let key = cell_key(cx + ddx, cy + ddy);
                let start = keys.partition_point(|e| e.0 < key);
                for e in keys[start..].iter().take_while(|e| e.0 == key) {
                    let j = e.1 as usize;
                    // Each unordered pair is visited once.
                    if j <= i {
                        continue;
                    }
                    let Some(pos_j) = cands.position(j) else { continue };
                    let Some(pj) = cands.parents[pos_j] else { continue };
                    // Coalescence only occurs between siblings: fragments
                    // must share a parent host.
                    if pi != pj {
                        continue;
                    }
                    let d_j = cands.d_to_parent[pos_j];
                    if is_bound_pair(b, i, j, m_p, (d_i + d_j) * 0.5) {
                        in_capture[i] = true;
                        in_capture[j] = true;
                        pairs.push((i, j));
                    }
                }
            }
        }
    }
    (pairs, in_capture)
}

/// Whether fragments `i` and `j` are close and slow enough to stay bound
/// around a host of mass `m_p` at mean orbital distance `d_mean`.
fn is_bound_pair(b: &Bodies, i: usize, j: usize, m_p: f64, d_mean: f64) -> bool {
    let dx = b.x[j] - b.x[i];
    let dy = b.y[j] - b.y[i];
    let dz = b.z[j] - b.z[i];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    // Mutual Hill radius of the pair around the host, evaluated at the mean
    // orbital distance. The separation must lie well inside it for the pair
    // to remain bound long enough to merge.
    let m_ij = b.m[i] + b.m[j];
    let r_hill = (m_ij / (3.0 * m_p)).cbrt() * d_mean;
    if dist >= DEBRIS_CAPTURE_HILL_FRAC * r_hill {
        return false;
    }

    // Relative speed must fall below the escape speed of the pair,
    // otherwise the encounter is unbound.
    let dvx = b.vx[j] - b.vx[i];
    let dvy = b.vy[j] - b.vy[i];
    let dvz = b.vz[j] - b.vz[i];
    let vrel = (dvx * dvx + dvy * dvy + dvz * dvz).sqrt();
    let v_esc = (2.0 * G * m_ij / dist.max(1e-12)).sqrt();
    vrel < DEBRIS_CAPTURE_VEL_RATIO * v_esc
}

/// Advance the capture streak for fragments currently in capture and reset
/// the rest. Saturating add guards against overflow in very long runs.
fn advance_capture_streaks(b: &mut Bodies, eligible: &[usize], in_capture: &[bool]) {
    for &i in eligible {
        b.coalesce_counter[i] = if in_capture[i] {
            b.coalesce_counter[i].saturating_add(1)
        } else {
            0
        };
    }
}

/// Merge the pairs whose members have both been bound for long enough.
/// Returns whether anything merged (the caller then compacts the arrays).
fn merge_captured_pairs(b: &mut Bodies, mut pairs: Vec<(usize, usize)>) -> bool {
    let persisted = |b: &Bodies, i: usize| b.coalesce_counter[i] >= DEBRIS_CAPTURE_STEPS;
    pairs.retain(|&(i, j)| persisted(b, i) && persisted(b, j));

    let mut merged_any = false;
    for (i, j) in pairs {
        // A body may appear in several pairs; earlier merges in this pass
        // can invalidate later ones, so re-check membership and streak.
        if b.dead[i] || b.dead[j] || !persisted(b, i) || !persisted(b, j) {
            continue;
        }
        merge_impl(b, i, j, /*silent=*/ true);
        b.stats.coalesces += 1;
        merged_any = true;
    }
    merged_any
}

pub fn density_gcc(b: &Bodies, i: usize) -> f64 {
    if matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
        b.bulk_density[i]
    } else {
        EARTH_RHO_GCC * b.m[i] * (R_EARTH_GM / b.r[i].max(1e-12)).powi(3)
    }
}

/// Interpolate between fluid (w = 0) and rigid (w = 1) Roche behaviour
/// based on the body's bulk density.
pub fn rigidity(rho_gcc: f64) -> f64 {
    ((rho_gcc - ROCHE_RHO_FLUID) / (ROCHE_RHO_RIGID - ROCHE_RHO_FLUID)).clamp(0.0, 1.0)
}

pub fn roche_limit(b: &Bodies, i: usize, p: usize) -> f64 {
    let w = rigidity(density_gcc(b, i));
    let coeff = ROCHE_FLUID * (1.0 - w) + ROCHE_RIGID * w;
    coeff * b.r[i] * (b.m[p] / b.m[i]).cbrt()
}

/// Break up planet-class bodies that enter the Roche limit of a
/// sufficiently massive primary.
pub(super) fn tidal_breakup(b: &mut Bodies) {
    let n = b.len();

    // Candidate primaries: stars, plus any planet or dwarf planet
    // heavy enough to dismember a smaller companion.
    let prims: Vec<usize> = (0..n)
        .filter(|&p| {
            is_host_kind(b.kind[p])
                || (matches!(b.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet)
                    && b.m[p] >= GIANT_MASS)
        })
        .collect();
    if prims.is_empty() {
        b.disrupting.iter_mut().for_each(|f| *f = false);
        return;
    }
    let t_now = b.time;

    // Iterate in reverse so a disruption that appends fragments to
    // `b` doesn't shift indices we still need to visit.
    for i in (0..n).rev() {
        if b.dead[i]
            || !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
            || b.origin[i].is_rubble()
            || b.m[i] < TIDAL_MIN_MASS
            || t_now < b.imm[i]
        {
            continue;
        }

        let tb = b.r[i];

        // Pre-filter: skip primaries whose Hill-scaled Roche condition
        // cannot be met within the currently searched distance. k6 is
        // the cube of the fluid Roche coefficient times r_i, which
        // bounds the geometric part of the criterion.
        let kf = ROCHE_FLUID * tb;
        let k2 = kf * kf;
        let k6 = k2 * k2 * k2;

        // Rigidity-weighted coefficient for this body. Only depends
        // on `i`, so compute once before looping over primaries.
        let w = rigidity(density_gcc(b, i));
        let coeff = ROCHE_FLUID * (1.0 - w) + ROCHE_RIGID * w;
        let k_self = coeff * tb; // coeff * b.r[i]
        let mi = b.m[i];

        let mut inside = false;
        for &p in &prims {
            // Skip self, dead primaries, and primaries too light to
            // meaningfully perturb body i.
            if p == i || b.dead[p] || b.m[p] < TIDAL_MASS_RATIO * mi {
                continue;
            }
            let dx = b.x[i] - b.x[p];
            let dy = b.y[i] - b.y[p];
            let dz = b.z[i] - b.z[p];
            let d2 = dx * dx + dy * dy + dz * dz;
            let q = b.m[p] / mi;

            // Cheap early-out: d^6 vs. the Roche denominator.
            if d2 * d2 * d2 >= k6 * q * q {
                continue;
            }

            // Full Roche radius for this (i, p) pair.
            let roche = k_self * q.cbrt();
            if d2 < roche * roche {
                inside = true;
                b.disrupting[i] = true;

                // Only dismember on the outward leg of the orbit
                // (rdot >= 0); on the inbound leg the body is still
                // approaching and disruption is deferred.
                let rdot = dx * (b.vx[i] - b.vx[p]) + dy * (b.vy[i] - b.vy[p]) + dz * (b.vz[i] - b.vz[p]);
                if rdot >= 0.0 {
                    let _ = disrupt(b, i, p);
                }
                break;
            }
        }

        // Clear the flag for bodies that have moved back out of the
        // Roche limit (unless they were disrupted this step).
        if !inside && !b.dead[i] {
            b.disrupting[i] = false;
        }
    }

    if b.any_dead() {
        b.compact();
    }
}

/// Dismember body `i` under the tide of primary `p`, producing
/// 4–12 fragments strung along the primary-facing axis with a
/// linear velocity kick.
pub(super) fn disrupt(b: &mut Bodies, i: usize, p: usize) -> bool {
    let mi = b.m[i];

    // Fragment count scales logarithmically with mass, clamped so
    // every disruption yields a visible debris string.
    let n = ((DISRUPT_FRAG_BASE + DISRUPT_FRAG_LOG * (mi + 1.0).log10()).round() as usize)
        .clamp(4, 12)
        .min(room(b, 0) + 1);
    if n < 3 {
        return false;
    }

    mark_energy(b);
    let l_before = l_orb(b, i) + b.spin[i];
    let text = format!("{} torn apart by {}", tag(b, i), tag(b, p));

    let axes = tidal_axes(b, i, p);
    // Draw fragment mass weights first so the geometry (fragment
    // spacing, radii) can be sized to the largest fragment.
    let (ms, rmax) = fragment_masses(b, mi, n);
    let origin = (b.x[i], b.y[i], b.z[i]);
    let bvel = (b.vx[i], b.vy[i], b.vz[i]);
    let prim_r = b.contact_r(p);

    let pieces = build_fragments(b, i, p, &axes, &ms, rmax);
    b.dead[i] = true;
    spawn_fragments(b, pieces, bvel, l_before);

    flash_at(b, origin.0, origin.1, origin.2, (prim_r * 0.5).max(0.05));
    b.stats.tidal += 1;
    b.log(EventKind::Tidal, text, origin.0, origin.1);
    true
}

/// Unit vector from primary to body, and a unit tangent to it.
struct TidalAxes {
    u: (f64, f64, f64),
    t: (f64, f64, f64),
}

/// Tangent: in-plane (-uy, ux) for a planar pair, otherwise the part of the
/// relative velocity perpendicular to the tidal axis.
fn tidal_axes(b: &Bodies, i: usize, p: usize) -> TidalAxes {
    let d = ((b.x[i] - b.x[p]).powi(2) + (b.y[i] - b.y[p]).powi(2) + (b.z[i] - b.z[p]).powi(2))
        .sqrt()
        .max(1e-12);
    let (ux, uy, uz) = ((b.x[i] - b.x[p]) / d, (b.y[i] - b.y[p]) / d, (b.z[i] - b.z[p]) / d);
    let (rvx, rvy, rvz) = (b.vx[i] - b.vx[p], b.vy[i] - b.vy[p], b.vz[i] - b.vz[p]);
    let t = if uz == 0.0 && rvz == 0.0 {
        (-uy, ux, 0.0)
    } else {
        let vr = rvx * ux + rvy * uy + rvz * uz;
        let (qx, qy, qz) = (rvx - vr * ux, rvy - vr * uy, rvz - vr * uz);
        let qn = (qx * qx + qy * qy + qz * qz).sqrt();
        if qn > 1e-12 {
            (qx / qn, qy / qn, qz / qn)
        } else {
            let pn = (ux * ux + uy * uy).sqrt().max(1e-12);
            (-uy / pn, ux / pn, 0.0)
        }
    };
    TidalAxes { u: (ux, uy, uz), t }
}

/// Random fragment masses summing to `mi`, and the largest fragment radius.
fn fragment_masses(b: &mut Bodies, mi: f64, n: usize) -> (Vec<f64>, f64) {
    let mut ms: Vec<f64> = (0..n).map(|_| DEBRIS_WEIGHT_MIN + b.rng.next_f64()).collect();
    let wsum: f64 = ms.iter().sum();
    let mut rmax: f64 = 0.0;
    for m in ms.iter_mut() {
        *m = mi * *m / wsum;
        rmax = rmax.max(planet_true_r(*m));
    }
    (ms, rmax)
}

/// Fragments strung along the tidal axis. Positions and kicks are symmetric
/// about the original body's centre. Draws two jitter numbers per fragment.
fn build_fragments(
    b: &mut Bodies,
    i: usize,
    p: usize,
    axes: &TidalAxes,
    ms: &[f64],
    rmax: f64,
) -> Vec<Spawn> {
    let n = ms.len();
    let (ux, uy, uz) = axes.u;
    let (tx, ty, tz) = axes.t;
    let step = DISRUPT_STEP * rmax;
    let half = (((n - 1) as f64) / 2.0 * step).max(1e-6);
    let vk = DISRUPT_KICK * (G * b.m[i] / b.r[i].max(1e-6)).sqrt();
    let (bx, by, bz, bvx, bvy, bvz) = (b.x[i], b.y[i], b.z[i], b.vx[i], b.vy[i], b.vz[i]);
    let prim_r = b.contact_r(p);
    let prim = (b.x[p], b.y[p], b.z[p]);

    let mut pieces: Vec<Spawn> = Vec::with_capacity(n);
    for (k, &mk) in ms.iter().enumerate() {
        let s = (k as f64 - (n - 1) as f64 / 2.0) * step;
        let kick = vk * s / half;

        // Start position: offset along the tidal axis plus a small perpendicular jitter.
        let jx = b.rng.next_f64() - DISRUPT_JITTER;
        let jy = b.rng.next_f64() - DISRUPT_JITTER;
        let pos = (bx + ux * s + tx * jx * rmax, by + uy * s + ty * jy * rmax, bz + uz * s + tz * jy * rmax);
        let (fx, fy, fz) = keep_outside_primary(pos, prim, prim_r + 1.5 * planet_true_r(mk), axes.u);

        pieces.push(Spawn {
            x: fx,
            y: fy,
            z: fz,
            vx: bvx + ux * kick,
            vy: bvy + uy * kick,
            vz: bvz + uz * kick,
            m: mk,
            kind: BodyKind::Planet,
            origin: BodyOrigin::TidalDebris,
            // Fragments inherit the composition of the body they came from.
            class: b.class[i],
        });
    }
    pieces
}

/// Push a fragment that would spawn inside the primary back out to `qmin`.
fn keep_outside_primary(
    pos: (f64, f64, f64),
    prim: (f64, f64, f64),
    qmin: f64,
    axis: (f64, f64, f64),
) -> (f64, f64, f64) {
    let (mut qx, mut qy, mut qz) = (pos.0 - prim.0, pos.1 - prim.1, pos.2 - prim.2);
    let mut qd = (qx * qx + qy * qy + qz * qz).sqrt();
    if qd >= qmin {
        return pos;
    }
    if qd < 1e-9 {
        (qx, qy, qz) = axis;
        qd = 1.0;
    }
    (prim.0 + qx / qd * qmin, prim.1 + qy / qd * qmin, prim.2 + qz / qd * qmin)
}

/// Add the fragments, correcting each velocity so the cloud's centre of mass
/// moves with the original body (`bvel`). The orbital angular momentum the
/// fragments lost goes into their spin.
fn spawn_fragments(b: &mut Bodies, pieces: Vec<Spawn>, bvel: (f64, f64, f64), l_before: f64) {
    // Momentum kick of the cloud relative to the original body.
    let (mut pvx, mut pvy, mut pvz, mut mt) = (0.0, 0.0, 0.0, 0.0);
    for s in &pieces {
        pvx += s.m * (s.vx - bvel.0);
        pvy += s.m * (s.vy - bvel.1);
        pvz += s.m * (s.vz - bvel.2);
        mt += s.m;
    }
    let t_imm = b.time + IMMUNITY_TIME;
    let first = b.len();
    let mut l_after = 0.0;
    for s in pieces {
        let (vx, vy, vz) = (s.vx - pvx / mt, s.vy - pvy / mt, s.vz - pvz / mt);
        let k = b.push(s.x, s.y, vx, vy, s.m, s.kind);
        b.z[k] = s.z;
        b.vz[k] = vz;
        b._pz[k] = s.z;
        b.imm[k] = t_imm;
        b.origin[k] = BodyOrigin::TidalDebris;
        b.class[k] = s.class;
        l_after += s.m * (s.x * vy - s.y * vx);
    }
    let dl = l_before - l_after;
    for k in first..b.len() {
        b.spin[k] = dl * b.m[k] / mt;
    }
}

/// Sublimate ice from comet bodies near their host star.
///
/// A comet (`BodyOrigin::Comet`) is active only when it is inside its
/// host's snow line AND has `BodyClass::Icy` — an icy body outside the
/// snow line is dormant, and a rocky body is inert regardless of
/// distance. Near perihelion the comet loses mass to sublimation;
/// that mass goes to the nearest dust field. Below `COMET_DEAD_MASS`
/// the comet is removed — it has exhausted its volatiles.
///
/// Rate model: `dm = rate × m × dt` with `rate ∝ flux = L / r²`. At
/// 1 AU from a Sun-like star the loss is `COMET_SUBLIMATE_FLUX_COEFF`
/// per time unit, capped at `COMET_MAX_FRAC_PER_STEP × m` per step.
///
/// No RNG is drawn, so the RNG stream is unchanged from before this
/// feature and every downstream test that hashes on belt-body
/// positions is unaffected.
pub(super) fn comet_sublimate(b: &mut Bodies, dt: f64) {
    // Fast path — most steps have no comets at all.
    if !b.origin.iter().any(|&o| o == BodyOrigin::Comet) {
        return;
    }
    let n = b.len();
    let hosts = host_list(b);
    if hosts.is_empty() {
        return;
    }
    let t_now = b.time;

    // Host luminosity depends only on the host, so evaluate it once
    // per host rather than once per comet.
    let host_lum: Vec<f64> = hosts
        .iter()
        .map(|&h| luminosity_of(b, h).max(0.0))
        .collect();

    // Deferred operations, so the borrow checker stays happy across
    // the `dust::deposit` calls below.
    let mut mass_updates: Vec<(usize, f64)> = Vec::new();
    let mut to_deposit: Vec<(usize, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)> = Vec::new();
    let mut to_kill: Vec<usize> = Vec::new();

    for i in 0..n {
        if b.dead[i] || b.origin[i] != BodyOrigin::Comet {
            continue;
        }
        if b.class[i] != BodyClass::Icy {
            continue;
        }
        if t_now < b.imm[i] {
            continue;
        }

        // Nearest host determines both the snow-line test and the
        // sublimation flux.
        let mut r_min = f64::INFINITY;
        let mut h_idx = 0usize;
        for (k, &hk) in hosts.iter().enumerate() {
            let d = (b.x[i] - b.x[hk]).hypot(b.y[i] - b.y[hk]).hypot(b.z[i] - b.z[hk]);
            if d < r_min {
                r_min = d;
                h_idx = k;
            }
        }
        let h = hosts[h_idx];
        let lum = host_lum[h_idx];
        if lum <= 0.0 {
            continue;
        }

        let r_au = r_min / au_in_gm();
        let snow_line_au = 2.7 * lum.sqrt();
        if r_au >= snow_line_au {
            continue;
        }

        let flux = lum / (r_au * r_au).max(1e-9);
        let rate = COMET_SUBLIMATE_FLUX_COEFF * flux;
        let dm = (rate * b.m[i] * dt).min(COMET_MAX_FRAC_PER_STEP * b.m[i]);
        if dm <= 0.0 {
            continue;
        }

        // Anti-sunward unit vector for the dust's initial momentum.
        let (x, y, z, vx, vy, vz) = (b.x[i], b.y[i], b.z[i], b.vx[i], b.vy[i], b.vz[i]);
        let dx = b.x[i] - b.x[h];
        let dy = b.y[i] - b.y[h];
        let dz = b.z[i] - b.z[h];
        let r = dx.hypot(dy).hypot(dz).max(1e-9);
        let (ux, uy, uz) = (dx / r, dy / r, dz / r);

        to_deposit.push((i, x, y, z, vx, vy, vz, dm, ux, uy, uz));
        let new_m = b.m[i] - dm;
        if new_m < COMET_DEAD_MASS {
            to_kill.push(i);
        } else {
            mass_updates.push((i, new_m));
        }
    }

    if !mass_updates.is_empty() || !to_kill.is_empty() {
        mark_energy(b);
    }
    for (i, m) in mass_updates {
        b.set_mass(i, m);
    }
    for (i, x, y, z, vx, vy, vz, dm, ux, uy, uz) in to_deposit {
        if b.comet_grains[i] as usize >= COMET_GRAIN_CAP {
            continue;
        }
        let added = dust::deposit(b, x, y, z, vx, vy, vz, dm, Some((ux, uy, uz)), 0.005);
        b.comet_grains[i] = b.comet_grains[i].saturating_add(added as u32);
    }

    let any_killed = !to_kill.is_empty();
    for i in to_kill {
        let text = format!("{} exhausted its volatiles", tag(b, i));
        let (x, y) = (b.x[i], b.y[i]);
        b.dead[i] = true;
        b.log(EventKind::Comet, text, x, y);
    }
    if any_killed {
        b.compact();
    }
}

/// Remove bodies that have escaped the system, promoting heavy ejectees
/// to rogues along the way.
pub(super) fn cull_escaped(b: &mut Bodies) {
    let n = b.len();
    if n == 0 {
        return;
    }

    // Total mass of every non-planetary body — the barycentre that
    // determines whether a body beyond CULL_DIST is unbound.
    let mut prim_m = 0.0;
    for i in 0..n {
        if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
            prim_m += b.m[i];
        }
    }
    let cd2 = CULL_DIST * CULL_DIST;
    let hcd2 = HARD_CULL_DIST * HARD_CULL_DIST;
    let v_esc_sq = 2.0 * G * prim_m;
    let hosts: Vec<usize> = (0..n)
        .filter(|&i| !b.dead[i] && is_host_kind(b.kind[i]))
        .collect();

    let mut gone: Vec<usize> = Vec::new();
    let mut promoted: Vec<usize> = Vec::new();
    for i in 0..n {
        if b.dead[i] || !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
            continue;
        }
        let d2 = b.x[i] * b.x[i] + b.y[i] * b.y[i] + b.z[i] * b.z[i];
        if d2 < cd2 {
            continue;
        }
        // A planet near any host star is not an escapee, even if that
        // star is far from the origin (wide companions, e.g. Proxima).
        let near_host = hosts.iter().any(|&h| {
            let dx = b.x[i] - b.x[h];
            let dy = b.y[i] - b.y[h];
            let dz = b.z[i] - b.z[h];
            dx * dx + dy * dy + dz * dz < cd2
        });
        if near_host {
            continue;
        }
        let v2 = b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i] + b.vz[i] * b.vz[i];

        // Hard-cull everything past the outer boundary. This applies
        // regardless of origin, motion, or mass — it's the final
        // safety net against unbounded accumulation.
        if d2 > hcd2 {
            gone.push(i);
            continue;
        }

        // Past CULL_DIST but inside HARD_CULL_DIST: cull unbound
        // outbound bodies unless they are rogues (which we track the
        // full distance), or heavy enough to *become* rogues.
        let d = d2.sqrt();
        let unbound_out = v2 > v_esc_sq / d && b.x[i] * b.vx[i] + b.y[i] * b.vy[i] + b.z[i] * b.vz[i] > 0.0;
        if !unbound_out {
            continue;
        }

        match b.origin[i] {
            BodyOrigin::RoguePlanet => {
                // A rogue is tracked all the way to HARD_CULL_DIST
                // so the user can watch it cross the outer system.
            }
            BodyOrigin::Interstellar => {
                // ISOs cull at CULL_DIST — same threshold as the
                // interstellar-object feature established. A real
                // ISO is unobservable within a few tens of AU;
                // tracking it to 6,680 AU would show nothing new.
                gone.push(i);
            }
            _ => {
                if b.m[i] >= ROGUE_MIN_MASS {
                    // Heavy ejectee → promote to rogue, keep tracking.
                    promoted.push(i);
                } else {
                    // Small ejectee → cull as before.
                    gone.push(i);
                }
            }
        }
    }

    if gone.is_empty() && promoted.is_empty() {
        return;
    }
    mark_energy(b);

    // Promotion comes first so its log entry precedes any subsequent
    // cull (a body can't be both promoted and culled in one pass —
    // promoted bodies are inside HARD_CULL_DIST).
    for i in promoted {
        b.origin[i] = BodyOrigin::RoguePlanet;
        let (x, y) = (b.x[i], b.y[i]);
        let text = format!("{} ejected into interstellar space", tag(b, i));
        b.log(EventKind::Eject, text, x, y);
    }

    for i in gone {
        let (kind, text) = match b.origin[i] {
            BodyOrigin::Interstellar => (
                EventKind::Interstellar,
                format!("{} left the system outbound", tag(b, i)),
            ),
            BodyOrigin::RoguePlanet => {
                (EventKind::Escape, format!("{} left the system", tag(b, i)))
            }
            _ => (
                EventKind::Escape,
                format!("{} escaped the system", tag(b, i)),
            ),
        };
        let (x, y) = (b.x[i], b.y[i]);
        b.dead[i] = true;
        b.stats.escaped += 1;
        b.log(kind, text, x, y);
    }
    b.compact();
}
