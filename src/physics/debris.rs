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
    let t_now = b.time;

    // Quick pass to decide whether the expensive path is worth taking.
    // Simulations with no rubble, or exactly one fragment, exit here.
    let mut n_eligible = 0usize;
    for i in 0..n {
        if !b.dead[i]
            && b.origin[i].is_rubble()
            && b.m[i] < DEBRIS_COALESCE_MAX_MASS
            && t_now >= b.imm[i]
        {
            n_eligible += 1;
        }
    }
    if n_eligible < 2 {
        // No pair can form, so no counters can advance — but stale
        // counters from a previous step must still be cleared so a
        // fragment that leaves the neighbourhood resets its streak.
        for i in 0..n {
            if !b.dead[i] && b.origin[i].is_rubble() {
                b.coalesce_counter[i] = 0;
            }
        }
        return;
    }

    let hosts = host_list(b);

    // Collect the eligible fragments once; every later pass iterates
    // over this compact list rather than scanning all bodies.
    let mut eligible: Vec<usize> = Vec::with_capacity(n_eligible);
    for i in 0..n {
        if !b.dead[i]
            && b.origin[i].is_rubble()
            && b.m[i] < DEBRIS_COALESCE_MAX_MASS
            && t_now >= b.imm[i]
        {
            eligible.push(i);
        }
    }

    // Reverse map: body index → position in `eligible`, or NOT_ELIGIBLE.
    debug_assert!(n <= u32::MAX as usize);
    const NOT_ELIGIBLE: u32 = u32::MAX;
    let mut body_to_eligible: Vec<u32> = vec![NOT_ELIGIBLE; n];
    for (pos, &i) in eligible.iter().enumerate() {
        body_to_eligible[i] = pos as u32;
    }

    // Rubble bodies that were not selected still need their counter
    // cleared, so a fragment that temporarily leaves the size window
    // doesn't resume a stale capture streak when it re-enters.
    for i in 0..n {
        if !b.dead[i] && b.origin[i].is_rubble() && body_to_eligible[i] == NOT_ELIGIBLE {
            b.coalesce_counter[i] = 0;
        }
    }

    let parents: Vec<Option<usize>> = eligible
        .iter()
        .map(|&i| b.parent_cache.get(i).copied().flatten())
        .collect();

    // Distance from each fragment to its parent, computed once here
    // and reused by every pair evaluation the fragment participates in.
    let mut d_to_parent: Vec<f64> = Vec::with_capacity(eligible.len());
    for (pos, &i) in eligible.iter().enumerate() {
        match parents[pos] {
            Some(pi) => {
                let dx = b.x[i] - b.x[pi];
                let dy = b.y[i] - b.y[pi];
                d_to_parent.push((dx * dx + dy * dy).sqrt());
            }
            None => d_to_parent.push(0.0),
        }
    }

    // Grid-cell size is sized to the largest fragment's interaction
    // radius so a 3×3 neighbourhood always covers the capture range.
    let max_cr = eligible
        .iter()
        .map(|&i| b.contact_r(i))
        .fold(0.0f64, f64::max);
    let cell = (DEBRIS_SEARCH_RADII * max_cr).max(1e-4);

    // Bucket every parented fragment into its grid cell. Fragments
    // without a parent can't be captured, so they're omitted.
    let mut keys: Vec<(u64, u32)> = Vec::with_capacity(eligible.len());
    for (pos, &i) in eligible.iter().enumerate() {
        if parents[pos].is_none() {
            continue;
        }
        let cx = (b.x[i] / cell).floor() as i64;
        let cy = (b.y[i] / cell).floor() as i64;
        keys.push((cell_key(cx, cy), i as u32));
    }
    keys.par_sort_unstable();

    let mut in_capture = vec![false; n];
    let mut pairs: Vec<(usize, usize)> = Vec::new();

    for (pos_i, &i) in eligible.iter().enumerate() {
        let Some(pi) = parents[pos_i] else { continue };

        // Quantities shared by every candidate partner j for this i.
        let m_p = effective_host_mass(b, &hosts, i, pi);
        let d_i = d_to_parent[pos_i];
        let cx = (b.x[i] / cell).floor() as i64;
        let cy = (b.y[i] / cell).floor() as i64;

        for ddx in -1..=1i64 {
            for ddy in -1..=1i64 {
                let key = cell_key(cx + ddx, cy + ddy);
                let start = keys.partition_point(|e| e.0 < key);
                for e in &keys[start..] {
                    if e.0 != key {
                        break;
                    }
                    let j = e.1 as usize;
                    // Each unordered pair is visited once.
                    if j <= i {
                        continue;
                    }
                    let pos_j = body_to_eligible[j];
                    if pos_j == NOT_ELIGIBLE {
                        continue;
                    }
                    let pos_j = pos_j as usize;
                    let Some(pj) = parents[pos_j] else { continue };
                    // Coalescence only occurs between siblings —
                    // fragments must share a parent host.
                    if pi != pj {
                        continue;
                    }

                    let dx = b.x[j] - b.x[i];
                    let dy = b.y[j] - b.y[i];
                    let dist2 = dx * dx + dy * dy;
                    let dist = dist2.sqrt();

                    // Mutual Hill radius of the pair around the host,
                    // evaluated at the mean orbital distance.
                    let d_j = d_to_parent[pos_j];
                    let m_ij = b.m[i] + b.m[j];
                    let r_hill = (m_ij / (3.0 * m_p)).cbrt() * ((d_i + d_j) * 0.5);

                    // Separation must lie well inside the Hill sphere
                    // for the pair to remain bound long enough to merge.
                    if dist >= DEBRIS_CAPTURE_HILL_FRAC * r_hill {
                        continue;
                    }

                    // Relative speed must fall below the escape speed
                    // of the pair, otherwise the encounter is unbound.
                    let dvx = b.vx[j] - b.vx[i];
                    let dvy = b.vy[j] - b.vy[i];
                    let vrel = (dvx * dvx + dvy * dvy).sqrt();
                    let v_esc = (2.0 * G * m_ij / dist.max(1e-12)).sqrt();
                    if vrel >= DEBRIS_CAPTURE_VEL_RATIO * v_esc {
                        continue;
                    }

                    in_capture[i] = true;
                    in_capture[j] = true;
                    pairs.push((i, j));
                }
            }
        }
    }

    // Advance the capture streak for fragments currently in capture;
    // reset any that fell out of it. Saturating add guards against
    // overflow during very long runs.
    for &i in &eligible {
        if in_capture[i] {
            b.coalesce_counter[i] = b.coalesce_counter[i].saturating_add(1);
        } else {
            b.coalesce_counter[i] = 0;
        }
    }

    if pairs.is_empty() {
        return;
    }

    // Keep only pairs where both members have persisted long enough.
    pairs.retain(|&(i, j)| {
        b.coalesce_counter[i] >= DEBRIS_CAPTURE_STEPS
            && b.coalesce_counter[j] >= DEBRIS_CAPTURE_STEPS
    });

    let mut merged_any = false;
    for (i, j) in pairs {
        // A body may appear in several pairs; earlier merges in this
        // pass can invalidate later ones. Re-check both membership and
        // streak before merging.
        if b.dead[i] || b.dead[j] {
            continue;
        }
        if b.coalesce_counter[i] < DEBRIS_CAPTURE_STEPS
            || b.coalesce_counter[j] < DEBRIS_CAPTURE_STEPS
        {
            continue;
        }
        merge_impl(b, i, j, /*silent=*/ true);
        b.stats.coalesces += 1;
        merged_any = true;
    }

    if merged_any {
        b.compact();
    }
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
            let d2 = dx * dx + dy * dy;
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
                let rdot = dx * (b.vx[i] - b.vx[p]) + dy * (b.vy[i] - b.vy[p]);
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

    // Unit vector from primary to body i, and its tangent.
    let d = ((b.x[i] - b.x[p]).powi(2) + (b.y[i] - b.y[p]).powi(2))
        .sqrt()
        .max(1e-12);
    let (ux, uy) = ((b.x[i] - b.x[p]) / d, (b.y[i] - b.y[p]) / d);
    let (tx, ty) = (-uy, ux);

    // Draw fragment mass weights first so the geometry (fragment
    // spacing, radii) can be sized to the largest fragment.
    let mut ms: Vec<f64> = (0..n)
        .map(|_| DEBRIS_WEIGHT_MIN + b.rng.next_f64())
        .collect();
    let wsum: f64 = ms.iter().sum();
    let mut rmax: f64 = 0.0;
    for m in ms.iter_mut() {
        *m = mi * *m / wsum;
        rmax = rmax.max(planet_true_r(*m));
    }

    let step = DISRUPT_STEP * rmax;
    let half = (((n - 1) as f64) / 2.0 * step).max(1e-6);
    let vk = DISRUPT_KICK * (G * mi / b.r[i].max(1e-6)).sqrt();
    let (bx, by, bvx, bvy) = (b.x[i], b.y[i], b.vx[i], b.vy[i]);
    let prim_r = b.contact_r(p);
    let (px0, py0) = (b.x[p], b.y[p]);

    b.dead[i] = true;

    // Build the fragments. Positions and kicks are symmetric about
    // the original body's centre along the tidal axis.
    let mut pieces: Vec<Spawn> = Vec::with_capacity(n);
    let (mut pvx, mut pvy, mut mt) = (0.0, 0.0, 0.0);
    for (k, &mk) in ms.iter().enumerate() {
        let s = (k as f64 - (n - 1) as f64 / 2.0) * step;
        let kick = vk * s / half;

        // Start position: offset along the tidal axis plus a small
        // perpendicular jitter.
        let mut fx = bx + ux * s + tx * (b.rng.next_f64() - DISRUPT_JITTER) * rmax;
        let mut fy = by + uy * s + ty * (b.rng.next_f64() - DISRUPT_JITTER) * rmax;

        // Push any fragment that would spawn inside the primary back
        // out to the contact radius plus a safety margin.
        let mut qx = fx - px0;
        let mut qy = fy - py0;
        let mut qd = (qx * qx + qy * qy).sqrt();
        let qmin = prim_r + 1.5 * planet_true_r(mk);
        if qd < qmin {
            if qd < 1e-9 {
                qx = ux;
                qy = uy;
                qd = 1.0;
            }
            fx = px0 + qx / qd * qmin;
            fy = py0 + qy / qd * qmin;
        }

        let vx = bvx + ux * kick;
        let vy = bvy + uy * kick;

        // Track momentum so we can later subtract the centre-of-mass
        // contribution and keep the fragment cloud recoiling.
        pvx += mk * (vx - bvx);
        pvy += mk * (vy - bvy);
        mt += mk;

        pieces.push(Spawn {
            x: fx,
            y: fy,
            vx,
            vy,
            m: mk,
            kind: BodyKind::Planet,
            origin: BodyOrigin::TidalDebris,
            // Fragments inherit the composition of the body they
            // came from. `i` is still valid here — `b.dead[i]` is
            // set above, but `class[i]` is untouched.
            class: b.class[i],
        });
    }

    // Spawn fragments and correct each one's velocity so the cloud's
    // centre of mass moves with the original body.
    let t_imm = b.time + IMMUNITY_TIME;
    let first = b.len();
    let mut l_after = 0.0;
    for s in pieces {
        let (vx, vy) = (s.vx - pvx / mt, s.vy - pvy / mt);
        let k = b.push(s.x, s.y, vx, vy, s.m, s.kind);
        b.imm[k] = t_imm;
        b.origin[k] = BodyOrigin::TidalDebris;
        b.class[k] = s.class;
        l_after += s.m * (s.x * vy - s.y * vx);
    }

    // Redistribute the orbital angular momentum the fragments lost
    // back into their spin, so total L is conserved.
    let dl = l_before - l_after;
    for k in first..b.len() {
        b.spin[k] = dl * b.m[k] / mt;
    }

    flash_at(b, bx, by, (prim_r * 0.5).max(0.05));
    b.stats.tidal += 1;
    b.log(EventKind::Tidal, text, bx, by);
    true
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
    let mut to_deposit: Vec<(f64, f64, f64, f64, f64, f64, f64)> = Vec::new();
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
            let d = (b.x[i] - b.x[hk]).hypot(b.y[i] - b.y[hk]);
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
        let (x, y, vx, vy) = (b.x[i], b.y[i], b.vx[i], b.vy[i]);
        let dx = b.x[i] - b.x[h];
        let dy = b.y[i] - b.y[h];
        let r = dx.hypot(dy).max(1e-9);
        let (ux, uy) = (dx / r, dy / r);

        to_deposit.push((x, y, vx, vy, dm, ux, uy));
        let new_m = b.m[i] - dm;
        if new_m < COMET_DEAD_MASS {
            to_kill.push(i);
        } else {
            mass_updates.push((i, new_m));
        }
    }

    for (i, m) in mass_updates {
        b.set_mass(i, m);
    }
    for (x, y, vx, vy, dm, ux, uy) in to_deposit {
        // Drop to 0.005 (about 22 m/s). The dust gently leaves the
        // nucleus and is carried into a curved tail purely by solar
        // radiation pressure.
        dust::deposit(b, x, y, vx, vy, dm, Some((ux, uy)), 0.005);
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

    let mut gone: Vec<usize> = Vec::new();
    let mut promoted: Vec<usize> = Vec::new();
    for i in 0..n {
        if b.dead[i] || !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
            continue;
        }
        let d2 = b.x[i] * b.x[i] + b.y[i] * b.y[i];
        if d2 < cd2 {
            continue;
        }
        let v2 = b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i];

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
        let unbound_out = v2 > v_esc_sq / d && b.x[i] * b.vx[i] + b.y[i] * b.vy[i] > 0.0;
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
