//! Debris-disk coalescence, Roche limit, tidal break-up, escape culling.

use super::*;

// ---- Fix 8: count eligible first, exit before the O(n) host scan and the
// ---- n-sized Vec. Hoist `m_p` and `d_i` out of the candidate loop.
pub(super) fn coalesce_debris(b: &mut Bodies) {
    let n = b.len();
    if n < 2 {
        return;
    }
    let t_now = b.time;

    // Count eligible bodies first. The common case (few or zero fragments)
    // exits without allocating the per-body eligibility map or scanning
    // the host list.
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
        // Reset the counter on every live rubble body — the same set
        // the old `body_to_eligible` + `eligible` loop reached.
        for i in 0..n {
            if !b.dead[i] && b.origin[i].is_rubble() {
                b.coalesce_counter[i] = 0;
            }
        }
        return;
    }

    let hosts = host_list(b);
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

    let mut body_to_eligible: Vec<Option<usize>> = vec![None; n];
    for (pos, &i) in eligible.iter().enumerate() {
        body_to_eligible[i] = Some(pos);
    }

    for (i, slot) in body_to_eligible.iter().enumerate() {
        if !b.dead[i] && b.origin[i].is_rubble() && slot.is_none() {
            b.coalesce_counter[i] = 0;
        }
    }

    let parents: Vec<Option<usize>> = eligible
        .iter()
        .map(|&i| b.parent_cache.get(i).copied().flatten())
        .collect();

    let max_cr = eligible
        .iter()
        .map(|&i| b.contact_r(i))
        .fold(0.0f64, f64::max);
    let cell = (DEBRIS_SEARCH_RADII * max_cr).max(1e-4);

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
        // Hoisted: constant across all candidate j for this i.
        let m_p = effective_host_mass(b, &hosts, i, pi);
        let d_i = ((b.x[i] - b.x[pi]).powi(2) + (b.y[i] - b.y[pi]).powi(2)).sqrt();
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
                    if j <= i {
                        continue;
                    }
                    let Some(pos_j) = body_to_eligible[j] else {
                        continue;
                    };
                    let Some(pj) = parents[pos_j] else { continue };
                    if pi != pj {
                        continue;
                    }

                    let dx = b.x[j] - b.x[i];
                    let dy = b.y[j] - b.y[i];
                    let dist2 = dx * dx + dy * dy;
                    let dist = dist2.sqrt();

                    let d_j = ((b.x[j] - b.x[pi]).powi(2) + (b.y[j] - b.y[pi]).powi(2)).sqrt();
                    let m_ij = b.m[i] + b.m[j];
                    let r_hill = (m_ij / (3.0 * m_p)).cbrt() * ((d_i + d_j) * 0.5);

                    if dist >= DEBRIS_CAPTURE_HILL_FRAC * r_hill {
                        continue;
                    }

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

    let to_merge: Vec<(usize, usize)> = pairs
        .into_iter()
        .filter(|&(i, j)| {
            b.coalesce_counter[i] >= DEBRIS_CAPTURE_STEPS
                && b.coalesce_counter[j] >= DEBRIS_CAPTURE_STEPS
        })
        .collect();

    let mut merged_any = false;
    for (i, j) in to_merge {
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

pub fn rigidity(rho_gcc: f64) -> f64 {
    ((rho_gcc - ROCHE_RHO_FLUID) / (ROCHE_RHO_RIGID - ROCHE_RHO_FLUID)).clamp(0.0, 1.0)
}

pub fn roche_limit(b: &Bodies, i: usize, p: usize) -> f64 {
    let w = rigidity(density_gcc(b, i));
    let coeff = ROCHE_FLUID * (1.0 - w) + ROCHE_RIGID * w;
    coeff * b.r[i] * (b.m[p] / b.m[i]).cbrt()
}

pub(super) fn tidal_breakup(b: &mut Bodies) {
    let n = b.len();
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
        let kf = ROCHE_FLUID * tb;
        let k2 = kf * kf;
        let k6 = k2 * k2 * k2;
        let mut inside = false;
        for &p in &prims {
            if p == i || b.dead[p] || b.m[p] < TIDAL_MASS_RATIO * b.m[i] {
                continue;
            }
            let dx = b.x[i] - b.x[p];
            let dy = b.y[i] - b.y[p];
            let d2 = dx * dx + dy * dy;
            let q = b.m[p] / b.m[i];
            if d2 * d2 * d2 >= k6 * q * q {
                continue;
            }
            let roche = roche_limit(b, i, p);
            if d2 < roche * roche {
                inside = true;
                b.disrupting[i] = true;
                let rdot = dx * (b.vx[i] - b.vx[p]) + dy * (b.vy[i] - b.vy[p]);
                if rdot >= 0.0 {
                    let _ = disrupt(b, i, p);
                }
                break;
            }
        }
        if !inside && !b.dead[i] {
            b.disrupting[i] = false;
        }
    }
    if b.any_dead() {
        b.compact();
    }
}

pub(super) fn disrupt(b: &mut Bodies, i: usize, p: usize) -> bool {
    let mi = b.m[i];
    let n = ((DISRUPT_FRAG_BASE + DISRUPT_FRAG_LOG * (mi + 1.0).log10()).round() as usize)
        .clamp(4, 12)
        .min(room(b, 0) + 1);
    if n < 3 {
        return false;
    }
    mark_energy(b);
    let l_before = l_orb(b, i) + b.spin[i];
    let text = format!("{} torn apart by {}", tag(b, i), tag(b, p));
    let d = ((b.x[i] - b.x[p]).powi(2) + (b.y[i] - b.y[p]).powi(2))
        .sqrt()
        .max(1e-12);
    let (ux, uy) = ((b.x[i] - b.x[p]) / d, (b.y[i] - b.y[p]) / d);
    let (tx, ty) = (-uy, ux);

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
    let mut pieces: Vec<Spawn> = Vec::with_capacity(n);
    let (mut pvx, mut pvy, mut mt) = (0.0, 0.0, 0.0);
    for (k, &mk) in ms.iter().enumerate() {
        let s = (k as f64 - (n - 1) as f64 / 2.0) * step;
        let kick = vk * s / half;
        let mut fx = bx + ux * s + tx * (b.rng.next_f64() - DISRUPT_JITTER) * rmax;
        let mut fy = by + uy * s + ty * (b.rng.next_f64() - DISRUPT_JITTER) * rmax;
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
            // A tidally disrupted body sheds material of its own
            // composition, so every fragment inherits `b.class[i]`
            // (the body being torn apart). `i` is still valid — the
            // `b.dead[i] = true` below runs *after* this loop.
            class: b.class[i],
        });
    }
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
    // Fast path: no comets → no work. `advance` calls this on every
    // step, so the early return matters for large simulations.
    if !b.origin.iter().any(|&o| o == BodyOrigin::Comet) {
        return;
    }
    let n = b.len();
    let hosts = host_list(b);
    if hosts.is_empty() {
        return;
    }
    let t_now = b.time;

    // Deferred operations, so the borrow checker stays happy across the
    // `dust::deposit` calls.
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
        // Nearest host.
        let mut r_min = f64::INFINITY;
        let mut h = hosts[0];
        for &k in &hosts {
            let d = (b.x[i] - b.x[k]).hypot(b.y[i] - b.y[k]);
            if d < r_min {
                r_min = d;
                h = k;
            }
        }
        let r_au = r_min / au_in_gm();
        let lum = luminosity_of(b, h).max(0.0);
        if lum <= 0.0 {
            continue;
        }
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
        let (x, y, vx, vy) = (b.x[i], b.y[i], b.vx[i], b.vy[i]);
        let dx = b.x[i] - b.x[h];
        let dy = b.y[i] - b.y[h];
        let r = dx.hypot(dy).max(1e-9);
        let (ux, uy) = (dx / r, dy / r); // anti-sunward
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
        // 0.02 reduces the ejection velocity to ~0.09 km/s (90 m/s).
        // This allows the dust to retain the comet's orbital momentum,
        // causing the tail to curve behind the comet rather than shoot
        // out in a straight line.
        dust::deposit(b, x, y, vx, vy, dm, Some((ux, uy)), 0.02);
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

pub(super) fn cull_escaped(b: &mut Bodies) {
    let n = b.len();
    if n == 0 {
        return;
    }
    let mut prim_m = 0.0;
    for i in 0..n {
        if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
            prim_m += b.m[i];
        }
    }
    let cd2 = CULL_DIST * CULL_DIST;
    let hcd2 = HARD_CULL_DIST * HARD_CULL_DIST;
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
        let unbound_out =
            v2 > (2.0 * G * prim_m) / d2.sqrt() && b.x[i] * b.vx[i] + b.y[i] * b.vy[i] > 0.0;
        if !unbound_out {
            continue;
        }
        match b.origin[i] {
            BodyOrigin::RoguePlanet => {
                // A rogue is tracked all the way to HARD_CULL_DIST so
                // the user can watch it cross the outer system.
            }
            BodyOrigin::Interstellar => {
                // ISOs cull at CULL_DIST — same threshold as Feature 7
                // established. A real interstellar object is
                // unobservable within a few tens of AU; tracking it to
                // 6,680 AU would show nothing new. The `gone` loop's
                // match below keeps its `EventKind::Interstellar` log
                // kind, so the "left the system outbound" message is
                // preserved.
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
