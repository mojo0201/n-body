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
    for i in 0..n {
        if b.dead[i] || !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
            continue;
        }
        let d2 = b.x[i] * b.x[i] + b.y[i] * b.y[i];
        if d2 < cd2 {
            continue;
        }
        let v2 = b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i];
        if d2 > hcd2
            || (v2 > (2.0 * G * prim_m) / d2.sqrt() && b.x[i] * b.vx[i] + b.y[i] * b.vy[i] > 0.0)
        {
            gone.push(i);
        }
    }
    if gone.is_empty() {
        return;
    }
    mark_energy(b);
    for i in gone {
        let text = format!("{} escaped the system", tag(b, i));
        let (x, y) = (b.x[i], b.y[i]);
        b.dead[i] = true;
        b.stats.escaped += 1;
        b.log(EventKind::Escape, text, x, y);
    }
    b.compact();
}
