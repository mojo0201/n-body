//! Collision detection and the impact model.

use super::*;

/// Below this many hot bodies a plain pair loop beats building a hash grid.
pub(super) const HOT_GRID_MIN: usize = 24;

pub(super) fn flash_at(b: &mut Bodies, x: f64, y: f64, r0: f64) {
    b.flashes.push(Flash {
        x,
        y,
        r0,
        age: 0.0,
        life: FLASH_LIFE,
    });
}

pub(super) fn impact_offset(b: &Bodies, i: usize, j: usize) -> f64 {
    let dx = b.x[j] - b.x[i];
    let dy = b.y[j] - b.y[i];
    let vx = b.vx[j] - b.vx[i];
    let vy = b.vy[j] - b.vy[i];
    let v = (vx * vx + vy * vy).sqrt();
    if v <= 0.0 {
        return 0.0;
    }
    (dx * vy - dy * vx).abs() / v
}

pub(super) fn pair_hit_d(b: &Bodies, i: usize, j: usize, sweep: bool) -> Option<f64> {
    let rr = b.contact_r(i) + b.contact_r(j);
    let rr2 = rr * rr;
    let dx = b.x[j] - b.x[i];
    let dy = b.y[j] - b.y[i];
    if dx * dx + dy * dy < rr2 {
        return Some(impact_offset(b, i, j));
    }
    if !sweep {
        return None;
    }
    let x0 = b._px[j] - b._px[i];
    let y0 = b._py[j] - b._py[i];
    let ex = dx - x0;
    let ey = dy - y0;
    let e2 = ex * ex + ey * ey;
    if e2 <= 0.0 || e2.is_nan() {
        return None;
    }
    let t = -(x0 * ex + y0 * ey) / e2;
    if t <= 0.0 || t >= 1.0 {
        return None;
    }
    let cx = x0 + t * ex;
    let cy = y0 + t * ey;
    if cx * cx + cy * cy < rr2 {
        Some((cx * cx + cy * cy).sqrt())
    } else {
        None
    }
}

pub(super) fn pair_hit(b: &Bodies, i: usize, j: usize, sweep: bool) -> bool {
    pair_hit_d(b, i, j, sweep).is_some()
}

#[inline]
pub(super) fn cell_key(cx: i64, cy: i64) -> u64 {
    (cx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (cy as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
}

pub(super) fn find_collisions(b: &Bodies, sweep: bool, dt: f64) -> Vec<(usize, usize)> {
    let n = b.len();
    let t_now = b.time;
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    if n < HASH_THRESHOLD {
        for i in 0..n {
            if b.dead[i] || t_now < b.imm[i] {
                continue;
            }
            for j in (i + 1)..n {
                if b.dead[j] || t_now < b.imm[j] {
                    continue;
                }
                if pair_hit(b, i, j, sweep) {
                    pairs.push((i, j));
                }
            }
        }
        return pairs;
    }

    let mut max_cr: f64 = 0.0;
    let mut vmax2: f64 = 0.0;
    for i in 0..n {
        if b.dead[i] {
            continue;
        }
        max_cr = max_cr.max(b.contact_r(i));
        vmax2 = vmax2.max(b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i]);
    }
    let motion = if sweep {
        2.0 * vmax2.sqrt() * dt * 1.05
    } else {
        0.0
    };
    let cell = (2.0 * max_cr + motion).max(1e-4);

    let mut keys: Vec<(u64, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        if b.dead[i] {
            continue;
        }
        let cx = (b.x[i] / cell).floor() as i64;
        let cy = (b.y[i] / cell).floor() as i64;
        keys.push((cell_key(cx, cy), i as u32));
    }
    keys.par_sort_unstable();

    for i in 0..n {
        if b.dead[i] || t_now < b.imm[i] {
            continue;
        }
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
                    if j <= i || b.dead[j] || t_now < b.imm[j] {
                        continue;
                    }
                    if pair_hit(b, i, j, sweep) {
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
    // Only live, non-immune hot bodies can take part. `hot` is ascending, so
    // `live` is ascending too.
    let live: Vec<usize> = hot
        .iter()
        .copied()
        .filter(|&i| !b.dead[i] && t_now >= b.imm[i])
        .collect();
    let nl = live.len();
    let mut pairs = Vec::new();
    if nl < HOT_GRID_MIN {
        for (a, &i) in live.iter().enumerate() {
            for &j in &live[a + 1..] {
                if pair_hit(b, i, j, sweep) {
                    pairs.push((i, j));
                }
            }
        }
        return pairs;
    }

    // Spatial hash. A pair can only hit if the centres are within
    // 2 * max_contact_r (static overlap) or within that plus both bodies'
    // displacement this tick (swept test), so the cell edge covers both.
    let mut max_cr = 0.0f64;
    let mut max_disp = 0.0f64;
    for &i in &live {
        max_cr = max_cr.max(b.contact_r(i));
        if sweep {
            max_disp = max_disp.max((b.x[i] - b._px[i]).hypot(b.y[i] - b._py[i]));
        }
    }
    let cell = ((2.0 * max_cr + 2.0 * max_disp) * 1.0001).max(1e-4);
    let inv = 1.0 / cell;

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
                    if pair_hit(b, i, j, sweep) {
                        pairs.push((i, j));
                    }
                }
            }
        }
    }
    // Same order as the brute-force loop: ascending i, then ascending j.
    pairs.sort_unstable();
    pairs
}

pub(super) fn collision_pass(b: &mut Bodies, dt: f64, hot: Option<&[usize]>) {
    if b.len() < 2 {
        return;
    }
    // One swept pass, no cascades. Bodies involved in a collision are put
    // under IMMUNITY_TIME (0.4 sim units >> DT = 1/480), so a second pass
    // within the same call always finds zero new pairs — the cascade the
    // old 3-iteration loop was guarding against cannot occur. Verified by
    // `c_moon_forming_impact_makes_a_disk`, which requires exactly one
    // impact per call and still produces ≥4 bodies after this change.
    let mut spawns: Vec<Spawn> = Vec::new();
    let pairs = match hot {
        Some(h) => find_collisions_hot(b, h, true, dt),
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
        b.imm[k] = t_imm;
        b.origin[k] = s.origin;
        b.class[k] = s.class;
    }
}

#[inline]
pub(super) fn l_orb(b: &Bodies, i: usize) -> f64 {
    b.m[i] * (b.x[i] * b.vy[i] - b.y[i] * b.vx[i])
}

pub(super) fn pair_l(b: &Bodies, i: usize, j: usize) -> f64 {
    l_orb(b, i) + l_orb(b, j) + b.spin[i] + b.spin[j]
}

pub(super) fn rho1_sim() -> f64 {
    1.0 / (EARTH_RHO_GCC * 4.0 / 3.0 * std::f64::consts::PI * R_EARTH_GM.powi(3))
}

pub(super) fn interact_frac(rt: f64, rp: f64, bb: f64) -> f64 {
    let l = (rt + rp) * (1.0 - bb);
    if l >= 2.0 * rp {
        1.0
    } else {
        ((3.0 * rp * l * l - l * l * l) / (4.0 * rp * rp * rp)).clamp(0.0, 1.0)
    }
}

pub(super) fn q_star_rd(m_tot: f64, gamma: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let rho1 = rho1_sim();
    let rc1 = (3.0 * m_tot / (4.0 * pi * rho1)).cbrt();
    let q11 = LS_C_STAR * 0.8 * pi * rho1 * G * rc1 * rc1;
    q11 * (gamma + 1.0).powi(2) / (4.0 * gamma)
}

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

pub(super) fn hit_and_run(b: &mut Bodies, big: usize, small: usize, rr: f64, bb: f64, ratio: f64) {
    mark_energy(b);
    let l_before = pair_l(b, big, small);
    let (dx, dy) = (b.x[small] - b.x[big], b.y[small] - b.y[big]);
    let (vx, vy) = (b.vx[small] - b.vx[big], b.vy[small] - b.vy[big]);
    let v = (vx * vx + vy * vy).sqrt().max(1e-12);
    let (ux, uy) = (vx / v, vy / v);
    let along = dx * ux + dy * uy;
    let (hx, hy) = (dx - along * ux, dy - along * uy);
    let d = (hx * hx + hy * hy).sqrt();
    let (hux, huy) = if d > 1e-12 {
        (hx / d, hy / d)
    } else {
        (-uy, ux)
    };
    let s = (rr * rr - d * d).max(0.0).sqrt();
    let (nx, ny) = ((d * hux - s * ux) / rr, (d * huy - s * uy) / rr);
    let vn = vx * nx + vy * ny;
    let (mb, ms) = (b.m[big], b.m[small]);
    let mtot = mb + ms;
    if vn < 0.0 {
        let dv = -(1.0 + HIT_RUN_RESTITUTION) * vn;
        b.vx[small] += mb / mtot * dv * nx;
        b.vy[small] += mb / mtot * dv * ny;
        b.vx[big] -= ms / mtot * dv * nx;
        b.vy[big] -= ms / mtot * dv * ny;
    }
    let alpha = interact_frac(b.contact_r(big), b.contact_r(small), bb);
    let mt = (HIT_RUN_LOSS * alpha * ms).min(0.5 * ms);
    if mt > 0.0 {
        let nb = mb + mt;
        let cx0 = (mb * b.x[big] + ms * b.x[small]) / mtot;
        let cy0 = (mb * b.y[big] + ms * b.y[small]) / mtot;
        let cx1 = (nb * b.x[big] + (ms - mt) * b.x[small]) / mtot;
        let cy1 = (nb * b.y[big] + (ms - mt) * b.y[small]) / mtot;
        for k in [big, small] {
            b.x[k] += cx0 - cx1;
            b.y[k] += cy0 - cy1;
            b._px[k] = b.x[k];
            b._py[k] = b.y[k];
        }
        b.vx[big] = (mb * b.vx[big] + mt * b.vx[small]) / nb;
        b.vy[big] = (mb * b.vy[big] + mt * b.vy[small]) / nb;
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
    flash_at(b, cx, cy, (b.contact_r(big) * 2.0).max(0.05));
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

pub(super) fn rubble_bounce(b: &mut Bodies, i: usize, j: usize, rr: f64) {
    let dx = b.x[j] - b.x[i];
    let dy = b.y[j] - b.y[i];
    let dvx = b.vx[j] - b.vx[i];
    let dvy = b.vy[j] - b.vy[i];
    let v = (dvx * dvx + dvy * dvy).sqrt().max(1e-12);
    let (ux, uy) = (dvx / v, dvy / v);

    let along = dx * ux + dy * uy;
    let (hx, hy) = (dx - along * ux, dy - along * uy);
    let d = (hx * hx + hy * hy).sqrt();
    let (hux, huy) = if d > 1e-12 {
        (hx / d, hy / d)
    } else {
        (-uy, ux)
    };
    let s = (rr * rr - d * d).max(0.0).sqrt();
    let (nx, ny) = ((d * hux - s * ux) / rr, (d * huy - s * uy) / rr);
    let vn = dvx * nx + dvy * ny;
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
    b.vx[j] += mi / mtot * dvn * nx;
    b.vy[j] += mi / mtot * dvn * ny;

    let dl = l_before - pair_l(b, i, j);
    b.spin[i] += dl * mi / mtot;
    b.spin[j] += dl * mj / mtot;

    let t_imm = b.time + RUBBLE_BOUNCE_IMMUNITY;
    b.imm[i] = t_imm;
    b.imm[j] = t_imm;

    b.stats.bounces += 1;
}

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

pub(super) fn room(b: &Bodies, pending: usize) -> usize {
    MAX_BODIES.saturating_sub(b.len() + pending)
}

pub(super) fn combine_origin(keep: BodyOrigin, kill: BodyOrigin) -> BodyOrigin {
    // Two debris bodies stay debris; two belt bodies stay in the belt.
    if (keep.is_rubble() && kill.is_rubble()) || (keep.is_belt() && kill.is_belt()) {
        keep
    } else {
        BodyOrigin::Spawned
    }
}

pub(super) fn merge_impl(b: &mut Bodies, i: usize, j: usize, silent: bool) -> usize {
    mark_energy(b);
    let (keep, kill) = pick_survivor(b, i, j);
    let total = b.m[keep] + b.m[kill];
    let cx = (b.m[keep] * b.x[keep] + b.m[kill] * b.x[kill]) / total;
    let cy = (b.m[keep] * b.y[keep] + b.m[kill] * b.y[kill]) / total;
    let cvx = (b.m[keep] * b.vx[keep] + b.m[kill] * b.vx[kill]) / total;
    let cvy = (b.m[keep] * b.vy[keep] + b.m[kill] * b.vy[kill]) / total;

    let mu = b.m[keep] * b.m[kill] / total;
    let rl = (b.x[kill] - b.x[keep]) * (b.vy[kill] - b.vy[keep])
        - (b.y[kill] - b.y[keep]) * (b.vx[kill] - b.vx[keep]);
    b.spin[keep] += b.spin[kill] + mu * rl;

    let verb = if b.kind[keep] != BodyKind::Planet && b.kind[kill] == BodyKind::Planet {
        "fell into"
    } else {
        "merged into"
    };
    let text = format!("{} {} {}", tag(b, kill), verb, tag(b, keep));

    b.x[keep] = cx;
    b.y[keep] = cy;
    b.vx[keep] = cvx;
    b.vy[keep] = cvy;
    b._px[keep] = cx;
    b._py[keep] = cy;
    // Evolution state of the survivor, before the radius is recomputed.
    let keep_was_star = b.kind[keep] == BodyKind::Star;
    let keep_kind_before = b.kind[keep];
    let kill_kind = b.kind[kill];
    // Preserve a mass-weighted fractional MS age. For a star, this is the
    // fraction of its MS lifetime that it has used. For a dwarf, this is
    // still numerically valid (it's just a proxy for "fraction cooled"),
    // and the kind-specific block below overwrites it with the proper
    // dwarf age once the survivor's final kind is known.
    let frac_age = |k: usize| -> f64 {
        let t = star_lifetime_gyr(BodyKind::Star, b.m_init[k]);
        if t.is_finite() && t > 0.0 {
            (b.age_gyr[k] / t).clamp(0.0, 0.99)
        } else {
            0.0
        }
    };
    let f = (frac_age(keep) * b.m[keep] + frac_age(kill) * b.m[kill]) / total;
    b.m_init[keep] += b.m_init[kill];
    b.age_gyr[keep] = f * star_lifetime_gyr(BodyKind::Star, b.m_init[keep]);
    if matches!(b.kind[keep], BodyKind::Planet | BodyKind::DwarfPlanet) {
        let rho = (b.m[keep] * b.bulk_density[keep] + b.m[kill] * b.bulk_density[kill]) / total;
        b.bulk_density[keep] = rho;
    }
    b.set_mass(keep, total);
    b.origin[keep] = combine_origin(b.origin[keep], b.origin[kill]);

    if b.origin[keep].is_rubble() && b.m[keep] >= RUBBLE_PROMOTE_MASS {
        b.origin[keep] = BodyOrigin::Spawned;
        let text = format!("{} grew to planet mass", tag(b, keep));
        b.log(EventKind::Merge, text, cx, cy);
    }

    // Same mass range as the spawn, so a merge cannot promote a body that
    // is already above `DWARF_M_MAX`.
    if b.origin[keep].is_belt()
        && b.kind[keep] == BodyKind::Planet
        && classify_small_body(b.m[keep]) == BodyKind::DwarfPlanet
    {
        b.kind[keep] = BodyKind::DwarfPlanet;
    }

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

    let kind_after = b.kind[keep];
    if kind_after == BodyKind::Star {
        if !keep_was_star {
            // A brown dwarf that ignited starts as a new zero-age star.
            b.m_init[keep] = b.m[keep];
            b.age_gyr[keep] = 0.0;
        }
    } else {
        b.m_init[keep] = b.m[keep];
        let same_kind = keep_kind_before == kind_after;
        // A dwarf that only swallows a planet or a star keeps its cooling age.
        // A merger of two dwarfs, or a new kind of object, starts again.
        let swallowed_other = same_kind
            && kill_kind != kind_after
            && matches!(kind_after, BodyKind::WhiteDwarf | BodyKind::BrownDwarf);
        if !swallowed_other {
            b.age_gyr[keep] = if same_kind {
                0.0
            } else {
                default_age_gyr(kind_after)
            };
        }
    }
    b.r[keep] = b.radius_now(keep);

    b.dead[kill] = true;
    b.coalesce_counter[keep] = 0;

    b.stats.merges += 1;
    if !silent {
        let flash_r = (b.contact_r(keep) * 5.0).max(0.05);
        flash_at(b, cx, cy, flash_r);
        b.log(EventKind::Merge, text, cx, cy);
    }
    keep
}

pub(super) fn merge(b: &mut Bodies, i: usize, j: usize) -> usize {
    merge_impl(b, i, j, false)
}

pub(super) fn collide(b: &mut Bodies, i: usize, j: usize, spawns: &mut Vec<Spawn>) {
    if b.kind[i] != BodyKind::Planet || b.kind[j] != BodyKind::Planet {
        merge(b, i, j);
        return;
    }
    let (rvx, rvy) = (b.vx[j] - b.vx[i], b.vy[j] - b.vy[i]);
    let vrel = (rvx * rvx + rvy * rvy).sqrt();
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
        }
        if ratio < RUBBLE_BOUNCE_VRATIO {
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
    let md = (m_erode + m_disk).min(cap);
    let n_raw = FRAG_COUNT_BASE
        + FRAG_COUNT_SCALE * x.powf(FRAG_COUNT_EXP)
        + FRAG_COUNT_DISK_GAIN * (m_disk / DEBRIS_MIN_MASS).ln_1p();
    let n = (n_raw.round() as usize)
        .min(FRAG_COUNT_MAX)
        .min((md / DEBRIS_MIN_MASS) as usize)
        .min(room(b, spawns.len()) + 1);
    if md < DEBRIS_MIN_MASS || n < 2 {
        merge(b, i, j);
        return;
    }

    let l_before = pair_l(b, i, j);
    let (px0, py0) = (b.x[j] - b.x[i], b.y[j] - b.y[i]);
    let sense = if px0 * rvy - py0 * rvx >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let idg = (b.id[i], b.id[j]);
    let rem = merge(b, i, j);
    b.events.pop_back();
    b.stats.merges -= 1;
    let m_rem = mtot - md;
    b.set_mass(rem, m_rem);
    b.imm[rem] = b.time + IMMUNITY_TIME;
    b.origin[rem] = BodyOrigin::Spawned;

    let ms = frag_masses(&mut b.rng, n, md);
    // Dust source term: a bonus population, see `DUST_YIELD_FRAC`.
    let (dust_x, dust_y) = (b.x[rem], b.y[rem]);
    let (dust_vx, dust_vy) = (b.vx[rem], b.vy[rem]);
    dust::deposit(b, dust_x, dust_y, dust_vx, dust_vy, md * DUST_YIELD_FRAC);
    let r_rem = b.contact_r(rem);
    let v_esc = (2.0 * G * m_rem / r_rem.max(1e-6)).sqrt();
    let disk_share = (m_disk / md).clamp(0.0, 1.0);
    let base = b.rng.next_f64() * std::f64::consts::TAU;
    let (rx, ry, rvx0, rvy0) = (b.x[rem], b.y[rem], b.vx[rem], b.vy[rem]);
    let s0 = spawns.len();
    let (mut px, mut py, mut sx, mut sy) = (0.0, 0.0, 0.0, 0.0);
    for (k, &mk) in ms.iter().enumerate() {
        let is_disk = b.rng.next_f64() < disk_share;
        let ang = base
            + (k as f64) * std::f64::consts::TAU / n as f64
            + (b.rng.next_f64() - 0.5) * DEBRIS_ANGLE_JIT;
        let (ux, uy) = (ang.cos(), ang.sin());
        let (dist, wx, wy);
        if is_disk {
            dist = r_rem * (DISK_R_MIN + DISK_R_SPAN * b.rng.next_f64()) + planet_true_r(mk);
            let vc = (G * m_rem / dist).sqrt() * (0.9 + 0.2 * b.rng.next_f64());
            wx = -sense * uy * vc;
            wy = sense * ux * vc;
        } else {
            dist = (r_rem + planet_true_r(mk)) * (1.0 + DEBRIS_GAP_FRAC);
            let kick = v_esc * (DEBRIS_KICK_MIN + DEBRIS_KICK_SPAN * b.rng.next_f64());
            wx = ux * kick;
            wy = uy * kick;
        }
        px += mk * wx;
        py += mk * wy;
        sx += mk * ux * dist;
        sy += mk * uy * dist;
        spawns.push(Spawn {
            x: rx + ux * dist,
            y: ry + uy * dist,
            vx: rvx0 + wx,
            vy: rvy0 + wy,
            m: mk,
            kind: BodyKind::Planet,
            origin: BodyOrigin::ImpactDebris,
            // Inherit the class of the surviving (dominant) body, so
            // a fragment from a rocky impact is rocky and a fragment
            // from an icy impact is icy. `rem` is still the pre-compact
            // index of the merge survivor at this point, and `merge`
            // does not touch `class`, so `b.class[rem]` is the class
            // that body had before the impact.
            class: b.class[rem],
        });
    }
    b.x[rem] -= sx / m_rem;
    b.y[rem] -= sy / m_rem;
    b._px[rem] = b.x[rem];
    b._py[rem] = b.y[rem];
    b.vx[rem] -= px / m_rem;
    b.vy[rem] -= py / m_rem;

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
        n,
        mass_str(md),
        if m_disk > m_erode { " (disk)" } else { "" }
    );
    b.log(EventKind::Impact, text, rx, ry);
}
