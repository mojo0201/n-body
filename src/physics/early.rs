//! Early-system evolution: gas-disk migration, dispersal, then the scattering
//! that follows.
//!
//! Generated systems are placed dynamically quiet. Real ones are shaped in
//! their first ~100 Myr: planets migrate in the disk and lock into resonance
//! chains, and when the gas disperses, packed chains and crowded giants go
//! unstable and scatter, collide or eject each other. A literal 100 Myr
//! integration is out of reach (1 yr is ~7 time units), so this runs a
//! *time-compressed* version with a fixed step budget:
//!
//!   1. Disk phase: every planet host gets a full disk whose strength decays
//!      exponentially (`EARLY_DISK_EFOLD` time units), driving the normal
//!      migration/eccentricity-damping model (`gas_disk_migration`).
//!   2. Relaxation phase: the disk is gone; the system evolves under pure
//!      N-body gravity, collisions and ejections for `EARLY_RELAX_UNITS`.
//!
//! Before the disk phase, some giant-planet pairs are moved closer together
//! (`youngify_giants`): a generated system is placed Hill-stable, but real
//! giant systems are often born packed and scatter once the gas leaves; the
//! relaxation phase then plays that out (eccentric giants, ejections, mergers).
//!
//! The relaxation phase is ~10⁻⁵ of a real 100 Myr, so it only catches the
//! fast instabilities; slow ones are left to the live simulation.
//! The pass runs on a *sandbox copy* holding only the hosts and their planets
//! (moons, belts and dust would make it hundreds of times slower), then writes
//! the outcome back: survivors keep their moons (shifted along with them),
//! planets that merged are removed with theirs, ejected ones leave with theirs.
//! Belts and dust are not part of the pass (a limitation: they are not stirred
//! by migrating giants). Deterministic: depends only on the generated state.

use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

static EARLY_EVOLVE: AtomicBool = AtomicBool::new(true);

/// Whether newly generated systems go through early evolution.
pub fn set_early_evolution(on: bool) {
    EARLY_EVOLVE.store(on, Ordering::Relaxed);
}
pub fn early_evolution() -> bool {
    EARLY_EVOLVE.load(Ordering::Relaxed)
}

/// Disk e-folding time, in simulation time units (~15 yr). With migration
/// sped up 300×, this gives a 10 M⊕ planet at 1 AU a ~0.6 e-fold shrink, a
/// Jupiter at 5 AU ~0.4: most planets move a little, and only light inner
/// ones reach the cavity edge.
pub const EARLY_DISK_EFOLD: f64 = 150.0;
/// Disk phase length in e-folds.
const DISK_EFOLDS: f64 = 5.0;
/// Disk-free relaxation, time units (~550 yr).
pub const EARLY_RELAX_UNITS: f64 = 4000.0;
/// Hard cap on integrator steps (bounds generation time).
const EARLY_MAX_STEPS: u64 = 3_000_000;

#[derive(Clone, Debug, Default)]
pub struct EarlyReport {
    pub planets_before: usize,
    pub planets_after: usize,
    pub merged: usize,
    pub ejected: usize,
    /// Giant pairs that were moved closer before the run.
    pub packed: usize,
    pub skipped: bool,
}

/// Planets that orbit a host directly and are real planets, not belts.
fn is_primary_planet(b: &Bodies, i: usize) -> bool {
    !b.dead[i]
        && b.kind[i] == BodyKind::Planet
        && b.origin[i] == BodyOrigin::Spawned
        && b.m[i] >= 0.05
        && find_parent(b, i).is_some_and(|h| is_host_kind(b.kind[h]))
}

/// Run the time-compressed early evolution on `b` (a freshly spawned system).
pub fn evolve_early(b: &mut Bodies) -> EarlyReport {
    let n = b.len();
    let planets: Vec<usize> = (0..n).filter(|&i| is_primary_planet(b, i)).collect();
    let mut rep = EarlyReport { planets_before: planets.len(), planets_after: planets.len(), ..Default::default() };
    if planets.len() < 2 {
        rep.skipped = true;
        return rep;
    }
    // Which planet carries each other body (moons ride with their planet).
    let mut carrier: Vec<Option<usize>> = vec![None; n];
    for &p in &planets {
        carrier[p] = Some(p);
    }
    for i in 0..n {
        if b.dead[i] || carrier[i].is_some() || is_host_kind(b.kind[i]) {
            continue;
        }
        let mut cur = i;
        for _ in 0..4 {
            match find_parent(b, cur) {
                Some(p) if carrier[p].is_some() => {
                    carrier[i] = carrier[p];
                    break;
                }
                Some(p) if !is_host_kind(b.kind[p]) => cur = p,
                _ => break,
            }
        }
    }
    // Sandbox: hosts + primary planets only.
    let mut sb = b.clone();
    for i in 0..n {
        let keep = (is_host_kind(b.kind[i]) && !b.dead[i]) || carrier[i] == Some(i);
        if !keep {
            sb.dead[i] = true;
        }
    }
    sb.compact();
    sb.dust_fields.clear();
    sb.parent_cache_valid = false;
    compute_accel(&mut sb);
    rebase_energy(&mut sb);

    // Disk lifetime/mass varies by orders of magnitude between systems: draw the
    // e-folding time log-uniformly over a decade, so most systems migrate little.
    let mut drng = XorShift::new(sb.id.iter().fold(0x51ED_270Bu128, |h, &i| h.wrapping_mul(131).wrapping_add(i as u128)));
    let efold = EARLY_DISK_EFOLD * 10f64.powf(-drng.next_f64());
    let mut n_young = 0;
    let mut packed = false;
    let mut t = 0.0;
    let t_disk = DISK_EFOLDS * efold;
    let t_end = t_disk + EARLY_RELAX_UNITS;
    let mut steps = 0u64;
    let mut dt = DT;
    while t < t_end && steps < EARLY_MAX_STEPS {
        if steps.is_multiple_of(200) {
            dt = choose_dt(&mut sb);
            // Disk strength follows exp(−t/efold) through the host age; none once dispersed.
            let age = if t < t_disk { t / efold * 0.002 } else { 1.0 };
            for i in 0..sb.len() {
                if sb.kind[i] == BodyKind::Star && !sb.dead[i] {
                    sb.age_gyr[i] = age;
                }
            }
        }
        if t >= t_disk && !packed {
            // The gas is gone: some giant pairs now turn out to be too close.
            packed = true;
            n_young = youngify_giants(&mut sb);
        }
        let h = dt.min(pair_dt(&sb));
        yoshida(&mut sb, h);
        if t < t_disk {
            gas_disk_migration(&mut sb, h);
        }
        merge_touching(&mut sb);
        t += h;
        steps += 1;
    }
    // Write back. Survivors: shift the planet and its moons by the planet's change.
    let mut shift: Vec<Option<[f64; 6]>> = vec![None; n];
    let mut lost = vec![false; n];
    for i in 0..n {
        if b.dead[i] {
            continue;
        }
        let host = is_host_kind(b.kind[i]);
        if !host && carrier[i] != Some(i) {
            continue;
        }
        match sb.index_of_id(b.id[i]) {
            Some(j) if !sb.dead[j] => {
                shift[i] = Some([
                    sb.x[j] - b.x[i], sb.y[j] - b.y[i], sb.z[j] - b.z[i],
                    sb.vx[j] - b.vx[i], sb.vy[j] - b.vy[i], sb.vz[j] - b.vz[i],
                ]);
                if !host {
                    b.m[i] = sb.m[j];
                    b.r[i] = sb.r[j];
                    b.kind[i] = sb.kind[j];
                    b.bulk_density[i] = sb.bulk_density[j];
                    b.class[i] = sb.class[j];
                    b.spin[i] = sb.spin[j];
                    b.m_init[i] = sb.m_init[j];
                }
            }
            _ => lost[i] = true,
        }
    }
    for i in 0..n {
        if b.dead[i] {
            continue;
        }
        let key = if is_host_kind(b.kind[i]) { Some(i) } else { carrier[i] };
        let Some(c) = key else { continue };
        if lost[c] {
            b.dead[i] = true;
            continue;
        }
        if let Some(d) = shift[c] {
            // Moons keep their offset from their planet: shift by the planet's change.
            b.x[i] += d[0]; b.y[i] += d[1]; b.z[i] += d[2];
            b.vx[i] += d[3]; b.vy[i] += d[4]; b.vz[i] += d[5];
            b._px[i] = b.x[i]; b._py[i] = b.y[i]; b._pz[i] = b.z[i];
        }
    }
    for &p in &planets {
        if let (Some(_), false) = (shift[p], lost[p]) {
            if let Some(j) = sb.index_of_id(b.id[p]) {
                if let Some(h) = find_parent(&sb, j) {
                    let _ = h;
                } else {
                    rep.ejected += 1;
                }
            }
        }
    }
    rep.merged = planets.iter().filter(|&&p| lost[p]).count();
    rep.packed = n_young;
    b.parent_cache_valid = false;
    b.compact();
    rep.planets_after = (0..b.len()).filter(|&i| is_primary_planet(b, i)).count();
    rep
}

/// Base step: 1/40 of the shortest planet period, at most 8 DT.
fn choose_dt(sb: &mut Bodies) -> f64 {
    let n = sb.len();
    for i in 0..n {
        sb.parent_cache.resize(n, None);
        sb.parent_cache[i] = if sb.dead[i] { None } else { find_parent(sb, i) };
    }
    sb.parent_cache_valid = true;
    let mut p_min = f64::INFINITY;
    for i in 0..n {
        if sb.dead[i] || is_host_kind(sb.kind[i]) {
            continue;
        }
        let Some(h) = sb.parent_cache[i] else { continue };
        let (dx, dy, dz) = (sb.x[i] - sb.x[h], sb.y[i] - sb.y[h], sb.z[i] - sb.z[h]);
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        let p = std::f64::consts::TAU * (r * r * r / (G * (sb.m[h] + sb.m[i]))).sqrt();
        p_min = p_min.min(p);
    }
    (p_min / 40.0).clamp(DT / 8.0, 8.0 * DT)
}

/// Step limit from the closest approaching pair: 1/25 of the free-fall-like time.
fn pair_dt(sb: &Bodies) -> f64 {
    let n = sb.len();
    let mut lim = f64::INFINITY;
    for i in 0..n {
        if sb.dead[i] {
            continue;
        }
        for j in i + 1..n {
            if sb.dead[j] {
                continue;
            }
            let (dx, dy, dz) = (sb.x[j] - sb.x[i], sb.y[j] - sb.y[i], sb.z[j] - sb.z[i]);
            let r2 = dx * dx + dy * dy + dz * dz;
            let t = (r2 * r2.sqrt() / (G * (sb.m[i] + sb.m[j]))).sqrt();
            lim = lim.min(t / 25.0);
        }
    }
    lim.max(DT / 64.0)
}

fn kick(sb: &mut Bodies, dt: f64) {
    let n = sb.len();
    let mut a = vec![[0.0f64; 3]; n];
    for i in 0..n {
        if sb.dead[i] {
            continue;
        }
        for j in i + 1..n {
            if sb.dead[j] {
                continue;
            }
            let (dx, dy, dz) = (sb.x[j] - sb.x[i], sb.y[j] - sb.y[i], sb.z[j] - sb.z[i]);
            let r2 = dx * dx + dy * dy + dz * dz + 1e-12;
            let f = G / (r2 * r2.sqrt());
            for (k, d) in [dx, dy, dz].into_iter().enumerate() {
                a[i][k] += f * sb.m[j] * d;
                a[j][k] -= f * sb.m[i] * d;
            }
        }
    }
    for i in 0..n {
        sb.vx[i] += a[i][0] * dt;
        sb.vy[i] += a[i][1] * dt;
        sb.vz[i] += a[i][2] * dt;
    }
}

fn drift(sb: &mut Bodies, dt: f64) {
    for i in 0..sb.len() {
        sb.x[i] += sb.vx[i] * dt;
        sb.y[i] += sb.vy[i] * dt;
        sb.z[i] += sb.vz[i] * dt;
    }
}

/// Fourth-order Yoshida step (three kicks).
fn yoshida(sb: &mut Bodies, h: f64) {
    let c2 = 2f64.powf(1.0 / 3.0);
    let w1 = 1.0 / (2.0 - c2);
    let w0 = -c2 * w1;
    drift(sb, 0.5 * w1 * h);
    kick(sb, w1 * h);
    drift(sb, 0.5 * (w0 + w1) * h);
    kick(sb, w0 * h);
    drift(sb, 0.5 * (w0 + w1) * h);
    kick(sb, w1 * h);
    drift(sb, 0.5 * w1 * h);
}

/// Perfectly inelastic merger of any two bodies whose surfaces touch (a planet
/// touching a host is swallowed). Momentum is conserved.
fn merge_touching(sb: &mut Bodies) {
    let n = sb.len();
    for i in 0..n {
        if sb.dead[i] {
            continue;
        }
        for j in i + 1..n {
            if sb.dead[j] {
                continue;
            }
            let (dx, dy, dz) = (sb.x[j] - sb.x[i], sb.y[j] - sb.y[i], sb.z[j] - sb.z[i]);
            let rr = sb.r[i] + sb.r[j];
            if dx * dx + dy * dy + dz * dz >= rr * rr {
                continue;
            }
            let (k, o) = if sb.m[i] >= sb.m[j] { (i, j) } else { (j, i) };
            let mt = sb.m[k] + sb.m[o];
            let w = |a: f64, b: f64| (a * sb.m[k] + b * sb.m[o]) / mt;
            let (x, y, z) = (w(sb.x[k], sb.x[o]), w(sb.y[k], sb.y[o]), w(sb.z[k], sb.z[o]));
            let (vx, vy, vz) = (w(sb.vx[k], sb.vx[o]), w(sb.vy[k], sb.vy[o]), w(sb.vz[k], sb.vz[o]));
            sb.dead[o] = true;
            sb.set_mass(k, mt);
            sb.x[k] = x; sb.y[k] = y; sb.z[k] = z;
            sb.vx[k] = vx; sb.vy[k] = vy; sb.vz[k] = vz;
            sb.m_init[k] = sb.m_init[k].max(mt);
            if i == o {
                break;
            }
        }
    }
}

/// Giants (≥ this many M⊕) take part in the packing step.
const GIANT_MIN_M: f64 = 30.0;
/// Chance that a given adjacent giant pair starts packed.
const PACK_PROB: f64 = 0.4;
/// Packed spacing, in mutual Hill radii (the two-planet Hill limit is 3.46).
const PACK_DELTA: (f64, f64) = (1.8, 3.4);

/// Move the outer planet of some adjacent giant pairs inward to 1.8–3.4 mutual
/// Hill radii, keeping its eccentricity and phase. Returns the number moved.
fn youngify_giants(sb: &mut Bodies) -> usize {
    let mut rng = XorShift::new(sb.id.iter().fold(0x9E37_79B9u128, |h, &i| h.wrapping_mul(31).wrapping_add(i as u128)) ^ 0xE4_12_1A);
    let au = au_in_gm();
    let n = sb.len();
    sb.parent_cache.resize(n, None);
    let mut by_host: Vec<(usize, f64, usize)> = Vec::new(); // (host, a, planet)
    for i in 0..n {
        if sb.dead[i] || sb.kind[i] != BodyKind::Planet || sb.m[i] < GIANT_MIN_M {
            continue;
        }
        let Some(h) = find_parent(sb, i) else { continue };
        let (dx, dy, dz) = (sb.x[i] - sb.x[h], sb.y[i] - sb.y[h], sb.z[i] - sb.z[h]);
        by_host.push((h, (dx * dx + dy * dy + dz * dz).sqrt(), i));
    }
    by_host.sort_by(|p, q| p.0.cmp(&q.0).then(p.1.partial_cmp(&q.1).unwrap()));
    let mut moved = 0;
    for w in 1..by_host.len() {
        let (h, _, inner) = by_host[w - 1];
        let (h2, _, outer) = by_host[w];
        if h != h2 || rng.next_f64() >= PACK_PROB {
            continue;
        }
        let rel = |i: usize| {
            ([sb.x[i] - sb.x[h], sb.y[i] - sb.y[h], sb.z[i] - sb.z[h]],
             [sb.vx[i] - sb.vx[h], sb.vy[i] - sb.vy[h], sb.vz[i] - sb.vz[h]])
        };
        let (ri, _) = rel(inner);
        let (ro, vo) = rel(outer);
        let a_in = (ri[0] * ri[0] + ri[1] * ri[1] + ri[2] * ri[2]).sqrt();
        let a_out = (ro[0] * ro[0] + ro[1] * ro[1] + ro[2] * ro[2]).sqrt();
        let hill = 0.5 * (a_in + a_out) * ((sb.m[inner] + sb.m[outer]) / (3.0 * sb.m[h])).cbrt();
        let delta = PACK_DELTA.0 + (PACK_DELTA.1 - PACK_DELTA.0) * rng.next_f64();
        let a_new = a_in + delta * hill;
        if !(a_new < a_out) || a_new < 0.02 * au {
            continue;
        }
        let f = a_new / a_out;
        let vf = 1.0 / f.sqrt();
        sb.x[outer] = sb.x[h] + ro[0] * f;
        sb.y[outer] = sb.y[h] + ro[1] * f;
        sb.z[outer] = sb.z[h] + ro[2] * f;
        sb.vx[outer] = sb.vx[h] + vo[0] * vf;
        sb.vy[outer] = sb.vy[h] + vo[1] * vf;
        sb.vz[outer] = sb.vz[h] + vo[2] * vf;
        moved += 1;
    }
    moved
}
