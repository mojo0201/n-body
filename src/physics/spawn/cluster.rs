//! Spawn: ring and Plummer-cluster layout for multi-system spawns.

use super::*;

/// Largest distance (Gm) from the system's centre of mass to any body of
/// `b`. A body on a bound orbit about the heaviest host counts with its
/// osculating apoapsis, an unbound body with its present distance.
pub(in crate::physics) fn system_extent_gm(b: &Bodies) -> f64 {
    let n = b.len();
    let Some(primary) = (0..n)
        .filter(|&i| is_host_kind(b.kind[i]))
        .max_by(|&i, &j| b.m[i].total_cmp(&b.m[j]))
    else {
        return 0.0;
    };
    let mut ext = 0.0f64;
    for i in 0..n {
        let mut r = b.x[i]
            .hypot(b.y[i])
            .max((b.x[i] - b.x[primary]).hypot(b.y[i] - b.y[primary]));
        if i != primary {
            let o = orbit_info(b, i, primary);
            if o.bound && o.apo.is_finite() {
                r = r.max(o.apo);
            }
        }
        ext = ext.max(r);
    }
    ext
}

/// Ring radius for `n` systems whose widest one reaches `widest_gm`.
/// The ring is never smaller than 3 times that extent.
pub fn ring_radius_for_extent(n: usize, widest_gm: f64) -> f64 {
    if n <= 1 {
        0.0
    } else {
        solar_ring_radius_gm(n).max(3.0 * widest_gm)
    }
}

/// Sum term of the ring formula: v^2 = G M s / R for n equal masses.
pub(in crate::physics) fn ring_sum(n: usize) -> f64 {
    (1..n)
        .map(|k| 1.0 / (std::f64::consts::PI * k as f64 / n as f64).sin())
        .sum::<f64>()
        * 0.25
}

/// Polar angle of system `k` of `n`: a regular polygon plus a jitter of
/// up to +-0.15 polygon steps. `u` is a uniform draw in [0, 1).
pub(in crate::physics) fn ring_theta(n: usize, k: usize, u: f64) -> f64 {
    std::f64::consts::TAU * k as f64 / n as f64
        + (u - 0.5) * 0.3 * (std::f64::consts::TAU / n as f64)
}

/// State (x, y, vx, vy) of a system of `mass` on an orbit of semi-major
/// axis `a` about the ring centre. The central term is `mass * s`, the
/// value that makes e = 0 reproduce the circular ring speed.
pub(in crate::physics) fn ring_slot(mass: f64, s: f64, a: f64, e: f64, nu: f64, omega: f64) -> (f64, f64, f64, f64) {
    kepler_state(G * mass * s, a, e, nu, omega)
}

/// Position and velocity of every system on the ring. The semi-major axis
/// scatters by +-15 %, the eccentricity is Rayleigh(0.05), and the system
/// sits at its jittered polar angle. One system gets zero state.
pub(in crate::physics) fn ring_slots(n: usize, seed: u32, ring: f64, masses: &[f64]) -> Vec<(f64, f64, f64, f64)> {
    if n <= 1 {
        return vec![(0.0, 0.0, 0.0, 0.0); n];
    }
    let mut rng = XorShift::new(seed ^ 0x51AB_C0DE);
    let s = ring_sum(n);
    (0..n)
        .map(|k| {
            let theta = ring_theta(n, k, rng.next_f64());
            let a_k = ring * (0.85 + 0.30 * rng.next_f64());
            let e_k = sample_rayleigh(&mut rng, 0.05).min(0.5);
            let nu = true_anomaly(
                solve_kepler(rng.next_f64() * std::f64::consts::TAU, e_k),
                e_k,
            );
            // The polar angle is nu + omega, so omega follows from theta.
            ring_slot(masses[k], s, a_k, e_k, nu, theta - nu)
        })
        .collect()
}

/// Minimum centre separation, in summed reaches.
pub(in crate::physics) const SEP_FACTOR: f64 = 2.0;

/// Reach = this quantile of body distances from the system centre. Using a
/// quantile (not the max) keeps one scattered KBO from sizing the cluster.
pub(in crate::physics) const REACH_QUANTILE: f64 = 0.9;

/// Stable reach of a system in Gm: 90 % quantile of distance from its
/// centre of mass, times 1.5 margin. Replaces `system_extent_gm` for layout.
pub(in crate::physics) fn system_reach_gm(b: &Bodies) -> f64 {
    let n = b.len();
    if n == 0 {
        return 0.0;
    }
    let mt: f64 = b.m.iter().sum();
    let cx = (0..n).map(|i| b.m[i] * b.x[i]).sum::<f64>() / mt;
    let cy = (0..n).map(|i| b.m[i] * b.y[i]).sum::<f64>() / mt;
    let mut d: Vec<f64> = (0..n).map(|i| (b.x[i] - cx).hypot(b.y[i] - cy)).collect();
    d.sort_by(|a, c| a.total_cmp(c));
    let k = ((n as f64 * REACH_QUANTILE) as usize).min(n - 1);
    (d[k] * 1.5).max(200.0)
}

/// (x, y, vx, vy) for every system. Plummer surface density
/// M(<R) = M R^2 / (R^2 + a^2), isotropic velocities with
/// sigma_1D^2 = G M / (6 sqrt(r^2 + a^2)) (virial equilibrium).
/// The scale radius `a` grows until the separation constraint holds.
pub(in crate::physics) fn cluster_slots(seed: u32, reaches: &[f64], masses: &[f64]) -> Vec<(f64, f64, f64, f64)> {
    let n = reaches.len();
    if n <= 1 {
        return vec![(0.0, 0.0, 0.0, 0.0); n];
    }
    let m_tot: f64 = masses.iter().sum();
    let mean_reach = reaches.iter().sum::<f64>() / n as f64;
    let mut a = SEP_FACTOR * mean_reach * (n as f64).sqrt();
    let mut rng = XorShift::new(seed ^ 0xC1A5_7E25);
    let mut pos: Vec<(f64, f64)> = Vec::new();
    'grow: for _attempt in 0..200 {
        pos.clear();
        for k in 0..n {
            let mut placed = false;
            for _ in 0..200 {
                let u = rng.next_f64().min(0.95); // truncate the tail
                let r = a * (u / (1.0 - u)).sqrt();
                let th = rng.next_f64() * std::f64::consts::TAU;
                let p = (r * th.cos(), r * th.sin());
                let ok = pos.iter().enumerate().all(|(j, q)| {
                    (p.0 - q.0).hypot(p.1 - q.1) >= SEP_FACTOR * (reaches[k] + reaches[j])
                });
                if ok {
                    pos.push(p);
                    placed = true;
                    break;
                }
            }
            if !placed {
                a *= 1.1;
                continue 'grow;
            }
        }
        break;
    }
    pos.iter()
        .map(|&(x, y)| {
            let sigma = (G * m_tot / (6.0 * (x * x + y * y + a * a).sqrt())).sqrt();
            (
                x,
                y,
                sample_normal(&mut rng, 0.0, sigma),
                sample_normal(&mut rng, 0.0, sigma),
            )
        })
        .collect()
}

/// Spawn `n_systems` independent random systems on a ring of radius
/// `solar_ring_radius_gm(n_systems)`.
///
/// Each system has `body_count` bodies (per-system) and its own derived
/// seed, so the systems are statistically independent. The systems sit on
/// the ring and are given the circular-ring velocity so they (approximately)
/// orbit the common centre rather than falling into one clump. The ring
/// formula uses each system's own mass for its speed.
pub fn spawn_random_systems(
    b: &mut Bodies,
    body_count: usize,
    seed: u32,
    n_systems: usize,
) -> Vec<SystemInfo> {
    spawn_random_systems_impl(b, body_count, seed, n_systems).0
}

/// `spawn_random_systems`, and the index range of every system.
pub(in crate::physics) fn spawn_random_systems_impl(
    b: &mut Bodies,
    body_count: usize,
    seed: u32,
    n_systems: usize,
) -> (Vec<SystemInfo>, Vec<(usize, usize)>) {
    let n = n_systems.clamp(1, MAX_SOLAR_SYSTEMS);
    // With realistic belts, `body_count` is a *ceiling* on each system,
    // not its actual count (a typical system has ~15 planets + ~10–60
    // belt tracers). So `body_count * n` overestimates. Still, cap it
    // to `MAX_BODIES` so a pathological slider pair cannot exhaust
    // memory.
    let per_system = (MAX_BODIES / n).max(1);
    let body_count = body_count.max(1).min(per_system);
    *b = Bodies::with_capacity(body_count * n + 16);

    // Per-system seed. Multiply by an odd constant so consecutive k give
    // well-separated draws even from the same master seed.
    let sys_seed = |k: usize| seed.wrapping_add((k as u32).wrapping_mul(0x9E37_79B9));

    // Spawn into scratch `Bodies`: `spawn_system_with_info` recentres the
    // COM to the origin, and we translate it to its slot later.
    let mut temps: Vec<(Bodies, SystemInfo)> = Vec::with_capacity(n);
    for k in 0..n {
        let mut temp = Bodies::with_capacity(body_count);
        let info = spawn_system_with_info(&mut temp, body_count, sys_seed(k));
        temps.push((temp, info));
    }

    // M5: heavier systems get more bodies. Spawn again with the scaled
    // count (same seed, so the same primary). One system is left alone.
    if n > 1 {
        let masses: Vec<f64> = temps.iter().map(|t| t.0.m.iter().sum::<f64>()).collect();
        let avg = masses.iter().sum::<f64>() / n as f64;
        for k in 0..n {
            let scale = (masses[k] / avg).clamp(0.5, 2.0);
            let count = ((body_count as f64 * scale).round() as usize).max(1);
            if count != body_count {
                let mut temp = Bodies::with_capacity(count);
                let info = spawn_system_with_info(&mut temp, count, sys_seed(k));
                temps[k] = (temp, info);
            }
        }
    }

    let reaches: Vec<f64> = temps.iter().map(|t| system_reach_gm(&t.0)).collect();
    let masses: Vec<f64> = temps.iter().map(|t| t.0.m.iter().sum::<f64>()).collect();
    let slots = cluster_slots(seed, &reaches, &masses);

    let mut infos = Vec::with_capacity(n);
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(n);
    for (k, (temp, info)) in temps.iter().enumerate() {
        let (x, y, vx, vy) = slots[k];
        let (start, end) = append_system(b, temp, x, y);
        for i in start..end {
            b.vx[i] += vx;
            b.vy[i] += vy;
        }
        ranges.push((start, end));
        infos.push(*info);
    }
    // The first system's metallicity stands for the ring.
    b.metallicity = temps[0].0.metallicity;

    // Runtime RNG (fragment masses, kick angles) must depend on the master seed.
    b.rng = XorShift::new(seed ^ 0xDEAD_BEEF);
    recenter_com(b);
    compute_accel(b);
    rebase_energy(b);
    // Refresh the parent cache so orbit lines draw correctly on frame 1.
    // Do it during spawn (user-triggered) rather than in `advance`: the
    // O(n²) cost would otherwise stall the first physics tick after spawn
    // while holding the bodies lock.
    b.refresh_parent_cache();
    b.dust_fields.clear();
    dust::init_fields(b, Some(&ranges));
    (infos, ranges)
}
