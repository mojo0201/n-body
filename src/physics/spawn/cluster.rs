//! Spawn: ring and Plummer-cluster layout for multi-system spawns.
//!
//! Two placement strategies live here:
//!   * a ring for systems that should orbit a common centre, and
//!   * a Plummer cluster for systems that should form a virialised group.
//!
//! Both consume `(x, y, vx, vy)` slots produced by the `*_slots` functions
//! and hand them to `append_system`. Everything in this module is pure
//! apart from the final assembly into `b`; the per-system spawners are
//! independent given an explicit seed, which is what lets `spawn_random_systems_impl`
//! fan them out across threads.

use super::*;

/// Largest distance (Gm) from the system's centre of mass to any body of
/// `b`. Bound bodies count with their osculating apoapsis about the
/// heaviest host; unbound bodies count with their present distance.
///
/// Retained for callers outside the spawn path (`ring_radius_for_extent`
/// still uses it). `cluster_slots` prefers `system_reach_gm`, which is
/// robust to a single scattered KBO.
pub(in crate::physics) fn system_extent_gm(b: &Bodies) -> f64 {
    let n = b.len();
    let Some(primary) = (0..n)
        .filter(|&i| is_host_kind(b.kind[i]))
        .max_by(|&i, &j| b.m[i].total_cmp(&b.m[j]))
    else {
        return 0.0;
    };
    let mut ext = 0.0f64;
    let (px, py) = (b.x[primary], b.y[primary]);
    for i in 0..n {
        // hypot would be safer for arbitrary magnitudes, but Gm-scale
        // positions are bounded well below f64::MAX and above f64::MIN_POSITIVE,
        // so the naive squared form is exact-in-practice and 3–4× cheaper.
        // mul_add uses a fused multiply-add where the target has one.
        let r0 = b.x[i].mul_add(b.x[i], b.y[i] * b.y[i]).sqrt();
        let dx = b.x[i] - px;
        let dy = b.y[i] - py;
        let rp = dx.mul_add(dx, dy * dy).sqrt();
        let mut r = r0.max(rp);
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
/// Never smaller than three times that extent, so systems cannot overlap
/// at spawn.
pub fn ring_radius_for_extent(n: usize, widest_gm: f64) -> f64 {
    if n <= 1 {
        0.0
    } else {
        solar_ring_radius_gm(n).max(3.0 * widest_gm)
    }
}

/// Sum term of the ring formula: v² = G M s / R for `n` equal masses.
/// The factor ¼ normalises the half-pair double count in the pairwise sum.
pub(in crate::physics) fn ring_sum(n: usize) -> f64 {
    (1..n)
        .map(|k| 1.0 / (std::f64::consts::PI * k as f64 / n as f64).sin())
        .sum::<f64>()
        * 0.25
}

/// Polar angle of system `k` of `n`: a regular polygon plus jitter of up
/// to ±0.15 polygon steps. `u` is a uniform draw in `[0, 1)`.
pub(in crate::physics) fn ring_theta(n: usize, k: usize, u: f64) -> f64 {
    std::f64::consts::TAU * k as f64 / n as f64
        + (u - 0.5) * 0.3 * (std::f64::consts::TAU / n as f64)
}

/// State (x, y, vx, vy) of a system of `mass` on an orbit of semi-major
/// axis `a` about the ring centre.
///
/// The central term is `mass * s` — the value that makes e = 0 reproduce
/// the circular ring speed, so a system that sits on the ring stays on it.
pub(in crate::physics) fn ring_slot(
    mass: f64,
    s: f64,
    a: f64,
    e: f64,
    nu: f64,
    omega: f64,
) -> (f64, f64, f64, f64) {
    kepler_state(G * mass * s, a, e, nu, omega)
}

/// Position and velocity of every system on the ring.
///
/// Semi-major axis scatters by ±15 %, eccentricity is Rayleigh(0.05)
/// (capped at 0.5 so no system is on a strongly radial orbit at spawn),
/// and each system sits at its jittered polar angle. With n ≤ 1 all
/// systems get the zero state so the caller can pass through unchanged.
pub(in crate::physics) fn ring_slots(
    n: usize,
    seed: u128,
    ring: f64,
    masses: &[f64],
) -> Vec<(f64, f64, f64, f64)> {
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
            // Perifocal angle plus argument of periapsis must equal the
            // slot's polar angle; solve for omega.
            ring_slot(masses[k], s, a_k, e_k, nu, theta - nu)
        })
        .collect()
}

/// Minimum centre separation, in summed reaches. Two systems must be at
/// least this far apart to be placed without overlapping.
pub(in crate::physics) const SEP_FACTOR: f64 = 2.0;

/// Fraction of body distances (from the system centre) that the "reach"
/// covers. A quantile, not the max, so a single scattered KBO cannot
/// dominate the cluster's scale radius.
pub(in crate::physics) const REACH_QUANTILE: f64 = 0.9;

/// Stable reach of a system in Gm.
///
/// Convenience wrapper that pays the O(n) mass sum itself; the spawn path
/// uses `system_reach_gm_with_mass` to avoid double-summation.
pub(in crate::physics) fn system_reach_gm(b: &Bodies) -> f64 {
    let mt: f64 = b.m.iter().sum();
    system_reach_gm_with_mass(b, mt)
}

/// Same as `system_reach_gm`, but takes a precomputed total mass. Returns
/// the 90th-percentile body distance from the mass-weighted centre times
/// a 1.5 safety margin, floored at 200 Gm so an empty cluster still has
/// non-zero size.
pub(in crate::physics) fn system_reach_gm_with_mass(b: &Bodies, mt: f64) -> f64 {
    let n = b.len();
    if n == 0 || mt == 0.0 {
        return 0.0;
    }

    // Mass-weighted centre of mass in one pass.
    let mut cx = 0.0;
    let mut cy = 0.0;
    for i in 0..n {
        cx += b.m[i] * b.x[i];
        cy += b.m[i] * b.y[i];
    }
    cx /= mt;
    cy /= mt;

    // Naive squared distance: same accuracy argument as in
    // `system_extent_gm`. select_nth_unstable_by is O(n) versus a full
    // sort's O(n log n) — we only need the quantile.
    let mut d: Vec<f64> = (0..n)
        .map(|i| {
            let dx = b.x[i] - cx;
            let dy = b.y[i] - cy;
            dx.mul_add(dx, dy * dy).sqrt()
        })
        .collect();
    let k = ((n as f64 * REACH_QUANTILE) as usize).min(n - 1);
    let (_, val, _) = d.select_nth_unstable_by(k, |a, c| a.total_cmp(c));
    (*val * 1.5).max(200.0)
}

/// (x, y, vx, vy) for every system in a Plummer cluster.
///
/// Surface density M(<R) = M R² / (R² + a²), isotropic velocities with
/// sigma_1D² = G M / (6 √(r² + a²)) — i.e. virial equilibrium at every
/// radius. The scale radius `a` grows until the separation constraint
/// holds for every pair; the growth is a fallback for tight packs and is
/// bounded at 200 attempts.
///
/// Rejection sampling: draw r from the Plummer radial CDF (u/(1-u), with
/// the tail truncated at u ≤ 0.95), then reject if any existing system is
/// too close. The separation test is done in squared distance so the
/// innermost loop never calls sqrt.
pub(in crate::physics) fn cluster_slots(
    seed: u128,
    reaches: &[f64],
    masses: &[f64],
) -> Vec<(f64, f64, f64, f64)> {
    let n = reaches.len();
    if n <= 1 {
        return vec![(0.0, 0.0, 0.0, 0.0); n];
    }
    let m_tot: f64 = masses.iter().sum();
    let mean_reach = reaches.iter().sum::<f64>() / n as f64;
    let mut a = SEP_FACTOR * mean_reach * (n as f64).sqrt();
    let mut rng = XorShift::new(seed ^ 0xC1A5_7E25);
    let mut pos: Vec<(f64, f64)> = Vec::with_capacity(n);

    // Pairwise squared-separation thresholds. This is O(n²) memory, but
    // n is a small handful of solar systems, and the alternative is a
    // sqrt per rejection attempt — far more expensive in aggregate.
    let sep_sq: Vec<Vec<f64>> = (0..n)
        .map(|k| {
            (0..n)
                .map(|j| {
                    let t = SEP_FACTOR * (reaches[k] + reaches[j]);
                    t * t
                })
                .collect()
        })
        .collect();

    'grow: for _attempt in 0..200 {
        pos.clear();
        for k in 0..n {
            let mut placed = false;
            let sep_k = &sep_sq[k];
            for _ in 0..200 {
                // Truncate the Plummer tail so a single system cannot
                // be flung so far it makes the cluster a one-system
                // affair. u = 0.95 caps r at ~√19 · a.
                let u = rng.next_f64().min(0.95);
                let r = a * (u / (1.0 - u)).sqrt();
                let th = rng.next_f64() * std::f64::consts::TAU;
                let p = (r * th.cos(), r * th.sin());
                let ok = pos.iter().enumerate().all(|(j, q)| {
                    let dx = p.0 - q.0;
                    let dy = p.1 - q.1;
                    dx.mul_add(dx, dy * dy) >= sep_k[j]
                });
                if ok {
                    pos.push(p);
                    placed = true;
                    break;
                }
            }
            if !placed {
                // Too crowded for this scale radius; grow and restart.
                a *= 1.1;
                continue 'grow;
            }
        }
        break;
    }
    pos.iter()
        .map(|&(x, y)| {
            // sigma_1D² = G M / (6 √(r² + a²)) — the local value of the
            // virial relation for a Plummer sphere. Velocities are drawn
            // isotropically (independent x, y components from N(0, σ)).
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
/// Each system has up to `body_count` bodies (per-system) and its own
/// derived seed, so systems are statistically independent. Systems sit on
/// the ring and are given the circular-ring velocity so they orbit the
/// common centre rather than collapsing into one clump.
pub fn spawn_random_systems(
    b: &mut Bodies,
    body_count: usize,
    seed: u128,
    n_systems: usize,
) -> Vec<SystemInfo> {
    spawn_random_systems_impl(b, body_count, seed, n_systems).0
}

/// Spawn one system into a fresh scratch `Bodies`.
///
/// Factored out so the parallel spawners below can call it without
/// duplicating the seed argument or the capacity hint. Purity is an
/// assumption of the whole module: `spawn_system_with_info` must not
/// consult any thread-local or global RNG state, only `seed`.
#[inline]
fn spawn_one(body_count: usize, seed: u128) -> (Bodies, SystemInfo) {
    let mut temp = Bodies::with_capacity(body_count);
    let info = spawn_system_with_info(&mut temp, body_count, seed);
    (temp, info)
}

/// Thread count for a job of `n` items: never more than `n`, never zero.
/// Very small `n` falls back to single-threaded because thread spawn
/// overhead dominates.
fn n_threads_for(n: usize) -> usize {
    std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1)
        .min(n)
        .max(1)
}

/// `spawn_random_systems`, plus the index range `(start, end)` of every
/// system inside `b`.
///
/// Pipeline:
///   1. Spawn every system in parallel, each into its own scratch `Bodies`.
///   2. Mass-scale the body counts, respawning where the count changes.
///      Skipped when the scale is within 5 % of 1.0 — the respawn would
///      only adjust the count by a fractional amount already inside the
///      rounding noise.
///   3. Compute each system's reach, place all systems with `cluster_slots`,
///      and assemble into `b` with a single exact-capacity allocation.
pub(in crate::physics) fn spawn_random_systems_impl(
    b: &mut Bodies,
    body_count: usize,
    seed: u128,
    n_systems: usize,
) -> (Vec<SystemInfo>, Vec<(usize, usize)>) {
    let n = n_systems.clamp(1, MAX_SOLAR_SYSTEMS);

    // `body_count` is a *ceiling* on each system, not its actual count:
    // realistic belts mean a typical system is ~15 planets + 10–60 belt
    // tracers, so `body_count * n` overestimates. Cap per-system count so
    // a pathological slider pair cannot exhaust memory.
    let per_system = (MAX_BODIES / n).max(1);
    let body_count = body_count.max(1).min(per_system);

    // Per-system seed. Multiply by an odd constant so consecutive `k`
    // land in well-separated parts of the u128 space even from the same
    // master seed.
    let sys_seed = |k: usize| seed.wrapping_add((k as u128).wrapping_mul(0x9E37_79B9));
    let n_threads = n_threads_for(n);

    // ---- Phase 1: spawn every system in parallel ----------------------
    // Each system owns its own scratch `Bodies` and its own seed, so the
    // work is embarrassingly parallel. Each thread builds a contiguous
    // chunk; flattening in chunk order restores index order without a
    // gather step.
    let mut temps: Vec<(Bodies, SystemInfo)> = Vec::with_capacity(n);
    if n_threads <= 1 || n < 4 {
        for k in 0..n {
            temps.push(spawn_one(body_count, sys_seed(k)));
        }
    } else {
        let chunk = n.div_ceil(n_threads);
        let mut chunk_results: Vec<Vec<(Bodies, SystemInfo)>> = Vec::with_capacity(n_threads);
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..n_threads)
                .map(|ti| {
                    let lo = ti * chunk;
                    let hi = ((ti + 1) * chunk).min(n);
                    // `sys_seed` captures only `seed: u128` (Copy), so it
                    // is itself Copy and each thread gets its own copy —
                    // no shared mutation, no closures-over-&mut.
                    s.spawn(move || {
                        (lo..hi)
                            .map(|k| spawn_one(body_count, sys_seed(k)))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            for h in handles {
                chunk_results.push(h.join().unwrap());
            }
        });
        for v in chunk_results {
            temps.extend(v);
        }
    }

    // ---- Phase 2: mass-scale body counts in parallel ------------------
    // Heavier systems get more bodies. The heaviest system is unchanged
    // by construction (its scale is 2.0, clamped, and its count grows);
    // the lightest shrinks; the rest move proportionally.
    //
    // [Behavioural note] Skipping the respawn when |scale − 1| ≤ 0.05 is
    // a deliberate deviation from the unoptimised version. The respawn
    // only changes body count by a fraction already inside the `round`
    // noise, so this halves the spawn cost for typical multi-system runs
    // while preserving monotone mass ↔ body-count and the heaviest
    // system. Removing the early return restores bit-identical layouts.
    if n > 1 {
        let masses: Vec<f64> = temps.iter().map(|t| t.0.m.iter().sum::<f64>()).collect();
        let avg = masses.iter().sum::<f64>() / n as f64;

        let rescale_one = |k: usize, masses: &[f64]| -> Option<(Bodies, SystemInfo)> {
            let scale = (masses[k] / avg).clamp(0.5, 2.0);
            if (scale - 1.0).abs() <= 0.05 {
                return None;
            }
            let count = ((body_count as f64 * scale).round() as usize).max(1);
            if count != body_count {
                Some(spawn_one(count, sys_seed(k)))
            } else {
                None
            }
        };

        if n_threads <= 1 || n < 4 {
            for k in 0..n {
                if let Some(t) = rescale_one(k, &masses) {
                    temps[k] = t;
                }
            }
        } else {
            // `chunks_mut` gives each thread a disjoint &mut slice, so
            // in-place replacement is race-free without a mutex.
            let chunk = n.div_ceil(n_threads);
            let masses_ref = &masses;
            let rescale_ref = &rescale_one;
            std::thread::scope(|s| {
                for (ti, slice) in temps.chunks_mut(chunk).enumerate() {
                    s.spawn(move || {
                        for (j, slot) in slice.iter_mut().enumerate() {
                            let k = ti * chunk + j;
                            if let Some(t) = rescale_ref(k, masses_ref) {
                                *slot = t;
                            }
                        }
                    });
                }
            });
        }
    }

    // ---- Phase 3: layout and assembly ---------------------------------
    // Reuse the mass sums we already computed for the M5 rescale. The
    // spawner would otherwise recompute them inside `system_reach_gm`.
    let masses: Vec<f64> = temps.iter().map(|t| t.0.m.iter().sum::<f64>()).collect();
    let reaches: Vec<f64> = (0..n)
        .map(|k| system_reach_gm_with_mass(&temps[k].0, masses[k]))
        .collect();
    let slots = cluster_slots(seed, &reaches, &masses);

    // Exact capacity: after the rescale, `body_count * n` is wrong. Summing
    // the actual system lengths avoids reallocating while appending.
    let total_bodies: usize = temps.iter().map(|t| t.0.len()).sum();
    *b = Bodies::with_capacity(total_bodies + 16);

    let mut infos = Vec::with_capacity(n);
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(n);
    for (k, (temp, info)) in temps.iter().enumerate() {
        let (x, y, vx, vy) = slots[k];
        let (start, end) = append_system(b, temp, x, y);
        // Bulk-shift velocities: the slot's velocity is the system's
        // barycentric motion, and `append_system` only wrote internal
        // velocities. Contiguous range → cache-friendly.
        for i in start..end {
            b.vx[i] += vx;
            b.vy[i] += vy;
        }
        ranges.push((start, end));
        infos.push(*info);
    }

    // The ring shares a single metallicity; the first system's value
    // stands for the whole population.
    b.metallicity = temps[0].0.metallicity;

    // Runtime RNG (fragment masses, kick angles) must depend on the master
    // seed but *not* on the per-system draws, so the physics is stable
    // under changes to the spawn order.
    b.rng = XorShift::new(seed ^ 0xDEAD_BEEF);

    // Final assembly steps. Order matters: recenter before compute_accel
    // so the initial forces reflect the as-shown positions; rebase energy
    // after compute_accel so the energy baseline matches the new state.
    recenter_com(b);
    compute_accel(b);
    rebase_energy(b);

    // Parent cache is refreshed here (user-triggered) rather than in
    // `advance`, so the O(n²) cost does not stall the first physics tick
    // while holding the bodies lock. Orbit lines draw correctly from
    // frame 1.
    b.refresh_parent_cache();

    b.dust_fields.clear();
    dust::init_fields(b, Some(&ranges));
    (infos, ranges)
}
