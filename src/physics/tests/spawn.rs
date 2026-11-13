//! Tests: initial conditions, multi-system layout, spawn variants.

use super::super::*;
use super::shared::*;

// ===========================================================================
// g — initial-condition statistics
// ===========================================================================
#[test]
fn g_ic_random_architecture_sane() {
    // Belts are common (see `debris_belt_occurrence_fraction`). What must
    // always hold is that the count never exceeds the ceiling and that
    // small bodies are present.
    // Belt occurrence is probabilistic (debris_belt_occurrence_fraction),
    // and ~15 % of primaries are remnants whose belts are usually empty.
    // Expected belt fraction: 0.85 × 0.85 + 0.15 × 0.30 ≈ 0.77.
    let mut with_belt = 0;
    for seed in 0..200u128 {
        let mut b = Bodies::with_capacity(100);
        spawn_system(&mut b, 100, seed);
        assert!(b.len() <= 100, "seed {} produced {} bodies", seed, b.len());
        assert!(primary_index(&b).is_some(), "seed {} has no primary", seed);
        if (1..b.len()).any(|i| b.origin[i].is_belt()) {
            with_belt += 1;
        }
    }
    assert!(
        with_belt >= 140,
        "only {}/200 seeds produced a belt",
        with_belt
    );
}

#[test]
fn g_ic_primary_diversity() {
    let mut seen = [false; 5];
    for seed in 0..1500u128 {
        let mut b = Bodies::with_capacity(30);
        spawn_system(&mut b, 30, seed);
        match b.kind[0] {
            BodyKind::Star => seen[0] = true,
            BodyKind::BrownDwarf => seen[1] = true,
            BodyKind::WhiteDwarf => seen[2] = true,
            BodyKind::NeutronStar => seen[3] = true,
            BodyKind::BlackHole => seen[4] = true,
            _ => {}
        }
    }
    assert!(seen[0], "no MS stars in 1500 seeds");
    assert!(seen[1], "no brown dwarfs in 1500 seeds");
    assert!(seen[2], "no white dwarfs in 1500 seeds");
    assert!(seen[3], "no neutron stars in 1500 seeds");
    assert!(seen[4], "no black holes in 1500 seeds");
}

#[test]
fn g_ic_primary_imf_mass_vs_remnant_split() {
    // Before aging: 85 % of primaries are stellar-mass draws.
    let mut rng = XorShift::new(12345);
    let n = 20_000;
    let stars = (0..n)
        .filter(|_| sample_primary(&mut rng).1 == BodyKind::Star)
        .count();
    let f = stars as f64 / n as f64;
    assert!((0.82..=0.88).contains(&f), "stellar-mass fraction {}", f);
}

#[test]
fn g_ic_primary_mix_realistic() {
    // After aging: most primaries are still Stars, but a real share of the
    // 85 % stellar draws has already died, and every compact kind occurs.
    let n = 3000u128;
    let (mut ms, mut bd, mut wd, mut ns, mut bh) = (0, 0, 0, 0, 0);
    for seed in 0..n {
        let mut b = Bodies::with_capacity(3);
        spawn_system(&mut b, 3, seed);
        match b.kind[0] {
            BodyKind::Star => ms += 1,
            BodyKind::BrownDwarf => bd += 1,
            BodyKind::WhiteDwarf => wd += 1,
            BodyKind::NeutronStar => ns += 1,
            BodyKind::BlackHole => bh += 1,
            _ => panic!("primary must be a host"),
        }
    }
    let f = |c: i32| c as f64 / n as f64;
    assert!((0.50..=0.85).contains(&f(ms)), "star fraction {}", f(ms));
    assert!(
        (0.05..=0.15).contains(&f(bd)),
        "brown dwarf fraction {}",
        f(bd)
    );
    assert!(wd > 0 && ns > 0 && bh > 0, "wd {} ns {} bh {}", wd, ns, bh);
    // Remnants now come only from aged stars.
    assert!(f(wd) > 0.0, "white dwarf fraction {}", f(wd));
}

#[test]
fn g_ic_imf_median_mass() {
    let mut ms: Vec<f64> = Vec::new();
    for seed in 0..1000u128 {
        let mut b = Bodies::with_capacity(30);
        spawn_system(&mut b, 30, seed);
        if b.kind[0] == BodyKind::Star {
            ms.push(b.m[0] / M_SUN_ME);
        }
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ms[ms.len() / 2];
    println!("IMF median mass: {:.3} Msun over {} stars", med, ms.len());
    assert!(med > 0.20 && med < 0.60, "median {}", med);
}

#[test]
fn g_ic_hot_jupiter_rate() {
    let au = au_in_gm();
    let (mut hosts, mut hj) = (0u32, 0u32);
    for seed in 0..2000u128 {
        let mut b = Bodies::with_capacity(30);
        spawn_system(&mut b, 30, seed);
        assert!(b.len() <= 30, "seed {}: {} bodies", seed, b.len());
        let m_sun = b.m[0] / M_SUN_ME;
        if b.kind[0] != BodyKind::Star || !(0.5..=2.0).contains(&m_sun) {
            continue;
        }
        hosts += 1;
        let found = (1..b.len()).any(|i| {
            let d = ((b.x[i] - b.x[0]).powi(2) + (b.y[i] - b.y[0]).powi(2)).sqrt();
            d / au < 0.1 && b.m[i] > 50.0 && b.m[i] < 0.1 * M_SUN_ME
        });
        if found {
            hj += 1;
        }
    }
    let f = hj as f64 / hosts as f64;
    println!("hot Jupiter rate: {}/{} = {:.4}", hj, hosts, f);
    assert!(f > 0.003 && f < 0.03, "rate {}", f);
}

// ===========================================================================
// j — compaction invariants
// ===========================================================================
#[test]
fn j_compact_remaps_parent_cache() {
    let mut b = Bodies::with_capacity(8);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(10.0, 0.0, 0.0, 0.1, 1.0, BodyKind::Planet);
    b.push(20.0, 0.0, 0.0, 0.1, 1.0, BodyKind::Planet);
    b.parent_cache = vec![None, Some(0), Some(1)];
    b.dead[1] = true;
    b.compact();
    assert_eq!(b.len(), 2);
    assert_eq!(b.parent_cache, vec![None, None], "dead parent -> None");

    let mut c = Bodies::with_capacity(8);
    for k in 0..4 {
        c.push(k as f64, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    }
    c.parent_cache = vec![None, None, Some(3), Some(2)];
    c.dead[0] = true;
    c.compact();
    assert_eq!(c.parent_cache, vec![None, Some(2), Some(1)]);
}

// ===========================================================================
// t — multi-system spawn
// ===========================================================================
/// Each of the 3 spawned suns must remain near the ring radius after a few
/// steps; the mutual gravity of the ring is in equilibrium with the
/// tangential velocity that `spawn_solar_systems` gave them.
#[test]
fn t_multi_system_ring() {
    let mut b = Bodies::with_capacity(32);
    let (_radius, ranges) = spawn_solar_systems_impl(&mut b, 3);
    // Record initial sun positions.
    let suns: Vec<usize> = (0..b.len())
        .filter(|&i| b.kind[i] == BodyKind::Star)
        .collect();
    assert_eq!(suns.len(), 3);
    let initial: Vec<(f64, f64)> = suns.iter().map(|&i| (b.x[i], b.y[i])).collect();
    for _ in 0..50 {
        advance(&mut b, DT);
    }
    // After 50 steps, each sun is still inside the same cluster: within
    // 3x its initial distance from the origin (the cluster is virialised,
    // so it moves but does not fly apart on this timescale).
    for (k, &i) in suns.iter().enumerate() {
        let (x0, y0) = initial[k];
        let r0 = x0.hypot(y0).max(1.0);
        let r = b.x[i].hypot(b.y[i]);
        assert!(r < 3.0 * r0, "sun {i}: r={r:.0} from r0={r0:.0}",);
    }
    let _ = ranges;
}

/// Every planet in a 3-system spawn must be bound to its own sun (not to
/// a foreign one). Skip belt and debris bodies — they may be unbound or
/// have high eccentricity by design.
#[test]
fn t_multi_system_planets_bound_to_own_sun() {
    let mut b = Bodies::with_capacity(32);
    spawn_solar_systems(&mut b, 3);
    let suns: Vec<usize> = (0..b.len())
        .filter(|&i| b.kind[i] == BodyKind::Star)
        .collect();
    assert_eq!(suns.len(), 3);
    for i in 0..b.len() {
        if b.kind[i] != BodyKind::Planet {
            continue;
        }
        if b.origin[i] != BodyOrigin::Spawned {
            continue;
        }
        if b.m[i] < 0.01 {
            continue;
        }
        // Skip moons: a moon's velocity is dominated by its own orbit
        // about its planet, so "bound to nearest sun" is the wrong check.
        if is_moon(&b, i) {
            continue;
        }
        // Nearest sun by euclidean distance.
        let mut nearest = suns[0];
        let mut best = f64::INFINITY;
        for &k in &suns {
            let d = (b.x[i] - b.x[k]).hypot(b.y[i] - b.y[k]);
            if d < best {
                best = d;
                nearest = k;
            }
        }
        let o = orbit_info(&b, i, nearest);
        assert!(o.bound, "planet {} not bound to nearest sun {}", i, nearest);
        // Preset eccentricities: Mercury 0.2056, most < 0.1. Cap at 0.3.
        assert!(o.ecc < 0.3, "planet {} ecc {:.4} too high", i, o.ecc);
    }
}

// ===========================================================================
// aa — spawn variants
// ===========================================================================
#[test]
fn aa_random_systems_ring_and_count() {
    let mut b = Bodies::with_capacity(64);
    let infos = spawn_random_systems(&mut b, 20, 42, 3);
    // `n=20` is a ceiling, not a target: with realistic spawning, each
    // system typically ends up with 5–15 bodies. 3 systems → 15–45
    // bodies expected; accept 10..=60.
    assert!(
        (10..=60).contains(&b.len()),
        "3 systems of n=20 produced {} bodies",
        b.len()
    );
    assert_eq!(infos.len(), 3);
    let max_r: f64 = (0..b.len())
        .map(|i| b.x[i].hypot(b.y[i]))
        .fold(0.0, f64::max);
    assert!(
        max_r > 1000.0,
        "systems should not pile at origin: {}",
        max_r
    );
}

#[test]
fn solar_preset_loads_at_present_age() {
    let preset = solar_system_preset();
    assert_eq!(preset.name, "Solar System");
    assert_eq!(preset.planets.len(), 21);
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    // Belt bodies now join the 9 (Sun + 8 planets). The exact count
    // depends on PRESET_BELT_BUDGET, so check the lower bound.
    assert!(b.len() >= 9, "Solar System spawned {} bodies", b.len());
    assert_eq!(b.kind[0], BodyKind::Star);
    assert_eq!(b.kind[0], BodyKind::Star);
    // The preset's star is spawned at its present age (4.6 Gyr).
    assert!(
        (b.age_gyr[0] - 4.6).abs() < 1e-9,
        "preset star age = {}",
        b.age_gyr[0]
    );
    assert_eq!(b.m[0], M_SUN_ME);
    let earth = (0..b.len())
        .find(|&i| (b.m[i] - 1.0).abs() < 1e-9)
        .expect("earth");
    let o = orbit_info(&b, earth, 0);
    assert!((o.semi_major / au_in_gm() - 1.0).abs() < 0.01);
}

#[test]
fn solar_ring_systems_have_independent_phases() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_systems(&mut b, 3);
    // Each system has 9 planets + ~200 belts. The suns are still 3.
    assert!(
        b.len() >= 27,
        "expected at least 27 bodies, got {}",
        b.len()
    );
    let suns: Vec<usize> = (0..b.len())
        .filter(|&i| b.kind[i] == BodyKind::Star)
        .collect();
    assert_eq!(suns.len(), 3);
    // Each system's Earth is at sun + 3 (Sun, Mercury, Venus, Earth).
    let ang: Vec<f64> = suns
        .iter()
        .map(|&s| (b.y[s + 3] - b.y[s]).atan2(b.x[s + 3] - b.x[s]))
        .collect();
    assert!((ang[0] - ang[1]).abs() > 1e-6 && (ang[1] - ang[2]).abs() > 1e-6);
}

#[test]
fn solar_preset_has_belt_bodies() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let n_belt = (0..b.len()).filter(|&i| b.origin[i].is_belt()).count();
    assert!(n_belt > 100, "belt bodies: {}", n_belt);
}

#[test]
fn solar_preset_has_every_belt_population() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    for o in [
        BodyOrigin::AsteroidBelt,
        BodyOrigin::KuiperCold,
        BodyOrigin::KuiperHot,
        BodyOrigin::KuiperResonant,
        BodyOrigin::KuiperScattered,
        BodyOrigin::Trojan,
        BodyOrigin::Hilda,
    ] {
        let n = (0..b.len()).filter(|&i| b.origin[i] == o).count();
        assert!(n > 0, "no bodies of origin {:?}", o);
    }
}

#[test]
fn mean_anomaly_sampling_is_uniform_in_time() {
    // For e = 0.5, a body spends (π - 2e) / (2π) of its *time* with r < a,
    // i.e. (π/2 - e)/π ≈ 0.34. A true-anomaly sampler would give 2/3
    // instead. So this distinguishes the two.
    let mut b = Bodies::with_capacity(1000);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mut rng = XorShift::new(7);
    for _ in 0..1000 {
        push_kepler(&mut b, 0, 5.0, 0.5, 1.0, &mut rng, BodyKind::Planet);
    }
    let near = (1..b.len()).filter(|&i| b.x[i].hypot(b.y[i]) < 5.0).count();
    let f = near as f64 / 1000.0;
    let expect = (std::f64::consts::FRAC_PI_2 - 0.5) / std::f64::consts::PI;
    assert!(
        (f - expect).abs() < 0.05,
        "fraction r<a: {} (expect {})",
        f,
        expect
    );
}

#[test]
fn random_systems_never_exceed_body_count() {
    // `body_count` is a ceiling: sparse systems may under-fill it, but no
    // system may exceed it.
    for seed in 0..200u128 {
        let mut b = Bodies::with_capacity(100);
        spawn_system(&mut b, 100, seed);
        assert!(b.len() <= 100, "seed {}: {} bodies", seed, b.len());
    }
}

#[test]
fn default_density_matches_mass_radius_law() {
    // `push` seeds `bulk_density` so that `radius_from_density` reproduces
    // `planet_true_r` at the body's initial mass. This is what keeps a
    // freshly-spawned planet visually the same size as before the
    // bulk-density change.
    let mut b = Bodies::with_capacity(4);
    for &m in &[0.01, 1.0, 5.0, 100.0] {
        let i = b.push(1.0, 0.0, 0.0, 0.0, m, BodyKind::Planet);
        assert!((b.radius_now(i) - planet_true_r(m)).abs() / planet_true_r(m) < 1e-9);
    }
}

/// `spawn_solar_system` must produce the same body count as `spawn_from_preset`.
/// The UI's "systems" row multiplies this count by systems; the old code
/// hard-coded 9 and the row read "1 × 9" while the body row read "214".
#[test]
fn solar_system_spawn_count_matches_preset() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let mut b2 = Bodies::with_capacity(0);
    spawn_from_preset(&mut b2, &solar_system_preset());
    assert_eq!(b.len(), b2.len());
    assert!(
        b.len() > 100,
        "Solar System produced only {} bodies",
        b.len()
    );
}

/// `fit` on the Solar preset must include Neptune (30 AU ≈ 4500 Gm).
#[test]
fn fit_solar_includes_neptune() {
    let neptune_gm = 30.07 * au_in_gm();
    assert!(
        solar_extent_gm(1) >= neptune_gm,
        "solar_extent {} < Neptune {}",
        solar_extent_gm(1),
        neptune_gm
    );
}

/// R on the Solar preset must change the planet phases. Two spawns with
/// different phase seeds must produce different angles for the same body.
#[test]
fn solar_reset_changes_phases() {
    let mut b1 = Bodies::with_capacity(0);
    spawn_solar_system_seeded(&mut b1, 0xC0DE);
    let mut b2 = Bodies::with_capacity(0);
    spawn_solar_system_seeded(&mut b2, 0xC0DF);
    assert_eq!(b1.len(), b2.len());
    let earth1 = (0..b1.len())
        .find(|&i| (b1.m[i] - 1.0).abs() < 1e-9)
        .expect("earth 1");
    let earth2 = (0..b2.len())
        .find(|&i| (b2.m[i] - 1.0).abs() < 1e-9)
        .expect("earth 2");
    let a1 = b1.y[earth1].atan2(b1.x[earth1]);
    let a2 = b2.y[earth2].atan2(b2.x[earth2]);
    assert!((a1 - a2).abs() > 1e-6, "phases identical: {a1} vs {a2}");
}

/// Collisions inside `step_block` must trigger a tree rebuild; otherwise
/// the stale body indices either panic or silently compute wrong forces.
/// This exercises the `stats_before != stats_after` rebuild path.
#[test]
fn step_block_with_collisions_runs_clean() {
    let mut b = Bodies::with_capacity(800);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mut rng = XorShift::new(7);
    for k in 0..696 {
        let r = 5.0 + (k as f64) * 0.05;
        let th = rng.next_f64() * std::f64::consts::TAU;
        let vc = (G * M_SUN_ME / r).sqrt();
        b.push(
            r * th.cos(),
            r * th.sin(),
            -vc * th.sin(),
            vc * th.cos(),
            0.5,
            BodyKind::Planet,
        );
    }
    // Twenty overlapping pairs, to trigger many merges in one step_block.
    for k in 0..20 {
        let r = 5.0 + (k as f64) * 1.5;
        let th = 0.1 + (k as f64) * 0.3;
        let vc = (G * M_SUN_ME / r).sqrt();
        let (x, y) = (r * th.cos(), r * th.sin());
        b.push(x, y, -vc * th.sin(), vc * th.cos(), 1.0, BodyKind::Planet);
        b.push(
            x + 0.01,
            y,
            -vc * th.sin(),
            vc * th.cos(),
            1.0,
            BodyKind::Planet,
        );
    }
    compute_accel(&mut b);
    for step in 0..5 {
        advance(&mut b, DT);
        fuzz_assert_sane(&b, &format!("collision step {step}"));
    }
    assert!(b.stats.merges >= 20, "expected ≥ 20 merges");
}

#[test]
fn m1_ring_accommodates_wide_binary() {
    assert!((ring_radius_for_extent(3, 0.0) - solar_ring_radius_gm(3)).abs() < 1e-9);
    assert!((ring_radius_for_extent(2, 44_880.0) - 3.0 * 44_880.0).abs() < 1e-6);
    assert_eq!(ring_radius_for_extent(1, 1e9), 0.0);
    for seed in 0..500u128 {
        let mut b = Bodies::with_capacity(0);
        let (infos, ranges) = spawn_random_systems_impl(&mut b, 30, seed, 2);
        let wide = infos
            .iter()
            .any(|i| i.binary.is_some_and(|bi| bi.a_au > 100.0));
        if !wide {
            continue;
        }
        // The two systems must not overlap: centre distance exceeds the sum
        // of their reach about their own centres of mass.
        let c: Vec<(f64, f64)> = ranges.iter().map(|&r| range_com(&b, r)).collect();
        let reach: Vec<f64> = ranges
            .iter()
            .zip(&c)
            .map(|(&(s, e), &(cx, cy))| {
                (s..e)
                    .map(|i| (b.x[i] - cx).hypot(b.y[i] - cy))
                    .fold(0.0, f64::max)
            })
            .collect();
        let d = (c[0].0 - c[1].0).hypot(c[0].1 - c[1].1);
        assert!(
            d > reach[0] + reach[1],
            "seed {seed}: d {d} reach {reach:?}"
        );
        return;
    }
    panic!("no wide binary in 500 seeds - adjust seed range");
}

#[test]
fn m2_radial_scatter() {
    let mut b = Bodies::with_capacity(0);
    let _ = spawn_random_systems(&mut b, 5, 42, 6);
    let prim: Vec<f64> = (0..b.len())
        .filter(|&i| is_host_kind(b.kind[i])) // Star | BD | WD | NS | BH
        .map(|i| b.x[i].hypot(b.y[i]))
        .collect();
    assert!(prim.len() >= 6);
    let ring = solar_ring_radius_gm(6);
    let min_r = prim.iter().copied().fold(f64::INFINITY, f64::min);
    let max_r = prim.iter().copied().fold(0.0, f64::max);
    assert!(max_r - min_r > 0.05 * ring, "no radial scatter");
}

#[test]
fn m3_angular_jitter() {
    let n = 6usize;
    let step = std::f64::consts::TAU / n as f64;
    let lo = ring_theta(n, 2, 0.0);
    let hi = ring_theta(n, 2, 1.0);
    assert!(((hi - lo) / step - 0.3).abs() < 1e-12);
    assert!((ring_theta(n, 2, 0.5) - 2.0 * step).abs() < 1e-12);
    let masses = vec![1.0; n];
    let a = ring_slots(n, 1, 1e4, &masses);
    let c = ring_slots(n, 2, 1e4, &masses);
    let differs = (0..n).any(|k| {
        let ta = a[k].1.atan2(a[k].0);
        let tc = c[k].1.atan2(c[k].0);
        (ta - tc).abs() > 0.05 * step
    });
    assert!(differs, "seeds give the same angles");
}

#[test]
fn m4_per_system_speed() {
    let s = ring_sum(4);
    let speed = |m: f64| {
        let (_, _, vx, vy) = ring_slot(m, s, 1e4, 0.0, 0.0, 0.0);
        vx.hypot(vy)
    };
    assert!((speed(10.0) / speed(1.0) - 10f64.sqrt()).abs() < 1e-9);
    // e = 0 gives the circular ring speed.
    assert!((speed(2.0) - (G * 2.0 * s / 1e4).sqrt()).abs() < 1e-9);
    // In a spawn the systems do not share one speed.
    let mut b = Bodies::with_capacity(0);
    let (_, ranges) = spawn_random_systems_impl(&mut b, 5, 42, 4);
    let v: Vec<f64> = ranges
        .iter()
        .map(|&(s, e)| {
            let m: f64 = (s..e).map(|i| b.m[i]).sum();
            let px: f64 = (s..e).map(|i| b.m[i] * b.vx[i]).sum::<f64>() / m;
            let py: f64 = (s..e).map(|i| b.m[i] * b.vy[i]).sum::<f64>() / m;
            px.hypot(py)
        })
        .collect();
    assert!(v.iter().all(|x| x.is_finite()));
    let (lo, hi) = (
        v.iter().copied().fold(f64::INFINITY, f64::min),
        v.iter().copied().fold(0.0, f64::max),
    );
    assert!(hi > lo * 1.0001, "all systems have the same speed: {v:?}");
}

#[test]
fn m5_body_count_scales_with_mass() {
    let mut any_scaled = false;
    for seed in 0..20u128 {
        let mut b = Bodies::with_capacity(0);
        let (_, ranges) = spawn_random_systems_impl(&mut b, 30, seed, 4);
        for &(s, e) in &ranges {
            // Realistic spawning occasionally produces degenerate systems: a
            // brown-dwarf primary + brown-dwarf companion, no planets, no
            // belts. That is a legitimate 2-body outcome (P ≈ 8 % × 20 % ≈
            // 1.7 % per BD system), so the floor is 2, not 3.
            assert!((2..=60).contains(&(e - s)), "seed {seed}: size {}", e - s);
            any_scaled |= e - s != 30;
        }
    }
    assert!(any_scaled, "no system was rescaled");
    // One system is never rescaled.
    let mut b = Bodies::with_capacity(0);
    let _ = spawn_random_systems(&mut b, 30, 5, 1);
    assert!(b.len() <= 30, "{} bodies", b.len());
}

#[test]
fn m6_eccentric_mutual_orbits() {
    let s = ring_sum(6);
    let (m, a, e) = (3.0, 1e4, 0.3);
    let mu = G * m * s;
    for nu in [0.0, 1.0, 3.0] {
        let (x, y, vx, vy) = ring_slot(m, s, a, e, nu, 0.7);
        let r = x.hypot(y);
        let energy = 0.5 * (vx * vx + vy * vy) - mu / r;
        assert!((energy + mu / (2.0 * a)).abs() < 1e-9 * mu / a, "nu {nu}");
    }
    let (xp, yp, _, _) = ring_slot(m, s, a, e, 0.0, 0.0);
    let (xa, ya, _, _) = ring_slot(m, s, a, e, std::f64::consts::PI, 0.0);
    assert!(xa.hypot(ya) > 1.5 * xp.hypot(yp));
}

#[test]
fn spawn_compact_chain_is_bimodal() {
    let (mut zero, mut one, mut chain) = (0u32, 0u32, 0u32);
    for seed in 0..1000u128 {
        let b = solar_mass_system(seed, 100);
        match count_compact_and_giants(&b).0 {
            0 => zero += 1,
            1 => one += 1,
            2..=6 => chain += 1,
            _ => {}
        }
    }
    assert!(one < 50, "exactly one compact planet: {}/1000", one);
    assert!(chain > 300, "2-6 compact planets: {}/1000", chain);
    assert!(zero > 400, "no compact planet: {}/1000", zero);
}

#[test]
fn spawn_compact_and_giant_coexist() {
    let mut both = 0u32;
    for seed in 0..1000u128 {
        let b = solar_mass_system(seed, 100);
        let (c, g) = count_compact_and_giants(&b);
        if c >= 3 && g >= 1 {
            both += 1;
        }
    }
    assert!(both > 20, "systems with a chain and a giant: {}", both);
}

#[test]
fn spawn_outer_belt_mass_realistic() {
    // Solar analogue with a Jupiter: the outer belt carries `outer_mass`.
    let centre = (0.0, 0.0, 0.0, 0.0, M_SUN_ME);
    let planets = [(1usize, 317.8, 5.2)];
    let mut spec_masses = Vec::new();
    let mut ratios = Vec::new();
    for seed in 0..300u128 {
        let mut rng = XorShift::new(seed.wrapping_mul(7919) + 1);
        let mut b = Bodies::with_capacity(2000);
        b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
        let spec = belt_spec_for(
            &b,
            M_SUN_ME,
            0.0,
            4.6,
            &planets,
            2.7,
            0.1,
            f64::INFINITY,
            &mut rng,
        );
        let Some(outer) = &spec.outer else { continue };
        let frac: f64 = outer.populations.iter().map(|p| p.mass_frac).sum();
        let expected = spec.outer_mass * frac;
        spec_masses.push(spec.outer_mass);
        spawn_belts(&mut b, centre, &spec, 2.7, 1000, &mut rng);
        let got: f64 = (0..b.len())
            .filter(|&i| {
                matches!(
                    b.origin[i],
                    BodyOrigin::KuiperCold
                        | BodyOrigin::KuiperHot
                        | BodyOrigin::KuiperResonant
                        | BodyOrigin::KuiperScattered
                )
            })
            .map(|i| b.m[i])
            .sum();
        ratios.push(got / expected);
    }
    assert!(
        spec_masses.len() > 150,
        "outer belts: {}",
        spec_masses.len()
    );
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, c| a.total_cmp(c));
        v[v.len() / 2]
    };
    // Observed cold classical Kuiper belt: ~0.03 Me.
    let m_med = median(&mut spec_masses);
    assert!(
        m_med > 0.03 / 5.0 && m_med < 0.03 * 5.0,
        "median outer mass {}",
        m_med
    );
    let r_med = median(&mut ratios);
    assert!(
        r_med > 0.2 && r_med < 5.0,
        "tracer mass / outer_mass = {}",
        r_med
    );
}

#[test]
fn outer_belt_mass_barely_declines_with_age() {
    let old = belt_mass_earth_outer(1.0, 0.0, 10.0, &mut XorShift::new(42));
    let young = belt_mass_earth_outer(1.0, 0.0, 0.0, &mut XorShift::new(42));
    assert!(old > 0.5 * young, "old {} young {}", old, young);
}

#[test]
fn debris_belt_occurrence_is_near_universal() {
    assert!(debris_belt_occurrence_fraction(1.0, 10.0) >= 0.80);
}

#[test]
fn spawn_bd_binaries_exist() {
    let mut n = 0;
    for seed in 0..500u128 {
        let mut b = Bodies::with_capacity(30);
        let info = spawn_system_with_info(&mut b, 30, seed);
        if b.kind[0] == BodyKind::BrownDwarf && info.binary.is_some() {
            n += 1;
        }
    }
    assert!(n > 5, "brown-dwarf binaries: {}", n);
}

#[test]
fn sample_primary_bd_mass_skews_low() {
    let mut rng = XorShift::new(11);
    let (mut low, mut total) = (0u32, 0u32);
    let mid = (BD_M_MIN + BD_M_MAX) / 2.0;
    while total < 5000 {
        let (m, kind) = sample_primary(&mut rng);
        if kind == BodyKind::BrownDwarf {
            total += 1;
            if m < mid {
                low += 1;
            }
        }
    }
    assert!(
        low as f64 / total as f64 > 0.55,
        "below midpoint: {}/{}",
        low,
        total
    );
}

#[test]
fn spawn_never_exceeds_body_count() {
    for seed in 0..200u128 {
        let mut b = Bodies::with_capacity(100);
        spawn_system(&mut b, 100, seed);
        assert!(b.len() <= 100, "seed {}: {} bodies", seed, b.len());
    }
}

#[test]
fn outer_belt_radii_follow_falling_surface_density() {
    // Sigma ~ a^-1.5: more bodies in the inner half of [10, 50] AU than a
    // flat draw gives (median ~26 AU, not 30 AU), and all inside the range.
    let mut rng = XorShift::new(3);
    let mut v: Vec<f64> = (0..5000)
        .map(|_| draw_a_outer(&mut rng, 10.0, 50.0, &[]).unwrap())
        .collect();
    assert!(v.iter().all(|&a| (10.0..=50.0).contains(&a)));
    v.sort_by(|a, c| a.total_cmp(c));
    assert!(v[2500] < 28.0, "median {}", v[2500]);
}

#[test]
fn spawn_lone_giants_are_eccentric() {
    let mut es = Vec::new();
    for seed in 0..600u128 {
        let b = solar_mass_system(seed, 100);
        let giants: Vec<usize> = (1..b.len())
            .filter(|&i| {
                b.origin[i] == BodyOrigin::Spawned
                    && b.kind[i] == BodyKind::Planet
                    && b.m[i] > 50.0
                    && !is_moon(&b, i)
            })
            .collect();
        if giants.len() != 1 {
            continue;
        }
        let i = giants[0];
        let (dx, dy) = (b.x[i] - b.x[0], b.y[i] - b.y[0]);
        let (dvx, dvy) = (b.vx[i] - b.vx[0], b.vy[i] - b.vy[0]);
        let mu = G * (b.m[0] + b.m[i]);
        let r = dx.hypot(dy);
        let rv = dx * dvx + dy * dvy;
        let v2 = dvx * dvx + dvy * dvy;
        let ex = ((v2 - mu / r) * dx - rv * dvx) / mu;
        let ey = ((v2 - mu / r) * dy - rv * dvy) / mu;
        es.push(ex.hypot(ey));
    }
    assert!(es.len() > 30, "lone giants: {}", es.len());
    let mean = es.iter().sum::<f64>() / es.len() as f64;
    assert!(mean > 0.12 && mean < 0.30, "mean e {}", mean);
}

#[test]
fn spawn_compact_chains_include_resonant_systems() {
    let (mut chains, mut resonant) = (0u32, 0u32);
    for seed in 0..800u128 {
        let b = solar_mass_system(seed, 100);
        let mut a: Vec<f64> = (1..b.len())
            .filter(|&i| {
                b.origin[i] == BodyOrigin::Spawned
                    && b.kind[i] == BodyKind::Planet
                    && b.m[i] <= 30.0
                    && !is_moon(&b, i)
            })
            .map(|i| sma_about_primary(&b, i))
            .collect();
        if a.len() < 3 {
            continue;
        }
        chains += 1;
        a.sort_by(|x, y| x.total_cmp(y));
        let near = a
            .windows(2)
            .filter(|w| {
                let pr = (w[1] / w[0]).powf(1.5);
                (1.49..1.55).contains(&pr) || (1.98..2.06).contains(&pr)
            })
            .count();
        if near >= 2 {
            resonant += 1;
        }
    }
    assert!(chains > 40, "chains: {}", chains);
    assert!(
        resonant as f64 / chains as f64 > 0.15,
        "resonant chains: {}/{}",
        resonant,
        chains
    );
}

#[test]
fn spawn_unstable_giant_systems_exist() {
    // Tightly packed giants: adjacent pairs inside ~4.6 mutual Hill radii
    // (stable 1.5-2.2 period spacing gives > 10).
    let mut packed = 0u32;
    for seed in 0..1000u128 {
        let b = solar_mass_system(seed, 100);
        let mut g: Vec<(f64, f64)> = (1..b.len())
            .filter(|&i| {
                b.origin[i] == BodyOrigin::Spawned
                    && b.kind[i] == BodyKind::Planet
                    && b.m[i] > 50.0
                    && !is_moon(&b, i)
            })
            .map(|i| (sma_about_primary(&b, i), b.m[i]))
            .collect();
        g.sort_by(|x, y| x.0.total_cmp(&y.0));
        let tight = g.windows(2).any(|w| {
            let rh = ((w[0].1 + w[1].1) / (3.0 * b.m[0])).cbrt() * 0.5 * (w[0].0 + w[1].0);
            (w[1].0 - w[0].0) / rh < 4.6
        });
        if tight {
            packed += 1;
        }
    }
    assert!(packed > 15, "packed giant systems: {}/1000", packed);
}

#[test]
fn spawn_triples_and_quadruples_exist() {
    let (mut hosts, mut triples, mut quads) = (0u32, 0u32, 0u32);
    for seed in 0..4000u128 {
        let mut b = Bodies::with_capacity(20);
        spawn_system(&mut b, 20, seed);
        assert!(b.len() <= 20, "seed {}: {} bodies", seed, b.len());
        if b.kind[0] != BodyKind::Star {
            continue;
        }
        hosts += 1;
        let stars = (0..b.len())
            .filter(|&i| b.kind[i] == BodyKind::Star)
            .count();
        if stars >= 3 {
            triples += 1;
        }
        if stars >= 4 {
            quads += 1;
        }
    }
    assert!(
        triples as f64 / hosts as f64 > 0.03,
        "triples or more: {}/{}",
        triples,
        hosts
    );
    assert!(quads >= 3, "quadruples: {}", quads);
}

#[test]
fn hierarchical_stars_respect_stability_limit() {
    for seed in 0..3000u128 {
        let mut b = Bodies::with_capacity(20);
        let info = spawn_system_with_info(&mut b, 20, seed);
        let Some(bi) = info.binary else { continue };
        let stars: Vec<usize> = (0..b.len())
            .filter(|&i| b.kind[i] == BodyKind::Star && i != 0 && i != bi.companion)
            .collect();
        for &s in &stars {
            // Distance of the outer star from the inner pair's centre of mass.
            let (m0, m1) = (b.m[0], b.m[bi.companion]);
            let cx = (m0 * b.x[0] + m1 * b.x[bi.companion]) / (m0 + m1);
            let cy = (m0 * b.y[0] + m1 * b.y[bi.companion]) / (m0 + m1);
            let d_au = (b.x[s] - cx).hypot(b.y[s] - cy) / au_in_gm();
            assert!(
                d_au > 2.0 * bi.a_au * (1.0 - bi.ecc),
                "seed {}: outer star at {} AU, inner binary a = {} AU",
                seed,
                d_au,
                bi.a_au
            );
        }
    }
}

#[test]
fn spawn_interstellar_object_is_inbound_and_logged() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let n = spawn_interstellar_object(&mut b).expect("ISO should spawn");
    assert_eq!(b.origin[n], BodyOrigin::Interstellar);
    // Starts near 0.9 × CULL_DIST from the primary.
    let r0 = (b.x[n] * b.x[n] + b.y[n] * b.y[n]).sqrt();
    assert!(
        (r0 - 0.9 * CULL_DIST).abs() < 1e-6 * CULL_DIST,
        "ISO start radius {} vs 0.9 × CULL_DIST",
        r0
    );
    // Inbound: velocity dotted with the outward radial vector is negative.
    let (dx, dy) = (b.x[n] - b.x[0], b.y[n] - b.y[0]);
    let (vx, vy) = (b.vx[n] - b.vx[0], b.vy[n] - b.vy[0]);
    assert!(
        dx * vx + dy * vy < 0.0,
        "ISO must be inbound: r·v = {}",
        dx * vx + dy * vy
    );
    // Unbound relative to the primary: specific orbital energy > 0.
    let v2 = vx * vx + vy * vy;
    let r = (dx * dx + dy * dy).sqrt();
    let e_spec = 0.5 * v2 - G * b.m[0] / r;
    assert!(e_spec > 0.0, "ISO must be unbound: E = {}", e_spec);
    // Logged as "entered".
    assert!(
        b.events
            .iter()
            .any(|e| { matches!(e.kind, EventKind::Interstellar) && e.text.contains("entered") }),
        "ISO entry must be logged (no Interstellar event found)"
    );
}

#[test]
fn spawn_rogue_planet_is_heavy_and_inbound() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let n = spawn_rogue_planet(&mut b).expect("rogue should spawn");
    assert_eq!(b.origin[n], BodyOrigin::RoguePlanet);
    assert!(
        b.m[n] >= ROGUE_MIN_MASS,
        "rogue mass {} below threshold {}",
        b.m[n],
        ROGUE_MIN_MASS
    );
    assert!(
        b.m[n] <= 10.0 * JUPITER_MASS,
        "rogue mass {} above upper bound",
        b.m[n]
    );
    let r0 = (b.x[n] * b.x[n] + b.y[n] * b.y[n]).sqrt();
    assert!(
        (r0 - 0.9 * CULL_DIST).abs() < 1e-6 * CULL_DIST,
        "rogue start radius {} vs 0.9 × CULL_DIST",
        r0
    );
    let (dx, dy) = (b.x[n] - b.x[0], b.y[n] - b.y[0]);
    let (vx, vy) = (b.vx[n] - b.vx[0], b.vy[n] - b.vy[0]);
    assert!(dx * vx + dy * vy < 0.0, "rogue must be inbound");
    let v2 = vx * vx + vy * vy;
    let r = (dx * dx + dy * dy).sqrt();
    assert!(0.5 * v2 - G * b.m[0] / r > 0.0, "rogue must be unbound");
}

#[test]
fn rogue_planet_is_not_a_moon() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let n = spawn_rogue_planet(&mut b).expect("rogue should spawn");
    b.refresh_parent_cache();
    assert_eq!(
        find_parent(&b, n),
        None,
        "rogue must not report a parent — it's unbound"
    );
}
