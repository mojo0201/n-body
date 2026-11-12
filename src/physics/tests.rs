//! Physics tests.
//!
//! Loaded as `physics::tests` via `#[cfg(test)] mod tests;` in
//! `physics.rs`, so `use super::*;` sees every private item (`Tree`,
//! `collision_pass`, `find_collisions`, `pair_hit`, `interact_frac`,
//! `impact_offset`, `dt_limit`, `rung_for`, `disrupt`, `coalesce_debris`,
//! `belt_spec_for`, `spawn_belts`, `belt_mass_earth`, `kirkwood_gaps`,
//! `radius_from_density`, the IC samplers, and so on) without widening the
//! crate's public API.
//!
//! Test naming convention: a letter prefix groups tests by area; a
//! descriptive snake_case name follows. `cargo test -- --list` shows the
//! grouping at a glance.
//!
//!   a  conservation invariants (L, p, mass) across merges
//!   b  hit-and-run
//!   c  debris / moon-forming impacts
//!   d  grazing vs head-on
//!   e  tidal disruption timing
//!   f  radius laws
//!   g  initial conditions (statistical)
//!   h  rubble coalescence
//!   i  radius / survivor / primary / Chandrasekhar
//!   j  compaction invariants
//!   k  binaries and circumbinary stability
//!   l  Barnes-Hut vs direct
//!   m  collision detection (hash vs brute force, immunity)
//!   n  integrator (rungs, dt limits, energy conservation)
//!   o  orbit_info, circularize, host_distance
//!   p  escape culling
//!   q  energy diagnostics
//!   r  kind transitions via merge
//!   s  user edits
//!   t  multi-system spawn
//!   u  RNG / samplers
//!   v  geometry helpers (Hill, Roche, impact, interact_frac, dt_limit)
//!   w  naming helpers (class/threshold/format)
//!   x  star luminosity, temperature, lifetime, habitability
//!   y  samplers (Rayleigh, power-law, normal, IMF median)
//!   z  small utilities (time unit, index_of_id, log ring buffer, cell key)
//!   aa spawn variants
//!   ab fast (hot) collision path matches exhaustive scan
//!   ac rubble bounce (momentum, spin, immunity)
//!   ad fragmentation (frag_masses, density, rigidity, disrupt)
//!   ae integrator internals (rungs, flashes, Yoshida beats leapfrog)
//!   af silent merge (no log, no flash, conservation)
//!   ag orbit / anomaly helpers
//!   ah osculating-ellipse geometry
//!   ai accuracy (tight BH error, tight energy drift, element preservation)
//!   aj realism (Solar System reproduces real periods)
//!   ak fuzz smoke (invariants across many seeds; tree-rebuild stress)
//!   al stellar evolution (aging, remnants, adiabatic expansion, belts,
//!      bulk density, random spawn)
//!
//! Belts and bulk density are tested under `al_*` (they were introduced by
//! the evolution-then-belts prompt pair) and under `mean_anomaly_*`,
//! `ices_are_lighter_than_rocks`, `belt_*`, `kirkwood_*`, `trojans_*`,
//! `random_systems_never_exceed_body_count`, `default_density_*`.

use super::*;

// ---------------------------------------------------------------------------
// Shared test helpers
// ---------------------------------------------------------------------------

/// Two overlapping planets, with a given relative speed ratio and impact
/// parameter. Used by every collision-conservation test.
fn pair(m_big: f64, m_small: f64, ratio: f64, bb: f64) -> Bodies {
    let mut b = Bodies::with_capacity(8);
    let big = b.push(0.0, 0.0, 0.0, 0.0, m_big, BodyKind::Planet);
    let small = b.push(0.0, 0.0, 0.0, 0.0, m_small, BodyKind::Planet);
    let rr = b.contact_r(big) + b.contact_r(small);
    let v_esc = (2.0 * G * (m_big + m_small) / rr).sqrt();
    let d = bb * rr;
    let s = (rr * rr - d * d).sqrt();
    b.x[small] = -s * 0.999;
    b.y[small] = d * 0.999;
    b.vx[small] = ratio * v_esc;
    b._px[small] = b.x[small];
    b._py[small] = b.y[small];
    b
}

fn shift(b: &mut Bodies, x: f64, y: f64, vx: f64, vy: f64) {
    for k in 0..b.len() {
        b.x[k] += x;
        b.y[k] += y;
        b._px[k] = b.x[k];
        b._py[k] = b.y[k];
        b.vx[k] += vx;
        b.vy[k] += vy;
    }
}

fn momentum(b: &Bodies) -> (f64, f64) {
    (
        (0..b.len()).map(|k| b.m[k] * b.vx[k]).sum(),
        (0..b.len()).map(|k| b.m[k] * b.vy[k]).sum(),
    )
}

fn mass(b: &Bodies) -> f64 {
    b.m.iter().sum()
}

fn bound_prograde(b: &Bodies, sense: f64) -> usize {
    (1..b.len())
        .filter(|&k| {
            let (dx, dy) = (b.x[k] - b.x[0], b.y[k] - b.y[0]);
            let (dvx, dvy) = (b.vx[k] - b.vx[0], b.vy[k] - b.vy[0]);
            let e = 0.5 * (dvx * dvx + dvy * dvy) - G * b.m[0] / (dx * dx + dy * dy).sqrt();
            e < 0.0 && sense * (dx * dvy - dy * dvx) > 0.0
        })
        .count()
}

fn assert_conserved(b0: (f64, f64, f64, f64), b: &Bodies) {
    let (l0, px0, py0, m0) = b0;
    let l1 = total_angular_momentum(b);
    let (px1, py1) = momentum(b);
    assert!(
        (l1 - l0).abs() < 1e-9 * (1.0 + l0.abs()),
        "L {} -> {}",
        l0,
        l1
    );
    assert!((px1 - px0).abs() < 1e-9 * (1.0 + px0.abs()), "px");
    assert!((py1 - py0).abs() < 1e-9 * (1.0 + py0.abs()), "py");
    assert!((mass(b) - m0).abs() < 1e-12 * m0, "mass");
}

fn snapshot(b: &Bodies) -> (f64, f64, f64, f64) {
    let (px, py) = momentum(b);
    (total_angular_momentum(b), px, py, mass(b))
}

/// Osculating semi-major axis of body `i` about a centre
/// `(x, y, vx, vy, mass)`.
fn osc_a(b: &Bodies, i: usize, c: (f64, f64, f64, f64, f64)) -> f64 {
    let (dx, dy) = (b.x[i] - c.0, b.y[i] - c.1);
    let (dvx, dvy) = (b.vx[i] - c.2, b.vy[i] - c.3);
    let r = dx.hypot(dy);
    let v2 = dvx * dvx + dvy * dvy;
    1.0 / (2.0 / r - v2 / (G * c.4))
}

fn star_pair_centre(b: &Bodies) -> (f64, f64, f64, f64, f64) {
    let m = b.m[0] + b.m[1];
    (
        (b.m[0] * b.x[0] + b.m[1] * b.x[1]) / m,
        (b.m[0] * b.y[0] + b.m[1] * b.y[1]) / m,
        (b.m[0] * b.vx[0] + b.m[1] * b.vx[1]) / m,
        (b.m[0] * b.vy[0] + b.m[1] * b.vy[1]) / m,
        m,
    )
}

/// True iff body `i` is a moon of a planet (its parent is a Planet or
/// DwarfPlanet, not a star or remnant). Use this instead of
/// `find_parent(b, i).is_some()` when you mean "is a moon": the latter
/// is true for *every* primary planet, since planets orbit a star.
fn is_moon(b: &Bodies, i: usize) -> bool {
    matches!(
        find_parent(b, i),
        Some(p) if matches!(b.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet)
    )
}

// ===========================================================================
// a — merge conservation
// ===========================================================================

#[test]
fn a_merge_conserves_angular_momentum() {
    let mut b = pair(1.0, 0.3, 0.4, 0.6);
    shift(&mut b, 40.0, -25.0, 3.0, 2.0);
    b.spin[0] = 0.01;
    b.spin[1] = -0.004;
    let s0 = snapshot(&b);
    assert!(s0.0.abs() > 1e-3, "test needs a non-zero L");
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1);
    assert_eq!(b.stats.merges, 1);
    assert_conserved(s0, &b);
}

#[test]
fn a_star_merge_conserves_angular_momentum() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(0.5, 0.2, -3.0, 40.0, 1.0, BodyKind::Planet);
    let s0 = snapshot(&b);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1);
    assert_conserved(s0, &b);
}

#[test]
fn a_debris_conserves_everything() {
    let mut b = pair(1.0, 0.107, 1.0, 0.7);
    shift(&mut b, -30.0, 12.0, 1.5, -0.7);
    let s0 = snapshot(&b);
    collision_pass(&mut b, DT, None);
    assert!(b.len() > 3, "expected fragments, got {} bodies", b.len());
    assert_conserved(s0, &b);
}

// ===========================================================================
// b — hit-and-run
// ===========================================================================

#[test]
fn b_hit_and_run_at_high_b() {
    let mut b = pair(1.0, 0.5, 1.5, 0.9);
    shift(&mut b, 10.0, -4.0, 0.5, 0.3);
    let (mb0, ms0) = (b.m[0], b.m[1]);
    let v0 = (b.vx[1] - b.vx[0], b.vy[1] - b.vy[0]);
    let s0 = snapshot(&b);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 2, "both bodies must survive");
    assert_eq!(b.stats.merges, 0);
    assert_eq!(b.stats.impacts, 1);
    assert!(
        b.m[0] > mb0 && b.m[1] < ms0,
        "light body loses mass to the heavy body"
    );
    let v1 = (b.vx[1] - b.vx[0], b.vy[1] - b.vy[0]);
    let (n0, n1) = (v0.0.hypot(v0.1), v1.0.hypot(v1.1));
    let cos = (v0.0 * v1.0 + v0.1 * v1.1) / (n0 * n1);
    assert!(cos < 0.9999, "path must be deflected");
    assert!(n1 < n0, "bounce is inelastic");
    assert_conserved(s0, &b);
    let mut h = pair(1.0, 0.5, 1.5, 0.1);
    collision_pass(&mut h, DT, None);
    assert_ne!(h.len(), 2, "head-on impact must not be a hit-and-run");
}

// ===========================================================================
// c — moon-forming impacts
// ===========================================================================

#[test]
fn c_moon_forming_impact_makes_a_disk() {
    for &ratio in &[0.7, 0.85, 1.0] {
        let mut b = pair(1.0, 0.107, ratio, 0.7);
        let sense = -1.0;
        collision_pass(&mut b, DT, None);
        assert_eq!(b.stats.impacts, 1, "ratio {}", ratio);
        let ejected = mass(&b) - b.m[0];
        let bound = bound_prograde(&b, sense);
        eprintln!(
            "ratio {:.2}: bodies {} ejected {:.4} M_E ({:.2} lunar) bound prograde {} spin {:.4}",
            ratio,
            b.len(),
            ejected,
            ejected / 0.0123,
            bound,
            b.spin[0]
        );
        assert!(b.len() >= 4);
        assert!(ejected > 0.005 && ejected < 0.1, "ejected {}", ejected);
        assert!(b.m[0] > 1.0, "remnant keeps the target mass");
        assert!(
            bound * 2 >= b.len() - 1,
            "most fragments must be bound and prograde"
        );
    }
}

// ===========================================================================
// d — grazing vs head-on
// ===========================================================================

#[test]
fn d_grazing_differs_from_head_on() {
    let sense = -1.0;
    let mut head = pair(1.0, 0.107, 1.2, 0.0);
    let mut graze = pair(1.0, 0.107, 1.2, 0.7);
    collision_pass(&mut head, DT, None);
    collision_pass(&mut graze, DT, None);
    let (bh, bg) = (bound_prograde(&head, sense), bound_prograde(&graze, sense));
    eprintln!(
        "head-on: bodies {} spin {:.4} bound-prograde {} | grazing: bodies {} spin {:.4} bound-prograde {}",
        head.len(), head.spin[0], bh, graze.len(), graze.spin[0], bg
    );
    assert!(
        graze.spin[0].abs() > 5.0 * head.spin[0].abs() + 1e-12,
        "grazing spin must be larger"
    );
    assert!(bg > bh, "grazing impact must put more fragments into orbit");
}

// ===========================================================================
// e — tidal break-up timing
// ===========================================================================

#[test]
fn e_tidal_break_up_waits_for_pericentre() {
    let mut b = Bodies::with_capacity(32);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let j = b.push(1.0, 0.0, 0.0, 0.0, 318.0, BodyKind::Planet);
    let r0 = 0.95 * roche_limit(&b, j, 0);
    let vc = (G * M_SUN_ME / r0).sqrt();
    b.x[j] = r0;
    b.vx[j] = -0.15 * vc;
    b.vy[j] = 0.9 * vc;
    compute_accel(&mut b);
    advance(&mut b, DT);
    assert_eq!(
        b.stats.tidal, 0,
        "must not break up on entering the Roche limit"
    );
    assert!(b.disrupting[1], "must be marked");
    let mut steps = 1;
    while b.stats.tidal == 0 && steps < 200 {
        advance(&mut b, DT);
        steps += 1;
    }
    assert_eq!(b.stats.tidal, 1, "must break up after pericentre");
    let rm = (1..b.len()).map(|k| b.x[k].hypot(b.y[k])).sum::<f64>() / (b.len() - 1) as f64;
    eprintln!(
        "broke up after {} steps at mean r {:.3} (start {:.3})",
        steps, rm, r0
    );
    assert!(
        rm < 0.9 * r0,
        "break-up must happen deeper than where the body entered"
    );
}

// ===========================================================================
// f — radius laws
// ===========================================================================

#[test]
fn f_radius_law() {
    let rj = planet_true_r(JUPITER_MASS) / R_EARTH_GM;
    assert!((rj - 11.21).abs() < 0.2, "Jupiter {}", rj);
    let mars = planet_true_r(0.107) / R_EARTH_GM * 6371.0;
    assert!((mars - 3390.0).abs() / 3390.0 < 0.1, "Mars {}", mars);
    let moon = planet_true_r(0.0123) / R_EARTH_GM * 6371.0;
    assert!((moon - 1737.0).abs() / 1737.0 < 0.1, "Moon {}", moon);
    let mut prev = 0.0;
    let mut m = 1e-4;
    while m < 3000.0 {
        let r = planet_true_r(m);
        if m < 90.0 {
            assert!(r > prev, "radius must grow with mass below 90 M_E");
        }
        prev = r;
        m *= 1.07;
    }
    let eps = 1e-9;
    let (lo, hi) = (
        radius_earth(SMALL_BODY_M - eps),
        radius_earth(SMALL_BODY_M + eps),
    );
    assert!((lo - hi).abs() / hi < 1e-6, "jump at SMALL_BODY_M");
}

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
    for seed in 0..200u32 {
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
    for seed in 0..1500u32 {
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
    let n = 3000u32;
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
    for seed in 0..1000u32 {
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
    for seed in 0..2000u32 {
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
// h — rubble coalescence
// ===========================================================================

#[test]
fn h_coalesce_counter_accumulates_and_merges() {
    let mut b = Bodies::with_capacity(8);
    let _star = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);

    let au = au_in_gm();
    let vc_star = (G * M_SUN_ME / au).sqrt();
    let p = b.push(au, 0.0, 0.0, vc_star, 1.0, BodyKind::Planet);

    let m_frag = 0.0005;
    let r_orb = 0.2;
    let vc_p = (G * 1.0 / r_orb).sqrt();
    let ang = 0.001_f64;

    let i = b.push(
        au + r_orb,
        0.0,
        0.0,
        vc_star + vc_p,
        m_frag,
        BodyKind::Planet,
    );
    let j = b.push(
        au + r_orb * ang.cos(),
        r_orb * ang.sin(),
        -vc_p * ang.sin(),
        vc_star + vc_p * ang.cos(),
        m_frag,
        BodyKind::Planet,
    );
    b.origin[i] = BodyOrigin::ImpactDebris;
    b.origin[j] = BodyOrigin::ImpactDebris;
    b.imm[i] = 0.0;
    b.imm[j] = 0.0;

    assert_eq!(find_parent(&b, i), Some(p));
    assert_eq!(find_parent(&b, j), Some(p));
    b.refresh_parent_cache();

    for _ in 0..DEBRIS_CAPTURE_STEPS {
        coalesce_debris(&mut b);
    }

    assert_eq!(b.len(), 3, "the two fragments should have coalesced");
    assert!(
        (b.m[2] - 2.0 * m_frag).abs() < 1e-9,
        "survivor has both masses"
    );
    assert_eq!(b.stats.coalesces, 1);
    assert_eq!(b.stats.merges, 1);
}

// ===========================================================================
// i — radius / survivor / primary / Chandrasekhar
// ===========================================================================

#[test]
fn i_radius_laws() {
    let r_bd = physical_radius(BodyKind::BrownDwarf, BD_M_MIN) / R_EARTH_GM;
    assert!((r_bd - 11.0).abs() < 1.0, "BD radius: {}", r_bd);

    let r_wd = physical_radius(BodyKind::WhiteDwarf, 0.6 * M_SUN_ME) / R_EARTH_GM;
    assert!(r_wd > 0.7 && r_wd < 2.0, "WD radius: {}", r_wd);

    let r_ns = physical_radius(BodyKind::NeutronStar, 1.4 * M_SUN_ME);
    assert!((r_ns - 1e-5).abs() < 1e-6, "NS radius: {}", r_ns);

    let r_dp = physical_radius(BodyKind::DwarfPlanet, 0.002);
    let r_p = physical_radius(BodyKind::Planet, 0.002);
    assert!((r_dp - r_p).abs() < 1e-12);
}

#[test]
fn i_survivor_ranking() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0 * M_SUN_ME, BodyKind::Star);
    b.push(0.001, 0.0, 0.0, 0.0, 1.4 * M_SUN_ME, BodyKind::NeutronStar);
    let (keep, _) = pick_survivor(&b, 0, 1);
    assert_eq!(b.kind[keep], BodyKind::NeutronStar, "NS outranks star");

    let mut b2 = Bodies::with_capacity(4);
    b2.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b2.push(0.001, 0.0, 0.0, 0.0, 0.5, BodyKind::DwarfPlanet);
    let (keep, _) = pick_survivor(&b2, 0, 1);
    assert_eq!(
        b2.kind[keep],
        BodyKind::Planet,
        "planet outranks dwarf planet"
    );
}

#[test]
fn i_primary_eligibility() {
    let mut b = Bodies::with_capacity(4);
    b.push(
        0.0,
        0.0,
        0.0,
        0.0,
        50.0 * JUPITER_MASS,
        BodyKind::BrownDwarf,
    );
    b.push(10.0, 0.0, 0.0, 0.3, 1.0, BodyKind::Planet);
    assert_eq!(primary_index(&b), Some(0), "BD can be primary");
    assert_eq!(find_parent(&b, 1), Some(0), "BD can host planets");
}

#[test]
fn i_wd_chandrasekhar_collapse() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, WD_M_MAX * 0.9, BodyKind::WhiteDwarf);
    b.push(0.005, 0.0, 0.0, 0.0, 0.6 * M_SUN_ME, BodyKind::WhiteDwarf);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1);
    assert_eq!(b.kind[0], BodyKind::NeutronStar, "should become NS");
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
// k — binaries and circumbinary stability
// ===========================================================================

#[test]
fn k_hw_formulas_match_published_values() {
    // Equal-mass circular binary: S-type ~0.27 a_b, P-type ~2.4 a_b.
    let s = hw_critical_s(1.0, 0.0, 0.5);
    let p = hw_critical_p(1.0, 0.0, 0.5);
    assert!((s - 0.274).abs() < 0.01, "S {}", s);
    assert!((p - 2.39).abs() < 0.05, "P {}", p);
    assert!(hw_critical_s(1.0, 0.5, 0.5) < s);
    assert!(hw_critical_p(1.0, 0.5, 0.5) > p);
}

#[test]
fn k_binary_ic_respects_stability_limits() {
    let au = au_in_gm();
    let (mut stars, mut bins, mut n_s, mut n_p) = (0, 0, 0, 0);
    for seed in 0..600u32 {
        let mut b = Bodies::with_capacity(60);
        let info = spawn_system_with_info(&mut b, 60, seed);
        assert!(b.len() <= 60, "seed {}: {} bodies", seed, b.len());
        // Companion statistics are measured for MS-star primaries only.
        // Brown dwarfs get a BD companion with 20% probability (Luhman 16
        // and similar systems) — skip them; they don't count toward the
        // MS binary fraction.
        if b.kind[0] == BodyKind::BrownDwarf {
            continue;
        }
        // Remnants (WD/NS/BH) never get a companion in `spawn_system_with_info`:
        // `wants_companion` is false for every kind other than Star and
        // BrownDwarf. Assert that invariant explicitly.
        if b.kind[0] != BodyKind::Star {
            assert!(
                info.binary.is_none(),
                "remnant primaries must not have companions"
            );
            continue;
        }
        stars += 1;
        let Some(bi) = info.binary else { continue };
        bins += 1;
        assert_eq!(bi.companion, 1);
        assert_eq!(b.kind[1], BodyKind::Star);
        assert!(
            b.m[1] <= b.m[0] && b.m[1] >= STAR_M_MIN * 0.999,
            "seed {}",
            seed
        );
        assert!(
            bi.period_days >= BIN_P_MIN_D * 0.999,
            "seed {} P {}",
            seed,
            bi.period_days
        );
        assert!(
            bi.a_au <= BIN_A_MAX_AU * 1.001,
            "seed {} a {}",
            seed,
            bi.a_au
        );
        let peri = bi.a_au * au * (1.0 - bi.ecc);
        assert!(
            peri >= BIN_PERI_CLEAR * (b.r[0] + b.r[1]) * 0.999,
            "seed {}",
            seed
        );
        if bi.period_days < BIN_CIRC_P_D {
            assert_eq!(bi.ecc, 0.0);
        }
        let bary = star_pair_centre(&b);
        let rel = (b.x[0], b.y[0], b.vx[0], b.vy[0], b.m[0] + b.m[1]);
        let a_rel = osc_a(&b, 1, rel) / au;
        assert!(
            (a_rel / bi.a_au - 1.0).abs() < 1e-6,
            "seed {} {} vs {}",
            seed,
            a_rel,
            bi.a_au
        );
        // Moons orbit their host *planet*, not the star (S-type) or the
        // binary barycentre (P-type). Measured against either reference,
        // a moon's osculating `a` is not the stability quantity: its
        // velocity includes v_planet, so it can land well outside the
        // stability limit or even on a hyperbolic orbit (a < 0) while
        // remaining perfectly bound to its parent planet. Skip them.
        // `find_parent` returns a Planet exactly for moons; hosts
        // (including a binary's lead star for circumbinary planets)
        // return a Star.
        let is_moon = |bb: &Bodies, i: usize| -> bool {
            matches!(
                find_parent(bb, i),
                Some(p) if matches!(bb.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet)
            )
        };
        match bi.arch {
            BinaryArch::PType => {
                n_p += 1;
                let lim = HW_MARGIN_P * bi.a_crit_au * au;
                for i in 2..b.len() {
                    // Skip non-planets (e.g. the third star in a triple). A
                    // third star orbits the whole inner system (m_in + m_third),
                    // not just the inner pair, so `osc_a` about `star_pair_centre`
                    // (m0+m1) is not a stability quantity for it — it can be
                    // negative even when the star is bound.
                    if b.origin[i] != BodyOrigin::Spawned
                        || !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
                        || is_moon(&b, i)
                    {
                        continue;
                    }
                    let a = osc_a(&b, i, bary);
                    assert!(
                        a >= lim * 0.999,
                        "seed {} body {} a {} < {}",
                        seed,
                        i,
                        a / au,
                        lim / au
                    );
                }
            }
            BinaryArch::STypePrimary | BinaryArch::STypeSecondary => {
                n_s += 1;
                let h = if bi.arch == BinaryArch::STypePrimary {
                    0
                } else {
                    1
                };
                let c = (b.x[h], b.y[h], b.vx[h], b.vy[h], b.m[h]);
                let lim = HW_MARGIN_S * bi.a_crit_au * au;
                for i in 2..b.len() {
                    // Same reasoning as the P-type branch: exclude third
                    // (and fourth) stars, which are not planetary stability
                    // tests.
                    if b.origin[i] != BodyOrigin::Spawned
                        || !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
                        || is_moon(&b, i)
                    {
                        continue;
                    }
                    let a = osc_a(&b, i, c);
                    assert!(
                        a > 0.0 && a <= lim * 1.001,
                        "seed {} body {} a {} > {}",
                        seed,
                        i,
                        a / au,
                        lim / au
                    );
                }
            }
        }
    }
    let frac = bins as f64 / stars as f64;
    eprintln!(
        "binaries {}/{} = {:.2}  S {} P {}",
        bins, stars, frac, n_s, n_p
    );
    assert!(frac > 0.2 && frac < 0.6, "binary fraction {}", frac);
    assert!(n_s > 0 && n_p > 0, "both architectures must occur");
}

#[test]
fn k_parent_follows_binary() {
    let mut b = Bodies::with_capacity(8);
    let (m1, m2) = (M_SUN_ME, 0.5 * M_SUN_ME);
    let mt = m1 + m2;
    let a_b = 0.1 * au_in_gm();
    let v = (G * mt / a_b).sqrt();
    b.push(-m2 / mt * a_b, 0.0, 0.0, -m2 / mt * v, m1, BodyKind::Star);
    b.push(m1 / mt * a_b, 0.0, 0.0, m1 / mt * v, m2, BodyKind::Star);
    let mut rng = XorShift::new(7);
    let a_p = 3.0 * au_in_gm();
    let circ = push_kepler_about(
        &mut b,
        (0.0, 0.0, 0.0, 0.0, mt),
        a_p,
        0.0,
        1.0,
        &mut rng,
        BodyKind::Planet,
    );
    let near2 = push_kepler(
        &mut b,
        1,
        0.01 * au_in_gm(),
        0.0,
        1.0,
        &mut rng,
        BodyKind::Planet,
    );
    assert_eq!(binary_partner(&b, 0), Some(1));
    assert_eq!(binary_partner(&b, 1), Some(0));
    assert_eq!(
        find_parent(&b, circ),
        Some(0),
        "circumbinary planet -> heavier star"
    );
    assert_eq!(
        find_parent(&b, near2),
        Some(1),
        "S-type planet -> its own star"
    );
    b.refresh_parent_cache();
    let c = orbit_reference(&b, &host_list(&b), circ, (0.0, 0.0, 0.0, 0.0, 0.0));
    assert!(
        (c.4 - mt).abs() < 1e-9,
        "circumbinary orbit is drawn about the pair"
    );
    let s = orbit_reference(&b, &host_list(&b), near2, (0.0, 0.0, 0.0, 0.0, 0.0));
    assert!(
        (s.4 - m2).abs() < 1e-9,
        "S-type orbit is drawn about its own star"
    );
    assert!(!compute_orbit_segments(&b, 1.0, true, true).is_empty());
}

#[test]
fn k_circumbinary_planet_stays_bound() {
    let au = au_in_gm();
    let (m1, m2) = (M_SUN_ME, 0.6 * M_SUN_ME);
    let mt = m1 + m2;
    let (a_b_au, e_b) = (0.2, 0.1);
    let a_b = a_b_au * au;
    let mut b = Bodies::with_capacity(8);
    let (x, y, vx, vy) = kepler_state(G * mt, a_b, e_b, 0.7, 0.3);
    b.push(
        -m2 / mt * x,
        -m2 / mt * y,
        -m2 / mt * vx,
        -m2 / mt * vy,
        m1,
        BodyKind::Star,
    );
    b.push(
        m1 / mt * x,
        m1 / mt * y,
        m1 / mt * vx,
        m1 / mt * vy,
        m2,
        BodyKind::Star,
    );
    let a_p = HW_MARGIN_P * hw_critical_p(a_b_au, e_b, m2 / mt) * au;
    let mut rng = XorShift::new(11);
    push_kepler_about(
        &mut b,
        (0.0, 0.0, 0.0, 0.0, mt),
        a_p,
        0.0,
        1.0,
        &mut rng,
        BodyKind::Planet,
    );
    recenter_com(&mut b);
    compute_accel(&mut b);
    rebase_energy(&mut b);
    for _ in 0..4000 {
        advance(&mut b, DT);
    }
    assert_eq!(b.len(), 3, "nothing merged, escaped or broke up");
    let bary = star_pair_centre(&b);
    let a_end = osc_a(&b, 2, bary);
    let drift = energy_drift_pct(&b).unwrap().abs();
    eprintln!(
        "planet a {:.3} -> {:.3} AU, binary sep {:.4} AU, energy drift {:.4}%, max rung {}",
        a_p / au,
        a_end / au,
        (b.x[1] - b.x[0]).hypot(b.y[1] - b.y[0]) / au,
        drift,
        b.stats.max_rung
    );
    assert!(
        a_end > 0.8 * a_p && a_end < 1.25 * a_p,
        "planet a {} -> {}",
        a_p / au,
        a_end / au
    );
    assert!(drift < 0.5, "energy drift {}%", drift);
    assert_eq!(b.stats.merges + b.stats.tidal, 0);
}

// ===========================================================================
// l — Barnes-Hut vs direct
// ===========================================================================

/// `Tree::accel_for` must agree with `force_direct_on` to within the
/// documented BH error (~1 % for θ = 0.5) on a realistic configuration.
/// This is the correctness gate for the entire BH path — if it drifts, the
/// physics still runs but with the wrong answer.
#[test]
fn l_bh_vs_direct_random() {
    let mut b = Bodies::with_capacity(500);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mut rng = XorShift::new(123);
    for _ in 0..499 {
        let a = 5.0 + 500.0 * rng.next_f64();
        let th = rng.next_f64() * std::f64::consts::TAU;
        let vc = (G * M_SUN_ME / a).sqrt();
        let m = 0.1 + 100.0 * rng.next_f64();
        b.push(
            a * th.cos(),
            a * th.sin(),
            -vc * th.sin(),
            vc * th.cos(),
            m,
            BodyKind::Planet,
        );
    }
    let tree = Tree::build(&b);
    let mut worst = 0.0_f64;
    let mut worst_i = 0usize;
    for i in 0..b.len() {
        let (rx, ry, _) = force_direct_on(&b, i);
        let (ax, ay, _) = tree.accel_for(&b, i);
        let rmag = (rx * rx + ry * ry).sqrt().max(1e-30);
        let dmag = ((ax - rx).powi(2) + (ay - ry).powi(2)).sqrt();
        let rel = dmag / rmag;
        if rel > worst {
            worst = rel;
            worst_i = i;
        }
    }
    assert!(
        worst < 0.10,
        "worst BH relative error {:.4} at body {} exceeds 10 %",
        worst,
        worst_i
    );
}

/// A body's own tree leaf must be skipped; a body alone in the tree must
/// feel zero force from itself.
#[test]
fn l_bh_zero_self_force() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let tree = Tree::build(&b);
    let (ax, ay, _) = tree.accel_for(&b, 0);
    assert_eq!((ax, ay), (0.0, 0.0));
}

/// BH must respect Newton's third law at the body level for a 2-body system:
/// a_0 + a_1 = 0 and the force points from heavy → light.
#[test]
fn l_bh_third_law() {
    let mut b = Bodies::with_capacity(2);
    b.push(-5.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(5.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    let (a0x, a0y, _) = tree.accel_for(&b, 0);
    let (a1x, a1y, _) = tree.accel_for(&b, 1);
    assert!((a0x + a1x).abs() < 1e-12, "third law x: {} {}", a0x, a1x);
    assert!((a0y + a1y).abs() < 1e-12, "third law y: {} {}", a0y, a1y);
    assert!(a0x > 0.0, "body 0 is pulled toward +x");
    assert!(a1x < 0.0, "body 1 is pulled toward -x");
}

/// Two coincident bodies stress the tree's `coincident` node path. Neither
/// body should get a NaN/inf force.
#[test]
fn l_bh_handles_coincident() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tree = Tree::build(&b);
    let (ax, ay, _) = tree.accel_for(&b, 0);
    assert!(ax.is_finite() && ay.is_finite(), "body 0: {} {}", ax, ay);
    let (ax, ay, _) = tree.accel_for(&b, 1);
    assert!(ax.is_finite() && ay.is_finite(), "body 1: {} {}", ax, ay);
}

// ===========================================================================
// m — collision detection
// ===========================================================================

/// Two overlapping bodies on a tiny `n` use the brute-force path; verify
/// it detects the pair.
#[test]
fn m_find_collisions_detects_overlap() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let r = b.contact_r(0);
    b.push(r - 0.001, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let pairs = find_collisions(&b, false, DT);
    assert_eq!(pairs, vec![(0, 1)]);
}

/// The hash-grid path (n >= `HASH_THRESHOLD`) must find exactly the same
/// pairs as an exhaustive pairwise scan of the same configuration.
#[test]
fn m_find_collisions_hash_matches_bruteforce() {
    // 80 bodies on a 0.01-spaced line. Each has r ≈ 0.0064 (1 M_earth),
    // so adjacent pairs overlap (sep 0.01 < sum 0.0127) and pairs two
    // or more steps apart do not.
    let mut b = Bodies::with_capacity(80);
    for k in 0..80 {
        b.push(k as f64 * 0.01, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    }
    assert!(b.len() >= HASH_THRESHOLD, "hash path is what we're testing");

    let mut reference: Vec<(usize, usize)> = Vec::new();
    for i in 0..b.len() {
        for j in (i + 1)..b.len() {
            if pair_hit(&b, i, j, false) {
                reference.push((i, j));
            }
        }
    }
    reference.sort_unstable();

    let pairs = find_collisions(&b, false, DT);
    assert_eq!(pairs, reference);
    assert!(!pairs.is_empty(), "test should produce collisions");
}

/// A body under `IMMUNITY_TIME` must not be reported as colliding.
#[test]
fn m_find_collisions_respects_immunity() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let r = b.contact_r(0);
    b.push(r - 0.001, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    // Sanity: without immunity, they collide.
    assert_eq!(find_collisions(&b, false, DT), vec![(0, 1)]);
    // Now make body 0 immune.
    b.imm[0] = 10.0;
    assert!(find_collisions(&b, false, DT).is_empty());
    // And immune on the other body.
    b.imm[0] = 0.0;
    b.imm[1] = 10.0;
    assert!(find_collisions(&b, false, DT).is_empty());
}

// ===========================================================================
// n — integrator
// ===========================================================================

/// `rung_for(lim, dt)` must land exactly on the documented boundary
/// (dt/2^k for integer k) and clamp to MAX_RUNG.
#[test]
fn n_rung_for_boundaries() {
    assert_eq!(rung_for(f64::INFINITY, DT), 0);
    assert_eq!(rung_for(0.0, DT), MAX_RUNG);
    assert_eq!(rung_for(DT, DT), 0);
    assert_eq!(rung_for(DT / 8.0, DT), 3);
    assert_eq!(rung_for(DT / 16.0, DT), 4);
    assert_eq!(rung_for(DT / 32.0, DT), 5);
    // Just below a power-of-two boundary pushes to the next rung.
    assert_eq!(rung_for(DT / 8.0 * 0.99, DT), 4);
}

/// `dt_limit` is monotonically decreasing in both the gravitational term
/// and the velocity term.
#[test]
fn n_dt_limit_monotonic() {
    // Stronger gravity -> smaller dt.
    let strong = dt_limit(1.0, 0.0);
    let weak = dt_limit(0.01, 0.0);
    assert!(weak > strong, "weaker gravity: {} > {}", weak, strong);
    // Faster relative motion -> smaller dt.
    let slow = dt_limit(0.0, 0.001);
    let fast = dt_limit(0.0, 0.1);
    assert!(slow > fast, "slower motion: {} > {}", slow, fast);
}

/// The block-leapfrog path (`kmax > 0`) must conserve energy to ~0.5 %
/// over hundreds of steps on a circular orbit.
#[test]
fn n_block_leapfrog_conserves_energy() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0;
    let mu = G * M_SUN_ME;
    let v = (mu / a).sqrt();
    b.push(a, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    compute_accel(&mut b);
    rebase_energy(&mut b);
    for _ in 0..500 {
        advance(&mut b, DT);
    }
    let drift = energy_drift_pct(&b).unwrap().abs();
    assert!(drift < 0.5, "drift {}%", drift);
    assert!(
        b.stats.max_rung > 0,
        "test should exercise the block path (rung > 0)"
    );
}

// ===========================================================================
// o — orbit info, circularize, host_distance
// ===========================================================================

/// `orbit_info` must recover (a, e, r, v) exactly for a circular orbit.
#[test]
fn o_orbit_info_circular_round_trip() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0;
    let mu = G * (M_SUN_ME + 1.0);
    let v = (mu / a).sqrt();
    b.push(a, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    let o = orbit_info(&b, 1, 0);
    assert!(o.bound);
    assert!(o.ecc.abs() < 1e-9, "ecc {}", o.ecc);
    assert!((o.semi_major - a).abs() < 1e-9, "a {}", o.semi_major);
    assert!((o.dist - a).abs() < 1e-9);
    assert!((o.speed_rel - v).abs() < 1e-9);
    let expected_period = std::f64::consts::TAU * (a * a * a / mu).sqrt();
    assert!(
        (o.period - expected_period).abs() / expected_period < 1e-9,
        "period {} vs {}",
        o.period,
        expected_period
    );
}

/// `orbit_info` must recover (a, e) for an elliptical orbit generated by
/// `kepler_state`.
#[test]
fn o_orbit_info_elliptical_round_trip() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 10.0;
    let e = 0.3;
    let mu = G * (M_SUN_ME + 1.0);
    let (x, y, vx, vy) = kepler_state(mu, a, e, 1.1, 0.4);
    b.push(x, y, vx, vy, 1.0, BodyKind::Planet);
    let o = orbit_info(&b, 1, 0);
    assert!(o.bound);
    assert!((o.ecc - e).abs() < 1e-8, "ecc {} vs {}", o.ecc, e);
    assert!(
        (o.semi_major - a).abs() < 1e-8,
        "a {} vs {}",
        o.semi_major,
        a
    );
    assert!((o.peri - a * (1.0 - e)).abs() < 1e-8);
    assert!((o.apo - a * (1.0 + e)).abs() < 1e-8);
}

/// A body above escape speed must be reported as unbound with infinite
/// `semi_major` / `period` and `ecc > 1`.
#[test]
fn o_orbit_info_hyperbolic_unbound() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let r = 5.0;
    let v_esc = (2.0 * G * M_SUN_ME / r).sqrt();
    b.push(r, 0.0, 0.0, 2.0 * v_esc, 1.0, BodyKind::Planet);
    let o = orbit_info(&b, 1, 0);
    assert!(!o.bound);
    assert_eq!(o.semi_major, f64::INFINITY);
    assert_eq!(o.period, f64::INFINITY);
    assert!(o.ecc > 1.0, "ecc {}", o.ecc);
}

/// `circularize` must zero the eccentricity and preserve the orbital radius.
#[test]
fn o_circularize_zeroes_eccentricity() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0;
    let mu = G * (M_SUN_ME + 1.0);
    let (x, y, vx, vy) = kepler_state(mu, a, 0.4, 1.2, 0.5);
    b.push(x, y, vx, vy, 1.0, BodyKind::Planet);
    let r_before = (b.x[1] * b.x[1] + b.y[1] * b.y[1]).sqrt();
    assert!(circularize(&mut b, 1));
    let o = orbit_info(&b, 1, 0);
    assert!(o.ecc < 1e-10, "ecc after circularize: {}", o.ecc);
    let r_after = (b.x[1] * b.x[1] + b.y[1] * b.y[1]).sqrt();
    assert!((r_after - r_before).abs() < 1e-10);
}

/// `host_distance` for a moon must return the distance to the *host star*,
/// not to its parent planet — that's the value the renderer uses for the
/// planet colour gradient, and getting it wrong would collapse a Jupiter
/// moon onto Jupiter's distance.
#[test]
fn o_host_distance_from_moon() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0 * au_in_gm();
    let v_jup = (G * M_SUN_ME / a).sqrt();
    b.push(a, 0.0, 0.0, v_jup, 318.0, BodyKind::Planet);
    // Moon 0.1 Gm from Jupiter, at Jupiter's velocity plus the moon's own
    // circular velocity around Jupiter.
    let v_moon = v_jup + (G * 318.0 / 0.1).sqrt();
    b.push(a + 0.1, 0.0, 0.0, v_moon, 0.01, BodyKind::Planet);
    b.refresh_parent_cache();
    assert_eq!(find_parent(&b, 2), Some(1), "moon must orbit Jupiter");
    let d = host_distance(&b, 2);
    let expected = a + 0.1;
    assert!(
        (d - expected).abs() < 0.01,
        "host_distance = {} (expected ~{})",
        d,
        expected
    );
}

// ===========================================================================
// p — escape culling
// ===========================================================================

/// A planet beyond CULL_DIST moving radially outward at > v_esc must be
/// culled and logged.
#[test]
fn p_escape_cull_removes_outward_unbound() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let d = CULL_DIST * 2.0;
    let v_esc = (2.0 * G * M_SUN_ME / d).sqrt();
    b.push(d, 0.0, 2.0 * v_esc, 0.0, 1.0, BodyKind::Planet);
    cull_escaped(&mut b);
    assert_eq!(b.len(), 1);
    assert_eq!(b.stats.escaped, 1);
}

/// A planet beyond CULL_DIST moving radially inward must be kept (it may
/// still return) even if its speed exceeds v_esc.
#[test]
fn p_escape_cull_keeps_inward() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let d = CULL_DIST * 2.0;
    let v_esc = (2.0 * G * M_SUN_ME / d).sqrt();
    b.push(d, 0.0, -2.0 * v_esc, 0.0, 1.0, BodyKind::Planet);
    cull_escaped(&mut b);
    assert_eq!(b.len(), 2, "inward body must not be culled");
    assert_eq!(b.stats.escaped, 0);
}

/// Beyond HARD_CULL_DIST the body is removed regardless of velocity.
#[test]
fn p_escape_cull_hard_removes_regardless() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let d = HARD_CULL_DIST * 1.5;
    b.push(d, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    cull_escaped(&mut b);
    assert_eq!(b.len(), 1);
}

// ===========================================================================
// q — energy diagnostics
// ===========================================================================

/// On a stable solar system the raw integrator drift stays under 0.5 %.
#[test]
fn q_energy_drift_small_on_solar() {
    let mut b = Bodies::with_capacity(16);
    spawn_solar_system(&mut b);
    let mut max_drift = 0.0_f64;
    for _ in 0..500 {
        advance(&mut b, DT);
        if let Some(d) = energy_drift_pct(&b) {
            max_drift = max_drift.max(d.abs());
        }
    }
    assert!(max_drift < 0.5, "max drift {:.4}%", max_drift);
}

/// Above `ENERGY_MAX_N` the diagnostic helper must return `None` rather
/// than perform the O(n²) sum.
#[test]
fn q_energy_drift_returns_none_above_limit() {
    let mut b = Bodies::with_capacity(ENERGY_MAX_N + 10);
    for k in 0..ENERGY_MAX_N + 10 {
        b.push(k as f64, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    }
    rebase_energy(&mut b);
    assert_eq!(energy_drift_pct(&b), None);
    assert_eq!(energy_dissipated_pct(&b), None);
}

/// `settle_energy` must accumulate `energy_offset` by exactly
/// (post-collision energy - pre-collision energy).
#[test]
fn q_settle_energy_computes_offset() {
    let mut b = pair(1.0, 0.3, 0.4, 0.6);
    shift(&mut b, 40.0, -25.0, 3.0, 2.0);
    rebase_energy(&mut b);
    let e_pre = total_energy(&b).unwrap();
    collision_pass(&mut b, DT, None);
    settle_energy(&mut b);
    let e_post = total_energy(&b).unwrap();
    let expected = e_post - e_pre;
    assert!(
        (b.stats.energy_offset - expected).abs() < 1e-9 * (1.0 + expected.abs()),
        "offset {} vs expected {}",
        b.stats.energy_offset,
        expected
    );
}

// ===========================================================================
// r — kind transitions via merge
// ===========================================================================

/// A sub-`STAR_M_MIN` "star" that gains a tiny bit of mass must be
/// demoted to a brown dwarf.
#[test]
fn r_star_to_bd_transition() {
    let mut b = Bodies::with_capacity(2);
    let tiny_star_mass = 0.5 * STAR_M_MIN;
    b.push(0.0, 0.0, 0.0, 0.0, tiny_star_mass, BodyKind::Star);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1);
    assert_eq!(b.kind[0], BodyKind::BrownDwarf, "demoted to BD");
    assert!(b.m[0] >= BD_M_MIN);
}

/// Two brown dwarfs whose summed mass exceeds `BD_M_MAX` must ignite.
#[test]
fn r_bd_to_star_transition() {
    let mut b = Bodies::with_capacity(2);
    // 45 M_jup each = 14310 M_earth; sum 28620 > BD_M_MAX = 25440.
    let m_bd = 45.0 * JUPITER_MASS;
    b.push(0.0, 0.0, 0.0, 0.0, m_bd, BodyKind::BrownDwarf);
    b.push(0.0, 0.0, 0.0, 0.0, m_bd, BodyKind::BrownDwarf);
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1);
    assert_eq!(b.kind[0], BodyKind::Star, "ignited as a star");
    assert!(b.m[0] >= STAR_M_MIN);
}

// ===========================================================================
// s — user edits
// ===========================================================================

/// `scale_mass(i, 2)` then `scale_mass(i, 0.5)` must return to the original
/// mass and radius.
#[test]
fn s_scale_mass_round_trip() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let (m0, r0) = (b.m[0], b.r[0]);
    scale_mass(&mut b, 0, 2.0);
    assert!((b.m[0] - 2.0 * m0).abs() < 1e-12);
    scale_mass(&mut b, 0, 0.5);
    assert!((b.m[0] - m0).abs() < 1e-12);
    assert!((b.r[0] - r0).abs() < 1e-12, "radius round-trips too");
}

/// `set_kind(BlackHole)` on a low-mass body must clamp its mass to the
/// black-hole minimum.
#[test]
fn s_set_kind_clamps_mass() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    set_kind(&mut b, 0, BodyKind::BlackHole);
    assert_eq!(b.kind[0], BodyKind::BlackHole);
    assert!(b.m[0] >= BH_M_MIN, "mass {} < BH_M_MIN", b.m[0]);
}

/// Deleting a body must compact the arrays and shift surviving ids down.
#[test]
fn s_delete_body_compacts() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tail_id = b.push(2.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let tail = b.id[tail_id];
    delete_body(&mut b, 1);
    assert_eq!(b.len(), 2);
    assert_eq!(b.id[1], tail, "surviving body shifts down");
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
// u — RNG / samplers
// ===========================================================================

/// Same seed -> same sequence; different seeds differ within 100 draws.
#[test]
fn u_xor_shift_deterministic() {
    let mut a = XorShift::new(42);
    let mut b = XorShift::new(42);
    for k in 0..100 {
        assert_eq!(a.next_u32(), b.next_u32(), "same seed, draw {}", k);
    }
    let mut a = XorShift::new(42);
    let mut c = XorShift::new(43);
    let mut differ = false;
    for _ in 0..100 {
        if a.next_u32() != c.next_u32() {
            differ = true;
            break;
        }
    }
    assert!(differ, "different seeds should diverge");
}

/// The Kroupa IMF sampler must actually span its declared range
/// [0.08, 50] M_sun and never produce a value outside it.
#[test]
fn u_imf_spans_range() {
    let mut rng = XorShift::new(7);
    let mut min_m = f64::INFINITY;
    let mut max_m = f64::NEG_INFINITY;
    for _ in 0..10_000 {
        let m = sample_imf_mass(&mut rng);
        assert!(
            (0.08 - 1e-9..=50.0 + 1e-9).contains(&m),
            "out of range {}",
            m
        );
        min_m = min_m.min(m);
        max_m = max_m.max(m);
    }
    assert!(min_m < 0.1, "min {} never near the low end", min_m);
    assert!(max_m > 5.0, "max {} never near the high end", max_m);
}

/// The Poisson sampler's empirical mean matches its rate parameter.
#[test]
fn u_poisson_mean() {
    let mut rng = XorShift::new(11);
    let lambda = 3.0;
    let n = 5000u64;
    let sum: u64 = (0..n).map(|_| poisson(&mut rng, lambda) as u64).sum();
    let mean = sum as f64 / n as f64;
    // stddev of mean ≈ sqrt(3 / 5000) ≈ 0.0245; tolerance 0.2 is ~8 sigma.
    assert!(
        (mean - lambda).abs() < 0.2,
        "mean {} vs lambda {}",
        mean,
        lambda
    );
}

/// `solve_kepler(M, e)` must satisfy Kepler's equation M = E - e sin E to
/// machine precision across a uniform sweep of M.
#[test]
fn u_kepler_equation_roundtrip() {
    let e = 0.3;
    for k in 0..20 {
        let m = k as f64 / 20.0 * std::f64::consts::TAU;
        let ea = solve_kepler(m, e);
        let m_verify = ea - e * ea.sin();
        assert!((m_verify - m).abs() < 1e-10, "M={} vs {}", m, m_verify);
    }
}

// ===========================================================================
// v — geometry helpers
// ===========================================================================

/// `interact_frac` is bounded by [0, 1] and decreases monotonically with
/// impact parameter.
#[test]
fn v_interact_frac_bounds() {
    for rp in [0.001, 0.01, 0.1] {
        let rt = rp * 2.0;
        let mut prev = f64::INFINITY;
        for k in 0..=100 {
            let bb = k as f64 / 100.0;
            let f = interact_frac(rt, rp, bb);
            assert!((0.0..=1.0).contains(&f), "bb {} -> {}", bb, f);
            assert!(f <= prev + 1e-9, "bb {} -> {} (was {})", bb, f, prev);
            prev = f;
        }
        // Head-on: full interaction.
        assert!((interact_frac(rt, rp, 0.0) - 1.0).abs() < 1e-9);
        // Grazing: minimal interaction.
        assert!(interact_frac(rt, rp, 1.0) < 0.05);
    }
}

/// A perfectly head-on encounter has impact parameter zero.
#[test]
fn v_impact_offset_head_on_zero() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(-1.0, 0.0, 1.0, 0.0, 1.0, BodyKind::Planet);
    // dx = -1, dy = 0, vx = 1, vy = 0 -> offset = |(-1)*0 - 0*1| / 1 = 0.
    assert!((impact_offset(&b, 0, 1) - 0.0).abs() < 1e-10);
}

/// A purely tangential encounter has impact parameter equal to the
/// separation.
#[test]
fn v_impact_offset_grazing_max() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(-1.0, 0.0, 0.0, 1.0, 1.0, BodyKind::Planet);
    // dx = -1, dy = 0, vx = 0, vy = 1 -> offset = |(-1)*1 - 0*0| / 1 = 1.
    assert!((impact_offset(&b, 0, 1) - 1.0).abs() < 1e-10);
}

/// Earth's Hill radius around the Sun must come out to ~0.01 AU.
#[test]
fn v_hill_radius_earth_sun() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    b.push(au, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let r = hill_radius(&b, 1, 0);
    let expected = au * (1.0 / (3.0 * M_SUN_ME)).cbrt();
    assert!(
        (r - expected).abs() < 1e-9 * au,
        "hill {} vs {}",
        r,
        expected
    );
    // Sanity: about 0.01 AU.
    assert!((r / au - 0.01).abs() < 1e-4);
}

/// The Roche limit of an Earth-density body around the Sun must lie
/// between the rigid (1.26 × R) and fluid (2.44 × R) coefficient bounds,
/// scaled by (M/m)^(1/3).
#[test]
fn v_roche_limit_earth_sun() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    b.push(au, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let roche = roche_limit(&b, 1, 0);
    let r_e = planet_true_r(1.0);
    let scale = (M_SUN_ME / 1.0).cbrt();
    let lo = ROCHE_RIGID * r_e * scale;
    let hi = ROCHE_FLUID * r_e * scale;
    assert!(
        roche >= lo - 1e-9 && roche <= hi + 1e-9,
        "roche {} not in [{}, {}]",
        roche,
        lo,
        hi
    );
    // Earth has rho ≈ 5.5 g/cm³, above ROCHE_RHO_RIGID = 4, so it should
    // be fully rigid: roche ≈ ROCHE_RIGID * r_e * scale.
    let expected_rigid = ROCHE_RIGID * r_e * scale;
    assert!(
        (roche - expected_rigid).abs() / expected_rigid < 0.01,
        "Earth should be fully rigid: {} vs {}",
        roche,
        expected_rigid
    );
}

// ===========================================================================
// w — naming helpers
// ===========================================================================

#[test]
fn w_planet_class_thresholds() {
    assert_eq!(planet_class(0.5), "rocky");
    assert_eq!(planet_class(4.999), "rocky");
    assert_eq!(planet_class(5.0), "super-earth");
    assert_eq!(planet_class(9.999), "super-earth");
    assert_eq!(planet_class(10.0), "ice giant");
    assert_eq!(planet_class(49.999), "ice giant");
    assert_eq!(planet_class(50.0), "gas giant");
    assert_eq!(planet_class(317.999), "gas giant");
    assert_eq!(planet_class(318.0), "super-jupiter");
}

#[test]
fn w_star_class_thresholds() {
    assert_eq!(star_class(0.1 * M_SUN_ME), 'M');
    assert_eq!(star_class(0.6 * M_SUN_ME), 'K');
    assert_eq!(star_class(1.0 * M_SUN_ME), 'G');
    assert_eq!(star_class(1.3 * M_SUN_ME), 'F');
    assert_eq!(star_class(2.0 * M_SUN_ME), 'A');
    assert_eq!(star_class(5.0 * M_SUN_ME), 'B');
}

#[test]
fn w_bd_class_thresholds() {
    assert_eq!(bd_class(60.0 * JUPITER_MASS), 'M');
    assert_eq!(bd_class(40.0 * JUPITER_MASS), 'L');
    assert_eq!(bd_class(25.0 * JUPITER_MASS), 'T');
    assert_eq!(bd_class(15.0 * JUPITER_MASS), 'Y');
}

#[test]
fn w_mass_str_formats() {
    let s = mass_str(M_SUN_ME);
    assert!(s.contains("M☉") && s.contains("1.00"), "{}", s);
    let s = mass_str(1.0);
    assert!(s.contains("M⊕") && s.contains("1.00"), "{}", s);
    let s = mass_str(300.0);
    assert!(s.contains("M⊕") && s.contains("300"), "{}", s);
}

#[test]
fn w_body_class_moon_vs_planet() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0 * au_in_gm();
    let v = (G * M_SUN_ME / a).sqrt();
    // 200 M_E: comfortably a gas giant (50 <= m < 318).
    b.push(a, 0.0, 0.0, v, 200.0, BodyKind::Planet);
    let v_moon = v + (G * 200.0 / 0.1).sqrt();
    b.push(a + 0.1, 0.0, 0.0, v_moon, 0.01, BodyKind::Planet);
    b.refresh_parent_cache();
    assert_eq!(body_class(&b, 1), "gas giant");
    assert_eq!(body_class(&b, 2), "moon");
}

// ===========================================================================
// x — star & habitability
// ===========================================================================

#[test]
fn x_luminosity_mass_law() {
    assert!((star_luminosity(BodyKind::Star, M_SUN_ME) - 1.0).abs() < 1e-6);
    // 10 M☉ → 10^3.5 = 3162
    let l10 = star_luminosity(BodyKind::Star, 10.0 * M_SUN_ME);
    assert!((l10 / 3162.28 - 1.0).abs() < 0.01, "10 M☉ L = {}", l10);
    assert!(star_luminosity(BodyKind::Planet, 1.0) == 0.0);
}

#[test]
fn x_effective_temperature() {
    let t = star_temperature(BodyKind::Star, M_SUN_ME);
    assert!((t - 5772.0).abs() < 10.0, "sun T = {}", t);
    // M dwarf: T ∝ m^0.475, so 0.2 M☉ → ~2,900 K.
    let t02 = star_temperature(BodyKind::Star, 0.2 * M_SUN_ME);
    assert!(t02 < 4000.0, "0.2 M☉ T = {}", t02);
    // 5 M☉: 5^0.475 = 2.15 → ~12,400 K.
    let t5 = star_temperature(BodyKind::Star, 5.0 * M_SUN_ME);
    assert!(t5 > 12_000.0, "5 M☉ T = {}", t5);
    // 10 M☉: 10^0.475 = 2.99 → ~17,200 K.
    let t10 = star_temperature(BodyKind::Star, 10.0 * M_SUN_ME);
    assert!(t10 > 16_000.0, "10 M☉ T = {}", t10);
}

#[test]
fn x_main_sequence_lifetime() {
    assert!((star_lifetime_gyr(BodyKind::Star, M_SUN_ME) - 10.0).abs() < 0.01);
    let l10 = star_lifetime_gyr(BodyKind::Star, 10.0 * M_SUN_ME);
    assert!((l10 - 0.0316).abs() < 1e-3, "10 M☉ lifetime {}", l10);
    assert!(star_lifetime_gyr(BodyKind::BrownDwarf, BD_M_MIN).is_infinite());
}

#[test]
fn x_habitable_zone_and_flux() {
    let (inner, outer) = habitable_zone_au(BodyKind::Star, M_SUN_ME);
    assert!(inner < 1.0 && outer > 1.0, "HZ {:?}..{:?}", inner, outer);
    let (i, o) = habitable_zone_au(BodyKind::Planet, 1.0);
    assert!(i.is_nan() && o.is_nan());

    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    assert!((stellar_flux(&b, au, 0.0, 1) - 1.0).abs() < 1e-6);
    assert!((stellar_flux(&b, 2.0 * au, 0.0, 1) - 0.25).abs() < 1e-6);
    assert_eq!(stellar_flux(&b, 0.0, 0.0, 0), 0.0);
    // Earth's equilibrium temperature should be ~255 K.
    assert!((equilibrium_temp(1.0) - 255.0).abs() < 1.0);
}

// ===========================================================================
// y — samplers
// ===========================================================================

#[test]
fn y_rayleigh_mean_matches_theory() {
    let mut rng = XorShift::new(3);
    let sigma = 0.1_f64;
    let n = 20_000;
    let mean: f64 = (0..n)
        .map(|_| sample_rayleigh(&mut rng, sigma))
        .sum::<f64>()
        / n as f64;
    let expected = sigma * (std::f64::consts::PI / 2.0).sqrt();
    assert!(
        (mean - expected).abs() < 0.01,
        "mean {} vs {}",
        mean,
        expected
    );
}

#[test]
fn y_power_law_in_range_and_biased_low() {
    let mut rng = XorShift::new(5);
    let (lo, hi, p) = (0.1, 5.0, 1.5);
    let mut below_median = 0;
    for _ in 0..2000 {
        let m = sample_mass_power_law(&mut rng, lo, hi, p);
        assert!(m >= lo - 1e-9 && m <= hi + 1e-9);
        if m < (lo + hi) * 0.5 {
            below_median += 1;
        }
    }
    // p = 1.5 puts most samples below the midpoint.
    assert!(
        below_median > 1200,
        "distribution should skew low: {}",
        below_median
    );
}

#[test]
fn y_normal_mean_var() {
    let mut rng = XorShift::new(7);
    let n = 20_000;
    let mut s = 0.0;
    let mut ss = 0.0;
    for _ in 0..n {
        let x = sample_normal(&mut rng, 2.0, 0.5);
        s += x;
        ss += x * x;
    }
    let m = s / n as f64;
    let v = ss / n as f64 - m * m;
    assert!((m - 2.0).abs() < 0.05, "mean {}", m);
    assert!((v - 0.25).abs() < 0.02, "var {}", v);
}

#[test]
fn y_imf_median_around_02_msun() {
    // Kroupa (2001) → median ~0.2 M☉ for the 0.08..50 range.
    let mut rng = XorShift::new(42);
    let mut ms: Vec<f64> = (0..20_000).map(|_| sample_imf_mass(&mut rng)).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ms[ms.len() / 2];
    assert!(med > 0.15 && med < 0.35, "IMF median {}", med);
}

// ===========================================================================
// z — small utilities
// ===========================================================================

#[test]
fn z_time_unit_is_about_fifty_days() {
    let days = time_unit_s() / 86_400.0;
    assert!(days > 40.0 && days < 60.0, "{} days", days);
}

#[test]
fn z_index_of_id() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let idx = b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let id = b.id[idx];
    assert_eq!(b.index_of_id(id), Some(1));
    assert_eq!(b.index_of_id(9999), None);
}

#[test]
fn z_set_mass_updates_radius() {
    // `set_mass` recomputes the radius from the body's bulk density, not
    // the mass-radius law (which is only used to seed `bulk_density`).
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let r1 = b.r[0];
    b.set_mass(0, 4.0);
    assert!(b.r[0] > r1);
    assert!((b.r[0] - radius_from_density(4.0, b.bulk_density[0])).abs() < 1e-15);
}

#[test]
fn z_log_ring_buffer_wraps() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    for k in 0..EVENT_LOG_MAX + 50 {
        b.log(EventKind::User, format!("e{}", k), 0.0, 0.0);
    }
    assert_eq!(b.events.len(), EVENT_LOG_MAX);
    assert!(b
        .events
        .back()
        .unwrap()
        .text
        .contains(&format!("{}", EVENT_LOG_MAX + 49)));
}

#[test]
fn z_contact_r_black_hole() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 10.0 * M_SUN_ME, BodyKind::BlackHole);
    let expected = (BH_CAPTURE_HORIZONS * b.r[0]).max(BH_MIN_CAPTURE);
    assert!((b.contact_r(0) - expected).abs() < 1e-15);
    let mut p = Bodies::with_capacity(1);
    p.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    assert_eq!(p.contact_r(0), p.r[0]);
}

#[test]
fn z_recenter_com_zeroes_both() {
    let mut b = Bodies::with_capacity(2);
    b.push(1.0, 2.0, 3.0, 4.0, M_SUN_ME, BodyKind::Star);
    b.push(-1.0, -2.0, -3.0, -4.0, 1.0, BodyKind::Planet);
    recenter_com(&mut b);
    let (px, py) = momentum(&b);
    assert!(px.abs() < 1e-9 && py.abs() < 1e-9);
    let mt: f64 = b.m.iter().sum();
    let cx: f64 = (0..b.len()).map(|i| b.m[i] * b.x[i]).sum::<f64>() / mt;
    let cy: f64 = (0..b.len()).map(|i| b.m[i] * b.y[i]).sum::<f64>() / mt;
    assert!(cx.abs() < 1e-9 && cy.abs() < 1e-9);
}

#[test]
fn z_cell_key_distinct_nearby() {
    let k00 = cell_key(0, 0);
    assert_eq!(k00, cell_key(0, 0));
    assert_ne!(k00, cell_key(1, 0));
    assert_ne!(k00, cell_key(0, 1));
}

#[test]
fn z_au_in_gm() {
    assert!((au_in_gm() - 149.6).abs() < 0.1);
}

#[test]
fn z_solar_extent_monotone() {
    assert!((solar_extent_gm(1) - 5000.0).abs() < 1.0);
    assert!(solar_extent_gm(3) > solar_extent_gm(1));
    assert!(solar_extent_gm(6) > solar_extent_gm(3));
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

// ===========================================================================
// ab — fast (hot) collision path
// ===========================================================================

#[test]
fn ab_hot_small_list_matches_full() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let r = b.contact_r(0);
    b.push(r - 0.001, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let hot = vec![0usize, 1usize];
    assert_eq!(find_collisions_hot(&b, &hot, false, DT), vec![(0, 1)]);
}

#[test]
fn ab_hot_grid_matches_full_path() {
    // 30 hot bodies on a 0.01 grid → exercises the grid branch.
    let mut b = Bodies::with_capacity(30);
    for k in 0..30 {
        b.push(k as f64 * 0.01, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    }
    const { assert!(30 >= HOT_GRID_MIN) };
    let hot: Vec<usize> = (0..30).collect();
    let mut full = find_collisions(&b, false, DT);
    let mut fast = find_collisions_hot(&b, &hot, false, DT);
    full.sort_unstable();
    fast.sort_unstable();
    assert_eq!(full, fast);
    assert!(!full.is_empty());
}

// ===========================================================================
// ac — rubble bounce
// ===========================================================================

#[test]
fn ac_rubble_bounce_noop_on_graze() {
    // `rubble_bounce` returns early when `vn >= 0`. In a head-on
    // configuration `vn = -s·v/rr` is always < 0 (s = sqrt(rr² - d²), v > 0),
    // so head-on always fires regardless of approach vs separation sign.
    // The guard only fires for a *graze*: perpendicular offset d >= rr,
    // which makes s = 0 -> vn = 0.
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 0.001, BodyKind::Planet);
    b.push(0.0, 0.0, 1.0, 0.0, 0.001, BodyKind::Planet);
    b.origin[0] = BodyOrigin::ImpactDebris;
    b.origin[1] = BodyOrigin::ImpactDebris;
    let rr = b.contact_r(0) + b.contact_r(1);
    // Move body 1 perpendicular to its velocity by more than rr.
    b.y[1] = rr + 0.001;
    let v0x = b.vx[1];
    let v0y = b.vy[1];
    let (px0, py0) = momentum(&b);
    rubble_bounce(&mut b, 0, 1, rr);
    assert_eq!(b.vx[1], v0x);
    assert_eq!(b.vy[1], v0y);
    assert_eq!(b.stats.bounces, 0);
    let (px1, py1) = momentum(&b);
    assert_eq!(px1, px0);
    assert_eq!(py1, py0);
}

// ===========================================================================
// ad — fragmentation & disruption
// ===========================================================================

#[test]
fn ad_frag_masses_sum_and_skew() {
    let mut rng = XorShift::new(11);
    let md = 0.1;

    // Per-draw invariants hold for every n.
    for n in [2, 5, 20] {
        for _ in 0..50 {
            let ms = frag_masses(&mut rng, n, md);
            assert_eq!(ms.len(), n);
            let sum: f64 = ms.iter().sum();
            assert!((sum - md).abs() < 1e-9, "n={} sum {} vs {}", n, sum, md);
            assert!(ms.iter().all(|&m| m > 0.0), "n={} non-positive mass", n);
        }
    }

    // The Pareto tail (FRAG_SLOPE = 2.85 → beta = 0.95 < 1) has infinite
    // mean, so a *single* draw's CV is not reliable. Average it.
    for n in [5, 20] {
        let draws = 100;
        let mut cv_sum = 0.0;
        for _ in 0..draws {
            let ms = frag_masses(&mut rng, n, md);
            let mean = md / n as f64;
            let var: f64 = ms.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            cv_sum += var.sqrt() / mean;
        }
        let cv_mean = cv_sum / draws as f64;
        assert!(
            cv_mean > 0.20,
            "n={} mean CV over {} draws = {:.3} (should be heavy-tailed)",
            n,
            draws,
            cv_mean
        );
    }
}

#[test]
fn ad_density_and_rigidity() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    let d = density_gcc(&b, 0);
    assert!((d - 5.51).abs() < 0.5, "Earth density {}", d);
    assert!((rigidity(6.0) - 1.0).abs() < 1e-9);
    assert!((rigidity(1.0) - 0.0).abs() < 1e-9);
    let mid = (ROCHE_RHO_FLUID + ROCHE_RHO_RIGID) * 0.5;
    assert!((rigidity(mid) - 0.5).abs() < 1e-9);
}

#[test]
fn ad_disrupt_fragment_count_and_conservation() {
    let mut b = Bodies::with_capacity(20);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let p = b.push(1.0, 0.0, 0.0, 0.0, 10.0, BodyKind::Planet);
    let before = b.len();
    let l0 = total_angular_momentum(&b);
    let (px0, py0) = momentum(&b);
    assert!(disrupt(&mut b, p, 0));
    // compact() removes the dead planet; new_len = before - 1 + n_frag.
    b.compact();
    let fragments = b.len() - before + 1;
    assert!((4..=12).contains(&fragments), "fragments {}", fragments);
    let (px1, py1) = momentum(&b);
    assert!((px1 - px0).abs() < 1e-8 * (1.0 + px0.abs()));
    assert!((py1 - py0).abs() < 1e-8 * (1.0 + py0.abs()));
    assert!((total_angular_momentum(&b) - l0).abs() < 1e-8 * (1.0 + l0.abs()));
}

// ===========================================================================
// ae — integrator internals
// ===========================================================================

#[test]
fn ae_assign_rungs_uses_dtl() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.dtl[0] = f64::INFINITY;
    b.dtl[1] = DT / 8.0;
    assert_eq!(assign_rungs(&mut b, DT), 3);
    assert_eq!(b.rung[0], 0);
    assert_eq!(b.rung[1], 3);
}

#[test]
fn ae_flash_lifecycle() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    flash_at(&mut b, 1.0, 2.0, 0.1);
    age_flashes(&mut b, FLASH_LIFE * 0.5);
    assert_eq!(b.flashes.len(), 1);
    age_flashes(&mut b, FLASH_LIFE);
    assert_eq!(b.flashes.len(), 0);
}

#[test]
fn ae_yoshida_energy_conservation() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0;
    let v = (G * M_SUN_ME / a).sqrt();
    b.push(a, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    compute_accel(&mut b);
    rebase_energy(&mut b);
    for _ in 0..200 {
        yoshida_step(&mut b, DT);
    }
    let drift = energy_drift_pct(&b).unwrap().abs();
    assert!(drift < 0.05, "yoshida drift {:.4}%", drift);
}

#[test]
fn ae_yoshida_beats_leapfrog() {
    // Equal force-evaluation cost: 100 Yoshida calls = 300 leapfrog steps.
    let run = |step: fn(&mut Bodies, f64), n: usize| -> f64 {
        let mut b = Bodies::with_capacity(2);
        b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
        let a = 5.0;
        let v = (G * M_SUN_ME / a).sqrt();
        b.push(a, 0.0, 0.0, v, 1.0, BodyKind::Planet);
        compute_accel(&mut b);
        rebase_energy(&mut b);
        for _ in 0..n {
            step(&mut b, DT);
        }
        energy_drift_pct(&b).unwrap().abs()
    };
    let dl = run(leapfrog_step, 300);
    let dy = run(yoshida_step, 100);
    eprintln!("leapfrog drift {:.4e}, yoshida drift {:.4e}", dl, dy);
    assert!(dy < dl, "4th-order should beat 2nd at equal cost");
}

// ===========================================================================
// af — silent merge
// ===========================================================================

#[test]
fn af_silent_merge_does_not_log_or_flash() {
    let mut b = pair(1.0, 0.1, 0.3, 0.5);
    let l0 = total_angular_momentum(&b);
    let (px0, py0) = momentum(&b);
    let m0 = mass(&b);
    let ev0 = b.events.len();
    let fl0 = b.flashes.len();
    merge_impl(&mut b, 0, 1, /*silent=*/ true);
    // merge_impl marks the victim dead but leaves compaction to the caller.
    assert!(b.dead.iter().any(|&d| d), "victim should be dead");
    b.compact();
    assert_eq!(b.len(), 1);
    assert_eq!(b.events.len(), ev0);
    assert_eq!(b.flashes.len(), fl0);
    assert_eq!(b.stats.merges, 1);
    assert!((mass(&b) - m0).abs() < 1e-12);
    let (px1, py1) = momentum(&b);
    assert!((px1 - px0).abs() < 1e-9);
    assert!((py1 - py0).abs() < 1e-9);
    assert!((total_angular_momentum(&b) - l0).abs() < 1e-9);
}

// ===========================================================================
// ag — orbit / anomaly helpers
// ===========================================================================

#[test]
fn ag_push_kepler_yields_requested_elements() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mut rng = XorShift::new(5);
    let a = 5.0;
    push_kepler(&mut b, 0, a, 0.0, 1.0, &mut rng, BodyKind::Planet);
    let o = orbit_info(&b, 1, 0);
    assert!(o.bound);
    // push_kepler and orbit_info both use G * (m_star + m_planet).
    assert!(
        (o.semi_major / a - 1.0).abs() < 1e-9,
        "a {:.8} vs {}",
        o.semi_major,
        a
    );
    assert!(o.ecc < 1e-4, "ecc {}", o.ecc);
}

#[test]
fn ag_true_anomaly_endpoints() {
    // Circular: ν = E
    for k in 0..20 {
        let e_anom = k as f64 / 20.0 * std::f64::consts::TAU;
        let nu = true_anomaly(e_anom, 0.0);
        let d = (nu - e_anom).rem_euclid(std::f64::consts::TAU);
        assert!(d < 1e-9 || (std::f64::consts::TAU - d) < 1e-9);
    }
    assert!(true_anomaly(0.0, 0.3).abs() < 1e-9);
    assert!((true_anomaly(std::f64::consts::PI, 0.3) - std::f64::consts::PI).abs() < 1e-9);
}

#[test]
fn ag_orbital_period_earth_one_year() {
    let p = orbital_period_days(au_in_gm(), M_SUN_ME);
    assert!((p - 365.25).abs() < 1.0, "Earth year {}", p);
}

#[test]
fn ag_binary_fraction_monotone() {
    assert!(binary_fraction(0.3) < binary_fraction(1.0));
    assert!(binary_fraction(1.0) < binary_fraction(3.0));
    for m in [0.1, 0.5, 1.0, 2.0, 5.0] {
        let f = binary_fraction(m);
        assert!((0.0..=1.0).contains(&f));
    }
}

// ===========================================================================
// ah — osculating ellipse geometry
// ===========================================================================

#[test]
fn ah_osculating_ellipse_circular() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0;
    let v = (G * M_SUN_ME / a).sqrt();
    b.push(a, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    let pts = osculating_ellipse(&b, 1, 0.0, 0.0, 0.0, 0.0, M_SUN_ME, 1.0).unwrap();
    assert!(pts.len() >= 32, "expected at least 32 points");
    for p in &pts {
        let r = ((p[0] as f64).powi(2) + (p[1] as f64).powi(2)).sqrt();
        assert!((r - a).abs() < 1e-3, "point r={} vs {}", r, a);
    }
}

#[test]
fn ah_osculating_ellipse_hyperbolic_none() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0;
    let v_esc = (2.0 * G * M_SUN_ME / a).sqrt();
    b.push(a, 0.0, 0.0, 2.0 * v_esc, 1.0, BodyKind::Planet);
    assert!(osculating_ellipse(&b, 1, 0.0, 0.0, 0.0, 0.0, M_SUN_ME, 1.0).is_none());
}

// ===========================================================================
// ai — accuracy (tighter than the baseline l/n/q tests)
// ===========================================================================

#[test]
fn ai_bh_error_below_documented_bound() {
    // Worst *per-body* error at θ = 0.5. The module doc's "~1 %" is the
    // mean error; the worst-case body can reach ~θ² (~25 %), and empirically
    // ~10 % for this configuration. 15 % catches regressions without
    // pretending θ = 0.5 is more accurate than it is.
    for seed in 0..5u32 {
        let mut b = Bodies::with_capacity(400);
        b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
        let mut rng = XorShift::new(seed);
        for _ in 0..399 {
            let a = 5.0 + 200.0 * rng.next_f64();
            let th = rng.next_f64() * std::f64::consts::TAU;
            let vc = (G * M_SUN_ME / a).sqrt();
            let m = 0.1 + 50.0 * rng.next_f64();
            b.push(
                a * th.cos(),
                a * th.sin(),
                -vc * th.sin(),
                vc * th.cos(),
                m,
                BodyKind::Planet,
            );
        }
        let tree = Tree::build(&b);
        let mut worst = 0.0_f64;
        for i in 0..b.len() {
            let (rx, ry, _) = force_direct_on(&b, i);
            let (ax, ay, _) = tree.accel_for(&b, i);
            let rmag = (rx * rx + ry * ry).sqrt().max(1e-30);
            let dmag = ((ax - rx).powi(2) + (ay - ry).powi(2)).sqrt();
            worst = worst.max(dmag / rmag);
        }
        assert!(
            worst < 0.15,
            "seed {}: worst relative BH error {:.4}",
            seed,
            worst
        );
    }
}

#[test]
fn ai_solar_energy_drift_tight() {
    // 4th-order integrator should keep the Solar System within 0.05 %
    // energy over hundreds of steps.
    let mut b = Bodies::with_capacity(16);
    spawn_solar_system(&mut b);
    let mut worst = 0.0_f64;
    for _ in 0..500 {
        advance(&mut b, DT);
        if let Some(d) = energy_drift_pct(&b) {
            worst = worst.max(d.abs());
        }
    }
    assert!(worst < 0.05, "solar drift {:.4}%", worst);
}

#[test]
fn ai_orbital_elements_preserved_over_one_orbit() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let (x, y, vx, vy) = kepler_state(G * M_SUN_ME, 5.0, 0.3, 1.0, 0.0);
    b.push(x, y, vx, vy, 1.0, BodyKind::Planet);
    let o0 = orbit_info(&b, 1, 0);
    compute_accel(&mut b);
    rebase_energy(&mut b);
    let n = (o0.period / DT) as usize;
    for _ in 0..n {
        advance(&mut b, DT);
    }
    let o1 = orbit_info(&b, 1, 0);
    assert!(
        (o1.semi_major / o0.semi_major - 1.0).abs() < 0.02,
        "a: {} -> {}",
        o0.semi_major,
        o1.semi_major
    );
    assert!(
        (o1.ecc - o0.ecc).abs() < 0.02,
        "e: {} -> {}",
        o0.ecc,
        o1.ecc
    );
}

// ===========================================================================
// aj — realism: Solar System matches known periods
// ===========================================================================

#[test]
fn aj_solar_system_reproduces_real_periods() {
    let mut b = Bodies::with_capacity(16);
    spawn_solar_system(&mut b);
    assert!(b.len() >= 9, "expected at least 9 bodies, got {}", b.len());
    // (mass in M_E as it appears in solar_system.json, real sidereal period in days)
    let expected = [
        (0.0553_f64, 87.97_f64),
        (0.8150, 224.70),
        (1.0000, 365.25),
        (0.1070, 686.98),
        (317.80, 4332.59),
        (95.160, 10759.22),
        (14.540, 30688.5),
        (17.150, 60182.0),
    ];
    for &(m_target, real) in &expected {
        let body = (1..b.len())
            .find(|&i| (b.m[i] - m_target).abs() < 1e-3)
            .unwrap_or_else(|| panic!("no body with mass ~{} M⊕", m_target));
        let o = orbit_info(&b, body, 0);
        let p_days = o.period * time_unit_s() / 86_400.0;
        assert!(
            (p_days / real - 1.0).abs() < 0.05,
            "body {} (m={} M⊕): {:.1} d vs {:.1} d",
            body,
            m_target,
            p_days,
            real
        );
    }
}

// ===========================================================================
// ak — fuzz smoke tests
// ===========================================================================
//
// These tests intentionally do *not* check physics outcomes — those are
// covered by the specific groups. The point here is:
//
//   1. No NaN/Inf ever reaches the state vectors.
//   2. Mass and radius stay strictly positive per body.
//   3. Mass never *increases* across a run (all operations conserve exactly
//      or lose mass via culling; nothing gains).
//   4. n >= BH_THRESHOLD exercises the block-leapfrog path with a live
//      Barnes-Hut tree, including compact()-triggered tree rebuilds inside
//      step_block — the exact class of bug that used to crash at scale.
//
// `FUZZ_ITERS` env var (opt-in) multiplies the run length on the three
// cheap tests. The tree-rebuild test ignores it because it's already heavy.

fn fuzz_iters(default: usize) -> usize {
    std::env::var("FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Assert every state vector field is finite and every physical quantity
/// is in its allowed range.
fn fuzz_assert_sane(b: &Bodies, ctx: &str) {
    for i in 0..b.len() {
        assert!(b.x[i].is_finite(), "{}: x[{}] = {}", ctx, i, b.x[i]);
        assert!(b.y[i].is_finite(), "{}: y[{}] = {}", ctx, i, b.y[i]);
        assert!(b.vx[i].is_finite(), "{}: vx[{}] = {}", ctx, i, b.vx[i]);
        assert!(b.vy[i].is_finite(), "{}: vy[{}] = {}", ctx, i, b.vy[i]);
        assert!(b.ax[i].is_finite(), "{}: ax[{}] = {}", ctx, i, b.ax[i]);
        assert!(b.ay[i].is_finite(), "{}: ay[{}] = {}", ctx, i, b.ay[i]);
        assert!(
            b.m[i].is_finite() && b.m[i] > 0.0,
            "{}: m[{}] = {}",
            ctx,
            i,
            b.m[i]
        );
        assert!(
            b.r[i].is_finite() && b.r[i] >= 0.0,
            "{}: r[{}] = {}",
            ctx,
            i,
            b.r[i]
        );
        assert!(
            b.bulk_density[i].is_finite() && b.bulk_density[i] > 0.0,
            "{}: bulk_density[{}] = {}",
            ctx,
            i,
            b.bulk_density[i]
        );
        assert!(
            b.spin[i].is_finite(),
            "{}: spin[{}] = {}",
            ctx,
            i,
            b.spin[i]
        );
        assert!(!b.dead[i], "{}: body {} dead post-step", ctx, i);
    }
    assert!(b.time.is_finite(), "{}: time = {}", ctx, b.time);
    // Flashes are aged and pruned every step; the buffer should never
    // grow without bound even during sustained tidal disruption.
    assert!(
        b.flashes.len() < 10_000,
        "{}: {} live flashes (leak)",
        ctx,
        b.flashes.len()
    );
    // Event log is a ring buffer of EVENT_LOG_MAX; sanity check the wrap.
    assert!(
        b.events.len() <= EVENT_LOG_MAX,
        "{}: event log overflow",
        ctx
    );
}

/// Solar system: 500 steps, exact conservation (no mergers, no escapes).
#[test]
fn ak_fuzz_solar_conservation() {
    let steps = fuzz_iters(500);
    let mut b = Bodies::with_capacity(16);
    spawn_solar_system(&mut b);
    let m0 = mass(&b);
    let l0 = total_angular_momentum(&b);
    let n0 = b.len();

    for step in 0..steps {
        advance(&mut b, DT);
        if step % 25 == 0 || step + 1 == steps {
            fuzz_assert_sane(&b, &format!("solar step {}", step));
        }
    }

    assert_eq!(b.len(), n0, "solar system lost bodies");
    assert_eq!(b.stats.merges, 0, "unexpected merger in solar system");
    assert_eq!(b.stats.escaped, 0, "unexpected escape in solar system");
    let m1 = mass(&b);
    assert!(
        (m1 - m0).abs() / m0 < 1e-12,
        "mass changed: {} -> {}",
        m0,
        m1
    );
    // The Solar System is no longer Yoshida-only once moons are present:
    // the Moon's dtl < DT, so kmax ≥ 1 and step_block integrates the
    // system. Symplectic drift over 500 steps is ~1e-4; 1e-3 still
    // catches a regression to leapfrog-only (which would be ~1e-2).
    let l1 = total_angular_momentum(&b);
    assert!(
        (l1 - l0).abs() / (1.0 + l0.abs()) < 1e-3,
        "L changed: {} -> {}",
        l0,
        l1
    );
}

/// 10 random seeds × 200 steps × n=30. Stresses mergers, escapes and
/// fragment spawning at small-n (direct-sum path). Asserts mass never
/// grows and no NaN ever appears.
#[test]
fn ak_fuzz_random_invariants() {
    let steps = fuzz_iters(200);
    let n = 30usize;
    let mut worst_mass_loss = 0.0_f64;

    for seed in 0..10u32 {
        let mut b = Bodies::with_capacity(n);
        spawn_system(&mut b, n, seed);
        let m0 = mass(&b);
        assert!(m0 > 0.0);

        for step in 0..steps {
            advance(&mut b, DT);
            if step % 25 == 0 || step + 1 == steps {
                fuzz_assert_sane(&b, &format!("seed {} step {}", seed, step));
            }
            // Mass may only decrease (culling). It must never grow.
            let m = mass(&b);
            assert!(
                m <= m0 * 1.0001,
                "seed {} step {}: mass grew {} -> {}",
                seed,
                step,
                m0,
                m
            );
        }
        let loss = (m0 - mass(&b)) / m0;
        worst_mass_loss = worst_mass_loss.max(loss);
    }
    eprintln!(
        "ak_fuzz_random_invariants: worst mass loss {:.4}",
        worst_mass_loss
    );
}

/// Construct a system with n > BH_THRESHOLD and 20 pairs that overlap
/// *immediately*, forcing 20 merges on the very first tick of step_block.
/// This triggers `compact()` — which invalidates the cached Barnes-Hut
/// tree — inside the tick loop, exercising the rebuild-on-len-change guard
/// in `step_block`. If that guard were missing, this test would panic with
/// an index-out-of-bounds in `Tree::accel_for` within the first tick.
///
/// Setup notes:
///   * Overlapping pairs at 0.01 Gm separation (contact sum ~0.0127), with
///     zero relative velocity. Guaranteed to merge on tick 1 — no swept
///     detection needed, no dependence on drift timing.
///   * 20 pairs. 20 merges still hits `b.len() != tree_len` many times
///     inside one step_block.
///   * Each pair at a distinct radius and angle so no pair-pair overlap.
///   * 10 steps. Merges happen on tick 1, so 10 steps is enough to verify
///     the survivor stays sane and the tree didn't corrupt state.
///   * FUZZ_ITERS deliberately ignored; this test is already heavy
///     (n≈650, kmax≈6 → 64 sub-ticks) and the merge scenario is fully
///     exercised in the first advance.
#[test]
fn ak_fuzz_many_collisions_tree_rebuild() {
    let steps = 10;
    let mut b = Bodies::with_capacity(800);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mut rng = XorShift::new(7);

    // 600 background bodies between r=5 and r=35, spread angularly.
    for k in 0..600 {
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

    // 20 overlapping pairs at distinct radii, already within contact sum.
    // Body 2 sits 0.01 Gm to the +x of body 1 with the SAME velocity, so
    // pair_hit returns Some on the very first collision_pass call.
    for k in 0..20 {
        let r = 5.0 + (k as f64) * 1.5; // radii 5, 6.5, 8, ..., 33.5
        let th = 0.1 + (k as f64) * 0.3; // distinct angles
        let vc = (G * M_SUN_ME / r).sqrt();
        let (cos_th, sin_th) = (th.cos(), th.sin());
        let vx = -vc * sin_th;
        let vy = vc * cos_th;
        let (x1, y1) = (r * cos_th, r * sin_th);
        b.push(x1, y1, vx, vy, 1.0, BodyKind::Planet);
        // +0.01 Gm in x, identical velocity → static overlap → immediate merge.
        b.push(x1 + 0.01, y1, vx, vy, 1.0, BodyKind::Planet);
    }

    assert!(
        b.len() >= BH_THRESHOLD,
        "test setup needs n >= BH_THRESHOLD ({}), got {}",
        BH_THRESHOLD,
        b.len()
    );
    compute_accel(&mut b);

    let m0 = mass(&b);
    let n0 = b.len();

    for step in 0..steps {
        advance(&mut b, DT);
        fuzz_assert_sane(&b, &format!("tree-rebuild step {}", step));
    }

    assert!(
        b.stats.merges >= 20,
        "expected >= 20 merges, got {}",
        b.stats.merges
    );
    assert!(b.len() < n0, "merge should have shrunk the array");
    assert!(mass(&b) <= m0 * 1.001, "mass grew unexpectedly");
}

/// Edge case: a single-body system advances forever without panic or NaN.
/// Cheap sanity that `advance` doesn't require n >= 2 to terminate cleanly.
#[test]
fn ak_fuzz_degenerate_n1() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    compute_accel(&mut b);
    rebase_energy(&mut b);
    for step in 0..500 {
        advance(&mut b, DT);
        if step % 100 == 0 {
            fuzz_assert_sane(&b, &format!("n1 step {}", step));
        }
    }
    assert_eq!(b.len(), 1);
    assert!(b.x[0].abs() < 1e-9 && b.y[0].abs() < 1e-9);
}

// ===========================================================================
// al — stellar evolution
// ===========================================================================

fn sun_system(planet_au: &[f64]) -> Bodies {
    let mut b = Bodies::with_capacity(8);
    let sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    for &a in planet_au {
        push_orbit(&mut b, sun, a, 0.0, 1.0, 0.0);
    }
    recenter_com(&mut b);
    compute_accel(&mut b);
    rebase_energy(&mut b);
    b.refresh_parent_cache();
    b
}

fn run_to_remnant(b: &mut Bodies, max_steps: usize) -> usize {
    for k in 0..max_steps {
        advance(b, DT);
        if !b.kind.contains(&BodyKind::Star) {
            return k;
        }
    }
    panic!("star did not die in {max_steps} steps");
}

#[test]
fn al_evolution_is_off_by_default() {
    let mut b = sun_system(&[1.0]);
    let (m, r) = (b.m[0], b.r[0]);
    for _ in 0..200 {
        advance(&mut b, DT);
    }
    assert_eq!(b.age_gyr[0], 0.0, "age should not advance");
    assert_eq!(b.m[0], m, "mass unchanged");
    assert_eq!(b.r[0], r, "radius unchanged");
    assert_eq!(b.kind[0], BodyKind::Star);
}

#[test]
fn al_age_zero_matches_the_old_relations() {
    // At age 0 the evolution model reproduces the pre-evolution relations
    // exactly, so presets look identical with evolution on and off.
    for s in [0.1_f64, 0.5, 1.0, 2.0, 7.0, 30.0] {
        let m = s * M_SUN_ME;
        let st = star_state(m, m, 0.0);
        assert_eq!(
            st.radius_gm,
            physical_radius(BodyKind::Star, m),
            "star_state at age 0 must match physical_radius"
        );
        assert_eq!(
            st.luminosity,
            star_luminosity(BodyKind::Star, m),
            "star_state at age 0 must match star_luminosity"
        );
        let t = star_temperature(BodyKind::Star, m);
        assert!((st.temperature - t).abs() < 1e-9 * t);
        assert_eq!(st.phase, StellarPhase::MainSequence);
    }
}

#[test]
fn al_giant_phase_is_bright_large_and_cool() {
    let m = M_SUN_ME;
    let t_ms = star_lifetime_gyr(BodyKind::Star, m);
    let late = star_state(m, m * 0.6, t_ms * (1.0 + EVO_POST_MS_FRAC * 0.95));
    assert_eq!(late.phase, StellarPhase::RedGiant);
    assert!(late.radius_gm > 100.0 * R_SUN_GM);
    assert!(late.luminosity > 1000.0);
    assert!(late.temperature < 4500.0);
    // Radius and luminosity are continuous across the MS→giant transition.
    let before = star_state(m, m, t_ms * 0.999_999);
    let after = star_state(m, m, t_ms * 1.000_001);
    assert!((after.radius_gm / before.radius_gm - 1.0).abs() < 1e-3);
    assert!((after.luminosity / before.luminosity - 1.0).abs() < 1e-2);
}

#[test]
fn al_sunlike_star_becomes_a_white_dwarf() {
    let mut b = sun_system(&[30.0]);
    set_evolution_rate(&mut b, 4.0);
    run_to_remnant(&mut b, 5000);
    assert_eq!(b.kind[0], BodyKind::WhiteDwarf);
    // Kalirai relation for 1 M_SUN_ME ≈ 0.503 M_SUN_ME.
    let ms = b.m[0] / M_SUN_ME;
    assert!((ms - 0.503).abs() < 0.02, "WD mass {ms}");
    assert!(b.r[0] < 0.05);
    assert!(b.events.iter().any(|e| e.kind == EventKind::Evolve));
}

#[test]
fn al_massive_star_becomes_a_neutron_star() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, 12.0 * M_SUN_ME, BodyKind::Star);
    let host = b.len() - 1;
    push_orbit(&mut b, host, 50.0, 0.0, 1.0, 0.0);
    b.sn_kicks = false;
    compute_accel(&mut b);
    rebase_energy(&mut b);
    set_evolution_rate(&mut b, 10.0);
    run_to_remnant(&mut b, 1000);
    assert_eq!(b.kind[0], BodyKind::NeutronStar);
    let ms = b.m[0] / M_SUN_ME;
    // Section-B model: NS mass = 1.4 + 0.6 * (12-8)/(20-8) = 1.6.
    assert!((ms - 1.6).abs() < 0.05, "NS mass {ms}");
}

#[test]
fn al_very_massive_star_becomes_a_black_hole() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, 25.0 * M_SUN_ME, BodyKind::Star);
    let host = b.len() - 1;
    push_orbit(&mut b, host, 50.0, 0.0, 1.0, 0.0);
    b.sn_kicks = false;
    compute_accel(&mut b);
    rebase_energy(&mut b);
    set_evolution_rate(&mut b, 10.0);
    run_to_remnant(&mut b, 1000);
    assert_eq!(b.kind[0], BodyKind::BlackHole);
    let ms = b.m[0] / M_SUN_ME;
    // Section-B model: BH mass = max(0.25 * 25, BH_M_MIN) = 6.25.
    assert!((ms - 6.25).abs() < 0.05, "BH mass {ms}");
}

#[test]
fn al_supernova_kick_is_optional() {
    for kicks in [false, true] {
        let mut b = Bodies::with_capacity(2);
        b.push(0.0, 0.0, 0.0, 0.0, 25.0 * M_SUN_ME, BodyKind::Star);
        b.sn_kicks = kicks;
        become_remnant(&mut b, 0);
        let v = b.vx[0].hypot(b.vy[0]);
        if kicks {
            assert!(v > 0.0, "BH kick should give velocity");
        } else {
            assert_eq!(v, 0.0, "kicks off → no velocity");
        }
    }
}

#[test]
fn al_roche_lobe_matches_eggleton() {
    let rl = roche_lobe_radius(1.0, 1.0, 1.0);
    // Eggleton 1983: q = 1 gives ≈ 0.379.
    assert!((rl - 0.379).abs() < 0.002, "equal-mass RL = {rl}");
    assert!(
        roche_lobe_radius(1.0, 0.2, 1.0) < rl,
        "lighter donor has smaller lobe"
    );
}

#[test]
fn al_transfer_conserves_momentum_and_angular_momentum() {
    let mut b = Bodies::with_capacity(2);
    let d = b.push(10.0, -20.0, 1.0, 2.0, M_SUN_ME, BodyKind::Star);
    let a = b.push(100.0, 50.0, -3.0, 0.5, 0.6 * M_SUN_ME, BodyKind::WhiteDwarf);
    let totals = |b: &Bodies| {
        let mut t = (0.0, 0.0, 0.0);
        for i in 0..b.len() {
            t.0 += b.m[i] * b.vx[i];
            t.1 += b.m[i] * b.vy[i];
            t.2 += b.m[i] * (b.x[i] * b.vy[i] - b.y[i] * b.vx[i]);
        }
        t
    };
    let before = totals(&b);
    let dm = 0.01 * M_SUN_ME;
    assert!(!transfer_mass(&mut b, d, a, dm), "under Chandrasekhar");
    let after = totals(&b);
    let lost = dm * (1.0 - MT_EFF_COMPACT);
    let (px, py) = (after.0 + lost * 1.0, after.1 + lost * 2.0);
    let l = after.2 + lost * (10.0 * 2.0 - (-20.0) * 1.0);
    let scale = before.0.abs().max(before.1.abs()).max(before.2.abs());
    assert!(
        (px - before.0).abs() < 1e-9 * scale,
        "px {px} vs {}",
        before.0
    );
    assert!(
        (py - before.1).abs() < 1e-9 * scale,
        "py {py} vs {}",
        before.1
    );
    assert!((l - before.2).abs() < 1e-9 * scale, "L {l} vs {}", before.2);
}

#[test]
fn al_white_dwarf_crossing_the_limit_explodes() {
    let mut b = Bodies::with_capacity(2);
    let d = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = b.push(100.0, 0.0, 0.0, 0.0, 1.39 * M_SUN_ME, BodyKind::WhiteDwarf);
    assert!(
        transfer_mass(&mut b, d, a, 0.05 * M_SUN_ME),
        "should trigger Type Ia"
    );
    assert!(b.dead[a]);
    assert!(b.events.iter().any(|e| e.text.contains("Type Ia")));
}

#[test]
fn al_white_dwarf_cooling_decreases_luminosity() {
    let m = 0.6 * M_SUN_ME;
    let l1 = wd_luminosity(m, 1.0);
    let l10 = wd_luminosity(m, 10.0);
    let ratio = l1 / l10;
    let want = 10f64.powf(WD_COOL_EXP);
    assert!(
        (ratio / want - 1.0).abs() < 0.01,
        "10 Gyr fainter by {ratio} (want {want})"
    );
    assert!(wd_temperature(m, 0.0) > 50_000.0);
    assert!(wd_temperature(m, 10.0) < 6_000.0);
}

#[test]
fn al_brown_dwarf_cools_and_changes_class() {
    let m = 60.0 * JUPITER_MASS;
    let young = bd_state(m, 0.1);
    let old = bd_state(m, 10.0);
    assert!(young.luminosity > 100.0 * old.luminosity);
    assert!(young.radius_gm > old.radius_gm);
    let cy = bd_class_for_temp(young.temperature);
    let co = bd_class_for_temp(old.temperature);
    assert_ne!(cy, co, "young {cy}, old {co} should differ");
    assert!(young.temperature > old.temperature);
}

#[test]
fn al_default_ages_for_dwarfs_are_set() {
    let mut b = Bodies::with_capacity(2);
    let w = b.push(0.0, 0.0, 0.0, 0.0, 0.6 * M_SUN_ME, BodyKind::WhiteDwarf);
    let d = b.push(
        1e4,
        0.0,
        0.0,
        0.0,
        40.0 * JUPITER_MASS,
        BodyKind::BrownDwarf,
    );
    assert_eq!(b.age_gyr[w], WD_DEFAULT_AGE_GYR);
    assert_eq!(b.age_gyr[d], BD_DEFAULT_AGE_GYR);
    assert_eq!(default_age_gyr(BodyKind::WhiteDwarf), WD_DEFAULT_AGE_GYR);
    assert_eq!(default_age_gyr(BodyKind::BrownDwarf), BD_DEFAULT_AGE_GYR);
    assert_eq!(default_age_gyr(BodyKind::Star), 0.0);
}

#[test]
fn al_compact_moves_evolution_fields_with_their_bodies() {
    let mut b = Bodies::with_capacity(4);
    for k in 0..3 {
        b.push(
            k as f64,
            0.0,
            0.0,
            0.0,
            (k as f64 + 1.0) * M_SUN_ME,
            BodyKind::Star,
        );
        b.age_gyr[k] = k as f64;
        b.rlof[k] = k == 2;
    }
    b.dead[0] = true;
    b.compact();
    assert_eq!(b.len(), 2);
    assert_eq!(b.age_gyr, vec![1.0, 2.0]);
    assert_eq!(b.m_init, vec![2.0 * M_SUN_ME, 3.0 * M_SUN_ME]);
    assert_eq!(b.rlof, vec![false, true]);
}

#[test]
fn al_merging_stars_rejuvenates_the_survivor() {
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(0.5, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.age_gyr[0] = 5.0; // half MS lifetime
    let keep = merge(&mut b, 0, 1);
    assert_eq!(b.m_init[keep], 2.0 * M_SUN_ME, "m_init = sum");
    // The survivor keeps the mass-weighted fraction of MS life.
    let t_ms = star_lifetime_gyr(BodyKind::Star, 2.0 * M_SUN_ME);
    let expected = 0.25 * t_ms; // (0.5*1 + 0.0*1) / 2 = 0.25 fraction used
    assert!(
        (b.age_gyr[keep] - expected).abs() < 1e-6,
        "age {} vs expected {}",
        b.age_gyr[keep],
        expected
    );
}

#[test]
fn al_class_name_reports_red_giant_after_aging() {
    let mut b = sun_system(&[1.0]);
    let t_ms = star_lifetime_gyr(BodyKind::Star, M_SUN_ME);
    set_star_age(&mut b, 0, t_ms * 1.12);
    assert_eq!(class_name(&b, 0), "red giant");
}

#[test]
fn al_giant_luminosity_reaches_nearby_planets() {
    let mut b = sun_system(&[1.0]);
    let (x, y) = (b.x[1], b.y[1]);
    let f0 = stellar_flux(&b, x, y, 1);
    let t_ms = star_lifetime_gyr(BodyKind::Star, M_SUN_ME);
    set_star_age(&mut b, 0, t_ms * 1.12);
    assert_eq!(phase_of(&b, 0), Some(StellarPhase::RedGiant));
    let f1 = stellar_flux(&b, x, y, 1);
    assert!(f1 > 50.0 * f0, "flux {f0} → {f1}");
}

#[test]
fn al_set_star_age_clamps_to_total_life() {
    let mut b = sun_system(&[]);
    let life = star_total_life_gyr(M_SUN_ME);
    set_star_age(&mut b, 0, life * 2.0);
    // Setter clamps to end * 0.999999, so still a star, just barely.
    assert!(b.age_gyr[0] < life);
    assert!(b.age_gyr[0] > life * 0.999_990);
    assert_eq!(b.kind[0], BodyKind::Star);
}

#[test]
fn al_age_advances_only_when_rate_is_positive() {
    let mut b = sun_system(&[]);
    // Rate 0: age frozen.
    advance(&mut b, DT);
    assert_eq!(b.age_gyr[0], 0.0);
    // Rate 1 Gyr per tu: after one advance, age = 1 * DT Gyr.
    set_evolution_rate(&mut b, 1.0);
    advance(&mut b, DT);
    assert!((b.age_gyr[0] - DT).abs() < 1e-12);
    // Negative rate (invalid) → 0, no advance.
    set_evolution_rate(&mut b, -5.0);
    advance(&mut b, DT);
    assert!(
        (b.age_gyr[0] - DT).abs() < 1e-12,
        "negative rate should freeze"
    );
}

// ---- Prompt 1: random ages, remnant spawns, adiabatic vs SN mass loss ----

#[test]
fn al_wd_formation_expands_orbits() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a0 = 50.0;
    let v = (G * M_SUN_ME / a0).sqrt();
    b.push(a0, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    compute_accel(&mut b);
    let o0 = orbit_info(&b, 1, 0);
    become_remnant(&mut b, 0);
    assert_eq!(b.kind[0], BodyKind::WhiteDwarf);
    let o1 = orbit_info(&b, 1, 0);
    // m_old = 1 M_sun, m_new ≈ 0.5 M_sun; a should roughly double.
    let expected = o0.semi_major * (M_SUN_ME / b.m[0]);
    assert!(
        (o1.semi_major / expected - 1.0).abs() < 0.05,
        "a {} vs expected {}",
        o1.semi_major,
        expected
    );
    // The orbit stays bound and keeps its shape.
    assert!(o1.bound);
    assert!((o1.ecc - o0.ecc).abs() < 0.05, "e {} -> {}", o0.ecc, o1.ecc);
}

#[test]
fn al_wd_formation_moves_satellites_with_their_parent() {
    // A planet with a moon around a star that becomes a white dwarf: the
    // planet's orbit expands, the moon keeps its orbit about the planet.
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a0 = 50.0;
    let v = (G * M_SUN_ME / a0).sqrt();
    let p = b.push(a0, 0.0, 0.0, v, 300.0, BodyKind::Planet);
    let d = 0.2;
    let vm = (G * 300.0 / d).sqrt();
    let moon = b.push(a0 + d, 0.0, 0.0, v + vm, 0.01, BodyKind::Planet);
    compute_accel(&mut b);
    let before = orbit_info(&b, moon, p);
    become_remnant(&mut b, 0);
    let after = orbit_info(&b, moon, p);
    assert!(
        (after.semi_major / before.semi_major - 1.0).abs() < 1e-6,
        "moon a {} -> {}",
        before.semi_major,
        after.semi_major
    );
}

#[test]
fn al_wd_formation_leaves_other_systems_alone() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    // A bound neighbour 200 AU away, as on the ring of systems.
    b.push(30_000.0, 0.0, 0.0, 5.0, M_SUN_ME, BodyKind::Star);
    let (x, y, vx, vy) = (b.x[1], b.y[1], b.vx[1], b.vy[1]);
    become_remnant(&mut b, 0);
    assert_eq!((b.x[1], b.y[1], b.vx[1], b.vy[1]), (x, y, vx, vy));
}

#[test]
fn al_wd_formation_expands_close_companion() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let d = 1000.0;
    let v = (G * 2.0 * M_SUN_ME / d).sqrt();
    b.push(d, 0.0, 0.0, v, 0.3 * M_SUN_ME, BodyKind::Star);
    become_remnant(&mut b, 0);
    let o = orbit_info(&b, 1, 0);
    assert!(o.bound);
    assert!(o.semi_major > 1.5 * d, "a {}", o.semi_major);
}

#[test]
fn al_sn_unbinds_wide_planet() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 12.0 * M_SUN_ME, BodyKind::Star);
    let a0 = 50.0;
    let v = (G * 12.0 * M_SUN_ME / a0).sqrt();
    b.push(a0, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    compute_accel(&mut b);
    b.sn_kicks = false;
    become_remnant(&mut b, 0);
    let o = orbit_info(&b, 1, 0);
    // v_circ(12) > v_esc(1.6): planet becomes unbound.
    assert!(!o.bound, "wide planet should be unbound by SN");
}

#[test]
fn al_random_age_produces_some_remnants() {
    let mut n_rem = 0;
    for seed in 0..200u32 {
        let mut b = Bodies::with_capacity(10);
        spawn_system(&mut b, 10, seed);
        if !matches!(b.kind[0], BodyKind::Star | BodyKind::BrownDwarf) {
            n_rem += 1;
        }
    }
    assert!(
        n_rem >= 5,
        "expected at least 5 remnant primaries, got {}",
        n_rem
    );
}

#[test]
fn al_random_ages_are_consistent() {
    for seed in 0..300u32 {
        let mut b = Bodies::with_capacity(12);
        spawn_system(&mut b, 12, seed);
        for i in 0..b.len() {
            assert!(b.age_gyr[i].is_finite() && b.age_gyr[i] >= 0.0);
            if b.kind[i] == BodyKind::Star {
                // A star is never older than its life, and its radius and
                // mass match its evolutionary state.
                assert!(b.age_gyr[i] <= star_total_life_gyr(b.m_init[i]) + 1e-9);
                assert!(b.m[i] <= b.m_init[i] + 1e-9);
                assert!((b.r[i] - b.radius_now(i)).abs() <= 1e-12 * b.r[i].max(1.0));
            }
        }
    }
}

#[test]
fn al_random_systems_keep_ages_on_the_ring() {
    let mut b = Bodies::with_capacity(0);
    spawn_random_systems(&mut b, 10, 7, 4);
    let mut aged = 0;
    for i in 0..b.len() {
        if b.kind[i] == BodyKind::Star && b.age_gyr[i] > 0.0 {
            aged += 1;
        }
        assert!((b.r[i] - b.radius_now(i)).abs() <= 1e-12 * b.r[i].max(1.0));
    }
    // Four systems of random ages: at least one aged star or a remnant host.
    let remnants = (0..b.len())
        .filter(|&i| {
            matches!(
                b.kind[i],
                BodyKind::WhiteDwarf | BodyKind::NeutronStar | BodyKind::BlackHole
            )
        })
        .count();
    assert!(aged + remnants > 0);
}

#[test]
fn solar_preset_loads_at_present_age() {
    let preset = solar_system_preset();
    assert_eq!(preset.name, "Solar System");
    assert_eq!(preset.planets.len(), 13);
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
fn smoke_evolution_runs_without_panic() {
    for seed in 0..40u32 {
        let mut b = Bodies::with_capacity(0);
        spawn_random_systems(&mut b, 30, seed, 1);
        set_evolution_rate(&mut b, 1.0);
        for _ in 0..600 {
            advance(&mut b, DT);
        }
        assert!((0..b.len()).all(|i| b.x[i].is_finite() && b.vx[i].is_finite()));
    }
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    set_evolution_rate(&mut b, 100.0);
    for _ in 0..3000 {
        advance(&mut b, DT);
    }
    assert_eq!(b.kind[0], BodyKind::WhiteDwarf);
}

// ---- Prompt 2: belts, bulk density, mean-anomaly sampling ----

#[test]
fn belt_spec_returns_belt_for_most_systems() {
    let mut with_belt = 0;
    for seed in 0..300u32 {
        let mut b = Bodies::with_capacity(50);
        spawn_system(&mut b, 50, seed);
        if (0..b.len()).any(|i| b.origin[i].is_belt()) {
            with_belt += 1;
        }
    }
    // Belt occurrence is ~85 % for MS primaries, ~30 % for remnants, and
    // ~15 % of primaries are remnants. Expected ~77 %.
    assert!(with_belt > 180, "belt systems: {}/300", with_belt);
}

#[test]
fn kirkwood_gaps_at_jupiter_resonances() {
    let gaps = kirkwood_gaps(5.2026);
    assert!(gaps.iter().any(|g| (g.a_au - 2.50).abs() < 0.05)); // 3:1
    assert!(gaps.iter().any(|g| (g.a_au - 2.82).abs() < 0.05)); // 5:2
    assert!(gaps.iter().any(|g| (g.a_au - 2.96).abs() < 0.05)); // 7:3
    assert!(gaps.iter().any(|g| (g.a_au - 3.28).abs() < 0.05)); // 2:1
}

#[test]
fn belt_bodies_have_physical_bulk_density() {
    for seed in 0..40u32 {
        let mut b = Bodies::with_capacity(50);
        spawn_system(&mut b, 50, seed);
        for i in 0..b.len() {
            if b.origin[i].is_belt() {
                assert!(
                    b.bulk_density[i] > 0.1 && b.bulk_density[i] < 10.0,
                    "seed {} body {} density {}",
                    seed,
                    i,
                    b.bulk_density[i]
                );
            }
        }
    }
}

#[test]
fn belt_bodies_are_never_moons() {
    for seed in 0..30u32 {
        let mut b = Bodies::with_capacity(50);
        spawn_system(&mut b, 50, seed);
        b.refresh_parent_cache();
        for i in 0..b.len() {
            if b.origin[i].is_belt() {
                assert!(
                    find_parent(&b, i).is_none(),
                    "seed {} body {} classified as moon",
                    seed,
                    i
                );
                assert_ne!(body_class(&b, i), "moon");
            }
        }
    }
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
fn trojans_sit_sixty_degrees_from_jupiter() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let jup = (0..b.len())
        .find(|&i| (b.m[i] - 317.80).abs() < 1e-2)
        .expect("Jupiter");
    let ang = |i: usize| (b.y[i] - b.y[0]).atan2(b.x[i] - b.x[0]);
    let aj = ang(jup);
    let mut n = 0;
    for i in 0..b.len() {
        if b.origin[i] == BodyOrigin::Trojan {
            let mut d = (ang(i) - aj).abs();
            if d > std::f64::consts::PI {
                d = std::f64::consts::TAU - d;
            }
            assert!(
                (d - std::f64::consts::FRAC_PI_3).abs() < 0.25,
                "Trojan {} is {:.2} rad from Jupiter",
                i,
                d
            );
            n += 1;
        }
    }
    assert!(n > 0);
}

#[test]
fn belt_mass_scatter_is_log_normal() {
    let mut rng = XorShift::new(42);
    let masses: Vec<f64> = (0..100)
        .map(|_| belt_mass_earth(1.0, 0.0, 1.0, &mut rng))
        .collect();
    let min = masses.iter().copied().fold(f64::INFINITY, f64::min);
    let max = masses.iter().copied().fold(0.0f64, f64::max);
    assert!(max / min > 20.0, "spread {}x", max / min);
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
fn ices_are_lighter_than_rocks() {
    let r_rock = radius_from_density(1.0, 5.0);
    let r_ice = radius_from_density(1.0, 1.0);
    assert!(r_ice > r_rock, "rock {} ice {}", r_rock, r_ice);
}

#[test]
fn random_systems_never_exceed_body_count() {
    // `body_count` is a ceiling: sparse systems may under-fill it, but no
    // system may exceed it.
    for seed in 0..200u32 {
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

#[test]
fn belt_dwarf_planets_use_dwarf_kind() {
    for seed in 0..200u32 {
        let mut b = Bodies::with_capacity(100);
        spawn_system(&mut b, 100, seed);
        for i in 0..b.len() {
            if b.origin[i].is_belt() {
                assert_eq!(
                    b.kind[i] == BodyKind::DwarfPlanet,
                    b.m[i] >= BELT_DWARF_M_MIN
                );
            }
        }
    }
}

#[test]
fn preset_library_loads_all() {
    // The count and the full slug list are checked by
    // `preset_library_has_all_sixteen`; this test just proves the
    // library loads and contains the six originals.
    let presets = load_presets();
    assert!(presets.len() >= 6, "library shrunk: {}", presets.len());
    let slugs: Vec<&str> = presets.iter().map(|(s, _)| s.as_str()).collect();
    for s in [
        "solar_system",
        "trappist_1",
        "hot_jupiter",
        "vega",
        "hr_8799",
        "hd_40307",
    ] {
        assert!(slugs.contains(&s), "missing preset {}", s);
    }
}

#[test]
fn every_preset_spawns_and_runs() {
    for (slug, preset) in load_presets() {
        let mut b = Bodies::with_capacity(0);
        spawn_from_preset(&mut b, preset);
        assert!(!b.is_empty(), "preset {} spawned empty", slug);
        assert!(b.m[0] > 0.0);
        let n0 = b.len();
        for _ in 0..100 {
            advance(&mut b, DT);
        }
        for i in 0..b.len() {
            assert!(
                b.x[i].is_finite() && b.y[i].is_finite(),
                "preset {} body {} NaN",
                slug,
                i
            );
            assert!(b.m[i] > 0.0, "preset {} body {} dead", slug, i);
        }
        println!("{slug}: {n0} bodies, {} after 100 steps", b.len());
    }
}

#[test]
fn preset_by_slug_is_idempotent() {
    let p1 = load_preset_by_slug("solar_system").expect("solar_system");
    let p2 = load_preset_by_slug("solar_system").expect("solar_system");
    assert_eq!(p1.name, p2.name);
    assert_eq!(p1.planets.len(), p2.planets.len());
    assert!(load_preset_by_slug("nope").is_none());
}

#[test]
fn vega_has_no_planets_but_has_a_belt() {
    let p = load_preset_by_slug("vega").expect("vega");
    assert_eq!(p.planets.len(), 0);
    assert!(p.belts.is_some());
}

#[test]
fn trappist_1_has_seven_planets() {
    let p = load_preset_by_slug("trappist_1").expect("trappist_1");
    assert_eq!(p.planets.len(), 7);
    // The preset ships a small outer belt; assert it stays under 1e-3 M⊕.
    if let Some(b) = &p.belts {
        assert!(b.total_mass_earth < 1e-3, "TRAPPIST-1 belt too heavy");
    }
}

#[test]
fn hot_jupiter_has_outer_belt() {
    let p = load_preset_by_slug("hot_jupiter").expect("hot_jupiter");
    let b = p.belts.as_ref().expect("51 Peg belt");
    assert!(b.outer.is_some(), "51 Peg has no outer belt");
    // The inner belt exists but is a minor component (mass_frac ≤ 0.05).
    if let Some(inner) = &b.inner {
        assert!(
            inner.mass_frac <= 0.05,
            "inner mass_frac {}",
            inner.mass_frac
        );
    }
}

// ===========================================================================
// al2 — fixes for CRITICAL.txt
// ===========================================================================

/// The giant-phase envelope is lost over the whole post-MS phase, so
/// `evolve_stars` must call `adiabatic_expand` on every mass decrease.
/// The old code only expanded once, in `become_remnant`, after the mass had
/// already dropped to `m_rem`.
#[test]
fn evolve_sun_to_wd_expands_planet_orbit() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    // Start well outside the Sun's eventual giant radius (peak ≈ 139 Gm);
    // otherwise the planet is swallowed mid-giant and there is nothing
    // left to measure. With a0 = 500 Gm, a_peak ≈ 994 Gm and the planet
    // stays bound and outside the envelope.
    let a0 = 500.0;
    let v = (G * M_SUN_ME / a0).sqrt();
    b.push(a0, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    compute_accel(&mut b);
    rebase_energy(&mut b);
    set_evolution_rate(&mut b, 4.0);
    let o0 = orbit_info(&b, 1, 0);
    for _ in 0..5000 {
        advance(&mut b, DT);
        if b.kind[0] == BodyKind::WhiteDwarf {
            break;
        }
    }
    assert_eq!(b.kind[0], BodyKind::WhiteDwarf, "sun should become WD");
    assert_eq!(b.len(), 2, "planet must not have been swallowed");
    let o1 = orbit_info(&b, 1, 0);
    let expected = M_SUN_ME / b.m[0];
    let ratio = o1.semi_major / o0.semi_major;
    assert!(
        (ratio / expected - 1.0).abs() < 0.15,
        "a ratio {:.3} vs expected {:.3} (m_old {:.0} m_new {:.0})",
        ratio,
        expected,
        M_SUN_ME,
        b.m[0]
    );
    assert!(o1.bound);
}

/// Two brown dwarfs merging to a mass in `[BD_M_MAX, STAR_M_MIN)` must not
/// ignite (they're below the H-burning limit), and must conserve mass.
#[test]
fn merge_never_creates_mass_across_bd_star_gap() {
    // Just above BD_M_MAX (25 440 M⊕), below STAR_M_MIN (26 636 M⊕):
    // midpoint ≈ 26 037; use 13 018 per body.
    let m_each = 13_018.0;
    assert!(2.0 * m_each > BD_M_MAX);
    assert!(2.0 * m_each < STAR_M_MIN);
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, m_each, BodyKind::BrownDwarf);
    b.push(0.001, 0.0, 0.0, 0.0, m_each, BodyKind::BrownDwarf);
    let m0: f64 = b.m.iter().sum();
    collision_pass(&mut b, DT, None);
    assert_eq!(b.len(), 1);
    assert_eq!(b.kind[0], BodyKind::BrownDwarf, "still below STAR_M_MIN");
    assert!(
        (b.m[0] - m0).abs() < 1e-9 * m0,
        "mass created: {} -> {}",
        m0,
        b.m[0]
    );
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
fn s_rng_differs_across_seeds() {
    let mut a = Bodies::with_capacity(0);
    spawn_random_systems(&mut a, 100, 42, 1);
    let r1 = a.rng.next_u32();
    let mut b = Bodies::with_capacity(0);
    spawn_random_systems(&mut b, 100, 43, 1);
    let r2 = b.rng.next_u32();
    assert_ne!(r1, r2);
}

#[test]
fn s_seed_base62_roundtrip() {
    for seed in [
        0u32,
        1,
        61,
        62,
        63,
        1000,
        u32::MAX / 2,
        u32::MAX - 1,
        u32::MAX,
    ] {
        let s = seed_to_base62(seed);
        assert_eq!(s.len(), 6);
        assert_eq!(seed_from_base62(&s), Some(seed));
    }
    assert_eq!(seed_from_base62(""), None);
    assert_eq!(seed_from_base62("0000001"), None); // too long
    assert_eq!(seed_from_base62("!@#$%^"), None); // bad chars
    assert_eq!(seed_from_base62("zzzzzz"), None); // above u32::MAX
}

#[test]
fn s_preset_dwarf_planets_are_dwarfs() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    // The 5 dwarfs, by mass.
    for m in [0.000159, 0.00218, 0.00066, 0.00052, 0.00278] {
        let i = (0..b.len())
            .find(|&i| (b.m[i] - m).abs() < 1e-6 && b.origin[i] == BodyOrigin::Spawned)
            .unwrap_or_else(|| panic!("no body of mass {m}"));
        assert_eq!(b.kind[i], BodyKind::DwarfPlanet, "mass {}", m);
    }
}

#[test]
fn s_preset_phases_are_not_all_periapsis() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let mut n_at_peri = 0;
    for i in 1..9 {
        let o = orbit_info(&b, i, 0);
        if (o.dist / o.peri - 1.0).abs() < 1e-3 {
            n_at_peri += 1;
        }
    }
    assert!(n_at_peri < 2, "all planets at periapsis: {}", n_at_peri);
}

#[test]
fn s_remnant_fraction_is_realistic() {
    let (mut wd, mut ns, mut bh, mut n) = (0, 0, 0, 0);
    for seed in 0..2000u32 {
        let mut b = Bodies::with_capacity(3);
        spawn_system(&mut b, 3, seed);
        n += 1;
        match b.kind[0] {
            BodyKind::WhiteDwarf => wd += 1,
            BodyKind::NeutronStar => ns += 1,
            BodyKind::BlackHole => bh += 1,
            _ => {}
        }
    }
    let bhbns = (ns + bh) as f64 / n as f64;
    assert!(bhbns < 0.10, "NS+BH fraction {}", bhbns);
    assert!((wd as f64 / n as f64) < 0.35, "WD fraction too high");
}

#[test]
fn s_moons_spawn_around_giants() {
    let mut total_moons = 0;
    for seed in 0..100u32 {
        let mut b = Bodies::with_capacity(80);
        spawn_system(&mut b, 80, seed);
        b.refresh_parent_cache();
        for i in 0..b.len() {
            if b.origin[i] != BodyOrigin::Spawned || b.kind[i] != BodyKind::Planet {
                continue;
            }
            if let Some(p) = find_parent(&b, i) {
                if b.kind[p] == BodyKind::Planet && b.m[p] >= 50.0 {
                    total_moons += 1;
                }
            }
        }
    }
    assert!(total_moons > 10, "expected moons, got {}", total_moons);
}

#[test]
fn s_body_count_with_moons_never_exceeds_ceiling() {
    for seed in 0..100u32 {
        let mut b = Bodies::with_capacity(100);
        spawn_system(&mut b, 100, seed);
        assert!(b.len() <= 100, "seed {}: {} bodies", seed, b.len());
    }
}

#[test]
fn s_warm_jupiters_exist() {
    let mut n = 0;
    for seed in 0..500u32 {
        let mut b = Bodies::with_capacity(50);
        spawn_system(&mut b, 50, seed);
        for i in 0..b.len() {
            if b.origin[i] != BodyOrigin::Spawned || b.m[i] < 50.0 || b.m[i] > 318.0 {
                continue;
            }
            let a_au = ((b.x[i] - b.x[0]).powi(2) + (b.y[i] - b.y[0]).powi(2)).sqrt() / au_in_gm();
            if a_au > 0.1 && a_au < 4.0 {
                n += 1;
                break;
            }
        }
    }
    assert!(n > 10, "warm Jupiters found: {}", n);
}

// ===========================================================================
// m — multi-system ring layout
// ===========================================================================

fn range_com(b: &Bodies, (s, e): (usize, usize)) -> (f64, f64) {
    let m: f64 = (s..e).map(|i| b.m[i]).sum();
    (
        (s..e).map(|i| b.m[i] * b.x[i]).sum::<f64>() / m,
        (s..e).map(|i| b.m[i] * b.y[i]).sum::<f64>() / m,
    )
}

#[test]
fn m1_ring_accommodates_wide_binary() {
    assert!((ring_radius_for_extent(3, 0.0) - solar_ring_radius_gm(3)).abs() < 1e-9);
    assert!((ring_radius_for_extent(2, 44_880.0) - 3.0 * 44_880.0).abs() < 1e-6);
    assert_eq!(ring_radius_for_extent(1, 1e9), 0.0);
    for seed in 0..500u32 {
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
        .filter(|&i| matches!(b.kind[i], BodyKind::Star | BodyKind::BrownDwarf))
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
    for seed in 0..20u32 {
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

// ===========================================================================
// dust — statistical dust field
// ===========================================================================

#[test]
fn dust_field_initialized_for_solar() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    assert_eq!(b.dust_fields.len(), 1);
    assert!(
        b.dust_fields[0].extent_gm > 4000.0,
        "extent {}",
        b.dust_fields[0].extent_gm
    );
    assert!(!b.dust_fields[0].grains.is_empty());
}

#[test]
fn dust_field_initialized_for_multi_system() {
    let mut b = Bodies::with_capacity(0);
    let _ = spawn_random_systems(&mut b, 30, 42, 3);
    assert_eq!(b.dust_fields.len(), 3);
}

#[test]
fn dust_field_survives_compact() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let primary = b.dust_fields[0].primary;
    b.dead[primary] = true;
    b.compact();
    assert!(
        b.dust_fields.is_empty(),
        "field with dead primary should be dropped"
    );
}

#[test]
fn dust_field_primary_is_remapped_by_compact() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    // Remove the last body: the field must stay, with its primary index kept.
    let last = b.len() - 1;
    b.dead[last] = true;
    let primary = b.dust_fields[0].primary;
    b.compact();
    assert_eq!(b.dust_fields.len(), 1);
    assert_eq!(b.dust_fields[0].primary, primary);
}

#[test]
fn dust_field_survives_advance() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    for _ in 0..100 {
        advance(&mut b, DT);
    }
    let f = &b.dust_fields[0];
    assert!(f
        .grains
        .iter()
        .all(|g| g.x.is_finite() && g.y.is_finite() && g.mass > 0.0));
}

// ===========================================================================
// Spawning realism and belt occurrence
// ===========================================================================

/// A solar-mass, 4.6-Gyr, solar-metallicity host with the main-sequence
/// architecture of `spawn_architecture` (no binary, so the compact-chain
/// statistics are isolated).
fn solar_mass_system(seed: u32, n: usize) -> Bodies {
    let mut b = Bodies::with_capacity(n);
    b.rng = XorShift::new(seed ^ 0xA5A5_5A5A);
    b.metallicity = 0.0;
    let mut rng = XorShift::new(seed);
    let i = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.age_gyr[i] = 4.6;
    b.set_mass(i, M_SUN_ME);
    spawn_architecture(&mut b, i, n - 1, &mut rng);
    b.refresh_parent_cache();
    b
}

/// (compact planets, giants) among the spawned planets of `b`. Moons are
/// excluded. Compact: <= 30 Me. Giant: > 50 Me.
fn count_compact_and_giants(b: &Bodies) -> (usize, usize) {
    let (mut c, mut g) = (0, 0);
    for i in 1..b.len() {
        if b.origin[i] != BodyOrigin::Spawned || b.kind[i] != BodyKind::Planet || is_moon(b, i) {
            continue;
        }
        if b.m[i] > 50.0 {
            g += 1;
        } else if b.m[i] <= 30.0 {
            c += 1;
        }
    }
    (c, g)
}

#[test]
fn al_spawned_ages_within_universe_age() {
    for seed in 0..500u32 {
        let mut b = Bodies::with_capacity(30);
        spawn_system(&mut b, 30, seed);
        for i in 0..b.len() {
            if b.kind[i] == BodyKind::Star {
                assert!(
                    b.age_gyr[i] <= AGE_UNIVERSE_GYR,
                    "seed {} star {} age {}",
                    seed,
                    i,
                    b.age_gyr[i]
                );
            }
        }
    }
}

#[test]
fn spawn_compact_chain_is_bimodal() {
    let (mut zero, mut one, mut chain) = (0u32, 0u32, 0u32);
    for seed in 0..1000u32 {
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
    for seed in 0..1000u32 {
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
    for seed in 0..300u32 {
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
    for seed in 0..500u32 {
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
    for seed in 0..200u32 {
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

/// Semi-major axis (Gm) of body `i` about body 0, from vis-viva.
fn sma_about_primary(b: &Bodies, i: usize) -> f64 {
    let (dx, dy) = (b.x[i] - b.x[0], b.y[i] - b.y[0]);
    let (dvx, dvy) = (b.vx[i] - b.vx[0], b.vy[i] - b.vy[0]);
    let r = dx.hypot(dy);
    let mu = G * (b.m[0] + b.m[i]);
    1.0 / (2.0 / r - (dvx * dvx + dvy * dvy) / mu)
}

#[test]
fn spawn_lone_giants_are_eccentric() {
    let mut es = Vec::new();
    for seed in 0..600u32 {
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
    for seed in 0..800u32 {
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
    for seed in 0..1000u32 {
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
    for seed in 0..4000u32 {
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
    for seed in 0..3000u32 {
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
fn o_moon_orbit_reference_is_the_planet() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let a = 5.0 * au_in_gm();
    let v_jup = (G * M_SUN_ME / a).sqrt();
    b.push(a, 0.0, 0.0, v_jup, 318.0, BodyKind::Planet);
    let v_moon = v_jup + (G * 318.0 / 0.1).sqrt();
    b.push(a + 0.1, 0.0, 0.0, v_moon, 0.01, BodyKind::Planet);
    b.refresh_parent_cache();
    assert_eq!(find_parent(&b, 2), Some(1));
    let hosts = host_list(&b);
    let r = orbit_reference(&b, &hosts, 2, (0.0, 0.0, 0.0, 0.0, 0.0));
    assert!((r.0 - b.x[1]).abs() < 1e-12);
    assert!((r.1 - b.y[1]).abs() < 1e-12);
    assert!((r.4 - b.m[1]).abs() < 1e-12);
}

#[test]
fn belt_body_distance_matches_primary() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    b.refresh_parent_cache();
    let prim = primary_index(&b).unwrap();
    let belt = (0..b.len()).find(|&i| b.origin[i].is_belt()).unwrap();
    assert!(find_parent(&b, belt).is_none());
    let o = orbit_info(&b, belt, prim);
    assert!(o.dist > 0.0);
    assert!(o.period.is_finite());
}

#[test]
fn preset_names_are_attached_to_bodies() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let earth = (0..b.len())
        .find(|&i| (b.m[i] - 1.0).abs() < 1e-9)
        .expect("Earth");
    assert_eq!(b.name[earth].as_deref(), Some("Earth"));
    assert_eq!(b.name[0].as_deref(), Some("Sun"));
    let belt = (0..b.len())
        .find(|&i| b.origin[i].is_belt())
        .expect("belt body");
    assert_eq!(b.name[belt], None);
}

#[test]
fn named_body_tag_uses_name() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let i = b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.name[i] = Some("Earth".to_string());
    let t = tag(&b, i);
    assert!(t.starts_with("Earth"), "tag was {:?}", t);
    assert!(t.contains("M⊕"), "tag was {:?}", t);
    let t0 = tag(&b, 0);
    assert!(t0.starts_with("#1 "), "tag was {:?}", t0);
}

#[test]
fn compact_moves_names_with_bodies() {
    let mut b = Bodies::with_capacity(3);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(2.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.name[1] = Some("A".to_string());
    b.name[2] = Some("B".to_string());
    b.dead[1] = true;
    b.compact();
    assert_eq!(b.len(), 2);
    assert_eq!(b.name[1].as_deref(), Some("B"));
}

#[test]
fn merge_keeps_survivor_name() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(0.001, 0.0, 0.0, 0.0, 0.01, BodyKind::Planet);
    b.name[0] = Some("Earth".to_string());
    b.name[1] = Some("Moon".to_string());
    merge(&mut b, 0, 1);
    b.compact();
    assert_eq!(b.len(), 1);
    assert_eq!(b.name[0].as_deref(), Some("Earth"));
}

#[test]
fn event_log_uses_names() {
    let mut b = Bodies::with_capacity(2);
    b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.push(0.001, 0.0, 0.0, 0.0, 0.01, BodyKind::Planet);
    b.name[0] = Some("Earth".to_string());
    b.name[1] = Some("Moon".to_string());
    collision_pass(&mut b, DT, None);
    let txt = &b.events.back().unwrap().text;
    assert!(txt.contains("Earth"), "event {:?}", txt);
    assert!(txt.contains("Moon"), "event {:?}", txt);
}

#[test]
fn s_spawn_moon_creates_a_bound_moon() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    b.refresh_parent_cache();
    let earth = (0..b.len())
        .find(|&i| b.name[i].as_deref() == Some("Earth"))
        .unwrap_or_else(|| (0..b.len()).find(|&i| (b.m[i] - 1.0).abs() < 1e-9).unwrap());
    let n0 = b.len();
    let mut rng = XorShift::new(7);
    let m = spawn_moon(&mut b, earth, &mut rng).expect("moon should fit");
    assert_eq!(b.len(), n0 + 1);
    b.refresh_parent_cache();
    assert_eq!(find_parent(&b, m), Some(earth));
}

#[test]
fn s_add_ring_creates_small_particles() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let jup = (0..b.len())
        .find(|&i| (b.m[i] - 317.80).abs() < 1e-2)
        .expect("Jupiter");
    let n0 = b.len();
    let mut rng = XorShift::new(11);
    let parts = add_ring(&mut b, jup, 25, &mut rng);
    assert_eq!(parts.len(), 25);
    assert_eq!(b.len(), n0 + 25);
    let total: f64 = parts.iter().map(|&k| b.m[k]).sum();
    assert!(total < 1e-3 * b.m[jup]);
}

#[test]
fn s_spawn_moon_rejects_star() {
    let mut b = Bodies::with_capacity(1);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let mut rng = XorShift::new(1);
    assert!(spawn_moon(&mut b, 0, &mut rng).is_none());
}

#[test]
fn solar_preset_moons_spawn_with_parents() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    b.refresh_parent_cache();
    let moon_count = (1..b.len())
        .filter(|&i| {
            matches!(
                find_parent(&b, i),
                Some(p) if matches!(b.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet)
            )
        })
        .count();
    assert_eq!(
        moon_count, 8,
        "expected 8 Solar-System moons, got {}",
        moon_count
    );
}

#[test]
fn solar_moon_periods_match_real() {
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    b.refresh_parent_cache();
    let expected = [
        (0.0123_f64, 27.32), // Moon
        (0.015, 1.769),      // Io
        (0.008, 3.551),      // Europa
        (0.025, 7.155),      // Ganymede
        (0.018, 16.689),     // Callisto
        (0.0225, 15.945),    // Titan
        (0.00359, 5.877),    // Triton
        (0.000254, 6.387),   // Charon
    ];
    for &(m_target, real) in &expected {
        // Tolerance 1e-5 (not 1e-4): Ceres m=0.000159 is within 1e-4 of
        // Charon m=0.000254, so 1e-4 would pick Ceres (which orbits the
        // Sun) and fail the "parent is a planet" check below.
        let i = (1..b.len())
            .find(|&i| b.origin[i] == BodyOrigin::Spawned && (b.m[i] - m_target).abs() < 1e-5)
            .unwrap_or_else(|| panic!("no spawned body of mass {} M⊕", m_target));
        let p =
            find_parent(&b, i).unwrap_or_else(|| panic!("moon of mass {} has no parent", m_target));
        assert!(matches!(
            b.kind[p],
            BodyKind::Planet | BodyKind::DwarfPlanet
        ));
        let o = orbit_info(&b, i, p);
        assert!(o.bound);
        let p_days = o.period * time_unit_s() / 86_400.0;
        assert!(
            (p_days / real - 1.0).abs() < 0.05,
            "moon of mass {}: {:.3} d vs {:.3} d",
            m_target,
            p_days,
            real
        );
    }
}

#[test]
fn legacy_presets_without_moons_still_load() {
    for (slug, preset) in load_presets() {
        if slug != "solar_system" {
            for p in &preset.planets {
                assert!(
                    p.moons.is_empty(),
                    "{}/{} unexpectedly has moons",
                    slug,
                    p.name
                );
            }
        }
    }
}

#[test]
fn moon_schema_round_trips() {
    let json = r#"{
        "name": "Fixture",
        "star": { "mass_earth": 332946.0, "age_gyr": 4.6, "metallicity": 0.0 },
        "planets": [
            { "name": "p", "a_au": 1.0, "e": 0.0, "m_earth": 1.0,
              "moons": [
                  { "name": "m1", "a_au": 0.00257, "e": 0.0549, "m_earth": 0.0123,
                    "density_gcc": 3.34 }
              ] }
        ]
    }"#;
    let preset: SystemPreset = serde_json::from_str(json).expect("parse");
    assert_eq!(preset.planets[0].moons.len(), 1);
    assert_eq!(preset.planets[0].moons[0].name, "m1");
}

#[test]
fn preset_library_has_all_sixteen() {
    let presets = load_presets();
    assert_eq!(presets.len(), 16, "library length");
    let slugs: Vec<&str> = presets.iter().map(|(s, _)| s.as_str()).collect();
    for s in [
        "solar_system",
        "trappist_1",
        "hot_jupiter",
        "vega",
        "hr_8799",
        "hd_40307",
        "gamma_cephei",
        "kepler_16",
        "algol",
        "luhman_16",
        "psr_b1257_plus_12",
        "wd_1145_plus_017",
        "wasp_12b",
        "hd_80606",
        "cygnus_x1",
        "kepler_90",
    ] {
        assert!(slugs.contains(&s), "missing preset {s}");
    }
}

#[test]
fn gamma_cephei_is_s_type() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("gamma_cephei").unwrap());
    assert_eq!(b.kind[0], BodyKind::Star);
    assert_eq!(b.kind[1], BodyKind::Star);
    let planet = (0..b.len())
        .find(|&i| b.name[i].as_deref() == Some("γ Cep b"))
        .unwrap();
    assert_eq!(
        find_parent(&b, planet),
        Some(0),
        "S-type planet must orbit the primary"
    );
}

#[test]
fn kepler_16_is_p_type() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("kepler_16").unwrap());
    let planet = (0..b.len())
        .find(|&i| b.name[i].as_deref() == Some("Kepler-16 b"))
        .unwrap();
    let bary_x = (b.m[0] * b.x[0] + b.m[1] * b.x[1]) / (b.m[0] + b.m[1]);
    let bary_y = (b.m[0] * b.y[0] + b.m[1] * b.y[1]) / (b.m[0] + b.m[1]);
    let sep = (b.x[1] - b.x[0]).hypot(b.y[1] - b.y[0]);
    let d = (b.x[planet] - bary_x).hypot(b.y[planet] - bary_y);
    assert!(d > 3.0 * sep, "P-type: d {} vs 3×sep {}", d, 3.0 * sep);
}

#[test]
fn algol_is_a_triple() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("algol").unwrap());
    let n_stars = (0..b.len())
        .filter(|&i| b.kind[i] == BodyKind::Star)
        .count();
    assert_eq!(n_stars, 3, "Algol is a triple");
}

#[test]
fn psr_b1257_host_is_a_neutron_star() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("psr_b1257_plus_12").unwrap());
    assert_eq!(b.kind[0], BodyKind::NeutronStar);
    let n = (1..b.len())
        .filter(|&i| b.kind[i] == BodyKind::Planet)
        .count();
    assert_eq!(n, 3);
}

#[test]
fn wd_1145_host_is_a_white_dwarf() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("wd_1145_plus_017").unwrap());
    assert_eq!(b.kind[0], BodyKind::WhiteDwarf);
    let n_belt = (0..b.len()).filter(|&i| b.origin[i].is_belt()).count();
    assert!(n_belt > 100);
}

#[test]
fn luhman_16_is_a_bd_binary() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("luhman_16").unwrap());
    assert_eq!(b.kind[0], BodyKind::BrownDwarf);
    assert_eq!(b.kind[1], BodyKind::BrownDwarf);
}

#[test]
fn cygnus_x1_host_is_a_black_hole() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("cygnus_x1").unwrap());
    assert_eq!(b.kind[0], BodyKind::Star);
    assert_eq!(b.kind[1], BodyKind::BlackHole);
}

#[test]
fn hd_80606_b_has_extreme_eccentricity() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("hd_80606").unwrap());
    let o = orbit_info(&b, 1, 0);
    assert!(o.bound);
    assert!(o.ecc > 0.9, "HD 80606 b e = {}", o.ecc);
    let peri_au = o.peri / au_in_gm();
    assert!((peri_au - 0.031).abs() < 0.01);
}

#[test]
fn kepler_90_has_eight_planets() {
    let mut b = Bodies::with_capacity(0);
    spawn_from_preset(&mut b, &load_preset_by_slug("kepler_90").unwrap());
    let n = (1..b.len())
        .filter(|&i| b.kind[i] == BodyKind::Planet && b.origin[i] == BodyOrigin::Spawned)
        .count();
    assert_eq!(n, 8);
}

#[test]
fn legacy_presets_parse_without_companions() {
    for (slug, preset) in load_presets() {
        if matches!(
            slug.as_str(),
            "solar_system" | "trappist_1" | "hot_jupiter" | "vega" | "hr_8799" | "hd_40307"
        ) {
            assert!(preset.companions.is_empty(), "{slug} grew companions");
            assert!(preset.star.kind.is_none(), "{slug} grew a kind");
        }
    }
}
