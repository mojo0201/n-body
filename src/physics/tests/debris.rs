//! Tests for tidal disruption, rubble coalescence, Roche-limit physics,
//! comet sublimation, escape culling, and belt-body spawning.

use super::super::*;

// ===========================================================================
// e — Tidal break-up timing
// ===========================================================================
#[test]
fn e_tidal_break_up_waits_for_pericentre() {
    // A planet crossing into the Roche limit while inbound must be
    // marked `disrupting` but not dismembered until the outward leg.
    // Verify that break-up occurs, and that it happens deeper in the
    // potential than the entry point.
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
    // The exact radius depends on the sub-step that first sees the outward
    // leg, so only require a clear inward displacement; the point of the test
    // is *when* it breaks (after pericentre), not the exact radius.
    assert!(
        rm < 0.95 * r0,
        "break-up must happen deeper than where the body entered"
    );
    assert!(steps > 1, "must not break up on the entry step");
}

// ===========================================================================
// h — Rubble coalescence
// ===========================================================================
#[test]
fn h_coalesce_counter_accumulates_and_merges() {
    // Two rubble fragments on nearly identical orbits around the same
    // planet must merge after DEBRIS_CAPTURE_STEPS consecutive steps of
    // capture, conserving mass and incrementing the coalesce counters.
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
// i — coalesce_debris edge cases
// ===========================================================================
#[test]
fn coalesce_noop_without_rubble() {
    // No rubble bodies at all: function must exit before touching anything.
    let mut b = Bodies::with_capacity(8);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    b.push(1.0, 0.0, 0.0, 1.0, 1.0, BodyKind::Planet);
    let before = b.len();
    coalesce_debris(&mut b);
    assert_eq!(b.len(), before);
    assert_eq!(b.stats.coalesces, 0);
}

#[test]
fn coalesce_noop_with_one_rubble_clears_counter() {
    // Exactly one rubble fragment in an otherwise-populated system:
    // no pair can form, but the fragment's stale capture streak must
    // still be reset so a fragment that leaves the size window
    // doesn't resume a stale counter when it re-enters.
    let mut b = Bodies::with_capacity(8);
    let _star = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let i = b.push(1.0, 0.0, 0.0, 1.0, 0.001, BodyKind::Planet);
    b.origin[i] = BodyOrigin::ImpactDebris;
    b.coalesce_counter[i] = 7;
    coalesce_debris(&mut b);
    assert_eq!(b.len(), 2);
    assert_eq!(b.coalesce_counter[i], 0);
}

#[test]
fn coalesce_skips_heavy_fragments() {
    // Rubble above DEBRIS_COALESCE_MAX_MASS is excluded from coalescence
    // even when geometrically capable of pairing.
    let mut b = Bodies::with_capacity(8);
    let _star = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    let vc = (G * M_SUN_ME / au).sqrt();
    let heavy = DEBRIS_COALESCE_MAX_MASS * 1.001;
    let i = b.push(au, 0.0, 0.0, vc, heavy, BodyKind::Planet);
    let j = b.push(au, 1e-4, 0.0, vc, heavy, BodyKind::Planet);
    b.origin[i] = BodyOrigin::ImpactDebris;
    b.origin[j] = BodyOrigin::ImpactDebris;
    b.refresh_parent_cache();
    for _ in 0..(DEBRIS_CAPTURE_STEPS + 5) {
        coalesce_debris(&mut b);
    }
    assert_eq!(b.len(), 3, "heavy rubble must not merge");
    assert_eq!(b.stats.coalesces, 0);
}

#[test]
fn coalesce_does_not_merge_before_streak() {
    // Same geometry as the positive test, but stop one step short of
    // DEBRIS_CAPTURE_STEPS: no merge may occur yet.
    let mut b = Bodies::with_capacity(8);
    let _star = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    let vc_star = (G * M_SUN_ME / au).sqrt();
    let p = b.push(au, 0.0, 0.0, vc_star, 1.0, BodyKind::Planet);
    let m = 0.0005;
    let r_orb = 0.2;
    let vc_p = (G * 1.0 / r_orb).sqrt();
    let ang = 0.001_f64;
    let i = b.push(au + r_orb, 0.0, 0.0, vc_star + vc_p, m, BodyKind::Planet);
    let j = b.push(
        au + r_orb * ang.cos(),
        r_orb * ang.sin(),
        -vc_p * ang.sin(),
        vc_star + vc_p * ang.cos(),
        m,
        BodyKind::Planet,
    );
    b.origin[i] = BodyOrigin::ImpactDebris;
    b.origin[j] = BodyOrigin::ImpactDebris;
    b.imm[i] = 0.0;
    b.imm[j] = 0.0;
    assert_eq!(find_parent(&b, i), Some(p));
    assert_eq!(find_parent(&b, j), Some(p));
    b.refresh_parent_cache();

    // `DEBRIS_CAPTURE_STEPS >= 2` is checked at compile time below; the loop
    // uses `STEPS - 1` so the merge has not happened yet.
    for _ in 0..(DEBRIS_CAPTURE_STEPS - 1) {
        coalesce_debris(&mut b);
    }
    assert_eq!(b.len(), 4, "must not merge before DEBRIS_CAPTURE_STEPS");
    assert_eq!(b.stats.coalesces, 0);
}

#[test]
fn coalesce_resets_counter_when_pair_separates() {
    // Two rubble bodies far outside one another's Hill spheres must
    // have any stale capture streak wiped on this step.
    let mut b = Bodies::with_capacity(8);
    let _star = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    let vc = (G * M_SUN_ME / au).sqrt();
    let _p = b.push(au, 0.0, 0.0, vc, 1.0, BodyKind::Planet);
    let m = 0.0005;
    let i = b.push(au + 1.0, 0.0, 0.0, vc, m, BodyKind::Planet);
    let j = b.push(au + 100.0, 0.0, 0.0, vc, m, BodyKind::Planet);
    b.origin[i] = BodyOrigin::ImpactDebris;
    b.origin[j] = BodyOrigin::ImpactDebris;
    b.imm[i] = 0.0;
    b.imm[j] = 0.0;
    b.refresh_parent_cache();
    b.coalesce_counter[i] = 999;
    b.coalesce_counter[j] = 999;
    coalesce_debris(&mut b);
    assert_eq!(b.coalesce_counter[i], 0);
    assert_eq!(b.coalesce_counter[j], 0);
    assert_eq!(b.stats.coalesces, 0);
}

// ===========================================================================
// j — Density, rigidity, Roche limit
// ===========================================================================
#[test]
fn density_gcc_planet_uses_bulk_density() {
    // Planets and dwarf planets report their tabulated bulk density.
    let mut b = Bodies::with_capacity(2);
    let i = b.push(1.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.bulk_density[i] = 5.5;
    assert!((density_gcc(&b, i) - 5.5).abs() < 1e-12);
    b.bulk_density[i] = 1.5;
    assert!((density_gcc(&b, i) - 1.5).abs() < 1e-12);
}

#[test]
fn rigidity_endpoints_and_clamping() {
    // Rigidity interpolates 0..1 between the fluid and rigid density
    // thresholds and saturates at both endpoints.
    assert_eq!(rigidity(ROCHE_RHO_FLUID * 0.5), 0.0);
    assert_eq!(rigidity(ROCHE_RHO_FLUID), 0.0);
    assert_eq!(rigidity(ROCHE_RHO_RIGID), 1.0);
    assert_eq!(rigidity(ROCHE_RHO_RIGID * 2.0), 1.0);
    let mid = 0.5 * (ROCHE_RHO_FLUID + ROCHE_RHO_RIGID);
    assert!((rigidity(mid) - 0.5).abs() < 1e-9);
}

#[test]
fn roche_limit_cube_root_mass_ratio() {
    // The Roche limit scales as (M_primary / m_body)^(1/3) at fixed
    // body radius: an 8× mass ratio must produce a factor-of-2 limit.
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let j1 = b.push(1.0, 0.0, 0.0, 0.0, 100.0, BodyKind::Planet);
    let j2 = b.push(2.0, 0.0, 0.0, 0.0, 800.0, BodyKind::Planet);
    b.bulk_density[j1] = 5.0;
    b.bulk_density[j2] = 5.0;
    b.r[j2] = b.r[j1];
    let rl1 = roche_limit(&b, j1, 0);
    let rl2 = roche_limit(&b, j2, 0);
    assert!((rl1 / rl2 - 2.0).abs() < 1e-9, "roche: {} vs {}", rl1, rl2);
}

// ===========================================================================
// k — tidal_breakup edge cases
// ===========================================================================
#[test]
fn tidal_clears_flags_without_primaries() {
    // No candidate primaries: the function must reset every
    // `disrupting` flag and take no other action.
    let mut b = Bodies::with_capacity(4);
    let i = b.push(0.0, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.disrupting[i] = true;
    tidal_breakup(&mut b);
    assert!(!b.disrupting[i], "flags must be cleared when no primaries");
}

#[test]
fn tidal_skips_rubble_body() {
    // Rubble inside the Roche limit must never be re-disrupted:
    // fragments are already post-break-up material.
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let j = b.push(1.0, 0.0, 0.0, 0.0, 318.0, BodyKind::Planet);
    let r0 = 0.5 * roche_limit(&b, j, 0);
    b.x[j] = r0;
    b.vy[j] = 0.9 * (G * M_SUN_ME / r0).sqrt();
    b.origin[j] = BodyOrigin::TidalDebris;
    b.imm[j] = 0.0;
    tidal_breakup(&mut b);
    assert_eq!(b.stats.tidal, 0);
}

#[test]
fn tidal_skips_light_body() {
    // A body below TIDAL_MIN_MASS is too small to model fragmentation
    // meaningfully and must be skipped even at deep pericentre.
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let j = b.push(1.0, 0.0, 0.0, 0.0, TIDAL_MIN_MASS * 0.5, BodyKind::Planet);
    let r0 = 0.1 * roche_limit(&b, j, 0);
    b.x[j] = r0;
    b.vy[j] = 0.9 * (G * M_SUN_ME / r0).sqrt();
    b.imm[j] = 0.0;
    tidal_breakup(&mut b);
    assert_eq!(b.stats.tidal, 0);
}

#[test]
fn tidal_skips_immune_body() {
    // Bodies still inside their spawn immunity window cannot disrupt.
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let j = b.push(1.0, 0.0, 0.0, 0.0, 318.0, BodyKind::Planet);
    let r0 = 0.5 * roche_limit(&b, j, 0);
    b.x[j] = r0;
    b.vy[j] = 0.9 * (G * M_SUN_ME / r0).sqrt();
    b.imm[j] = b.time + 1e9;
    tidal_breakup(&mut b);
    assert_eq!(b.stats.tidal, 0);
}

#[test]
fn tidal_marks_but_defers_when_inbound() {
    // Inbound leg (rdot < 0): the body must be flagged, but
    // disruption is deferred until pericentre passage.
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let j = b.push(1.0, 0.0, 0.0, 0.0, 318.0, BodyKind::Planet);
    let r0 = 0.5 * roche_limit(&b, j, 0);
    b.x[j] = r0;
    b.vy[j] = 0.9 * (G * M_SUN_ME / r0).sqrt();
    b.vx[j] = -0.5 * (G * M_SUN_ME / r0).sqrt();
    b.imm[j] = 0.0;
    tidal_breakup(&mut b);
    assert_eq!(b.stats.tidal, 0, "must not disrupt while inbound");
    assert!(b.disrupting[j], "but must remain flagged");
}

#[test]
fn tidal_clears_flag_when_moving_out() {
    // Body well outside the Roche limit: any stale `disrupting` flag
    // must be cleared.
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let j = b.push(1.0, 0.0, 0.0, 0.0, 318.0, BodyKind::Planet);
    let r0 = 100.0 * roche_limit(&b, j, 0);
    b.x[j] = r0;
    b.vy[j] = 0.5 * (G * M_SUN_ME / r0).sqrt();
    b.imm[j] = 0.0;
    b.disrupting[j] = true;
    tidal_breakup(&mut b);
    assert!(!b.disrupting[j], "flag must be cleared on exit");
}

// ===========================================================================
// l — disrupt invariants
// ===========================================================================
#[test]
fn disrupt_conserves_mass() {
    // The sum of fragment masses must equal the original body's mass.
    let mut b = Bodies::with_capacity(32);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let j = b.push(1.0, 0.0, 0.0, 0.9, 318.0, BodyKind::Planet);
    b.imm[j] = 0.0;
    let m_before = b.m[j];
    assert!(disrupt(&mut b, j, 0));
    let m_after: f64 = (0..b.len())
        .filter(|&k| k != 0 && !b.dead[k])
        .map(|k| b.m[k])
        .sum();
    assert!(
        (m_after - m_before).abs() / m_before < 1e-10,
        "mass {} -> {}",
        m_before,
        m_after
    );
}

#[test]
fn disrupt_conserves_momentum() {
    // Total linear momentum of the fragment cloud must equal the
    // momentum of the original body.
    let mut b = Bodies::with_capacity(32);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let j = b.push(1.0, 0.0, 0.3, 0.9, 318.0, BodyKind::Planet);
    b.imm[j] = 0.0;
    let px_before = b.m[j] * b.vx[j];
    let py_before = b.m[j] * b.vy[j];
    assert!(disrupt(&mut b, j, 0));
    let mut px_after = 0.0;
    let mut py_after = 0.0;
    for k in 0..b.len() {
        if k == 0 || b.dead[k] {
            continue;
        }
        px_after += b.m[k] * b.vx[k];
        py_after += b.m[k] * b.vy[k];
    }
    assert!((px_after - px_before).abs() <= 1e-10 * px_before.abs().max(1e-30));
    assert!((py_after - py_before).abs() <= 1e-10 * py_before.abs().max(1e-30));
}

#[test]
fn disrupt_conserves_angular_momentum() {
    // Orbital + spin angular momentum is conserved: the deficit in
    // orbital L is redistributed into fragment spin.
    let mut b = Bodies::with_capacity(32);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let j = b.push(1.0, 0.0, 0.3, 0.9, 318.0, BodyKind::Planet);
    b.imm[j] = 0.0;
    let l_before = l_orb(&b, j) + b.spin[j];
    assert!(disrupt(&mut b, j, 0));
    let mut l_after = 0.0;
    for k in 0..b.len() {
        if k == 0 || b.dead[k] {
            continue;
        }
        l_after += l_orb(&b, k) + b.spin[k];
    }
    assert!(
        (l_after - l_before).abs() / l_before.abs().max(1e-30) < 1e-10,
        "L {} -> {}",
        l_before,
        l_after
    );
}

#[test]
fn disrupt_fragments_inherit_class() {
    // Every fragment inherits the parent body's composition class, so
    // an icy planet leaves an icy debris string.
    let mut b = Bodies::with_capacity(32);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let j = b.push(1.0, 0.0, 0.0, 0.9, 318.0, BodyKind::Planet);
    b.class[j] = BodyClass::Icy;
    b.imm[j] = 0.0;
    assert!(disrupt(&mut b, j, 0));
    let mut checked = 0;
    for k in 0..b.len() {
        if k == 0 || b.dead[k] {
            continue;
        }
        assert_eq!(b.class[k], BodyClass::Icy);
        checked += 1;
    }
    assert!(checked >= 3, "expected fragments to inspect");
}

// ===========================================================================
// m — comet_sublimate
// ===========================================================================
#[test]
fn comet_sublimate_no_comets_is_noop() {
    // Fast-path: without any comet bodies, the function must not
    // touch the body list.
    let mut b = Bodies::with_capacity(4);
    b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let before = b.len();
    comet_sublimate(&mut b, 1.0);
    assert_eq!(b.len(), before);
}

#[test]
fn comet_sublimate_requires_icy_class() {
    // Only Icy comets sublimate. Rocky (non-Icy) comets are inert
    // regardless of distance to the host. Skip if the default class
    // happens to already be Icy — the invariant under test is the
    // *non-Icy* case.
    let mut b = Bodies::with_capacity(4);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    let c = b.push(0.1 * au, 0.0, 0.0, 0.0, 1e-5, BodyKind::Planet);
    b.origin[c] = BodyOrigin::Comet;
    b.imm[c] = 0.0;
    if b.class[c] == BodyClass::Icy {
        return;
    }
    let m_before = b.m[c];
    comet_sublimate(&mut b, 1.0);
    assert_eq!(b.m[c], m_before);
}

#[test]
fn comet_sublimate_outside_snow_line_is_inert() {
    // Beyond the snow line, ice stays frozen: no mass loss.
    let mut b = Bodies::with_capacity(4);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    let c = b.push(200.0 * au, 0.0, 0.0, 0.0, 1e-5, BodyKind::Planet);
    b.origin[c] = BodyOrigin::Comet;
    b.class[c] = BodyClass::Icy;
    b.imm[c] = 0.0;
    let m_before = b.m[c];
    comet_sublimate(&mut b, 1.0);
    assert_eq!(b.m[c], m_before);
}

#[test]
fn comet_sublimate_inside_snow_line_loses_mass() {
    // Inside the snow line an active icy comet must lose mass, but
    // not all of it in a single step.
    let mut b = Bodies::with_capacity(4);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    let c = b.push(0.1 * au, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.origin[c] = BodyOrigin::Comet;
    b.class[c] = BodyClass::Icy;
    b.imm[c] = 0.0;
    let m_before = b.m[c];
    comet_sublimate(&mut b, 1.0);
    assert!(b.m[c] < m_before, "{} -> {}", m_before, b.m[c]);
    assert!(b.m[c] > 0.0);
}

#[test]
fn comet_sublimate_immunity_blocks_loss() {
    // A comet still inside its spawn immunity window cannot sublimate.
    let mut b = Bodies::with_capacity(4);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    let c = b.push(0.1 * au, 0.0, 0.0, 0.0, 1.0, BodyKind::Planet);
    b.origin[c] = BodyOrigin::Comet;
    b.class[c] = BodyClass::Icy;
    b.imm[c] = b.time + 1e9;
    let m_before = b.m[c];
    comet_sublimate(&mut b, 1.0);
    assert_eq!(b.m[c], m_before);
}

#[test]
fn comet_sublimate_kills_below_dead_mass() {
    // A comet pushed repeatedly below COMET_DEAD_MASS must eventually
    // be removed from the body list, leaving only the star.
    let mut b = Bodies::with_capacity(4);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    let c = b.push(
        0.1 * au,
        0.0,
        0.0,
        0.0,
        COMET_DEAD_MASS * 1.5,
        BodyKind::Planet,
    );
    b.origin[c] = BodyOrigin::Comet;
    b.class[c] = BodyClass::Icy;
    b.imm[c] = 0.0;
    let mut iterations = 0;
    while b.len() > 1 {
        comet_sublimate(&mut b, 1e6);
        iterations += 1;
        assert!(
            iterations < 10_000,
            "comet never fell below COMET_DEAD_MASS"
        );
    }
    assert_eq!(b.len(), 1, "only the star should remain");
}

// ===========================================================================
// n — cull_escaped
// ===========================================================================
#[test]
fn cull_escaped_empty_system_is_noop() {
    // Empty body list: no work, no panic.
    let mut b = Bodies::with_capacity(0);
    cull_escaped(&mut b);
    assert_eq!(b.len(), 0);
}

#[test]
fn cull_keeps_bound_body_inside_cull_distance() {
    // Circular orbit well inside CULL_DIST: body must survive.
    let mut b = Bodies::with_capacity(4);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let au = au_in_gm();
    let r = 1.0 * au;
    let v = (G * M_SUN_ME / r).sqrt();
    let _p = b.push(r, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    cull_escaped(&mut b);
    assert_eq!(b.len(), 2);
    assert!(!b.dead[1]);
}

#[test]
fn cull_keeps_bound_body_between_cull_and_hard_cull() {
    // Bound body past CULL_DIST but below HARD_CULL_DIST: must survive.
    let mut b = Bodies::with_capacity(4);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let r = 0.5 * (CULL_DIST + HARD_CULL_DIST);
    let v = (G * M_SUN_ME / r).sqrt(); // circular
    let _p = b.push(r, 0.0, 0.0, v, 1.0, BodyKind::Planet);
    cull_escaped(&mut b);
    assert_eq!(b.len(), 2, "bound body must survive");
}

#[test]
fn cull_keeps_inbound_unbound_body() {
    // Unbound but moving inward: not outbound, so not culled.
    let mut b = Bodies::with_capacity(4);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let r = 0.5 * (CULL_DIST + HARD_CULL_DIST);
    let v_esc = (2.0 * G * M_SUN_ME / r).sqrt();
    let _p = b.push(
        r,
        0.0,
        -2.0 * v_esc,
        0.0,
        ROGUE_MIN_MASS * 0.5,
        BodyKind::Planet,
    );
    cull_escaped(&mut b);
    assert_eq!(b.len(), 2, "inbound unbound body must be kept");
}

#[test]
fn cull_kills_body_beyond_hard_cull() {
    // HARD_CULL_DIST is absolute: everything past it is removed
    // regardless of velocity or mass.
    let mut b = Bodies::with_capacity(4);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let r = HARD_CULL_DIST * 1.01;
    let _p = b.push(r, 0.0, 100.0, 0.0, 1.0, BodyKind::Planet);
    cull_escaped(&mut b);
    assert_eq!(b.len(), 1, "past HARD_CULL_DIST must be removed");
}

#[test]
fn cull_promotes_heavy_ejectee_to_rogue() {
    // Outbound unbound ejectee above ROGUE_MIN_MASS: promoted to rogue,
    // not culled, so the user can track it out to HARD_CULL_DIST.
    let mut b = Bodies::with_capacity(4);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let r = 0.5 * (CULL_DIST + HARD_CULL_DIST);
    let v_esc = (2.0 * G * M_SUN_ME / r).sqrt();
    let p = b.push(
        r,
        0.0,
        1.5 * v_esc,
        0.0,
        ROGUE_MIN_MASS * 2.0,
        BodyKind::Planet,
    );
    b.origin[p] = BodyOrigin::ImpactDebris;
    cull_escaped(&mut b);
    assert_eq!(b.len(), 2, "heavy ejectee must be promoted, not culled");
    assert_eq!(b.origin[p], BodyOrigin::RoguePlanet);
}

#[test]
fn cull_kills_light_ejectee() {
    // Outbound unbound ejectee below ROGUE_MIN_MASS: culled, and the
    // escaped counter incremented.
    let mut b = Bodies::with_capacity(4);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let r = 0.5 * (CULL_DIST + HARD_CULL_DIST);
    let v_esc = (2.0 * G * M_SUN_ME / r).sqrt();
    let _p = b.push(
        r,
        0.0,
        1.5 * v_esc,
        0.0,
        ROGUE_MIN_MASS * 0.5,
        BodyKind::Planet,
    );
    let before_escaped = b.stats.escaped;
    cull_escaped(&mut b);
    assert_eq!(b.len(), 1, "light ejectee must be culled");
    assert!(b.stats.escaped > before_escaped);
}

#[test]
fn cull_keeps_rogue_between_cull_and_hard_cull() {
    // A rogue body already beyond CULL_DIST must be tracked all the
    // way to HARD_CULL_DIST.
    let mut b = Bodies::with_capacity(4);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let r = 0.5 * (CULL_DIST + HARD_CULL_DIST);
    let v_esc = (2.0 * G * M_SUN_ME / r).sqrt();
    let p = b.push(
        r,
        0.0,
        1.5 * v_esc,
        0.0,
        ROGUE_MIN_MASS * 2.0,
        BodyKind::Planet,
    );
    b.origin[p] = BodyOrigin::RoguePlanet;
    cull_escaped(&mut b);
    assert_eq!(b.len(), 2, "rogue must persist until HARD_CULL_DIST");
    assert_eq!(b.origin[p], BodyOrigin::RoguePlanet);
}

#[test]
fn cull_kills_iso_at_cull_distance() {
    // Interstellar objects cull at CULL_DIST, not HARD_CULL_DIST —
    // they are effectively unobservable once outbound and far.
    let mut b = Bodies::with_capacity(4);
    let _sun = b.push(0.0, 0.0, 0.0, 0.0, M_SUN_ME, BodyKind::Star);
    let r = CULL_DIST * 1.01;
    let v_esc = (2.0 * G * M_SUN_ME / r).sqrt();
    let p = b.push(
        r,
        0.0,
        1.5 * v_esc,
        0.0,
        ROGUE_MIN_MASS * 2.0,
        BodyKind::Planet,
    );
    b.origin[p] = BodyOrigin::Interstellar;
    cull_escaped(&mut b);
    assert_eq!(b.len(), 1, "ISO culls at CULL_DIST");
}

// ===========================================================================
// o — Belt spawning, bulk density, mean-anomaly sampling
// ===========================================================================
#[test]
fn belt_spec_returns_belt_for_most_systems() {
    // Belts should appear in the majority of generated systems. The
    // probability is ~85 % for main-sequence primaries, ~30 % for
    // remnants, with ~15 % of primaries being remnants — expected
    // overall occurrence around 77 %. Threshold of 180/300 is a
    // comfortable two-sigma-plus floor.
    let mut with_belt = 0;
    for seed in 0..300u128 {
        let mut b = Bodies::with_capacity(50);
        spawn_system(&mut b, 50, seed);
        if (0..b.len()).any(|i| b.origin[i].is_belt()) {
            with_belt += 1;
        }
    }
    assert!(with_belt > 180, "belt systems: {}/300", with_belt);
}

#[test]
fn kirkwood_gaps_at_jupiter_resonances() {
    // Kirkwood gap positions must match the classic Jupiter
    // mean-motion resonances to within 0.05 AU.
    let gaps = kirkwood_gaps(5.2026);
    assert!(gaps.iter().any(|g| (g.a_au - 2.50).abs() < 0.05)); // 3:1
    assert!(gaps.iter().any(|g| (g.a_au - 2.82).abs() < 0.05)); // 5:2
    assert!(gaps.iter().any(|g| (g.a_au - 2.96).abs() < 0.05)); // 7:3
    assert!(gaps.iter().any(|g| (g.a_au - 3.28).abs() < 0.05)); // 2:1
}

#[test]
fn belt_bodies_have_physical_bulk_density() {
    // Every spawned belt body must carry a bulk density in the
    // physical range (rock through ice, with headroom for porosity).
    for seed in 0..40u128 {
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
    // Belt bodies orbit the primary, not another planet — the parent
    // finder must never classify one as a moon.
    for seed in 0..30u128 {
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
fn trojans_sit_sixty_degrees_from_jupiter() {
    // Jupiter Trojans must occupy the L4/L5 points, roughly 60°
    // leading or trailing Jupiter in heliocentric longitude.
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    let jup = (0..b.len())
        .find(|&i| (b.m[i] - 317.80).abs() < 1e-2)
        .expect("Jupiter");
    // Angles are measured in 3D: the Solar preset is tilted, so a heliocentric
    // *longitude* difference in the xy plane is not the L4/L5 separation.
    let dir = |i: usize| {
        let (dx, dy, dz) = (b.x[i] - b.x[0], b.y[i] - b.y[0], b.z[i] - b.z[0]);
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        [dx / r, dy / r, dz / r]
    };
    let uj = dir(jup);
    let mut n = 0;
    for i in 0..b.len() {
        if b.origin[i] == BodyOrigin::Trojan {
            let u = dir(i);
            let cos = (u[0] * uj[0] + u[1] * uj[1] + u[2] * uj[2]).clamp(-1.0, 1.0);
            let d = cos.acos();
            // L4/L5 ± the ±20° libration jitter the spawner adds.
            assert!(
                (d - std::f64::consts::FRAC_PI_3).abs() < 0.4,
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
    // Sampled belt-body masses must span at least an order of
    // magnitude — the log-normal draw should produce a wide spread.
    let mut rng = XorShift::new(42);
    let masses: Vec<f64> = (0..100)
        .map(|_| belt_mass_earth(1.0, 0.0, 1.0, &mut rng))
        .collect();
    let min = masses.iter().copied().fold(f64::INFINITY, f64::min);
    let max = masses.iter().copied().fold(0.0f64, f64::max);
    assert!(max / min > 20.0, "spread {}x", max / min);
}

#[test]
fn ices_are_lighter_than_rocks() {
    // For a fixed mass, an icy body (lower density) is larger.
    let r_rock = radius_from_density(1.0, 5.0);
    let r_ice = radius_from_density(1.0, 1.0);
    assert!(r_ice > r_rock, "rock {} ice {}", r_rock, r_ice);
}

#[test]
fn belt_dwarf_planets_use_dwarf_kind() {
    // A belt body's kind is DwarfPlanet exactly when its mass falls in
    // the dwarf-planet window; outside that band it is a plain Planet.
    for seed in 0..200u128 {
        let mut b = Bodies::with_capacity(100);
        spawn_system(&mut b, 100, seed);
        for i in 0..b.len() {
            if b.origin[i].is_belt() {
                assert_eq!(
                    b.kind[i] == BodyKind::DwarfPlanet,
                    (BELT_DWARF_M_MIN..=DWARF_M_MAX).contains(&b.m[i])
                );
            }
        }
    }
}

#[test]
fn belt_body_distance_matches_primary() {
    // Orbit info for a belt body relative to the system primary must
    // yield a positive distance and finite period.
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
fn belt_classification_uses_thresholds() {
    // classify_small_body must return Planet just below the dwarf
    // window, DwarfPlanet inside it, and Planet again above the
    // upper bound.
    const { assert!(BELT_DWARF_M_MIN < DWARF_M_MAX) };
    assert_eq!(
        classify_small_body(BELT_DWARF_M_MIN * 0.999),
        BodyKind::Planet
    );
    assert_eq!(classify_small_body(BELT_DWARF_M_MIN), BodyKind::DwarfPlanet);
    assert_eq!(classify_small_body(DWARF_M_MAX), BodyKind::DwarfPlanet);
    assert_eq!(classify_small_body(DWARF_M_MAX * 1.001), BodyKind::Planet);
}

#[test]
fn classify_small_body_matches_belt_spawn() {
    // Spawned belt bodies must already carry the kind that
    // classify_small_body would assign to them, both for random seeds
    // and for the Solar-System preset (which exercises the heavy belt
    // and the Trojans).
    let mut checked = 0usize;
    let mut check = |b: &Bodies, ctx: &str| {
        for i in 0..b.len() {
            if b.origin[i].is_belt() {
                checked += 1;
                assert_eq!(
                    b.kind[i],
                    classify_small_body(b.m[i]),
                    "{ctx} body {i} mass {}",
                    b.m[i]
                );
            }
        }
    };
    for seed in 0..100u128 {
        let mut b = Bodies::with_capacity(100);
        spawn_system(&mut b, 100, seed);
        check(&b, &format!("seed {seed}"));
    }
    let mut b = Bodies::with_capacity(0);
    spawn_solar_system(&mut b);
    check(&b, "solar_system");
    assert!(checked > 0, "no belt bodies were checked");
}

// ===========================================================================
// Compile-time invariants
// ===========================================================================

/// The capture streak has to be longer than one step for the "not merged yet"
/// half of `h_coalesce_counter_accumulates_and_merges` to mean anything.
const _: () = assert!(DEBRIS_CAPTURE_STEPS >= 2);
