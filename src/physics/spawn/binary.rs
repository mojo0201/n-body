//! Spawn: primaries, companions, outer stars and binary planets.

use super::*;

/// Push the primary of a random system at the origin and give it an age.
///
/// A star older than its total life is spawned as its remnant (kind, mass
/// and cooling age), keeping `m_init`. Younger stars are spread uniformly over
/// their life (not over Gyr), so a giant has also lost the mass its age implies.
/// Ages never exceed `AGE_UNIVERSE_GYR`.
/// Non-stellar primaries get a random age in the range that suits their kind.
pub(in crate::physics) fn spawn_aged_primary(b: &mut Bodies, m_init: f64, kind: BodyKind, rng: &mut XorShift) -> usize {
    if kind != BodyKind::Star {
        let age = match kind {
            BodyKind::WhiteDwarf => 0.1 + 9.9 * rng.next_f64(),
            BodyKind::BrownDwarf => 0.1 + 4.9 * rng.next_f64(),
            BodyKind::NeutronStar | BodyKind::BlackHole => 1.0 + 9.0 * rng.next_f64(),
            k => default_age_gyr(k),
        };
        let i = b.push(0.0, 0.0, 0.0, 0.0, m_init, kind);
        b.age_gyr[i] = age;
        // The brown dwarf radius depends on age.
        b.set_mass(i, m_init);
        return i;
    }
    let t_total = star_total_life_gyr(m_init);
    // No star is older than the universe, so the sampling range is capped.
    let t_cap = t_total.min(AGE_UNIVERSE_GYR);
    // With ages uniform over the age of the universe, the chance that a star
    // is older than its total life is 1 - t_total / AGE_UNIVERSE_GYR.
    let p_remnant = (1.0 - t_cap / AGE_UNIVERSE_GYR).clamp(0.0, 0.95);
    let u = rng.next_f64();
    if u < p_remnant {
        // The star has already evolved past its total life.
        let (rkind, m_rem) = remnant_of(m_init);
        let cool_age = match rkind {
            BodyKind::WhiteDwarf => 0.1 + 9.9 * rng.next_f64(), // 0.1-10 Gyr
            BodyKind::NeutronStar => 1.0 + 9.0 * rng.next_f64(), // 1-10 Gyr
            BodyKind::BlackHole => 1.0 + 9.0 * rng.next_f64(),  // 1-10 Gyr
            _ => 0.0,
        };
        let i = b.push(0.0, 0.0, 0.0, 0.0, m_rem, rkind);
        b.m_init[i] = m_init; // preserve initial mass
        b.age_gyr[i] = cool_age;
        i
    } else {
        // Main sequence or giant. Uniform in lifetime, not in Gyr, and never
        // older than the universe.
        let age = rng.next_f64() * t_cap;
        let i = b.push(0.0, 0.0, 0.0, 0.0, m_init, BodyKind::Star);
        b.age_gyr[i] = age;
        // Mass at this age (a giant has shed part of its envelope) and the
        // evolved radius, as `evolve_stars` would have left them.
        let m_now = star_mass_at(m_init, age).min(m_init);
        b.set_mass(i, m_now);
        i
    }
}

/// Place a stellar companion on a bound orbit about `prim` (which sits at the
/// origin) and move both stars onto their barycentric orbits. Returns `None`
/// if the primary is too light to have a stellar companion. The companion
/// co-forms with the primary, so it has the same age (`age_gyr`).
pub(in crate::physics) fn spawn_companion(
    b: &mut Bodies,
    prim: usize,
    rng: &mut XorShift,
    age_gyr: f64,
) -> Option<BinaryInfo> {
    let m1 = b.m[prim];
    // A brown-dwarf primary gets a brown-dwarf companion (mass >= BD_M_MIN).
    let is_bd = b.kind[prim] == BodyKind::BrownDwarf;
    let m_floor = if is_bd { BD_M_MIN } else { STAR_M_MIN };
    let q_lo = m_floor / m1;
    if q_lo >= 1.0 {
        return None;
    }
    // Mass ratio (Duchene & Kraus 2013): rises toward q = 1 for G/K primaries,
    // roughly flat for M dwarfs. Density ~ q^alpha. Brown-dwarf pairs are
    // strongly peaked at q ~ 1 (Burgasser 2007: alpha ~ 4).
    let alpha = if is_bd {
        4.0
    } else if m1 > 0.5 * M_SUN_ME {
        0.5
    } else {
        0.0
    };
    let q = q_lo + (1.0 - q_lo) * rng.next_f64().powf(1.0 / (1.0 + alpha));
    let m2 = q * m1;
    let mt = m1 + m2;
    let au = au_in_gm();
    let tu = time_unit_s();
    // Evolved radii: a giant primary is much larger than its main-sequence size.
    let r1 = b.r[prim].max(physical_radius(BodyKind::Star, m1));
    let r2 = if is_bd {
        r1
    } else {
        star_state(m2, m2, age_gyr)
            .radius_gm
            .max(physical_radius(BodyKind::Star, m2))
    };
    let clear = BIN_PERI_CLEAR * (r1 + r2);
    let a_of_p =
        |p_days: f64| (G * mt * (p_days * 86_400.0 / tu / std::f64::consts::TAU).powi(2)).cbrt();

    // Log-normal period, truncated to the range the integrator handles.
    let mut chosen: Option<(f64, f64, f64)> = None;
    for _ in 0..64 {
        // Brown-dwarf binaries are tight: log10(a/AU) ~ N(0.55, 0.45), so the
        // median is ~3.5 AU and nearly all lie inside 20 AU (Burgasser 2007).
        let p_days = if is_bd {
            let a_bd_au = 10f64.powf(sample_normal(rng, 0.55, 0.45)).min(20.0);
            orbital_period_days(a_bd_au * au, mt)
        } else {
            10f64.powf(sample_normal(rng, BIN_LOGP_MEAN, BIN_LOGP_SIGMA))
        };
        if p_days < BIN_P_MIN_D {
            continue;
        }
        let a = a_of_p(p_days);
        if a / au > BIN_A_MAX_AU {
            continue;
        }
        // Circularised when tight; otherwise thermal, f(e) = 2e.
        let e = if p_days < BIN_CIRC_P_D {
            0.0
        } else if is_bd {
            rng.next_f64() * 0.3
        } else {
            rng.next_f64().sqrt() * 0.95
        };
        if a * (1.0 - e) < clear {
            continue;
        }
        chosen = Some((p_days, a, e));
        break;
    }
    let (p_days, a, e) = chosen.unwrap_or_else(|| {
        let a = a_of_p(BIN_P_MIN_D).max(clear);
        (orbital_period_days(a, mt), a, 0.0)
    });

    // Phase uniform in mean anomaly (time), not true anomaly.
    let mean_anom = rng.next_f64() * std::f64::consts::TAU;
    let nu = true_anomaly(solve_kepler(mean_anom, e), e);
    let omega = rng.next_f64() * std::f64::consts::TAU;
    let (rx, ry, rvx, rvy) = kepler_state(G * mt, a, e, nu, omega);
    let (f1, f2) = (m2 / mt, m1 / mt);
    b.x[prim] = -f1 * rx;
    b.y[prim] = -f1 * ry;
    b._px[prim] = b.x[prim];
    b._py[prim] = b.y[prim];
    b.vx[prim] = -f1 * rvx;
    b.vy[prim] = -f1 * rvy;
    let comp_kind = if is_bd {
        BodyKind::BrownDwarf
    } else {
        BodyKind::Star
    };
    let comp = b.push(f2 * rx, f2 * ry, f2 * rvx, f2 * rvy, m2, comp_kind);
    b.age_gyr[comp] = age_gyr;
    b.set_mass(comp, m2); // radius at this age

    Some(BinaryInfo {
        companion: comp,
        period_days: p_days,
        a_au: a / au,
        ecc: e,
        mass_ratio: q,
        arch: BinaryArch::PType,
        a_crit_au: 0.0,
    })
}

/// A planned outer star of a hierarchical triple or quadruple.
#[derive(Clone, Copy, Debug)]
pub(in crate::physics) struct OuterStar {
    /// Mass in Earth masses.
    pub(in crate::physics) m: f64,
    /// Semi-major axis about the centre of mass of everything inside it, AU.
    pub(in crate::physics) a_au: f64,
    pub(in crate::physics) ecc: f64,
}

/// Plan a star on a wide orbit about an inner system of mass `m_in` whose
/// own widest orbit is `a_in_au`. `m1` is the primary mass, used to scale
/// the mass ratio. The orbit respects the Mardling & Aarseth (2001)
/// stability limit, a_out / a_in > 2.8 [(1+q)(1+e)/sqrt(1-e)]^(2/5), with a
/// log-uniform factor of 1-20 on top. Returns `None` if no such star fits.
pub(in crate::physics) fn plan_outer_star(m1: f64, m_in: f64, a_in_au: f64, rng: &mut XorShift) -> Option<OuterStar> {
    let q_lo = STAR_M_MIN / m1;
    if q_lo >= 1.0 {
        return None;
    }
    let q = q_lo + (1.0 - q_lo) * rng.next_f64().powf(1.0 / 1.5);
    let m = q * m1;
    let ecc = rng.next_f64().sqrt() * 0.8;
    let q_out = m / m_in;
    let crit = 2.8 * ((1.0 + q_out) * (1.0 + ecc) / (1.0 - ecc).sqrt()).powf(0.4);
    // The periapsis must also clear 3x the inner orbit: the formula alone
    // lets very eccentric orbits dip close to the inner system.
    let a_au = (a_in_au * crit * 10f64.powf(1.3 * rng.next_f64())).max(3.0 * a_in_au / (1.0 - ecc));
    if !a_au.is_finite() || a_au > OUTER_A_MAX_AU {
        return None;
    }
    Some(OuterStar { m, a_au, ecc })
}

/// Add the planned star on its orbit about the centre of mass of every body
/// already in `b`. The inner system moves as a unit about the new common
/// centre of mass, so its internal orbits are unchanged.
pub(in crate::physics) fn place_outer_star(b: &mut Bodies, s: &OuterStar, age_gyr: f64, rng: &mut XorShift) {
    let au = au_in_gm();
    let n0 = b.len();
    let m_in: f64 = b.m[..n0].iter().sum();
    let mt = m_in + s.m;
    let mean_anom = rng.next_f64() * std::f64::consts::TAU;
    let nu = true_anomaly(solve_kepler(mean_anom, s.ecc), s.ecc);
    let omega = rng.next_f64() * std::f64::consts::TAU;
    let (rx, ry, rvx, rvy) = kepler_state(G * mt, s.a_au * au, s.ecc, nu, omega);
    let (f_in, f_out) = (s.m / mt, m_in / mt);
    // Centre of mass of the inner system before the shift.
    let (mut cx, mut cy, mut cvx, mut cvy) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..n0 {
        cx += b.m[i] * b.x[i];
        cy += b.m[i] * b.y[i];
        cvx += b.m[i] * b.vx[i];
        cvy += b.m[i] * b.vy[i];
    }
    (cx, cy, cvx, cvy) = (cx / m_in, cy / m_in, cvx / m_in, cvy / m_in);
    for i in 0..n0 {
        b.x[i] -= f_in * rx;
        b.y[i] -= f_in * ry;
        b.vx[i] -= f_in * rvx;
        b.vy[i] -= f_in * rvy;
        b._px[i] = b.x[i];
        b._py[i] = b.y[i];
    }
    let star = b.push(
        cx + f_out * rx,
        cy + f_out * ry,
        cvx + f_out * rvx,
        cvy + f_out * rvy,
        s.m,
        BodyKind::Star,
    );
    b.age_gyr[star] = age_gyr;
    b.set_mass(star, s.m);
}

/// Choose S-type or P-type, then build the planetary system inside the
/// Holman-Wiegert stability limits.
pub(in crate::physics) fn spawn_binary_planets(
    b: &mut Bodies,
    prim: usize,
    bi: &mut BinaryInfo,
    remaining: usize,
    a_outer_limit_au: f64,
    rng: &mut XorShift,
) {
    let au = au_in_gm();
    let comp = bi.companion;
    let (m1, m2) = (b.m[prim], b.m[comp]);
    let mt = m1 + m2;
    let p0 = compact_period_p0(rng);

    // Candidate S-type host: usually the primary.
    let host = if rng.next_f64() < 0.5 * m2 / m1 {
        comp
    } else {
        prim
    };
    let other = if host == prim { comp } else { prim };
    let a_crit_s = hw_critical_s(bi.a_au, bi.ecc, b.m[other] / mt);
    let a_hi_s = HW_MARGIN_S * a_crit_s;
    let r_host_au = (b.r[host] / au).max(1e-5);
    let inner_s = inner_edge_au(p0, b.m[host], r_host_au);

    let p_prob = if bi.a_au <= BIN_P_TYPE_BELOW_AU {
        1.0
    } else if bi.a_au >= BIN_S_TYPE_ABOVE_AU {
        0.0
    } else {
        1.0 - (bi.a_au / BIN_P_TYPE_BELOW_AU).ln()
            / (BIN_S_TYPE_ABOVE_AU / BIN_P_TYPE_BELOW_AU).ln()
    };
    // S-type needs room for the compact inner planets.
    let s_ok = a_hi_s >= 3.0 * inner_s;
    let use_p = !s_ok || rng.next_f64() < p_prob;

    if use_p {
        let a_crit_p = hw_critical_p(bi.a_au, bi.ecc, m2 / mt);
        let lum = luminosity_of(b, prim) + luminosity_of(b, comp);
        let r_au = (b.r[prim].max(b.r[comp]) / au).max(1e-5);
        // Circumbinary planets sit outside the binary's disk cavity: the
        // innermost one has P ~ 50-200 d, not the 2-20 d of a stellar disk.
        let p0_cb = 50.0 * 86_400.0 / time_unit_s() * (1.0 + 3.0 * rng.next_f64());
        let inner = inner_edge_au(p0_cb, mt, r_au);
        bi.arch = BinaryArch::PType;
        bi.a_crit_au = a_crit_p;
        // The stars were placed about the barycentre, which is still the origin.
        let fr = Frame {
            centre: (0.0, 0.0, 0.0, 0.0, mt),
            m_class_sun: m1 / M_SUN_ME,
            lum: lum.max(1e-6),
            inner_au: inner,
            a_lo_au: inner.max(HW_MARGIN_P * a_crit_p),
            a_hi_au: a_outer_limit_au,
            occ_scale: BIN_OCC_P,
            m_star: mt,
            age_gyr: b.age_gyr[if m1 >= m2 { prim } else { comp }],
        };
        spawn_ms_frame(b, &fr, remaining, rng);
    } else {
        bi.arch = if host == prim {
            BinaryArch::STypePrimary
        } else {
            BinaryArch::STypeSecondary
        };
        bi.a_crit_au = a_crit_s;
        let fr = Frame {
            centre: (b.x[host], b.y[host], b.vx[host], b.vy[host], b.m[host]),
            m_class_sun: b.m[host] / M_SUN_ME,
            lum: (luminosity_of(b, host) + 0.5 * luminosity_of(b, other)).max(1e-6),
            inner_au: inner_s,
            a_lo_au: inner_s,
            a_hi_au: a_hi_s.min(a_outer_limit_au),
            occ_scale: BIN_OCC_S,
            m_star: b.m[host],
            age_gyr: b.age_gyr[host],
        };
        spawn_ms_frame(b, &fr, remaining, rng);
    }
}
