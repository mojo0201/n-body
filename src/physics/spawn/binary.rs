//! Spawn: primaries, companions, outer stars and binary planets.

use super::*;

/// Push the primary of a random system at the origin and give it an age.
///
/// Ages are drawn uniformly over cosmic time (0 to `AGE_UNIVERSE_GYR`).
/// A star whose sampled age exceeds its total lifetime is instead spawned as
/// its remnant (kind, mass and cooling age), preserving `m_init`. Surviving
/// stars are placed on the main sequence or giant branch at the mass and radius
/// `evolve_stars` would have produced by that age. Ages are capped at
/// `AGE_UNIVERSE_GYR`. Non-stellar primaries get an age drawn from the range
/// appropriate to their kind.
pub(in crate::physics) fn spawn_aged_primary(
    b: &mut Bodies,
    m_init: f64,
    kind: BodyKind,
    rng: &mut XorShift,
) -> usize {
    if kind != BodyKind::Star {
        // Cooling-age ranges reflect the observable populations of each
        // non-stellar primary kind.
        let age = match kind {
            BodyKind::WhiteDwarf => 0.1 + 9.9 * rng.next_f64(),
            BodyKind::BrownDwarf => 0.1 + 4.9 * rng.next_f64(),
            BodyKind::NeutronStar | BodyKind::BlackHole => 1.0 + 9.0 * rng.next_f64(),
            k => default_age_gyr(k),
        };
        let i = b.push(0.0, 0.0, 0.0, 0.0, m_init, kind);
        b.age_gyr[i] = age;
        // Radius (e.g. a cooling brown dwarf) depends on age.
        b.set_mass(i, m_init);
        return i;
    }
    let t_total = star_total_life_gyr(m_init);
    // Sampling range is capped at the age of the universe.
    let t_cap = t_total.min(AGE_UNIVERSE_GYR);
    // With uniform sampling over cosmic time, the fraction of stars drawn as
    // remnants equals the fraction of the universe's age that lies beyond the
    // lifetime. Low-mass stars have t_cap = AGE_UNIVERSE_GYR, so p_remnant = 0.
    let p_remnant = (1.0 - t_cap / AGE_UNIVERSE_GYR).clamp(0.0, 1.0);
    let u = rng.next_f64();
    if u < p_remnant {
        // The star has evolved past its total lifetime: spawn as a remnant.
        let (rkind, m_rem) = remnant_of(m_init);
        // The sampled age is uniform on [t_total, AGE_UNIVERSE_GYR], so the
        // cooling age is uniform on [0, AGE_UNIVERSE_GYR - t_total].
        let cool_window = (AGE_UNIVERSE_GYR - t_total).max(0.1);
        let cool_age = match rkind {
            BodyKind::WhiteDwarf | BodyKind::NeutronStar | BodyKind::BlackHole => {
                rng.next_f64() * cool_window
            }
            _ => 0.0,
        };
        let i = b.push(0.0, 0.0, 0.0, 0.0, m_rem, rkind);
        b.m_init[i] = m_init; // keep progenitor mass for evolution bookkeeping
        b.age_gyr[i] = cool_age;
        i
    } else {
        // Surviving star: uniform in lifetime, never older than the universe.
        let age = rng.next_f64() * t_cap;
        let i = b.push(0.0, 0.0, 0.0, 0.0, m_init, BodyKind::Star);
        b.age_gyr[i] = age;
        // Apply the mass loss and radius growth `evolve_stars` would have
        // produced, so the star starts out self-consistent.
        let m_now = star_mass_at(m_init, age).min(m_init);
        b.set_mass(i, m_now);
        i
    }
}

/// Place a stellar companion on a bound orbit about `prim` (which sits at the
/// origin) and move both stars onto their barycentric orbits. Returns `None`
/// if the primary is too light to have a stellar companion. The companion
/// co-forms with the primary, so it inherits `age_gyr`.
pub(in crate::physics) fn spawn_companion(
    b: &mut Bodies,
    prim: usize,
    rng: &mut XorShift,
    age_gyr: f64,
) -> Option<BinaryInfo> {
    let m1 = b.m[prim];
    // Brown-dwarf primaries get brown-dwarf companions; otherwise the
    // companion floor is the hydrogen-burning limit.
    let is_bd = b.kind[prim] == BodyKind::BrownDwarf;
    let m_floor = if is_bd { BD_M_MIN } else { STAR_M_MIN };
    let q_lo = m_floor / m1;
    if q_lo >= 1.0 {
        return None;
    }
    // Mass-ratio distribution (Moe & Di Stefano 2017; Burgasser 2007):
    // density ~ q^alpha on [q_lo, 1]. Flat for stellar primaries and strongly
    // peaked at q ~ 1 for brown-dwarf pairs. Primaries above 0.5 M☉ also have a
    // "twin" excess at q > 0.95 (10 % here; tune this value).
    let alpha = if is_bd { 4.0 } else { 0.0 };
    let twin = !is_bd && m1 > 0.5 * M_SUN_ME && q_lo < 0.95 && rng.next_f64() < 0.10;
    let q = if twin {
        0.95 + 0.05 * rng.next_f64()
    } else {
        // Inverse CDF of density ~ q^alpha on [q_lo, 1].
        let k = 1.0 + alpha;
        let c = q_lo.powf(k);
        (c + (1.0 - c) * rng.next_f64()).powf(1.0 / k)
    };
    let m2 = q * m1;
    let mt = m1 + m2;
    let au = au_in_gm();
    let tu = time_unit_s();

    // Evolved radii: a giant primary is much larger than its MS size.
    let r1 = b.r[prim].max(physical_radius(BodyKind::Star, m1));
    let r2 = if is_bd {
        r1
    } else {
        star_state(m2, m2, age_gyr)
            .radius_gm
            .max(physical_radius(BodyKind::Star, m2))
    };
    let clear = BIN_PERI_CLEAR * (r1 + r2);

    // Kepler's third law in code units:
    //   a(P) = (G·mt·(P_days·86400/tu/TAU)²)^(1/3).
    // Hoist the constant factor out of the rejection loop.
    let day_unit = 86_400.0 / tu / std::f64::consts::TAU;
    let a_k = G * mt * day_unit * day_unit;
    let a_of_p = |p_days: f64| (a_k * p_days * p_days).cbrt();
    // Compare in gm units to avoid dividing `a` by `au` each iteration.
    let a_max_gm = BIN_A_MAX_AU * au;

    // Rejection-sample a period; if 64 tries fail (e.g. an extreme m1 that
    // forces a small q_lo), fall back to the integrator's minimum period
    // stretched outward to clear the collision limit.
    let (p_days, a, e) = 'search: {
        for _ in 0..64 {
            let p_days = if is_bd {
                // Brown-dwarf binaries: log10(a/AU) ~ N(0.55, 0.45),
                // truncated at 20 AU (Burgasser 2007).
                let a_bd_au = 10f64.powf(sample_normal(rng, 0.55, 0.45));
                if a_bd_au > 20.0 {
                    continue; // truncate by rejection, not by clamping
                }
                orbital_period_days(a_bd_au * au, mt)
            } else {
                10f64.powf(sample_normal(rng, BIN_LOGP_MEAN, BIN_LOGP_SIGMA))
            };
            if p_days < BIN_P_MIN_D {
                continue;
            }
            let a = a_of_p(p_days);
            if a > a_max_gm {
                continue;
            }
            // Tight binaries are tidally circularised; wider ones follow a
            // thermal distribution f(e) = 2e, while brown-dwarf pairs stay
            // nearly circular.
            let e = if p_days < BIN_CIRC_P_D {
                0.0
            } else if is_bd {
                rng.next_f64() * 0.3
            } else {
                // Upper envelope of observed e versus P (Moe & Kratter 2021):
                // e_max = 1 - (P / 2 d)^(-2/3), capped at 0.95.
                let e_max = (1.0 - (p_days / 2.0).powf(-2.0 / 3.0)).clamp(0.0, 0.95);
                rng.next_f64().sqrt() * e_max
            };
            if a * (1.0 - e) < clear {
                continue;
            }
            break 'search (p_days, a, e);
        }
        let a = a_of_p(BIN_P_MIN_D).max(clear);
        (orbital_period_days(a, mt), a, 0.0)
    };

    // Uniform in mean anomaly (time), not in true anomaly.
    let mean_anom = rng.next_f64() * std::f64::consts::TAU;
    let nu = true_anomaly(solve_kepler(mean_anom, e), e);
    let omega = rng.next_f64() * std::f64::consts::TAU;
    let (rx, ry, rvx, rvy) = kepler_state(G * mt, a, e, nu, omega);
    // Split the relative state into barycentric pieces. `prim` was at the
    // origin, so this simultaneously moves it and leaves the companion
    // opposite across the barycentre.
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
/// own widest orbit is `a_in_au`.
///
/// The orbit respects the Mardling & Aarseth (2001) stability limit
///   a_out / a_in > 2.8 · [(1+q)(1+e)/√(1-e)]^(2/5),
/// multiplied by a log-uniform factor in [1, 20] so stars don't pile up right
/// at the stability edge. An extra periapsis constraint keeps highly
/// eccentric orbits from grazing the inner system. `m1` is the primary mass,
/// used only to scale the companion's mass ratio. Returns `None` if the drawn
/// orbit is non-finite or exceeds `OUTER_A_MAX_AU`.
pub(in crate::physics) fn plan_outer_star(
    m1: f64,
    m_in: f64,
    a_in_au: f64,
    rng: &mut XorShift,
) -> Option<OuterStar> {
    // Mass-ratio density ~ q^1.5: wide, likely-captured outer stars share the
    // companion peak-toward-unity seen in field binaries.
    let q_lo = STAR_M_MIN / m1;
    if q_lo >= 1.0 {
        return None;
    }
    // Inverse CDF of density ~ q^1.5 on [q_lo, 1].
    let c = q_lo.powf(2.5);
    let q = (c + (1.0 - c) * rng.next_f64()).powf(1.0 / 2.5);
    let m = q * m1;
    // Thermal eccentricity distribution truncated at 0.8.
    let ecc = rng.next_f64().sqrt() * 0.8;
    let q_out = m / m_in;
    let crit = 2.8 * ((1.0 + q_out) * (1.0 + ecc) / (1.0 - ecc).sqrt()).powf(0.4);
    // The formula alone lets very eccentric orbits dip toward the inner
    // system, so additionally require periapsis > 3·a_in.
    let a_au = (a_in_au * crit * 10f64.powf(1.3 * rng.next_f64())).max(3.0 * a_in_au / (1.0 - ecc));
    if !a_au.is_finite() || a_au > OUTER_A_MAX_AU {
        return None;
    }
    Some(OuterStar { m, a_au, ecc })
}

/// Add the planned outer star on its orbit about the centre of mass of every
/// body already in `b`. The inner system is translated as a rigid unit about
/// the new common COM, so its internal relative orbits are unchanged.
pub(in crate::physics) fn place_outer_star(
    b: &mut Bodies,
    s: &OuterStar,
    age_gyr: f64,
    rng: &mut XorShift,
) {
    let au = au_in_gm();
    let n0 = b.len();
    let m_in: f64 = b.m[..n0].iter().sum();
    let mt = m_in + s.m;
    let mean_anom = rng.next_f64() * std::f64::consts::TAU;
    let nu = true_anomaly(solve_kepler(mean_anom, s.ecc), s.ecc);
    let omega = rng.next_f64() * std::f64::consts::TAU;
    let (rx, ry, rvx, rvy) = kepler_state(G * mt, s.a_au * au, s.ecc, nu, omega);
    let (f_in, f_out) = (s.m / mt, m_in / mt);
    let (dx, dy, dvx, dvy) = (f_in * rx, f_in * ry, f_in * rvx, f_in * rvy);

    // Single fused pass: accumulate the pre-shift COM and translate every
    // inner body by the same barycentric offset. Reading the pre-shift
    // values first keeps the COM bit-identical to a separate loop.
    let (mut cx, mut cy, mut cvx, mut cvy) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..n0 {
        let mi = b.m[i];
        let xi = b.x[i];
        let yi = b.y[i];
        let vxi = b.vx[i];
        let vyi = b.vy[i];
        cx += mi * xi;
        cy += mi * yi;
        cvx += mi * vxi;
        cvy += mi * vyi;
        let nx = xi - dx;
        let ny = yi - dy;
        b.x[i] = nx;
        b.y[i] = ny;
        b.vx[i] = vxi - dvx;
        b.vy[i] = vyi - dvy;
        b._px[i] = nx;
        b._py[i] = ny;
    }
    // The outer star sits at the (shifted) inner-system COM plus its own
    // barycentric offset.
    let star = b.push(
        cx / m_in + f_out * rx,
        cy / m_in + f_out * ry,
        cvx / m_in + f_out * rvx,
        cvy / m_in + f_out * rvy,
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

    // S-type host: usually the primary, occasionally the secondary when the
    // mass ratio favours it.
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

    // P-type probability ramps linearly in log(a): close binaries always get
    // circumbinary planets, wide ones never do.
    let p_prob = if bi.a_au <= BIN_P_TYPE_BELOW_AU {
        1.0
    } else if bi.a_au >= BIN_S_TYPE_ABOVE_AU {
        0.0
    } else {
        1.0 - (bi.a_au / BIN_P_TYPE_BELOW_AU).ln()
            / (BIN_S_TYPE_ABOVE_AU / BIN_P_TYPE_BELOW_AU).ln()
    };
    // S-type needs enough room between the host's inner edge and the
    // Holman-Wiegert limit to fit a compact planetary system.
    let s_ok = a_hi_s.min(a_outer_limit_au) >= 3.0 * inner_s;
    // P-type also needs its stable zone to start inside the outer limit.
    let p_ok = HW_MARGIN_P * hw_critical_p(bi.a_au, bi.ecc, m2 / mt) < a_outer_limit_au;
    let use_p = match (s_ok, p_ok) {
        (true, true) => rng.next_f64() < p_prob,
        (true, false) => false,
        (false, _) => true,
    };

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
        // The binary was placed about the barycentre, which is still at the
        // origin, so the P-type frame uses a zero-offset COM at rest.
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
        // No circumbinary planets are known around binaries with P < ~7 d.
        if bi.period_days >= 7.0 {
            spawn_ms_frame(b, &fr, remaining, rng);
        }
    } else {
        bi.arch = if host == prim {
            BinaryArch::STypePrimary
        } else {
            BinaryArch::STypeSecondary
        };
        bi.a_crit_au = a_crit_s;
        // S-type planets orbit the host's current state; the companion
        // contributes half its light to the outer-HZ estimate only.
        let fr = Frame {
            centre: (b.x[host], b.y[host], b.vx[host], b.vy[host], b.m[host]),
            m_class_sun: b.m[host] / M_SUN_ME,
            lum: (luminosity_of(b, host) + 0.5 * luminosity_of(b, other)).max(1e-6),
            inner_au: inner_s,
            a_lo_au: inner_s,
            a_hi_au: a_hi_s.min(a_outer_limit_au),
            occ_scale: BIN_OCC_S * (bi.a_au / 200.0).clamp(0.1, 1.0),
            m_star: b.m[host],
            age_gyr: b.age_gyr[host],
        };
        spawn_ms_frame(b, &fr, remaining, rng);
    }
}
