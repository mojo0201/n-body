//! Spawn: planetary architectures for main-sequence and evolved hosts.

use super::*;

pub(in crate::physics) fn spawn_architecture(
    b: &mut Bodies,
    prim: usize,
    remaining: usize,
    rng: &mut XorShift,
) {
    match b.kind[prim] {
        BodyKind::Star | BodyKind::BrownDwarf => {
            let au = au_in_gm();
            let r_star_au = (b.r[prim] / au).max(1e-5);
            let p0 = compact_period_p0(rng);
            let inner = inner_edge_au(p0, b.m[prim], r_star_au);
            let fr = Frame {
                centre: (b.x[prim], b.y[prim], b.vx[prim], b.vy[prim], b.m[prim]),
                m_class_sun: b.m[prim] / M_SUN_ME,
                lum: luminosity_of(b, prim).max(1e-6),
                inner_au: inner,
                a_lo_au: inner,
                a_hi_au: f64::INFINITY,
                occ_scale: 1.0,
                m_star: b.m[prim],
                age_gyr: b.age_gyr[prim],
            };
            spawn_ms_frame(b, &fr, remaining, rng);
        }
        BodyKind::WhiteDwarf | BodyKind::NeutronStar | BodyKind::BlackHole => {
            spawn_evolved_system(b, prim, remaining, rng);
        }
        _ => {}
    }
}

/// Where and how a main-sequence-style planetary system is built.
pub(in crate::physics) struct Frame {
    /// Centre of the orbits: x, y, vx, vy and the central mass (one star, or
    /// the barycentre of a pair with the total mass).
    pub(in crate::physics) centre: (f64, f64, f64, f64, f64),
    /// Mass in solar masses that selects the planet-count class.
    pub(in crate::physics) m_class_sun: f64,
    /// Total luminosity heating the system, solar units.
    pub(in crate::physics) lum: f64,
    /// Physical inner edge of the compact planets (sets the snow line floor).
    pub(in crate::physics) inner_au: f64,
    /// Nothing is placed inside this semi-major axis (P-type stability limit).
    pub(in crate::physics) a_lo_au: f64,
    /// Nothing is placed outside this semi-major axis (S-type stability limit).
    pub(in crate::physics) a_hi_au: f64,
    /// Multiplier on planet-occurrence rates.
    pub(in crate::physics) occ_scale: f64,
    /// Mass in Earth masses of the star (or the pair) that the belts orbit.
    pub(in crate::physics) m_star: f64,
    /// Age of the host star in Gyr.
    pub(in crate::physics) age_gyr: f64,
}

/// Orbital period of the innermost compact planet: 2-20 days, log-uniform
/// (the Kepler-like inner edge).
pub(in crate::physics) fn compact_period_p0(rng: &mut XorShift) -> f64 {
    10f64.powf(0.3 + rng.next_f64()) * 86_400.0 / time_unit_s()
}

pub(in crate::physics) fn inner_edge_au(p0: f64, mass: f64, r_star_au: f64) -> f64 {
    let a0_gm = (G * mass * (p0 / std::f64::consts::TAU).powi(2)).cbrt();
    (a0_gm / au_in_gm()).max(5.0 * r_star_au).max(0.005)
}

/// Intersection of `[lo, hi]` with the allowed `[a_lo, a_hi]`. If they do not
/// overlap, use the nearest part of the allowed range instead.
#[allow(dead_code)]
pub(in crate::physics) fn fit_range(lo: f64, hi: f64, a_lo: f64, a_hi: f64) -> (f64, f64) {
    let (l, h) = (lo.max(a_lo), hi.min(a_hi));
    if l < h {
        (l, h)
    } else if hi < a_lo {
        (a_lo, (a_lo * 1.6).min(a_hi))
    } else {
        ((a_hi * 0.5).max(a_lo), a_hi)
    }
}

pub(in crate::physics) fn spawn_ms_frame(
    b: &mut Bodies,
    fr: &Frame,
    remaining: usize,
    rng: &mut XorShift,
) {
    let au = au_in_gm();
    let m_sun = fr.m_class_sun;
    let inner_au = fr.inner_au;
    let (a_lo, a_hi) = (fr.a_lo_au, fr.a_hi_au);
    let snow_line_au = (2.7 * fr.lum.sqrt()).max(inner_au * 3.0);
    // Giant occurrence scales as 10^(1.5 [Fe/H]) (Fischer & Valenti 2005).
    // Small planets show no such trend (Buchhave 2012), so it applies to
    // giants only.
    let giant_occ = fr.occ_scale * 10f64.powf(1.5 * b.metallicity);

    let max_planets_here = MAX_PLANETS.min(remaining);
    let compact_mean = (if m_sun < 0.5 {
        2.5
    } else if m_sun < 1.5 {
        1.5
    } else {
        1.0
    }) * fr.occ_scale;
    // Compact chains are "peas in a pod": ~40 % of stars have one, and it
    // holds 2 or more planets (never a lone one).
    let mut n_compact = if rng.next_f64() < P_COMPACT_CHAIN {
        (2 + poisson(rng, (compact_mean - 1.0).max(0.1))).min(max_planets_here)
    } else {
        0
    };
    // Keep a third of the planet budget for giants.
    let compact_cap = (max_planets_here * 2) / 3;
    n_compact = n_compact.min(compact_cap);

    let has_giant = rng.next_f64() < (0.20 * giant_occ).min(0.95);
    let p_second = (0.10 * giant_occ).min(0.5);
    let unstable_giants = has_giant && rng.next_f64() < P_UNSTABLE_GIANTS;
    let n_giants = if unstable_giants {
        2 + (rng.next_f64() < 0.4) as usize + (rng.next_f64() < 0.15) as usize
    } else if has_giant {
        if rng.next_f64() < 1.0 - p_second {
            1
        } else {
            2
        }
    } else {
        0
    }
    .min(max_planets_here.saturating_sub(n_compact));

    // Hot Jupiter (~1 % of FGK hosts, ~0.1 % of M dwarfs). Hot Jupiters are
    // mostly lone (Steffen 2012), so the compact chain is dropped for them.
    // Decided here so the slot is taken out of every later budget.
    let hj_frac = if (0.5..2.0).contains(&m_sun) {
        0.01
    } else {
        0.001
    };
    // Capped at P ~ 7 d (a = 0.07 AU * m^(1/3)): hot Jupiters orbit within
    // about 0.1 AU, and the chain-edge scaling alone reaches 0.5 AU.
    let a_hj = (inner_au * (1.5 + rng.next_f64() * 2.0))
        .min(0.07 * m_sun.cbrt())
        .max(inner_au * 1.2);
    let has_hj = rng.next_f64() < (hj_frac * giant_occ).min(0.5)
        && n_giants < max_planets_here
        && a_hj >= a_lo
        && a_hj <= a_hi;
    if has_hj {
        n_compact = 0;
    }

    // Every planet placed below, for the belt-architecture choice.
    let mut placed: Vec<(usize, f64, f64)> = Vec::new();

    // ---- Compact super-Earths / sub-Neptunes (peas in a pod) ----
    // Radius valley (Fulton 2017): each system is rocky (median ~2 Me) or
    // volatile-rich (median ~8 Me). Intra-system scatter is small
    // (Weiss 2018). The mass cap keeps the chain compact (no giants).
    let rocky_system = rng.next_f64() < 0.55;
    let log_mu = if rocky_system { 0.30 } else { 0.90 };
    let log_sigma_sys = 0.25;
    let log_sigma_intra = 0.10;
    let sys_scale = sample_normal(rng, 0.0, log_sigma_sys);
    let mut a_au = inner_au.max(a_lo);
    let mut placed_c = 0usize;
    // ~30 % of chains sit near 3:2 or 2:1 period commensurabilities, just
    // wide of exact (1-2 %), as in the Kepler period-ratio peaks and in
    // TRAPPIST-1-like systems.
    let resonant_chain = rng.next_f64() < 0.30;
    for _ in 0..n_compact {
        if a_au > a_hi {
            break;
        }
        let z = sample_normal(rng, 0.0, 1.0);
        let m = 10f64
            .powf(log_mu + sys_scale + log_sigma_intra * z)
            .clamp(0.1, 30.0);
        let e = sample_rayleigh(rng, 0.06);
        let k = spawn_planet(
            b,
            fr.centre,
            a_au * au,
            e,
            m,
            BodyKind::Planet,
            snow_line_au,
            rng,
        );
        placed.push((k, m, a_au));
        placed_c += 1;
        a_au *= if resonant_chain {
            let pr = if rng.next_f64() < 0.5 { 1.5 } else { 2.0 };
            (pr * (1.01 + 0.01 * rng.next_f64())).powf(2.0 / 3.0)
        } else {
            1.4 + rng.next_f64() * 0.8 // 1.4-2.2 (Fabrycky 2014, Weiss 2018)
        };
    }

    // ---- Hot Jupiter ----
    if has_hj {
        let m = sample_mass_power_law(rng, 50.0, 300.0, MASS_PLAW_GIANT);
        let e = rng.next_f64() * 0.05;
        let k = spawn_planet(
            b,
            fr.centre,
            a_hj * au,
            e,
            m,
            BodyKind::Planet,
            snow_line_au,
            rng,
        );
        placed.push((k, m, a_hj));
        a_au = a_au.max(a_hj * 1.5);
    }

    // ---- Warm Jupiter (~10% of giant hosts) ----
    // Real systems have giants at 0.1-1 AU in ~10% of cases (WASP-107,
    // WASP-117). Place one inside the snow line with moderate eccentricity.
    let has_warm =
        rng.next_f64() < (0.10 * giant_occ).min(0.5) && n_giants < max_planets_here && !has_hj;
    let mut placed_w = 0usize;
    if has_warm {
        let a_warm = inner_au + (snow_line_au - inner_au) * (0.3 + 0.6 * rng.next_f64());
        if a_warm >= a_lo && a_warm <= a_hi {
            let m = sample_mass_power_law(rng, MASS_GIANT_MIN, MASS_GIANT_MAX, MASS_PLAW_GIANT);
            let e = sample_rayleigh(rng, 0.15);
            let k = spawn_planet(
                b,
                fr.centre,
                a_warm * au,
                e,
                m,
                BodyKind::Planet,
                snow_line_au,
                rng,
            );
            placed.push((k, m, a_warm));
            placed_w += 1;
            a_au = a_au.max(a_warm * 1.2);
        }
    }

    // ---- Giants near the snow line, outside the compact chain ----
    let mut a_au = (snow_line_au * (1.5 + rng.next_f64() * 1.0)).max(a_au);
    let mut placed_g = 0usize;
    for _ in 0..n_giants {
        if a_au > a_hi {
            break;
        }
        let m = sample_mass_power_law(rng, MASS_GIANT_MIN, MASS_GIANT_MAX, MASS_PLAW_GIANT);
        // Observed giants are fairly eccentric (mean ~0.2 for a lone giant,
        // Kipping 2013); multi-giant systems are rounder.
        let sigma_e = if n_giants == 1 { 0.15 } else { 0.08 };
        let e = sample_rayleigh(rng, sigma_e).min(0.7);
        let k = spawn_planet(
            b,
            fr.centre,
            a_au * au,
            e,
            m,
            BodyKind::Planet,
            snow_line_au,
            rng,
        );
        placed.push((k, m, a_au));
        placed_g += 1;
        a_au *= if unstable_giants {
            // Packed to 2-3.5 mutual Hill radii (equal-mass pairs are Hill
            // stable only beyond 2*sqrt(3) = 3.46), so the system scatters.
            let mu_h = (2.0 * m / (3.0 * fr.m_star)).cbrt();
            let h = (2.0 + 1.5 * rng.next_f64()) * mu_h;
            1.0 + h / (1.0 - 0.5 * h)
        } else {
            1.5 + rng.next_f64() * 0.7
        };
    }

    // ---- Moons around giant planets ----
    // Real gas giants host moons. Draw 1-4 per giant with ~70 % probability,
    // mass ratio 1e-4 to 1e-2. Moons use the background-debris budget, so
    // the body count never exceeds the ceiling.
    let mut moons_spawned = 0usize;
    let giants: Vec<(usize, f64, f64)> = placed.iter().copied().filter(|g| g.1 >= 50.0).collect();
    for (giant_idx, _gm, _ga) in giants {
        if rng.next_f64() > 0.7 {
            continue;
        }
        let n_moons = 1 + (rng.next_f64() * 4.0).floor() as usize; // 1..=4
        if moons_spawned + n_moons
            > remaining.saturating_sub(placed_c + placed_g + placed_w + has_hj as usize) / 10
        {
            break; // do not eat the whole budget; skip moons here
        }
        let m_p = b.m[giant_idx];
        let r_giant_gm = b.r[giant_idx];
        let d_star = (b.x[giant_idx] - fr.centre.0).hypot(b.y[giant_idx] - fr.centre.1);
        let r_hill = d_star * (m_p / (3.0 * fr.m_star)).cbrt();
        let a_lo_m = 5.0 * r_giant_gm;
        let a_hi_m = MOON_HILL_FRAC * r_hill;
        if a_lo_m >= a_hi_m {
            continue;
        }
        for k in 0..n_moons {
            let m_ratio = 10f64.powf(-4.0 + rng.next_f64() * 2.0);
            let m_moon = m_p * m_ratio;
            let frac = (k as f64 + 1.0) / (n_moons as f64 + 1.0);
            let a_moon = a_lo_m + (a_hi_m - a_lo_m) * frac;
            let angle = rng.next_f64() * std::f64::consts::TAU;
            let (sa, ca) = angle.sin_cos();
            let v_circ = (G * m_p / a_moon).sqrt();
            let moon_idx = b.push(
                b.x[giant_idx] + a_moon * ca,
                b.y[giant_idx] + a_moon * sa,
                b.vx[giant_idx] - v_circ * sa,
                b.vy[giant_idx] + v_circ * ca,
                m_moon,
                BodyKind::Planet,
            );
            // Moons inherit the giant's class, so they stay Icy.
            b.class[moon_idx] = b.class[giant_idx];
            b.bulk_density[moon_idx] = draw_density(rng, b.class[moon_idx]);
            b.r[moon_idx] = b.radius_now(moon_idx);
            moons_spawned += 1;
        }
    }

    // ---- Belts: archetype from star, age, metallicity and planets ----
    let spec = belt_spec_for(
        b,
        fr.m_star,
        b.metallicity,
        fr.age_gyr,
        &placed,
        snow_line_au,
        a_lo,
        a_hi,
        rng,
    );
    let belt_budget =
        remaining.saturating_sub(placed_c + placed_g + placed_w + has_hj as usize + moons_spawned);
    let len_before = b.len();
    spawn_belts(b, fr.centre, &spec, snow_line_au, belt_budget, rng);
    let belt_budget_used = b.len() - len_before;
    spawn_background(
        b,
        fr.centre,
        snow_line_au,
        belt_budget.saturating_sub(belt_budget_used),
        rng,
    );
}

/// A white dwarf, neutron star or black hole: sometimes a scattered belt of
/// leftover bodies, and background debris for the rest of the budget.
pub(in crate::physics) fn spawn_evolved_system(
    b: &mut Bodies,
    prim: usize,
    remaining: usize,
    rng: &mut XorShift,
) {
    let m_sun = b.m[prim] / M_SUN_ME;
    let spec = belt_spec_for_evolved(m_sun, b.age_gyr[prim], rng);
    let centre = (b.x[prim], b.y[prim], b.vx[prim], b.vy[prim], b.m[prim]);
    // Deliberate simplification: a snow line of 3 AU gives a mix of rocky and
    // icy debris. The geometric snow line of a white dwarf (L ~ 1e-3 Lsun) is
    // ~0.1 AU, which would make every body icy.
    let snow_line_au_for_density = 3.0;
    let len_before = b.len();
    spawn_belts(b, centre, &spec, snow_line_au_for_density, remaining, rng);
    let belt_budget_used = b.len() - len_before;
    spawn_background(
        b,
        centre,
        snow_line_au_for_density,
        remaining.saturating_sub(belt_budget_used),
        rng,
    );
}
