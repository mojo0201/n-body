//! Spawn: planetary architectures for main-sequence and evolved hosts.
//!
//! One entry point (`spawn_architecture`) dispatches to two builders:
//! `spawn_ms_frame` for stars and brown dwarfs, and `spawn_evolved_system`
//! for stellar remnants. Both draw from the same `remaining` body budget and
//! share the `XorShift` stream, so a given seed always yields the same bodies.

use super::*;

/// Build the planetary system around one primary.
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

/// Geometry and physics of a main-sequence system, threaded through
/// `spawn_ms_frame` unchanged. `centre` is either a single star or the
/// barycentre of a pair (with the total mass); `m_star` is what the belts
/// and moons orbit.
pub(in crate::physics) struct Frame {
    /// Orbital centre: x, y, vx, vy and central mass.
    pub(in crate::physics) centre: (f64, f64, f64, f64, f64),
    /// Host mass in solar masses — selects the planet-count class.
    pub(in crate::physics) m_class_sun: f64,
    /// Total luminosity in solar units — sets the snow line.
    pub(in crate::physics) lum: f64,
    /// Physical inner edge (overrides a Keplerian inner edge for compact
    /// planets; also a floor on the snow line).
    pub(in crate::physics) inner_au: f64,
    /// P-type stability limit (nothing inside this).
    pub(in crate::physics) a_lo_au: f64,
    /// S-type stability limit (nothing outside this).
    pub(in crate::physics) a_hi_au: f64,
    /// Multiplier on planet-occurrence rates (0 = suppress all planets).
    pub(in crate::physics) occ_scale: f64,
    /// Mass in Earth masses of the star or pair that belts and moons orbit.
    pub(in crate::physics) m_star: f64,
    /// Host age in Gyr — drives belt archetype and dynamical heating.
    pub(in crate::physics) age_gyr: f64,
}

/// Period of the innermost compact planet: 2–20 d, log-uniform. This is the
/// Kepler-like inner edge — chains of super-Earths start here, not at the
/// stellar surface.
pub(in crate::physics) fn compact_period_p0(rng: &mut XorShift) -> f64 {
    10f64.powf(0.3 + rng.next_f64()) * 86_400.0 / time_unit_s()
}

/// Semi-major axis corresponding to `p0`, floored at 5 stellar radii and at
/// 0.005 AU (so very compact hosts still get a non-degenerate range).
pub(in crate::physics) fn inner_edge_au(p0: f64, mass: f64, r_star_au: f64) -> f64 {
    let a0_gm = (G * mass * (p0 / std::f64::consts::TAU).powi(2)).cbrt();
    (a0_gm / au_in_gm()).max(5.0 * r_star_au).max(0.005)
}

/// Intersect `[lo, hi]` with the allowed `[a_lo, a_hi]`; if they miss,
/// fall back to the nearest part of the allowed range. Currently unused
/// (`#[allow(dead_code)]`) — kept as a utility for callers that need it.
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

    // Snow line from equilibrium temperature; floored at 3× inner edge so
    // hot inner systems still get a physically distinct volatile zone.
    let snow_line_au = (2.7 * fr.lum.sqrt()).max(inner_au * 3.0);

    // Giant occurrence scales as 10^(1.5 [Fe/H]) (Fischer & Valenti 2005).
    // Small planets show no such trend (Buchhave 2012) — apply to giants
    // only, never to the compact chain.
    let giant_occ = fr.occ_scale * 10f64.powf(1.5 * b.metallicity);

    // ---- Planet-count classes ----
    let max_planets_here = MAX_PLANETS.min(remaining);

    // Compact chains ("peas in a pod"): ~40 % of stars have one, and it
    // holds 2+ planets — never a lone one. Count is Poisson about a mean
    // that decreases with host mass (M dwarfs pack more).
    let compact_mean = (if m_sun < 0.5 {
        2.5
    } else if m_sun < 1.5 {
        1.5
    } else {
        1.0
    }) * fr.occ_scale;
    let mut n_compact = if rng.next_f64() < P_COMPACT_CHAIN {
        (2 + poisson(rng, (compact_mean - 1.0).max(0.1))).min(max_planets_here)
    } else {
        0
    };
    // Reserve a third of the slot budget for giants.
    let compact_cap = (max_planets_here * 2) / 3;
    n_compact = n_compact.min(compact_cap);

    // Giants, and the fraction of giant hosts that are "unstable" (packed
    // inside mutual Hill stability so the system will scatter).
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

    // Hot Jupiters: ~1 % of FGK hosts, ~0.1 % of M dwarfs, and mostly lone
    // (Steffen 2012). Decide up front so the slot is removed from every
    // later budget (compact chain, giants, moons).
    //
    // Orbit is capped at P ~ 7 d (a = 0.07 AU · m^(1/3)); without the cap the
    // inner-edge scaling alone would reach ~0.5 AU for some hosts.
    let hj_frac = if (0.5..2.0).contains(&m_sun) {
        0.01
    } else {
        0.001
    };
    let a_hj = (inner_au * (1.5 + rng.next_f64() * 2.0)).min(0.07 * m_sun.cbrt());
    let has_hj = rng.next_f64() < (hj_frac * giant_occ).min(0.5)
        && n_giants < max_planets_here
        && a_hj >= inner_au * 1.2
        && a_hj >= a_lo
        && a_hj <= a_hi;
    if has_hj {
        n_compact = 0;
    }

    // Every planet placed so far, as (index, mass, semi-major axis). Passed
    // to `belt_spec_for` and iterated for moon seeding.
    let mut placed: Vec<(usize, f64, f64)> = Vec::new();

    // ---- Compact super-Earths / sub-Neptunes ----
    // Radius valley (Fulton 2017): each system is one composition class —
    // rocky (median ~2 Me) or volatile-rich (median ~8 Me). Intra-system
    // scatter is small (Weiss 2018). Mass cap at 30 Me keeps chains compact.
    let rocky_system = rng.next_f64() < 0.55;
    let log_mu = if rocky_system { 0.30 } else { 0.90 };
    let log_sigma_sys = 0.25;
    let log_sigma_intra = 0.10;
    let sys_scale = sample_normal(rng, 0.0, log_sigma_sys);
    // Per-system log-mass mean, constant across the chain — hoisted.
    let log_base = log_mu + sys_scale;

    let mut a_au = inner_au.max(a_lo);
    let mut placed_c = 0usize;

    // ~30 % of chains sit near a 3:2 or 2:1 period ratio, just wide of
    // exact (1–2 %) — matches the Kepler period-ratio peaks and
    // TRAPPIST-1-like architectures.
    let resonant_chain = rng.next_f64() < 0.30;

    for _ in 0..n_compact {
        if a_au > a_hi {
            break;
        }
        let z = sample_normal(rng, 0.0, 1.0);
        let m = 10f64.powf(log_base + log_sigma_intra * z).clamp(0.1, 30.0);
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
            // Near but not exactly at a 3:2 or 2:1 commensurability.
            let pr = if rng.next_f64() < 0.5 { 1.5 } else { 2.0 };
            (pr * (1.01 + 0.01 * rng.next_f64())).powf(2.0 / 3.0)
        } else {
            // Non-resonant spacing, 1.4–2.2 in semi-major axis
            // (Fabrycky 2014, Weiss 2018).
            1.4 + rng.next_f64() * 0.8
        };
    }

    // ---- Hot Jupiter ----
    if has_hj {
        let m = sample_mass_power_law(rng, 50.0, 300.0, MASS_PLAW_GIANT);
        let e = rng.next_f64() * 0.05; // tidally circularised
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

    // ---- Warm Jupiter (~10 % of giant hosts, never with a hot Jupiter) ----
    // Real systems (WASP-107, WASP-117) put one giant at 0.1–1 AU with
    // moderate eccentricity — inside the snow line.
    let has_warm = rng.next_f64() < (0.10 * giant_occ).min(0.5)
        && n_compact + n_giants < max_planets_here
        && !has_hj;
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

    // ---- Giants near or beyond the snow line ----
    // Start at 1.5–2.5× the snow line, and no closer than any compact or
    // warm-Jupiter planet already placed.
    let mut a_au = (snow_line_au * (1.5 + rng.next_f64() * 1.0)).max(a_au);
    let mut placed_g = 0usize;
    for _ in 0..n_giants {
        if a_au > a_hi {
            break;
        }
        let m = sample_mass_power_law(rng, MASS_GIANT_MIN, MASS_GIANT_MAX, MASS_PLAW_GIANT);
        // Observed giants are fairly eccentric (mean e ~ 0.2 for a lone
        // giant, Kipping 2013); multi-giant systems are rounder.
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
            // Pack to 2–3.5 mutual Hill radii. Equal-mass pairs are Hill
            // stable only beyond 2·sqrt(3) = 3.46, so this configuration
            // is dynamically unstable and will scatter.
            let mu_h = (2.0 * m / (3.0 * fr.m_star)).cbrt();
            let h = (2.0 + 1.5 * rng.next_f64()) * mu_h;
            1.0 + h / (1.0 - 0.5 * h)
        } else {
            1.5 + rng.next_f64() * 0.7
        };
    }

    // ---- Moons around giant planets ----
    // Gas giants host moons in ~70 % of systems (1–4 moons, mass ratio
    // 1e-4 to 1e-2 of the planet). Moons draw from the background budget,
    // so the body count stays under the ceiling.
    //
    // `base_placed` and `moon_budget_limit` are invariant across the moon
    // loop; hoist them so `base_placed` is also reusable for the belt budget.
    let base_placed = placed_c + placed_g + placed_w + has_hj as usize;
    let moon_budget_limit = remaining.saturating_sub(base_placed) / 10;
    let mut moons_spawned = 0usize;
    // Iterate `placed` directly (skipping non-giants *before* the RNG roll,
    // matching the original filter order so the RNG stream is unchanged).
    for &(giant_idx, gm, _) in placed.iter() {
        if gm < 50.0 {
            continue;
        }
        if rng.next_f64() > 0.7 {
            continue;
        }
        let n_moons = 1 + (rng.next_f64() * 4.0).floor() as usize; // 1..=4
        if moons_spawned + n_moons > moon_budget_limit {
            // Don't eat the whole background budget on moons.
            break;
        }
        let m_p = b.m[giant_idx];
        let r_giant_gm = b.r[giant_idx];
        let d_star = (b.x[giant_idx] - fr.centre.0).hypot(b.y[giant_idx] - fr.centre.1);
        // Hill radius of the giant about the host; place moons between
        // 5 giant radii and a fraction of the Hill radius.
        let r_hill = d_star * (m_p / (3.0 * fr.m_star)).cbrt();
        let a_lo_m = 5.0 * r_giant_gm;
        let a_hi_m = MOON_HILL_FRAC * r_hill;
        if a_lo_m >= a_hi_m {
            continue;
        }
        // Spread moons evenly in semi-major axis within the allowed range.
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
            // Inherit the giant's composition class so moons stay Icy.
            b.class[moon_idx] = b.class[giant_idx];
            b.bulk_density[moon_idx] = draw_density(rng, b.class[moon_idx]);
            b.r[moon_idx] = b.radius_now(moon_idx);
            moons_spawned += 1;
        }
    }

    // ---- Belts ----
    // Archetype chosen from host mass, metallicity, age, and the planets
    // already placed (a giant inside the belt gaps it).
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
    let belt_budget = remaining.saturating_sub(base_placed + moons_spawned);
    let len_before = b.len();
    spawn_belts(b, fr.centre, &spec, snow_line_au, belt_budget, rng);
    let belt_budget_used = b.len() - len_before;

    // ---- Comets ----
    // Reserve up to 4 slots from what belts left over; the rest goes to
    // background debris.
    let avail_after_belts = belt_budget.saturating_sub(belt_budget_used);
    let comet_budget = avail_after_belts.min(4);
    let len_before_comets = b.len();
    spawn_comets(b, fr.centre, snow_line_au, comet_budget, rng);
    let comet_budget_used = b.len() - len_before_comets;

    spawn_background(
        b,
        fr.centre,
        snow_line_au,
        avail_after_belts.saturating_sub(comet_budget_used),
        rng,
    );
}

/// Stellar remnant (white dwarf, neutron star, or black hole): sometimes a
/// scattered belt of leftover bodies, plus background debris for the rest
/// of the budget.
pub(in crate::physics) fn spawn_evolved_system(
    b: &mut Bodies,
    prim: usize,
    remaining: usize,
    rng: &mut XorShift,
) {
    let m_sun = b.m[prim] / M_SUN_ME;
    let spec = belt_spec_for_evolved(m_sun, b.age_gyr[prim], rng);
    let centre = (b.x[prim], b.y[prim], b.vx[prim], b.vy[prim], b.m[prim]);

    // Deliberate simplification: a fixed 3 AU snow line gives a physically
    // mixed debris population. The geometric snow line of a white dwarf
    // (L ~ 1e-3 Lsun) sits at ~0.1 AU, which would make every body Icy —
    // not what is observed around polluted WDs.
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
