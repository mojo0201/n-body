//! Spawn: belts, trojans and background populations.

use super::*;

/// Fewest tracer bodies of one belt population (so a thin belt is visible).
pub(in crate::physics) const BELT_MIN_TRACERS: usize = 3;

/// Reference belt mass (M⊕) for tracer scaling: a belt of this mass
/// gets `BELT_TRACER_SCALE` tracers.
pub(in crate::physics) const BELT_REF_MASS: f64 = 0.01;

/// Tracers per `BELT_REF_MASS` of belt mass. Each tracer therefore
/// stands for `BELT_REF_MASS / BELT_TRACER_SCALE` ≈ 2×10⁻⁵ M⊕ of real
/// debris — roughly Ceres's mass.
pub(in crate::physics) const BELT_TRACER_SCALE: f64 = 500.0;

/// Smallest mass a single tracer may carry (M⊕). Below this, the
/// "tracer" is sub-Ceres and not worth simulating.
pub(in crate::physics) const BELT_MIN_TRACER_MASS: f64 = 1e-6;

/// Largest mass a single tracer may carry (M⊕). Prevents a heavy belt
/// from producing body-scale collisions that dominate the dynamics.
pub(in crate::physics) const BELT_MAX_TRACER_MASS: f64 = 0.1;

/// Fraction of `spec.total_mass` that the Hilda block adds on top of the
/// main belt.
pub(in crate::physics) const HILDA_MASS_FRAC: f64 = 0.05;

/// Probability that a system of mass `m_sun` has a leftover-planetesimal
/// belt at all.
///
/// This is the intrinsic occurrence rate, not a detection rate. Belts form
/// during planet formation and persist even when they are too faint for IR
/// excess surveys; the ~2-15 % detection rates in Bryden 2006 / Eiroa 2013
/// are a different quantity. `age_gyr` is accepted for symmetry with the
/// (now removed) detection-rate model; the intrinsic rate is age-independent.
pub(in crate::physics) fn debris_belt_occurrence_fraction(m_sun: f64, age_gyr: f64) -> f64 {
    let _ = age_gyr;
    if m_sun < 0.5 {
        0.80
    } else if m_sun < 2.5 {
        0.85
    } else {
        0.90
    }
}

/// Median debris-disk mass in Earth masses, times a log-normal scatter.
/// Calibrated so a 1 M☉, solar-metallicity, 4.6-Gyr system has ~1.5×10⁻³
/// M⊕ — the real Kuiper + asteroid belt. The age factor uses a
/// 300-Myr collisional-depletion timescale (Wyatt 2008), so a 100-Myr
/// disk is about 4× more massive than a 5-Gyr one.
pub(in crate::physics) fn belt_mass_earth(
    m_sun: f64,
    feh: f64,
    age_gyr: f64,
    rng: &mut XorShift,
) -> f64 {
    let baseline = 0.02 * m_sun * 10f64.powf(1.5 * feh);
    let scatter = 10f64.powf(sample_normal(rng, 0.0, 1.0));
    baseline * scatter * (1.0 + age_gyr / 0.3).powf(-1.0)
}

/// Outer (Kuiper-like) belt mass in Earth masses, times a log-normal scatter.
///
/// Calibrated so a solar analogue at 4.6 Gyr has ~0.03 M⊕, the observed mass
/// of the cold classical Kuiper belt. The scatter is the same 10^N(0,1) as
/// the inner belt. The age factor `(1 + t/3 Gyr)^-0.3` is mild (~1.3x at
/// 4.6 Gyr): a primordial cold belt is barely collisionally depleted, unlike
/// the inner belt in `belt_mass_earth`.
pub(in crate::physics) fn belt_mass_earth_outer(
    m_sun: f64,
    feh: f64,
    age_gyr: f64,
    rng: &mut XorShift,
) -> f64 {
    let baseline = 0.035 * m_sun * 10f64.powf(1.5 * feh);
    let scatter = 10f64.powf(sample_normal(rng, 0.0, 1.0));
    baseline * scatter * (1.0 + age_gyr / 3.0).powf(-0.3)
}

/// Kirkwood gaps: the 3:1, 5:2, 7:3 and 2:1 mean-motion resonances with a
/// giant at `a_giant_au`. A resonance p:q (giant : body periods) lies at
/// a = a_giant * (q/p)^(2/3).
pub(in crate::physics) fn kirkwood_gaps(a_giant_au: f64) -> Vec<Gap> {
    let mu = JUPITER_MASS / M_SUN_ME;
    let w = a_giant_au * mu.powf(2.0 / 7.0) * 0.05;
    let at = |p: f64, q: f64| a_giant_au * (q / p).powf(2.0 / 3.0);
    let mut v = vec![
        Gap {
            a_au: at(3.0, 1.0),
            width_au: w * 1.5,
        },
        Gap {
            a_au: at(5.0, 2.0),
            width_au: w * 0.8,
        },
        Gap {
            a_au: at(7.0, 3.0),
            width_au: w * 0.6,
        },
        Gap {
            a_au: at(2.0, 1.0),
            width_au: w * 2.0,
        },
    ];
    v.sort_by(|a, b| a.a_au.total_cmp(&b.a_au));
    v
}

/// Number of tracers a belt of mass `mass` (M⊕) gets, capped at `budget`.
/// Linear in mass: doubling the mass doubles the count. Two floors apply:
/// a physical one (`BELT_MIN_TRACER_MASS`) and a visibility one
/// (`BELT_MIN_TRACERS`), which never exceeds `budget`.
pub(in crate::physics) fn tracers_for_mass(mass: f64, budget: usize) -> usize {
    if mass <= 0.0 || !mass.is_finite() {
        return 0;
    }
    let by_mass = (BELT_TRACER_SCALE * mass / BELT_REF_MASS).round() as usize;
    let by_floor = (mass / BELT_MIN_TRACER_MASS).floor() as usize;
    by_mass
        .min(by_floor)
        .min(budget)
        .max(BELT_MIN_TRACERS.min(budget))
}

pub(in crate::physics) fn crosses_gap(a_au: f64, gaps: &[Gap]) -> bool {
    gaps.iter().any(|g| (a_au - g.a_au).abs() < g.width_au)
}

/// Draw a semi-major axis in `[lo, hi]` AU that is outside every gap.
/// Returns `None` when 16 draws all fall in gaps.
pub(in crate::physics) fn draw_a_outside_gaps(
    rng: &mut XorShift,
    lo: f64,
    hi: f64,
    gaps: &[Gap],
) -> Option<f64> {
    for _ in 0..16 {
        let a = lo + (hi - lo) * rng.next_f64();
        if !crosses_gap(a, gaps) {
            return Some(a);
        }
    }
    None
}

/// Draw a semi-major axis in `[lo, hi]` AU for an outer (Kuiper-like) belt,
/// outside every gap. The surface density falls as a^-1.5, so the number of
/// bodies per unit a goes as a^-0.5 (flat-in-a would put too much mass far
/// out). Falls back to a flat draw if `hi` is not finite.
pub(in crate::physics) fn draw_a_outer(
    rng: &mut XorShift,
    lo: f64,
    hi: f64,
    gaps: &[Gap],
) -> Option<f64> {
    if !(hi.is_finite() && lo > 0.0 && hi > lo) {
        return draw_a_outside_gaps(rng, lo, hi, gaps);
    }
    let (s_lo, s_hi) = (lo.sqrt(), hi.sqrt());
    for _ in 0..16 {
        let s = s_lo + (s_hi - s_lo) * rng.next_f64();
        let a = s * s;
        if !crosses_gap(a, gaps) {
            return Some(a);
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(in crate::physics) fn spawn_belt_body(
    b: &mut Bodies,
    centre: (f64, f64, f64, f64, f64),
    a_gm: f64,
    e: f64,
    m: f64,
    origin: BodyOrigin,
    snow_line_au: f64,
    rng: &mut XorShift,
) -> usize {
    let e = e.clamp(0.0, 0.9);
    let mean_anom = rng.next_f64() * std::f64::consts::TAU;
    let nu = true_anomaly(solve_kepler(mean_anom, e), e);
    let omega = rng.next_f64() * std::f64::consts::TAU;
    let (cx, cy, cvx, cvy, mc) = centre;
    let (x, y, vx, vy) = kepler_state(G * mc, a_gm, e, nu, omega);
    let kind = classify_small_body(m);
    let i = b.push(cx + x, cy + y, cvx + vx, cvy + vy, m, kind);
    b.origin[i] = origin;
    b.bulk_density[i] = draw_density(rng, (a_gm / au_in_gm()) < snow_line_au);
    b.r[i] = b.radius_now(i);
    i
}

/// Angle (rad) of the giant planet nearest to `a_au` about the centre, if
/// the system has one within 10 %.
pub(in crate::physics) fn giant_angle(
    b: &Bodies,
    centre: (f64, f64, f64, f64, f64),
    a_au: f64,
) -> Option<f64> {
    let au = au_in_gm();
    let mut best: Option<(f64, f64)> = None;
    for i in 0..b.len() {
        if b.dead[i]
            || b.kind[i] != BodyKind::Planet
            || b.origin[i] != BodyOrigin::Spawned
            || b.m[i] <= 50.0
        {
            continue;
        }
        let (dx, dy) = (b.x[i] - centre.0, b.y[i] - centre.1);
        let d = (dx.hypot(dy) / au - a_au).abs();
        if d < 0.1 * a_au && best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, dy.atan2(dx)));
        }
    }
    best.map(|(_, ang)| ang)
}

/// Mass (Earth masses) of the giant planet nearest to `a_au` about the
/// centre, if the system has one within 10 %.
pub(in crate::physics) fn giant_mass_near(
    b: &Bodies,
    centre: (f64, f64, f64, f64, f64),
    a_au: f64,
) -> Option<f64> {
    let au = au_in_gm();
    let mut best: Option<(f64, f64)> = None;
    for i in 0..b.len() {
        if b.dead[i]
            || b.kind[i] != BodyKind::Planet
            || b.origin[i] != BodyOrigin::Spawned
            || b.m[i] <= 50.0
        {
            continue;
        }
        let d = ((b.x[i] - centre.0).hypot(b.y[i] - centre.1) / au - a_au).abs();
        if d < 0.1 * a_au && best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, b.m[i]));
        }
    }
    best.map(|(_, m)| m)
}

/// A Trojan at the L4 (`+`) or L5 (`-`) point of a giant at `a_gm`.
/// `giant_angle` is the angle of that giant, when it is known.
pub(in crate::physics) fn spawn_trojan(
    b: &mut Bodies,
    centre: (f64, f64, f64, f64, f64),
    a_gm: f64,
    giant_angle: f64,
    lead: f64,
    m: f64,
    rng: &mut XorShift,
) -> usize {
    let e = rng.next_f64() * 0.05;
    // A little scatter around the Lagrange point.
    let jitter = (rng.next_f64() - 0.5) * 0.2;
    let omega = giant_angle + lead * std::f64::consts::PI / 3.0 + jitter;
    let (cx, cy, cvx, cvy, mc) = centre;
    let (x, y, vx, vy) = kepler_state(G * mc, a_gm, e, 0.0, omega);
    let kind = classify_small_body(m);
    let i = b.push(cx + x, cy + y, cvx + vx, cvy + vy, m, kind);
    b.origin[i] = BodyOrigin::Trojan;
    b.bulk_density[i] = draw_density(rng, true);
    b.r[i] = b.radius_now(i);
    i
}

/// One part of a belt that will receive bodies.
pub(in crate::physics) enum BeltPart<'a> {
    Inner(&'a InnerBelt),
    Outer(&'a OuterPopulation),
    Trojan(&'a Trojan),
}

/// A small population of scattered background bodies — interstellar
/// interlopers and escaped belt members. Real systems have a handful of
/// these, not hundreds. Spawns 0–8 bodies with a Poisson mean of 2.
pub(in crate::physics) fn spawn_background(
    b: &mut Bodies,
    centre: (f64, f64, f64, f64, f64),
    snow_line_au: f64,
    budget: usize,
    rng: &mut XorShift,
) {
    let n = poisson(rng, 2.0).min(8).min(budget);
    for _ in 0..n {
        let a_au = 5.0 + rng.next_f64() * 51.0;
        let e = sample_rayleigh(rng, 0.30).min(0.9);
        let m = sample_mass_power_law(rng, 1e-6, 1e-4, 1.8);
        spawn_belt_body(
            b,
            centre,
            a_au * au_in_gm(),
            e,
            m,
            BodyOrigin::BackgroundDebris,
            snow_line_au,
            rng,
        );
    }
}

/// Spawn the bodies of `spec` about `centre`, using at most `budget` bodies.
///
/// The bodies are tracers: a thin belt still gets a few, and a massive one
/// is capped by the budget. When the parts want more than the budget, it is
/// shared by the square root of their mass fraction, so a light inner belt
/// stays visible next to a heavy Kuiper belt. `fill_au` is the range for
/// background debris that takes whatever budget is left (`None`: no fill).
pub(in crate::physics) fn spawn_belts(
    b: &mut Bodies,
    centre: (f64, f64, f64, f64, f64),
    spec: &BeltSpec,
    snow_line_au: f64,
    budget: usize,
    rng: &mut XorShift,
) {
    if budget == 0 {
        return;
    }
    let au = au_in_gm();
    let _start = b.len();

    // Hildas: a 3:2 group about a Jupiter-like giant. They take their
    // share of the budget first.
    // Only a Jupiter-like host (300-350 Earth masses) gets Hildas.
    let hilda_giant = spec
        .trojans
        .iter()
        .find(|t| {
            giant_mass_near(b, centre, t.giant_a_au).is_some_and(|m| (300.0..=350.0).contains(&m))
        })
        .map(|t| t.giant_a_au);
    let n_hilda = if hilda_giant.is_some() {
        (budget / 20).clamp(1, 30).min(budget)
    } else {
        0
    };
    let avail = budget - n_hilda;

    // Plan: (part, wanted count, mass weight). The weight is used only when
    // the total demand exceeds the budget.
    let mut plan: Vec<(BeltPart, usize, f64)> = Vec::new();
    let want = |mass: f64| -> usize { tracers_for_mass(mass, avail) };
    if let Some(inner) = &spec.inner {
        plan.push((
            BeltPart::Inner(inner),
            want(spec.total_mass * inner.mass_frac),
            inner.mass_frac,
        ));
    }
    if let Some(outer) = &spec.outer {
        for pop in &outer.populations {
            plan.push((
                BeltPart::Outer(pop),
                want(spec.outer_mass * pop.mass_frac),
                pop.mass_frac,
            ));
        }
    }
    for t in &spec.trojans {
        plan.push((
            BeltPart::Trojan(t),
            want(spec.total_mass * t.mass_frac),
            t.mass_frac,
        ));
    }

    // Never plan more parts than there are bodies: keep the parts that want
    // the most tracers. (Keeping every part would overshoot the budget.)
    if plan.len() > avail {
        plan.sort_by_key(|a| std::cmp::Reverse(a.1));
        plan.truncate(avail);
    }

    let wanted: usize = plan.iter().map(|p| p.1).sum();
    if wanted > avail {
        // Proportional down-scale. Weighting by `mass_frac` (not `sqrt`) is
        // the physically correct share: a population with twice the mass
        // keeps twice the tracers.
        let mut assigned = 0usize;
        for (_, n, _) in plan.iter_mut() {
            *n = (((avail as f64) * (*n as f64) / (wanted as f64)).floor() as usize).max(1);
            assigned += *n;
        }
        // The per-part floor of 1 can overshoot; take the excess back from
        // the largest parts (every part keeps at least one tracer).
        while assigned > avail {
            let Some(big) = plan.iter_mut().filter(|p| p.1 > 1).max_by_key(|p| p.1) else {
                break;
            };
            big.1 -= 1;
            assigned -= 1;
        }
        // Distribute the small remainder (from flooring) to the first
        // populations so the total is exactly `avail`.
        let mut leftover = avail.saturating_sub(assigned);
        for (_, n, _) in plan.iter_mut() {
            if leftover == 0 {
                break;
            }
            *n += 1;
            leftover -= 1;
        }
    }

    for (part, n, _) in &plan {
        match part {
            BeltPart::Inner(inner) => {
                let m_total = spec.total_mass * inner.mass_frac;
                let masses =
                    draw_masses_to_total(rng, *n, inner.m_min, inner.m_max, inner.plaw, m_total);
                for &m in masses.iter() {
                    let Some(a_au) =
                        draw_a_outside_gaps(rng, inner.a_lo_au, inner.a_hi_au, &spec.gaps)
                    else {
                        continue;
                    };
                    let e = sample_rayleigh(rng, inner.e_sigma);
                    spawn_belt_body(
                        b,
                        centre,
                        a_au * au,
                        e,
                        m,
                        BodyOrigin::AsteroidBelt,
                        snow_line_au,
                        rng,
                    );
                }
            }
            BeltPart::Outer(pop) => {
                let m_total = spec.outer_mass * pop.mass_frac;
                let masses = draw_masses_to_total(rng, *n, pop.m_min, pop.m_max, pop.plaw, m_total);
                for &m in masses.iter() {
                    let Some(a_au) = draw_a_outer(rng, pop.a_lo_au, pop.a_hi_au, &spec.gaps) else {
                        continue;
                    };
                    let e = if let Some(q_giant) = pop.q_frac {
                        let q = q_giant * (0.9 + 0.2 * rng.next_f64());
                        (1.0 - q / a_au).clamp(0.05, 0.85)
                    } else {
                        sample_rayleigh(rng, pop.e_sigma)
                    };
                    spawn_belt_body(b, centre, a_au * au, e, m, pop.kind, snow_line_au, rng);
                }
            }
            BeltPart::Trojan(troj) => {
                let ang = giant_angle(b, centre, troj.giant_a_au)
                    .unwrap_or_else(|| rng.next_f64() * std::f64::consts::TAU);
                let m_total = spec.total_mass * troj.mass_frac;
                let masses = draw_masses_to_total(
                    rng,
                    *n,
                    MASS_ASTEROID_MIN,
                    MASS_ASTEROID_MAX,
                    MASS_PLAW_ASTEROID,
                    m_total,
                );
                for (k, &m) in masses.iter().enumerate() {
                    let lead = if (k as f64) < troj.leading_frac * *n as f64 {
                        1.0
                    } else {
                        -1.0
                    };
                    spawn_trojan(b, centre, troj.giant_a_au * au, ang, lead, m, rng);
                }
            }
        }
    }

    if let Some(g_au) = hilda_giant {
        let a_hilda_gm = g_au * au * (2.0f64 / 3.0).powf(2.0 / 3.0);
        let m_total = spec.total_mass * HILDA_MASS_FRAC;
        let masses = draw_masses_to_total(
            rng,
            n_hilda,
            MASS_ASTEROID_MIN,
            MASS_ASTEROID_MAX,
            MASS_PLAW_ASTEROID,
            m_total,
        );
        for &m in masses.iter() {
            let e = sample_rayleigh(rng, 0.15);
            spawn_belt_body(
                b,
                centre,
                a_hilda_gm,
                e,
                m,
                BodyOrigin::Hilda,
                snow_line_au,
                rng,
            );
        }
    }
}
