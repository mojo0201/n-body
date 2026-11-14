//! Spawn belts, trojans and background populations.
//!
//! Belt members are "tracers": a handful of bodies each carrying a real
//! debris mass and standing in for a much larger swarm. Counts and masses
//! are calibrated against the observed asteroid and Kuiper belts, and
//! every tracer is capped in mass so it can never dominate collisions.

use super::*;

/// Fewest tracers of one belt population, so even a thin belt is visible.
pub(in crate::physics) const BELT_MIN_TRACERS: usize = 3;

/// Reference belt mass (M⊕) that receives `BELT_TRACER_SCALE` tracers.
pub(in crate::physics) const BELT_REF_MASS: f64 = 0.01;

/// Tracers per `BELT_REF_MASS` of belt mass. Each tracer therefore stands
/// for `BELT_REF_MASS / BELT_TRACER_SCALE` ≈ 2×10⁻⁵ M⊕ of real debris —
/// roughly the mass of Ceres.
pub(in crate::physics) const BELT_TRACER_SCALE: f64 = 500.0;

/// Smallest mass (M⊕) a single tracer may carry. Below this the tracer
/// is sub-Ceres and not worth simulating.
pub(in crate::physics) const BELT_MIN_TRACER_MASS: f64 = 1e-4;

/// Largest mass (M⊕) a single tracer may carry. Prevents a heavy belt
/// from producing body-scale collisions that dominate the dynamics.
pub(in crate::physics) const BELT_MAX_TRACER_MASS: f64 = 0.1;

/// Hilda mass as a fraction of `spec.total_mass`. Real Hildas total
/// ~10⁻⁶ M⊕ — ~0.04% of the asteroid belt — so 0.05% is the right
/// order of magnitude and leaves room for the dominant body to be a
/// realistic ~150 km object.
pub(in crate::physics) const HILDA_MASS_FRAC: f64 = 0.0005;

/// Intrinsic probability that a star of mass `m_sun` hosts a leftover
/// planetesimal belt. This is an occurrence rate, not a detection rate:
/// belts form generically and persist even when too faint for IR-excess
/// surveys (whose 2–15 % rates are a different quantity). `age_gyr` is
/// accepted for signature symmetry; the rate is age-independent.
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

/// Median inner (asteroid-like) belt mass in Earth masses, with a
/// log-normal scatter. Calibrated so a solar analogue at 4.6 Gyr has
/// ~1.5×10⁻³ M⊕ — the real asteroid belt. The `(1 + t/0.3 Gyr)⁻¹`
/// factor models collisional depletion over a 300-Myr timescale
/// (Wyatt 2008): a 100-Myr belt is ~4× heavier than a 5-Gyr one.
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

/// Median outer (Kuiper-like) belt mass in Earth masses, with a
/// log-normal scatter. Calibrated so a solar analogue at 4.6 Gyr has
/// ~0.03 M⊕ — the observed cold classical Kuiper belt. The mild
/// `(1 + t/3 Gyr)⁻⁰·³` factor reflects that a primordial cold belt is
/// barely collisionally depleted, unlike the inner belt above.
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

/// Kirkwood gaps: the 3:1, 5:2, 7:3 and 2:1 mean-motion resonances with
/// a giant at `a_giant_au`. A p:q resonance (giant : body periods) lies
/// at `a = a_giant * (q/p)^(2/3)`.
pub(in crate::physics) fn kirkwood_gaps(a_giant_au: f64) -> Vec<Gap> {
    let mu = JUPITER_MASS / M_SUN_ME;
    let w = a_giant_au * mu.powf(2.0 / 7.0) * 0.05;
    let at = |p: f64, q: f64| a_giant_au * (q / p).powf(2.0 / 3.0);
    // The (q/p)^(2/3) factors are 0.48, 0.54, 0.57, 0.63, so this literal
    // order is already ascending; no post-sort is needed.
    vec![
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
    ]
}

/// Number of tracers a belt of mass `mass` (M⊕) receives, capped at
/// `budget`. Linear in mass: doubling the mass doubles the count. Two
/// floors apply — a physical one (`BELT_MIN_TRACER_MASS`) and a
/// visibility one (`BELT_MIN_TRACERS`) — neither of which exceeds
/// `budget`.
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
/// Returns `None` if 16 consecutive draws all land in gaps.
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

/// Draw a semi-major axis in `[lo, hi]` AU for an outer (Kuiper-like)
/// belt, outside every gap. The surface density falls as a⁻¹·⁵, so the
/// number of bodies per unit a goes as a⁻⁰·⁵; drawing sqrt(a) uniformly
/// gives exactly that distribution. Falls back to a flat draw when `hi`
/// is not finite.
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

/// Spawn one belt body on a Keplerian orbit about `centre`. `a_au` is
/// the semi-major axis in AU; `centre` is (x, y, vx, vy, mass). The AU
/// value is converted to gm only where `kepler_state` needs it, so
/// `class_for` sees exactly the `a_au` the caller used for gap checks.
#[allow(clippy::too_many_arguments)]
pub(in crate::physics) fn spawn_belt_body(
    b: &mut Bodies,
    centre: (f64, f64, f64, f64, f64),
    a_au: f64,
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
    let (x, y, vx, vy) = kepler_state(G * mc, a_au * au_in_gm(), e, nu, omega);
    let kind = classify_small_body(m);
    let i = b.push(cx + x, cy + y, cvx + vx, cvy + vy, m, kind);
    b.origin[i] = origin;
    b.class[i] = class_for(origin, a_au, snow_line_au);
    b.bulk_density[i] = draw_density(rng, b.class[i]);
    b.r[i] = b.radius_now(i);
    i
}

/// Angle (rad) and mass (M⊕) of the spawned giant nearest to `a_au`
/// about `centre`, if one lies within 10 %. A single planet scan yields
/// both, so callers that need both (trojan placement + Hilda gating)
/// only pay for one pass.
pub(in crate::physics) fn giant_near(
    b: &Bodies,
    centre: (f64, f64, f64, f64, f64),
    a_au: f64,
) -> Option<(f64, f64)> {
    let au = au_in_gm();
    let mut best: Option<(f64, f64, f64)> = None; // (delta, angle, mass)
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
        if d < 0.1 * a_au && best.is_none_or(|(bd, _, _)| d < bd) {
            best = Some((d, dy.atan2(dx), b.m[i]));
        }
    }
    best.map(|(_, ang, m)| (ang, m))
}

/// Angle (rad) of the spawned giant nearest to `a_au` about `centre`,
/// if one lies within 10 %. Thin wrapper over [`giant_near`].
pub(in crate::physics) fn giant_angle(
    b: &Bodies,
    centre: (f64, f64, f64, f64, f64),
    a_au: f64,
) -> Option<f64> {
    giant_near(b, centre, a_au).map(|(ang, _)| ang)
}

/// Mass (M⊕) of the spawned giant nearest to `a_au` about `centre`, if
/// one lies within 10 %. Thin wrapper over [`giant_near`].
pub(in crate::physics) fn giant_mass_near(
    b: &Bodies,
    centre: (f64, f64, f64, f64, f64),
    a_au: f64,
) -> Option<f64> {
    giant_near(b, centre, a_au).map(|(_, m)| m)
}

/// Spawn a Trojan at the L4 (`lead = +1`) or L5 (`lead = -1`) point of
/// a giant at `a_gm`. `giant_angle` is that giant's current azimuth.
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
    // Trojan clouds are icy planetesimals trapped at L4/L5.
    b.class[i] = BodyClass::Icy;
    b.bulk_density[i] = draw_density(rng, b.class[i]);
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
/// interlopers and escaped belt members. Real systems have a handful,
/// not hundreds: Poisson mean of 2, capped at 8.
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
            a_au,
            e,
            m,
            BodyOrigin::BackgroundDebris,
            snow_line_au,
            rng,
        );
    }
}

/// Spawn the bodies of `spec` about `centre`, using at most `budget`
/// bodies.
///
/// Tracers, not real bodies: a thin belt still gets a few, a massive one
/// is capped by the budget. When the parts want more than the budget, it
/// is shared in proportion to each part's demand (linear in mass), so a
/// population with twice the mass keeps twice the tracers.
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

    // Resolve the giant each trojan cluster sits near, once per cluster:
    // its mass gates the Hilda block below, its angle places the L4/L5
    // bodies. One planet scan per cluster instead of two.
    let trojan_giants: Vec<Option<(f64, f64)>> = spec
        .trojans
        .iter()
        .map(|t| giant_near(b, centre, t.giant_a_au))
        .collect();

    // Hildas: a 3:2 group about a Jupiter-like giant (300–350 M⊕). They
    // take their share of the budget first.
    let hilda_giant = spec
        .trojans
        .iter()
        .zip(&trojan_giants)
        .find(|(_, g)| g.is_some_and(|(_, m)| (300.0..=350.0).contains(&m)))
        .map(|(t, _)| t.giant_a_au);
    let n_hilda = if hilda_giant.is_some() {
        (budget / 20).clamp(1, 30).min(budget)
    } else {
        0
    };
    let avail = budget - n_hilda;

    // Plan: (part, wanted count, mass weight). The weight only matters
    // if total demand exceeds `avail`.
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

    // Keep only the parts that want the most tracers if we have more
    // parts than slots. A selection is O(n) and leaves the prefix as the
    // correct set; ordering within it does not matter downstream.
    if plan.len() > avail {
        plan.select_nth_unstable_by_key(avail, |p| std::cmp::Reverse(p.1));
        plan.truncate(avail);
    }

    let wanted: usize = plan.iter().map(|p| p.1).sum();
    if wanted > avail {
        // Proportional down-scale. Weighting by `mass_frac` (not sqrt) is
        // the physically correct share: a population with twice the mass
        // keeps twice the tracers.
        let mut assigned = 0usize;
        for (_, n, _) in plan.iter_mut() {
            *n = (((avail as f64) * (*n as f64) / (wanted as f64)).floor() as usize).max(1);
            assigned += *n;
        }
        // The per-part floor of 1 can overshoot; take the excess back
        // from the largest parts (each keeps at least one tracer).
        while assigned > avail {
            let Some(big) = plan.iter_mut().filter(|p| p.1 > 1).max_by_key(|p| p.1) else {
                break;
            };
            big.1 -= 1;
            assigned -= 1;
        }
        // Distribute the flooring remainder so the total hits `avail`
        // exactly. Which parts receive it is arbitrary; only the total
        // is meaningful.
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
                        a_au,
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
                    spawn_belt_body(b, centre, a_au, e, m, pop.kind, snow_line_au, rng);
                }
            }
            BeltPart::Trojan(troj) => {
                // Reuse the giant angle resolved above; fall back to a
                // random azimuth if no giant was found nearby.
                let ang = spec
                    .trojans
                    .iter()
                    .position(|t| std::ptr::eq(t, *troj))
                    .and_then(|idx| trojan_giants[idx])
                    .map(|(a, _)| a)
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
        // 3:2 resonance with the giant sits at a = a_giant·(2/3)^(2/3).
        let a_hilda_au = g_au * (2.0f64 / 3.0).powf(2.0 / 3.0);
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
                a_hilda_au,
                e,
                m,
                BodyOrigin::Hilda,
                snow_line_au,
                rng,
            );
        }
    }
}

/// Spawn highly eccentric, icy comets from the outer system.
///
/// Real comets have extreme orbits. To make their tails actually visible
/// we draw a perihelion `q` close to the star and a large `a` in the
/// outer system, then derive `e = 1 − q/a` — the classic sungrazer /
/// long-period orbit.
pub(in crate::physics) fn spawn_comets(
    b: &mut Bodies,
    centre: (f64, f64, f64, f64, f64),
    snow_line_au: f64,
    budget: usize,
    rng: &mut XorShift,
) {
    if budget == 0 {
        return;
    }
    // A system has 1–3 comets on average (Poisson mean 1.5).
    let n = poisson(rng, 1.5).clamp(0, 4).min(budget);

    for _ in 0..n {
        // Perihelion 0.2–3.0 AU — inside the snow line, so they become active.
        let q_au = 0.2 + rng.next_f64() * 2.8;

        // Semi-major axis 20–500 AU (Kuiper belt to inner Oort cloud).
        // Capped at 500 AU so they don't immediately hit HARD_CULL_DIST.
        let a_au = 20.0 + rng.next_f64() * 480.0;

        // e = 1 − q/a, clamped to keep the orbit well-defined.
        let e = (1.0 - q_au / a_au).clamp(0.0, 0.99);

        // Comet nucleus mass: 1e-12 to 1e-9 Earth masses.
        let m = sample_mass_power_law(rng, 1e-12, 1e-9, 1.8);

        // spawn_belt_body builds the Keplerian orbit; class_for assigns
        // BodyClass::Icy to Comet origins automatically.
        spawn_belt_body(b, centre, a_au, e, m, BodyOrigin::Comet, snow_line_au, rng);
    }
}
