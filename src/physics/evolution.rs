//! Stellar evolution, end-of-life remnants, brown- and white-dwarf cooling
//! tracks, and Roche-lobe mass transfer in binaries.
//!
//! Stars have a main-sequence lifetime, mild MS growth, and a giant /
//! supergiant swelling phase. Mass is lost during the giant phase
//! (adiabatically) or in a supernova (impulsively). White dwarfs and
//! brown dwarfs cool along Mestel / Burrows tracks. Binary mass transfer
//! conserves linear and angular momentum and can change a star's mass,
//! luminosity, and eventual fate.

use super::*;

/// Bolometric luminosity in L☉ for a body of mass `m` (internal units).
///
/// Stars follow the standard main-sequence mass-luminosity relation
/// L ∝ M^3.5. Brown dwarfs scale as M^2.5 (no stable hydrogen burning).
/// Remnants are given flat representative values here; their real
/// brightness comes from the cooling tracks (`wd_luminosity`,
/// `bd_state`) in `luminosity_of`.
pub fn star_luminosity(kind: BodyKind, m: f64) -> f64 {
    match kind {
        BodyKind::Star => {
            let s = m / M_SUN_ME;
            s * s * s * s.sqrt()
        }
        BodyKind::BrownDwarf => {
            let x = m / JUPITER_MASS / 80.0;
            x * x * x.sqrt() * 1e-4
        }
        BodyKind::WhiteDwarf => 1e-3,
        BodyKind::NeutronStar => 1e-5,
        _ => 0.0,
    }
}

/// Effective surface temperature in K.
///
/// Main-sequence stars use Stefan-Boltzmann, T = T☉ · (L/L☉)^(1/4) ·
/// (R☉/R)^(1/2). Brown dwarfs use a linear ramp from ~500 K at the
/// planetary-mass end toward the hydrogen-burning limit. Remnants are
/// given flat values; the WD branch is superseded by the cooling
/// track in `wd_temperature`.
pub fn star_temperature(kind: BodyKind, m: f64) -> f64 {
    match kind {
        BodyKind::Star => {
            let l = star_luminosity(kind, m);
            let r = (physical_radius(kind, m) / R_SUN_GM).max(1e-6);
            5772.0 * (l / (r * r)).sqrt().sqrt()
        }
        BodyKind::BrownDwarf => {
            let mj = m / JUPITER_MASS;
            500.0 + 2000.0 * (mj / 80.0)
        }
        BodyKind::WhiteDwarf => 15_000.0,
        BodyKind::NeutronStar => 600_000.0,
        _ => 0.0,
    }
}

/// Main-sequence lifetime in Gyr.
///
/// From the fuel-burn-rate argument: available fuel ∝ M, burn rate ∝ L
/// ∝ M^3.5, so t_MS ∝ M^-2.5. Normalized so 1 M☉ lasts ~10 Gyr.
/// Brown dwarfs and remnants have no main sequence and return infinity.
pub fn star_lifetime_gyr(kind: BodyKind, m: f64) -> f64 {
    match kind {
        BodyKind::Star => {
            let s = m / M_SUN_ME;
            10.0 / (s * s * s.sqrt())
        }
        BodyKind::BrownDwarf | BodyKind::WhiteDwarf | BodyKind::NeutronStar => f64::INFINITY,
        _ => 0.0,
    }
}

/// Evolutionary stage of a star.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StellarPhase {
    MainSequence,
    RedGiant,
    RedSupergiant,
}

/// Snapshot of a star's structure at one moment of its life.
#[derive(Clone, Copy, Debug)]
pub struct StarState {
    pub phase: StellarPhase,
    /// Progress through the current phase, from 0 to 1.
    pub progress: f64,
    pub radius_gm: f64,
    /// In L☉.
    pub luminosity: f64,
    /// Surface temperature in K.
    pub temperature: f64,
}

/// Hermite smoothstep on [0, 1]. Eases the temperature ramp so the
/// giant branch cools in gradually rather than linearly.
pub(super) fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Fraction of MS lifetime spent in the giant phase. Massive stars
/// burn a shorter (relative) tail; low- and intermediate-mass stars
/// spend a larger fraction of their total life post-MS.
pub(super) fn post_ms_frac(m_init: f64) -> f64 {
    if m_init >= EVO_MASSIVE_M {
        EVO_POST_MS_FRAC_MASSIVE
    } else {
        EVO_POST_MS_FRAC
    }
}

/// Total lifetime (MS + giant phase) in Gyr.
pub fn star_total_life_gyr(m_init: f64) -> f64 {
    let t_ms = star_lifetime_gyr(BodyKind::Star, m_init);
    t_ms * (1.0 + post_ms_frac(m_init))
}

/// Object left when a star of initial mass `m_init` dies, and its mass.
///
///  * Low/intermediate mass: white dwarf via the Kalirai et al. (2008)
///    initial-final mass relation, capped at the Chandrasekhar limit.
///  * Massive: neutron star, linear in mass up to the BH threshold.
///  * Very massive: black hole, ~25% of the progenitor's mass.
pub fn remnant_of(m_init: f64) -> (BodyKind, f64) {
    let s = m_init / M_SUN_ME;
    if m_init < EVO_MASSIVE_M {
        let wd = (0.109 * s + 0.394).clamp(0.2, WD_M_MAX / M_SUN_ME);
        (BodyKind::WhiteDwarf, (wd * M_SUN_ME).min(m_init))
    } else if m_init < EVO_BH_PROGENITOR_M {
        let t = (m_init - EVO_MASSIVE_M) / (EVO_BH_PROGENITOR_M - EVO_MASSIVE_M);
        (BodyKind::NeutronStar, NS_M_MIN + 0.6 * M_SUN_ME * t)
    } else {
        (BodyKind::BlackHole, (0.25 * m_init).max(BH_M_MIN))
    }
}

/// Structure of a star from its initial mass, current mass, and age.
///
/// Main sequence: radius and luminosity grow by fixed small factors
/// over the MS lifetime, and T follows from L and R. Giant phase: the
/// star swells to the giant (or supergiant) radius and cools along a
/// smoothstep ramp to the giant effective temperature; L again follows
/// from the new R and T.
///
/// Age 0 gives exactly the static `physical_radius` / `star_luminosity`
/// relations, so presets are unchanged until evolution is switched on.
pub fn star_state(m_init: f64, m: f64, age_gyr: f64) -> StarState {
    let t_ms = star_lifetime_gyr(BodyKind::Star, m_init).max(1e-12);

    // Main-sequence (R, L, T) at fraction f of the MS, for a given
    // current mass. Factored out because the giant branch also needs
    // the endpoint (f = 1) as its starting point.
    let ms = |mass: f64, f: f64| -> (f64, f64, f64) {
        let r = physical_radius(BodyKind::Star, mass) * (1.0 + EVO_MS_RADIUS_GROWTH * f);
        let l = star_luminosity(BodyKind::Star, mass) * (1.0 + EVO_MS_LUM_GROWTH * f);
        let rr = r / R_SUN_GM;
        let t = 5772.0 * (l / (rr * rr)).sqrt().sqrt();
        (r, l, t)
    };

    if age_gyr < t_ms {
        let f = (age_gyr / t_ms).clamp(0.0, 1.0);
        let (r, l, t) = ms(m, f);
        return StarState {
            phase: StellarPhase::MainSequence,
            progress: f,
            radius_gm: r,
            luminosity: l,
            temperature: t,
        };
    }

    // Giant phase. Radius and phase come from the initial mass (which
    // is constant during the giant branch); the star reaches the MS
    // endpoint at p = 0 and the peak giant radius at p = 1.
    let massive = m_init >= EVO_MASSIVE_M;
    let post = t_ms * post_ms_frac(m_init);
    let p = ((age_gyr - t_ms) / post).clamp(0.0, 1.0);
    let (r0, _l0, t0) = ms(m_init, 1.0);

    let (r_peak, t_giant, phase) = if massive {
        let r = EVO_SUPERGIANT_R_SUN * (m_init / (20.0 * M_SUN_ME)).sqrt();
        (
            r.clamp(500.0, 1500.0) * R_SUN_GM,
            EVO_SUPERGIANT_T,
            StellarPhase::RedSupergiant,
        )
    } else {
        let r = EVO_GIANT_R_SUN * (m_init / M_SUN_ME).sqrt();
        (r * R_SUN_GM, EVO_GIANT_T, StellarPhase::RedGiant)
    };
    let r_peak = r_peak.max(r0);
    let r = r0 * (r_peak / r0).powf(p);
    let t = t0 + (t_giant - t0) * smoothstep(p / EVO_COOL_FRAC);
    let l = (r / R_SUN_GM).powi(2) * (t / 5772.0).powi(4);
    StarState {
        phase,
        progress: p,
        radius_gm: r,
        luminosity: l,
        temperature: t,
    }
}

/// Mass at the given age, after envelope loss but before mass transfer.
///
/// Low-mass stars peel off their envelope during the giant phase
/// (gradually, steepening near the end: ∝ p²). Massive stars lose a
/// fixed fraction of their mass to winds but keep at least their
/// remnant mass. Whatever is left leaves abruptly in the supernova,
/// which is handled in `become_remnant`, not here.
///
/// `t_ms` is passed in so callers that already have it avoid a
/// redundant `star_lifetime_gyr` evaluation in the hot loop.
pub(super) fn star_mass_at_pre(m_init: f64, t_ms: f64, age_gyr: f64) -> f64 {
    let t_ms = t_ms.max(1e-12);
    let post = t_ms * post_ms_frac(m_init);
    let p = ((age_gyr - t_ms) / post).clamp(0.0, 1.0);
    if p <= 0.0 {
        return m_init;
    }
    let (_, m_rem) = remnant_of(m_init);
    if m_init >= EVO_MASSIVE_M {
        let m_pre = (m_init * (1.0 - EVO_WIND_LOSS_MASSIVE)).max(m_rem);
        m_init - (m_init - m_pre) * p
    } else {
        m_init - (m_init - m_rem) * p * p
    }
}

/// Convenience wrapper that computes `t_ms` on the fly.
pub(super) fn star_mass_at(m_init: f64, age_gyr: f64) -> f64 {
    let t_ms = star_lifetime_gyr(BodyKind::Star, m_init).max(1e-12);
    star_mass_at_pre(m_init, t_ms, age_gyr)
}

/// Classify a body's stellar phase without computing full structure.
/// Non-stars return `None`. This mirrors `star_state`'s branch choice,
/// so callers that only need the phase skip the (more expensive)
/// radius/luminosity computation.
pub fn phase_of(b: &Bodies, i: usize) -> Option<StellarPhase> {
    if b.kind[i] != BodyKind::Star {
        return None;
    }
    let t_ms = star_lifetime_gyr(BodyKind::Star, b.m_init[i]).max(1e-12);
    Some(if b.age_gyr[i] < t_ms {
        StellarPhase::MainSequence
    } else if b.m_init[i] >= EVO_MASSIVE_M {
        StellarPhase::RedSupergiant
    } else {
        StellarPhase::RedGiant
    })
}

/// Luminosity of body `i` in L☉, including evolution. Each kind uses
/// its appropriate track: `star_state` for stars, the Mestel law for
/// white dwarfs, the Burrows track for brown dwarfs, and the flat
/// values from `star_luminosity` for everything else.
pub fn luminosity_of(b: &Bodies, i: usize) -> f64 {
    match b.kind[i] {
        BodyKind::Star => star_state(b.m_init[i], b.m[i], b.age_gyr[i]).luminosity,
        BodyKind::WhiteDwarf => wd_luminosity(b.m[i], b.age_gyr[i]),
        BodyKind::BrownDwarf => bd_state(b.m[i], b.age_gyr[i]).luminosity,
        k => star_luminosity(k, b.m[i]),
    }
}

/// Surface temperature of body `i` in K, including evolution.
pub fn temperature_of(b: &Bodies, i: usize) -> f64 {
    match b.kind[i] {
        BodyKind::Star => star_state(b.m_init[i], b.m[i], b.age_gyr[i]).temperature,
        BodyKind::WhiteDwarf => wd_temperature(b.m[i], b.age_gyr[i]),
        BodyKind::BrownDwarf => bd_state(b.m[i], b.age_gyr[i]).temperature,
        k => star_temperature(k, b.m[i]),
    }
}

/// Habitable zone (inner, outer) in AU.
///
/// Standard inverse-square scaling L = flux · r², with the solar-flux
/// multipliers 1.1 (inner, runaway greenhouse) and 0.36 (outer, early
/// Mars limit). Returns (NaN, NaN) for a non-luminous body.
pub fn habitable_zone_of(b: &Bodies, i: usize) -> (f64, f64) {
    let l = luminosity_of(b, i);
    if l <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    ((l / 1.1).sqrt(), (l / 0.36).sqrt())
}

/// Time remaining until star `i` dies, in Gyr. Infinite for anything
/// that does not evolve (remnants, brown dwarfs, planets).
pub fn star_time_left_gyr(b: &Bodies, i: usize) -> f64 {
    if b.kind[i] != BodyKind::Star {
        return f64::INFINITY;
    }
    (star_total_life_gyr(b.m_init[i]) - b.age_gyr[i]).max(0.0)
}

/// Enable or disable stellar evolution. `gyr_per_time_unit` is the
/// number of simulated Gyr that pass per simulation time unit.
/// Non-finite or non-positive values disable evolution.
pub fn set_evolution_rate(b: &mut Bodies, gyr_per_time_unit: f64) {
    b.evolution_rate = if gyr_per_time_unit.is_finite() && gyr_per_time_unit > 0.0 {
        gyr_per_time_unit
    } else {
        0.0
    };
}

/// Force a star to a given age. Clamped to just before its death so
/// it remains a star. Mass, radius, and velocity-dependent forces are
/// refreshed via `set_mass`.
pub fn set_star_age(b: &mut Bodies, i: usize, age_gyr: f64) {
    if b.kind[i] != BodyKind::Star || !age_gyr.is_finite() {
        return;
    }
    let end = star_total_life_gyr(b.m_init[i]);
    let age = age_gyr.clamp(0.0, end * 0.999_999);
    b.age_gyr[i] = age;
    let target = star_mass_at(b.m_init[i], age);
    let m = target.min(b.m[i]);
    b.set_mass(i, m);
    let text = format!("{} set to age {:.3} Gyr", tag(b, i), age);
    let (x, y) = (b.x[i], b.y[i]);
    b.log(EventKind::User, text, x, y);
    user_modified(b);
}

// ---- Dwarf cooling -----------------------------------------------------------

/// Initial age assigned to a newly created body. White and brown dwarfs
/// start partly cooled so their presets match the pre-evolution
/// appearance at default age.
pub fn default_age_gyr(kind: BodyKind) -> f64 {
    match kind {
        BodyKind::WhiteDwarf => WD_DEFAULT_AGE_GYR,
        BodyKind::BrownDwarf => BD_DEFAULT_AGE_GYR,
        _ => 0.0,
    }
}

/// White dwarf luminosity in L☉ under the Mestel cooling law.
///
/// A degenerate core with no nuclear source radiates its stored
/// thermal energy, giving L ∝ M · (t + t₀)^(-α), with α ≈ 1.4 for
/// the classic Mestel solution (see `WD_COOL_EXP`). `t₀` smooths the
/// initial hot, non-degenerate phase.
pub fn wd_luminosity(m: f64, cooling_age_gyr: f64) -> f64 {
    let t = cooling_age_gyr.max(0.0) + WD_COOL_T0_GYR;
    WD_COOL_L_REF * (m / (0.6 * M_SUN_ME)) * t.powf(-WD_COOL_EXP)
}

/// White dwarf effective temperature in K, from L and the (fixed) WD
/// radius via Stefan-Boltzmann.
pub fn wd_temperature(m: f64, cooling_age_gyr: f64) -> f64 {
    let r = physical_radius(BodyKind::WhiteDwarf, m) / R_SUN_GM;
    5772.0 * (wd_luminosity(m, cooling_age_gyr) / (r * r)).powf(0.25)
}

/// Brown dwarf structure at one moment of its life.
#[derive(Clone, Copy, Debug)]
pub struct DwarfState {
    pub radius_gm: f64,
    /// In L☉.
    pub luminosity: f64,
    /// Surface temperature in K.
    pub temperature: f64,
}

/// Brown dwarf radius, luminosity, and temperature from mass and age.
///
/// Young brown dwarfs are puffy and bright, then contract and dim
/// along the Burrows et al. (2001) tracks:
///   L ∝ M^2.64 · t^-1.3
/// The radius relaxes toward `physical_radius` on a timescale of
/// `BD_R_CONTRACT_GYR`, normalized so that at `BD_DEFAULT_AGE_GYR`
/// the radius matches the static preset value (zero excess).
pub fn bd_state(m: f64, age_gyr: f64) -> DwarfState {
    let mj = m / JUPITER_MASS;
    let age = age_gyr.max(0.0);
    let shrink = (-age / BD_R_CONTRACT_GYR).exp() - (-BD_DEFAULT_AGE_GYR / BD_R_CONTRACT_GYR).exp();
    let r = physical_radius(BodyKind::BrownDwarf, m) * (1.0 + BD_R_YOUNG_EXCESS * shrink);
    let t = age.max(BD_AGE_FLOOR_GYR) / BD_DEFAULT_AGE_GYR;
    let l = BD_L_REF * (mj / 80.0).powf(BD_L_MASS_EXP) * t.powf(-BD_L_TIME_EXP);
    let temp = 5772.0 * (l / (r / R_SUN_GM).powi(2)).powf(0.25);
    DwarfState {
        radius_gm: r,
        luminosity: l,
        temperature: temp,
    }
}

// ---- Mass transfer in binaries ----------------------------------------------

/// Roche-lobe radius of the donor in a binary with separation `sep`.
///
/// Eggleton (1983) approximation, accurate to ~1% over all mass ratios:
///   r_L / sep = 0.49 q^(2/3) / (0.6 q^(2/3) + ln(1 + q^(1/3)))
/// with q = m_donor / m_other.
pub fn roche_lobe_radius(sep: f64, m_donor: f64, m_other: f64) -> f64 {
    let q = (m_donor / m_other).max(1e-12);
    let q13 = q.cbrt();
    let q23 = q13 * q13;
    sep * 0.49 * q23 / (0.6 * q23 + (1.0 + q13).ln())
}

/// Add `dm` to a star's initial mass.
///
/// A main-sequence star keeps the same fraction of its (now longer)
/// MS lifetime, so accretion makes it appear rejuvenated. A post-MS
/// star keeps its age but must not overrun the new end of life, so
/// its age is clamped to just below the new total lifetime.
pub(super) fn add_mass_to_star(b: &mut Bodies, i: usize, dm: f64) {
    let t0 = star_lifetime_gyr(BodyKind::Star, b.m_init[i]);
    if b.age_gyr[i] < t0 {
        let f = (b.age_gyr[i] / t0).clamp(0.0, 0.99);
        b.m_init[i] += dm;
        b.age_gyr[i] = f * star_lifetime_gyr(BodyKind::Star, b.m_init[i]);
    } else {
        b.m_init[i] += dm;
        let t_ms = star_lifetime_gyr(BodyKind::Star, b.m_init[i]);
        let end = t_ms * (1.0 + post_ms_frac(b.m_init[i]));
        b.age_gyr[i] = b.age_gyr[i].min(0.999 * end);
    }
}

/// Move `dm` from donor `d` to accretor `a`.
///
/// Compact accretors (WD, NS, BH) retain only `MT_EFF_COMPACT`; the
/// rest is carried out of the system by the donor's wind (not tracked).
///
/// Momentum bookkeeping: the accretor absorbs the incoming mass at the
/// donor's velocity. Because the mass moves along the line of centers,
/// that transfer would change the system's angular momentum, so an
/// equal-and-opposite corrective impulse is applied to donor and
/// accretor to cancel the change. Linear and angular momentum are both
/// conserved as a result.
///
/// After the transfer, the accretor's fate is checked:
///   * super-Chandrasekhar white dwarf → Type Ia supernova (destroyed);
///   * heavy neutron star → collapse to black hole;
///   * brown dwarf crossing the hydrogen-burning limit → ignites.
///
/// Returns true if the accretor was destroyed.
pub(super) fn transfer_mass(b: &mut Bodies, d: usize, a: usize, dm: f64) -> bool {
    let eff = if matches!(
        b.kind[a],
        BodyKind::WhiteDwarf | BodyKind::NeutronStar | BodyKind::BlackHole
    ) {
        MT_EFF_COMPACT
    } else {
        1.0
    };
    let dm_acc = dm * eff;
    let m_d = b.m[d] - dm;
    let m_a_old = b.m[a];
    let m_a = m_a_old + dm_acc;
    let (vdx, vdy) = (b.vx[d], b.vy[d]);

    // New accretor velocity from momentum mixing.
    let vax = (m_a_old * b.vx[a] + dm_acc * vdx) / m_a;
    let vay = (m_a_old * b.vy[a] + dm_acc * vdy) / m_a;

    // Tangential component of the donor velocity relative to the
    // accretor, used to build the corrective impulse pair.
    let sx = b.x[d] - b.x[a];
    let sy = b.y[d] - b.y[a];
    let s = sx.hypot(sy).max(1e-12);
    let (hx, hy) = (sx / s, sy / s);
    let along = vdx * hx + vdy * hy;
    let px = dm_acc * (vdx - along * hx);
    let py = dm_acc * (vdy - along * hy);

    b.vx[d] += px / m_d;
    b.vy[d] += py / m_d;
    b.vx[a] = vax - px / m_a;
    b.vy[a] = vay - py / m_a;

    b.set_mass(d, m_d);
    if b.kind[a] == BodyKind::Star {
        add_mass_to_star(b, a, dm_acc);
    }
    b.set_mass(a, m_a);
    if b.kind[a] != BodyKind::Star {
        b.m_init[a] = m_a;
    }

    let (x, y) = (b.x[a], b.y[a]);
    match b.kind[a] {
        BodyKind::WhiteDwarf if m_a > WD_M_MAX => {
            let text = format!("{} exploded as a Type Ia supernova", tag(b, a));
            flash_at(b, x, y, 5.0);
            b.log(EventKind::Evolve, text, x, y);
            b.dead[a] = true;
            return true;
        }
        BodyKind::NeutronStar if m_a > NS_M_MAX => {
            b.kind[a] = BodyKind::BlackHole;
            b.set_mass(a, m_a);
            let text = format!("{} collapsed to a black hole", tag(b, a));
            b.log(EventKind::Evolve, text, x, y);
        }
        BodyKind::BrownDwarf if m_a >= STAR_M_MIN => {
            b.kind[a] = BodyKind::Star;
            b.m_init[a] = m_a;
            b.age_gyr[a] = 0.0;
            b.set_mass(a, m_a);
            let text = format!("{} ignited as a star", tag(b, a));
            b.log(EventKind::Evolve, text, x, y);
        }
        _ => {}
    }
    false
}

/// Least mass the donor can reach before transfer stalls.
///
/// A still-main-sequence star stops at the hydrogen-burning limit;
/// a post-MS star stops at its future remnant mass (the core it
/// cannot shed). `t_ms` is the donor's MS lifetime, passed in so the
/// hot loop doesn't re-evaluate it.
pub(super) fn donor_floor_pre(b: &Bodies, i: usize, t_ms: f64) -> f64 {
    if b.age_gyr[i] < t_ms {
        STAR_M_MIN
    } else {
        remnant_of(b.m_init[i]).1
    }
}

/// One step of Roche-lobe overflow.
///
/// For every star that fills its Roche lobe with respect to its binary
/// partner, transfer mass at the dynamical rate scaled by the cube of
/// the overflow depth, capped by a small fraction of the donor per
/// step and by the donor floor. A giant that has been stripped of its
/// entire envelope is converted to its remnant in place.
///
/// Returns true if any mass changed (the caller must recompute forces).
pub(super) fn mass_transfer_step(b: &mut Bodies, dt: f64) -> bool {
    let hosts = host_list(b);
    if hosts.len() < 2 {
        return false;
    }
    let mut changed = false;
    let mut destroyed = false;
    for &d in &hosts {
        if b.dead[d] || b.kind[d] != BodyKind::Star {
            continue;
        }
        let rd = b.r[d];
        let reach = MT_NEAR_RADII * rd;
        let near = hosts
            .iter()
            .any(|&k| k != d && (b.x[k] - b.x[d]).hypot(b.y[k] - b.y[d]) < reach);
        let partner = if near {
            binary_partner_in(b, &hosts, d)
        } else {
            None
        };

        // Overflow fraction x ∈ (0, 1]: how far inside the Roche lobe
        // the donor's surface has moved. 0 means it exactly fills.
        let overflow = partner.filter(|&a| !b.dead[a]).and_then(|a| {
            let sep = (b.x[a] - b.x[d]).hypot(b.y[a] - b.y[d]);
            let x = ((rd - roche_lobe_radius(sep, b.m[d], b.m[a])) / rd).clamp(0.0, 1.0);
            (x > 0.0).then_some((a, x))
        });

        let Some((a, x)) = overflow else {
            if b.rlof[d] {
                b.rlof[d] = false;
                let text = format!("{} stopped transferring mass", tag(b, d));
                let (px, py) = (b.x[d], b.y[d]);
                b.log(EventKind::Evolve, text, px, py);
            }
            continue;
        };
        if !b.rlof[d] {
            b.rlof[d] = true;
            let text = format!("{} began transferring mass to {}", tag(b, d), tag(b, a));
            let (px, py) = (b.x[d], b.y[d]);
            b.log(EventKind::Evolve, text, px, py);
        }

        let t_ms = star_lifetime_gyr(BodyKind::Star, b.m_init[d]).max(1e-12);
        let floor = donor_floor_pre(b, d, t_ms);
        let room = (b.m[d] - floor).max(0.0);
        let t_dyn = (rd.powi(3) / (G * b.m[d])).sqrt();
        let rate = MT_RATE_COEFF * b.m[d] / t_dyn * x.powi(3);
        let dm = (rate * dt).min(MT_MAX_FRAC_PER_STEP * b.m[d]).min(room);
        if dm > 0.0 {
            mark_energy(b);
            if transfer_mass(b, d, a, dm) {
                destroyed = true;
            }
            changed = true;
        }

        // A low-mass giant with no envelope left is a bare core: a
        // white dwarf. Convert in place so the next step sees a WD.
        if b.m_init[d] < EVO_MASSIVE_M && b.age_gyr[d] >= t_ms && b.m[d] - floor <= 1e-9 * floor {
            b.rlof[d] = false;
            become_remnant(b, d);
            changed = true;
        }
    }
    if destroyed {
        b.compact();
        b.parent_cache_valid = false;
    }
    changed
}

/// km/s per simulation velocity unit. Used to convert supernova kick
/// dispersions (given in km/s) into simulation units.
pub(super) fn vel_unit_kms() -> f64 {
    LEN_KM / time_unit_s()
}

/// Rescale the orbit of everything bound to a host that lost mass
/// slowly, i.e. adiabatically with respect to the orbital period.
///
/// Under adiabatic mass loss the orbital action a·M is invariant, so
/// separations grow as M_old / M_new and orbital velocities shrink by
/// the inverse factor. This is not expressible by simply changing the
/// host's mass while leaving the state alone — that would correspond
/// to impulsive loss. So positions and velocities are rewritten
/// explicitly here.
///
/// Only bodies orbiting `i` are touched:
///   * its planets and debris, whose stored parent is `i`;
///   * a close bound stellar companion. Wide pairs are left alone,
///     because on a ring they cannot be told apart from neighboring
///     systems that must not move.
/// Their own satellites (moons, S-type planets) ride along with them
/// so their internal orbits stay intact. Linear momentum is not
/// conserved here: the ejected envelope carries the difference.
pub(super) fn adiabatic_expand(b: &mut Bodies, i: usize, m_old: f64, m_new: f64) {
    /// Companion stars closer than this (about 100 AU, in Gm) expand
    /// with the orbit; wider pairs stay put.
    const ADIABATIC_COMPANION_MAX_GM: f64 = 15_000.0;
    let n = b.len();
    if m_new >= m_old || n < 2 {
        return;
    }
    let hosts = host_list(b);
    let use_cache = b.parent_cache_valid && b.parent_cache.len() == n;
    let parents: Vec<Option<usize>> = (0..n)
        .map(|k| {
            if k == i || b.dead[k] {
                None
            } else if use_cache {
                b.parent_cache[k]
            } else {
                find_parent(b, k)
            }
        })
        .collect();

    // Host state at the start of the expansion.
    let (sx, sy, svx, svy) = (b.x[i], b.y[i], b.vx[i], b.vy[i]);

    // (dx, dy, dvx, dvy) to apply to each body.
    let mut delta: Vec<Option<[f64; 4]>> = vec![None; n];
    for k in 0..n {
        if k == i || b.dead[k] {
            continue;
        }
        let direct = if is_host_kind(b.kind[k]) {
            // Close bound stellar companion only; wide pairs may be
            // other systems.
            (b.x[k] - sx).hypot(b.y[k] - sy) < ADIABATIC_COMPANION_MAX_GM
                && binary_partner_in(b, &hosts, k) == Some(i)
        } else {
            parents[k] == Some(i)
        };
        if !direct {
            continue;
        }

        // Two-body parameter: include the satellite's own mass in the
        // total, so a massive companion shrinks less than a light one.
        let rho = (m_new + b.m[k]) / (m_old + b.m[k]);
        let (rx, ry) = (b.x[k] - sx, b.y[k] - sy);
        let (rvx, rvy) = (b.vx[k] - svx, b.vy[k] - svy);
        delta[k] = Some([
            rx / rho - rx,
            ry / rho - ry,
            rvx * rho - rvx,
            rvy * rho - rvy,
        ]);
    }

    // Satellites of a moved body ride along with it. Their own orbit
    // around their parent is preserved because both get the same delta.
    let mut ride: Vec<Option<[f64; 4]>> = vec![None; n];
    for k in 0..n {
        if k == i || b.dead[k] || delta[k].is_some() {
            continue;
        }
        if let Some(p) = parents[k] {
            if p != i {
                ride[k] = delta[p];
            }
        }
    }

    for k in 0..n {
        if let Some(d) = delta[k].or(ride[k]) {
            b.x[k] += d[0];
            b.y[k] += d[1];
            b.vx[k] += d[2];
            b.vy[k] += d[3];
            b._px[k] = b.x[k];
            b._py[k] = b.y[k];
        }
    }
}

/// End of a star's life: turn it into its remnant.
///
/// The evolutionary path matters:
///   * White dwarfs form by slow (adiabatic) envelope ejection, so
///     surrounding orbits expand with `adiabatic_expand`.
///   * Neutron stars and black holes form in a sudden supernova: the
///     envelope leaves before the orbiting bodies can respond, so
///     velocities are kept as they are and wide planets may become
///     unbound. A stochastic kick is added if `sn_kicks` is enabled.
///     Black holes receive the same impulse as a canonical neutron
///     star, so a heavier remnant moves more slowly.
pub(super) fn become_remnant(b: &mut Bodies, i: usize) {
    mark_energy(b);
    let (kind, m_rem) = remnant_of(b.m_init[i]);
    let m_old = b.m[i];
    let before = tag(b, i);
    let (x, y) = (b.x[i], b.y[i]);
    if kind == BodyKind::WhiteDwarf {
        adiabatic_expand(b, i, m_old, m_rem.min(m_old));
    }
    b.kind[i] = kind;
    b.bulk_density[i] = EARTH_RHO_GCC;
    b.set_mass(i, m_rem.min(b.m[i]));
    b.age_gyr[i] = 0.0;
    b.m_init[i] = b.m[i];
    b.origin[i] = BodyOrigin::Spawned;
    match kind {
        BodyKind::WhiteDwarf => {
            let text = format!("{} shed its envelope and became {}", before, tag(b, i));
            flash_at(b, x, y, 1.0);
            b.log(EventKind::Evolve, text, x, y);
        }
        _ => {
            let what = if kind == BodyKind::BlackHole {
                "black hole"
            } else {
                "neutron star"
            };
            let text = format!("{} exploded as a supernova and left a {}", before, what);
            flash_at(b, x, y, 5.0);
            b.log(EventKind::Evolve, text, x, y);
            if b.sn_kicks {
                // Same momentum kick for BH as for NS, so heavier
                // remnants recoil more slowly.
                let scale = (NS_M_MIN / b.m[i]).min(1.0);
                let sigma = SN_KICK_SIGMA_KMS / vel_unit_kms() * scale;
                let kx = sample_normal(&mut b.rng, 0.0, sigma);
                let ky = sample_normal(&mut b.rng, 0.0, sigma);
                b.vx[i] += kx;
                b.vy[i] += ky;
            }
        }
    }
    b._px[i] = b.x[i];
    b._py[i] = b.y[i];
    b.parent_cache_valid = false;
}

/// Advance every evolving body by `dt` simulation time.
///
/// Order of operations:
///   1. White and brown dwarfs: age only; their structure follows from
///      age alone, so just refresh the radius.
///   2. Stars: detect the MS → giant transition (for logging), advance
///      mass loss, and collect those that reached end of life.
///   3. Convert dying stars to remnants.
///   4. Run one mass-transfer step if the feature is enabled.
///   5. Recompute accelerations if mass or membership changed.
///
/// A no-op while `evolution_rate` is zero.
pub(super) fn evolve_stars(b: &mut Bodies, dt: f64) {
    if b.evolution_rate <= 0.0 {
        return;
    }
    let dage = dt * b.evolution_rate;
    let mut dying: Vec<usize> = Vec::new();
    let mut mass_changed = false;
    for i in 0..b.len() {
        if b.dead[i] {
            continue;
        }
        match b.kind[i] {
            BodyKind::WhiteDwarf => {
                b.age_gyr[i] += dage;
                continue;
            }
            BodyKind::BrownDwarf => {
                b.age_gyr[i] += dage;
                b.r[i] = b.radius_now(i);
                continue;
            }
            BodyKind::Star => {}
            _ => continue,
        }
        let m0 = b.m_init[i];
        let t_ms = star_lifetime_gyr(BodyKind::Star, m0).max(1e-12);
        let end = t_ms * (1.0 + post_ms_frac(m0));

        // Detect the MS → giant crossing within this step (only for
        // logging; the structure itself doesn't need it).
        let leaving_ms = b.age_gyr[i] < t_ms && b.age_gyr[i] + dage >= t_ms;
        let before = leaving_ms.then(|| tag(b, i));

        b.age_gyr[i] += dage;
        let age = b.age_gyr[i];
        if age >= end {
            dying.push(i);
            continue;
        }
        if let Some(before) = before {
            let text = format!("{} left the main sequence and began to swell", before);
            let (x, y) = (b.x[i], b.y[i]);
            b.log(EventKind::Evolve, text, x, y);
        }

        if age < t_ms {
            // Still on the MS: no mass loss, just grow the radius in
            // place. Luminosity and T are recomputed lazily by
            // `luminosity_of` / `temperature_of` when needed.
            let f = (age / t_ms).clamp(0.0, 1.0);
            b.r[i] = physical_radius(BodyKind::Star, b.m[i]) * (1.0 + EVO_MS_RADIUS_GROWTH * f);
        } else {
            // Post-MS: mass loss.
            let target = star_mass_at_pre(m0, t_ms, age);
            if target < b.m[i] {
                mark_energy(b);
                adiabatic_expand(b, i, b.m[i], target);
                b.set_mass(i, target);
                mass_changed = true;
            } else {
                b.r[i] = b.radius_now(i);
            }
        }
    }
    for &i in &dying {
        become_remnant(b, i);
    }
    if b.mass_transfer && mass_transfer_step(b, dt) {
        mass_changed = true;
    }
    // Masses changed, so the stored accelerations are stale.
    if mass_changed || !dying.is_empty() {
        compute_accel(b);
    }
}
