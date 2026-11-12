//! Stellar evolution, remnants, brown-dwarf and white-dwarf cooling, mass transfer.

use super::*;

pub fn star_luminosity(kind: BodyKind, m: f64) -> f64 {
    match kind {
        BodyKind::Star => (m / M_SUN_ME).powf(3.5),
        BodyKind::BrownDwarf => {
            let mj = m / JUPITER_MASS;
            (mj / 80.0).powf(2.5) * 1e-4
        }
        BodyKind::WhiteDwarf => 1e-3,
        BodyKind::NeutronStar => 1e-5,
        _ => 0.0,
    }
}

pub fn star_temperature(kind: BodyKind, m: f64) -> f64 {
    match kind {
        BodyKind::Star => {
            let l = star_luminosity(kind, m);
            let r = (physical_radius(kind, m) / R_SUN_GM).max(1e-6);
            5772.0 * (l / (r * r)).powf(0.25)
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

pub fn star_lifetime_gyr(kind: BodyKind, m: f64) -> f64 {
    match kind {
        BodyKind::Star => 10.0 * (m / M_SUN_ME).powf(-2.5),
        BodyKind::BrownDwarf | BodyKind::WhiteDwarf | BodyKind::NeutronStar => f64::INFINITY,
        _ => 0.0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StellarPhase {
    MainSequence,
    RedGiant,
    RedSupergiant,
}

/// Structure of a star at one moment of its life.
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

pub(super) fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub(super) fn post_ms_frac(m_init: f64) -> f64 {
    if m_init >= EVO_MASSIVE_M {
        EVO_POST_MS_FRAC_MASSIVE
    } else {
        EVO_POST_MS_FRAC
    }
}

/// Total life of a star in Gyr: main sequence plus giant phase.
pub fn star_total_life_gyr(m_init: f64) -> f64 {
    let t_ms = star_lifetime_gyr(BodyKind::Star, m_init);
    t_ms * (1.0 + post_ms_frac(m_init))
}

/// Object left when a star of initial mass `m_init` dies, and its mass.
/// White dwarf mass follows the initial-final mass relation of
/// Kalirai et al. (2008). Neutron stars span 1.4-2.0 M☉. Black holes keep
/// a quarter of the initial mass.
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

/// Structure of a star from its initial mass, current mass and age.
///
/// Age 0 gives exactly the present-day relations (`physical_radius`,
/// `star_luminosity`), so presets look the same until evolution is switched on.
pub fn star_state(m_init: f64, m: f64, age_gyr: f64) -> StarState {
    let t_ms = star_lifetime_gyr(BodyKind::Star, m_init).max(1e-12);
    if age_gyr < t_ms {
        let f = (age_gyr / t_ms).clamp(0.0, 1.0);
        let r = physical_radius(BodyKind::Star, m) * (1.0 + EVO_MS_RADIUS_GROWTH * f);
        let l = star_luminosity(BodyKind::Star, m) * (1.0 + EVO_MS_LUM_GROWTH * f);
        let t = 5772.0 * (l / (r / R_SUN_GM).powi(2)).powf(0.25);
        return StarState {
            phase: StellarPhase::MainSequence,
            progress: f,
            radius_gm: r,
            luminosity: l,
            temperature: t,
        };
    }
    // Giant phase. Structure follows the initial mass, which is constant.
    let massive = m_init >= EVO_MASSIVE_M;
    let post = t_ms * post_ms_frac(m_init);
    let p = ((age_gyr - t_ms) / post).clamp(0.0, 1.0);
    let r0 = physical_radius(BodyKind::Star, m_init) * (1.0 + EVO_MS_RADIUS_GROWTH);
    let l0 = star_luminosity(BodyKind::Star, m_init) * (1.0 + EVO_MS_LUM_GROWTH);
    let t0 = 5772.0 * (l0 / (r0 / R_SUN_GM).powi(2)).powf(0.25);
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

/// Mass a star has at `age_gyr`, before any other change. Low-mass stars lose
/// their envelope in the giant phase (most of it at the end). Massive stars
/// lose a fraction in winds. The rest leaves in the supernova.
pub(super) fn star_mass_at(m_init: f64, age_gyr: f64) -> f64 {
    let t_ms = star_lifetime_gyr(BodyKind::Star, m_init).max(1e-12);
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

/// Phase of body `i`, or `None` if it is not a star.
pub fn phase_of(b: &Bodies, i: usize) -> Option<StellarPhase> {
    (b.kind[i] == BodyKind::Star).then(|| star_state(b.m_init[i], b.m[i], b.age_gyr[i]).phase)
}

/// Luminosity of body `i` in L☉, with evolution included.
pub fn luminosity_of(b: &Bodies, i: usize) -> f64 {
    match b.kind[i] {
        BodyKind::Star => star_state(b.m_init[i], b.m[i], b.age_gyr[i]).luminosity,
        BodyKind::WhiteDwarf => wd_luminosity(b.m[i], b.age_gyr[i]),
        BodyKind::BrownDwarf => bd_state(b.m[i], b.age_gyr[i]).luminosity,
        k => star_luminosity(k, b.m[i]),
    }
}

/// Surface temperature of body `i` in K, with evolution included.
pub fn temperature_of(b: &Bodies, i: usize) -> f64 {
    match b.kind[i] {
        BodyKind::Star => star_state(b.m_init[i], b.m[i], b.age_gyr[i]).temperature,
        BodyKind::WhiteDwarf => wd_temperature(b.m[i], b.age_gyr[i]),
        BodyKind::BrownDwarf => bd_state(b.m[i], b.age_gyr[i]).temperature,
        k => star_temperature(k, b.m[i]),
    }
}

/// Habitable zone (inner, outer) of body `i` in AU, with evolution included.
pub fn habitable_zone_of(b: &Bodies, i: usize) -> (f64, f64) {
    let l = luminosity_of(b, i);
    if l <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    ((l / 1.1).sqrt(), (l / 0.36).sqrt())
}

/// Remaining life of star `i` in Gyr. Infinite for bodies that do not evolve.
pub fn star_time_left_gyr(b: &Bodies, i: usize) -> f64 {
    if b.kind[i] != BodyKind::Star {
        return f64::INFINITY;
    }
    (star_total_life_gyr(b.m_init[i]) - b.age_gyr[i]).max(0.0)
}

/// Switch stellar evolution on or off. `gyr_per_time_unit` is the stellar
/// time that passes in one simulation time unit (0 or less = off).
pub fn set_evolution_rate(b: &mut Bodies, gyr_per_time_unit: f64) {
    b.evolution_rate = if gyr_per_time_unit.is_finite() && gyr_per_time_unit > 0.0 {
        gyr_per_time_unit
    } else {
        0.0
    };
}

/// Put star `i` at a given age. Mass, radius and forces follow. A star older
/// than its life is placed just before its death.
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
/// Age a body has when it is created. White dwarfs and brown dwarfs start
/// partly cooled, so presets keep about the brightness they had before.
pub fn default_age_gyr(kind: BodyKind) -> f64 {
    match kind {
        BodyKind::WhiteDwarf => WD_DEFAULT_AGE_GYR,
        BodyKind::BrownDwarf => BD_DEFAULT_AGE_GYR,
        _ => 0.0,
    }
}

/// White dwarf luminosity in L☉ for a given cooling age (Mestel law).
pub fn wd_luminosity(m: f64, cooling_age_gyr: f64) -> f64 {
    let t = cooling_age_gyr.max(0.0) + WD_COOL_T0_GYR;
    WD_COOL_L_REF * (m / (0.6 * M_SUN_ME)) * t.powf(-WD_COOL_EXP)
}

/// White dwarf surface temperature in K for a given cooling age.
pub fn wd_temperature(m: f64, cooling_age_gyr: f64) -> f64 {
    let r = physical_radius(BodyKind::WhiteDwarf, m) / R_SUN_GM;
    5772.0 * (wd_luminosity(m, cooling_age_gyr) / (r * r)).powf(0.25)
}

/// Structure of a brown dwarf at one moment of its life.
#[derive(Clone, Copy, Debug)]
pub struct DwarfState {
    pub radius_gm: f64,
    /// In L☉.
    pub luminosity: f64,
    /// Surface temperature in K.
    pub temperature: f64,
}

/// Brown dwarf radius, luminosity and temperature from mass and age. Young
/// brown dwarfs are larger and brighter. They shrink and fade as L ∝ M^2.64 t^-1.3
/// (Burrows et al. 2001). At `BD_DEFAULT_AGE_GYR` the radius equals `physical_radius`.
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

// ---- Mass transfer in binaries -----------------------------------------------
/// Roche-lobe radius of a donor (Eggleton 1983), in the unit of `sep`.
pub fn roche_lobe_radius(sep: f64, m_donor: f64, m_other: f64) -> f64 {
    let q = (m_donor / m_other).max(1e-12);
    let q13 = q.cbrt();
    let q23 = q13 * q13;
    sep * 0.49 * q23 / (0.6 * q23 + (1.0 + q13).ln())
}

/// Least mass a donor can have. A main-sequence star stays above the
/// hydrogen-burning limit. A giant keeps its core (the future remnant).
pub(super) fn donor_floor(b: &Bodies, i: usize) -> f64 {
    let m0 = b.m_init[i];
    if b.age_gyr[i] < star_lifetime_gyr(BodyKind::Star, m0) {
        STAR_M_MIN
    } else {
        remnant_of(m0).1
    }
}

/// A star that gains mass keeps the same fraction of its main-sequence life,
/// so it looks younger (it gets fresh fuel).
pub(super) fn add_mass_to_star(b: &mut Bodies, i: usize, dm: f64) {
    let t0 = star_lifetime_gyr(BodyKind::Star, b.m_init[i]);
    if b.age_gyr[i] < t0 {
        let f = (b.age_gyr[i] / t0).clamp(0.0, 0.99);
        b.m_init[i] += dm;
        b.age_gyr[i] = f * star_lifetime_gyr(BodyKind::Star, b.m_init[i]);
    } else {
        b.m_init[i] += dm;
        let end = star_total_life_gyr(b.m_init[i]);
        b.age_gyr[i] = b.age_gyr[i].min(0.999 * end);
    }
}

/// Move `dm` from donor `d` to accretor `a`. A compact accretor keeps only
/// `MT_EFF_COMPACT` of it. The rest leaves the system with the donor's velocity.
///
/// Linear momentum and angular momentum are both conserved. The accretor
/// takes the donor's momentum. A pair of equal and opposite impulses then
/// restores the angular momentum that the move of mass would change.
/// Returns true if the accretor was destroyed (Type Ia supernova).
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

    let vax = (m_a_old * b.vx[a] + dm_acc * vdx) / m_a;
    let vay = (m_a_old * b.vy[a] + dm_acc * vdy) / m_a;

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

/// One step of Roche-lobe overflow for every star that fills its lobe.
/// Returns true if any mass changed.
pub(super) fn mass_transfer_step(b: &mut Bodies, dt: f64) -> bool {
    let hosts = host_list(b);
    if hosts.len() < 2 {
        return false;
    }
    let mut changed = false;
    let mut destroyed = false;
    let mut stripped: Vec<usize> = Vec::new();
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
        let floor = donor_floor(b, d);
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
        // A low-mass giant with no envelope left is a bare core: a white dwarf.
        let t_ms = star_lifetime_gyr(BodyKind::Star, b.m_init[d]);
        if b.m_init[d] < EVO_MASSIVE_M && b.age_gyr[d] >= t_ms && b.m[d] - floor <= 1e-9 * floor {
            stripped.push(d);
        }
    }
    for d in stripped {
        if !b.dead[d] {
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

/// km/s per simulation velocity unit.
pub(super) fn vel_unit_kms() -> f64 {
    LEN_KM / time_unit_s()
}

/// Orbits that loosely follow a host that loses mass slowly (adiabatic loss).
///
/// For a mass loss slow against the orbital period, `a * M` stays constant,
/// the eccentricity and orientation stay the same, and the body keeps its
/// place on the orbit. In the host frame that scales the separation by
/// `M_old / M_new` and the velocity by `M_new / M_old`. Positions and
/// velocities alone cannot express this (a plain change of mass at fixed
/// state is the *sudden* case), so the state is rewritten here.
///
/// Only what orbits `i` moves: planets and debris whose parent is `i`, and a
/// close bound stellar companion (wide pairs stay as they are). Their own satellites (moons, S-type planets)
/// travel with them, so those orbits stay intact. Bodies of other systems
/// are left alone. Linear momentum is not conserved: the envelope carries it.
pub(super) fn adiabatic_expand(b: &mut Bodies, i: usize, m_old: f64, m_new: f64) {
    /// Companion stars closer than this (about 100 AU, in Gm) expand with the orbit.
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
    let (sx, sy, svx, svy) = (b.x[i], b.y[i], b.vx[i], b.vy[i]);
    // (dx, dy, dvx, dvy) to add to each body.
    let mut delta: Vec<Option<[f64; 4]>> = vec![None; n];
    for k in 0..n {
        if k == i || b.dead[k] {
            continue;
        }
        let direct = if is_host_kind(b.kind[k]) {
            // Only a close companion. Wider pairs cannot be told from
            // neighbouring systems on a ring, which must stay where they are.
            (b.x[k] - sx).hypot(b.y[k] - sy) < ADIABATIC_COMPANION_MAX_GM
                && binary_partner_in(b, &hosts, k) == Some(i)
        } else {
            parents[k] == Some(i)
        };
        if !direct {
            continue;
        }
        // Include the body's own mass in the two-body parameter.
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
    // Satellites of a moved body keep their orbit by moving with it.
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

/// End of a star's life: it becomes a white dwarf, or explodes and leaves a
/// neutron star or black hole.
///
/// A white dwarf forms by slow (adiabatic) mass loss, so the orbits around it
/// expand (`adiabatic_expand`). A supernova throws the mass out at once: the
/// velocities stay as they were, so wide planets may become unbound.
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
                // Black holes get the same momentum as neutron stars, so a
                // heavier remnant moves slower.
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

/// Age every star by `dt` and apply the results: radius, mass loss and the
/// last stage of life. Does nothing while `b.evo_rate` is 0.
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
        let t_ms = star_lifetime_gyr(BodyKind::Star, m0);
        let end = star_total_life_gyr(m0);
        let leaving_ms = b.age_gyr[i] < t_ms && b.age_gyr[i] + dage >= t_ms;
        let before = if leaving_ms { Some(tag(b, i)) } else { None };
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
        let target = star_mass_at(m0, age);
        if target < b.m[i] {
            mark_energy(b);
            adiabatic_expand(b, i, b.m[i], target);
            b.set_mass(i, target);
            mass_changed = true;
        } else {
            b.r[i] = b.radius_now(i);
        }
    }
    for &i in &dying {
        become_remnant(b, i);
    }
    if b.mass_transfer && mass_transfer_step(b, dt) {
        mass_changed = true;
    }
    // Masses changed, so the stored accelerations are out of date.
    if mass_changed || !dying.is_empty() {
        compute_accel(b);
    }
}
