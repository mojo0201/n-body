//! N-body physics: SoA bodies, direct / Barnes-Hut gravity, Yoshida 4th-order
//! integrator, swept collision detection at true physical radii, Roche-limit
//! tidal disruption, debris-disk coalescence, stellar evolution (main
//! sequence, giant branch, white dwarf / neutron star / black hole remnants),
//! energy bookkeeping and an event log.
//!
//! Units: length 1 Gm (1e6 km), mass 1 Earth mass, G = 7.4. One time unit is
//! about 50 days (see `time_unit_s`).

#![allow(dead_code)]

pub mod barnes_hut;
pub mod collisions;
pub mod debris;
pub mod dust;
pub mod edit;
pub mod energy;
pub mod evolution;
pub mod forces;
pub mod integrator;
pub mod naming;
pub mod orbit_lines;
pub mod orbits;
pub mod presets;
pub mod rng;
pub mod spawn;
#[cfg(test)]
mod tests;

pub use barnes_hut::*;
// `collisions` has no public items, so a plain `use` re-exports it to siblings.
use collisions::*;
pub use debris::*;
pub use dust::*;
pub use edit::*;
pub use energy::*;
pub use evolution::*;
pub use forces::*;
pub use integrator::*;
pub use naming::*;
pub use orbit_lines::*;
pub use orbits::*;
pub use presets::*;
pub use rng::*;
pub use spawn::*;

use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::OnceLock;

static DUST_VERSION_COUNTER: AtomicU64 = AtomicU64::new(1);

pub const G: f64 = 7.4;

/// Softening length is 1e-3 Gm (1000 km). Larger than the true radius of
/// a neutron star (1e-5 Gm) and a stellar-mass black hole's horizon, so
/// for those bodies it acts as a gravitational regularizer rather than
/// "softening below the surface".
pub const SOFT2: f64 = 1e-6;

pub const DT: f64 = 1.0 / 480.0;

pub const Y_W1: f64 = 1.3512071919596578;

pub const Y_W0: f64 = -1.7024143839193153;

/// Barnes-Hut opening angle. 0.35 is very conservative (~0.1 % force error);
/// 0.5 is the standard default (~1 % error) and roughly 1.5-2× cheaper at
/// n ≥ 600. For a visual sim the extra error is invisible.
pub const BH_THETA: f64 = 0.5;

/// Below this body count the exact O(n^2) pair sum is used (serial and symmetric
/// below `PAR_DIRECT_MIN`, parallel above it).
pub const BH_THRESHOLD: usize = 600;

pub const PAR_DIRECT_MIN: usize = 150;

/// With fewer active bodies than this, a direct sum beats building a tree.
pub const DIRECT_ACTIVE_MAX: usize = 64;

// ---- Fix 1: PAR_ACTIVE_MIN removed. The `active * n >= PAR_WORK_MIN`
// ---- check alone is the right gate: at n = 10 000 one body is ~50 µs
// ---- of work, far above rayon's ~10 µs dispatch. At small n,
// ---- PAR_WORK_MIN already keeps short ticks serial.
/// Below this much work (active bodies × all bodies) threads cost more than they save.
pub const PAR_WORK_MIN: usize = 20_000;

/// Block time steps: body i takes steps of dt / 2^rung, at most `MAX_RUNG` halvings.
pub const MAX_RUNG: u8 = 6;

/// Step limit as a fraction of the pair dynamical time sqrt(d^3 / (G M)).
pub const ETA_GRAV: f64 = 0.015;

/// Step limit as a fraction of the pair crossing time d / v_rel (catches fast fly-bys).
pub const ETA_VEL: f64 = 0.03;

/// Systems this small and quiet use the 4th-order Yoshida step instead of block leapfrog.
pub const YOSHIDA_MAX_N: usize = 1500;

pub const AU_KM: f64 = 1.495_978_707e8;

pub const LEN_KM: f64 = 1e6;

pub const MASS_KG: f64 = 5.9722e24;

pub const G_SI: f64 = 6.6743e-11;

pub const M_SUN_ME: f64 = 332_946.0;

pub const R_SUN_GM: f64 = 0.696;

pub const R_EARTH_GM: f64 = 0.006371;

pub const STAR_MR_EXP: f64 = 0.8;

pub const SCHWARZSCHILD_GM_PER_ME: f64 = 8.87e-12;

pub const STAR_M: f64 = M_SUN_ME;

pub const STAR_M_MIN: f64 = 0.08 * M_SUN_ME;

pub const BH_M_MIN: f64 = 3.0 * M_SUN_ME;

pub const JUPITER_MASS: f64 = 318.0;

// Brown dwarfs — deuterium-fusing, ~13–80 M♃, roughly Jupiter-sized.
pub const BD_M_MIN: f64 = 13.0 * JUPITER_MASS;

pub const BD_M_MAX: f64 = 80.0 * JUPITER_MASS;

pub const BD_R_EARTH: f64 = 11.21;

pub const BD_R_EXP: f64 = -0.04;

// White dwarfs — degenerate carbon/oxygen cores.
pub const WD_M_MIN: f64 = 0.5 * M_SUN_ME;

pub const WD_M_MAX: f64 = 1.4 * M_SUN_ME;

pub const WD_R_EARTH: f64 = 1.0;

pub const WD_R_EXP: f64 = -1.0 / 3.0;

// Neutron stars.
pub const NS_M_MIN: f64 = 1.4 * M_SUN_ME;

pub const NS_M_MAX: f64 = 2.5 * M_SUN_ME;

pub const NS_R_GM: f64 = 1e-5;

// Dwarf planets.
pub const DWARF_M_MAX: f64 = 0.013;

/// A belt body at or above this mass (Earth masses) is a dwarf planet.
pub const BELT_DWARF_M_MIN: f64 = 1e-3;

pub const BH_CAPTURE_HORIZONS: f64 = 3.0;

pub const BH_MIN_CAPTURE: f64 = 1e-3;

// ---- Stellar evolution -----------------------------------------------------
//
// Evolution is OFF until `Bodies::evo_rate` is set (see `set_evolution_rate`).
// A star keeps `BodyKind::Star` during its whole life; its phase follows from
// its age and initial mass (`star_state`). At the end of its life it becomes a
// white dwarf, neutron star or black hole (`remnant_of`).
/// Radius growth over the main sequence (age 0 = the preset's present state).
pub const EVO_MS_RADIUS_GROWTH: f64 = 0.4;

/// Luminosity growth over the main sequence.
pub const EVO_MS_LUM_GROWTH: f64 = 0.5;

/// Length of the giant phase as a fraction of the main-sequence lifetime.
pub const EVO_POST_MS_FRAC: f64 = 0.15;

pub const EVO_POST_MS_FRAC_MASSIVE: f64 = 0.10;

/// Stars at or above this initial mass end as supernovae (neutron star or black hole).
pub const EVO_MASSIVE_M: f64 = 8.0 * M_SUN_ME;

/// Stars at or above this initial mass leave a black hole, not a neutron star.
pub const EVO_BH_PROGENITOR_M: f64 = 20.0 * M_SUN_ME;

/// Peak giant radius of a 1 M☉ star (scales with sqrt(mass)), in R☉.
pub const EVO_GIANT_R_SUN: f64 = 200.0;

/// Peak supergiant radius of a 20 M☉ star (scales with sqrt(mass)), in R☉.
pub const EVO_SUPERGIANT_R_SUN: f64 = 800.0;

pub const EVO_GIANT_T: f64 = 3500.0;

pub const EVO_SUPERGIANT_T: f64 = 3600.0;

/// Fraction of the giant phase over which the surface cools to its final temperature.
pub const EVO_COOL_FRAC: f64 = 0.25;

/// Fraction of its mass a massive star loses in winds before it explodes.
pub const EVO_WIND_LOSS_MASSIVE: f64 = 0.2;

/// Natal kick: per-component Maxwellian sigma for a neutron star (Hobbs et al. 2005).
pub const SN_KICK_SIGMA_KMS: f64 = 265.0;

/// A rate that makes a 1 M☉ star live about 10 time units (about 500 days).
pub const EVO_RATE_DEMO: f64 = 1.0;

// White dwarf cooling: Mestel law, L ∝ M t^-7/6 (t = cooling age).
/// Luminosity in L☉ of a 0.6 M☉ white dwarf at a cooling age of 1 Gyr.
pub const WD_COOL_L_REF: f64 = 2e-4;

pub const WD_COOL_EXP: f64 = 7.0 / 6.0;

/// Offset that keeps a new white dwarf finite (10 000 yr).
pub const WD_COOL_T0_GYR: f64 = 1e-5;

/// Cooling age of a white dwarf made by `push`. A 0.6 M☉ star of this age has L = 1e-3 L☉.
pub const WD_DEFAULT_AGE_GYR: f64 = 0.25;

// Brown dwarf cooling: L ∝ M^2.64 t^-1.3.
/// Luminosity in L☉ of an 80 M♃ brown dwarf at `BD_DEFAULT_AGE_GYR`.
pub const BD_L_REF: f64 = 1e-4;

pub const BD_L_MASS_EXP: f64 = 2.64;

pub const BD_L_TIME_EXP: f64 = 1.3;

/// Young brown dwarfs stay at the brightness of this age.
pub const BD_AGE_FLOOR_GYR: f64 = 0.05;

pub const BD_DEFAULT_AGE_GYR: f64 = 1.0;

/// Extra radius (fraction) of a newborn brown dwarf, and its contraction time.
pub const BD_R_YOUNG_EXCESS: f64 = 0.5;

pub const BD_R_CONTRACT_GYR: f64 = 0.3;

/// Age of the universe in Gyr. No star of a random system is older than this,
/// however long its total lifetime is (a 0.5 Msun star lives ~56 Gyr).
pub const AGE_UNIVERSE_GYR: f64 = 13.8;

// Mass transfer in binaries (Roche-lobe overflow).
/// Transfer rate = coeff * M_donor / t_dyn * x^3, with x = (R - R_lobe) / R.
pub const MT_RATE_COEFF: f64 = 10.0;

/// A donor loses at most this fraction of its mass in one step.
pub const MT_MAX_FRAC_PER_STEP: f64 = 0.02;

/// Fraction of the transferred mass that a white dwarf, neutron star or black hole keeps.
pub const MT_EFF_COMPACT: f64 = 0.5;

/// A star can only overflow if a companion is within this many stellar radii.
pub const MT_NEAR_RADII: f64 = 25.0;

// ---- Realistic initial conditions ---------------------------------------
pub const MAX_PLANETS: usize = 15;

/// Fraction of main-sequence stars that host a compact chain of planets.
const P_COMPACT_CHAIN: f64 = 0.40;

/// Fraction of giant-planet hosts whose giants form tightly packed and are
/// dynamically unstable (2-4 giants within ~2-3.5 mutual Hill radii). The
/// N-body integrator then scatters, ejects or merges them: a "young unstable
/// system". The other hosts keep the stable 1.5-2.2 period spacing.
const P_UNSTABLE_GIANTS: f64 = 0.20;

/// Chance that a binary has a third star in a hierarchical orbit. About 70 %
/// of these fit the stability limit, so ~10 % of Sun-like stars end up in a
/// triple or higher multiple (Raghavan 2010, Tokovinin 2014).
const P_TRIPLE_GIVEN_BINARY: f64 = 0.30;

/// Chance that a triple has a fourth star (a 3+1 quadruple).
const P_QUAD_GIVEN_TRIPLE: f64 = 0.20;

/// Outermost allowed semi-major axis of a hierarchical star, AU.
const OUTER_A_MAX_AU: f64 = 3000.0;

pub const GIANT_FRACTION: f64 = 0.40;

pub const ROCKY_A_MIN: f64 = 0.4;

pub const ROCKY_A_MAX: f64 = 2.5;

pub const GIANT_A_MIN: f64 = 4.0;

pub const GIANT_A_MAX: f64 = 15.0;

pub const ASTEROID_A_MIN: f64 = 2.0;

pub const ASTEROID_A_SPAN: f64 = 1.3;

pub const KUIPER_A_MIN: f64 = 20.0;

pub const KUIPER_A_SPAN: f64 = 20.0;

pub const ROCKY_SPACING_LO: f64 = 1.35;

pub const ROCKY_SPACING_HI: f64 = 1.60;

pub const GIANT_SPACING_LO: f64 = 1.50;

pub const GIANT_SPACING_HI: f64 = 2.00;

pub const PLANET_JITTER: f64 = 0.10;

pub const ECC_ROCKY: f64 = 0.04;

pub const ECC_GIANT: f64 = 0.07;

pub const ECC_ASTEROID: f64 = 0.15;

pub const ECC_KUIPER: f64 = 0.12;

pub const MASS_PLAW_ROCKY: f64 = 1.5;

pub const MASS_PLAW_GIANT: f64 = 1.2;

pub const MASS_PLAW_ASTEROID: f64 = 1.7;

pub const MASS_PLAW_KUIPER: f64 = 1.8;

pub const MASS_ROCKY_MIN: f64 = 0.1;

pub const MASS_ROCKY_MAX: f64 = 5.0;

pub const MASS_GIANT_MIN: f64 = 50.0;

pub const MASS_GIANT_MAX: f64 = 300.0;

pub const MASS_ASTEROID_MIN: f64 = 1e-5;

pub const MASS_ASTEROID_MAX: f64 = 1e-3;

pub const MASS_KUIPER_MIN: f64 = 1e-6;

pub const MASS_KUIPER_MAX: f64 = 0.013;

// ---- Binary stars ----------------------------------------------------------
/// Binary period distribution (Raghavan et al. 2010): log-normal in log10(days).
pub const BIN_LOGP_MEAN: f64 = 5.03;

pub const BIN_LOGP_SIGMA: f64 = 2.28;

/// Shortest binary period generated, in days. Below about 2 d the block time
/// steps (`MAX_RUNG` halvings) can no longer resolve the orbit.
pub const BIN_P_MIN_D: f64 = 3.0;

/// Widest binary generated. Keeps both stars (and S-type planets around the
/// companion) well inside `CULL_DIST`.
pub const BIN_A_MAX_AU: f64 = 300.0;

/// Binaries tighter than this are tidally circularised (e = 0).
pub const BIN_CIRC_P_D: f64 = 7.0;

/// Periastron must clear this many summed stellar radii.
pub const BIN_PERI_CLEAR: f64 = 2.0;

/// Systems with fewer bodies than this stay single.
pub const BIN_MIN_BODIES: usize = 6;

/// S-type planets stay inside this fraction of the Holman-Wiegert limit,
/// P-type planets outside this multiple of it.
pub const HW_MARGIN_S: f64 = 0.7;

pub const HW_MARGIN_P: f64 = 1.25;

/// Below / above these separations the architecture is P-type / S-type; in
/// between the probability falls linearly in log(a).
pub const BIN_P_TYPE_BELOW_AU: f64 = 0.3;

pub const BIN_S_TYPE_ABOVE_AU: f64 = 20.0;

/// A body farther than this many binary separations from the barycentre is
/// treated as orbiting the pair as a whole.
pub const CIRCUMBINARY_FACTOR: f64 = 3.0;

/// Rough planet-occurrence suppression in binaries (S-type, P-type). Order of
/// magnitude only.
pub const BIN_OCC_S: f64 = 0.8;

pub const BIN_OCC_P: f64 = 0.6;

pub const CULL_DIST: f64 = 100_000.0;

pub const HARD_CULL_DIST: f64 = 1_000_000.0;

pub const MAX_BODIES: usize = 100_000;

pub const FLASH_LIFE: f64 = 0.6;

pub const MERGE_VEL_RATIO: f64 = 0.9;

pub const MERGE_B_WEIGHT: f64 = 0.5;

pub const DEBRIS_MAX_FRAC_HEADON: f64 = 0.6;

pub const HIT_RUN_B: f64 = 0.8;

pub const HIT_RUN_VEL_RATIO: f64 = 1.2;

pub const HIT_RUN_LOSS: f64 = 0.25;

pub const HIT_RUN_RESTITUTION: f64 = 0.5;

pub const LS_C_STAR: f64 = 1.9;

pub const FRAG_SLOPE: f64 = 2.85;

pub const FRAG_DYN_RANGE: f64 = 100.0;

pub const FRAG_COUNT_BASE: f64 = 3.0;

pub const FRAG_COUNT_SCALE: f64 = 6.0;

pub const FRAG_COUNT_EXP: f64 = 0.5;

pub const FRAG_COUNT_DISK_GAIN: f64 = 1.5;

pub const FRAG_COUNT_MAX: usize = 64;

pub const DISK_MASS_FRAC: f64 = 0.3;

pub const DISK_MAX_N: usize = 1500;

pub const DISK_R_MIN: f64 = 2.5;

pub const DISK_R_SPAN: f64 = 5.0;

pub const SMALL_BODY_M: f64 = 0.1;

pub const JUPITER_R_EARTH: f64 = 11.21;

pub const GIANT_R_EXP: f64 = -0.04;

pub const DEBRIS_VEL_RANGE: f64 = 2.5;

pub const DEBRIS_MAX_FRAC: f64 = 0.95;

pub const DEBRIS_MIN_MASS: f64 = 0.002;

pub const DEBRIS_COUNT_BASE: f64 = 3.0;

pub const DEBRIS_COUNT_SPAN: f64 = 7.0;

pub const DEBRIS_POWER_LAW: f64 = 1.5;

pub const DEBRIS_WEIGHT_MIN: f64 = 0.3;

pub const DEBRIS_KICK_MIN: f64 = 0.3;

pub const DEBRIS_KICK_SPAN: f64 = 0.7;

pub const DEBRIS_ANGLE_JIT: f64 = 0.4;

pub const DEBRIS_GAP_FRAC: f64 = 0.1;

pub const IMMUNITY_TIME: f64 = 0.4;

pub const RUBBLE_COALESCE_VRATIO: f64 = 1.0;

pub const RUBBLE_BOUNCE_VRATIO: f64 = 3.0;

pub const RUBBLE_BOUNCE_E: f64 = 0.35;

pub const RUBBLE_BOUNCE_IMMUNITY: f64 = 0.05;

pub const RUBBLE_PROMOTE_MASS: f64 = 0.05;

pub const DEBRIS_CAPTURE_HILL_FRAC: f64 = 0.3;

pub const DEBRIS_CAPTURE_STEPS: u32 = 200;

pub const DEBRIS_CAPTURE_VEL_RATIO: f64 = 0.5;

pub const DEBRIS_COALESCE_MAX_MASS: f64 = 0.05;

pub const DEBRIS_SEARCH_RADII: f64 = 30.0;

pub const TIDAL_MIN_MASS: f64 = 0.05;

pub const TIDAL_MASS_RATIO: f64 = 20.0;

pub const GIANT_MASS: f64 = 50.0;

pub const DISRUPT_KICK: f64 = 0.2;

pub const DISRUPT_STEP: f64 = 2.6;

pub const DISRUPT_JITTER: f64 = 0.5;

pub const DISRUPT_FRAG_BASE: f64 = 4.0;

pub const DISRUPT_FRAG_LOG: f64 = 2.0;

pub const ROCHE_RIGID: f64 = 1.26;

pub const ROCHE_FLUID: f64 = 2.44;

pub const ROCHE_RHO_FLUID: f64 = 1.5;

pub const ROCHE_RHO_RIGID: f64 = 4.0;

pub const EARTH_RHO_GCC: f64 = 5.51;

pub const MOON_HILL_FRAC: f64 = 0.5;

/// Dust added by a collision, as a fraction of its debris mass. Dust is a
/// bonus population: it is NOT taken from the debris bodies, so body mass
/// and momentum bookkeeping do not change (dust is not a body).
pub const DUST_YIELD_FRAC: f64 = 0.02;

/// Boost on the PR drag speed. A grain with beta folded in drifts from
/// 1 AU in about 10 yr at 100.
pub const PR_DRAG_BOOST: f64 = 100.0;

/// Initial dust mass as a fraction of each belt or rubble body's mass.
pub const DUST_TO_BELT_MASS: f64 = 0.3;

/// The dust field is advanced every this many physics steps (with a
/// matching `dt`): the drift per step is far below one cell.
pub const DUST_STEP_INTERVAL: u64 = 16;

pub const HASH_THRESHOLD: usize = 64;

pub const ENERGY_MAX_N: usize = 1500;

pub const EVENT_LOG_MAX: usize = 200;

pub const CLASS_ROCKY_MAX: f64 = 5.0;

pub const CLASS_SE_MAX: f64 = 10.0;

pub const CLASS_ICE_MAX: f64 = 50.0;

pub const CLASS_GAS_MAX: f64 = 318.0;

pub const CLASS_STAR_M_MAX: f64 = 0.45;

pub const CLASS_STAR_K_MAX: f64 = 0.8;

pub const CLASS_STAR_G_MAX: f64 = 1.2;

pub const CLASS_STAR_F_MAX: f64 = 1.5;

pub const CLASS_STAR_A_MAX: f64 = 2.5;

/// One simulation time unit in seconds.
pub fn time_unit_s() -> f64 {
    (G * (LEN_KM * 1000.0).powi(3) / (G_SI * MASS_KG)).sqrt()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyKind {
    Star,
    BrownDwarf,
    WhiteDwarf,
    NeutronStar,
    BlackHole,
    Planet,
    DwarfPlanet,
}

/// Where a body came from. Belt and debris origins decide its label, its
/// colour, and how it collides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyOrigin {
    Spawned,
    AsteroidBelt,
    KuiperCold,
    KuiperHot,
    KuiperResonant,
    KuiperScattered,
    Trojan,
    Hilda,
    ImpactDebris,
    TidalDebris,
    BackgroundDebris,
}

impl BodyOrigin {
    pub fn is_belt(self) -> bool {
        matches!(
            self,
            BodyOrigin::AsteroidBelt
                | BodyOrigin::KuiperCold
                | BodyOrigin::KuiperHot
                | BodyOrigin::KuiperResonant
                | BodyOrigin::KuiperScattered
                | BodyOrigin::Trojan
                | BodyOrigin::Hilda
        )
    }
    pub fn is_rubble(self) -> bool {
        matches!(
            self,
            BodyOrigin::ImpactDebris | BodyOrigin::TidalDebris | BodyOrigin::BackgroundDebris
        )
    }
}

fn is_host_kind(k: BodyKind) -> bool {
    matches!(
        k,
        BodyKind::Star
            | BodyKind::BrownDwarf
            | BodyKind::WhiteDwarf
            | BodyKind::NeutronStar
            | BodyKind::BlackHole
    )
}

fn radius_earth(m: f64) -> f64 {
    if m < SMALL_BODY_M {
        radius_earth(SMALL_BODY_M) * (m / SMALL_BODY_M).cbrt()
    } else if m < 2.04 {
        m.powf(0.279)
    } else {
        let nept = 0.79 * m.powf(0.589);
        let giant = JUPITER_R_EARTH * (m / JUPITER_MASS).powf(GIANT_R_EXP);
        nept.min(giant)
    }
}

/// True radius of a planet of mass `m` (Earth masses), in Gm.
pub fn planet_true_r(m: f64) -> f64 {
    R_EARTH_GM * radius_earth(m)
}

/// Bulk density in g/cm³ that makes `radius_from_density` reproduce the
/// mass-radius law `planet_true_r` for a planet. Other kinds use Earth's.
pub fn default_bulk_density(kind: BodyKind, m: f64) -> f64 {
    if matches!(kind, BodyKind::Planet | BodyKind::DwarfPlanet) {
        let r = planet_true_r(m);
        if r > 0.0 {
            3.0 * m * MASS_KG
                / (4.0 * std::f64::consts::PI * (r * LEN_KM * 1000.0).powi(3))
                / 1000.0 // kg/m³ -> g/cm³
        } else {
            3.0
        }
    } else {
        EARTH_RHO_GCC
    }
}

/// Radius in Gm of a body of mass `m` (Earth masses) and bulk density
/// `rho_gcc` (g/cm³).
pub fn radius_from_density(m: f64, rho_gcc: f64) -> f64 {
    let rho_kg_m3 = (rho_gcc * 1000.0).max(1e-6);
    let r_m = (3.0 * m * MASS_KG / (4.0 * std::f64::consts::PI * rho_kg_m3)).cbrt();
    r_m / (LEN_KM * 1000.0)
}

/// True radius in Gm. For a black hole this is the Schwarzschild radius.
pub fn physical_radius(kind: BodyKind, m: f64) -> f64 {
    match kind {
        BodyKind::Star => R_SUN_GM * (m / M_SUN_ME).powf(STAR_MR_EXP),
        BodyKind::BrownDwarf => R_EARTH_GM * BD_R_EARTH * (m / JUPITER_MASS).powf(BD_R_EXP),
        BodyKind::WhiteDwarf => R_EARTH_GM * WD_R_EARTH * (m / M_SUN_ME).powf(WD_R_EXP),
        BodyKind::NeutronStar => NS_R_GM,
        BodyKind::BlackHole => m * SCHWARZSCHILD_GM_PER_ME,
        BodyKind::Planet | BodyKind::DwarfPlanet => planet_true_r(m),
    }
}

pub struct Flash {
    pub x: f64,
    pub y: f64,
    pub r0: f64,
    pub age: f64,
    pub life: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    Merge,
    Impact,
    Tidal,
    Escape,
    User,
    Evolve,
}

pub struct Event {
    pub time: f64,
    pub kind: EventKind,
    pub text: String,
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Debug)]
pub struct Stats {
    pub merges: u64,
    pub impacts: u64,
    pub tidal: u64,
    pub escaped: u64,
    pub steps: u64,
    pub last_nsub: u32,
    pub max_rung: u8,
    pub energy0: f64,
    pub energy_offset: f64,
    pub coalesces: u64,
    pub bounces: u64,
}

struct Spawn {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    m: f64,
    kind: BodyKind,
    origin: BodyOrigin,
}

pub struct Bodies {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub ax: Vec<f64>,
    pub ay: Vec<f64>,
    pub _px: Vec<f64>,
    pub _py: Vec<f64>,
    pub m: Vec<f64>,
    pub r: Vec<f64>,
    pub kind: Vec<BodyKind>,
    pub dead: Vec<bool>,
    pub id: Vec<u64>,
    /// Human-readable name from the preset JSON, if any. `None` for
    /// procedurally generated bodies (procedural planets, debris, fragments).
    pub name: Vec<Option<String>>,
    pub imm: Vec<f64>,
    pub origin: Vec<BodyOrigin>,
    /// Bulk density in g/cm³ (planets and dwarf planets).
    pub bulk_density: Vec<f64>,
    pub rung: Vec<u8>,
    pub dtl: Vec<f64>,
    pub spin: Vec<f64>,
    pub disrupting: Vec<bool>,
    pub coalesce_counter: Vec<u32>,
    pub parent_cache: Vec<Option<usize>>,
    pub parent_cache_step: u64,
    pub parent_cache_valid: bool,
    pub flashes: Vec<Flash>,
    /// Statistical dust fields, one per system. They are not bodies.
    pub dust_fields: Vec<DustField>,
    pub time: f64,
    pub next_id: u64,
    pub stats: Stats,
    pub events: VecDeque<Event>,
    pub rng: XorShift,
    pub metallicity: f64,
    e_pre: Option<f64>,
    pub age_gyr: Vec<f64>,
    pub m_init: Vec<f64>,
    pub sn_kicks: bool,
    pub mass_transfer: bool,
    pub evolution_rate: f64,
    pub rlof: Vec<bool>,
}

impl Bodies {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            x: Vec::with_capacity(n),
            y: Vec::with_capacity(n),
            vx: Vec::with_capacity(n),
            vy: Vec::with_capacity(n),
            ax: Vec::with_capacity(n),
            ay: Vec::with_capacity(n),
            _px: Vec::with_capacity(n),
            _py: Vec::with_capacity(n),
            m: Vec::with_capacity(n),
            r: Vec::with_capacity(n),
            kind: Vec::with_capacity(n),
            dead: Vec::with_capacity(n),
            id: Vec::with_capacity(n),
            name: Vec::with_capacity(n),
            imm: Vec::with_capacity(n),
            origin: Vec::with_capacity(n),
            bulk_density: Vec::with_capacity(n),
            rung: Vec::with_capacity(n),
            dtl: Vec::with_capacity(n),
            spin: Vec::with_capacity(n),
            disrupting: Vec::with_capacity(n),
            coalesce_counter: Vec::with_capacity(n),
            parent_cache: Vec::with_capacity(n),
            parent_cache_step: 0,
            parent_cache_valid: false,
            flashes: Vec::new(),
            dust_fields: Vec::new(),
            time: 0.0,
            next_id: 1,
            stats: Stats {
                merges: 0,
                impacts: 0,
                tidal: 0,
                escaped: 0,
                steps: 0,
                last_nsub: 1,
                max_rung: 0,
                energy0: 0.0,
                energy_offset: 0.0,
                coalesces: 0,
                bounces: 0,
            },
            events: VecDeque::new(),
            rng: XorShift::new(1),
            metallicity: 0.0,
            e_pre: None,
            age_gyr: Vec::with_capacity(n),
            m_init: Vec::with_capacity(n),
            sn_kicks: true,
            mass_transfer: true,
            evolution_rate: 0.0,
            rlof: Vec::with_capacity(n),
        }
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    pub fn push(&mut self, x: f64, y: f64, vx: f64, vy: f64, m: f64, kind: BodyKind) -> usize {
        let i = self.x.len();
        self.x.push(x);
        self.y.push(y);
        self.vx.push(vx);
        self.vy.push(vy);
        self.ax.push(0.0);
        self.ay.push(0.0);
        self._px.push(x);
        self._py.push(y);
        self.m.push(m);
        let rho_default = default_bulk_density(kind, m);
        self.r.push(physical_radius(kind, m));
        self.kind.push(kind);
        self.dead.push(false);
        self.id.push(self.next_id);
        self.next_id += 1;
        self.name.push(None);
        self.imm.push(0.0);
        self.origin.push(BodyOrigin::Spawned);
        self.bulk_density.push(rho_default);
        self.rung.push(MAX_RUNG);
        self.dtl.push(f64::INFINITY);
        self.spin.push(0.0);
        self.disrupting.push(false);
        self.coalesce_counter.push(0);
        self.parent_cache.push(None);
        self.age_gyr.push(match kind {
            BodyKind::WhiteDwarf => WD_DEFAULT_AGE_GYR,
            BodyKind::BrownDwarf => BD_DEFAULT_AGE_GYR,
            _ => 0.0,
        });
        self.m_init.push(m);
        self.rlof.push(false);
        i
    }

    pub fn index_of_id(&self, id: u64) -> Option<usize> {
        self.id.iter().position(|&v| v == id)
    }

    pub fn set_mass(&mut self, i: usize, m: f64) {
        self.m[i] = m;
        self.r[i] = self.radius_now(i);
    }

    /// Current true radius of body `i`. A star uses its evolutionary state.
    pub fn radius_now(&self, i: usize) -> f64 {
        match self.kind[i] {
            BodyKind::Star => star_state(self.m_init[i], self.m[i], self.age_gyr[i]).radius_gm,
            BodyKind::BrownDwarf => bd_state(self.m[i], self.age_gyr[i]).radius_gm,
            BodyKind::Planet | BodyKind::DwarfPlanet => {
                radius_from_density(self.m[i], self.bulk_density[i])
            }
            k => physical_radius(k, self.m[i]),
        }
    }

    #[inline]
    pub fn contact_r(&self, i: usize) -> f64 {
        if self.kind[i] == BodyKind::BlackHole {
            (BH_CAPTURE_HORIZONS * self.r[i]).max(BH_MIN_CAPTURE)
        } else {
            self.r[i]
        }
    }

    pub fn log(&mut self, kind: EventKind, text: String, x: f64, y: f64) {
        self.events.push_back(Event {
            time: self.time,
            kind,
            text,
            x,
            y,
        });
        while self.events.len() > EVENT_LOG_MAX {
            self.events.pop_front();
        }
    }

    pub fn compact(&mut self) {
        let old_n = self.len();
        // old index -> new index (usize::MAX for removed bodies). Cached
        // parent indices must follow their bodies, or they point at the wrong
        // body (or past the end) until the next refresh.
        let mut remap = vec![usize::MAX; old_n];
        let mut write = 0;
        for (read, remap_slot) in remap.iter_mut().enumerate() {
            if self.dead[read] {
                continue;
            }
            *remap_slot = write;
            if write != read {
                self.x[write] = self.x[read];
                self.y[write] = self.y[read];
                self.vx[write] = self.vx[read];
                self.vy[write] = self.vy[read];
                self.ax[write] = self.ax[read];
                self.ay[write] = self.ay[read];
                self._px[write] = self._px[read];
                self._py[write] = self._py[read];
                self.m[write] = self.m[read];
                self.r[write] = self.r[read];
                self.kind[write] = self.kind[read];
                self.id[write] = self.id[read];
                self.name[write] = std::mem::take(&mut self.name[read]);
                self.imm[write] = self.imm[read];
                self.origin[write] = self.origin[read];
                self.bulk_density[write] = self.bulk_density[read];
                self.rung[write] = self.rung[read];
                self.dtl[write] = self.dtl[read];
                self.spin[write] = self.spin[read];
                self.disrupting[write] = self.disrupting[read];
                self.coalesce_counter[write] = self.coalesce_counter[read];
                self.parent_cache[write] = self.parent_cache[read];
                self.age_gyr[write] = self.age_gyr[read];
                self.m_init[write] = self.m_init[read];
                self.rlof[write] = self.rlof[read];
                self.dead[write] = false;
            }
            write += 1;
        }
        self.x.truncate(write);
        self.y.truncate(write);
        self.vx.truncate(write);
        self.vy.truncate(write);
        self.ax.truncate(write);
        self.ay.truncate(write);
        self._px.truncate(write);
        self._py.truncate(write);
        self.m.truncate(write);
        self.r.truncate(write);
        self.kind.truncate(write);
        self.dead.truncate(write);
        self.id.truncate(write);
        self.name.truncate(write);
        self.imm.truncate(write);
        self.origin.truncate(write);
        self.bulk_density.truncate(write);
        self.rung.truncate(write);
        self.dtl.truncate(write);
        self.spin.truncate(write);
        self.disrupting.truncate(write);
        self.coalesce_counter.truncate(write);
        self.parent_cache.truncate(write);
        self.age_gyr.truncate(write);
        self.m_init.truncate(write);
        self.rlof.truncate(write);
        // Dust fields follow their primary. A field whose primary died is dropped.
        let fields = std::mem::take(&mut self.dust_fields);
        self.dust_fields = fields
            .into_iter()
            .filter_map(|mut f| match remap.get(f.primary).copied() {
                Some(ni) if ni != usize::MAX => {
                    f.primary = ni;
                    Some(f)
                }
                _ => None,
            })
            .collect();
        for p in self.parent_cache.iter_mut() {
            *p = p
                .and_then(|k| remap.get(k).copied())
                .filter(|&v| v != usize::MAX);
        }
    }

    fn any_dead(&self) -> bool {
        self.dead.iter().any(|&d| d)
    }

    /// Recompute the parent of every body. This is O(n^2) because each
    /// `find_parent` call is O(n), so call it infrequently (every ~100
    /// steps), not every step. `coalesce_debris` and the UI both read the
    /// cache. Call `refresh_parent_cache` yourself after any operation
    /// (user edit, spawn) that can change parent relationships outside
    /// the normal step cadence.
    ///
    // ---- Fix 4: build the host list and partner table once, then use
    // ---- `find_parent_cached` per body. Cuts the redundant host_list
    // ---- rebuild and per-host partner search from every call.
    pub fn refresh_parent_cache(&mut self) {
        if self.parent_cache.len() != self.len() {
            self.parent_cache = vec![None; self.len()];
        }
        let ctx = build_parent_context(self);
        let bb: &Bodies = self;
        let parents: Vec<Option<usize>> = (0..bb.len())
            .into_par_iter()
            .map(|i| find_parent_cached(bb, i, &ctx))
            .collect();
        self.parent_cache = parents;
        self.parent_cache_step = self.stats.steps;
        self.parent_cache_valid = true;
    }
}

pub fn au_in_gm() -> f64 {
    AU_KM / LEN_KM
}

pub fn habitable_zone_au(kind: BodyKind, m: f64) -> (f64, f64) {
    let l = star_luminosity(kind, m);
    if l <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    ((l / 1.1).sqrt(), (l / 0.36).sqrt())
}

pub fn stellar_flux(b: &Bodies, x: f64, y: f64, skip: usize) -> f64 {
    let au = au_in_gm();
    let mut s = 0.0;
    for k in 0..b.len() {
        if k == skip {
            continue;
        }
        let l = luminosity_of(b, k);
        if l <= 0.0 {
            continue;
        }
        let d = (((b.x[k] - x).powi(2) + (b.y[k] - y).powi(2)).sqrt() / au).max(1e-6);
        s += l / (d * d);
    }
    s
}

pub fn equilibrium_temp(s: f64) -> f64 {
    255.0 * s.powf(0.25)
}

pub fn hill_radius(b: &Bodies, i: usize, h: usize) -> f64 {
    let d = ((b.x[i] - b.x[h]).powi(2) + (b.y[i] - b.y[h]).powi(2)).sqrt();
    d * (b.m[i] / (3.0 * b.m[h])).cbrt()
}

pub fn recenter_com(b: &mut Bodies) {
    let n = b.len();
    if n == 0 {
        return;
    }
    let (mut mt, mut cx, mut cy, mut px, mut py) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        mt += b.m[i];
        cx += b.m[i] * b.x[i];
        cy += b.m[i] * b.y[i];
        px += b.m[i] * b.vx[i];
        py += b.m[i] * b.vy[i];
    }
    if mt <= 0.0 {
        return;
    }
    cx /= mt;
    cy /= mt;
    px /= mt;
    py /= mt;
    for i in 0..n {
        b.x[i] -= cx;
        b.y[i] -= cy;
        b._px[i] = b.x[i];
        b._py[i] = b.y[i];
        b.vx[i] -= px;
        b.vy[i] -= py;
    }
}
