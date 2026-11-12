//! N-body physics: SoA bodies, direct / Barnes-Hut gravity, Yoshida 4th-order
//! integrator, swept collision detection at true physical radii, Roche-limit
//! tidal disruption, debris-disk coalescence, stellar evolution (main
//! sequence, giant branch, white dwarf / neutron star / black hole remnants),
//! energy bookkeeping and an event log.
//!
//! Units: length 1 Gm (1e6 km), mass 1 Earth mass, G = 7.4. One time unit is
//! about 50 days (see `time_unit_s`).

#![allow(dead_code)]

use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub mod dust;
pub use dust::{DustField, DustPoint, Grain};
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

// ---- Fix 2: fixed chunk size for `force_direct_on_parallel`. The value
// ---- must stay constant so the fp sum order is deterministic across
// ---- thread counts. 2048 pairs × ~5 ns ≈ 10 µs per chunk, roughly the
// ---- cost of a rayon dispatch.
const DIRECT_CHUNK: usize = 2048;

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
/// Fewest tracer bodies of one belt population (so a thin belt is visible).
const BELT_MIN_TRACERS: usize = 3;

pub const BH_CAPTURE_HORIZONS: f64 = 3.0;
pub const BH_MIN_CAPTURE: f64 = 1e-3;

/// Reference belt mass (M⊕) for tracer scaling: a belt of this mass
/// gets `BELT_TRACER_SCALE` tracers.
const BELT_REF_MASS: f64 = 0.01;
/// Tracers per `BELT_REF_MASS` of belt mass. Each tracer therefore
/// stands for `BELT_REF_MASS / BELT_TRACER_SCALE` ≈ 2×10⁻⁵ M⊕ of real
/// debris — roughly Ceres's mass.
const BELT_TRACER_SCALE: f64 = 500.0;
/// Smallest mass a single tracer may carry (M⊕). Below this, the
/// "tracer" is sub-Ceres and not worth simulating.
const BELT_MIN_TRACER_MASS: f64 = 1e-6;
/// Largest mass a single tracer may carry (M⊕). Prevents a heavy belt
/// from producing body-scale collisions that dominate the dynamics.
const BELT_MAX_TRACER_MASS: f64 = 0.1;
/// Fraction of `spec.total_mass` that the Hilda block adds on top of the
/// main belt.
const HILDA_MASS_FRAC: f64 = 0.05;

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

const ORBIT_TABLE_SIZE: usize = 512;
const ORBIT_PIXELS_PER_SEG: f64 = 20.0;
const ORBIT_MAX_BODIES: usize = 4000;

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

/// Base62 alphabet for seed display and input.
pub const SEED_ALPHABET: &[u8; 62] =
    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Seed as 6 base62 characters, big-endian, zero-padded. 62^6 > 2^32.
pub fn seed_to_base62(mut seed: u32) -> String {
    let mut buf = [b'0'; 6];
    for k in (0..6).rev() {
        buf[k] = SEED_ALPHABET[(seed as usize) % 62];
        seed /= 62;
    }
    String::from_utf8(buf.to_vec()).unwrap()
}

/// Parse a base62 seed of 1 to 6 characters. `None` if bad or above `u32::MAX`.
pub fn seed_from_base62(s: &str) -> Option<u32> {
    if s.is_empty() || s.len() > 6 {
        return None;
    }
    let mut v: u64 = 0;
    for c in s.bytes() {
        let d = SEED_ALPHABET.iter().position(|&x| x == c)? as u64;
        v = v.checked_mul(62)?.checked_add(d)?;
    }
    if v > u32::MAX as u64 {
        return None;
    }
    Some(v as u32)
}
pub const HASH_THRESHOLD: usize = 64;
/// Below this many hot bodies a plain pair loop beats building a hash grid.
const HOT_GRID_MIN: usize = 24;
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

// ---------------------------------------------------------------------------
// Body kinds and physical radii
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Naming helpers
// ---------------------------------------------------------------------------

pub fn planet_class(m: f64) -> &'static str {
    if m < CLASS_ROCKY_MAX {
        "rocky"
    } else if m < CLASS_SE_MAX {
        "super-earth"
    } else if m < CLASS_ICE_MAX {
        "ice giant"
    } else if m < CLASS_GAS_MAX {
        "gas giant"
    } else {
        "super-jupiter"
    }
}

pub fn star_class(m: f64) -> char {
    let s = m / M_SUN_ME;
    if s < CLASS_STAR_M_MAX {
        'M'
    } else if s < CLASS_STAR_K_MAX {
        'K'
    } else if s < CLASS_STAR_G_MAX {
        'G'
    } else if s < CLASS_STAR_F_MAX {
        'F'
    } else if s < CLASS_STAR_A_MAX {
        'A'
    } else {
        'B'
    }
}

pub fn bd_class(m: f64) -> char {
    let mj = m / JUPITER_MASS;
    if mj >= 55.0 {
        'M'
    } else if mj >= 35.0 {
        'L'
    } else if mj >= 20.0 {
        'T'
    } else {
        'Y'
    }
}

pub fn class_name(b: &Bodies, i: usize) -> String {
    match b.kind[i] {
        BodyKind::BlackHole => "black hole".to_string(),
        BodyKind::NeutronStar => "neutron star".to_string(),
        BodyKind::WhiteDwarf => "white dwarf".to_string(),
        BodyKind::BrownDwarf => format!(
            "{}-type brown dwarf",
            bd_class_for_temp(bd_state(b.m[i], b.age_gyr[i]).temperature)
        ),
        BodyKind::Star => match phase_of(b, i) {
            Some(StellarPhase::RedGiant) => "red giant".to_string(),
            Some(StellarPhase::RedSupergiant) => "red supergiant".to_string(),
            _ => format!("{}-type star", star_class(b.m[i])),
        },
        BodyKind::Planet | BodyKind::DwarfPlanet => match b.origin[i] {
            BodyOrigin::AsteroidBelt => "asteroid".to_string(),
            BodyOrigin::KuiperCold => "cold classical KBO".to_string(),
            BodyOrigin::KuiperHot => "hot classical KBO".to_string(),
            BodyOrigin::KuiperResonant => "resonant KBO".to_string(),
            BodyOrigin::KuiperScattered => "scattered KBO".to_string(),
            BodyOrigin::Trojan => "Trojan".to_string(),
            BodyOrigin::Hilda => "Hilda".to_string(),
            BodyOrigin::ImpactDebris => "fragment".to_string(),
            BodyOrigin::TidalDebris => "tidal fragment".to_string(),
            BodyOrigin::BackgroundDebris => "debris".to_string(),
            BodyOrigin::Spawned => {
                if b.kind[i] == BodyKind::Planet {
                    planet_class(b.m[i]).to_string()
                } else {
                    "dwarf planet".to_string()
                }
            }
        },
    }
}

pub fn body_class(b: &Bodies, i: usize) -> String {
    body_class_with_parent(b, i, find_parent(b, i))
}

/// Same as `body_class`, but reuses an already-computed parent so callers
/// that need it for other columns don't call `find_parent` twice.
pub fn body_class_with_parent(b: &Bodies, i: usize, parent: Option<usize>) -> String {
    if matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
        && b.origin[i] == BodyOrigin::Spawned
    {
        if let Some(p) = parent {
            if matches!(b.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet)
                && b.origin[p] == BodyOrigin::Spawned
            {
                return "moon".to_string();
            }
        }
    }
    class_name(b, i)
}

pub fn mass_str(m: f64) -> String {
    if m >= 1e4 {
        format!("{:.2} M☉", m / M_SUN_ME)
    } else if m < 10.0 {
        format!("{:.2} M⊕", m)
    } else {
        format!("{:.0} M⊕", m)
    }
}

pub fn tag(b: &Bodies, i: usize) -> String {
    match &b.name[i] {
        Some(n) => format!("{} ({})", n, mass_str(b.m[i])),
        None => format!("#{} {} ({})", b.id[i], class_name(b, i), mass_str(b.m[i])),
    }
}
// ---------------------------------------------------------------------------
// Barnes-Hut quadtree
// ---------------------------------------------------------------------------

const MIN_NODE_SIZE: f64 = 1e-9;

struct Node {
    cx: f64,
    cy: f64,
    size: f64,
    m: f64,
    com_x: f64,
    com_y: f64,
    body: i32,
    children: [i32; 4],
}

impl Node {
    fn empty() -> Self {
        Self {
            cx: 0.0,
            cy: 0.0,
            size: 0.0,
            m: 0.0,
            com_x: 0.0,
            com_y: 0.0,
            body: -1,
            children: [-1; 4],
        }
    }
}

pub struct Tree {
    nodes: Vec<Node>,
}

fn partition_in_place(idx: &mut [u32], mut pred: impl FnMut(u32) -> bool) -> usize {
    let mut lo = 0;
    let mut hi = idx.len();
    while lo < hi {
        if pred(idx[lo]) {
            lo += 1;
        } else {
            hi -= 1;
            idx.swap(lo, hi);
        }
    }
    lo
}

// ---- Fix 5: parallel tree build. Each subtree is built independently
// ---- and concatenated at the root with child offsets patched in the
// ---- fixed q0..q3 order. The resulting node array is bit-identical to
// ---- the serial version.
impl Tree {
    pub fn build(b: &Bodies) -> Self {
        let n = b.len();
        if n == 0 {
            return Tree { nodes: Vec::new() };
        }
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for i in 0..n {
            xmin = xmin.min(b.x[i]);
            xmax = xmax.max(b.x[i]);
            ymin = ymin.min(b.y[i]);
            ymax = ymax.max(b.y[i]);
        }
        let cx = (xmin + xmax) * 0.5;
        let cy = (ymin + ymax) * 0.5;
        let size = (xmax - xmin).max(ymax - ymin).max(1.0) * 0.5 + 1.0;
        let mut idx: Vec<u32> = (0..n as u32).collect();
        let nodes = build_subtree(b, &mut idx, cx, cy, size, 0);
        Tree { nodes }
    }

    #[inline]
    fn accel_for(&self, b: &Bodies, i: usize) -> (f64, f64, f64) {
        let (xi, yi, mi) = (b.x[i], b.y[i], b.m[i]);
        let (vxi, vyi) = (b.vx[i], b.vy[i]);
        let theta2 = BH_THETA * BH_THETA;
        let mut stack = [0u32; 256];
        let mut sp = 1usize;
        let (mut sx, mut sy) = (0.0, 0.0);
        // Pair time-step limits without per-node divisions. With
        // inv = 1 / d^3:  1/t_dyn^2 = G (mi + mj) inv,  v^2 / d^2 = v^2 inv d.
        let mut s_max = 0.0f64;
        let mut w_max = 0.0f64;
        while sp > 0 {
            sp -= 1;
            let n = &self.nodes[stack[sp] as usize];
            let dx = n.com_x - xi;
            let dy = n.com_y - yi;
            let d2 = dx * dx + dy * dy + SOFT2;
            let accept = if n.body >= 0 {
                if n.body as usize == i {
                    continue;
                }
                true
            } else if n.body == -2 {
                if d2 < 4.0 * SOFT2 {
                    continue;
                }
                true
            } else {
                (n.size * 2.0) * (n.size * 2.0) < theta2 * d2
            };
            if accept {
                let d = d2.sqrt();
                let inv = 1.0 / (d2 * d);
                let f = G * n.m * inv;
                sx += f * dx;
                sy += f * dy;
                let s = (mi + n.m) * inv;
                if s > s_max {
                    s_max = s;
                }
                if n.body >= 0 {
                    let j = n.body as usize;
                    let dvx = b.vx[j] - vxi;
                    let dvy = b.vy[j] - vyi;
                    let w = (dvx * dvx + dvy * dvy) * inv * d;
                    if w > w_max {
                        w_max = w;
                    }
                }
            } else {
                for q in 0..4 {
                    if n.children[q] >= 0 && sp < stack.len() {
                        stack[sp] = n.children[q] as u32;
                        sp += 1;
                    }
                }
            }
        }
        (sx, sy, dt_limit(s_max, w_max))
    }
}

/// Build one subtree in pre-order. The returned nodes still have child
/// indices local to this subtree; the caller offsets them when merging.
fn build_subtree(
    b: &Bodies,
    indices: &mut [u32],
    cx: f64,
    cy: f64,
    size: f64,
    depth: usize,
) -> Vec<Node> {
    let mut total_m = 0.0;
    let mut com_x = 0.0;
    let mut com_y = 0.0;
    for &i in indices.iter() {
        let i = i as usize;
        total_m += b.m[i];
        com_x += b.m[i] * b.x[i];
        com_y += b.m[i] * b.y[i];
    }
    let (com_x, com_y) = if total_m > 0.0 {
        (com_x / total_m, com_y / total_m)
    } else {
        (cx, cy)
    };
    let single = indices.len() == 1;
    let coincident = !single && size < MIN_NODE_SIZE;
    let root = Node {
        cx,
        cy,
        size,
        m: total_m,
        com_x,
        com_y,
        body: if single {
            indices[0] as i32
        } else if coincident {
            -2
        } else {
            -1
        },
        children: [-1; 4],
    };
    if indices.is_empty() || single || coincident {
        return vec![root];
    }
    let ny = partition_in_place(indices, |i| b.y[i as usize] < cy);
    // Snapshot the length *before* `split_at_mut` borrows `indices`
    // mutably; the sub-slices are still live below.
    let n_here = indices.len();
    let (low, high) = indices.split_at_mut(ny);
    let nx_low = partition_in_place(low, |i| b.x[i as usize] < cx);
    let nx_high = partition_in_place(high, |i| b.x[i as usize] < cx);
    let (q0, q1) = low.split_at_mut(nx_low);
    let (q2, q3) = high.split_at_mut(nx_high);
    let h = size * 0.5;

    // Parallelise the top few levels. Below that the slices are small
    // enough that dispatch cost dominates the work.
    let par = depth < 3 && n_here >= 512;
    let trees: [Vec<Node>; 4] = if par {
        let ((t0, t1), (t2, t3)) = rayon::join(
            || {
                rayon::join(
                    || build_subtree(b, q0, cx - h, cy - h, h, depth + 1),
                    || build_subtree(b, q1, cx + h, cy - h, h, depth + 1),
                )
            },
            || {
                rayon::join(
                    || build_subtree(b, q2, cx - h, cy + h, h, depth + 1),
                    || build_subtree(b, q3, cx + h, cy + h, h, depth + 1),
                )
            },
        );
        [t0, t1, t2, t3]
    } else {
        [
            build_subtree(b, q0, cx - h, cy - h, h, depth + 1),
            build_subtree(b, q1, cx + h, cy - h, h, depth + 1),
            build_subtree(b, q2, cx - h, cy + h, h, depth + 1),
            build_subtree(b, q3, cx + h, cy + h, h, depth + 1),
        ]
    };

    let mut out: Vec<Node> = Vec::with_capacity(1 + trees.iter().map(|t| t.len()).sum::<usize>());
    out.push(root);
    for (q, mut sub) in trees.into_iter().enumerate() {
        if sub.is_empty() {
            continue;
        }
        let offset = out.len() as i32;
        out[0].children[q] = offset;
        for node in sub.iter_mut() {
            for c in node.children.iter_mut() {
                if *c >= 0 {
                    *c += offset;
                }
            }
        }
        out.append(&mut sub);
    }
    out
}

/// Step limit from the largest `1/t_dyn^2 / G` and `v^2 / d^2` seen.
#[inline]
fn dt_limit(s_max: f64, w_max: f64) -> f64 {
    let t_dyn = (1.0 / (G * s_max)).sqrt();
    let t_vel = (1.0 / w_max).sqrt();
    (ETA_GRAV * t_dyn).min(ETA_VEL * t_vel)
}

// ---- Fix 2: serial `force_direct_on` kept byte-for-byte (same fp order).
// ---- `force_direct_range` is only used by the parallel path.

fn force_direct_on(b: &Bodies, i: usize) -> (f64, f64, f64) {
    let n = b.len();
    let (xs, ys, ms) = (&b.x[..n], &b.y[..n], &b.m[..n]);
    let (vxs, vys) = (&b.vx[..n], &b.vy[..n]);
    let (xi, yi, mi) = (xs[i], ys[i], ms[i]);
    let (vxi, vyi) = (vxs[i], vys[i]);
    let (mut sx, mut sy) = (0.0, 0.0);
    let mut s_max = 0.0f64;
    let mut w_max = 0.0f64;
    // Two ranges instead of a `j == i` branch in the hot loop.
    for range in [0..i, (i + 1)..n] {
        for j in range {
            let dx = xs[j] - xi;
            let dy = ys[j] - yi;
            let d2 = dx * dx + dy * dy + SOFT2;
            let d = d2.sqrt();
            let inv = 1.0 / (d2 * d);
            let f = G * ms[j] * inv;
            sx += f * dx;
            sy += f * dy;
            let s = (mi + ms[j]) * inv;
            let dvx = vxs[j] - vxi;
            let dvy = vys[j] - vyi;
            let w = (dvx * dvx + dvy * dvy) * inv * d;
            if s > s_max {
                s_max = s;
            }
            if w > w_max {
                w_max = w;
            }
        }
    }
    (sx, sy, dt_limit(s_max, w_max))
}

/// One contiguous range of the j-loop. Used only by `force_direct_on_parallel`.
#[inline]
fn force_direct_range(b: &Bodies, i: usize, range: std::ops::Range<usize>) -> (f64, f64, f64, f64) {
    let (xs, ys, ms) = (&b.x, &b.y, &b.m);
    let (vxs, vys) = (&b.vx, &b.vy);
    let (xi, yi, mi) = (xs[i], ys[i], ms[i]);
    let (vxi, vyi) = (vxs[i], vys[i]);
    let (mut sx, mut sy) = (0.0, 0.0);
    let mut s_max = 0.0f64;
    let mut w_max = 0.0f64;
    for j in range {
        let dx = xs[j] - xi;
        let dy = ys[j] - yi;
        let d2 = dx * dx + dy * dy + SOFT2;
        let d = d2.sqrt();
        let inv = 1.0 / (d2 * d);
        let f = G * ms[j] * inv;
        sx += f * dx;
        sy += f * dy;
        let s = (mi + ms[j]) * inv;
        let dvx = vxs[j] - vxi;
        let dvy = vys[j] - vyi;
        let w = (dvx * dvx + dvy * dvy) * inv * d;
        if s > s_max {
            s_max = s;
        }
        if w > w_max {
            w_max = w;
        }
    }
    (sx, sy, s_max, w_max)
}

/// Parallel variant of `force_direct_on` for the case where the active
/// set is smaller than the thread pool but each body's j-loop is long
/// enough to justify splitting. Fixed chunk boundaries keep the fp sum
/// order identical across thread counts.
fn force_direct_on_parallel(b: &Bodies, i: usize) -> (f64, f64, f64) {
    let n = b.len();
    let mut bounds: Vec<(usize, usize)> = Vec::new();
    let mut lo = 0;
    while lo < i {
        let hi = (lo + DIRECT_CHUNK).min(i);
        bounds.push((lo, hi));
        lo = hi;
    }
    lo = i + 1;
    while lo < n {
        let hi = (lo + DIRECT_CHUNK).min(n);
        bounds.push((lo, hi));
        lo = hi;
    }
    let chunks: Vec<(f64, f64, f64, f64)> = bounds
        .par_iter()
        .map(|&(lo, hi)| force_direct_range(b, i, lo..hi))
        .collect();
    let (mut sx, mut sy, mut s_max, mut w_max) = (0.0, 0.0, 0.0f64, 0.0f64);
    for c in chunks {
        sx += c.0;
        sy += c.1;
        s_max = s_max.max(c.2);
        w_max = w_max.max(c.3);
    }
    (sx, sy, dt_limit(s_max, w_max))
}

fn compute_accel_direct_sym(b: &mut Bodies) {
    let n = b.len();
    for i in 0..n {
        b.ax[i] = 0.0;
        b.ay[i] = 0.0;
    }
    let mut s_max = vec![0.0f64; n];
    let mut w_max = vec![0.0f64; n];
    for i in 0..n {
        let (xi, yi, mi, vxi, vyi) = (b.x[i], b.y[i], b.m[i], b.vx[i], b.vy[i]);
        let (mut axi, mut ayi) = (0.0, 0.0);
        let (mut si, mut wi) = (s_max[i], w_max[i]);
        for j in (i + 1)..n {
            let dx = b.x[j] - xi;
            let dy = b.y[j] - yi;
            let d2 = dx * dx + dy * dy + SOFT2;
            let d = d2.sqrt();
            let inv = 1.0 / (d2 * d);
            let fi = G * b.m[j] * inv;
            let fj = G * mi * inv;
            axi += fi * dx;
            ayi += fi * dy;
            b.ax[j] -= fj * dx;
            b.ay[j] -= fj * dy;
            let s = (mi + b.m[j]) * inv;
            let dvx = b.vx[j] - vxi;
            let dvy = b.vy[j] - vyi;
            let w = (dvx * dvx + dvy * dvy) * inv * d;
            if s > si {
                si = s;
            }
            if s > s_max[j] {
                s_max[j] = s;
            }
            if w > wi {
                wi = w;
            }
            if w > w_max[j] {
                w_max[j] = w;
            }
        }
        b.ax[i] += axi;
        b.ay[i] += ayi;
        s_max[i] = si;
        w_max[i] = wi;
    }
    for i in 0..n {
        b.dtl[i] = dt_limit(s_max[i], w_max[i]);
    }
}

fn eval_forces(b: &mut Bodies, active: &[u32]) {
    eval_forces_with(b, active, None);
}

/// Like `eval_forces`, but accepts a pre-built tree. Callers inside
/// `step_block` build the tree once and pass `Some(&tree)` for every
/// tick; everything else passes `None` and pays a rebuild.
///
// ---- Fix 1 + Fix 2: dispatch order is
// ----   1. tree (n >= BH_THRESHOLD, active >= DIRECT_ACTIVE_MAX)
// ----   2. serial (active * n < PAR_WORK_MIN)
// ----   3. parallel-within-body (few active, n >= 2*DIRECT_CHUNK)
// ----   4. parallel-across-bodies (everything else)
fn eval_forces_with(b: &mut Bodies, active: &[u32], tree: Option<&Tree>) {
    let n = b.len();
    if active.is_empty() {
        return;
    }
    let use_tree = n >= BH_THRESHOLD && active.len() >= DIRECT_ACTIVE_MAX;
    let threads = rayon::current_num_threads();
    let out: Vec<(f64, f64, f64)> = if use_tree {
        // Build only if the caller didn't hand us one.
        let owned;
        let tree_ref = if let Some(t) = tree {
            t
        } else {
            owned = Tree::build(b);
            &owned
        };
        let bb: &Bodies = b;
        active
            .par_iter()
            .map(|&i| tree_ref.accel_for(bb, i as usize))
            .collect()
    } else if active.len() * n < PAR_WORK_MIN {
        // Tiny: dispatch would cost more than the work.
        active
            .iter()
            .map(|&i| force_direct_on(b, i as usize))
            .collect()
    } else if active.len() < threads && n >= 2 * DIRECT_CHUNK {
        // Few active bodies, long j-loops: split each body's loop.
        let bb: &Bodies = b;
        active
            .iter()
            .map(|&i| force_direct_on_parallel(bb, i as usize))
            .collect()
    } else {
        // Plenty of active bodies: split over active.
        let bb: &Bodies = b;
        active
            .par_iter()
            .map(|&i| force_direct_on(bb, i as usize))
            .collect()
    };
    for (k, &i) in active.iter().enumerate() {
        let i = i as usize;
        b.ax[i] = out[k].0;
        b.ay[i] = out[k].1;
        b.dtl[i] = out[k].2;
    }
}

/// Recompute forces on every body.
pub fn compute_accel(b: &mut Bodies) {
    let n = b.len();
    if n == 0 {
        return;
    }
    if n < PAR_DIRECT_MIN {
        compute_accel_direct_sym(b);
    } else {
        let all: Vec<u32> = (0..n as u32).collect();
        eval_forces(b, &all);
    }
    let mut mt = 0.0;
    let mut px = 0.0;
    let mut py = 0.0;
    for i in 0..n {
        mt += b.m[i];
        px += b.m[i] * b.ax[i];
        py += b.m[i] * b.ay[i];
    }
    if mt > 0.0 {
        let cx = px / mt;
        let cy = py / mt;
        for i in 0..n {
            b.ax[i] -= cx;
            b.ay[i] -= cy;
        }
    }
}

// ---------------------------------------------------------------------------
// Integrator
// ---------------------------------------------------------------------------

#[inline]
fn leapfrog_step(b: &mut Bodies, dt: f64) {
    let n = b.len();
    for i in 0..n {
        b.vx[i] += 0.5 * b.ax[i] * dt;
        b.vy[i] += 0.5 * b.ay[i] * dt;
        b.x[i] += b.vx[i] * dt;
        b.y[i] += b.vy[i] * dt;
    }
    compute_accel(b);
    for i in 0..n {
        b.vx[i] += 0.5 * b.ax[i] * dt;
        b.vy[i] += 0.5 * b.ay[i] * dt;
    }
}

#[inline]
fn rung_for(lim: f64, dt: f64) -> u8 {
    if lim.is_finite() && lim < dt {
        ((dt / lim).log2().ceil() as i32).clamp(0, MAX_RUNG as i32) as u8
    } else {
        0
    }
}

fn assign_rungs(b: &mut Bodies, dt: f64) -> u8 {
    let mut kmax = 0;
    for i in 0..b.len() {
        let k = rung_for(b.dtl[i], dt);
        b.rung[i] = k;
        kmax = kmax.max(k);
    }
    kmax
}

fn age_flashes(b: &mut Bodies, dt: f64) {
    for f in &mut b.flashes {
        f.age += dt;
    }
    b.flashes.retain(|f| f.age < f.life);
}

fn snapshot_positions(b: &mut Bodies) {
    for i in 0..b.len() {
        b._px[i] = b.x[i];
        b._py[i] = b.y[i];
    }
}

fn yoshida_step(b: &mut Bodies, dt: f64) {
    snapshot_positions(b);
    leapfrog_step(b, Y_W1 * dt);
    leapfrog_step(b, Y_W0 * dt);
    leapfrog_step(b, Y_W1 * dt);
    age_flashes(b, dt);
    collision_pass(b, dt, None);
    settle_energy(b);
    b.time += dt;
}

// ---- Fix 3: per-rung index lists. Rebuilt only when the body set
// ---- changes (compact or any collision), then mutated in O(1) per rung
// ---- change via `move_rung`. `hot` (rung > 0) is kept sorted.

fn rebuild_rung_lists(
    b: &Bodies,
    kmax: u8,
    by_rung: &mut [Vec<u32>],
    bucket_pos: &mut Vec<u32>,
    hot: &mut Vec<usize>,
) {
    debug_assert_eq!(by_rung.len(), kmax as usize + 1);
    for v in by_rung.iter_mut() {
        v.clear();
    }
    if bucket_pos.len() != b.len() {
        bucket_pos.resize(b.len(), 0);
    }
    for (i, pos) in bucket_pos.iter_mut().enumerate() {
        let r = b.rung[i].min(kmax) as usize;
        *pos = by_rung[r].len() as u32;
        by_rung[r].push(i as u32);
    }
    hot.clear();
    for list in by_rung.iter().take(kmax as usize + 1).skip(1) {
        hot.extend(list.iter().map(|&x| x as usize));
    }
    hot.sort_unstable();
}

#[inline]
fn move_rung(i: u32, old_r: usize, new_r: usize, by_rung: &mut [Vec<u32>], bucket_pos: &mut [u32]) {
    let pos = bucket_pos[i as usize] as usize;
    let last = by_rung[old_r].pop().unwrap();
    if pos < by_rung[old_r].len() {
        by_rung[old_r][pos] = last;
        bucket_pos[last as usize] = pos as u32;
    }
    bucket_pos[i as usize] = by_rung[new_r].len() as u32;
    by_rung[new_r].push(i);
}

fn step_block(b: &mut Bodies, dt: f64, kmax: u8) {
    let ticks = 1u32 << kmax;
    let dtf = dt / ticks as f64;
    let n0 = b.len();
    for i in 0..n0 {
        let h = dt / (1u32 << b.rung[i]) as f64;
        b.vx[i] += 0.5 * b.ax[i] * h;
        b.vy[i] += 0.5 * b.ay[i] * h;
    }
    // Build the Barnes-Hut tree ONCE for the whole block. Bodies move at
    // most dt/tick per tick (≈ 1/15000 sim units), which is < 0.01 % of a
    // tree cell, so the topology is valid for all `ticks` sub-steps. The
    // node COM values drift by an amount far below the BH opening-angle
    // error, so reuse is safe. This replaces 2^kmax tree builds with one.
    //
    // EXCEPTION: `collision_pass` inside the tick loop can call
    // `compact()`, which shrinks `b` and remaps every index. The cached
    // tree still holds the *old* body indices and would panic or silently
    // read the wrong body. Rebuild the tree (and the per-rung lists)
    // whenever the body set changes.
    let mut tree = if b.len() >= BH_THRESHOLD {
        Some(Tree::build(&*b))
    } else {
        None
    };
    let mut tree_len = b.len();

    // Per-rung index lists, rebuilt whenever the body set changes.
    let mut by_rung: Vec<Vec<u32>> = (0..=(kmax as usize)).map(|_| Vec::new()).collect();
    let mut bucket_pos: Vec<u32> = Vec::with_capacity(b.len());
    let mut hot: Vec<usize> = Vec::new();
    rebuild_rung_lists(&*b, kmax, &mut by_rung, &mut bucket_pos, &mut hot);

    let mut active: Vec<u32> = Vec::new();
    for tick in 1..=ticks {
        let n = b.len();
        for i in 0..n {
            b._px[i] = b.x[i];
            b._py[i] = b.y[i];
            b.x[i] += b.vx[i] * dtf;
            b.y[i] += b.vy[i] * dtf;
        }
        // Active set: bodies whose rung r satisfies trailing_zeros(tick)
        // >= kmax - r. Union of the cached buckets for r >= r_min.
        let tz = tick.trailing_zeros() as usize;
        let r_min = (kmax as usize).saturating_sub(tz);
        active.clear();
        for list in by_rung.iter().take(kmax as usize + 1).skip(r_min) {
            active.extend_from_slice(list);
        }
        if b.len() != tree_len {
            tree = if b.len() >= BH_THRESHOLD {
                Some(Tree::build(&*b))
            } else {
                None
            };
            tree_len = b.len();
            rebuild_rung_lists(&*b, kmax, &mut by_rung, &mut bucket_pos, &mut hot);
        }
        eval_forces_with(b, &active, tree.as_ref());
        for &iu in &active {
            let i = iu as usize;
            let r = b.rung[i].min(kmax);
            let h = dt / (1u32 << r) as f64;
            b.vx[i] += 0.5 * b.ax[i] * h;
            b.vy[i] += 0.5 * b.ay[i] * h;
            if tick < ticks {
                let mut k = rung_for(b.dtl[i], dt).min(kmax);
                while k < r && tick % (1u32 << (kmax - k)) != 0 {
                    k += 1;
                }
                let old_r = r as usize;
                let new_r = k as usize;
                if new_r != old_r {
                    move_rung(iu, old_r, new_r, &mut by_rung, &mut bucket_pos);
                    match (old_r == 0, new_r == 0) {
                        (true, false) => match hot.binary_search(&i) {
                            Ok(_) => {}
                            Err(pos) => hot.insert(pos, i),
                        },
                        (false, true) => {
                            if let Ok(pos) = hot.binary_search(&i) {
                                hot.remove(pos);
                            }
                        }
                        _ => {}
                    }
                }
                b.rung[i] = k;
                let h2 = dt / (1u32 << k) as f64;
                b.vx[i] += 0.5 * b.ax[i] * h2;
                b.vy[i] += 0.5 * b.ay[i] * h2;
            }
        }
        age_flashes(b, dtf);
        let stats_before = (
            b.stats.merges,
            b.stats.impacts,
            b.stats.tidal,
            b.stats.coalesces,
            b.stats.bounces,
        );
        if !hot.is_empty() {
            collision_pass(b, dtf, Some(&hot));
            settle_energy(b);
        }
        let stats_after = (
            b.stats.merges,
            b.stats.impacts,
            b.stats.tidal,
            b.stats.coalesces,
            b.stats.bounces,
        );
        if b.len() != tree_len || stats_before != stats_after {
            tree = if b.len() >= BH_THRESHOLD {
                Some(Tree::build(&*b))
            } else {
                None
            };
            tree_len = b.len();
            rebuild_rung_lists(&*b, kmax, &mut by_rung, &mut bucket_pos, &mut hot);
        }
        b.time += dtf;
    }
}

/// Advance by `dt`: collisions, debris-disk coalescence, tidal break-up and
/// escape culling included.
pub fn advance(b: &mut Bodies, dt: f64) {
    let n = b.len();
    if n == 0 {
        return;
    }
    // Parent relationships change slowly (only when a body moves to a
    // different Hill sphere). Refreshing every 100 steps amortizes the
    // O(n^2) cost to a few ms per second even at n=1000.
    if !b.parent_cache_valid || b.stats.steps.saturating_sub(b.parent_cache_step) >= 100 {
        b.refresh_parent_cache();
    }
    let kmax = assign_rungs(b, dt);
    b.stats.max_rung = kmax;
    if kmax == 0 && n < YOSHIDA_MAX_N {
        yoshida_step(b, dt);
        b.stats.last_nsub = 1;
    } else {
        step_block(b, dt, kmax);
        b.stats.last_nsub = 1u32 << kmax;
    }
    evolve_stars(b, dt);
    dust::evolve(b, dt);
    coalesce_debris(b);
    tidal_breakup(b);
    cull_escaped(b);
    settle_energy(b);
    b.stats.steps += 1;
}

// ---------------------------------------------------------------------------
// Energy and momentum bookkeeping
// ---------------------------------------------------------------------------

pub fn total_energy(b: &Bodies) -> Option<f64> {
    let n = b.len();
    if n > ENERGY_MAX_N {
        return None;
    }
    let ke: f64 = (0..n)
        .map(|i| 0.5 * b.m[i] * (b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i]))
        .sum();
    let pe: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut s = 0.0;
            for j in (i + 1)..n {
                let dx = b.x[j] - b.x[i];
                let dy = b.y[j] - b.y[i];
                s -= G * b.m[i] * b.m[j] / (dx * dx + dy * dy + SOFT2).sqrt();
            }
            s
        })
        .sum();
    Some(ke + pe)
}

pub fn total_angular_momentum(b: &Bodies) -> f64 {
    let n = b.len();
    if n == 0 {
        return 0.0;
    }
    let (mut mt, mut cx, mut cy, mut vx, mut vy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        mt += b.m[i];
        cx += b.m[i] * b.x[i];
        cy += b.m[i] * b.y[i];
        vx += b.m[i] * b.vx[i];
        vy += b.m[i] * b.vy[i];
    }
    if mt <= 0.0 {
        return 0.0;
    }
    cx /= mt;
    cy /= mt;
    vx /= mt;
    vy /= mt;
    let mut l = 0.0;
    for i in 0..n {
        l += b.m[i] * ((b.x[i] - cx) * (b.vy[i] - vy) - (b.y[i] - cy) * (b.vx[i] - vx)) + b.spin[i];
    }
    l
}

fn mark_energy(b: &mut Bodies) {
    if b.e_pre.is_none() {
        b.e_pre = total_energy(b);
    }
}

fn settle_energy(b: &mut Bodies) {
    if let Some(e0) = b.e_pre.take() {
        if let Some(e1) = total_energy(b) {
            b.stats.energy_offset += e1 - e0;
        }
    }
}

pub fn rebase_energy(b: &mut Bodies) {
    b.stats.energy0 = total_energy(b).unwrap_or(0.0);
    b.stats.energy_offset = 0.0;
    b.e_pre = None;
}

pub fn energy_drift_pct(b: &Bodies) -> Option<f64> {
    let e = total_energy(b)?;
    let e0 = b.stats.energy0;
    if e0 == 0.0 {
        return None;
    }
    Some(((e - b.stats.energy_offset - e0) / e0.abs()) * 100.0)
}

pub fn energy_dissipated_pct(b: &Bodies) -> Option<f64> {
    let e0 = b.stats.energy0;
    if e0 == 0.0 || b.len() > ENERGY_MAX_N {
        return None;
    }
    Some(b.stats.energy_offset / e0.abs() * 100.0)
}

// ---------------------------------------------------------------------------
// Collisions
// ---------------------------------------------------------------------------

fn flash_at(b: &mut Bodies, x: f64, y: f64, r0: f64) {
    b.flashes.push(Flash {
        x,
        y,
        r0,
        age: 0.0,
        life: FLASH_LIFE,
    });
}

fn impact_offset(b: &Bodies, i: usize, j: usize) -> f64 {
    let dx = b.x[j] - b.x[i];
    let dy = b.y[j] - b.y[i];
    let vx = b.vx[j] - b.vx[i];
    let vy = b.vy[j] - b.vy[i];
    let v = (vx * vx + vy * vy).sqrt();
    if v <= 0.0 {
        return 0.0;
    }
    (dx * vy - dy * vx).abs() / v
}

fn pair_hit_d(b: &Bodies, i: usize, j: usize, sweep: bool) -> Option<f64> {
    let rr = b.contact_r(i) + b.contact_r(j);
    let rr2 = rr * rr;
    let dx = b.x[j] - b.x[i];
    let dy = b.y[j] - b.y[i];
    if dx * dx + dy * dy < rr2 {
        return Some(impact_offset(b, i, j));
    }
    if !sweep {
        return None;
    }
    let x0 = b._px[j] - b._px[i];
    let y0 = b._py[j] - b._py[i];
    let ex = dx - x0;
    let ey = dy - y0;
    let e2 = ex * ex + ey * ey;
    if e2 <= 0.0 || e2.is_nan() {
        return None;
    }
    let t = -(x0 * ex + y0 * ey) / e2;
    if t <= 0.0 || t >= 1.0 {
        return None;
    }
    let cx = x0 + t * ex;
    let cy = y0 + t * ey;
    if cx * cx + cy * cy < rr2 {
        Some((cx * cx + cy * cy).sqrt())
    } else {
        None
    }
}

fn pair_hit(b: &Bodies, i: usize, j: usize, sweep: bool) -> bool {
    pair_hit_d(b, i, j, sweep).is_some()
}

#[inline]
fn cell_key(cx: i64, cy: i64) -> u64 {
    (cx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (cy as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
}

fn find_collisions(b: &Bodies, sweep: bool, dt: f64) -> Vec<(usize, usize)> {
    let n = b.len();
    let t_now = b.time;
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    if n < HASH_THRESHOLD {
        for i in 0..n {
            if b.dead[i] || t_now < b.imm[i] {
                continue;
            }
            for j in (i + 1)..n {
                if b.dead[j] || t_now < b.imm[j] {
                    continue;
                }
                if pair_hit(b, i, j, sweep) {
                    pairs.push((i, j));
                }
            }
        }
        return pairs;
    }

    let mut max_cr: f64 = 0.0;
    let mut vmax2: f64 = 0.0;
    for i in 0..n {
        if b.dead[i] {
            continue;
        }
        max_cr = max_cr.max(b.contact_r(i));
        vmax2 = vmax2.max(b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i]);
    }
    let motion = if sweep {
        2.0 * vmax2.sqrt() * dt * 1.05
    } else {
        0.0
    };
    let cell = (2.0 * max_cr + motion).max(1e-4);

    let mut keys: Vec<(u64, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        if b.dead[i] {
            continue;
        }
        let cx = (b.x[i] / cell).floor() as i64;
        let cy = (b.y[i] / cell).floor() as i64;
        keys.push((cell_key(cx, cy), i as u32));
    }
    keys.par_sort_unstable();

    for i in 0..n {
        if b.dead[i] || t_now < b.imm[i] {
            continue;
        }
        let cx = (b.x[i] / cell).floor() as i64;
        let cy = (b.y[i] / cell).floor() as i64;
        for ddx in -1..=1i64 {
            for ddy in -1..=1i64 {
                let key = cell_key(cx + ddx, cy + ddy);
                let start = keys.partition_point(|e| e.0 < key);
                for e in &keys[start..] {
                    if e.0 != key {
                        break;
                    }
                    let j = e.1 as usize;
                    if j <= i || b.dead[j] || t_now < b.imm[j] {
                        continue;
                    }
                    if pair_hit(b, i, j, sweep) {
                        pairs.push((i, j));
                    }
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

fn find_collisions_hot(b: &Bodies, hot: &[usize], sweep: bool, dt: f64) -> Vec<(usize, usize)> {
    let _ = dt;
    if hot.len() * hot.len() > 4_000_000 {
        return find_collisions(b, sweep, dt);
    }
    let t_now = b.time;
    // Only live, non-immune hot bodies can take part. `hot` is ascending, so
    // `live` is ascending too.
    let live: Vec<usize> = hot
        .iter()
        .copied()
        .filter(|&i| !b.dead[i] && t_now >= b.imm[i])
        .collect();
    let nl = live.len();
    let mut pairs = Vec::new();
    if nl < HOT_GRID_MIN {
        for (a, &i) in live.iter().enumerate() {
            for &j in &live[a + 1..] {
                if pair_hit(b, i, j, sweep) {
                    pairs.push((i, j));
                }
            }
        }
        return pairs;
    }

    // Spatial hash. A pair can only hit if the centres are within
    // 2 * max_contact_r (static overlap) or within that plus both bodies'
    // displacement this tick (swept test), so the cell edge covers both.
    let mut max_cr = 0.0f64;
    let mut max_disp = 0.0f64;
    for &i in &live {
        max_cr = max_cr.max(b.contact_r(i));
        if sweep {
            max_disp = max_disp.max((b.x[i] - b._px[i]).hypot(b.y[i] - b._py[i]));
        }
    }
    let cell = ((2.0 * max_cr + 2.0 * max_disp) * 1.0001).max(1e-4);
    let inv = 1.0 / cell;

    let table = (2 * nl).next_power_of_two().max(16);
    let shift = 64 - table.trailing_zeros();
    let bucket = |cx: i64, cy: i64| (cell_key(cx, cy) >> shift) as usize;

    let cells: Vec<(i64, i64)> = live
        .iter()
        .map(|&i| ((b.x[i] * inv).floor() as i64, (b.y[i] * inv).floor() as i64))
        .collect();
    let mut start = vec![0u32; table + 1];
    for &(cx, cy) in &cells {
        start[bucket(cx, cy) + 1] += 1;
    }
    for k in 0..table {
        start[k + 1] += start[k];
    }
    let mut fill: Vec<u32> = start[..table].to_vec();
    let mut items = vec![0u32; nl];
    for (p, &(cx, cy)) in cells.iter().enumerate() {
        let k = bucket(cx, cy);
        items[fill[k] as usize] = p as u32;
        fill[k] += 1;
    }

    for (p, &i) in live.iter().enumerate() {
        let (cx, cy) = cells[p];
        let mut seen = [usize::MAX; 9];
        let mut ns = 0;
        for ddx in -1..=1i64 {
            for ddy in -1..=1i64 {
                let k = bucket(cx + ddx, cy + ddy);
                if seen[..ns].contains(&k) {
                    continue;
                }
                seen[ns] = k;
                ns += 1;
                for &q in &items[start[k] as usize..start[k + 1] as usize] {
                    let q = q as usize;
                    if q <= p {
                        continue;
                    }
                    let j = live[q];
                    if pair_hit(b, i, j, sweep) {
                        pairs.push((i, j));
                    }
                }
            }
        }
    }
    // Same order as the brute-force loop: ascending i, then ascending j.
    pairs.sort_unstable();
    pairs
}

fn collision_pass(b: &mut Bodies, dt: f64, hot: Option<&[usize]>) {
    if b.len() < 2 {
        return;
    }
    // One swept pass, no cascades. Bodies involved in a collision are put
    // under IMMUNITY_TIME (0.4 sim units >> DT = 1/480), so a second pass
    // within the same call always finds zero new pairs — the cascade the
    // old 3-iteration loop was guarding against cannot occur. Verified by
    // `c_moon_forming_impact_makes_a_disk`, which requires exactly one
    // impact per call and still produces ≥4 bodies after this change.
    let mut spawns: Vec<Spawn> = Vec::new();
    let pairs = match hot {
        Some(h) => find_collisions_hot(b, h, true, dt),
        None => find_collisions(b, true, dt),
    };
    for (i, j) in pairs {
        if b.dead[i] || b.dead[j] {
            continue;
        }
        if !pair_hit(b, i, j, true) {
            continue;
        }
        collide(b, i, j, &mut spawns);
    }
    if b.any_dead() {
        b.compact();
    }
    let t_imm = b.time + IMMUNITY_TIME;
    for s in spawns {
        let k = b.push(s.x, s.y, s.vx, s.vy, s.m, s.kind);
        b.imm[k] = t_imm;
        b.origin[k] = s.origin;
    }
}

// ---------------------------------------------------------------------------
// Impact model helpers
// ---------------------------------------------------------------------------

#[inline]
fn l_orb(b: &Bodies, i: usize) -> f64 {
    b.m[i] * (b.x[i] * b.vy[i] - b.y[i] * b.vx[i])
}

fn pair_l(b: &Bodies, i: usize, j: usize) -> f64 {
    l_orb(b, i) + l_orb(b, j) + b.spin[i] + b.spin[j]
}

fn rho1_sim() -> f64 {
    1.0 / (EARTH_RHO_GCC * 4.0 / 3.0 * std::f64::consts::PI * R_EARTH_GM.powi(3))
}

fn interact_frac(rt: f64, rp: f64, bb: f64) -> f64 {
    let l = (rt + rp) * (1.0 - bb);
    if l >= 2.0 * rp {
        1.0
    } else {
        ((3.0 * rp * l * l - l * l * l) / (4.0 * rp * rp * rp)).clamp(0.0, 1.0)
    }
}

fn q_star_rd(m_tot: f64, gamma: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let rho1 = rho1_sim();
    let rc1 = (3.0 * m_tot / (4.0 * pi * rho1)).cbrt();
    let q11 = LS_C_STAR * 0.8 * pi * rho1 * G * rc1 * rc1;
    q11 * (gamma + 1.0).powi(2) / (4.0 * gamma)
}

fn frag_masses(rng: &mut XorShift, n: usize, md: f64) -> Vec<f64> {
    let beta = FRAG_SLOPE / 3.0;
    let hi = FRAG_DYN_RANGE.powf(-beta);
    let mut v: Vec<f64> = (0..n)
        .map(|_| (1.0 - rng.next_f64() * (1.0 - hi)).powf(-1.0 / beta))
        .collect();
    let sum: f64 = v.iter().sum();
    for m in v.iter_mut() {
        *m *= md / sum;
    }
    v
}

fn hit_and_run(b: &mut Bodies, big: usize, small: usize, rr: f64, bb: f64, ratio: f64) {
    mark_energy(b);
    let l_before = pair_l(b, big, small);
    let (dx, dy) = (b.x[small] - b.x[big], b.y[small] - b.y[big]);
    let (vx, vy) = (b.vx[small] - b.vx[big], b.vy[small] - b.vy[big]);
    let v = (vx * vx + vy * vy).sqrt().max(1e-12);
    let (ux, uy) = (vx / v, vy / v);
    let along = dx * ux + dy * uy;
    let (hx, hy) = (dx - along * ux, dy - along * uy);
    let d = (hx * hx + hy * hy).sqrt();
    let (hux, huy) = if d > 1e-12 {
        (hx / d, hy / d)
    } else {
        (-uy, ux)
    };
    let s = (rr * rr - d * d).max(0.0).sqrt();
    let (nx, ny) = ((d * hux - s * ux) / rr, (d * huy - s * uy) / rr);
    let vn = vx * nx + vy * ny;
    let (mb, ms) = (b.m[big], b.m[small]);
    let mtot = mb + ms;
    if vn < 0.0 {
        let dv = -(1.0 + HIT_RUN_RESTITUTION) * vn;
        b.vx[small] += mb / mtot * dv * nx;
        b.vy[small] += mb / mtot * dv * ny;
        b.vx[big] -= ms / mtot * dv * nx;
        b.vy[big] -= ms / mtot * dv * ny;
    }
    let alpha = interact_frac(b.contact_r(big), b.contact_r(small), bb);
    let mt = (HIT_RUN_LOSS * alpha * ms).min(0.5 * ms);
    if mt > 0.0 {
        let nb = mb + mt;
        let cx0 = (mb * b.x[big] + ms * b.x[small]) / mtot;
        let cy0 = (mb * b.y[big] + ms * b.y[small]) / mtot;
        let cx1 = (nb * b.x[big] + (ms - mt) * b.x[small]) / mtot;
        let cy1 = (nb * b.y[big] + (ms - mt) * b.y[small]) / mtot;
        for k in [big, small] {
            b.x[k] += cx0 - cx1;
            b.y[k] += cy0 - cy1;
            b._px[k] = b.x[k];
            b._py[k] = b.y[k];
        }
        b.vx[big] = (mb * b.vx[big] + mt * b.vx[small]) / nb;
        b.vy[big] = (mb * b.vy[big] + mt * b.vy[small]) / nb;
        b.set_mass(big, nb);
        b.set_mass(small, ms - mt);
    }
    let l_after = pair_l(b, big, small);
    b.spin[big] += l_before - l_after;
    let t_imm = b.time + IMMUNITY_TIME;
    b.imm[big] = t_imm;
    b.imm[small] = t_imm;
    let (cx, cy) = (
        b.x[big] + nx * b.contact_r(big),
        b.y[big] + ny * b.contact_r(big),
    );
    flash_at(b, cx, cy, (b.contact_r(big) * 2.0).max(0.05));
    b.stats.impacts += 1;
    let text = format!(
        "hit-and-run #{} × #{} at {:.1}× escape speed, b = {:.2}: {} moved to the heavy body",
        b.id[big],
        b.id[small],
        ratio,
        bb,
        mass_str(mt)
    );
    b.log(EventKind::Impact, text, cx, cy);
}

fn rubble_bounce(b: &mut Bodies, i: usize, j: usize, rr: f64) {
    let dx = b.x[j] - b.x[i];
    let dy = b.y[j] - b.y[i];
    let dvx = b.vx[j] - b.vx[i];
    let dvy = b.vy[j] - b.vy[i];
    let v = (dvx * dvx + dvy * dvy).sqrt().max(1e-12);
    let (ux, uy) = (dvx / v, dvy / v);

    let along = dx * ux + dy * uy;
    let (hx, hy) = (dx - along * ux, dy - along * uy);
    let d = (hx * hx + hy * hy).sqrt();
    let (hux, huy) = if d > 1e-12 {
        (hx / d, hy / d)
    } else {
        (-uy, ux)
    };
    let s = (rr * rr - d * d).max(0.0).sqrt();
    let (nx, ny) = ((d * hux - s * ux) / rr, (d * huy - s * uy) / rr);
    let vn = dvx * nx + dvy * ny;
    if vn >= 0.0 {
        return;
    }

    mark_energy(b);
    let l_before = pair_l(b, i, j);
    let (mi, mj) = (b.m[i], b.m[j]);
    let mtot = mi + mj;
    let dvn = -(1.0 + RUBBLE_BOUNCE_E) * vn;
    b.vx[i] -= mj / mtot * dvn * nx;
    b.vy[i] -= mj / mtot * dvn * ny;
    b.vx[j] += mi / mtot * dvn * nx;
    b.vy[j] += mi / mtot * dvn * ny;

    let dl = l_before - pair_l(b, i, j);
    b.spin[i] += dl * mi / mtot;
    b.spin[j] += dl * mj / mtot;

    let t_imm = b.time + RUBBLE_BOUNCE_IMMUNITY;
    b.imm[i] = t_imm;
    b.imm[j] = t_imm;

    b.stats.bounces += 1;
}

fn pick_survivor(b: &Bodies, i: usize, j: usize) -> (usize, usize) {
    let rank = |k: usize| match b.kind[k] {
        BodyKind::BlackHole => 6,
        BodyKind::NeutronStar => 5,
        BodyKind::WhiteDwarf => 4,
        BodyKind::Star => 3,
        BodyKind::BrownDwarf => 2,
        BodyKind::Planet => 1,
        BodyKind::DwarfPlanet => 0,
    };
    let (ri, rj) = (rank(i), rank(j));
    if ri != rj {
        return if ri > rj { (i, j) } else { (j, i) };
    }
    if b.m[i] >= b.m[j] {
        (i, j)
    } else {
        (j, i)
    }
}

fn room(b: &Bodies, pending: usize) -> usize {
    MAX_BODIES.saturating_sub(b.len() + pending)
}

fn combine_origin(keep: BodyOrigin, kill: BodyOrigin) -> BodyOrigin {
    // Two debris bodies stay debris; two belt bodies stay in the belt.
    if (keep.is_rubble() && kill.is_rubble()) || (keep.is_belt() && kill.is_belt()) {
        keep
    } else {
        BodyOrigin::Spawned
    }
}

fn merge_impl(b: &mut Bodies, i: usize, j: usize, silent: bool) -> usize {
    mark_energy(b);
    let (keep, kill) = pick_survivor(b, i, j);
    let total = b.m[keep] + b.m[kill];
    let cx = (b.m[keep] * b.x[keep] + b.m[kill] * b.x[kill]) / total;
    let cy = (b.m[keep] * b.y[keep] + b.m[kill] * b.y[kill]) / total;
    let cvx = (b.m[keep] * b.vx[keep] + b.m[kill] * b.vx[kill]) / total;
    let cvy = (b.m[keep] * b.vy[keep] + b.m[kill] * b.vy[kill]) / total;

    let mu = b.m[keep] * b.m[kill] / total;
    let rl = (b.x[kill] - b.x[keep]) * (b.vy[kill] - b.vy[keep])
        - (b.y[kill] - b.y[keep]) * (b.vx[kill] - b.vx[keep]);
    b.spin[keep] += b.spin[kill] + mu * rl;

    let verb = if b.kind[keep] != BodyKind::Planet && b.kind[kill] == BodyKind::Planet {
        "fell into"
    } else {
        "merged into"
    };
    let text = format!("{} {} {}", tag(b, kill), verb, tag(b, keep));

    b.x[keep] = cx;
    b.y[keep] = cy;
    b.vx[keep] = cvx;
    b.vy[keep] = cvy;
    b._px[keep] = cx;
    b._py[keep] = cy;
    // Evolution state of the survivor, before the radius is recomputed.
    let keep_was_star = b.kind[keep] == BodyKind::Star;
    let keep_kind_before = b.kind[keep];
    let kill_kind = b.kind[kill];
    // Preserve a mass-weighted fractional MS age. For a star, this is the
    // fraction of its MS lifetime that it has used. For a dwarf, this is
    // still numerically valid (it's just a proxy for "fraction cooled"),
    // and the kind-specific block below overwrites it with the proper
    // dwarf age once the survivor's final kind is known.
    let frac_age = |k: usize| -> f64 {
        let t = star_lifetime_gyr(BodyKind::Star, b.m_init[k]);
        if t.is_finite() && t > 0.0 {
            (b.age_gyr[k] / t).clamp(0.0, 0.99)
        } else {
            0.0
        }
    };
    let f = (frac_age(keep) * b.m[keep] + frac_age(kill) * b.m[kill]) / total;
    b.m_init[keep] += b.m_init[kill];
    b.age_gyr[keep] = f * star_lifetime_gyr(BodyKind::Star, b.m_init[keep]);
    if matches!(b.kind[keep], BodyKind::Planet | BodyKind::DwarfPlanet) {
        let rho = (b.m[keep] * b.bulk_density[keep] + b.m[kill] * b.bulk_density[kill]) / total;
        b.bulk_density[keep] = rho;
    }
    b.set_mass(keep, total);
    b.origin[keep] = combine_origin(b.origin[keep], b.origin[kill]);

    if b.origin[keep].is_rubble() && b.m[keep] >= RUBBLE_PROMOTE_MASS {
        b.origin[keep] = BodyOrigin::Spawned;
        let text = format!("{} grew to planet mass", tag(b, keep));
        b.log(EventKind::Merge, text, cx, cy);
    }

    if b.origin[keep].is_belt() && b.kind[keep] == BodyKind::Planet && b.m[keep] >= BELT_DWARF_M_MIN
    {
        b.kind[keep] = BodyKind::DwarfPlanet;
    }

    if b.kind[keep] == BodyKind::WhiteDwarf && b.m[keep] > WD_M_MAX {
        b.kind[keep] = BodyKind::NeutronStar;
        b.set_mass(keep, NS_M_MIN * 1.07);
        let text = format!("{} collapsed to neutron star", tag(b, keep));
        b.log(EventKind::Merge, text, cx, cy);
    }
    if b.kind[keep] == BodyKind::Star && b.m[keep] < STAR_M_MIN {
        b.kind[keep] = BodyKind::BrownDwarf;
        b.set_mass(keep, b.m[keep].max(BD_M_MIN));
    }
    if b.kind[keep] == BodyKind::BrownDwarf && b.m[keep] < BD_M_MIN {
        b.kind[keep] = BodyKind::Planet;
        b.bulk_density[keep] = default_bulk_density(BodyKind::Planet, b.m[keep]);
    }
    if b.kind[keep] == BodyKind::BrownDwarf && b.m[keep] >= STAR_M_MIN {
        b.kind[keep] = BodyKind::Star;
        let text = format!("{} ignited as a star", tag(b, keep));
        b.log(EventKind::Merge, text, cx, cy);
    }

    let kind_after = b.kind[keep];
    if kind_after == BodyKind::Star {
        if !keep_was_star {
            // A brown dwarf that ignited starts as a new zero-age star.
            b.m_init[keep] = b.m[keep];
            b.age_gyr[keep] = 0.0;
        }
    } else {
        b.m_init[keep] = b.m[keep];
        let same_kind = keep_kind_before == kind_after;
        // A dwarf that only swallows a planet or a star keeps its cooling age.
        // A merger of two dwarfs, or a new kind of object, starts again.
        let swallowed_other = same_kind
            && kill_kind != kind_after
            && matches!(kind_after, BodyKind::WhiteDwarf | BodyKind::BrownDwarf);
        if !swallowed_other {
            b.age_gyr[keep] = if same_kind {
                0.0
            } else {
                default_age_gyr(kind_after)
            };
        }
    }
    b.r[keep] = b.radius_now(keep);

    b.dead[kill] = true;
    b.coalesce_counter[keep] = 0;

    b.stats.merges += 1;
    if !silent {
        let flash_r = (b.contact_r(keep) * 5.0).max(0.05);
        flash_at(b, cx, cy, flash_r);
        b.log(EventKind::Merge, text, cx, cy);
    }
    keep
}

fn merge(b: &mut Bodies, i: usize, j: usize) -> usize {
    merge_impl(b, i, j, false)
}

fn collide(b: &mut Bodies, i: usize, j: usize, spawns: &mut Vec<Spawn>) {
    if b.kind[i] != BodyKind::Planet || b.kind[j] != BodyKind::Planet {
        merge(b, i, j);
        return;
    }
    let (rvx, rvy) = (b.vx[j] - b.vx[i], b.vy[j] - b.vy[i]);
    let vrel = (rvx * rvx + rvy * rvy).sqrt();
    let rr = (b.contact_r(i) + b.contact_r(j)).max(1e-6);
    let mtot = b.m[i] + b.m[j];
    let ratio = vrel / (2.0 * G * mtot / rr).sqrt();
    let bb = (impact_offset(b, i, j) / rr).clamp(0.0, 1.0);
    let (big, small) = if b.m[i] >= b.m[j] { (i, j) } else { (j, i) };

    if b.origin[i].is_rubble() && b.origin[j].is_rubble() {
        if ratio < RUBBLE_COALESCE_VRATIO {
            merge(b, i, j);
            b.stats.coalesces += 1;
            return;
        }
        if ratio < RUBBLE_BOUNCE_VRATIO {
            rubble_bounce(b, i, j, rr);
            return;
        }
    }

    if bb > HIT_RUN_B && ratio > HIT_RUN_VEL_RATIO {
        hit_and_run(b, big, small, rr, bb, ratio);
        return;
    }
    if ratio < MERGE_VEL_RATIO * (1.0 - MERGE_B_WEIGHT * bb * bb) {
        merge(b, i, j);
        return;
    }

    let alpha = interact_frac(b.contact_r(big), b.contact_r(small), bb).max(1e-3);
    let m_int = alpha * b.m[small];
    let mu_a = b.m[big] * m_int / (b.m[big] + m_int);
    let x = 0.5 * mu_a * vrel * vrel / mtot / q_star_rd(mtot, b.m[small] / b.m[big]);
    let lr = if x <= 1.8 {
        1.0 - 0.5 * x
    } else {
        0.1 * (x / 1.8).powf(-1.5)
    };
    let m_erode = mtot * (1.0 - lr);
    let m_disk = if b.len() <= DISK_MAX_N {
        DISK_MASS_FRAC * m_int * 4.0 * bb * bb * (1.0 - bb * bb)
    } else {
        0.0
    };
    let cap = (DEBRIS_MAX_FRAC_HEADON + (DEBRIS_MAX_FRAC - DEBRIS_MAX_FRAC_HEADON) * bb) * mtot;
    let md = (m_erode + m_disk).min(cap);
    let n_raw = FRAG_COUNT_BASE
        + FRAG_COUNT_SCALE * x.powf(FRAG_COUNT_EXP)
        + FRAG_COUNT_DISK_GAIN * (m_disk / DEBRIS_MIN_MASS).ln_1p();
    let n = (n_raw.round() as usize)
        .min(FRAG_COUNT_MAX)
        .min((md / DEBRIS_MIN_MASS) as usize)
        .min(room(b, spawns.len()) + 1);
    if md < DEBRIS_MIN_MASS || n < 2 {
        merge(b, i, j);
        return;
    }

    let l_before = pair_l(b, i, j);
    let (px0, py0) = (b.x[j] - b.x[i], b.y[j] - b.y[i]);
    let sense = if px0 * rvy - py0 * rvx >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let idg = (b.id[i], b.id[j]);
    let rem = merge(b, i, j);
    b.events.pop_back();
    b.stats.merges -= 1;
    let m_rem = mtot - md;
    b.set_mass(rem, m_rem);
    b.imm[rem] = b.time + IMMUNITY_TIME;
    b.origin[rem] = BodyOrigin::Spawned;

    let ms = frag_masses(&mut b.rng, n, md);
    // Dust source term: a bonus population, see `DUST_YIELD_FRAC`.
    let (dust_x, dust_y) = (b.x[rem], b.y[rem]);
    let (dust_vx, dust_vy) = (b.vx[rem], b.vy[rem]);
    dust::deposit(b, dust_x, dust_y, dust_vx, dust_vy, md * DUST_YIELD_FRAC);
    let r_rem = b.contact_r(rem);
    let v_esc = (2.0 * G * m_rem / r_rem.max(1e-6)).sqrt();
    let disk_share = (m_disk / md).clamp(0.0, 1.0);
    let base = b.rng.next_f64() * std::f64::consts::TAU;
    let (rx, ry, rvx0, rvy0) = (b.x[rem], b.y[rem], b.vx[rem], b.vy[rem]);
    let s0 = spawns.len();
    let (mut px, mut py, mut sx, mut sy) = (0.0, 0.0, 0.0, 0.0);
    for (k, &mk) in ms.iter().enumerate() {
        let is_disk = b.rng.next_f64() < disk_share;
        let ang = base
            + (k as f64) * std::f64::consts::TAU / n as f64
            + (b.rng.next_f64() - 0.5) * DEBRIS_ANGLE_JIT;
        let (ux, uy) = (ang.cos(), ang.sin());
        let (dist, wx, wy);
        if is_disk {
            dist = r_rem * (DISK_R_MIN + DISK_R_SPAN * b.rng.next_f64()) + planet_true_r(mk);
            let vc = (G * m_rem / dist).sqrt() * (0.9 + 0.2 * b.rng.next_f64());
            wx = -sense * uy * vc;
            wy = sense * ux * vc;
        } else {
            dist = (r_rem + planet_true_r(mk)) * (1.0 + DEBRIS_GAP_FRAC);
            let kick = v_esc * (DEBRIS_KICK_MIN + DEBRIS_KICK_SPAN * b.rng.next_f64());
            wx = ux * kick;
            wy = uy * kick;
        }
        px += mk * wx;
        py += mk * wy;
        sx += mk * ux * dist;
        sy += mk * uy * dist;
        spawns.push(Spawn {
            x: rx + ux * dist,
            y: ry + uy * dist,
            vx: rvx0 + wx,
            vy: rvy0 + wy,
            m: mk,
            kind: BodyKind::Planet,
            origin: BodyOrigin::ImpactDebris,
        });
    }
    b.x[rem] -= sx / m_rem;
    b.y[rem] -= sy / m_rem;
    b._px[rem] = b.x[rem];
    b._py[rem] = b.y[rem];
    b.vx[rem] -= px / m_rem;
    b.vy[rem] -= py / m_rem;

    let mut l_after = l_orb(b, rem);
    for s in &spawns[s0..] {
        l_after += s.m * (s.x * s.vy - s.y * s.vx);
    }
    b.spin[rem] = l_before - l_after;

    b.stats.impacts += 1;
    let text = format!(
        "impact #{} × #{} at {:.1}× escape speed, b = {:.2}: {} fragments, {} ejected{}",
        idg.0,
        idg.1,
        ratio,
        bb,
        n,
        mass_str(md),
        if m_disk > m_erode { " (disk)" } else { "" }
    );
    b.log(EventKind::Impact, text, rx, ry);
}

// ---------------------------------------------------------------------------
// Debris-disk coalescence
// ---------------------------------------------------------------------------

// ---- Fix 8: count eligible first, exit before the O(n) host scan and the
// ---- n-sized Vec. Hoist `m_p` and `d_i` out of the candidate loop.
fn coalesce_debris(b: &mut Bodies) {
    let n = b.len();
    if n < 2 {
        return;
    }
    let t_now = b.time;

    // Count eligible bodies first. The common case (few or zero fragments)
    // exits without allocating the per-body eligibility map or scanning
    // the host list.
    let mut n_eligible = 0usize;
    for i in 0..n {
        if !b.dead[i]
            && b.origin[i].is_rubble()
            && b.m[i] < DEBRIS_COALESCE_MAX_MASS
            && t_now >= b.imm[i]
        {
            n_eligible += 1;
        }
    }
    if n_eligible < 2 {
        // Reset the counter on every live rubble body — the same set
        // the old `body_to_eligible` + `eligible` loop reached.
        for i in 0..n {
            if !b.dead[i] && b.origin[i].is_rubble() {
                b.coalesce_counter[i] = 0;
            }
        }
        return;
    }

    let hosts = host_list(b);
    let mut eligible: Vec<usize> = Vec::with_capacity(n_eligible);
    for i in 0..n {
        if !b.dead[i]
            && b.origin[i].is_rubble()
            && b.m[i] < DEBRIS_COALESCE_MAX_MASS
            && t_now >= b.imm[i]
        {
            eligible.push(i);
        }
    }

    let mut body_to_eligible: Vec<Option<usize>> = vec![None; n];
    for (pos, &i) in eligible.iter().enumerate() {
        body_to_eligible[i] = Some(pos);
    }

    for (i, slot) in body_to_eligible.iter().enumerate() {
        if !b.dead[i] && b.origin[i].is_rubble() && slot.is_none() {
            b.coalesce_counter[i] = 0;
        }
    }

    let parents: Vec<Option<usize>> = eligible
        .iter()
        .map(|&i| b.parent_cache.get(i).copied().flatten())
        .collect();

    let max_cr = eligible
        .iter()
        .map(|&i| b.contact_r(i))
        .fold(0.0f64, f64::max);
    let cell = (DEBRIS_SEARCH_RADII * max_cr).max(1e-4);

    let mut keys: Vec<(u64, u32)> = Vec::with_capacity(eligible.len());
    for (pos, &i) in eligible.iter().enumerate() {
        if parents[pos].is_none() {
            continue;
        }
        let cx = (b.x[i] / cell).floor() as i64;
        let cy = (b.y[i] / cell).floor() as i64;
        keys.push((cell_key(cx, cy), i as u32));
    }
    keys.par_sort_unstable();

    let mut in_capture = vec![false; n];
    let mut pairs: Vec<(usize, usize)> = Vec::new();

    for (pos_i, &i) in eligible.iter().enumerate() {
        let Some(pi) = parents[pos_i] else { continue };
        // Hoisted: constant across all candidate j for this i.
        let m_p = effective_host_mass(b, &hosts, i, pi);
        let d_i = ((b.x[i] - b.x[pi]).powi(2) + (b.y[i] - b.y[pi]).powi(2)).sqrt();
        let cx = (b.x[i] / cell).floor() as i64;
        let cy = (b.y[i] / cell).floor() as i64;

        for ddx in -1..=1i64 {
            for ddy in -1..=1i64 {
                let key = cell_key(cx + ddx, cy + ddy);
                let start = keys.partition_point(|e| e.0 < key);
                for e in &keys[start..] {
                    if e.0 != key {
                        break;
                    }
                    let j = e.1 as usize;
                    if j <= i {
                        continue;
                    }
                    let Some(pos_j) = body_to_eligible[j] else {
                        continue;
                    };
                    let Some(pj) = parents[pos_j] else { continue };
                    if pi != pj {
                        continue;
                    }

                    let dx = b.x[j] - b.x[i];
                    let dy = b.y[j] - b.y[i];
                    let dist2 = dx * dx + dy * dy;
                    let dist = dist2.sqrt();

                    let d_j = ((b.x[j] - b.x[pi]).powi(2) + (b.y[j] - b.y[pi]).powi(2)).sqrt();
                    let m_ij = b.m[i] + b.m[j];
                    let r_hill = (m_ij / (3.0 * m_p)).cbrt() * ((d_i + d_j) * 0.5);

                    if dist >= DEBRIS_CAPTURE_HILL_FRAC * r_hill {
                        continue;
                    }

                    let dvx = b.vx[j] - b.vx[i];
                    let dvy = b.vy[j] - b.vy[i];
                    let vrel = (dvx * dvx + dvy * dvy).sqrt();
                    let v_esc = (2.0 * G * m_ij / dist.max(1e-12)).sqrt();
                    if vrel >= DEBRIS_CAPTURE_VEL_RATIO * v_esc {
                        continue;
                    }

                    in_capture[i] = true;
                    in_capture[j] = true;
                    pairs.push((i, j));
                }
            }
        }
    }

    for &i in &eligible {
        if in_capture[i] {
            b.coalesce_counter[i] = b.coalesce_counter[i].saturating_add(1);
        } else {
            b.coalesce_counter[i] = 0;
        }
    }

    if pairs.is_empty() {
        return;
    }

    let to_merge: Vec<(usize, usize)> = pairs
        .into_iter()
        .filter(|&(i, j)| {
            b.coalesce_counter[i] >= DEBRIS_CAPTURE_STEPS
                && b.coalesce_counter[j] >= DEBRIS_CAPTURE_STEPS
        })
        .collect();

    let mut merged_any = false;
    for (i, j) in to_merge {
        if b.dead[i] || b.dead[j] {
            continue;
        }
        if b.coalesce_counter[i] < DEBRIS_CAPTURE_STEPS
            || b.coalesce_counter[j] < DEBRIS_CAPTURE_STEPS
        {
            continue;
        }
        merge_impl(b, i, j, /*silent=*/ true);
        b.stats.coalesces += 1;
        merged_any = true;
    }

    if merged_any {
        b.compact();
    }
}

// ---------------------------------------------------------------------------
// Roche limit and tidal break-up
// ---------------------------------------------------------------------------

pub fn density_gcc(b: &Bodies, i: usize) -> f64 {
    if matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
        b.bulk_density[i]
    } else {
        EARTH_RHO_GCC * b.m[i] * (R_EARTH_GM / b.r[i].max(1e-12)).powi(3)
    }
}

pub fn rigidity(rho_gcc: f64) -> f64 {
    ((rho_gcc - ROCHE_RHO_FLUID) / (ROCHE_RHO_RIGID - ROCHE_RHO_FLUID)).clamp(0.0, 1.0)
}

pub fn roche_limit(b: &Bodies, i: usize, p: usize) -> f64 {
    let w = rigidity(density_gcc(b, i));
    let coeff = ROCHE_FLUID * (1.0 - w) + ROCHE_RIGID * w;
    coeff * b.r[i] * (b.m[p] / b.m[i]).cbrt()
}

fn tidal_breakup(b: &mut Bodies) {
    let n = b.len();
    let prims: Vec<usize> = (0..n)
        .filter(|&p| {
            is_host_kind(b.kind[p])
                || (matches!(b.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet)
                    && b.m[p] >= GIANT_MASS)
        })
        .collect();
    if prims.is_empty() {
        b.disrupting.iter_mut().for_each(|f| *f = false);
        return;
    }
    let t_now = b.time;
    for i in (0..n).rev() {
        if b.dead[i]
            || !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
            || b.origin[i].is_rubble()
            || b.m[i] < TIDAL_MIN_MASS
            || t_now < b.imm[i]
        {
            continue;
        }
        let tb = b.r[i];
        let kf = ROCHE_FLUID * tb;
        let k2 = kf * kf;
        let k6 = k2 * k2 * k2;
        let mut inside = false;
        for &p in &prims {
            if p == i || b.dead[p] || b.m[p] < TIDAL_MASS_RATIO * b.m[i] {
                continue;
            }
            let dx = b.x[i] - b.x[p];
            let dy = b.y[i] - b.y[p];
            let d2 = dx * dx + dy * dy;
            let q = b.m[p] / b.m[i];
            if d2 * d2 * d2 >= k6 * q * q {
                continue;
            }
            let roche = roche_limit(b, i, p);
            if d2 < roche * roche {
                inside = true;
                b.disrupting[i] = true;
                let rdot = dx * (b.vx[i] - b.vx[p]) + dy * (b.vy[i] - b.vy[p]);
                if rdot >= 0.0 {
                    let _ = disrupt(b, i, p);
                }
                break;
            }
        }
        if !inside && !b.dead[i] {
            b.disrupting[i] = false;
        }
    }
    if b.any_dead() {
        b.compact();
    }
}

fn disrupt(b: &mut Bodies, i: usize, p: usize) -> bool {
    let mi = b.m[i];
    let n = ((DISRUPT_FRAG_BASE + DISRUPT_FRAG_LOG * (mi + 1.0).log10()).round() as usize)
        .clamp(4, 12)
        .min(room(b, 0) + 1);
    if n < 3 {
        return false;
    }
    mark_energy(b);
    let l_before = l_orb(b, i) + b.spin[i];
    let text = format!("{} torn apart by {}", tag(b, i), tag(b, p));
    let d = ((b.x[i] - b.x[p]).powi(2) + (b.y[i] - b.y[p]).powi(2))
        .sqrt()
        .max(1e-12);
    let (ux, uy) = ((b.x[i] - b.x[p]) / d, (b.y[i] - b.y[p]) / d);
    let (tx, ty) = (-uy, ux);

    let mut ms: Vec<f64> = (0..n)
        .map(|_| DEBRIS_WEIGHT_MIN + b.rng.next_f64())
        .collect();
    let wsum: f64 = ms.iter().sum();
    let mut rmax: f64 = 0.0;
    for m in ms.iter_mut() {
        *m = mi * *m / wsum;
        rmax = rmax.max(planet_true_r(*m));
    }
    let step = DISRUPT_STEP * rmax;
    let half = (((n - 1) as f64) / 2.0 * step).max(1e-6);
    let vk = DISRUPT_KICK * (G * mi / b.r[i].max(1e-6)).sqrt();
    let (bx, by, bvx, bvy) = (b.x[i], b.y[i], b.vx[i], b.vy[i]);
    let prim_r = b.contact_r(p);
    let (px0, py0) = (b.x[p], b.y[p]);

    b.dead[i] = true;
    let mut pieces: Vec<Spawn> = Vec::with_capacity(n);
    let (mut pvx, mut pvy, mut mt) = (0.0, 0.0, 0.0);
    for (k, &mk) in ms.iter().enumerate() {
        let s = (k as f64 - (n - 1) as f64 / 2.0) * step;
        let kick = vk * s / half;
        let mut fx = bx + ux * s + tx * (b.rng.next_f64() - DISRUPT_JITTER) * rmax;
        let mut fy = by + uy * s + ty * (b.rng.next_f64() - DISRUPT_JITTER) * rmax;
        let mut qx = fx - px0;
        let mut qy = fy - py0;
        let mut qd = (qx * qx + qy * qy).sqrt();
        let qmin = prim_r + 1.5 * planet_true_r(mk);
        if qd < qmin {
            if qd < 1e-9 {
                qx = ux;
                qy = uy;
                qd = 1.0;
            }
            fx = px0 + qx / qd * qmin;
            fy = py0 + qy / qd * qmin;
        }
        let vx = bvx + ux * kick;
        let vy = bvy + uy * kick;
        pvx += mk * (vx - bvx);
        pvy += mk * (vy - bvy);
        mt += mk;
        pieces.push(Spawn {
            x: fx,
            y: fy,
            vx,
            vy,
            m: mk,
            kind: BodyKind::Planet,
            origin: BodyOrigin::TidalDebris,
        });
    }
    let t_imm = b.time + IMMUNITY_TIME;
    let first = b.len();
    let mut l_after = 0.0;
    for s in pieces {
        let (vx, vy) = (s.vx - pvx / mt, s.vy - pvy / mt);
        let k = b.push(s.x, s.y, vx, vy, s.m, s.kind);
        b.imm[k] = t_imm;
        b.origin[k] = BodyOrigin::TidalDebris;
        l_after += s.m * (s.x * vy - s.y * vx);
    }
    let dl = l_before - l_after;
    for k in first..b.len() {
        b.spin[k] = dl * b.m[k] / mt;
    }
    flash_at(b, bx, by, (prim_r * 0.5).max(0.05));
    b.stats.tidal += 1;
    b.log(EventKind::Tidal, text, bx, by);
    true
}

fn cull_escaped(b: &mut Bodies) {
    let n = b.len();
    if n == 0 {
        return;
    }
    let mut prim_m = 0.0;
    for i in 0..n {
        if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
            prim_m += b.m[i];
        }
    }
    let cd2 = CULL_DIST * CULL_DIST;
    let hcd2 = HARD_CULL_DIST * HARD_CULL_DIST;
    let mut gone: Vec<usize> = Vec::new();
    for i in 0..n {
        if b.dead[i] || !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
            continue;
        }
        let d2 = b.x[i] * b.x[i] + b.y[i] * b.y[i];
        if d2 < cd2 {
            continue;
        }
        let v2 = b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i];
        if d2 > hcd2
            || (v2 > (2.0 * G * prim_m) / d2.sqrt() && b.x[i] * b.vx[i] + b.y[i] * b.vy[i] > 0.0)
        {
            gone.push(i);
        }
    }
    if gone.is_empty() {
        return;
    }
    mark_energy(b);
    for i in gone {
        let text = format!("{} escaped the system", tag(b, i));
        let (x, y) = (b.x[i], b.y[i]);
        b.dead[i] = true;
        b.stats.escaped += 1;
        b.log(EventKind::Escape, text, x, y);
    }
    b.compact();
}

// ---------------------------------------------------------------------------
// Parents, orbits and astrophysical helpers (used by the UI)
// ---------------------------------------------------------------------------

pub fn primary_index(b: &Bodies) -> Option<usize> {
    let mut best: Option<usize> = None;
    for i in 0..b.len() {
        if is_host_kind(b.kind[i]) && best.is_none_or(|k| b.m[i] > b.m[k]) {
            best = Some(i);
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Binary hosts
// ---------------------------------------------------------------------------

fn host_list(b: &Bodies) -> Vec<usize> {
    (0..b.len()).filter(|&k| is_host_kind(b.kind[k])).collect()
}

/// Nearest other host that `h` is gravitationally bound to, if any.
fn binary_partner_in(b: &Bodies, hosts: &[usize], h: usize) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for &k in hosts {
        if k == h {
            continue;
        }
        let (dx, dy) = (b.x[k] - b.x[h], b.y[k] - b.y[h]);
        let d = (dx * dx + dy * dy).sqrt().max(1e-12);
        let (dvx, dvy) = (b.vx[k] - b.vx[h], b.vy[k] - b.vy[h]);
        if dvx * dvx + dvy * dvy >= 2.0 * G * (b.m[h] + b.m[k]) / d {
            continue;
        }
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((k, d));
        }
    }
    best.map(|p| p.0)
}

/// Bound stellar companion of host `h`, if any.
pub fn binary_partner(b: &Bodies, h: usize) -> Option<usize> {
    binary_partner_in(b, &host_list(b), h)
}

/// Barycentre of a host binary, seen from a body that orbits the pair as a whole.
struct Barycentre {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    m: f64,
    /// The heavier star (lower index on a tie): the parent reported for
    /// circumbinary bodies, so every body in a circumbinary disk agrees.
    lead: usize,
}

/// If host `h` has a bound companion and body `i` is more than
/// `CIRCUMBINARY_FACTOR` separations from their barycentre, return that
/// barycentre.
fn circumbinary_frame(b: &Bodies, hosts: &[usize], h: usize, i: usize) -> Option<Barycentre> {
    let q = binary_partner_in(b, hosts, h)?;
    let sep = (b.x[q] - b.x[h]).hypot(b.y[q] - b.y[h]);
    let m = b.m[h] + b.m[q];
    let x = (b.m[h] * b.x[h] + b.m[q] * b.x[q]) / m;
    let y = (b.m[h] * b.y[h] + b.m[q] * b.y[q]) / m;
    if (b.x[i] - x).hypot(b.y[i] - y) <= CIRCUMBINARY_FACTOR * sep {
        return None;
    }
    let vx = (b.m[h] * b.vx[h] + b.m[q] * b.vx[q]) / m;
    let vy = (b.m[h] * b.vy[h] + b.m[q] * b.vy[q]) / m;
    let lead = if b.m[h] > b.m[q] || (b.m[h] == b.m[q] && h < q) {
        h
    } else {
        q
    };
    Some(Barycentre {
        x,
        y,
        vx,
        vy,
        m,
        lead,
    })
}

/// Same as `circumbinary_frame` but takes a precomputed partner table.
fn circumbinary_frame_cached(
    b: &Bodies,
    partners: &[Option<usize>],
    h: usize,
    i: usize,
) -> Option<Barycentre> {
    let q = partners.get(h).copied().flatten()?;
    let sep = (b.x[q] - b.x[h]).hypot(b.y[q] - b.y[h]);
    let m = b.m[h] + b.m[q];
    let x = (b.m[h] * b.x[h] + b.m[q] * b.x[q]) / m;
    let y = (b.m[h] * b.y[h] + b.m[q] * b.y[q]) / m;
    if (b.x[i] - x).hypot(b.y[i] - y) <= CIRCUMBINARY_FACTOR * sep {
        return None;
    }
    let vx = (b.m[h] * b.vx[h] + b.m[q] * b.vx[q]) / m;
    let vy = (b.m[h] * b.vy[h] + b.m[q] * b.vy[q]) / m;
    let lead = if b.m[h] > b.m[q] || (b.m[h] == b.m[q] && h < q) {
        h
    } else {
        q
    };
    Some(Barycentre {
        x,
        y,
        vx,
        vy,
        m,
        lead,
    })
}

/// Mass that governs the Hill sphere of body `i` about its parent host `p`:
/// the pair's total for circumbinary bodies, else the host's own mass.
fn effective_host_mass(b: &Bodies, hosts: &[usize], i: usize, p: usize) -> f64 {
    circumbinary_frame(b, hosts, p, i).map_or(b.m[p], |bc| bc.m)
}

/// Reference frame (x, y, vx, vy, mass) for drawing the orbit of body `i`:
/// the barycentre for circumbinary bodies, its host star for S-type bodies,
/// otherwise `default` (the primary, as before).
fn orbit_reference(
    b: &Bodies,
    hosts: &[usize],
    i: usize,
    default: (f64, f64, f64, f64, f64),
) -> (f64, f64, f64, f64, f64) {
    if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) || !b.parent_cache_valid {
        return default;
    }
    let p = match b.parent_cache.get(i).copied().flatten() {
        Some(p) => Some(p),
        None => {
            // Belt and rubble bodies have no `find_parent` entry (they are
            // deliberately excluded so they aren't labelled moons), but
            // they still orbit a host. Use the nearest one so multi-system
            // rings draw each body about its own system, not the heaviest
            // host globally.
            if b.origin[i].is_belt() || b.origin[i].is_rubble() {
                hosts.iter().copied().min_by(|&a, &c| {
                    let da = (b.x[a] - b.x[i]).powi(2) + (b.y[a] - b.y[i]).powi(2);
                    let dc = (b.x[c] - b.x[i]).powi(2) + (b.y[c] - b.y[i]).powi(2);
                    da.total_cmp(&dc)
                })
            } else {
                None
            }
        }
    };
    let Some(p) = p else {
        return default;
    };
    if p >= b.len() {
        return default;
    }
    // NEW: moons draw their orbit about the parent planet, not the star.
    if matches!(b.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet) {
        return (b.x[p], b.y[p], b.vx[p], b.vy[p], b.m[p]);
    }
    if !is_host_kind(b.kind[p]) {
        return default;
    }
    if let Some(bc) = circumbinary_frame(b, hosts, p, i) {
        return (bc.x, bc.y, bc.vx, bc.vy, bc.m);
    }
    (b.x[p], b.y[p], b.vx[p], b.vy[p], b.m[p])
}

// ---- Fix 4: cached parent finding. Direct callers (`body_class`,
// ---- `circularize`, UI) still use `find_parent`, which pays the full
// ---- cost. `refresh_parent_cache` builds a `ParentContext` once and
// ---- then uses `find_parent_cached` per body.

/// Cached context for `find_parent_cached`. Built once per refresh.
pub struct ParentContext {
    pub hosts: Vec<usize>,
    /// Indexed by body index. Only entries for hosts are populated.
    partners: Vec<Option<usize>>,
}

pub fn build_parent_context(b: &Bodies) -> ParentContext {
    let n = b.len();
    let hosts = host_list(b);
    let mut partners = vec![None; n];
    for &h in &hosts {
        partners[h] = binary_partner_in(b, &hosts, h);
    }
    ParentContext { hosts, partners }
}

/// Find the parent of body `i` using a prebuilt context. Bit-identical
/// to `find_parent` (same candidate set, same order, same partner value).
fn find_parent_cached(b: &Bodies, i: usize, ctx: &ParentContext) -> Option<usize> {
    let n = b.len();
    if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
        return None;
    }
    if b.origin[i].is_belt() {
        return None;
    }
    let mut host: Option<usize> = None;
    let mut best = f64::NEG_INFINITY;
    for &k in &ctx.hosts {
        if k == i {
            continue;
        }
        let d2 = (b.x[k] - b.x[i]).powi(2) + (b.y[k] - b.y[i]).powi(2) + SOFT2;
        let w = b.m[k] / d2;
        if w > best {
            best = w;
            host = Some(k);
        }
    }
    // Far outside a bound host pair, the pair acts as one parent: report the
    // heavier star so all bodies of a circumbinary disk agree.
    if let Some(h0) = host {
        if let Some(bc) = circumbinary_frame_cached(b, &ctx.partners, h0, i) {
            host = Some(bc.lead);
        }
    }
    let mut parent = host;
    let mut best_ratio = MOON_HILL_FRAC;
    if let Some(h) = host {
        for k in 0..n {
            if k == i
                || !matches!(b.kind[k], BodyKind::Planet | BodyKind::DwarfPlanet)
                || b.m[k] <= b.m[i]
            {
                continue;
            }
            let dh = ((b.x[k] - b.x[h]).powi(2) + (b.y[k] - b.y[h]).powi(2)).sqrt();
            let rh = dh * (b.m[k] / (3.0 * b.m[h])).cbrt();
            if rh <= 0.0 {
                continue;
            }
            let dk = ((b.x[k] - b.x[i]).powi(2) + (b.y[k] - b.y[i]).powi(2)).sqrt();
            let ratio = dk / rh;
            // `<=` (not `<`) so a moon spawned at exactly
            // MOON_HILL_FRAC * r_hill is recognised as a moon.
            if ratio <= best_ratio {
                best_ratio = ratio;
                parent = Some(k);
            }
        }
    }
    parent
}

pub fn find_parent(b: &Bodies, i: usize) -> Option<usize> {
    let n = b.len();
    if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
        return None;
    }
    if b.origin[i].is_belt() {
        return None;
    }
    let mut host: Option<usize> = None;
    let mut best = f64::NEG_INFINITY;
    for k in 0..n {
        if k == i || !is_host_kind(b.kind[k]) {
            continue;
        }
        let d2 = (b.x[k] - b.x[i]).powi(2) + (b.y[k] - b.y[i]).powi(2) + SOFT2;
        let w = b.m[k] / d2;
        if w > best {
            best = w;
            host = Some(k);
        }
    }
    // Far outside a bound host pair, the pair acts as one parent: report the
    // heavier star so all bodies of a circumbinary disk agree.
    if let Some(h0) = host {
        let hosts = host_list(b);
        if let Some(bc) = circumbinary_frame(b, &hosts, h0, i) {
            host = Some(bc.lead);
        }
    }
    let mut parent = host;
    let mut best_ratio = MOON_HILL_FRAC;
    if let Some(h) = host {
        for k in 0..n {
            if k == i
                || !matches!(b.kind[k], BodyKind::Planet | BodyKind::DwarfPlanet)
                || b.m[k] <= b.m[i]
            {
                continue;
            }
            let dh = ((b.x[k] - b.x[h]).powi(2) + (b.y[k] - b.y[h]).powi(2)).sqrt();
            let rh = dh * (b.m[k] / (3.0 * b.m[h])).cbrt();
            if rh <= 0.0 {
                continue;
            }
            let dk = ((b.x[k] - b.x[i]).powi(2) + (b.y[k] - b.y[i]).powi(2)).sqrt();
            let ratio = dk / rh;
            if ratio <= best_ratio {
                best_ratio = ratio;
                parent = Some(k);
            }
        }
    }
    parent
}

pub struct OrbitInfo {
    pub parent: usize,
    pub dist: f64,
    pub speed_rel: f64,
    pub bound: bool,
    pub semi_major: f64,
    pub ecc: f64,
    pub period: f64,
    pub peri: f64,
    pub apo: f64,
}

pub fn orbit_info(b: &Bodies, i: usize, par: usize) -> OrbitInfo {
    let dx = b.x[i] - b.x[par];
    let dy = b.y[i] - b.y[par];
    let dvx = b.vx[i] - b.vx[par];
    let dvy = b.vy[i] - b.vy[par];
    let r = (dx * dx + dy * dy).sqrt().max(1e-12);
    let v2 = dvx * dvx + dvy * dvy;
    let mu = G * (b.m[par] + b.m[i]);
    let rv = dx * dvx + dy * dvy;
    let ex = ((v2 - mu / r) * dx - rv * dvx) / mu;
    let ey = ((v2 - mu / r) * dy - rv * dvy) / mu;
    let e = (ex * ex + ey * ey).sqrt();
    let inv = 2.0 / r - v2 / mu;
    let bound = inv > 1e-12;
    let a = if bound { 1.0 / inv } else { f64::INFINITY };
    let period = if bound {
        std::f64::consts::TAU * (a * a * a / mu).sqrt()
    } else {
        f64::INFINITY
    };
    OrbitInfo {
        parent: par,
        dist: r,
        speed_rel: v2.sqrt(),
        bound,
        semi_major: a,
        ecc: e,
        period,
        peri: if bound { a * (1.0 - e) } else { f64::NAN },
        apo: if bound { a * (1.0 + e) } else { f64::NAN },
    }
}

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

// ---------------------------------------------------------------------------
// Stellar evolution
// ---------------------------------------------------------------------------

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

fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn post_ms_frac(m_init: f64) -> f64 {
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
fn star_mass_at(m_init: f64, age_gyr: f64) -> f64 {
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

/// Spectral class of a brown dwarf from its surface temperature.
pub fn bd_class_for_temp(t: f64) -> char {
    if t >= 2300.0 {
        'M'
    } else if t >= 1300.0 {
        'L'
    } else if t >= 500.0 {
        'T'
    } else {
        'Y'
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
fn donor_floor(b: &Bodies, i: usize) -> f64 {
    let m0 = b.m_init[i];
    if b.age_gyr[i] < star_lifetime_gyr(BodyKind::Star, m0) {
        STAR_M_MIN
    } else {
        remnant_of(m0).1
    }
}

/// A star that gains mass keeps the same fraction of its main-sequence life,
/// so it looks younger (it gets fresh fuel).
fn add_mass_to_star(b: &mut Bodies, i: usize, dm: f64) {
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
fn transfer_mass(b: &mut Bodies, d: usize, a: usize, dm: f64) -> bool {
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
fn mass_transfer_step(b: &mut Bodies, dt: f64) -> bool {
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
fn vel_unit_kms() -> f64 {
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
fn adiabatic_expand(b: &mut Bodies, i: usize, m_old: f64, m_new: f64) {
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
fn become_remnant(b: &mut Bodies, i: usize) {
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
fn evolve_stars(b: &mut Bodies, dt: f64) {
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

// ---------------------------------------------------------------------------
// User edits
// ---------------------------------------------------------------------------

pub fn user_modified(b: &mut Bodies) {
    b.parent_cache_valid = false;
    compute_accel(b);
    rebase_energy(b);
}

pub fn scale_mass(b: &mut Bodies, i: usize, factor: f64) {
    let m = b.m[i] * factor;
    if !m.is_finite() || m < 1e-4 {
        return;
    }
    if b.kind[i] == BodyKind::Star {
        // Keep the same fraction of the (new) lifetime.
        let f = b.age_gyr[i] / star_lifetime_gyr(BodyKind::Star, b.m_init[i]);
        b.m_init[i] *= factor;
        b.age_gyr[i] = f * star_lifetime_gyr(BodyKind::Star, b.m_init[i]);
    } else {
        b.m_init[i] = m;
    }
    b.set_mass(i, m);
    b.origin[i] = BodyOrigin::Spawned;
    let text = format!("{} mass scaled by {}", tag(b, i), factor);
    let (x, y) = (b.x[i], b.y[i]);
    b.log(EventKind::User, text, x, y);
    user_modified(b);
}

pub fn set_kind(b: &mut Bodies, i: usize, kind: BodyKind) {
    let m = b.m[i];
    let new_m = match kind {
        BodyKind::Star if m < STAR_M_MIN || m > 100.0 * M_SUN_ME => M_SUN_ME,
        BodyKind::BrownDwarf if m < BD_M_MIN || m > BD_M_MAX => m.clamp(BD_M_MIN, BD_M_MAX),
        BodyKind::WhiteDwarf if m < WD_M_MIN || m > WD_M_MAX => m.clamp(WD_M_MIN, WD_M_MAX),
        BodyKind::NeutronStar if m < NS_M_MIN || m > NS_M_MAX => m.clamp(NS_M_MIN, NS_M_MAX),
        BodyKind::BlackHole if m < BH_M_MIN => 10.0 * M_SUN_ME,
        BodyKind::DwarfPlanet if m > DWARF_M_MAX => DWARF_M_MAX,
        BodyKind::Planet if m >= 1000.0 => JUPITER_MASS,
        _ => m,
    };
    b.kind[i] = kind;
    b.origin[i] = BodyOrigin::Spawned;
    b.bulk_density[i] = default_bulk_density(kind, new_m);
    b.m_init[i] = new_m;
    b.age_gyr[i] = default_age_gyr(kind);
    b.set_mass(i, new_m);
    let text = format!("{} changed type", tag(b, i));
    let (x, y) = (b.x[i], b.y[i]);
    b.log(EventKind::User, text, x, y);
    user_modified(b);
}

pub fn delete_body(b: &mut Bodies, i: usize) {
    let text = format!("{} deleted", tag(b, i));
    let (x, y) = (b.x[i], b.y[i]);
    b.dead[i] = true;
    b.compact();
    b.log(EventKind::User, text, x, y);
    user_modified(b);
}

pub fn circularize(b: &mut Bodies, i: usize) -> bool {
    let Some(p) = find_parent(b, i) else {
        return false;
    };
    let dx = b.x[i] - b.x[p];
    let dy = b.y[i] - b.y[p];
    let r = (dx * dx + dy * dy).sqrt();
    if r < 1e-9 {
        return false;
    }
    let dvx = b.vx[i] - b.vx[p];
    let dvy = b.vy[i] - b.vy[p];
    let sign = if dx * dvy - dy * dvx >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let vc = (G * (b.m[p] + b.m[i]) / r).sqrt();
    b.vx[i] = b.vx[p] - sign * dy / r * vc;
    b.vy[i] = b.vy[p] + sign * dx / r * vc;
    let text = format!("{} circularized", tag(b, i));
    let (x, y) = (b.x[i], b.y[i]);
    b.log(EventKind::User, text, x, y);
    user_modified(b);
    true
}

/// Add a moon in a circular orbit around the planet at index `i`.
pub fn spawn_moon(b: &mut Bodies, i: usize, rng: &mut XorShift) -> Option<usize> {
    if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
        return None;
    }
    let host = find_parent(b, i).or_else(|| primary_index(b))?;
    if host == i {
        return None;
    }
    let d_star = (b.x[i] - b.x[host]).hypot(b.y[i] - b.y[host]);
    let r_hill = d_star * (b.m[i] / (3.0 * b.m[host])).cbrt();
    let a_lo = 5.0 * b.r[i];
    let a_hi = MOON_HILL_FRAC * r_hill;
    if a_lo >= a_hi {
        return None;
    }
    let a = a_lo * (a_hi / a_lo).powf(rng.next_f64());
    let angle = rng.next_f64() * std::f64::consts::TAU;
    let (sa, ca) = angle.sin_cos();
    let v_circ = (G * b.m[i] / a).sqrt();
    let m_moon = b.m[i] * 10f64.powf(-4.0 + rng.next_f64() * 2.0);
    let moon = b.push(
        b.x[i] + a * ca,
        b.y[i] + a * sa,
        b.vx[i] - v_circ * sa,
        b.vy[i] + v_circ * ca,
        m_moon,
        BodyKind::Planet,
    );
    b.bulk_density[moon] = draw_density(rng, false);
    b.r[moon] = b.radius_now(moon);
    b.parent_cache_valid = false;
    let (x, y) = (b.x[moon], b.y[moon]);
    let text = format!("{} spawned as a moon of {}", tag(b, moon), tag(b, i));
    b.log(EventKind::User, text, x, y);
    Some(moon)
}

/// Add a 25-particle planetary ring between 1.4 and 2.6 planet radii.
pub fn add_ring(b: &mut Bodies, i: usize, n: usize, rng: &mut XorShift) -> Vec<usize> {
    let mut out = Vec::new();
    if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
        return out;
    }
    let n = n.clamp(1, 30);
    let r0 = b.r[i];
    let m_total = 1e-4 * b.m[i];
    let radii: Vec<f64> = (0..n)
        .map(|_| r0 * 1.4 * (2.6_f64 / 1.4).powf(rng.next_f64()))
        .collect();
    let weights: Vec<f64> = radii.iter().map(|&r| (r / r0).sqrt()).collect();
    let wsum: f64 = weights.iter().sum();
    for (r, w) in radii.iter().zip(weights.iter()) {
        let m = m_total * w / wsum;
        let angle = rng.next_f64() * std::f64::consts::TAU;
        let (sa, ca) = angle.sin_cos();
        let v_circ = (G * b.m[i] / r).sqrt();
        let k = b.push(
            b.x[i] + r * ca,
            b.y[i] + r * sa,
            b.vx[i] - v_circ * sa,
            b.vy[i] + v_circ * ca,
            m,
            BodyKind::Planet,
        );
        b.origin[k] = BodyOrigin::ImpactDebris;
        b.bulk_density[k] = draw_density(rng, false);
        b.r[k] = b.radius_now(k);
        b.parent_cache_valid = false;
        out.push(k);
    }
    if !out.is_empty() {
        let (x, y) = (b.x[i], b.y[i]);
        let text = format!("ring added around {} ({} particles)", tag(b, i), out.len());
        b.log(EventKind::User, text, x, y);
    }
    out
}

// ---------------------------------------------------------------------------
// Spawn
// ---------------------------------------------------------------------------

fn sample_rayleigh(rng: &mut XorShift, sigma: f64) -> f64 {
    let u = rng.next_f64().clamp(1e-12, 1.0 - 1e-12);
    sigma * (-2.0 * (1.0 - u).ln()).sqrt()
}

fn sample_mass_power_law(rng: &mut XorShift, m_min: f64, m_max: f64, p: f64) -> f64 {
    let u = rng.next_f64();
    let inv = 1.0 - p;
    let a = m_min.powf(inv);
    let b = m_max.powf(inv);
    ((1.0 - u) * a + u * b).powf(1.0 / inv)
}

/// Draw `n` masses from a truncated power law, then scale so their sum is
/// `total`. Caps each tracer at `BELT_MAX_TRACER_MASS` and at `total/5`
/// (no single tracer should exceed 20 % of its belt — such a body would
/// act as a single perturber, not as a population). The power-law *shape*
/// is preserved; only the absolute scale changes, so relative body sizes
/// stay physical.
fn draw_masses_to_total(
    rng: &mut XorShift,
    n: usize,
    m_min: f64,
    m_max: f64,
    plaw: f64,
    total: f64,
) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if total <= 0.0 {
        return vec![0.0; n];
    }
    let mut v: Vec<f64> = (0..n)
        .map(|_| sample_mass_power_law(rng, m_min, m_max, plaw))
        .collect();
    let sum: f64 = v.iter().sum();
    if sum <= 0.0 {
        return v;
    }
    let scale = total / sum;
    let cap = BELT_MAX_TRACER_MASS.min(total / 5.0);
    for m in v.iter_mut() {
        *m = (*m * scale).min(cap);
    }
    v
}

fn spacing_range(n: usize, a_min: f64, a_max: f64, base_lo: f64, base_hi: f64) -> (f64, f64) {
    if n <= 1 {
        return (base_lo, base_hi);
    }
    let ratio_fit = (a_max / a_min).powf(1.0 / (n - 1) as f64);
    if ratio_fit >= base_hi {
        (base_lo, base_hi)
    } else {
        let scale = ratio_fit / base_hi;
        ((base_lo * scale).max(1.02), ratio_fit.max(1.02))
    }
}

/// Cartesian state of a body on a Kepler orbit about a fixed centre with
/// gravitational parameter `mu`, counter-clockwise, periapsis along `omega`.
fn kepler_state(mu: f64, a: f64, e: f64, nu: f64, omega: f64) -> (f64, f64, f64, f64) {
    let (sn, cn) = nu.sin_cos();
    let r = a * (1.0 - e * e) / (1.0 + e * cn);
    let h = (mu * a * (1.0 - e * e)).sqrt();
    let v_t = h / r;
    let v_r = mu / h * e * sn;
    let (x_orb, y_orb) = (r * cn, r * sn);
    let vx_orb = v_r * cn - v_t * sn;
    let vy_orb = v_r * sn + v_t * cn;
    let (sw, cw) = omega.sin_cos();
    (
        cw * x_orb - sw * y_orb,
        sw * x_orb + cw * y_orb,
        cw * vx_orb - sw * vy_orb,
        sw * vx_orb + cw * vy_orb,
    )
}

/// Solve Kepler's equation M = E - e sin E.
fn solve_kepler(mean_anom: f64, e: f64) -> f64 {
    let mut ea = if e < 0.8 {
        mean_anom
    } else {
        std::f64::consts::PI
    };
    for _ in 0..60 {
        let d = (ea - e * ea.sin() - mean_anom) / (1.0 - e * ea.cos());
        ea -= d;
        if d.abs() < 1e-13 {
            break;
        }
    }
    ea
}

fn true_anomaly(ecc_anom: f64, e: f64) -> f64 {
    2.0 * ((1.0 + e).sqrt() * (ecc_anom * 0.5).sin())
        .atan2((1.0 - e).sqrt() * (ecc_anom * 0.5).cos())
}

/// Push a test body on a Kepler orbit about an arbitrary centre (position,
/// velocity, central mass). Phase is uniform in mean anomaly (time).
fn push_kepler_about(
    b: &mut Bodies,
    centre: (f64, f64, f64, f64, f64),
    a: f64,
    e: f64,
    m: f64,
    rng: &mut XorShift,
    kind: BodyKind,
) -> usize {
    let (cx, cy, cvx, cvy, mc) = centre;
    let omega = rng.next_f64() * std::f64::consts::TAU;
    let mean_anom = rng.next_f64() * std::f64::consts::TAU;
    let nu = true_anomaly(solve_kepler(mean_anom, e), e);
    let (x, y, vx, vy) = kepler_state(G * (mc + m), a, e, nu, omega);
    b.push(cx + x, cy + y, cvx + vx, cvy + vy, m, kind)
}

/// Push a planet and give it a bulk density: rocky inside the snow line,
/// icy outside, light for giants. The radius follows the density.
#[allow(clippy::too_many_arguments)]
fn spawn_planet(
    b: &mut Bodies,
    centre: (f64, f64, f64, f64, f64),
    a_gm: f64,
    e: f64,
    m: f64,
    kind: BodyKind,
    snow_line_au: f64,
    rng: &mut XorShift,
) -> usize {
    let i = push_kepler_about(b, centre, a_gm, e, m, rng, kind);
    b.bulk_density[i] = if m >= GIANT_DENSITY_M_MIN {
        // Gas giants: 0.6 (Saturn) to 1.7 (Neptune) g/cm³, at any distance.
        0.6 + 1.1 * rng.next_f64()
    } else {
        draw_density(rng, (a_gm / au_in_gm()) < snow_line_au)
    };
    b.r[i] = b.radius_now(i);
    i
}

/// Planets at or above this mass (Earth masses) get gas-giant densities.
const GIANT_DENSITY_M_MIN: f64 = 50.0;

/// Bulk density in g/cm³: rocky inside the snow line, icy outside.
fn draw_density(rng: &mut XorShift, inside_snow: bool) -> f64 {
    if inside_snow {
        2.5 + 3.0 * rng.next_f64()
    } else {
        0.5 + 2.0 * rng.next_f64()
    }
}

/// Push a test body on a Kepler orbit about body `primary`.
fn push_kepler(
    b: &mut Bodies,
    primary: usize,
    a: f64,
    e: f64,
    m: f64,
    rng: &mut XorShift,
    kind: BodyKind,
) -> usize {
    let centre = (
        b.x[primary],
        b.y[primary],
        b.vx[primary],
        b.vy[primary],
        b.m[primary],
    );
    push_kepler_about(b, centre, a, e, m, rng, kind)
}

/// Kroupa (2001) IMF, two-piece: dN/dm ∝ m^-1.3 below 0.5 M☉, m^-2.3 above.
fn sample_imf_mass(rng: &mut XorShift) -> f64 {
    let u = rng.next_f64();
    if u < 0.76 {
        let u2 = u / 0.76;
        let a = 0.08_f64.powf(-0.3);
        let c = 0.50_f64.powf(-0.3);
        (a + u2 * (c - a)).powf(1.0 / -0.3)
    } else {
        let u2 = (u - 0.76) / 0.24;
        let a = 0.50_f64.powf(-1.3);
        let c = 50.0_f64.powf(-1.3);
        (a + u2 * (c - a)).powf(1.0 / -1.3)
    }
}

/// Draw the primary of a random system.
///
/// The 85 / 15 split between stellar-mass and brown-dwarf primaries is a
/// coarse population-synthesis approximation, not a direct IMF integral.
/// Kroupa (2001) with a lower cutoff of 0.08 M_sun gives ~85 % of draws
/// below 0.5 M_sun, most of which are M dwarfs. Doing the IMF integral
/// properly (integrating the mass function over [0.08, STAR_M_MIN] vs
/// [STAR_M_MIN, 50]) gives ~86 / 14, close enough that the constant is
/// retained for clarity. If `sample_imf_mass`'s range ever changes, this
/// split must be recomputed - it is not automatic.
///
/// Remnants (WD/NS/BH) are not drawn here. `spawn_aged_primary` converts
/// stars that outlive their lifetime into remnants. Drawing them here too
/// would double-count the field remnant fraction.
fn sample_primary(rng: &mut XorShift) -> (f64, BodyKind) {
    if rng.next_f64() < 0.85 {
        (sample_imf_mass(rng) * M_SUN_ME, BodyKind::Star)
    } else {
        // Log-uniform: the brown-dwarf mass function declines toward the
        // hydrogen-burning limit.
        let m = BD_M_MIN * (BD_M_MAX / BD_M_MIN).powf(rng.next_f64());
        (m, BodyKind::BrownDwarf)
    }
}

fn sample_normal(rng: &mut XorShift, mu: f64, sigma: f64) -> f64 {
    let u1 = rng.next_f64().max(1e-12);
    let u2 = rng.next_f64();
    mu + sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn poisson(rng: &mut XorShift, mean: f64) -> usize {
    if mean > 30.0 {
        return (mean + mean.sqrt() * sample_normal(rng, 0.0, 1.0)).max(0.0) as usize;
    }
    let l = (-mean).exp();
    let mut k = 0usize;
    let mut p = 1.0f64;
    loop {
        k += 1;
        p *= rng.next_f64();
        if p <= l {
            break;
        }
    }
    k - 1
}

/// Planetary architecture of a generated binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryArch {
    /// Planets orbit the primary, inside the Holman-Wiegert limit.
    STypePrimary,
    /// Planets orbit the companion, inside the Holman-Wiegert limit.
    STypeSecondary,
    /// Circumbinary: planets orbit the barycentre, outside the limit.
    PType,
}

#[derive(Clone, Copy, Debug)]
pub struct BinaryInfo {
    /// Body index of the companion (the primary is the heavier star, index 0).
    pub companion: usize,
    pub period_days: f64,
    /// Semi-major axis of the relative orbit, AU.
    pub a_au: f64,
    pub ecc: f64,
    /// m2 / m1, at most 1.
    pub mass_ratio: f64,
    pub arch: BinaryArch,
    /// Holman-Wiegert critical semi-major axis in AU: the outer edge of the
    /// stable zone for S-type (about the host star), the inner edge for P-type
    /// (about the barycentre). The planets respect it with `HW_MARGIN_S` /
    /// `HW_MARGIN_P`.
    pub a_crit_au: f64,
}

/// What `spawn_system` generated.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemInfo {
    pub binary: Option<BinaryInfo>,
}

/// Probability that a main-sequence star of mass `m_sun` (solar masses) has a
/// stellar companion (Raghavan 2010, Duchene & Kraus 2013), piecewise.
pub fn binary_fraction(m_sun: f64) -> f64 {
    if m_sun < 0.5 {
        0.30
    } else if m_sun < 0.8 {
        0.40
    } else if m_sun < 1.2 {
        0.46
    } else if m_sun < 1.5 {
        0.50
    } else if m_sun < 2.5 {
        0.65
    } else {
        0.80
    }
}

/// Holman & Wiegert (1999) critical semi-major axis for a planet orbiting one
/// star of a binary (S-type). `mu` is the perturber's mass fraction
/// m_pert / (m_host + m_pert). Result in the units of `a_b`.
pub fn hw_critical_s(a_b: f64, e: f64, mu: f64) -> f64 {
    let f = 0.464 - 0.380 * mu - 0.631 * e + 0.586 * mu * e + 0.150 * e * e - 0.198 * mu * e * e;
    (a_b * f).max(0.01 * a_b)
}

/// Holman & Wiegert (1999) critical semi-major axis for a circumbinary
/// planet (P-type). `mu` is m2 / (m1 + m2).
pub fn hw_critical_p(a_b: f64, e: f64, mu: f64) -> f64 {
    a_b * (1.60 + 5.10 * e - 2.22 * e * e + 4.12 * mu - 4.27 * e * mu - 5.09 * mu * mu
        + 4.61 * e * e * mu * mu)
}

fn orbital_period_days(a_gm: f64, m_tot: f64) -> f64 {
    std::f64::consts::TAU * (a_gm.powi(3) / (G * m_tot)).sqrt() * time_unit_s() / 86_400.0
}

pub fn spawn_system(b: &mut Bodies, body_count: usize, seed: u32) {
    spawn_system_with_info(b, body_count, seed);
}

// ---------------------------------------------------------------------------
// Ring layout for multi-system spawns (M1 to M6)
// ---------------------------------------------------------------------------

/// Largest distance (Gm) from the system's centre of mass to any body of
/// `b`. A body on a bound orbit about the heaviest host counts with its
/// osculating apoapsis, an unbound body with its present distance.
fn system_extent_gm(b: &Bodies) -> f64 {
    let n = b.len();
    let Some(primary) = (0..n)
        .filter(|&i| is_host_kind(b.kind[i]))
        .max_by(|&i, &j| b.m[i].total_cmp(&b.m[j]))
    else {
        return 0.0;
    };
    let mut ext = 0.0f64;
    for i in 0..n {
        let mut r = b.x[i]
            .hypot(b.y[i])
            .max((b.x[i] - b.x[primary]).hypot(b.y[i] - b.y[primary]));
        if i != primary {
            let o = orbit_info(b, i, primary);
            if o.bound && o.apo.is_finite() {
                r = r.max(o.apo);
            }
        }
        ext = ext.max(r);
    }
    ext
}

/// Ring radius for `n` systems whose widest one reaches `widest_gm`.
/// The ring is never smaller than 3 times that extent.
pub fn ring_radius_for_extent(n: usize, widest_gm: f64) -> f64 {
    if n <= 1 {
        0.0
    } else {
        solar_ring_radius_gm(n).max(3.0 * widest_gm)
    }
}

/// Sum term of the ring formula: v^2 = G M s / R for n equal masses.
fn ring_sum(n: usize) -> f64 {
    (1..n)
        .map(|k| 1.0 / (std::f64::consts::PI * k as f64 / n as f64).sin())
        .sum::<f64>()
        * 0.25
}

/// Polar angle of system `k` of `n`: a regular polygon plus a jitter of
/// up to +-0.15 polygon steps. `u` is a uniform draw in [0, 1).
fn ring_theta(n: usize, k: usize, u: f64) -> f64 {
    std::f64::consts::TAU * k as f64 / n as f64
        + (u - 0.5) * 0.3 * (std::f64::consts::TAU / n as f64)
}

/// State (x, y, vx, vy) of a system of `mass` on an orbit of semi-major
/// axis `a` about the ring centre. The central term is `mass * s`, the
/// value that makes e = 0 reproduce the circular ring speed.
fn ring_slot(mass: f64, s: f64, a: f64, e: f64, nu: f64, omega: f64) -> (f64, f64, f64, f64) {
    kepler_state(G * mass * s, a, e, nu, omega)
}

/// Position and velocity of every system on the ring. The semi-major axis
/// scatters by +-15 %, the eccentricity is Rayleigh(0.05), and the system
/// sits at its jittered polar angle. One system gets zero state.
fn ring_slots(n: usize, seed: u32, ring: f64, masses: &[f64]) -> Vec<(f64, f64, f64, f64)> {
    if n <= 1 {
        return vec![(0.0, 0.0, 0.0, 0.0); n];
    }
    let mut rng = XorShift::new(seed ^ 0x51AB_C0DE);
    let s = ring_sum(n);
    (0..n)
        .map(|k| {
            let theta = ring_theta(n, k, rng.next_f64());
            let a_k = ring * (0.85 + 0.30 * rng.next_f64());
            let e_k = sample_rayleigh(&mut rng, 0.05).min(0.5);
            let nu = true_anomaly(
                solve_kepler(rng.next_f64() * std::f64::consts::TAU, e_k),
                e_k,
            );
            // The polar angle is nu + omega, so omega follows from theta.
            ring_slot(masses[k], s, a_k, e_k, nu, theta - nu)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Plummer cluster layout (replaces the ring for spawn_systems / spawn_solar_systems)
// ---------------------------------------------------------------------------

/// Minimum centre separation, in summed reaches.
const SEP_FACTOR: f64 = 2.0;
/// Reach = this quantile of body distances from the system centre. Using a
/// quantile (not the max) keeps one scattered KBO from sizing the cluster.
const REACH_QUANTILE: f64 = 0.9;

/// Stable reach of a system in Gm: 90 % quantile of distance from its
/// centre of mass, times 1.5 margin. Replaces `system_extent_gm` for layout.
fn system_reach_gm(b: &Bodies) -> f64 {
    let n = b.len();
    if n == 0 {
        return 0.0;
    }
    let mt: f64 = b.m.iter().sum();
    let cx = (0..n).map(|i| b.m[i] * b.x[i]).sum::<f64>() / mt;
    let cy = (0..n).map(|i| b.m[i] * b.y[i]).sum::<f64>() / mt;
    let mut d: Vec<f64> = (0..n).map(|i| (b.x[i] - cx).hypot(b.y[i] - cy)).collect();
    d.sort_by(|a, c| a.total_cmp(c));
    let k = ((n as f64 * REACH_QUANTILE) as usize).min(n - 1);
    (d[k] * 1.5).max(200.0)
}

/// (x, y, vx, vy) for every system. Plummer surface density
/// M(<R) = M R^2 / (R^2 + a^2), isotropic velocities with
/// sigma_1D^2 = G M / (6 sqrt(r^2 + a^2)) (virial equilibrium).
/// The scale radius `a` grows until the separation constraint holds.
fn cluster_slots(seed: u32, reaches: &[f64], masses: &[f64]) -> Vec<(f64, f64, f64, f64)> {
    let n = reaches.len();
    if n <= 1 {
        return vec![(0.0, 0.0, 0.0, 0.0); n];
    }
    let m_tot: f64 = masses.iter().sum();
    let mean_reach = reaches.iter().sum::<f64>() / n as f64;
    let mut a = SEP_FACTOR * mean_reach * (n as f64).sqrt();
    let mut rng = XorShift::new(seed ^ 0xC1A5_7E25);
    let mut pos: Vec<(f64, f64)> = Vec::new();
    'grow: for _attempt in 0..200 {
        pos.clear();
        for k in 0..n {
            let mut placed = false;
            for _ in 0..200 {
                let u = rng.next_f64().min(0.95); // truncate the tail
                let r = a * (u / (1.0 - u)).sqrt();
                let th = rng.next_f64() * std::f64::consts::TAU;
                let p = (r * th.cos(), r * th.sin());
                let ok = pos.iter().enumerate().all(|(j, q)| {
                    (p.0 - q.0).hypot(p.1 - q.1) >= SEP_FACTOR * (reaches[k] + reaches[j])
                });
                if ok {
                    pos.push(p);
                    placed = true;
                    break;
                }
            }
            if !placed {
                a *= 1.1;
                continue 'grow;
            }
        }
        break;
    }
    pos.iter()
        .map(|&(x, y)| {
            let sigma = (G * m_tot / (6.0 * (x * x + y * y + a * a).sqrt())).sqrt();
            (
                x,
                y,
                sample_normal(&mut rng, 0.0, sigma),
                sample_normal(&mut rng, 0.0, sigma),
            )
        })
        .collect()
}

/// Spawn `n_systems` independent random systems on a ring of radius
/// `solar_ring_radius_gm(n_systems)`.
///
/// Each system has `body_count` bodies (per-system) and its own derived
/// seed, so the systems are statistically independent. The systems sit on
/// the ring and are given the circular-ring velocity so they (approximately)
/// orbit the common centre rather than falling into one clump. The ring
/// formula uses each system's own mass for its speed.
pub fn spawn_random_systems(
    b: &mut Bodies,
    body_count: usize,
    seed: u32,
    n_systems: usize,
) -> Vec<SystemInfo> {
    spawn_random_systems_impl(b, body_count, seed, n_systems).0
}

/// `spawn_random_systems`, and the index range of every system.
fn spawn_random_systems_impl(
    b: &mut Bodies,
    body_count: usize,
    seed: u32,
    n_systems: usize,
) -> (Vec<SystemInfo>, Vec<(usize, usize)>) {
    let n = n_systems.clamp(1, MAX_SOLAR_SYSTEMS);
    // With realistic belts, `body_count` is a *ceiling* on each system,
    // not its actual count (a typical system has ~15 planets + ~10–60
    // belt tracers). So `body_count * n` overestimates. Still, cap it
    // to `MAX_BODIES` so a pathological slider pair cannot exhaust
    // memory.
    let per_system = (MAX_BODIES / n).max(1);
    let body_count = body_count.max(1).min(per_system);
    *b = Bodies::with_capacity(body_count * n + 16);

    // Per-system seed. Multiply by an odd constant so consecutive k give
    // well-separated draws even from the same master seed.
    let sys_seed = |k: usize| seed.wrapping_add((k as u32).wrapping_mul(0x9E37_79B9));

    // Spawn into scratch `Bodies`: `spawn_system_with_info` recentres the
    // COM to the origin, and we translate it to its slot later.
    let mut temps: Vec<(Bodies, SystemInfo)> = Vec::with_capacity(n);
    for k in 0..n {
        let mut temp = Bodies::with_capacity(body_count);
        let info = spawn_system_with_info(&mut temp, body_count, sys_seed(k));
        temps.push((temp, info));
    }

    // M5: heavier systems get more bodies. Spawn again with the scaled
    // count (same seed, so the same primary). One system is left alone.
    if n > 1 {
        let masses: Vec<f64> = temps.iter().map(|t| t.0.m.iter().sum::<f64>()).collect();
        let avg = masses.iter().sum::<f64>() / n as f64;
        for k in 0..n {
            let scale = (masses[k] / avg).clamp(0.5, 2.0);
            let count = ((body_count as f64 * scale).round() as usize).max(1);
            if count != body_count {
                let mut temp = Bodies::with_capacity(count);
                let info = spawn_system_with_info(&mut temp, count, sys_seed(k));
                temps[k] = (temp, info);
            }
        }
    }

    let reaches: Vec<f64> = temps.iter().map(|t| system_reach_gm(&t.0)).collect();
    let masses: Vec<f64> = temps.iter().map(|t| t.0.m.iter().sum::<f64>()).collect();
    let slots = cluster_slots(seed, &reaches, &masses);

    let mut infos = Vec::with_capacity(n);
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(n);
    for (k, (temp, info)) in temps.iter().enumerate() {
        let (x, y, vx, vy) = slots[k];
        let (start, end) = append_system(b, temp, x, y);
        for i in start..end {
            b.vx[i] += vx;
            b.vy[i] += vy;
        }
        ranges.push((start, end));
        infos.push(*info);
    }
    // The first system's metallicity stands for the ring.
    b.metallicity = temps[0].0.metallicity;

    // Runtime RNG (fragment masses, kick angles) must depend on the master seed.
    b.rng = XorShift::new(seed ^ 0xDEAD_BEEF);
    recenter_com(b);
    compute_accel(b);
    rebase_energy(b);
    // Refresh the parent cache so orbit lines draw correctly on frame 1.
    // Do it during spawn (user-triggered) rather than in `advance`: the
    // O(n²) cost would otherwise stall the first physics tick after spawn
    // while holding the bodies lock.
    b.refresh_parent_cache();
    b.dust_fields.clear();
    dust::init_fields(b, Some(&ranges));
    (infos, ranges)
}

/// Like `spawn_system`, but also reports what was generated (binary
/// parameters and architecture).
pub fn spawn_system_with_info(b: &mut Bodies, body_count: usize, seed: u32) -> SystemInfo {
    *b = Bodies::with_capacity(body_count);
    b.rng = XorShift::new(seed ^ 0xA5A5_5A5A);
    let mut rng = XorShift::new(seed);

    // Host metallicity [Fe/H] in dex (log-normal in Z): mean 0, sigma 0.20.
    b.metallicity = sample_normal(&mut rng, 0.0, 0.20);
    let (m_init, kind0) = sample_primary(&mut rng);
    let prim = spawn_aged_primary(b, m_init, kind0, &mut rng);
    let kind = b.kind[prim];

    let mut remaining = body_count.saturating_sub(1);
    // Main-sequence hosts get a stellar companion with `binary_fraction`;
    // brown-dwarf hosts get a brown-dwarf companion in ~20 % of cases.
    // Remnant hosts keep their single-body architectures.
    let mut binary: Option<BinaryInfo> = None;
    let wants_companion = if kind == BodyKind::Star {
        rng.next_f64() < binary_fraction(m_init / M_SUN_ME)
    } else if kind == BodyKind::BrownDwarf {
        rng.next_f64() < 0.20
    } else {
        false
    };
    if wants_companion && remaining >= BIN_MIN_BODIES {
        let prim_age = b.age_gyr[prim];
        binary = spawn_companion(b, prim, &mut rng, prim_age);
        if binary.is_some() {
            remaining -= 1;
        }
    }
    // Hierarchical third (and fourth) star: planned now, so that its slot is
    // taken out of the budget and the planets stay inside its stability
    // limit; placed after the planets, around the whole inner system.
    let mut outer_stars: Vec<OuterStar> = Vec::new();
    if kind == BodyKind::Star {
        if let Some(bi) = &binary {
            let m1 = b.m[prim];
            let mut m_in = b.m[prim] + b.m[bi.companion];
            let mut a_in = bi.a_au;
            while outer_stars.len() < 2 && remaining > BIN_MIN_BODIES {
                let p_next = if outer_stars.is_empty() {
                    P_TRIPLE_GIVEN_BINARY
                } else {
                    P_QUAD_GIVEN_TRIPLE
                };
                if rng.next_f64() >= p_next {
                    break;
                }
                let Some(s) = plan_outer_star(m1, m_in, a_in, &mut rng) else {
                    break;
                };
                m_in += s.m;
                a_in = s.a_au;
                remaining -= 1;
                outer_stars.push(s);
            }
        }
    }
    // Planets must stay well inside the nearest outer star's periapsis.
    let a_outer_limit_au = outer_stars
        .first()
        .map_or(f64::INFINITY, |s| s.a_au * (1.0 - s.ecc) / 3.5);
    match binary.as_mut() {
        Some(bi) => spawn_binary_planets(b, prim, bi, remaining, a_outer_limit_au, &mut rng),
        None => spawn_architecture(b, prim, remaining, &mut rng),
    }
    let prim_age_now = b.age_gyr[prim];
    for s in &outer_stars {
        place_outer_star(b, s, prim_age_now, &mut rng);
    }

    recenter_com(b);
    compute_accel(b);
    rebase_energy(b);
    dust::init_fields(b, None);
    SystemInfo { binary }
}

/// Push the primary of a random system at the origin and give it an age.
///
/// A star older than its total life is spawned as its remnant (kind, mass
/// and cooling age), keeping `m_init`. Younger stars are spread uniformly over
/// their life (not over Gyr), so a giant has also lost the mass its age implies.
/// Ages never exceed `AGE_UNIVERSE_GYR`.
/// Non-stellar primaries get a random age in the range that suits their kind.
fn spawn_aged_primary(b: &mut Bodies, m_init: f64, kind: BodyKind, rng: &mut XorShift) -> usize {
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
fn spawn_companion(
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
struct OuterStar {
    /// Mass in Earth masses.
    m: f64,
    /// Semi-major axis about the centre of mass of everything inside it, AU.
    a_au: f64,
    ecc: f64,
}

/// Plan a star on a wide orbit about an inner system of mass `m_in` whose
/// own widest orbit is `a_in_au`. `m1` is the primary mass, used to scale
/// the mass ratio. The orbit respects the Mardling & Aarseth (2001)
/// stability limit, a_out / a_in > 2.8 [(1+q)(1+e)/sqrt(1-e)]^(2/5), with a
/// log-uniform factor of 1-20 on top. Returns `None` if no such star fits.
fn plan_outer_star(m1: f64, m_in: f64, a_in_au: f64, rng: &mut XorShift) -> Option<OuterStar> {
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
fn place_outer_star(b: &mut Bodies, s: &OuterStar, age_gyr: f64, rng: &mut XorShift) {
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
fn spawn_binary_planets(
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

fn spawn_architecture(b: &mut Bodies, prim: usize, remaining: usize, rng: &mut XorShift) {
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
struct Frame {
    /// Centre of the orbits: x, y, vx, vy and the central mass (one star, or
    /// the barycentre of a pair with the total mass).
    centre: (f64, f64, f64, f64, f64),
    /// Mass in solar masses that selects the planet-count class.
    m_class_sun: f64,
    /// Total luminosity heating the system, solar units.
    lum: f64,
    /// Physical inner edge of the compact planets (sets the snow line floor).
    inner_au: f64,
    /// Nothing is placed inside this semi-major axis (P-type stability limit).
    a_lo_au: f64,
    /// Nothing is placed outside this semi-major axis (S-type stability limit).
    a_hi_au: f64,
    /// Multiplier on planet-occurrence rates.
    occ_scale: f64,
    /// Mass in Earth masses of the star (or the pair) that the belts orbit.
    m_star: f64,
    /// Age of the host star in Gyr.
    age_gyr: f64,
}

/// Orbital period of the innermost compact planet: 2-20 days, log-uniform
/// (the Kepler-like inner edge).
fn compact_period_p0(rng: &mut XorShift) -> f64 {
    10f64.powf(0.3 + rng.next_f64()) * 86_400.0 / time_unit_s()
}

fn inner_edge_au(p0: f64, mass: f64, r_star_au: f64) -> f64 {
    let a0_gm = (G * mass * (p0 / std::f64::consts::TAU).powi(2)).cbrt();
    (a0_gm / au_in_gm()).max(5.0 * r_star_au).max(0.005)
}

/// Intersection of `[lo, hi]` with the allowed `[a_lo, a_hi]`. If they do not
/// overlap, use the nearest part of the allowed range instead.
#[allow(dead_code)]
fn fit_range(lo: f64, hi: f64, a_lo: f64, a_hi: f64) -> (f64, f64) {
    let (l, h) = (lo.max(a_lo), hi.min(a_hi));
    if l < h {
        (l, h)
    } else if hi < a_lo {
        (a_lo, (a_lo * 1.6).min(a_hi))
    } else {
        ((a_hi * 0.5).max(a_lo), a_hi)
    }
}

fn spawn_ms_frame(b: &mut Bodies, fr: &Frame, remaining: usize, rng: &mut XorShift) {
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
            b.bulk_density[moon_idx] = draw_density(rng, false); // icy
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
fn spawn_evolved_system(b: &mut Bodies, prim: usize, remaining: usize, rng: &mut XorShift) {
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

// ---------------------------------------------------------------------------
// Belts: archetypes, populations and spawning
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeltArchetype {
    SolarType,
    YoungDebrisDisk,
    HotJupiter,
    CompactChain,
    /// Old, quiet, giant-poor system with a modest outer belt only.
    Sparse,
    NoBelt,
    BinaryS,
    BinaryP,
    Evolved,
}

#[derive(Clone, Debug)]
pub struct BeltSpec {
    pub archetype: BeltArchetype,
    /// Total belt mass in Earth masses.
    pub total_mass: f64,
    /// Mass budget of the outer (Kuiper-like) populations in Earth masses.
    /// `total_mass` is the inner / asteroid / Trojan / Hilda budget.
    pub outer_mass: f64,
    pub inner: Option<InnerBelt>,
    pub outer: Option<OuterBelt>,
    pub trojans: Vec<Trojan>,
    pub gaps: Vec<Gap>,
}

#[derive(Clone, Debug)]
pub struct InnerBelt {
    pub a_lo_au: f64,
    pub a_hi_au: f64,
    pub mass_frac: f64,
    pub e_sigma: f64,
    pub plaw: f64,
    pub m_min: f64,
    pub m_max: f64,
}

#[derive(Clone, Debug)]
pub struct OuterBelt {
    pub populations: Vec<OuterPopulation>,
}

#[derive(Clone, Debug)]
pub struct OuterPopulation {
    pub kind: BodyOrigin,
    pub a_lo_au: f64,
    pub a_hi_au: f64,
    pub mass_frac: f64,
    pub e_sigma: f64,
    /// Pericentre in AU. When set, eccentricity is chosen to put the
    /// pericentre near this distance (scattered objects).
    pub q_frac: Option<f64>,
    pub plaw: f64,
    pub m_min: f64,
    pub m_max: f64,
}

#[derive(Clone, Debug)]
pub struct Trojan {
    pub giant_a_au: f64,
    pub leading_frac: f64,
    pub mass_frac: f64,
}

#[derive(Clone, Debug)]
pub struct Gap {
    pub a_au: f64,
    pub width_au: f64,
}

impl BeltSpec {
    pub fn empty() -> Self {
        Self {
            archetype: BeltArchetype::NoBelt,
            total_mass: 0.0,
            outer_mass: 0.0,
            inner: None,
            outer: None,
            trojans: Vec::new(),
            gaps: Vec::new(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.total_mass < 1e-6
            && self.outer_mass < 1e-6
            && self.inner.is_none()
            && self.outer.is_none()
    }
}

/// Probability that a system of mass `m_sun` has a leftover-planetesimal
/// belt at all.
///
/// This is the intrinsic occurrence rate, not a detection rate. Belts form
/// during planet formation and persist even when they are too faint for IR
/// excess surveys; the ~2-15 % detection rates in Bryden 2006 / Eiroa 2013
/// are a different quantity. `age_gyr` is accepted for symmetry with the
/// (now removed) detection-rate model; the intrinsic rate is age-independent.
fn debris_belt_occurrence_fraction(m_sun: f64, age_gyr: f64) -> f64 {
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
fn belt_mass_earth(m_sun: f64, feh: f64, age_gyr: f64, rng: &mut XorShift) -> f64 {
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
fn belt_mass_earth_outer(m_sun: f64, feh: f64, age_gyr: f64, rng: &mut XorShift) -> f64 {
    let baseline = 0.035 * m_sun * 10f64.powf(1.5 * feh);
    let scatter = 10f64.powf(sample_normal(rng, 0.0, 1.0));
    baseline * scatter * (1.0 + age_gyr / 3.0).powf(-0.3)
}

/// Kirkwood gaps: the 3:1, 5:2, 7:3 and 2:1 mean-motion resonances with a
/// giant at `a_giant_au`. A resonance p:q (giant : body periods) lies at
/// a = a_giant * (q/p)^(2/3).
fn kirkwood_gaps(a_giant_au: f64) -> Vec<Gap> {
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

/// Choose the belt architecture of a system from its star, age,
/// metallicity and planets. `planets` holds `(index, mass_earth, a_au)`.
#[allow(clippy::too_many_arguments)]
fn belt_spec_for(
    _b: &Bodies,
    m_star: f64,
    feh: f64,
    age_gyr: f64,
    planets: &[(usize, f64, f64)],
    snow_line_au: f64,
    a_lo_au: f64,
    a_hi_au: f64,
    rng: &mut XorShift,
) -> BeltSpec {
    let m_sun = m_star / M_SUN_ME;

    let p_belt = debris_belt_occurrence_fraction(m_sun, age_gyr);
    if rng.next_f64() > p_belt {
        return BeltSpec::empty();
    }

    let giants: Vec<&(usize, f64, f64)> = planets.iter().filter(|&&(_, m, _)| m > 50.0).collect();
    let has_hj = giants.iter().any(|&&(_, _, a)| a < 0.1);
    let n_compact = planets
        .iter()
        .filter(|&&(_, m, a)| m <= 30.0 && a < 1.0)
        .count();

    let mut spec = BeltSpec {
        archetype: BeltArchetype::NoBelt,
        total_mass: belt_mass_earth(m_sun, feh, age_gyr, rng),
        outer_mass: 0.0,
        inner: None,
        outer: None,
        trojans: Vec::new(),
        gaps: Vec::new(),
    };

    let single_outer = |a_lo: f64, a_hi: f64, e_sigma: f64, m_min: f64, m_max: f64| {
        Some(OuterBelt {
            populations: vec![OuterPopulation {
                kind: BodyOrigin::KuiperCold,
                a_lo_au: a_lo,
                a_hi_au: a_hi,
                mass_frac: 1.0,
                e_sigma,
                q_frac: None,
                plaw: MASS_PLAW_KUIPER,
                m_min,
                m_max,
            }],
        })
    };

    if has_hj {
        spec.archetype = BeltArchetype::HotJupiter;
        let a_giant = giants
            .iter()
            .map(|&&(_, _, a)| a)
            .fold(f64::INFINITY, f64::min);
        let start = (a_giant * 3.0).max(10.0).max(a_lo_au);
        let end = (start * 5.0).min(a_hi_au);
        if end > start {
            spec.outer = single_outer(start, end, 0.05, MASS_KUIPER_MIN, MASS_KUIPER_MAX);
        }
        spec.total_mass *= 0.1;
        spec.outer_mass = belt_mass_earth_outer(m_sun, feh, age_gyr, rng) * 0.1;
    } else if n_compact >= 4 {
        spec.archetype = BeltArchetype::CompactChain;
        let start = (snow_line_au * 5.0).max(a_lo_au);
        let end = (start * 3.0).min(a_hi_au);
        if end > start {
            spec.outer = single_outer(start, end, 0.08, MASS_KUIPER_MIN, MASS_KUIPER_MAX);
        }
        spec.total_mass *= 0.1;
        spec.outer_mass = belt_mass_earth_outer(m_sun, feh, age_gyr, rng) * 0.1;
    } else if giants.is_empty() && m_sun > 1.2 && age_gyr < 0.5 {
        spec.archetype = BeltArchetype::YoungDebrisDisk;
        let (start, end) = (a_lo_au.max(80.0), a_hi_au.min(120.0));
        if end > start {
            spec.outer = single_outer(start, end, 0.02, 1e-3, 1e-1);
        }
        spec.total_mass *= 100.0;
        spec.outer_mass = belt_mass_earth_outer(m_sun, feh, age_gyr, rng) * 100.0;
    } else if !giants.is_empty() {
        spec.archetype = BeltArchetype::SolarType;
        spec.outer_mass = belt_mass_earth_outer(m_sun, feh, age_gyr, rng);

        let a_giant_inner = giants
            .iter()
            .map(|&&(_, _, a)| a)
            .fold(f64::INFINITY, f64::min);
        let a_giant_outer = giants.iter().map(|&&(_, _, a)| a).fold(0.0, f64::max);
        let a_compact = planets
            .iter()
            .filter(|&&(_, m, _)| m <= 30.0)
            .map(|&(_, _, a)| a)
            .fold(0.0, f64::max);

        let ast_lo = (a_compact * 1.2).max(snow_line_au * 0.7).max(a_lo_au);
        let ast_hi = (a_giant_inner * 0.6).min(a_hi_au);
        if ast_hi > ast_lo {
            spec.inner = Some(InnerBelt {
                a_lo_au: ast_lo,
                a_hi_au: ast_hi,
                mass_frac: 0.04,
                e_sigma: 0.10,
                plaw: MASS_PLAW_ASTEROID,
                m_min: MASS_ASTEROID_MIN,
                m_max: MASS_ASTEROID_MAX,
            });
        }

        let kui_lo = (a_giant_outer * 1.5).max(snow_line_au * 10.0).max(a_lo_au);
        let kui_hi = (snow_line_au * 40.0).min(a_hi_au);
        if kui_hi > kui_lo {
            let span = kui_hi - kui_lo;
            let pop = |kind, a_lo, a_hi, mass_frac, e_sigma, q_frac| OuterPopulation {
                kind,
                a_lo_au: a_lo,
                a_hi_au: a_hi,
                mass_frac,
                e_sigma,
                q_frac,
                plaw: MASS_PLAW_KUIPER,
                m_min: MASS_KUIPER_MIN,
                m_max: MASS_KUIPER_MAX,
            };
            spec.outer = Some(OuterBelt {
                populations: vec![
                    pop(
                        BodyOrigin::KuiperCold,
                        kui_lo,
                        kui_lo + span * 0.4,
                        0.40,
                        0.03,
                        None,
                    ),
                    pop(BodyOrigin::KuiperHot, kui_lo, kui_hi, 0.30, 0.15, None),
                    pop(BodyOrigin::KuiperResonant, kui_lo, kui_hi, 0.10, 0.10, None),
                    pop(
                        BodyOrigin::KuiperScattered,
                        kui_lo,
                        kui_hi,
                        0.10,
                        0.30,
                        Some(a_giant_outer),
                    ),
                ],
            });
        }

        for &&(_, gm, ga) in &giants {
            if gm > 50.0 && ga > 0.3 {
                spec.trojans.push(Trojan {
                    giant_a_au: ga,
                    leading_frac: 0.5,
                    mass_frac: 0.06,
                });
                spec.gaps.extend(kirkwood_gaps(ga));
            }
        }
    } else {
        // Sparse: old, quiet, giant-poor system. Its inner belt was cleared
        // by migration or never formed, but a modest outer belt persists.
        // This is what ALMA sees around old K dwarfs (Matrà 2018).
        spec.archetype = BeltArchetype::Sparse;
        spec.outer_mass = belt_mass_earth_outer(m_sun, feh, age_gyr, rng) * 0.3;
        let start = (snow_line_au * 5.0).max(a_lo_au);
        let end = (start * 5.0).min(a_hi_au);
        if end > start {
            spec.outer = single_outer(start, end, 0.10, MASS_KUIPER_MIN, MASS_KUIPER_MAX);
        }
    }

    spec
}

fn belt_spec_for_evolved(m_remnant_sun: f64, _age_gyr: f64, rng: &mut XorShift) -> BeltSpec {
    if m_remnant_sun < 1.0 && rng.next_f64() < 0.3 {
        let total = 1e-4 * rng.next_f64();
        BeltSpec {
            archetype: BeltArchetype::Evolved,
            total_mass: total,
            outer_mass: total,
            inner: None,
            outer: Some(OuterBelt {
                populations: vec![OuterPopulation {
                    kind: BodyOrigin::KuiperScattered,
                    a_lo_au: 5.0,
                    a_hi_au: 40.0,
                    mass_frac: 1.0,
                    e_sigma: 0.30,
                    q_frac: None,
                    plaw: MASS_PLAW_KUIPER,
                    m_min: MASS_KUIPER_MIN,
                    m_max: MASS_KUIPER_MAX,
                }],
            }),
            trojans: Vec::new(),
            gaps: Vec::new(),
        }
    } else {
        BeltSpec::empty()
    }
}

/// Number of tracers a belt of mass `mass` (M⊕) gets, capped at `budget`.
/// Linear in mass: doubling the mass doubles the count. Two floors apply:
/// a physical one (`BELT_MIN_TRACER_MASS`) and a visibility one
/// (`BELT_MIN_TRACERS`), which never exceeds `budget`.
fn tracers_for_mass(mass: f64, budget: usize) -> usize {
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

fn crosses_gap(a_au: f64, gaps: &[Gap]) -> bool {
    gaps.iter().any(|g| (a_au - g.a_au).abs() < g.width_au)
}

/// Draw a semi-major axis in `[lo, hi]` AU that is outside every gap.
/// Returns `None` when 16 draws all fall in gaps.
fn draw_a_outside_gaps(rng: &mut XorShift, lo: f64, hi: f64, gaps: &[Gap]) -> Option<f64> {
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
fn draw_a_outer(rng: &mut XorShift, lo: f64, hi: f64, gaps: &[Gap]) -> Option<f64> {
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
fn spawn_belt_body(
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
    let kind = if m >= BELT_DWARF_M_MIN {
        BodyKind::DwarfPlanet
    } else {
        BodyKind::Planet
    };
    let i = b.push(cx + x, cy + y, cvx + vx, cvy + vy, m, kind);
    b.origin[i] = origin;
    b.bulk_density[i] = draw_density(rng, (a_gm / au_in_gm()) < snow_line_au);
    b.r[i] = b.radius_now(i);
    i
}

/// Angle (rad) of the giant planet nearest to `a_au` about the centre, if
/// the system has one within 10 %.
fn giant_angle(b: &Bodies, centre: (f64, f64, f64, f64, f64), a_au: f64) -> Option<f64> {
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
fn giant_mass_near(b: &Bodies, centre: (f64, f64, f64, f64, f64), a_au: f64) -> Option<f64> {
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
fn spawn_trojan(
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
    let kind = if m >= BELT_DWARF_M_MIN {
        BodyKind::DwarfPlanet
    } else {
        BodyKind::Planet
    };
    let i = b.push(cx + x, cy + y, cvx + vx, cvy + vy, m, kind);
    b.origin[i] = BodyOrigin::Trojan;
    b.bulk_density[i] = draw_density(rng, true);
    b.r[i] = b.radius_now(i);
    i
}

/// One part of a belt that will receive bodies.
enum BeltPart<'a> {
    Inner(&'a InnerBelt),
    Outer(&'a OuterPopulation),
    Trojan(&'a Trojan),
}

/// A small population of scattered background bodies — interstellar
/// interlopers and escaped belt members. Real systems have a handful of
/// these, not hundreds. Spawns 0–8 bodies with a Poisson mean of 2.
fn spawn_background(
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
fn spawn_belts(
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

// ---------------------------------------------------------------------------
// Solar System preset
// ---------------------------------------------------------------------------

/// A planetary system read from JSON (see `assets/presets/solar_system.json`).
fn default_dust_boost() -> f64 {
    1.0
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct SystemPreset {
    pub name: String,
    /// Multiplies the initial dust density. Default 1.
    #[serde(default = "default_dust_boost")]
    pub dust_boost: f64,
    pub star: StarPreset,
    #[serde(default)]
    pub companions: Vec<CompanionPreset>,
    pub planets: Vec<PlanetPreset>,
    /// Small-body belts (asteroids, Kuiper populations, Trojans, gaps).
    #[serde(default)]
    pub belts: Option<BeltPreset>,
}

fn default_phase() -> f64 {
    0.5
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct CompanionPreset {
    pub name: String,
    pub mass_earth: f64,
    #[serde(default)]
    pub age_gyr: Option<f64>,
    #[serde(default)]
    pub kind: Option<String>,
    pub a_au: f64,
    #[serde(default)]
    pub e: f64,
    #[serde(default = "default_phase")]
    pub phase: f64,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct BeltPreset {
    pub total_mass_earth: f64,
    /// Number of belt bodies to spawn. Default is `PRESET_BELT_BUDGET`.
    #[serde(default)]
    pub belt_budget: Option<usize>,
    #[serde(default)]
    pub inner: Option<InnerBeltPreset>,
    #[serde(default)]
    pub outer: Option<OuterBeltPreset>,
    #[serde(default)]
    pub trojans: Vec<TrojanPreset>,
    #[serde(default)]
    pub gaps: Vec<GapPreset>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct InnerBeltPreset {
    pub a_lo_au: f64,
    pub a_hi_au: f64,
    pub mass_frac: f64,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct OuterBeltPreset {
    pub populations: Vec<PopulationPreset>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct PopulationPreset {
    pub kind: String,
    pub a_lo_au: f64,
    pub a_hi_au: f64,
    pub mass_frac: f64,
    #[serde(default = "default_e_sigma")]
    pub e_sigma: f64,
    /// Pericentre in AU for scattered objects.
    #[serde(default)]
    pub q_frac: Option<f64>,
}

fn default_e_sigma() -> f64 {
    0.10
}

fn parse_body_kind(s: Option<&str>, mass: f64) -> BodyKind {
    match s {
        Some("white_dwarf") => BodyKind::WhiteDwarf,
        Some("neutron_star") => BodyKind::NeutronStar,
        Some("black_hole") => BodyKind::BlackHole,
        Some("brown_dwarf") => BodyKind::BrownDwarf,
        Some("star") | None => {
            if mass >= STAR_M_MIN {
                BodyKind::Star
            } else {
                BodyKind::BrownDwarf
            }
        }
        Some(other) => panic!("unknown body kind in preset: {other}"),
    }
}

fn place_companion(b: &mut Bodies, c: &CompanionPreset, default_age: f64) {
    let n = b.len();
    let (mut mt, mut cx, mut cy, mut vx, mut vy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        mt += b.m[i];
        cx += b.m[i] * b.x[i];
        cy += b.m[i] * b.y[i];
        vx += b.m[i] * b.vx[i];
        vy += b.m[i] * b.vy[i];
    }
    (cx, cy, vx, vy) = (cx / mt, cy / mt, vx / mt, vy / mt);
    let mt_total = mt + c.mass_earth;
    let a_gm = c.a_au * au_in_gm();
    let m_anom = c.phase * std::f64::consts::TAU;
    let nu = true_anomaly(solve_kepler(m_anom, c.e), c.e);
    let (rx, ry, rvx, rvy) = kepler_state(G * mt_total, a_gm, c.e, nu, 0.0);
    let f_in = c.mass_earth / mt_total;
    let f_out = mt / mt_total;
    for i in 0..n {
        b.x[i] -= f_in * rx;
        b.y[i] -= f_in * ry;
        b.vx[i] -= f_in * rvx;
        b.vy[i] -= f_in * rvy;
        b._px[i] = b.x[i];
        b._py[i] = b.y[i];
    }
    let kind = parse_body_kind(c.kind.as_deref(), c.mass_earth);
    let idx = b.push(
        cx + f_out * rx,
        cy + f_out * ry,
        vx + f_out * rvx,
        vy + f_out * rvy,
        c.mass_earth,
        kind,
    );
    b.name[idx] = Some(c.name.clone());
    b.age_gyr[idx] = c.age_gyr.unwrap_or(default_age);
    b.set_mass(idx, c.mass_earth);
}

fn push_circumbinary(b: &mut Bodies, a_au: f64, e: f64, m_me: f64, mean_anomaly: f64) -> usize {
    let n = b.len();
    let (mut mt, mut cx, mut cy, mut vx, mut vy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        if !is_host_kind(b.kind[i]) {
            continue;
        }
        mt += b.m[i];
        cx += b.m[i] * b.x[i];
        cy += b.m[i] * b.y[i];
        vx += b.m[i] * b.vx[i];
        vy += b.m[i] * b.vy[i];
    }
    (cx, cy, vx, vy) = (cx / mt, cy / mt, vx / mt, vy / mt);
    let a = a_au * au_in_gm();
    let mu = G * (mt + m_me);
    let nu = true_anomaly(solve_kepler(mean_anomaly, e), e);
    let (x, y, vx_rel, vy_rel) = kepler_state(mu, a, e, nu, 0.0);
    b.push(
        cx + x,
        cy + y,
        vx + vx_rel,
        vy + vy_rel,
        m_me,
        BodyKind::Planet,
    )
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct TrojanPreset {
    pub giant_a_au: f64,
    pub mass_frac: f64,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct GapPreset {
    pub a_au: f64,
    pub width_au: f64,
}

impl BeltPreset {
    pub fn to_spec(&self) -> BeltSpec {
        let inner = self.inner.as_ref().map(|i| InnerBelt {
            a_lo_au: i.a_lo_au,
            a_hi_au: i.a_hi_au,
            mass_frac: i.mass_frac,
            e_sigma: 0.10,
            plaw: MASS_PLAW_ASTEROID,
            m_min: MASS_ASTEROID_MIN,
            m_max: MASS_ASTEROID_MAX,
        });
        let outer = self.outer.as_ref().map(|o| OuterBelt {
            populations: o
                .populations
                .iter()
                .map(|p| {
                    let kind = match p.kind.as_str() {
                        "cold_classical" => BodyOrigin::KuiperCold,
                        "hot_classical" => BodyOrigin::KuiperHot,
                        "resonant" => BodyOrigin::KuiperResonant,
                        "scattered" => BodyOrigin::KuiperScattered,
                        _ => BodyOrigin::KuiperCold,
                    };
                    OuterPopulation {
                        kind,
                        a_lo_au: p.a_lo_au,
                        a_hi_au: p.a_hi_au,
                        mass_frac: p.mass_frac,
                        e_sigma: p.e_sigma,
                        q_frac: p.q_frac,
                        plaw: MASS_PLAW_KUIPER,
                        m_min: MASS_KUIPER_MIN,
                        m_max: MASS_KUIPER_MAX,
                    }
                })
                .collect(),
        });
        BeltSpec {
            archetype: BeltArchetype::SolarType,
            total_mass: self.total_mass_earth,
            outer_mass: self.total_mass_earth,
            inner,
            outer,
            trojans: self
                .trojans
                .iter()
                .map(|t| Trojan {
                    giant_a_au: t.giant_a_au,
                    leading_frac: 0.5,
                    mass_frac: t.mass_frac,
                })
                .collect(),
            gaps: self
                .gaps
                .iter()
                .map(|g| Gap {
                    a_au: g.a_au,
                    width_au: g.width_au,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct StarPreset {
    #[serde(default)]
    pub name: Option<String>,
    /// Star mass in Earth masses.
    pub mass_earth: f64,
    pub age_gyr: f64,
    /// [Fe/H] in dex.
    pub metallicity: f64,
    /// Primary kind: "white_dwarf" | "neutron_star" | "black_hole" |
    /// "brown_dwarf" | "star". Defaults by mass.
    #[serde(default)]
    pub kind: Option<String>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct PlanetPreset {
    pub name: String,
    pub a_au: f64,
    pub e: f64,
    pub m_earth: f64,
    /// Bulk density in g/cm³. When absent, it is drawn (rocky or icy).
    #[serde(default)]
    pub density_gcc: Option<f64>,
    /// True for IAU dwarf planets (Ceres, Pluto, ...).
    #[serde(default)]
    pub is_dwarf: bool,
    /// Moons of this planet. Each moon's `a_au` is measured from the
    /// planet, not the star.
    #[serde(default)]
    pub moons: Vec<MoonPreset>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct MoonPreset {
    pub name: String,
    /// Semi-major axis about the *planet*, in AU.
    pub a_au: f64,
    #[serde(default)]
    pub e: f64,
    pub m_earth: f64,
    #[serde(default)]
    pub density_gcc: Option<f64>,
    #[serde(default)]
    pub is_dwarf: bool,
}

/// The Solar System, embedded from `assets/presets/solar_system.json` at build time.
pub fn solar_system_preset() -> SystemPreset {
    serde_json::from_str(include_str!("../assets/presets/solar_system.json"))
        .expect("embedded solar_system.json is malformed")
}

/// The preset library: (slug, parsed preset) pairs, in display order.
/// Cached after the first call.
pub fn load_presets() -> &'static [(String, SystemPreset)] {
    static LIB: OnceLock<Vec<(String, SystemPreset)>> = OnceLock::new();
    LIB.get_or_init(|| {
        let sources: &[(&str, &str)] = &[
            (
                "solar_system",
                include_str!("../assets/presets/solar_system.json"),
            ),
            (
                "trappist_1",
                include_str!("../assets/presets/trappist_1.json"),
            ),
            (
                "kepler_90",
                include_str!("../assets/presets/kepler_90.json"),
            ),
            (
                "hot_jupiter",
                include_str!("../assets/presets/hot_jupiter.json"),
            ),
            ("wasp_12b", include_str!("../assets/presets/wasp_12b.json")),
            ("hd_80606", include_str!("../assets/presets/hd_80606.json")),
            ("vega", include_str!("../assets/presets/vega.json")),
            ("hr_8799", include_str!("../assets/presets/hr_8799.json")),
            ("hd_40307", include_str!("../assets/presets/hd_40307.json")),
            (
                "gamma_cephei",
                include_str!("../assets/presets/gamma_cephei.json"),
            ),
            (
                "kepler_16",
                include_str!("../assets/presets/kepler_16.json"),
            ),
            ("algol", include_str!("../assets/presets/algol.json")),
            (
                "luhman_16",
                include_str!("../assets/presets/luhman_16.json"),
            ),
            (
                "psr_b1257_plus_12",
                include_str!("../assets/presets/psr_b1257_plus_12.json"),
            ),
            (
                "wd_1145_plus_017",
                include_str!("../assets/presets/wd_1145_plus_017.json"),
            ),
            (
                "cygnus_x1",
                include_str!("../assets/presets/cygnus_x1.json"),
            ),
        ];
        sources
            .iter()
            .map(|(slug, src)| {
                let preset: SystemPreset = serde_json::from_str(src)
                    .unwrap_or_else(|e| panic!("preset {} is malformed: {}", slug, e));
                (slug.to_string(), preset)
            })
            .collect()
    })
}

/// Look up a preset by slug. Returns an owned clone, so the caller can spawn
/// it while it holds a mutable borrow of `Bodies`.
pub fn load_preset_by_slug(slug: &str) -> Option<SystemPreset> {
    load_presets()
        .iter()
        .find(|(s, _)| s == slug)
        .map(|(_, p)| p.clone())
}

/// Belt bodies of a preset system. About 200 reads as a populated belt.
const PRESET_BELT_BUDGET: usize = 200;

/// Seed of the planet phases of the single-system Solar System preset.
const SOLAR_PHASE_SEED: u32 = 0xC0DE;

/// Spawn a preset system with its star at the origin (centre of mass at rest).
pub fn spawn_from_preset(b: &mut Bodies, preset: &SystemPreset) {
    spawn_from_preset_seeded(b, preset, SOLAR_PHASE_SEED);
}

/// Spawn a preset with its own phase seed (fresh phases each time).
pub fn spawn_preset_with_seed(b: &mut Bodies, preset: &SystemPreset, phase_seed: u32) {
    spawn_from_preset_seeded(b, preset, phase_seed);
}

/// Like `spawn_from_preset`, with the seed that picks each planet's phase.
fn spawn_from_preset_seeded(b: &mut Bodies, preset: &SystemPreset, phase_seed: u32) {
    let n_stars = 1 + preset.companions.len();
    let n_moons: usize = preset.planets.iter().map(|p| p.moons.len()).sum();
    *b = Bodies::with_capacity(n_stars + preset.planets.len() + n_moons);
    b.rng = XorShift::new(phase_seed);
    b.metallicity = preset.star.metallicity;

    let star_kind = parse_body_kind(preset.star.kind.as_deref(), preset.star.mass_earth);
    let star = b.push(0.0, 0.0, 0.0, 0.0, preset.star.mass_earth, star_kind);
    b.name[star] = preset.star.name.clone();
    b.age_gyr[star] = preset.star.age_gyr;
    b.set_mass(star, preset.star.mass_earth);

    for comp in &preset.companions {
        place_companion(b, comp, preset.star.age_gyr);
    }

    let p_type_cutoff = preset
        .companions
        .first()
        .map(|c| c.a_au * HW_MARGIN_P)
        .unwrap_or(f64::INFINITY);
    let snow_line = 2.7 * luminosity_of(b, star).max(1e-6).sqrt();

    for p in &preset.planets {
        let angle = b.rng.next_f64() * std::f64::consts::TAU;
        let idx = if p.a_au > p_type_cutoff {
            push_circumbinary(b, p.a_au, p.e, p.m_earth, angle)
        } else {
            push_orbit(b, star, p.a_au, p.e, p.m_earth, angle)
        };
        b.name[idx] = Some(p.name.clone());
        if p.is_dwarf {
            b.kind[idx] = BodyKind::DwarfPlanet;
        }
        let rho = p
            .density_gcc
            .unwrap_or_else(|| draw_density(&mut b.rng, p.a_au < snow_line));
        b.bulk_density[idx] = rho;
        b.r[idx] = b.radius_now(idx);

        for moon in &p.moons {
            let mangle = b.rng.next_f64() * std::f64::consts::TAU;
            let mi = push_orbit(b, idx, moon.a_au, moon.e, moon.m_earth, mangle);
            b.name[mi] = Some(moon.name.clone());
            if moon.is_dwarf {
                b.kind[mi] = BodyKind::DwarfPlanet;
            }
            b.bulk_density[mi] = moon
                .density_gcc
                .unwrap_or_else(|| draw_density(&mut b.rng, false));
            b.r[mi] = b.radius_now(mi);
        }
    }

    if let Some(belt_preset) = &preset.belts {
        let spec = belt_preset.to_spec();
        let belt_budget = belt_preset.belt_budget.unwrap_or(PRESET_BELT_BUDGET);
        let centre = (b.x[star], b.y[star], b.vx[star], b.vy[star], b.m[star]);
        let mut rng = std::mem::replace(&mut b.rng, XorShift::new(1));
        spawn_belts(b, centre, &spec, snow_line, belt_budget, &mut rng);
        b.rng = rng;
    }

    recenter_com(b);
    compute_accel(b);
    rebase_energy(b);
    b.refresh_parent_cache();
    dust::init_fields(b, None);
    for f in b.dust_fields.iter_mut() {
        for g in f.grains.iter_mut() {
            g.mass *= preset.dust_boost;
        }
    }
}

/// Copy every body of `src` into `dst`, shifted by `(cx, cy)`. Returns the
/// index range the copies now occupy. Evolution state comes along.
fn append_system(dst: &mut Bodies, src: &Bodies, cx: f64, cy: f64) -> (usize, usize) {
    let start = dst.len();
    for i in 0..src.len() {
        let idx = dst.push(
            src.x[i] + cx,
            src.y[i] + cy,
            src.vx[i],
            src.vy[i],
            src.m[i],
            src.kind[i],
        );
        dst.origin[idx] = src.origin[i];
        dst.bulk_density[idx] = src.bulk_density[i];
        dst.age_gyr[idx] = src.age_gyr[i];
        dst.m_init[idx] = src.m_init[i];
        dst.spin[idx] = src.spin[i];
        dst.imm[idx] = src.imm[i];
        dst.rlof[idx] = src.rlof[i];
        dst.coalesce_counter[idx] = src.coalesce_counter[i];
        dst.name[idx] = src.name[i].clone();
        dst.r[idx] = dst.radius_now(idx);
    }
    (start, dst.len())
}

/// Push a planet on an orbit about `sun`, which may sit anywhere and move.
fn push_orbit(
    b: &mut Bodies,
    sun: usize,
    a_au: f64,
    e: f64,
    m_me: f64,
    mean_anomaly: f64,
) -> usize {
    let a = a_au * au_in_gm();
    // Include the orbiter's mass so a close-in moon of a comparable-mass
    // planet (Charon about Pluto) gets the right period.
    let mu = G * (b.m[sun] + m_me);
    // The angle is a mean anomaly, so the phase is uniform in time.
    let nu = true_anomaly(solve_kepler(mean_anomaly, e), e);
    let (x, y, vx, vy) = kepler_state(mu, a, e, nu, 0.0);
    let (sx, sy, svx, svy) = (b.x[sun], b.y[sun], b.vx[sun], b.vy[sun]);
    b.push(sx + x, sy + y, svx + vx, svy + vy, m_me, BodyKind::Planet)
}

/// Most Solar Systems the UI slider can ask for.
pub const MAX_SOLAR_SYSTEMS: usize = 256;
/// Distance in Gm between neighbouring systems on the ring (about 200 AU).
/// Neptune is at 4500 Gm, so a neighbour's tide on it stays below 1 %.
pub const SOLAR_SYSTEM_SPACING_GM: f64 = 30_000.0;
/// Outer edge of one system (Neptune's apoapsis plus a margin), in Gm.
const SOLAR_SYSTEM_RADIUS_GM: f64 = 5_000.0;

/// Radius of the ring that holds `n` systems, in Gm. Zero for one system.
pub fn solar_ring_radius_gm(n: usize) -> f64 {
    if n <= 1 {
        0.0
    } else {
        SOLAR_SYSTEM_SPACING_GM / (2.0 * (std::f64::consts::PI / n as f64).sin())
    }
}

/// Distance from the centre to the farthest planet of the outer system, in Gm.
pub fn solar_extent_gm(n: usize) -> f64 {
    solar_ring_radius_gm(n) + SOLAR_SYSTEM_RADIUS_GM
}

/// Distance from body `i` to the host star it orbits, in Gm. A moon uses the
/// host of its planet. Falls back to the distance from the origin when the
/// parent cache is stale. O(1), so the renderer can call it for every body.
pub fn host_distance(b: &Bodies, i: usize) -> f64 {
    let fallback = b.x[i].hypot(b.y[i]);
    if !b.parent_cache_valid {
        return fallback;
    }
    let mut p = b.parent_cache.get(i).copied().flatten();
    if let Some(q) = p {
        if q < b.len() && !is_host_kind(b.kind[q]) {
            p = b.parent_cache.get(q).copied().flatten();
        }
    }
    match p {
        Some(h) if h < b.len() && is_host_kind(b.kind[h]) => {
            (b.x[i] - b.x[h]).hypot(b.y[i] - b.y[h])
        }
        _ => fallback,
    }
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

pub fn spawn_solar_system(b: &mut Bodies) {
    spawn_solar_system_seeded(b, SOLAR_PHASE_SEED);
}

/// Single Solar System with a caller-chosen phase seed, so "Reload (R)"
/// actually gives fresh planet phases (the previous implementation
/// always used `SOLAR_PHASE_SEED`).
pub fn spawn_solar_system_seeded(b: &mut Bodies, seed: u32) {
    let preset = solar_system_preset();
    spawn_from_preset_seeded(b, &preset, seed);
}

/// Spawn `count` copies of the Solar System (clamped to 1..=`MAX_SOLAR_SYSTEMS`).
///
/// Each copy is built by `spawn_from_preset` from `solar_system_preset()`.
/// The systems sit on a ring, evenly spaced `SOLAR_SYSTEM_SPACING_GM` apart.
/// Each one moves at the circular speed about the common centre, so the group
/// stays together and does not fall into one clump. Each system has its own
/// random planet phases. System 0 keeps the seed of the single-system preset.
pub fn spawn_solar_systems(b: &mut Bodies, count: usize) {
    spawn_solar_systems_impl(b, count);
}

/// `spawn_solar_systems`, and the ring radius and index range of every system.
fn spawn_solar_systems_impl(b: &mut Bodies, count: usize) -> (f64, Vec<(usize, usize)>) {
    let n = count.clamp(1, MAX_SOLAR_SYSTEMS);
    let preset = solar_system_preset();
    *b = Bodies::with_capacity((1 + preset.planets.len()) * n + 16);
    b.metallicity = preset.star.metallicity;
    let mut temps: Vec<Bodies> = Vec::with_capacity(n);
    for k in 0..n {
        // Each system has its own planet phases.
        let seed = SOLAR_PHASE_SEED.wrapping_add((k as u32).wrapping_mul(0x9E37_79B9));
        let mut temp = Bodies::with_capacity(1 + preset.planets.len());
        spawn_from_preset_seeded(&mut temp, &preset, seed);
        temps.push(temp);
    }
    let reaches: Vec<f64> = temps.iter().map(system_reach_gm).collect();
    let masses: Vec<f64> = temps.iter().map(|t| t.m.iter().sum::<f64>()).collect();
    let slots = cluster_slots(SOLAR_PHASE_SEED, &reaches, &masses);
    let mut ranges = Vec::with_capacity(n);
    for (k, temp) in temps.iter().enumerate() {
        let (x, y, vx, vy) = slots[k];
        let (start, end) = append_system(b, temp, x, y);
        for i in start..end {
            b.vx[i] += vx;
            b.vy[i] += vy;
        }
        ranges.push((start, end));
    }
    // Runtime RNG must not stay at the default seed.
    b.rng = XorShift::new(SOLAR_PHASE_SEED ^ 0xDEAD_BEEF);
    recenter_com(b);
    compute_accel(b);
    rebase_energy(b);
    // Orbit lines and colours read the parent of every body. Fill the cache
    // now, or the first frame draws every system about the first star.
    b.refresh_parent_cache();
    b.dust_fields.clear();
    dust::init_fields(b, Some(&ranges));
    // Effective scale of the cluster, for the caller (`t_multi_system_ring`
    // divides initial sun radii by this). With the Plummer layout there is
    // no ring, so this is the largest initial sun distance from the origin.
    let radius = slots
        .iter()
        .map(|(x, y, _, _)| x.hypot(*y))
        .fold(0.0f64, f64::max)
        .max(1.0);
    (radius, ranges)
}

// ---------------------------------------------------------------------------
// Orbit lines: adaptive point count from apparent screen size
// ---------------------------------------------------------------------------

fn orbit_trig() -> &'static (Vec<f64>, Vec<f64>) {
    static T: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    T.get_or_init(|| {
        let mut s = Vec::with_capacity(ORBIT_TABLE_SIZE);
        let mut c = Vec::with_capacity(ORBIT_TABLE_SIZE);
        for k in 0..ORBIT_TABLE_SIZE {
            let th = (k as f64 / ORBIT_TABLE_SIZE as f64) * std::f64::consts::TAU;
            s.push(th.sin());
            c.push(th.cos());
        }
        (s, c)
    })
}

#[allow(clippy::too_many_arguments)]
pub fn osculating_ellipse(
    b: &Bodies,
    i: usize,
    sx: f64,
    sy: f64,
    svx: f64,
    svy: f64,
    sm: f64,
    zoom: f64,
) -> Option<Vec<[f32; 2]>> {
    let dx = b.x[i] - sx;
    let dy = b.y[i] - sy;
    let dvx = b.vx[i] - svx;
    let dvy = b.vy[i] - svy;
    let r = (dx * dx + dy * dy).sqrt();
    if r < 1e-9 {
        return None;
    }
    let v2 = dvx * dvx + dvy * dvy;
    let mu = G * (sm + b.m[i]);
    let inv_a = 2.0 / r - v2 / mu;
    if inv_a <= 1e-9 {
        return None;
    }
    let a = 1.0 / inv_a;
    if a > 100_000.0 {
        return None;
    }
    let rv = dx * dvx + dy * dvy;
    let ex = ((v2 - mu / r) * dx - rv * dvx) / mu;
    let ey = ((v2 - mu / r) * dy - rv * dvy) / mu;
    let e = (ex * ex + ey * ey).sqrt();
    if e >= 0.9999 {
        return None;
    }
    let b_ax = a * (1.0 - e * e).sqrt();
    let rot = ey.atan2(ex);
    let cx = -a * ex;
    let cy = -a * ey;
    let cr = rot.cos();
    let sr = rot.sin();

    let circum_px = a * zoom.max(0.001) * std::f64::consts::TAU;
    let n_pts_raw = (circum_px / ORBIT_PIXELS_PER_SEG).round() as usize;
    let n_pts = n_pts_raw.clamp(32, ORBIT_TABLE_SIZE).next_power_of_two();
    let stride = ORBIT_TABLE_SIZE / n_pts;

    let (sin_t, cos_t) = orbit_trig();
    let mut pts = Vec::with_capacity(n_pts);
    for k in 0..n_pts {
        let idx = k * stride;
        let ct = cos_t[idx] * a;
        let st = sin_t[idx] * b_ax;
        let px = cx + ct * cr - st * sr + sx;
        let py = cy + ct * sr + st * cr + sy;
        pts.push([px as f32, py as f32]);
    }
    Some(pts)
}

pub fn compute_orbit_segments(
    b: &Bodies,
    zoom: f64,
    show_planets: bool,
    show_debris: bool,
) -> Vec<([f32; 2], [f32; 2], [u8; 4])> {
    if !show_planets && !show_debris {
        return Vec::new();
    }
    let n = b.len();
    if n < 2 {
        return Vec::new();
    }
    let Some(si) = primary_index(b) else {
        return Vec::new();
    };
    let sx = b.x[si];
    let sy = b.y[si];
    let svx = b.vx[si];
    let svy = b.vy[si];
    let sm = b.m[si];
    let hosts = host_list(b);

    let n_planets = if show_planets {
        (0..n)
            .filter(|&i| i != si && !(b.origin[i].is_belt() || b.origin[i].is_rubble()))
            .count()
    } else {
        0
    };
    let n_debris = if show_debris {
        (0..n)
            .filter(|&i| i != si && (b.origin[i].is_belt() || b.origin[i].is_rubble()))
            .count()
    } else {
        0
    };
    let planet_stride = n_planets.div_ceil(ORBIT_MAX_BODIES).max(1);
    let debris_stride = if n_debris <= 256 { 1 } else { n_debris / 256 };

    let mut segs = Vec::with_capacity(n_planets.min(ORBIT_MAX_BODIES) * 64 + 256 * 64);
    let mut seen_planet = 0usize;
    let mut seen_debris = 0usize;

    for i in 0..n {
        if i == si {
            continue;
        }
        let is_debris = b.origin[i].is_belt() || b.origin[i].is_rubble();
        if is_debris {
            if !show_debris {
                continue;
            }
            let skip = !seen_debris.is_multiple_of(debris_stride);
            seen_debris += 1;
            if skip {
                continue;
            }
        } else {
            if !show_planets {
                continue;
            }
            let skip = !seen_planet.is_multiple_of(planet_stride);
            seen_planet += 1;
            if skip {
                continue;
            }
        }

        let (rx, ry, rvx, rvy, rm) = orbit_reference(b, &hosts, i, (sx, sy, svx, svy, sm));
        if let Some(pts) = osculating_ellipse(b, i, rx, ry, rvx, rvy, rm, zoom) {
            let color = if is_debris {
                [85u8, 85u8, 90u8, 110u8]
            } else {
                let d = host_distance(b, i) as f32;
                let t = (d / 800.0).clamp(0.0, 1.0);
                [
                    ((0.4 + 0.4 * t) * 255.0) as u8,
                    (0.55 * 255.0) as u8,
                    ((1.0 - 0.5 * t) * 255.0) as u8,
                    30u8,
                ]
            };
            let m = pts.len();
            for k in 0..m {
                segs.push((pts[k], pts[(k + 1) % m], color));
            }
        }
    }
    segs
}

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

pub struct XorShift {
    s: u32,
}

impl XorShift {
    pub fn new(seed: u32) -> Self {
        Self { s: seed.max(1) }
    }
    pub fn next_u32(&mut self) -> u32 {
        self.s = self.s.wrapping_add(0x6d2b79f5);
        let mut t = self.s;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        t ^ (t >> 14)
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64) / 4_294_967_296.0
    }
}

#[cfg(test)]
mod tests;
