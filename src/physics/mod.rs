//! N-body physics: structure-of-arrays bodies, direct / Barnes–Hut gravity,
//! a 4th-order Yoshida symplectic integrator for small systems and block
//! leapfrog for large ones, swept collision detection at true physical radii,
//! Roche-limit tidal disruption, debris-disk coalescence, stellar evolution
//! (main sequence → giant branch → white dwarf / neutron star / black hole),
//! energy bookkeeping and an event log.
//!
//! Units: length 1 Gm (1e6 km), mass 1 Earth mass, G = 7.4. One simulation
//! time unit is about 50 days (see `time_unit_s`).

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
pub mod save;
pub mod spawn;
#[cfg(test)]
mod tests;
pub mod tides;
pub mod yarkovsky;

pub use barnes_hut::*;
// `collisions` has no public items, so a plain `use` re-exports it to siblings.
use collisions::*;
pub use debris::*;
pub use dust::*;

/// Speed of light in simulation units (≈1.29×10⁶).
///
/// Derived once from `time_unit_s()` (≈4.31×10⁶ s) and the length unit
/// (`LEN_KM × 1000` m). Used by the post-Newtonian corrections and by
/// Poynting-Robertson drag in `dust.rs`.
pub fn c_sim() -> f64 {
    static C: OnceLock<f64> = OnceLock::new();
    *C.get_or_init(|| 299_792_458.0 * time_unit_s() / (LEN_KM * 1000.0))
}
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
pub use save::*;
pub use spawn::*;
pub use tides::*;
pub use yarkovsky::*;

use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::OnceLock;

static DUST_VERSION_COUNTER: AtomicU64 = AtomicU64::new(1);

// ---- Physics constants -----------------------------------------------------

/// Gravitational constant in simulation units. One time unit is 49.9 days
/// (see `time_unit_s`), so a body at 1 AU around 1 M☉ has an orbital period
/// of about 7.32 time units (1 year).
pub const G: f64 = 7.4;

/// Squared gravitational softening length: 1e-4 Gm (100 km). At the
/// distance of Phobos from Mars (9.4e-3 Gm) the force error is about
/// 1.5·(ε/r)² = 0.017 %. The old value (1000 km) gave 1.7 %. For a neutron
/// star (1e-5 Gm) or a stellar-mass black hole, this length is still larger
/// than the body, so close encounters stay regularised.
pub const SOFT2: f64 = 1e-8;

// ---- Integration -----------------------------------------------------------

/// Base step. Bodies on the finest rung advance by `DT`; a body on rung `r`
/// advances by `DT / 2^r`.
pub const DT: f64 = 1.0 / 480.0;

/// Yoshida (1990) 4th-order symplectic coefficients. The composite step uses
/// three drift–kick–drift sub-steps with weights `(Y_W1, Y_W0, Y_W1)`.
pub const Y_W1: f64 = 1.3512071919596578;

pub const Y_W0: f64 = -1.7024143839193153;

/// Barnes–Hut opening angle. 0.35 is conservative (~0.1 % force error);
/// 0.5 is the standard default (~1 % error) and roughly 1.5–2× cheaper at
/// n ≥ 600. For a visual sim the extra error is invisible.
pub const BH_THETA: f64 = 0.5;

/// Below this body count the exact O(n²) pair sum is used instead of a tree
/// (serial below `PAR_DIRECT_MIN`, parallel above it).
pub const BH_THRESHOLD: usize = 600;

/// Direct-sum floor: below this body count the inner loop stays serial even
/// if it is selected as the parallel path.
pub const PAR_DIRECT_MIN: usize = 150;

/// With fewer active bodies than this, a direct sum beats building a tree,
/// even if the system as a whole is large.
pub const DIRECT_ACTIVE_MAX: usize = 64;

/// Minimum work (active bodies × total bodies) before spawning rayon threads
/// pays for itself. At n = 10 000, a single active body is ~50 µs of work,
/// well above rayon's ~10 µs dispatch cost, so this threshold alone is the
/// right gate. At small n it keeps short ticks serial.
pub const PAR_WORK_MIN: usize = 20_000;

/// Block time steps: body i takes steps of `DT / 2^rung`, at most `MAX_RUNG`
/// halvings below the base step.
pub const MAX_RUNG: u8 = 8;

/// Per-pair dynamical-time limit: dt ≤ ETA_GRAV · sqrt(d³ / (G M)).
pub const ETA_GRAV: f64 = 0.015;

/// Per-pair crossing-time limit: dt ≤ ETA_VEL · d / v_rel. Catches fast
/// fly-bys that the dynamical-time limit alone misses.
pub const ETA_VEL: f64 = 0.03;

/// Systems this small and quiet use the 4th-order Yoshida step instead of
/// block leapfrog (whose per-body sub-steps have lower order).
pub const YOSHIDA_MAX_N: usize = 1500;

/// Largest work for the Yoshida path in one `advance`, in pair evaluations:
/// `n² · 3 · 2^kmax`. Above this, `advance` uses the block stepper.
pub const YOSHIDA_WORK_MAX: usize = 20_000_000;

// ---- Unit conversion -------------------------------------------------------

/// Astronomical unit in km (IAU 2012).
pub const AU_KM: f64 = 1.495_978_707e8;

/// One simulation length unit in km.
pub const LEN_KM: f64 = 1e6;

/// One simulation mass unit in kg (Earth mass).
pub const MASS_KG: f64 = 5.9722e24;

/// Gravitational constant in SI. Used only by `time_unit_s`.
pub const G_SI: f64 = 6.6743e-11;

// ---- Mass–radius relations -------------------------------------------------

/// Solar mass in Earth masses.
pub const M_SUN_ME: f64 = 332_946.0;

/// Solar radius in Gm.
pub const R_SUN_GM: f64 = 0.696;

/// Earth radius in Gm.
pub const R_EARTH_GM: f64 = 0.006371;

/// Main-sequence mass–radius exponent: R ∝ M^0.8.
pub const STAR_MR_EXP: f64 = 0.8;
/// Mass–radius exponent above 1 M☉ (CNO-cycle stars expand more slowly).
pub const STAR_MR_EXP_HI: f64 = 0.57;

/// Schwarzschild radius in Gm per Earth mass: R_s = 2GM/c².
pub const SCHWARZSCHILD_GM_PER_ME: f64 = 8.87e-12;

/// Mass of a 1 M☉ star in simulation units.
pub const STAR_M: f64 = M_SUN_ME;

/// Minimum mass for sustained hydrogen fusion: 0.08 M☉.
pub const STAR_M_MIN: f64 = 0.08 * M_SUN_ME;

/// Minimum mass for a stellar-mass black hole: 3 M☉.
pub const BH_M_MIN: f64 = 3.0 * M_SUN_ME;

/// Jupiter's mass in Earth masses.
pub const JUPITER_MASS: f64 = 318.0;

// ---- Brown dwarfs: deuterium-fusing, ~13–80 M♃, roughly Jupiter-sized ------

pub const BD_M_MIN: f64 = 13.0 * JUPITER_MASS;

pub const BD_M_MAX: f64 = 80.0 * JUPITER_MASS;

/// Brown dwarf radius in Earth radii at the reference mass.
pub const BD_R_EARTH: f64 = 11.21;

/// Brown dwarf mass–radius exponent (slightly negative: more massive is
/// slightly smaller).
pub const BD_R_EXP: f64 = -0.04;

// ---- White dwarfs: degenerate carbon / oxygen cores ------------------------

pub const WD_M_MIN: f64 = 0.5 * M_SUN_ME;

pub const WD_M_MAX: f64 = 1.4 * M_SUN_ME;

pub const WD_R_EARTH: f64 = 1.0;

/// White dwarf mass–radius exponent: R ∝ M^(-1/3).
pub const WD_R_EXP: f64 = -1.0 / 3.0;

// ---- Neutron stars ---------------------------------------------------------

pub const NS_M_MIN: f64 = 1.4 * M_SUN_ME;

pub const NS_M_MAX: f64 = 2.5 * M_SUN_ME;

/// Typical neutron-star radius: 10 km = 1e-5 Gm.
pub const NS_R_GM: f64 = 1e-5;

/// Largest mass still classed as a dwarf planet: 0.005 M⊕ (≈ Ceres is 1.6e-4).
pub const DWARF_M_MAX: f64 = 0.005;

/// Smallest belt object classed as a dwarf planet, roughly Pluto–Eris scale.
pub const BELT_DWARF_M_MIN: f64 = 1e-4;

/// Black holes capture bodies that pass within this many Schwarzschild radii.
pub const BH_CAPTURE_HORIZONS: f64 = 3.0;

/// Floor on the black-hole capture radius in Gm, so a tiny PBH still eats.
pub const BH_MIN_CAPTURE: f64 = 1e-3;

// ---- Stellar evolution -----------------------------------------------------
//
// Evolution is OFF until `Bodies::evolution_rate` is set (see
// `set_evolution_rate`). A star keeps `BodyKind::Star` through its whole life;
// its phase follows from its age and initial mass (`star_state`). At the end
// of its life it becomes a white dwarf, neutron star or black hole
// (`remnant_of`).

/// Radius growth over the main sequence (age 0 = the preset's present state).
pub const EVO_MS_RADIUS_GROWTH: f64 = 0.4;

/// Luminosity growth over the main sequence.
pub const EVO_MS_LUM_GROWTH: f64 = 0.5;

/// Length of the giant phase as a fraction of the main-sequence lifetime.
pub const EVO_POST_MS_FRAC: f64 = 0.15;

/// Same, for stars at or above `EVO_MASSIVE_M` (they leave the MS faster).
pub const EVO_POST_MS_FRAC_MASSIVE: f64 = 0.10;

/// Stars at or above this initial mass end as supernovae (NS or BH).
pub const EVO_MASSIVE_M: f64 = 8.0 * M_SUN_ME;

/// Stars at or above this initial mass leave a black hole, not a neutron star.
pub const EVO_BH_PROGENITOR_M: f64 = 20.0 * M_SUN_ME;

/// Peak giant radius of a 1 M☉ star (scales with sqrt(mass)), in R☉.
pub const EVO_GIANT_R_SUN: f64 = 200.0;

/// Peak supergiant radius of a 20 M☉ star (scales with sqrt(mass)), in R☉.
pub const EVO_SUPERGIANT_R_SUN: f64 = 800.0;

pub const EVO_GIANT_T: f64 = 3500.0;

pub const EVO_SUPERGIANT_T: f64 = 3600.0;

/// Fraction of the giant phase over which the surface cools to its final
/// temperature.
pub const EVO_COOL_FRAC: f64 = 0.25;

/// Fraction of its mass a massive star loses in winds before it explodes.
pub const EVO_WIND_LOSS_MASSIVE: f64 = 0.2;

/// Natal kick: per-component Maxwellian sigma for a neutron star (Hobbs et al.
/// 2005).
pub const SN_KICK_SIGMA_KMS: f64 = 265.0;

/// Demo rate: a 1 M☉ star would live about 10 time units (about 500 days).
/// Not used by the app. `main.rs` sets the real rate, `time_unit_s() /
/// SECONDS_PER_GYR`, times `aging_mult` (default 1.0).
pub const EVO_RATE_DEMO: f64 = 1.0;

// White dwarf cooling: Mestel law, L ∝ M t^(-7/6) with t = cooling age.

/// Luminosity in L☉ of a 0.6 M☉ white dwarf at a cooling age of 1 Gyr.
pub const WD_COOL_L_REF: f64 = 2e-4;

pub const WD_COOL_EXP: f64 = 7.0 / 6.0;

/// Offset that keeps a new white dwarf's luminosity finite (10 000 yr).
pub const WD_COOL_T0_GYR: f64 = 1e-5;

/// Cooling age of a white dwarf made by `push`. A 0.6 M☉ star of this age has
/// L = 1e-3 L☉.
pub const WD_DEFAULT_AGE_GYR: f64 = 0.25;

// Brown dwarf cooling: L ∝ M^2.64 t^(-1.3).

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
/// however long its total lifetime is (a 0.5 M☉ star lives ~56 Gyr).
pub const AGE_UNIVERSE_GYR: f64 = 13.8;

// Mass transfer in binaries (Roche-lobe overflow).

/// Transfer rate = coeff · M_donor / t_dyn · x³, with x = (R - R_lobe) / R.
pub const MT_RATE_COEFF: f64 = 10.0;

/// A donor loses at most this fraction of its mass in one step.
pub const MT_MAX_FRAC_PER_STEP: f64 = 0.02;

/// Fraction of the transferred mass that a white dwarf, neutron star or black
/// hole keeps (the rest is ejected from the system).
pub const MT_EFF_COMPACT: f64 = 0.5;

/// A star can only overflow if a companion is within this many stellar radii.
pub const MT_NEAR_RADII: f64 = 25.0;

// ---- Realistic initial conditions -----------------------------------------

pub const MAX_PLANETS: usize = 15;

/// Fraction of main-sequence stars that host a compact chain of planets.
const P_COMPACT_CHAIN: f64 = 0.40;

/// Fraction of giant-planet hosts whose giants form tightly packed and are
/// dynamically unstable (2–4 giants within ~2–3.5 mutual Hill radii). The
/// N-body integrator then scatters, ejects or merges them: a "young unstable
/// system". The other hosts keep the stable 1.5–2.2 period spacing.
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

pub const MASS_PLAW_ASTEROID: f64 = 11.0 / 6.0;

pub const MASS_PLAW_KUIPER: f64 = 11.0 / 6.0;

pub const MASS_ROCKY_MIN: f64 = 0.1;

pub const MASS_ROCKY_MAX: f64 = 5.0;

pub const MASS_GIANT_MIN: f64 = 50.0;

pub const MASS_GIANT_MAX: f64 = 3000.0; // about 9.4 Jupiter masses, below BD_M_MIN

pub const MASS_ASTEROID_MIN: f64 = 1e-10;

pub const MASS_ASTEROID_MAX: f64 = 1e-3;

pub const MASS_KUIPER_MIN: f64 = 1e-11;

pub const MASS_KUIPER_MAX: f64 = 0.013;

// ---- Binary stars ---------------------------------------------------------

/// Binary period distribution (Raghavan et al. 2010): log-normal in log10(days).
pub const BIN_LOGP_MEAN: f64 = 5.03;

pub const BIN_LOGP_SIGMA: f64 = 2.28;

/// Shortest binary period generated, in days. Below about 2 d the block time
/// steps (`MAX_RUNG` halvings) can no longer resolve the orbit.
pub const BIN_P_MIN_D: f64 = 3.0;

/// Widest binary generated. Keeps both stars (and S-type planets around the
/// companion) well inside `CULL_DIST`.
pub const BIN_A_MAX_AU: f64 = 300.0;

/// Binaries tighter than this period are tidally circularised (e = 0).
pub const BIN_CIRC_P_D: f64 = 7.0;

/// Periastron must clear this many summed stellar radii.
pub const BIN_PERI_CLEAR: f64 = 2.0;

/// Systems with fewer bodies than this stay single.
pub const BIN_MIN_BODIES: usize = 6;

/// S-type planets stay inside this fraction of the Holman–Wiegert limit,
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

// ---- Culling ---------------------------------------------------------------

/// Distance (Gm) at which ordinary escaped bodies are removed. Rogues are
/// exempt and continue out to `HARD_CULL_DIST`.
pub const CULL_DIST: f64 = 100_000.0;

/// Absolute maximum distance (Gm): even rogues and interstellar objects are
/// removed here.
pub const HARD_CULL_DIST: f64 = 1_000_000.0;

/// Hard cap on the number of bodies. `Bodies` panics or truncates above this.
pub const MAX_BODIES: usize = 256_000;

// ---- Collisions & debris ---------------------------------------------------

/// Lifetime of the expanding-ring flash drawn on a collision, in time units.
pub const FLASH_LIFE: f64 = 0.6;

/// Approach speed (as a fraction of escape velocity) below which two bodies
/// merge rather than produce debris.
pub const MERGE_VEL_RATIO: f64 = 0.9;

/// Fractional impact-parameter weight used to blend impact angles in mergers.
pub const MERGE_B_WEIGHT: f64 = 0.5;

/// Fraction of the smaller body's mass thrown off as debris in a head-on
/// impact.
pub const DEBRIS_MAX_FRAC_HEADON: f64 = 0.6;

/// Impact parameter above which an impact is a hit-and-run grazing pass.
pub const HIT_RUN_B: f64 = 0.8;

/// Minimum approach speed (as a fraction of escape velocity) for hit-and-run.
pub const HIT_RUN_VEL_RATIO: f64 = 1.2;

/// Mass fraction lost on a hit-and-run.
pub const HIT_RUN_LOSS: f64 = 0.25;

/// Coefficient of restitution for a hit-and-run bounce.
pub const HIT_RUN_RESTITUTION: f64 = 0.5;

/// Leinhardt & Stewart (2012) catastrophic-disruption scale factor for stars.
pub const LS_C_STAR: f64 = 1.9;

/// Leinhardt & Stewart (2012) velocity-coupling parameter `μ̄`. It sets the
/// exponent `2/(3μ̄) − 1` (about 0.80) of the mass-ratio factor in `Q*_RD`.
/// Check the value against the paper for the regime that you model.
pub const LS_MU_BAR: f64 = 0.37;

/// Power-law slope of the fragment mass distribution: dN/dm ∝ m^-slope.
pub const FRAG_SLOPE: f64 = 2.85;

/// Dynamic range (largest / smallest fragment mass) produced by one collision.
pub const FRAG_DYN_RANGE: f64 = 100.0;

/// Baseline fragment count, and the scaling of that count with impact energy.
pub const FRAG_COUNT_BASE: f64 = 3.0;

pub const FRAG_COUNT_SCALE: f64 = 6.0;

pub const FRAG_COUNT_EXP: f64 = 0.5;

/// Extra fragment count multiplier when the impact feeds a disk.
pub const FRAG_COUNT_DISK_GAIN: f64 = 1.5;

/// Hard cap on fragments produced by one collision.
pub const FRAG_COUNT_MAX: usize = 64;

/// Fraction of a disrupted body's mass that goes into the coalescence disk.
pub const DISK_MASS_FRAC: f64 = 0.3;

/// Cap on the number of disk bodies per host.
pub const DISK_MAX_N: usize = 1500;

/// Innermost disk orbital radius and radial span, in simulation units.
pub const DISK_R_MIN: f64 = 2.5;

pub const DISK_R_SPAN: f64 = 5.0;

/// Threshold below which a body counts as "small" for the mass-radius law.
pub const SMALL_BODY_M: f64 = 0.1;

/// Jupiter radius in Earth radii.
pub const JUPITER_R_EARTH: f64 = 11.21;

/// Gas-giant mass–radius exponent: R ∝ M^-0.04.
pub const GIANT_R_EXP: f64 = -0.04;

/// Spread of debris kick velocities, relative to escape velocity.
pub const DEBRIS_VEL_RANGE: f64 = 2.5;

/// Maximum fraction of the parent's mass that can end up as debris.
pub const DEBRIS_MAX_FRAC: f64 = 0.95;

/// Smallest mass (M⊕) that a debris fragment will carry.
pub const DEBRIS_MIN_MASS: f64 = 0.002;

/// Debris count: DEBRIS_COUNT_BASE + DEBRIS_COUNT_SPAN · (impact-energy ^ 0.5).
pub const DEBRIS_COUNT_BASE: f64 = 3.0;

pub const DEBRIS_COUNT_SPAN: f64 = 7.0;

pub const DEBRIS_POWER_LAW: f64 = 1.5;

/// Minimum mass weight when drawing fragment masses from the power law.
pub const DEBRIS_WEIGHT_MIN: f64 = 0.3;

/// Kick velocity floor and spread, relative to escape velocity.
pub const DEBRIS_KICK_MIN: f64 = 0.3;

pub const DEBRIS_KICK_SPAN: f64 = 0.7;

/// Random scatter in debris ejection angle, in radians.
pub const DEBRIS_ANGLE_JIT: f64 = 0.4;

/// Angular gap (fraction of 2π) between adjacent debris fragments.
pub const DEBRIS_GAP_FRAC: f64 = 0.1;

/// Time (simulation units) a fresh fragment is immune to further collisions.
pub const IMMUNITY_TIME: f64 = 0.4;

// Rubble coalescence: debris-rubble bodies (small, `is_rubble` origin) merge
// when their relative speed is below `RUBBLE_COALESCE_VRATIO` × escape; they
// bounce elastically below `RUBBLE_BOUNCE_VRATIO` × escape and merge into a
// real body once their combined mass exceeds `RUBBLE_PROMOTE_MASS`.

pub const RUBBLE_COALESCE_VRATIO: f64 = 1.0;

pub const RUBBLE_BOUNCE_VRATIO: f64 = 3.0;

pub const RUBBLE_BOUNCE_E: f64 = 0.35;

pub const RUBBLE_BOUNCE_IMMUNITY: f64 = 0.05;

pub const RUBBLE_PROMOTE_MASS: f64 = 0.05;

/// Fraction of a debris body's Hill radius within which capture can occur.
pub const DEBRIS_CAPTURE_HILL_FRAC: f64 = 0.3;

/// Capture search runs for this many steps before giving up.
pub const DEBRIS_CAPTURE_STEPS: u32 = 200;

/// Capture succeeds below this fraction of the Hill velocity.
pub const DEBRIS_CAPTURE_VEL_RATIO: f64 = 0.5;

/// Maximum mass a coalescing debris clump can reach before promotion.
pub const DEBRIS_COALESCE_MAX_MASS: f64 = 0.05;

/// Neighbour search radius, in debris radii.
pub const DEBRIS_SEARCH_RADII: f64 = 30.0;

// ---- Tidal disruption ------------------------------------------------------

/// Below this mass, bodies are not tidally disrupted (they hit or bounce).
pub const TIDAL_MIN_MASS: f64 = 0.05;

/// Body density must exceed the host's by this factor before Roche applies.
pub const TIDAL_MASS_RATIO: f64 = 20.0;

/// Mass threshold for "giant" classification.
pub const GIANT_MASS: f64 = 50.0;

/// Minimum mass for an ejected body to be promoted to a rogue planet
/// (rather than culled at `CULL_DIST` like escaped debris). Equal to
/// `GIANT_MASS` — microlensing surveys detect free-floating planets in the
/// gas-giant mass range, not smaller.
pub const ROGUE_MIN_MASS: f64 = GIANT_MASS;

/// Radial velocity kick of a fragment relative to the disruption point.
pub const DISRUPT_KICK: f64 = 0.2;

/// Fragment separation (in parent radii) along the disruption stream.
pub const DISRUPT_STEP: f64 = 2.6;

/// Random positional jitter of a fragment, in parent radii.
pub const DISRUPT_JITTER: f64 = 0.5;

/// Fragment count: DISRUPT_FRAG_BASE + DISRUPT_FRAG_LOG · log10(M_parent / M_min).
pub const DISRUPT_FRAG_BASE: f64 = 4.0;

pub const DISRUPT_FRAG_LOG: f64 = 2.0;

// ---- Roche limits ----------------------------------------------------------
//
// A rigid body disrupts at ROCHE_RIGID · R_host; a fluid body at
// ROCHE_FLUID · R_host. The fluid case depends on the density ratio; the
// rigid case is density-independent.

pub const ROCHE_RIGID: f64 = 1.26;

pub const ROCHE_FLUID: f64 = 2.44;

pub const ROCHE_RHO_FLUID: f64 = 1.5;

pub const ROCHE_RHO_RIGID: f64 = 4.0;

/// Earth's bulk density in g/cm³, used as the fallback for non-planets.
pub const EARTH_RHO_GCC: f64 = 5.51;

/// Moon-forming impacts produce debris inside this fraction of the Hill radius.
pub const MOON_HILL_FRAC: f64 = 0.5;

// ---- Dust ------------------------------------------------------------------
//
// Dust is a statistical field per system, tracked in `DustField` rather than
// as bodies. It does not participate in gravity, energy bookkeeping or the
// collision sweep, so it never changes body mass or momentum.

/// Dust added by a collision, as a fraction of its debris mass. Bonus
/// population: the dust is not taken from the debris, so body bookkeeping is
/// unchanged.
pub const DUST_YIELD_FRAC: f64 = 0.02;

/// Boost on the Poynting–Robertson drag speed. With β folded in, a grain
/// drifts from 1 AU in about 10 yr at 100.
pub const PR_DRAG_BOOST: f64 = 100.0;

/// When true, the demo speed-ups use their physical values: PR drag and
/// tides run with boost 1, and belt grinding is off (belt bodies still
/// collide through the collision code). Default: true.
static PHYSICAL_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_physical_mode(on: bool) {
    PHYSICAL_MODE.store(on, AtomicOrdering::Relaxed);
}

pub fn physical_mode() -> bool {
    PHYSICAL_MODE.load(AtomicOrdering::Relaxed)
}

/// Poynting–Robertson boost in use: 1.0 in physical mode.
pub fn pr_drag_boost() -> f64 {
    if physical_mode() {
        1.0
    } else {
        PR_DRAG_BOOST
    }
}

/// Tidal boost in use: 1.0 in physical mode.
pub fn tidal_boost() -> f64 {
    if physical_mode() {
        1.0
    } else {
        TIDAL_BOOST
    }
}

/// Belt grinding rate in use: 0.0 in physical mode.
pub fn belt_grind_per_orbit() -> f64 {
    if physical_mode() {
        0.0
    } else {
        BELT_GRIND_PER_ORBIT
    }
}

/// Initial dust mass injected per belt source body, as a fraction of the
/// body's own mass. The physical steady-state zodiacal cloud is roughly
/// belt mass × (PR-drag lifetime / collision timescale) ≈ 10⁻⁸ M⊕/M⊕;
pub const DUST_TO_BELT_MASS: f64 = 1e-7;

/// Advance the dust field every N steps (dust is slower than the bodies).
pub const DUST_STEP_INTERVAL: u64 = 2;

// ---- Comets -----------------------------------------------------------------

/// Sublimation rate coefficient: mass loss scales with stellar flux.
pub const COMET_SUBLIMATE_FLUX_COEFF: f64 = 1e-3;

/// Cap on the fraction of a comet's mass that can sublimate in one step, so
/// a comet near the star does not disappear in a single tick.
pub const COMET_MAX_FRAC_PER_STEP: f64 = 1e-4;

/// A comet below this mass (M⊕) is treated as exhausted and removed.
pub const COMET_DEAD_MASS: f64 = 1e-13;

/// Maximum comet-trail dust grains one comet may emit over its lifetime.
pub const COMET_GRAIN_CAP: usize = 5_000;

// ---- Miscellaneous ---------------------------------------------------------

/// Body count above which spatial hashing is used for the collision sweep.
pub const HASH_THRESHOLD: usize = 64;

/// Above this body count, energy bookkeeping is skipped (O(n²) exact energy).
pub const ENERGY_MAX_N: usize = 1500;

/// Maximum events retained in the ring buffer.
pub const EVENT_LOG_MAX: usize = 200;

// Mass class boundaries (Earth masses) used by naming and colour.

pub const CLASS_ROCKY_MAX: f64 = 5.0;

pub const CLASS_SE_MAX: f64 = 10.0;

pub const CLASS_ICE_MAX: f64 = 50.0;

pub const CLASS_GAS_MAX: f64 = 318.0;

// Stellar class boundaries (solar masses).

pub const CLASS_STAR_M_MAX: f64 = 0.45;

pub const CLASS_STAR_K_MAX: f64 = 0.8;

pub const CLASS_STAR_G_MAX: f64 = 1.2;

pub const CLASS_STAR_F_MAX: f64 = 1.5;

pub const CLASS_STAR_A_MAX: f64 = 2.5;

/// Seconds per simulation time unit, derived from `G`, `LEN_KM` and `MASS_KG`.
pub fn time_unit_s() -> f64 {
    (G * (LEN_KM * 1000.0).powi(3) / (G_SI * MASS_KG)).sqrt()
}

// ---- Enums -----------------------------------------------------------------

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
    /// A comet on a bound orbit (Halley, 67P, ...). Sublimates ice when inside
    /// its host's snow line, feeding the dust field.
    Comet,
    /// An unbound body passing through the system. Spawned by the user at the
    /// cull radius inbound; `cull_escaped` removes it on the way out and logs
    /// "left the system" rather than "escaped".
    Interstellar,
    /// A planet-mass unbound body. Either authored into a preset, or promoted
    /// from a heavy (`m >= ROGUE_MIN_MASS`) ejected planet. Exempt from the
    /// `CULL_DIST` cull so it stays visible through the outer system; culled
    /// at `HARD_CULL_DIST` like everything else.
    RoguePlanet,
}

impl BodyOrigin {
    #[inline]
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
    #[inline]
    pub fn is_rubble(self) -> bool {
        matches!(
            self,
            BodyOrigin::ImpactDebris | BodyOrigin::TidalDebris | BodyOrigin::BackgroundDebris
        )
    }
    #[inline]
    pub fn is_comet(self) -> bool {
        self == BodyOrigin::Comet
    }
}

/// Taxonomic class of a small body. Drives colour in `render.rs` and the bulk
/// density at spawn time. Stars and remnants ignore it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyClass {
    /// Carbonaceous — dark, ~75 % of asteroids.
    C,
    /// Silicaceous — stony, ~17 % of asteroids.
    S,
    /// Metallic — iron-nickel.
    M,
    /// Basaltic / volcanic — Vesta family.
    V,
    /// Water-ice dominated (Kuiper, Trojan, icy moons).
    Icy,
    /// Silicate rock (inner planets, Moon, Ceres).
    Rocky,
}

// ---- Gravity helpers -------------------------------------------------------

/// Is this kind a body that other bodies can orbit (star or remnant)?
#[inline]
pub fn is_host_kind(k: BodyKind) -> bool {
    matches!(
        k,
        BodyKind::Star
            | BodyKind::BrownDwarf
            | BodyKind::WhiteDwarf
            | BodyKind::NeutronStar
            | BodyKind::BlackHole
    )
}

/// Piecewise planet mass–radius law, in Earth radii.
///
/// * `m < SMALL_BODY_M`: constant-density extrapolation from the
///   `SMALL_BODY_M` threshold, R ∝ m^(1/3).
/// * `m < 2.04 M⊕`: rocky regime, R ∝ m^0.279.
/// * `m ≥ 2.04 M⊕`: min(Neptunian R ∝ m^0.589, Jovian R ∝ m^-0.04). A gas
///   giant is slightly *smaller* than a Neptune-mass body of equal mass, and
///   the `min` picks whichever branch applies.
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

/// Bulk density in g/cm³ that makes `radius_from_density` reproduce
/// `planet_true_r` for a planet. Non-planets fall back to Earth's density.
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

/// Radius in Gm of a body of mass `m` (Earth masses) with uniform bulk density
/// `rho_gcc` (g/cm³).
pub fn radius_from_density(m: f64, rho_gcc: f64) -> f64 {
    const K: f64 = 3.0 * MASS_KG / (4.0 * std::f64::consts::PI);
    const INV_GM_PER_M: f64 = 1.0 / (LEN_KM * 1000.0);
    let rho_kg_m3 = (rho_gcc * 1000.0).max(1e-6);
    (K * m / rho_kg_m3).cbrt() * INV_GM_PER_M
}

/// True radius in Gm. For a black hole this is the Schwarzschild radius; for
/// a neutron star it is the fixed `NS_R_GM`; for a planet it is
/// `planet_true_r`. Used for collisions and Roche geometry.
pub fn physical_radius(kind: BodyKind, m: f64) -> f64 {
    match kind {
        BodyKind::Star => {
            let x = m / M_SUN_ME;
            R_SUN_GM * x.powf(if x < 1.0 { STAR_MR_EXP } else { STAR_MR_EXP_HI })
        }
        BodyKind::BrownDwarf => R_EARTH_GM * BD_R_EARTH * (m / JUPITER_MASS).powf(BD_R_EXP),
        BodyKind::WhiteDwarf => R_EARTH_GM * WD_R_EARTH * (m / M_SUN_ME).powf(WD_R_EXP),
        BodyKind::NeutronStar => NS_R_GM,
        BodyKind::BlackHole => m * SCHWARZSCHILD_GM_PER_ME,
        BodyKind::Planet | BodyKind::DwarfPlanet => planet_true_r(m),
    }
}

// ---- Event / flash / stats records -----------------------------------------

/// Short-lived visual marker for a collision or disruption. Rendered as an
/// expanding ring; not physical, not a body. `expires_at` is precomputed as
/// `age + life` for the renderer's fade logic.
pub struct Flash {
    pub x: f64,
    pub y: f64,
    pub r0: f64,
    pub age: f64,
    pub life: f64,
    pub expires_at: f64,
}

/// Classification of a discrete event, used by the UI to colour log lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    Merge,
    Impact,
    Tidal,
    Escape,
    User,
    Evolve,
    /// A comet was born, exhausted its volatiles, or was otherwise destroyed
    /// by sublimation.
    Comet,
    /// An interstellar object entered or left the system. Kept distinct from
    /// `Escape` so the log can colour the (rare, dramatic) events differently
    /// from routine planet ejections.
    Interstellar,
    /// A heavy planet crossed `CULL_DIST` outbound and was promoted to a
    /// rogue. The body is not removed yet — it remains visible until
    /// `HARD_CULL_DIST`. Logged separately from `Escape` so the user can find
    /// the ejection event even after the body is gone.
    Eject,
}

/// A single discrete event, kept in a bounded ring buffer (`EVENT_LOG_MAX`).
/// `x` / `y` are the position *at the moment the event happened*, so a body
/// that moves after a merger still has its merge location highlighted.
pub struct Event {
    pub time: f64,
    pub kind: EventKind,
    pub text: String,
    pub x: f64,
    pub y: f64,
}

/// Cumulative counters and diagnostics, all `u64` because they only go up.
/// `last_nsub` and `max_rung` are last-step snapshots; `energy0` and
/// `energy_offset` track the energy bookkeeping.
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

/// One body waiting to be appended by a spawn routine. Bundled because a
/// debris or collision handler often builds a batch of them and pushes them
/// together; the `class` field is set explicitly rather than derived so
/// fragments inherit their parent's class.
struct Spawn {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    m: f64,
    kind: BodyKind,
    origin: BodyOrigin,
    /// Taxonomic class of the new body. Fresh spawns (`spawn_belt_body`) pick
    /// it via `class_for`; debris fragments inherit the parent's class so an
    /// icy moon sheds ice and a rocky planet sheds rock.
    class: BodyClass,
}

// ---- The simulation container ----------------------------------------------

/// The complete simulation state: SoA body storage, dust fields, event log,
/// RNG and all per-step bookkeeping.
///
/// Every per-body field is a `Vec` of the same length (`self.len()`); this
/// invariant is maintained by `push`, `compact` and any code that writes to
/// slices of them. Bodies marked `dead` remain in the vectors until `compact`
/// rewrites the storage.
pub struct Bodies {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub ax: Vec<f64>,
    pub ay: Vec<f64>,
    /// Previous-step positions for the leapfrog's velocity update.
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
    /// Per-body collision-immunity timer: simulation time left during which
    /// collisions with this body are ignored. Used so a freshly spawned
    /// fragment does not instantly re-collide with its parent.
    pub imm: Vec<f64>,
    pub origin: Vec<BodyOrigin>,
    /// Bulk density in g/cm³ (planets and dwarf planets only).
    pub bulk_density: Vec<f64>,
    /// Taxonomic class. Drives colour in `render.rs` and the bulk density at
    /// spawn time. Meaningless for stars and remnants.
    pub class: Vec<BodyClass>,
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
    pub comet_grains: Vec<u32>,
}
impl Bodies {
    /// Allocate SoA storage for `n` bodies up front. Growing individually
    /// reallocates each of the ~25 parallel vectors once per capacity bump, so
    /// a good estimate matters when spawning large systems.
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
            class: Vec::with_capacity(n),
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
            comet_grains: Vec::with_capacity(n),
        }
    }

    /// Current body count. Includes bodies marked `dead` but not yet compacted.
    #[inline]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Append a body with defaults for every per-body field not passed in
    /// (`name = None`, `origin = Spawned`, `class = Rocky`, `m_init = m`,
    /// white dwarfs / brown dwarfs get `WD_DEFAULT_AGE_GYR` / `BD_DEFAULT_AGE_GYR`).
    /// Returns the new index.
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
        self.class.push(BodyClass::Rocky);
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
        self.comet_grains.push(0);
        i
    }

    /// Linear scan for a body by stable ID. O(n); not on any hot path.
    pub fn index_of_id(&self, id: u64) -> Option<usize> {
        self.id.iter().position(|&v| v == id)
    }

    /// Update a body's mass and recompute its radius from the mass–radius law.
    pub fn set_mass(&mut self, i: usize, m: f64) {
        self.m[i] = m;
        self.r[i] = self.radius_now(i);
    }

    /// Current true radius of body `i`. Stars and brown dwarfs use their
    /// evolutionary state (age and initial mass); planets and dwarf planets
    /// use their bulk density; remnants use the fixed mass–radius relation.
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

    /// Radius at which body `i` counts as colliding. Black holes are widened
    /// to `BH_CAPTURE_HORIZONS` Schwarzschild radii (with a `BH_MIN_CAPTURE` Gm
    /// floor) so a near-miss at the true horizon still triggers capture;
    /// every other kind uses its physical radius.
    #[inline]
    pub fn contact_r(&self, i: usize) -> f64 {
        if self.kind[i] == BodyKind::BlackHole {
            (BH_CAPTURE_HORIZONS * self.r[i]).max(BH_MIN_CAPTURE)
        } else {
            self.r[i]
        }
    }

    /// Record a discrete event. Drops the oldest entry when the log fills.
    pub fn log(&mut self, kind: EventKind, text: String, x: f64, y: f64) {
        self.events.push_back(Event {
            time: self.time,
            kind,
            text,
            x,
            y,
        });
        if self.events.len() > EVENT_LOG_MAX {
            self.events.pop_front();
        }
    }

    /// Rewrite the SoA storage without dead bodies.
    ///
    /// Every dead entry shifts the surviving bodies after it down by one, so
    /// this is a full O(n · fields) copy. `remap` records where each surviving
    /// body landed, so cached parent indices and dust-field primary indices
    /// can be rewritten to match. When nothing is dead the function returns
    /// immediately (the identity remap would be a no-op, and re-indexing the
    /// parent cache / dust fields is guaranteed to leave them unchanged).
    pub fn compact(&mut self) {
        if !self.any_dead() {
            return;
        }
        let old_n = self.len();
        // old index -> new index (usize::MAX for removed bodies).
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
                self.class[write] = self.class[read];
                self.rung[write] = self.rung[read];
                self.dtl[write] = self.dtl[read];
                self.spin[write] = self.spin[read];
                self.disrupting[write] = self.disrupting[read];
                self.coalesce_counter[write] = self.coalesce_counter[read];
                self.parent_cache[write] = self.parent_cache[read];
                self.age_gyr[write] = self.age_gyr[read];
                self.m_init[write] = self.m_init[read];
                self.rlof[write] = self.rlof[read];
                self.comet_grains[write] = self.comet_grains[read];
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
        self.class.truncate(write);
        self.rung.truncate(write);
        self.dtl.truncate(write);
        self.spin.truncate(write);
        self.disrupting.truncate(write);
        self.coalesce_counter.truncate(write);
        self.parent_cache.truncate(write);
        self.age_gyr.truncate(write);
        self.m_init.truncate(write);
        self.rlof.truncate(write);
        self.comet_grains.truncate(write);
        // Dust fields follow their primary; a field whose primary died is dropped.
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
        // Parent indices must also follow their bodies.
        for p in self.parent_cache.iter_mut() {
            *p = p
                .and_then(|k| remap.get(k).copied())
                .filter(|&v| v != usize::MAX);
        }
    }

    fn any_dead(&self) -> bool {
        self.dead.iter().any(|&d| d)
    }

    /// Recompute `parent_cache` for every body.
    ///
    /// O(n²) overall because `find_parent_cached` scans the host table, so
    /// call it every ~100 steps rather than every step. Any operation that
    /// can change parent relationships outside the normal step cadence (user
    /// edit, spawn) should call it directly.
    pub fn refresh_parent_cache(&mut self) {
        let n = self.len();
        let mut cache = std::mem::take(&mut self.parent_cache);
        cache.resize(n, None);
        let ctx = build_parent_context(self);
        {
            let bb: &Bodies = self;
            cache
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, p)| *p = find_parent_cached(bb, i, &ctx));
        }
        self.parent_cache = cache;
        self.parent_cache_step = self.stats.steps;
        self.parent_cache_valid = true;
    }
}

// ---- Free functions --------------------------------------------------------

/// Length of one astronomical unit in simulation units (Gm).
#[inline]
pub fn au_in_gm() -> f64 {
    AU_KM / LEN_KM
}

/// Kopparapu et al. (2013) habitable-zone flux limits `(inner, outer)` in S⊕
/// for a star of effective temperature `teff` (K): runaway greenhouse (inner)
/// and maximum greenhouse (outer). Cool stars shift light to the infrared
/// where it is absorbed better, so their limits are lower than a Sun-like
/// star's. Valid 2600–7200 K; `teff` is clamped to that range.
pub fn hz_flux_limits(teff: f64) -> (f64, f64) {
    let t = teff.clamp(2600.0, 7200.0) - 5780.0;
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t2 * t2;
    let inner = 1.107 + 1.332e-4 * t + 1.580e-8 * t2 - 8.308e-12 * t3 - 1.931e-15 * t4;
    let outer = 0.356 + 6.171e-5 * t + 1.698e-9 * t2 - 3.198e-12 * t3 - 5.575e-16 * t4;
    (inner, outer)
}

/// Habitable zone of a host of `(kind, m)`, in AU, using the main-sequence
/// luminosity and temperature. Both are NaN if the host emits no light.
pub fn habitable_zone_au(kind: BodyKind, m: f64) -> (f64, f64) {
    let l = star_luminosity(kind, m);
    if l <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let (si, so) = hz_flux_limits(star_temperature(kind, m));
    ((l / si).sqrt(), (l / so).sqrt())
}

/// Sum the flux at `(x, y)` from every luminous body except `skip`.
///
/// Units: solar luminosities per (AU)². The distance is clamped at 1e-6 AU
/// (d² ≥ 1e-12 AU²) to keep a body sitting on a star finite; that distance
/// is far inside any real body, so the clamp never truncates a real value.
pub fn stellar_flux(b: &Bodies, x: f64, y: f64, skip: usize) -> f64 {
    let au = au_in_gm();
    let au2 = au * au;
    let n = b.len();
    let mut s = 0.0;
    let lo_end = skip.min(n);
    let hi_start = (skip + 1).min(n);

    for k in 0..lo_end {
        let l = luminosity_of(b, k);
        if l <= 0.0 {
            continue;
        }
        let dx = b.x[k] - x;
        let dy = b.y[k] - y;
        let d2 = (dx * dx + dy * dy) / au2;
        s += l / d2.max(1e-12);
    }
    for k in hi_start..n {
        let l = luminosity_of(b, k);
        if l <= 0.0 {
            continue;
        }
        let dx = b.x[k] - x;
        let dy = b.y[k] - y;
        let d2 = (dx * dx + dy * dy) / au2;
        s += l / d2.max(1e-12);
    }
    s
}

/// Equilibrium temperature in K for a body with Bond albedo `a` receiving
/// flux `s` (L☉ / AU²): T = 278.6 K · (s (1 − a))^¼, no greenhouse.
#[inline]
pub fn equilibrium_temp_albedo(s: f64, albedo: f64) -> f64 {
    278.6 * (s * (1.0 - albedo).max(0.0)).powf(0.25)
}

/// Equilibrium temperature for an Earth-like Bond albedo (0.3): 255 K at
/// Earth's insolation. Use `equilibrium_temp_albedo` for other albedos.
#[inline]
pub fn equilibrium_temp(s: f64) -> f64 {
    equilibrium_temp_albedo(s, 0.3)
}

/// Luminosity-weighted effective temperature of the light reaching `(x, y)`
/// (used to choose the habitable-zone limits). 5780 K if nothing shines.
pub fn stellar_flux_teff(b: &Bodies, x: f64, y: f64, skip: usize) -> f64 {
    let au2 = au_in_gm() * au_in_gm();
    let (mut sw, mut st) = (0.0, 0.0);
    for k in 0..b.len() {
        if k == skip {
            continue;
        }
        let l = luminosity_of(b, k);
        if l <= 0.0 {
            continue;
        }
        let dx = b.x[k] - x;
        let dy = b.y[k] - y;
        let f = l / ((dx * dx + dy * dy) / au2).max(1e-12);
        sw += f;
        st += f * temperature_of(b, k);
    }
    if sw > 0.0 { st / sw } else { 5780.0 }
}

/// How strongly black hole `i` is being fed, 0..1 (drives its accretion disk).
/// 1 while a bound donor overflows its Roche lobe onto it, 0.5 for a hole
/// orbited closely by a very luminous (wind-losing) star, otherwise 0 — an
/// isolated hole has no luminous disk.
pub fn accretion_level(b: &Bodies, i: usize) -> f64 {
    if i >= b.len() || b.kind[i] != BodyKind::BlackHole {
        return 0.0;
    }
    let au = au_in_gm();
    let mut level: f64 = 0.0;
    for k in 0..b.len() {
        if k == i || b.dead[k] || b.kind[k] != BodyKind::Star {
            continue;
        }
        let dx = b.x[k] - b.x[i];
        let dy = b.y[k] - b.y[i];
        let d = (dx * dx + dy * dy).sqrt();
        if k < b.rlof.len() && b.rlof[k] && d < 1e4 * b.r[i].max(1e-12) + 50.0 * au {
            return 1.0;
        }
        if d < 100.0 * au && luminosity_of(b, k) > 1e4 {
            level = level.max(0.5);
        }
    }
    level
}

/// Hill radius of body `i` around host `h`, in simulation units.
#[inline]
pub fn hill_radius(b: &Bodies, i: usize, h: usize) -> f64 {
    let dx = b.x[i] - b.x[h];
    let dy = b.y[i] - b.y[h];
    (dx * dx + dy * dy).sqrt() * (b.m[i] / (3.0 * b.m[h])).cbrt()
}

/// Shift every body so the centre of mass sits at the origin with zero total
/// momentum. `_px` / `_py` (the leapfrog's previous-position buffers) are also
/// reset so the next step does not see a spurious velocity kick from the
/// shift.
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
