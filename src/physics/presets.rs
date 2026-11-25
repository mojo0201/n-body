//! Data-driven system presets, belt specs, and Solar System spawning.
//!
//! Three layers live in this file:
//!
//! 1. **Presets** (`SystemPreset` and friends): serde-deserialized JSON
//!    describing a single system — star, companions, planets, moons, belts.
//! 2. **Belt specs** (`BeltSpec`, `InnerBelt`, `OuterBelt`, `OuterPopulation`):
//!    the *runtime* description of a small-body population. Presets map into
//!    specs, and so does the procedural `belt_spec_for` path used by the
//!    non-preset spawn code.
//! 3. **Spawners**: functions that turn a preset into `Bodies` — either a
//!    single system, or a cluster of Solar System copies arranged on a ring.

use super::*;

/// Coarse classification of a system's small-body architecture. Picked by
/// `belt_spec_for` from the star, its age/metallicity, and the planets'
/// arrangement; lets other code (UI labels, statistics) reason about "what
/// kind of system is this" without re-deriving it from the individual belts.
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

/// Runtime description of every small-body population in a system.
///
/// Two mass budgets are carried separately because they are drawn from
/// different physical populations and are spawned by different code paths:
///
/// * `total_mass` is the *inner* budget — main-belt asteroids, Hildas,
///   Trojans. Lumped together because the spawner treats them as "rocky,
///   short-period small bodies".
/// * `outer_mass` is the *outer* (Kuiper-like) budget, split across the
///   `OuterPopulation`s in `outer`.
///
/// Keeping them separate avoids a whole class of "belt mass went to the
/// wrong reservoir" bugs that appear when a single mass is divided between
/// inner and outer populations.
#[derive(Clone, Debug)]
pub struct BeltSpec {
    pub archetype: BeltArchetype,
    /// Inner/trojan/Hilda mass budget, in Earth masses.
    pub total_mass: f64,
    /// Outer/Kuiper mass budget, in Earth masses. Distinct from `total_mass`;
    /// see the type-level comment.
    pub outer_mass: f64,
    pub inner: Option<InnerBelt>,
    pub outer: Option<OuterBelt>,
    pub trojans: Vec<Trojan>,
    pub gaps: Vec<Gap>,
}

/// A single asteroid-belt-like annulus: a radial span, a share of the
/// system's inner-belt mass, and a broken-power-law size distribution.
#[derive(Clone, Debug)]
pub struct InnerBelt {
    pub a_lo_au: f64,
    pub a_hi_au: f64,
    /// Fraction of the parent belt's `total_mass` that this belt receives.
    pub mass_frac: f64,
    /// Rayleigh scale of orbital eccentricities. 0.10 is the classic
    /// main-belt value; higher values produce a more dynamically hot belt.
    pub e_sigma: f64,
    /// Log-log slope of the cumulative mass function.
    pub plaw: f64,
    pub m_min: f64,
    pub m_max: f64,
}

#[derive(Clone, Debug)]
pub struct OuterBelt {
    pub populations: Vec<OuterPopulation>,
}

/// One dynamical class of outer-belt body (cold classical, hot classical,
/// resonant, scattered). Populations overlap radially; they are separated
/// by their eccentricity distribution and, for scattered bodies, by a
/// pericentre prescription.
#[derive(Clone, Debug)]
pub struct OuterPopulation {
    pub kind: BodyOrigin,
    pub a_lo_au: f64,
    pub a_hi_au: f64,
    /// Fraction of the parent outer-belt mass assigned to this population.
    /// The `mass_frac` values across an `OuterBelt` sum to ~0.9 by design —
    /// the remainder is the "grains/dust" tail (see `spawn_belts`).
    pub mass_frac: f64,
    pub e_sigma: f64,
    /// When `Some(a_giant)`, eccentricities are drawn so the pericentre
    /// lands near `a_giant`. Used to model scattered objects, whose orbits
    /// were raised by close encounters with that giant.
    pub q_frac: Option<f64>,
    pub plaw: f64,
    pub m_min: f64,
    pub m_max: f64,
}

/// A Trojan cloud at a giant planet's L4/L5 point.
#[derive(Clone, Debug)]
pub struct Trojan {
    /// Semi-major axis of the parent giant, in AU.
    pub giant_a_au: f64,
    /// Fraction of Trojans placed at the leading (L4) point. The rest go to
    /// L5. Real clouds are slightly asymmetric; 0.5 is the neutral choice.
    pub leading_frac: f64,
    pub mass_frac: f64,
}

/// A Kirkwood-style gap in the belt, e.g. the 3:1 resonance with Jupiter.
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

    /// True when no belt would actually spawn anything. The thresholds sit
    /// above the minimum particle mass so that rounding noise can't cause
    /// a "belt exists but contains zero bodies" ghost state.
    pub fn is_empty(&self) -> bool {
        self.total_mass < 1e-6
            && self.outer_mass < 1e-6
            && self.inner.is_none()
            && self.outer.is_none()
    }
}

/// Pick the belt architecture of a procedurally generated system.
///
/// The decision tree below is deliberately shallow: the branches correspond
/// to observational categories (hot-Jupiter systems, compact chains, young
/// debris disks, Solar analogues, quiet old systems), not to any detailed
/// dynamical simulation. Each branch also scales the drawn belt masses by a
/// category-specific factor, so a hot-Jupiter system is not merely "a Solar
/// System with the giant moved inward" but a genuinely belt-poor system.
///
/// `planets` is `(index, mass_earth, a_au)`. Only the mass and orbital
/// radius are used here; the index is carried for symmetry with other
/// planetary summarisers and costs nothing.
#[allow(clippy::too_many_arguments)]
pub(super) fn belt_spec_for(
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

    // First gate: does this star even have a detectable debris belt?
    // Drawn before any geometry to keep the RNG stream independent of the
    // planet-arrangement statistics below.
    let p_belt = debris_belt_occurrence_fraction(m_sun, age_gyr);
    if rng.next_f64() > p_belt {
        return BeltSpec::empty();
    }

    let arch = PlanetSummary::of(planets, snow_line_au);
    let window = RadialWindow { lo_au: a_lo_au, hi_au: a_hi_au, snow_line_au };

    // Every archetype starts from a baseline "Solar-like" belt mass. The
    // branches scale that up or down to reflect how much the architecture
    // depletes small bodies.
    let mut spec = BeltSpec {
        archetype: BeltArchetype::NoBelt,
        total_mass: belt_mass_earth(m_sun, feh, age_gyr, rng),
        outer_mass: 0.0,
        inner: None,
        outer: None,
        trojans: Vec::new(),
        gaps: Vec::new(),
    };

    if arch.has_hot_jupiter {
        hot_jupiter_belt(&mut spec, &arch, &window, (m_sun, feh, age_gyr), rng);
    } else if arch.n_compact >= 4 && arch.giants.is_empty() {
        compact_chain_belt(&mut spec, &window, (m_sun, feh, age_gyr), rng);
    } else if arch.giants.is_empty() && m_sun > 1.2 && age_gyr < 0.5 {
        young_debris_disk_belt(&mut spec, &window, (m_sun, feh, age_gyr), rng);
    } else if !arch.giants.is_empty() {
        solar_type_belt(&mut spec, &arch, &window, (m_sun, feh, age_gyr), rng);
    } else {
        sparse_belt(&mut spec, &window, (m_sun, feh, age_gyr), rng);
    }

    spec
}

/// The caller's radial window for belts, plus the star's snow line.
struct RadialWindow {
    lo_au: f64,
    hi_au: f64,
    snow_line_au: f64,
}

/// Summary of a planet list: everything the belt archetypes branch on.
/// Computed in one pass so the decision tree stays linear in planet count.
struct PlanetSummary {
    /// `(mass_earth, a_au)` of every planet above 50 M⊕.
    giants: Vec<(f64, f64)>,
    has_hot_jupiter: bool,
    a_giant_inner: f64,
    a_giant_outer: f64,
    /// Small planets inside 1 AU (the TRAPPIST-1 / Kepler-90 regime).
    n_compact: usize,
    /// Outer edge of the compact chain inside the snow line.
    a_compact: f64,
    /// Outermost cold Neptune-class planet: sets the Kuiper resonances.
    a_ice: f64,
}

impl PlanetSummary {
    /// `planets` is `(index, mass_earth, a_au)`.
    fn of(planets: &[(usize, f64, f64)], snow_line_au: f64) -> Self {
        let mut sum = Self {
            giants: Vec::new(),
            has_hot_jupiter: false,
            a_giant_inner: f64::INFINITY,
            a_giant_outer: 0.0,
            n_compact: 0,
            a_compact: 0.0,
            a_ice: 0.0,
        };
        for &(_, m, a) in planets {
            if (10.0..=50.0).contains(&m) && a >= snow_line_au && a > sum.a_ice {
                sum.a_ice = a;
            }
            if m > 50.0 {
                sum.giants.push((m, a));
                if a < 0.1 {
                    sum.has_hot_jupiter = true;
                }
                sum.a_giant_inner = sum.a_giant_inner.min(a);
                sum.a_giant_outer = sum.a_giant_outer.max(a);
            }
            // Only planets inside the snow line belong to the compact chain.
            // A cold Neptune must not push the asteroid belt outward.
            if m <= 30.0 && a < snow_line_au {
                sum.a_compact = sum.a_compact.max(a);
                if a < 1.0 {
                    sum.n_compact += 1;
                }
            }
        }
        sum
    }
}

/// `(star mass in M☉, [Fe/H], age in Gyr)`: what the belt-mass laws take.
type StarProps = (f64, f64, f64);

/// An outer belt modelled as one cold classical population.
fn single_cold_belt(a_lo: f64, a_hi: f64, e_sigma: f64, m_min: f64, m_max: f64) -> Option<OuterBelt> {
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
}

/// Hot Jupiters: the giant migrated inward, scattering whatever belt was
/// inside its present orbit. Both budgets shrink by 10x and the surviving
/// outer belt is a narrow annulus just beyond the giant.
fn hot_jupiter_belt(
    spec: &mut BeltSpec,
    arch: &PlanetSummary,
    win: &RadialWindow,
    (m_sun, feh, age_gyr): StarProps,
    rng: &mut XorShift,
) {
    spec.archetype = BeltArchetype::HotJupiter;
    let start = (arch.a_giant_inner * 3.0).max(10.0).max(win.lo_au);
    let end = (start * 5.0).min(win.hi_au);
    if end > start {
        spec.outer = single_cold_belt(start, end, 0.05, MASS_KUIPER_MIN, MASS_KUIPER_MAX);
    }
    spec.total_mass *= 0.1;
    spec.outer_mass = belt_mass_earth_outer(m_sun, feh, age_gyr, rng) * 0.1;
}

/// Compact chains: four or more small planets inside 1 AU. Same depletion
/// as hot Jupiters; the belt sits just outside the snow line.
fn compact_chain_belt(
    spec: &mut BeltSpec,
    win: &RadialWindow,
    (m_sun, feh, age_gyr): StarProps,
    rng: &mut XorShift,
) {
    spec.archetype = BeltArchetype::CompactChain;
    let start = (win.snow_line_au * 5.0).max(win.lo_au);
    let end = (start * 3.0).min(win.hi_au);
    if end > start {
        spec.outer = single_cold_belt(start, end, 0.08, MASS_KUIPER_MIN, MASS_KUIPER_MAX);
    }
    spec.total_mass *= 0.1;
    spec.outer_mass = belt_mass_earth_outer(m_sun, feh, age_gyr, rng) * 0.1;
}

/// Young, massive, giant-free systems host the bright debris disks that
/// ALMA / Herschel see (Beta Pic analogues). Two orders of magnitude above
/// baseline, with an unusually low eccentricity: the dust is dynamically cold.
fn young_debris_disk_belt(
    spec: &mut BeltSpec,
    win: &RadialWindow,
    (m_sun, feh, age_gyr): StarProps,
    rng: &mut XorShift,
) {
    spec.archetype = BeltArchetype::YoungDebrisDisk;
    // Disk radius follows stellar luminosity: 8 to 16 snow lines.
    let (start, end) = (
        win.lo_au.max(win.snow_line_au * 8.0),
        win.hi_au.min(win.snow_line_au * 16.0),
    );
    if end > start {
        spec.outer = single_cold_belt(start, end, 0.02, 1e-3, 1e-1);
    }
    spec.total_mass *= 100.0;
    spec.outer_mass = belt_mass_earth_outer(m_sun, feh, age_gyr, rng) * 100.0;
}

/// Sparse: old, quiet, giant-poor. The inner belt was cleared by migration
/// or never formed, but a modest outer belt persists (ALMA, old K dwarfs;
/// Matrà 2018).
fn sparse_belt(
    spec: &mut BeltSpec,
    win: &RadialWindow,
    (m_sun, feh, age_gyr): StarProps,
    rng: &mut XorShift,
) {
    spec.archetype = BeltArchetype::Sparse;
    spec.outer_mass = belt_mass_earth_outer(m_sun, feh, age_gyr, rng) * 0.3;
    let start = (win.snow_line_au * 5.0).max(win.lo_au);
    let end = (start * 5.0).min(win.hi_au);
    if end > start {
        spec.outer = single_cold_belt(start, end, 0.10, MASS_KUIPER_MIN, MASS_KUIPER_MAX);
    }
}

/// Solar analogue: giants present, no hot Jupiter. The canonical
/// four-population Kuiper belt plus a main asteroid belt, Trojans at the
/// most massive giant, and Kirkwood gaps at that giant's resonances.
fn solar_type_belt(
    spec: &mut BeltSpec,
    arch: &PlanetSummary,
    win: &RadialWindow,
    (m_sun, feh, age_gyr): StarProps,
    rng: &mut XorShift,
) {
    spec.archetype = BeltArchetype::SolarType;
    spec.outer_mass = belt_mass_earth_outer(m_sun, feh, age_gyr, rng);
    // Computed before the inner belt so its mass fraction can take the
    // remainder of `total_mass`.
    let trojan_frac = add_trojans_and_gaps(spec, &arch.giants);
    spec.inner = main_asteroid_belt(arch, win, trojan_frac);
    spec.outer = classical_kuiper_belt(arch, win);
}

/// Only the single most massive giant gets a substantial Trojan cloud. Real
/// Jupiter Trojans are ~8% of the main belt; Saturn's are unobservable and
/// Neptune's likely a few per cent of Jupiter's. Below ~30 M⊕ the L4/L5
/// resonance is not deep enough to capture a cloud at all. Also adds the
/// Kirkwood gaps. Returns the Trojan mass fraction (0 when there is none).
fn add_trojans_and_gaps(spec: &mut BeltSpec, giants: &[(f64, f64)]) -> f64 {
    let Some(&(gm, ga)) = giants
        .iter()
        .filter(|(m, a)| *m >= 30.0 && *a > 0.3)
        .max_by(|a, b| a.0.total_cmp(&b.0))
    else {
        return 0.0;
    };
    let m_jup = gm / JUPITER_MASS;
    let frac = (0.08 * m_jup.powf(1.5)).clamp(0.005, 0.10);
    spec.trojans.push(Trojan { giant_a_au: ga, leading_frac: 0.5, mass_frac: frac });
    // `kirkwood_gaps` assumes a Jupiter-mass giant; width ~ mu^(2/7).
    let gap_scale = m_jup.powf(2.0 / 7.0);
    spec.gaps.extend(kirkwood_gaps(ga).into_iter().map(|g| Gap {
        width_au: g.width_au * gap_scale,
        ..g
    }));
    frac
}

/// The main belt sits between the outermost compact planet and the
/// innermost giant, never inside the snow line or outside the caller's
/// window. `total_mass` is the asteroid belt itself; Trojans and Hildas take
/// their share first and the inner belt gets the remainder, so a Jupiter
/// analogue spawns the belt plus an 8% Trojan cloud without over-spending
/// the calibrated total.
fn main_asteroid_belt(arch: &PlanetSummary, win: &RadialWindow, trojan_frac: f64) -> Option<InnerBelt> {
    let lo = (arch.a_compact * 1.2).max(win.snow_line_au * 0.7).max(win.lo_au);
    let hi = (arch.a_giant_inner * 0.6).min(win.hi_au);
    if hi <= lo {
        return None;
    }
    Some(InnerBelt {
        a_lo_au: lo,
        a_hi_au: hi,
        mass_frac: (1.0 - trojan_frac - belts::HILDA_MASS_FRAC).max(0.0),
        e_sigma: 0.10,
        plaw: MASS_PLAW_ASTEROID,
        m_min: MASS_ASTEROID_MIN,
        m_max: MASS_ASTEROID_MAX,
    })
}

/// The Kuiper belt starts just outside the outermost giant but at least 10
/// snow lines out. The four populations are the standard dynamical split;
/// their `mass_frac` sum to 0.9, the missing 0.1 being the dust tail that
/// `spawn_belts` adds to every outer belt.
fn classical_kuiper_belt(arch: &PlanetSummary, win: &RadialWindow) -> Option<OuterBelt> {
    let lo = (arch.a_giant_outer * 1.5)
        .max(win.snow_line_au * 10.0)
        .max(arch.a_ice * 1.3)
        .max(win.lo_au);
    let hi = (win.snow_line_au * 40.0).min(win.hi_au);
    if hi <= lo {
        return None;
    }
    let span = hi - lo;
    // Resonant objects need a perturber: 3:2 and 2:1 resonances of the cold
    // Neptune. Without a fit, they become hot classicals.
    let a_32 = arch.a_ice * 1.5f64.powf(2.0 / 3.0);
    let a_21 = arch.a_ice * 2.0f64.powf(2.0 / 3.0);
    let res_ok = arch.a_ice > 0.0 && a_32 >= lo && a_21 <= hi;
    let (r_kind, lo1, hi1, lo2, hi2) = if res_ok {
        (BodyOrigin::KuiperResonant, a_32 * 0.98, a_32 * 1.02, a_21 * 0.98, a_21 * 1.02)
    } else {
        (BodyOrigin::KuiperHot, lo, hi, lo, hi)
    };
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
    Some(OuterBelt {
        populations: vec![
            // Cold classicals: tight eccentricities, inner 40% of the belt.
            pop(BodyOrigin::KuiperCold, lo, lo + span * 0.4, 0.40, 0.03, None),
            // Hot classicals: wider eccentricities, full span.
            pop(BodyOrigin::KuiperHot, lo, hi, 0.30, 0.15, None),
            // Resonant objects (Plutinos etc.).
            pop(r_kind, lo1, hi1, 0.07, 0.10, None),
            pop(r_kind, lo2, hi2, 0.03, 0.10, None),
            // Scattered: pericentre tied to the outer giant.
            pop(BodyOrigin::KuiperScattered, lo, hi, 0.10, 0.30, Some(arch.a_giant_outer)),
        ],
    })
}

/// Belt around an evolved remnant (white dwarf, neutron star, black hole).
///
/// A low-mass remnant plus a 30 % die roll yields a small scattered belt
/// inside 40 AU — the observed signature of a tidally disrupted planet or
/// asteroid that fell onto the remnant. Everything else gets no belt, which
/// is consistent with the majority of observed evolved systems.
pub(super) fn belt_spec_for_evolved(
    m_remnant_sun: f64,
    _age_gyr: f64,
    rng: &mut XorShift,
) -> BeltSpec {
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

/// Multiplies the initial dust density. Default 1 (see `SystemPreset`).
pub(super) fn default_dust_boost() -> f64 {
    1.0
}

/// A planetary system read from JSON (see `assets/presets/solar_system.json`).
///
/// Presets are *declarative*: they name masses, orbits and flags, not
/// positions or velocities. `spawn_from_preset_seeded` resolves them into
/// concrete state via Kepler solves. The only non-declarative field is
/// `dust_boost`, a display knob that scales the spawned dust cloud.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Absent means "no belts at all", which `spawn_belts` skips.
    #[serde(default)]
    pub belts: Option<BeltPreset>,
}

/// Companion phase is uniform in *time*, so the mean anomaly is uniformly
/// distributed in [0, 1). 0.5 is the apocentre, which tends to look
/// "interesting" in the initial frame — companions visibly separated from
/// their primary rather than superimposed on it.
pub(super) fn default_phase() -> f64 {
    0.5
}

/// A secondary star (or compact object) in a preset.
///
/// Companions are placed in list order. Each `a_au` is the semi-major axis
/// about the centre of mass of *everything already placed* (the primary plus
/// earlier companions), so list inner companions first: Algol Aa2 (0.06 AU)
/// before Algol Ab (2.7 AU), Alpha Cen B before Proxima.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Argument of periapsis of the companion's orbit, degrees. Default 0.
    #[serde(default)]
    pub omega_deg: f64,
    /// Inclination to the system mid-plane (z = 0), degrees. Default 0.
    #[serde(default)]
    pub inc_deg: f64,
    /// Longitude of the ascending node, degrees. Default 0.
    #[serde(default)]
    pub node_deg: f64,
}

/// Belt definition inside a preset. The `mass_frac` fields inside `inner`
/// and each outer population are *relative* to `total_mass_earth`; the
/// spawner normalises them.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BeltPreset {
    /// Star the belt surrounds, by name, or "barycentre" for a circumbinary
    /// belt. Absent = the primary (the old behaviour).
    #[serde(default)]
    pub host: Option<String>,
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

/// Preset-side asteroid-belt description. Only the radial span and mass
/// share are user-controlled; the eccentricity scale, slope and size bounds
/// are filled in by `BeltPreset::to_spec` with main-belt-typical values so
/// that hand-edited JSON stays short.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InnerBeltPreset {
    pub a_lo_au: f64,
    pub a_hi_au: f64,
    pub mass_frac: f64,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuterBeltPreset {
    pub populations: Vec<PopulationPreset>,
}

/// Preset-side description of one outer-belt population. `kind` is matched
/// against a fixed string table in `BeltPreset::to_spec`; unknown kinds
/// silently fall back to `cold_classical` so a typo doesn't abort the whole
/// preset library load.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PopulationPreset {
    pub kind: String,
    pub a_lo_au: f64,
    pub a_hi_au: f64,
    pub mass_frac: f64,
    #[serde(default = "default_e_sigma")]
    pub e_sigma: f64,
    /// Pericentre in AU for scattered objects. When set, eccentricity is
    /// drawn so pericentre lands near this distance.
    #[serde(default)]
    pub q_frac: Option<f64>,
}

/// Default eccentricity scale for populations that don't specify one.
/// Matches the main-belt value used by `InnerBelt`'s Solar-analogue branch.
pub(super) fn default_e_sigma() -> f64 {
    0.10
}

/// Resolve a preset's `kind` string (or its absence) to a `BodyKind`.
///
/// The mass fallback is the important bit: a preset that says nothing about
/// kind still needs to be correctly classified. Anything at or above the
/// hydrogen-burning limit (`STAR_M_MIN`) is a star; below it, a brown dwarf.
/// Explicit compact-object kinds take precedence over the mass guess, so a
/// 1.2 M_sun white dwarf stays a white dwarf rather than being "corrected"
/// into a star.
pub(super) fn parse_body_kind(s: Option<&str>, mass: f64) -> BodyKind {
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

/// Insert a companion into `b` such that the existing bodies and the new
/// companion orbit their common centre of mass, with the COM at rest.
///
/// The existing bodies are rescaled by their COM-frame position: the
/// primary shifts by `-f_in * r` and the companion lands at `+f_out * r`,
/// where `f_in = m_companion / (M + m_companion)` and `f_out = M / (M +
/// m_companion)`. This reproduces the two-body solution exactly and keeps
/// any pre-existing planets glued to the primary's new position, since
/// their offsets relative to the primary are unchanged.
pub(super) fn place_companion(b: &mut Bodies, c: &CompanionPreset, default_age: f64) -> usize {
    let n = b.len();
    let (mut mt, mut cx, mut cy, mut cz) = (0.0, 0.0, 0.0, 0.0);
    let (mut vx, mut vy, mut vz) = (0.0, 0.0, 0.0);
    for i in 0..n {
        mt += b.m[i];
        cx += b.m[i] * b.x[i];
        cy += b.m[i] * b.y[i];
        cz += b.m[i] * b.z[i];
        vx += b.m[i] * b.vx[i];
        vy += b.m[i] * b.vy[i];
        vz += b.m[i] * b.vz[i];
    }
    (cx, cy, cz) = (cx / mt, cy / mt, cz / mt);
    (vx, vy, vz) = (vx / mt, vy / mt, vz / mt);
    let mt_total = mt + c.mass_earth;
    let a_gm = c.a_au * au_in_gm();
    let m_anom = c.phase * std::f64::consts::TAU;
    let nu = true_anomaly(solve_kepler(m_anom, c.e), c.e);
    let (inc, node) = (c.inc_deg.to_radians(), c.node_deg.to_radians());
    let omega = if node == 0.0 {
        c.omega_deg.to_radians()
    } else {
        (c.omega_deg - c.node_deg).to_radians()
    };
    let (rx, ry, rz, rvx, rvy, rvz) =
        kepler_state_incl(G * mt_total, a_gm, c.e, nu, omega, inc, node);
    let f_in = c.mass_earth / mt_total;
    let f_out = mt / mt_total;
    for i in 0..n {
        b.x[i] -= f_in * rx;
        b.y[i] -= f_in * ry;
        b.z[i] -= f_in * rz;
        b.vx[i] -= f_in * rvx;
        b.vy[i] -= f_in * rvy;
        b.vz[i] -= f_in * rvz;
        b._px[i] = b.x[i];
        b._py[i] = b.y[i];
        b._pz[i] = b.z[i];
    }
    let kind = parse_body_kind(c.kind.as_deref(), c.mass_earth);
    let idx = push3(
        b,
        [cx + f_out * rx, cy + f_out * ry, cz + f_out * rz],
        [vx + f_out * rvx, vy + f_out * rvy, vz + f_out * rvz],
        c.mass_earth,
        kind,
    );
    b.name[idx] = Some(c.name.clone());
    b.age_gyr[idx] = c.age_gyr.unwrap_or(default_age);
    b.set_mass(idx, c.mass_earth);
    idx
}

/// Holman-Wiegert S-type limit (AU) for a planet around a preset star.
/// `host` is `None` for the primary, `Some(k)` for `companions[k]`.
fn s_type_limit_au(preset: &SystemPreset, host: Option<usize>) -> Option<f64> {
    let m_star = preset.star.mass_earth;
    let (c, m_host, m_other) = match host {
        None => {
            let c = preset.companions.first()?;
            (c, m_star, c.mass_earth)
        }
        Some(k) => {
            let c = preset.companions.get(k)?;
            let inner = m_star
                + preset.companions[..k]
                    .iter()
                    .map(|x| x.mass_earth)
                    .sum::<f64>();
            (c, c.mass_earth, inner)
        }
    };
    Some(hw_critical_s(c.a_au, c.e, m_other / (m_host + m_other)))
}

/// Snow line (AU) for a host. Remnants use a fixed 3 AU, like
/// `spawn_evolved_system`, since their geometric snow line is far too small.
fn snow_line_for(b: &Bodies, host: usize) -> f64 {
    match b.kind[host] {
        BodyKind::WhiteDwarf | BodyKind::NeutronStar | BodyKind::BlackHole => 3.0,
        _ => 2.7 * luminosity_of(b, host).max(1e-6).sqrt(),
    }
}

/// Holman-Wiegert P-type limit (AU) about the first companion's binary.
fn p_type_limit_au(preset: &SystemPreset) -> Option<f64> {
    let c = preset.companions.first()?;
    let mu = c.mass_earth / (preset.star.mass_earth + c.mass_earth);
    Some(hw_critical_p(c.a_au, c.e, mu))
}

/// Innermost and outermost radius a belt preset spans, in AU.
fn belt_extent_au(bp: &BeltPreset) -> Option<(f64, f64)> {
    let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
    if let Some(i) = &bp.inner {
        lo = lo.min(i.a_lo_au);
        hi = hi.max(i.a_hi_au);
    }
    if let Some(o) = &bp.outer {
        for p in &o.populations {
            lo = lo.min(p.a_lo_au);
            hi = hi.max(p.a_hi_au);
        }
    }
    (hi > 0.0).then_some((lo, hi))
}

/// Mass-weighted centre (x, y, vx, vy, total mass) of the given stars.
fn host_barycentre(b: &Bodies, stars: &[usize]) -> (f64, f64, f64, f64, f64) {
    let (mut mt, mut cx, mut cy, mut vx, mut vy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &i in stars {
        mt += b.m[i];
        cx += b.m[i] * b.x[i];
        cy += b.m[i] * b.y[i];
        vx += b.m[i] * b.vx[i];
        vy += b.m[i] * b.vy[i];
    }
    (cx / mt, cy / mt, vx / mt, vy / mt, mt)
}

/// Push a planet that orbits the *barycentre* of the stellar hosts rather
/// than any single star. Used for P-type (circumbinary) planets, whose
/// `a_au` is well outside the binary separation so the binary can be
/// treated as a point mass.
///
/// Unlike `place_companion`, the existing bodies are not shifted: this is
/// called when adding a *planet*, and the hosts' mutual orbit is already
/// established. The two-body state is computed about the host COM in its
/// current (possibly moving) frame.
pub(super) fn push_circumbinary(
    b: &mut Bodies,
    a_au: f64,
    e: f64,
    m_me: f64,
    mean_anomaly: f64,
) -> usize {
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
#[serde(deny_unknown_fields)]
pub struct TrojanPreset {
    pub giant_a_au: f64,
    pub mass_frac: f64,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GapPreset {
    pub a_au: f64,
    pub width_au: f64,
}

impl BeltPreset {
    /// Resolve a preset-side belt into the runtime `BeltSpec`.
    ///
    /// Preset JSON is deliberately sparse: it exposes radial spans and mass
    /// shares, and this method fills in the physical parameters (eccentricity
    /// scales, slopes, size bounds) with the same defaults the procedural
    /// Solar-analogue branch uses. This keeps hand-written presets short
    /// without losing fidelity for the common case; a preset that needs
    /// unusual physics can still reach the runtime `BeltSpec` directly via
    /// `belt_spec_for`.
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
                    // Unknown kinds fall back to cold classicals rather
                    // than erroring: a typo in one population shouldn't
                    // kill the entire preset library load.
                    let kind = match p.kind.as_str() {
                        "cold_classical" => BodyOrigin::KuiperCold,
                        "hot_classical" => BodyOrigin::KuiperHot,
                        "resonant" => BodyOrigin::KuiperResonant,
                        "scattered" => BodyOrigin::KuiperScattered,
                        other => panic!("unknown belt population kind in preset: {other}"),
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

/// The primary of a preset system. `kind` is optional so simple presets
/// (a plain Sun-like star) can just specify mass, age and metallicity.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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

/// A planet in a preset. Orbits are expressed relative to the primary (or,
/// for P-type planets, the binary barycentre).
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanetPreset {
    pub name: String,
    /// Star this planet orbits, by name. "barycentre" forces a P-type orbit.
    /// Absent = old behaviour (primary, or barycentre past the cutoff).
    #[serde(default)]
    pub host: Option<String>,
    /// Argument of periapsis in degrees, counter-clockwise from +x. Default 0.
    /// When `node_deg` is set this is the *longitude* of periapsis
    /// (Ω + ω), as tabulated for the Solar System; the argument of
    /// periapsis used for the orbit is `omega_deg - node_deg`.
    #[serde(default)]
    pub omega_deg: f64,
    /// Inclination to the system mid-plane (z = 0), degrees. Default 0.
    /// Retrograde orbits (Halley) are above 90.
    #[serde(default)]
    pub inc_deg: f64,
    /// Longitude of the ascending node, degrees. Default 0.
    #[serde(default)]
    pub node_deg: f64,
    pub a_au: f64,
    pub e: f64,
    pub m_earth: f64,
    /// Bulk density in g/cm³. When absent, it is drawn (rocky or icy).
    #[serde(default)]
    pub density_gcc: Option<f64>,
    /// True for IAU dwarf planets (Ceres, Pluto, ...).
    #[serde(default)]
    pub is_dwarf: bool,
    /// Body origin: `"comet"` makes it a comet. Absent → `Spawned`.
    #[serde(default)]
    pub origin: Option<String>,
    /// Mean anomaly at spawn in [0, 1). When absent (the common case),
    /// a random phase is drawn as before. Set to `0.0` to start at
    /// perihelion, `0.5` at aphelion. Setting `Some(..)` consumes no RNG,
    /// so a pinned comet does not shift the phase stream for later bodies.
    #[serde(default)]
    pub phase: Option<f64>,
    /// Moons of this planet. Each moon's `a_au` is measured from the
    /// planet, not the star.
    #[serde(default)]
    pub moons: Vec<MoonPreset>,
}

/// A moon of a preset planet. The eccentricity defaults to 0 because most
/// Solar System moons are near-circular and a nonzero default would
/// surprise anyone editing the JSON by hand.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Inclination to the *planet's orbital plane*, degrees. Default 0.
    #[serde(default)]
    pub inc_deg: f64,
    /// Node of that tilt, degrees. Default 0.
    #[serde(default)]
    pub node_deg: f64,
}

/// The Solar System, embedded from `assets/presets/solar_system.json` at
/// build time.
///
/// The parse result is cached on first use because every call site — the
/// single-system spawner, the cluster spawner, the UI reload path — hits
/// this function, and `serde_json::from_str` is a few milliseconds per
/// invocation on the embedded blob. The returned value is an owned clone,
/// matching the original owned-return contract so call sites are unchanged.
pub fn solar_system_preset() -> SystemPreset {
    static CACHE: OnceLock<SystemPreset> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            serde_json::from_str(include_str!("../../assets/presets/solar_system.json"))
                .expect("embedded solar_system.json is malformed")
        })
        .clone()
}

/// The preset library: (slug, parsed preset) pairs, in display order.
///
/// Cached after the first call. Each entry is a distinct physical system:
/// exoplanet archetypes, compact objects, binary stars, debris disks, and
/// the Solar System itself. Parsing all ~40 embedded JSONs costs milliseconds,
/// so paying it once at startup is strictly preferable to paying it per
/// UI keystroke.
pub fn load_presets() -> &'static [(String, SystemPreset)] {
    static LIB: OnceLock<Vec<(String, SystemPreset)>> = OnceLock::new();
    LIB.get_or_init(|| PRESET_SOURCES.iter().map(|(name, json)| parse_preset(name, json)).collect())
}

/// Every preset is guaranteed to parse: the JSON is embedded at build time, so
/// a parse error here is a build regression, not a runtime condition. The
/// panic message names the offending slug.
fn parse_preset(name: &str, json: &str) -> (String, SystemPreset) {
    let preset: SystemPreset = serde_json::from_str(json)
        .unwrap_or_else(|e| panic!("failed to parse preset {name}: {e}"));
    if let Some(bp) = &preset.belts {
        let _ = bp.to_spec(); // panics on an unknown belt kind
    }
    (name.to_string(), preset)
}

/// The embedded JSON of `assets/presets/<file>.json`.
macro_rules! embed {
    ($file:literal) => {
        include_str!(concat!("../../assets/presets/", $file, ".json"))
    };
}

/// (slug, JSON) pairs in display order.
const PRESET_SOURCES: &[(&str, &str)] = &[
    ("2mass_j1207", embed!("2mass_j1207")),
    ("hot_jupiter", embed!("51_pegasi_b")),
    ("55_cancri", embed!("55_cancri")),
    ("algol", embed!("algol")),
    ("alpha_centauri", embed!("alpha_centauri")),
    ("au_microscopii", embed!("au_microscopii")),
    ("beta_pictoris", embed!("beta_pictoris")),
    ("cygnus_x1", embed!("cygnus_x1")),
    ("epsilon_eridani", embed!("epsilon_eridani")),
    ("fomalhaut", embed!("fomalhaut")),
    ("gaia_bh1", embed!("gaia_bh1")),
    ("gaia_bh3", embed!("gaia_bh3")),
    ("gamma_cephei", embed!("gamma_cephei")),
    ("gliese_876", embed!("gliese_876")),
    ("hd_40307", embed!("hd_40307")),
    ("hd_80606", embed!("hd_80606")),
    ("hl_tau", embed!("hl_tau")),
    ("hr_8799", embed!("hr_8799")),
    ("kelt_9", embed!("kelt_9")),
    ("kepler_16", embed!("kepler_16")),
    ("kepler_1625", embed!("kepler_1625")),
    ("kepler_36", embed!("kepler_36")),
    ("kepler_47", embed!("kepler_47")),
    ("kepler_51", embed!("kepler_51")),
    ("kepler_90", embed!("kepler_90")),
    ("luhman_16", embed!("luhman_16")),
    ("pds_70", embed!("pds_70")),
    ("psr_b1257_plus_12", embed!("psr_b1257_plus_12")),
    ("psr_b1620_26", embed!("psr_b1620_26")),
    ("sgr_a_star", embed!("sgr_a_star")),
    ("sirius", embed!("sirius")),
    ("solar_system", embed!("solar_system")),
    ("trappist_1", embed!("trappist_1")),
    ("vega", embed!("vega")),
    ("wasp_12b", embed!("wasp_12b")),
    ("wd_1145_plus_017", embed!("wd_1145_plus_017")),
    ("wd_1856_534", embed!("wd_1856_534")),
];

/// Look up a preset by slug. Returns an owned clone, so the caller can
/// mutate `Bodies` while the returned preset is still alive (the preset
/// library itself is behind an immutable `&'static` and would otherwise
/// create borrow conflicts at spawn sites).
pub fn load_preset_by_slug(slug: &str) -> Option<SystemPreset> {
    load_presets()
        .iter()
        .find(|(s, _)| s == slug)
        .map(|(_, p)| p.clone())
}

/// Belt bodies of a preset system without its own `belt_budget`. The frame rate
/// no longer limits this; the Solar System preset already uses 1000.
pub(super) const PRESET_BELT_BUDGET: usize = 600;

/// Seed of the planet phases of the single-system Solar System preset.
/// Kept distinct from the runtime RNG seed so a "Reload" with the default
/// seed reproduces the canonical Solar System layout.
pub(super) const SOLAR_PHASE_SEED: u128 = 0xC0DE;

/// Spawn a preset system with its star at the origin (centre of mass at rest).
pub fn spawn_from_preset(b: &mut Bodies, preset: &SystemPreset) {
    spawn_from_preset_seeded(b, preset, SOLAR_PHASE_SEED);
}

/// Spawn a preset with its own phase seed (fresh phases each time).
pub fn spawn_preset_with_seed(b: &mut Bodies, preset: &SystemPreset, phase_seed: u128) {
    spawn_from_preset_seeded(b, preset, phase_seed);
}

/// Build star, companions, planets, moons and belts into `b`, **without**
/// the expensive finalisation step (`recenter_com`, `compute_accel`,
/// `rebase_energy`, `refresh_parent_cache`, `dust::init_fields`).
///
/// The split exists so `spawn_solar_systems_impl` can build N temporary
/// systems cheaply: it needs each system's body list to compute its reach
/// and mass, but it would throw away any finalisation done on the temporary
/// `Bodies`. Doing finalisation once on the combined system turns an
/// O(N · n²) spawn into O(N · n + (N·n)²).
///
/// Every caller of this function must subsequently call the finalisation
/// step exactly once on the returned `Bodies`.
///
/// Returns one warning per preset flag that the masses contradict.
pub(super) fn spawn_from_preset_bodies(
    b: &mut Bodies,
    preset: &SystemPreset,
    phase_seed: u128,
) -> Vec<String> {
    let mut warnings: Vec<String> = Vec::new();
    reset_for_preset(b, preset, phase_seed);
    let star = place_primary_star(b, preset);
    let (hosts, star_idxs) = place_companions(b, preset, star, &mut warnings);

    let ctx = PlanetContext {
        star,
        snow_line: snow_line_for(b, star),
        // Planets farther than this from the primary are treated as
        // circumbinary and orbit the host barycentre instead. `HW_MARGIN_P`
        // is the standard P-type stability margin for a binary, so the choice
        // of reference frame matches the physical stability boundary.
        p_type_cutoff: preset
            .companions
            .first()
            .map(|c| c.a_au * HW_MARGIN_P)
            .unwrap_or(f64::INFINITY),
        hosts: &hosts,
    };
    for p in &preset.planets {
        place_preset_planet(b, preset, &ctx, p, &mut warnings);
    }
    if let Some(belt_preset) = &preset.belts {
        spawn_preset_belts(b, preset, belt_preset, &ctx, &star_idxs, &mut warnings);
    }
    warnings
}

/// Where the preset's planets and belts go: the primary, its snow line, the
/// circumbinary cutoff and the named hosts `(name, body index, slot)` (slot
/// is `None` for the primary).
struct PlanetContext<'a> {
    star: usize,
    snow_line: f64,
    p_type_cutoff: f64,
    hosts: &'a [(String, usize, Option<usize>)],
}

/// Start from an empty `Bodies` sized for the whole preset.
fn reset_for_preset(b: &mut Bodies, preset: &SystemPreset, phase_seed: u128) {
    let n_stars = 1 + preset.companions.len();
    let n_moons: usize = preset.planets.iter().map(|p| p.moons.len()).sum();
    // Reserve enough capacity for the belt spawn too. Belt budgets are
    // several hundred bodies per preset system; without this the inner `push` loop
    // triggers a reallocation cascade across every SoA column for every
    // belt body, which dominates spawn time for cluster spawns.
    let belt_capacity = preset
        .belts
        .as_ref()
        .map(|bp| bp.belt_budget.unwrap_or(PRESET_BELT_BUDGET))
        .unwrap_or(0);
    *b = Bodies::with_capacity(n_stars + preset.planets.len() + n_moons + belt_capacity);
    b.rng = XorShift::new(phase_seed);
    b.metallicity = preset.star.metallicity;
}

/// The primary star sits at the origin. Companions and planets are placed
/// relative to it, and the COM is recentred in the finalisation step.
fn place_primary_star(b: &mut Bodies, preset: &SystemPreset) -> usize {
    let kind = parse_body_kind(preset.star.kind.as_deref(), preset.star.mass_earth);
    let star = b.push(0.0, 0.0, 0.0, 0.0, preset.star.mass_earth, kind);
    b.name[star] = preset.star.name.clone();
    b.age_gyr[star] = preset.star.age_gyr;
    b.set_mass(star, preset.star.mass_earth);
    star
}

/// Place the companion stars. Returns the named hosts and every star index.
fn place_companions(
    b: &mut Bodies,
    preset: &SystemPreset,
    star: usize,
    warnings: &mut Vec<String>,
) -> (Vec<(String, usize, Option<usize>)>, Vec<usize>) {
    let mut hosts = Vec::with_capacity(1 + preset.companions.len());
    let mut star_idxs = vec![star];
    if let Some(n) = &preset.star.name {
        hosts.push((n.clone(), star, None));
    }
    for (k, comp) in preset.companions.iter().enumerate() {
        if k > 0 && comp.a_au < preset.companions[k - 1].a_au {
            warnings.push(format!(
                "preset: companion {} (a = {} AU) is listed after a wider one; list inner companions first",
                comp.name, comp.a_au
            ));
        }
        let ci = place_companion(b, comp, preset.star.age_gyr);
        hosts.push((comp.name.clone(), ci, Some(k)));
        star_idxs.push(ci);
    }
    (hosts, star_idxs)
}

/// Which body a planet orbits.
struct ResolvedHost {
    index: usize,
    slot: Option<usize>,
    /// The preset forced a circumbinary orbit around the barycentre.
    forced_p_type: bool,
}

/// Look up the planet's named host, warning about unknown names and about
/// planets beyond the S-type stability limit of their host.
fn resolve_planet_host(
    preset: &SystemPreset,
    ctx: &PlanetContext,
    p: &PlanetPreset,
    warnings: &mut Vec<String>,
) -> ResolvedHost {
    let mut host = ResolvedHost { index: ctx.star, slot: None, forced_p_type: false };
    match p.host.as_deref() {
        None => {}
        Some("barycentre") | Some("barycenter") => host.forced_p_type = true,
        Some(name) => match ctx.hosts.iter().find(|(n, _, _)| n == name) {
            Some(&(_, i, slot)) => {
                host.index = i;
                host.slot = slot;
            }
            None => warnings.push(format!(
                "preset: planet {} names unknown host '{name}'; using the primary",
                p.name
            )),
        },
    }
    if p.host.is_some() && !host.forced_p_type {
        if let Some(limit) = s_type_limit_au(preset, host.slot) {
            if p.a_au > limit {
                warnings.push(format!(
                    "preset: {} (a = {:.3} AU) is beyond the S-type stability limit of {:.3} AU for its host",
                    p.name, p.a_au, limit
                ));
            }
        }
    }
    host
}

/// Place one preset planet with its moons. The RNG draw order is part of the
/// reproducibility contract: phase, density, then per moon phase and density.
fn place_preset_planet(
    b: &mut Bodies,
    preset: &SystemPreset,
    ctx: &PlanetContext,
    p: &PlanetPreset,
    warnings: &mut Vec<String>,
) {
    // `phase: Some(..)` pins the mean anomaly without consuming an RNG draw.
    // This matters when a preset wants e.g. a comet exactly at perihelion
    // without shifting the phase stream for any body after it (which would
    // perturb reproducibility of other bodies).
    let angle = match p.phase {
        Some(ph) => ph * std::f64::consts::TAU,
        None => b.rng.next_f64() * std::f64::consts::TAU,
    };
    let host = resolve_planet_host(preset, ctx, p, warnings);
    let circumbinary = host.forced_p_type || (p.host.is_none() && p.a_au > ctx.p_type_cutoff);
    let idx = if circumbinary {
        push_circumbinary(b, p.a_au, p.e, p.m_earth, angle)
    } else {
        // `omega_deg` is a longitude of periapsis when a node is given.
        let omega = if p.node_deg == 0.0 {
            p.omega_deg.to_radians()
        } else {
            (p.omega_deg - p.node_deg).to_radians()
        };
        push_orbit_incl(
            b,
            host.index,
            p.a_au,
            p.e,
            p.m_earth,
            angle,
            omega,
            p.inc_deg.to_radians(),
            p.node_deg.to_radians(),
        )
    };
    let planet_snow_line = if circumbinary || host.index == ctx.star {
        ctx.snow_line
    } else {
        snow_line_for(b, host.index)
    };
    b.name[idx] = Some(p.name.clone());
    if p.is_dwarf {
        apply_dwarf_flag(b, idx, warnings);
    }
    let origin = match p.origin.as_deref() {
        Some("comet") => BodyOrigin::Comet,
        Some("rogue") => BodyOrigin::RoguePlanet,
        _ => BodyOrigin::Spawned,
    };
    b.origin[idx] = origin;
    b.class[idx] = planet_class(p.m_earth, origin, p.a_au, planet_snow_line);
    let cls = b.class[idx];
    let rho = p.density_gcc.unwrap_or_else(|| draw_density(&mut b.rng, cls));
    b.bulk_density[idx] = rho;
    b.r[idx] = b.radius_now(idx);

    place_preset_moons(b, p, idx, host.index, circumbinary, warnings);
}

/// Class comes from the body's position relative to the snow line, except
/// comets which are always Icy regardless of where they sit, and gas giants,
/// which are Icy wherever they sit (same rule as `spawn_planet`).
fn planet_class(m_earth: f64, origin: BodyOrigin, a_au: f64, snow_line_au: f64) -> BodyClass {
    if m_earth >= GIANT_DENSITY_M_MIN && matches!(origin, BodyOrigin::Spawned) {
        BodyClass::Icy
    } else {
        class_for(origin, a_au, snow_line_au)
    }
}

/// Place a planet's moons in its orbital plane (plus any preset tilt).
fn place_preset_moons(
    b: &mut Bodies,
    p: &PlanetPreset,
    planet: usize,
    host_idx: usize,
    circumbinary: bool,
    warnings: &mut Vec<String>,
) {
    let mut group = vec![planet];
    for moon in &p.moons {
        let mangle = b.rng.next_f64() * std::f64::consts::TAU;
        let normal = if circumbinary {
            [0.0, 0.0, 1.0]
        } else {
            let host_state = [
                b.x[host_idx],
                b.y[host_idx],
                b.z[host_idx],
                b.vx[host_idx],
                b.vy[host_idx],
                b.vz[host_idx],
            ];
            orbit_normal(b, planet, host_state)
        };
        let tilt = (moon.inc_deg.to_radians(), moon.node_deg.to_radians());
        let mi = push_orbit_group(b, &group, moon.a_au, moon.e, moon.m_earth, mangle, normal, tilt);
        group.push(mi);
        b.name[mi] = Some(moon.name.clone());
        if moon.is_dwarf {
            apply_dwarf_flag(b, mi, warnings);
        }
        // Moons inherit the parent planet's class, so Earth's Moon is Rocky
        // but Ganymede is Icy. This matches the physical trend where a
        // moon's composition is set by its formation location in the
        // circumplanetary disk, not by heliocentric distance.
        b.class[mi] = b.class[planet];
        let mcls = b.class[mi];
        b.bulk_density[mi] = moon
            .density_gcc
            .unwrap_or_else(|| draw_density(&mut b.rng, mcls));
        b.r[mi] = b.radius_now(mi);
    }
}

/// `(x, y, vx, vy, mass)` of the body a belt is centred on.
type BeltCentre = (f64, f64, f64, f64, f64);

/// Spawn the preset's belts around their host, warning when they sit inside
/// a stability limit or name an unknown host.
fn spawn_preset_belts(
    b: &mut Bodies,
    preset: &SystemPreset,
    belt_preset: &BeltPreset,
    ctx: &PlanetContext,
    star_idxs: &[usize],
    warnings: &mut Vec<String>,
) {
    let spec = belt_preset.to_spec();
    let belt_budget = belt_preset.belt_budget.unwrap_or(PRESET_BELT_BUDGET);
    let (centre, belt_snow_line) = belt_centre(b, preset, belt_preset, ctx, star_idxs, warnings);
    // Swap the RNG out and back so belt draws do not perturb the planet-draw
    // sequence. Reproducibility of planet phases across belt-presence
    // changes is a hard requirement for preset tests.
    let mut rng = std::mem::replace(&mut b.rng, XorShift::new(1));
    spawn_belts(b, centre, &spec, belt_snow_line, belt_budget, &mut rng);
    b.rng = rng;
}

/// Where the belt is centred, and the snow line to use for it.
fn belt_centre(
    b: &Bodies,
    preset: &SystemPreset,
    belt_preset: &BeltPreset,
    ctx: &PlanetContext,
    star_idxs: &[usize],
    warnings: &mut Vec<String>,
) -> (BeltCentre, f64) {
    let star = ctx.star;
    let primary_centre = (b.x[star], b.y[star], b.vx[star], b.vy[star], b.m[star]);
    let extent = belt_extent_au(belt_preset);
    match belt_preset.host.as_deref() {
        None => (primary_centre, ctx.snow_line),
        Some("barycentre") | Some("barycenter") => {
            if let (Some((lo, _)), Some(limit)) = (extent, p_type_limit_au(preset)) {
                if lo < limit {
                    warnings.push(format!(
                        "preset: circumbinary belt starts at {lo:.1} AU, inside the P-type limit of {limit:.1} AU"
                    ));
                }
            }
            (host_barycentre(b, star_idxs), ctx.snow_line)
        }
        Some(name) => match ctx.hosts.iter().find(|(n, _, _)| n == name) {
            Some(&(_, i, slot)) => {
                if let (Some((_, hi)), Some(limit)) = (extent, s_type_limit_au(preset, slot)) {
                    if hi > limit {
                        warnings.push(format!(
                            "preset: belt reaches {hi:.1} AU, beyond the S-type limit of {limit:.1} AU for its host"
                        ));
                    }
                }
                (
                    (b.x[i], b.y[i], b.vx[i], b.vy[i], b.m[i]),
                    2.7 * luminosity_of(b, i).max(1e-6).sqrt(),
                )
            }
            None => {
                warnings.push(format!(
                    "preset: belt names unknown host '{name}'; using the primary"
                ));
                (primary_centre, ctx.snow_line)
            }
        },
    }
}

/// Like `spawn_from_preset`, with the seed that picks each planet's phase.
/// Returns one warning for each preset flag that the masses contradict.
pub(super) fn spawn_from_preset_seeded(
    b: &mut Bodies,
    preset: &SystemPreset,
    phase_seed: u128,
) -> Vec<String> {
    let warnings = spawn_from_preset_bodies(b, preset, phase_seed);
    recenter_com(b);
    compute_accel(b);
    rebase_energy(b);
    b.refresh_parent_cache();
    dust::init_fields(b, None);
    // `dust_boost` defaults to 1.0; the common case is a no-op multiply
    // across every grain, so skip the traversal entirely when the boost is
    // the identity. Bit-for-bit identical to the unconditional version.
    if preset.dust_boost != 1.0 {
        for f in b.dust_fields.iter_mut() {
            for g in f.grains.iter_mut() {
                g.mass *= preset.dust_boost;
            }
        }
    }
    warnings
}

/// Copy every body of `src` into `dst`, shifted by `(cx, cy)`. Returns the
/// index range the copies now occupy. Evolution state (age, mass loss,
/// spin, tidal locks, merger counters) comes along so a cluster member
/// keeps its evolutionary history.
///
/// Velocities are *not* adjusted here: the caller adds its own systemic
/// velocity offset afterwards, because the offset depends on the slot the
/// system lands in rather than on the source system.
pub(super) fn append_system(dst: &mut Bodies, src: &Bodies, cx: f64, cy: f64) -> (usize, usize) {
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
        dst.z[idx] = src.z[i];
        dst._pz[idx] = src.z[i];
        dst.vz[idx] = src.vz[i];
        dst.origin[idx] = src.origin[i];
        dst.bulk_density[idx] = src.bulk_density[i];
        dst.class[idx] = src.class[i];
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
///
/// The relative state is computed in the sun's current frame; the caller's
/// existing motion of `sun` is added on top. This means a planet spawned
/// around a moving star gets the correct inertial velocity without any
/// special case for "moving star" vs. "stationary star".
pub(super) fn push_orbit(
    b: &mut Bodies,
    sun: usize,
    a_au: f64,
    e: f64,
    m_me: f64,
    mean_anomaly: f64,
) -> usize {
    push_orbit_omega(b, sun, a_au, e, m_me, mean_anomaly, 0.0)
}

/// `push_orbit` with an argument of periapsis `omega` in radians,
/// counter-clockwise from +x. `omega = 0.0` reproduces the old behaviour.
pub(super) fn push_orbit_omega(
    b: &mut Bodies,
    sun: usize,
    a_au: f64,
    e: f64,
    m_me: f64,
    mean_anomaly: f64,
    omega: f64,
) -> usize {
    push_orbit_incl(b, sun, a_au, e, m_me, mean_anomaly, omega, 0.0, 0.0)
}

/// `push_orbit_omega` with inclination and ascending node (radians), measured
/// from the mid-plane z = 0. `omega` is the argument of periapsis from the
/// node. With `inc == node == 0` the result is the planar orbit exactly.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_orbit_incl(
    b: &mut Bodies,
    sun: usize,
    a_au: f64,
    e: f64,
    m_me: f64,
    mean_anomaly: f64,
    omega: f64,
    inc: f64,
    node: f64,
) -> usize {
    let a = a_au * au_in_gm();
    // Include the orbiter's mass so a close-in moon of a comparable-mass
    // planet (Charon about Pluto) gets the right period.
    let mu = G * (b.m[sun] + m_me);
    // The angle is a mean anomaly, so the phase is uniform in time.
    let nu = true_anomaly(solve_kepler(mean_anomaly, e), e);
    let (x, y, z, vx, vy, vz) = kepler_state_incl(mu, a, e, nu, omega, inc, node);
    let (sx, sy, sz) = (b.x[sun], b.y[sun], b.z[sun]);
    let (svx, svy, svz) = (b.vx[sun], b.vy[sun], b.vz[sun]);
    push3(
        b,
        [sx + x, sy + y, sz + z],
        [svx + vx, svy + vy, svz + vz],
        m_me,
        BodyKind::Planet,
    )
}

/// Push a moon on an orbit about the centre of mass of `group`
/// (a planet plus its inner moons). The orbit speed uses the total
/// mass of the group, so outer moons see inner moons as part of the host.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_orbit_group(
    b: &mut Bodies,
    group: &[usize],
    a_au: f64,
    e: f64,
    m_me: f64,
    mean_anomaly: f64,
    normal: [f64; 3],
    tilt: (f64, f64),
) -> usize {
    let mt: f64 = group.iter().map(|&k| b.m[k]).sum();
    let (mut cx, mut cy, mut cz) = (0.0, 0.0, 0.0);
    let (mut cvx, mut cvy, mut cvz) = (0.0, 0.0, 0.0);
    for &k in group {
        cx += b.m[k] * b.x[k];
        cy += b.m[k] * b.y[k];
        cz += b.m[k] * b.z[k];
        cvx += b.m[k] * b.vx[k];
        cvy += b.m[k] * b.vy[k];
        cvz += b.m[k] * b.vz[k];
    }
    let (cx, cy, cz) = (cx / mt, cy / mt, cz / mt);
    let (cvx, cvy, cvz) = (cvx / mt, cvy / mt, cvz / mt);
    let a = a_au * au_in_gm();
    let nu = true_anomaly(solve_kepler(mean_anomaly, e), e);
    // Planar orbit about +z, then rotate it into the planet's plane.
    let (x, y, vx, vy) = kepler_state(G * (mt + m_me), a, e, nu, 0.0);
    let n2 = tilted_normal(normal, tilt);
    let (p, v) = if n2 == [0.0, 0.0, 1.0] {
        ([x, y, 0.0], [vx, vy, 0.0])
    } else {
        let (e1, e2) = plane_basis_pub(n2);
        (
            [0, 1, 2].map(|k| x * e1[k] + y * e2[k]),
            [0, 1, 2].map(|k| vx * e1[k] + vy * e2[k]),
        )
    };
    push3(
        b,
        [cx + p[0], cy + p[1], cz + p[2]],
        [cvx + v[0], cvy + v[1], cvz + v[2]],
        m_me,
        BodyKind::Planet,
    )
}

/// Most Solar Systems the UI slider can ask for.
pub const MAX_SOLAR_SYSTEMS: usize = 256;

/// Distance in Gm between neighbouring systems on the ring (about 200 AU).
/// Neptune is at 4500 Gm, so a neighbour's tide on it stays below 1 %.
pub const SOLAR_SYSTEM_SPACING_GM: f64 = 30_000.0;

/// Outer edge of one system (Neptune's apoapsis plus a margin), in Gm.
pub(super) const SOLAR_SYSTEM_RADIUS_GM: f64 = 5_000.0;

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

pub fn spawn_solar_system(b: &mut Bodies) {
    spawn_solar_system_seeded(b, SOLAR_PHASE_SEED);
}

/// Single Solar System with a caller-chosen phase seed, so "Reload (R)"
/// actually gives fresh planet phases (the previous implementation
/// always used `SOLAR_PHASE_SEED`).
pub fn spawn_solar_system_seeded(b: &mut Bodies, seed: u128) {
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
///
/// Performance note: each temporary system is built via
/// `spawn_from_preset_bodies` (no finalisation), because the temporary
/// `Bodies` are only inspected for reach and mass and are then copied into
/// the output. Finalising N temporaries would cost N full dust inits and N
/// O(n²) acceleration passes that are immediately discarded; the combined
/// system is finalised exactly once at the end instead.
pub(super) fn spawn_solar_systems_impl(b: &mut Bodies, count: usize) -> (f64, Vec<(usize, usize)>) {
    let n = count.clamp(1, MAX_SOLAR_SYSTEMS);
    let preset = solar_system_preset();

    // Capacity must cover the full body count (star + planets + moons +
    // belt bodies) per system; otherwise `append_system` reallocates every
    // SoA column once per system, which is a measurable fraction of spawn
    // time for n=256.
    let n_moons: usize = preset.planets.iter().map(|p| p.moons.len()).sum();
    let belt_capacity = preset
        .belts
        .as_ref()
        .map(|bp| bp.belt_budget.unwrap_or(PRESET_BELT_BUDGET))
        .unwrap_or(0);
    let per_system = 1 + preset.planets.len() + n_moons + belt_capacity;
    *b = Bodies::with_capacity(per_system * n + 16);
    b.metallicity = preset.star.metallicity;

    // Build each system as bodies only. `spawn_from_preset_bodies` replaces
    // `*temp` wholesale, so the placeholder capacity is irrelevant.
    let mut temps: Vec<Bodies> = Vec::with_capacity(n);
    for k in 0..n {
        // Each system has its own planet phases. The multiplier is an
        // odd prime so the sequence of phase seeds has no short cycle
        // relative to typical `n` values.
        let seed = SOLAR_PHASE_SEED.wrapping_add((k as u128).wrapping_mul(0x9E37_79B9));
        let mut temp = Bodies::with_capacity(0);
        spawn_from_preset_bodies(&mut temp, &preset, seed);
        temps.push(temp);
    }

    // Cluster layout needs each system's physical reach and total mass, so
    // compute those before deciding slot positions. Both are cheap
    // reductions over the temporary systems.
    let reaches: Vec<f64> = temps.iter().map(system_reach_gm).collect();
    let masses: Vec<f64> = temps.iter().map(|t| t.m.iter().sum::<f64>()).collect();
    let slots = cluster_slots(SOLAR_PHASE_SEED, &reaches, &masses);

    // Copy each system into the output at its slot, then add the slot's
    // systemic velocity. Copy and velocity offset are separate so the
    // copy logic stays position-only (see `append_system`).
    let mut ranges = Vec::with_capacity(n);
    // Several systems: each copy gets its own random orientation.
    if n > 1 {
        for (k, temp) in temps.iter_mut().enumerate() {
            orient_system(temp, 0x50_1A2 ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        }
    }
    for (k, temp) in temps.iter().enumerate() {
        let (x, y, vx, vy) = slots[k];
        let (start, end) = append_system(b, temp, x, y);
        for i in start..end {
            b.vx[i] += vx;
            b.vy[i] += vy;
        }
        ranges.push((start, end));
    }

    // The runtime RNG must not stay at the default seed; mixing in a
    // distinctive constant ensures cluster runs aren't accidentally
    // correlated with the phase seed used to lay out the systems.
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

/// Set the kind of a preset body flagged `is_dwarf` from its mass.
///
/// The mass is authoritative: a preset that flags an object as a dwarf but
/// gives it a planetary mass (or vice versa) gets a corrected kind plus a
/// logged warning. Silently honouring the flag would let bad presets inject
/// physically impossible bodies into the simulation, and silently honouring
/// the mass without logging would make debugging preset edits painful.
fn apply_dwarf_flag(b: &mut Bodies, i: usize, warnings: &mut Vec<String>) {
    let kind = classify_small_body(b.m[i]);
    b.kind[i] = kind;
    if kind != BodyKind::DwarfPlanet {
        let text = format!(
            "preset: {} is flagged dwarf but its mass is outside the dwarf range",
            tag(b, i)
        );
        let (x, y) = (b.x[i], b.y[i]);
        warnings.push(text.clone());
        b.log(EventKind::User, text, x, y);
    }
}
