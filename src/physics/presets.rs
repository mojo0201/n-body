//! Data-driven system presets, belt specs and the Solar System spawn.

use super::*;

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

/// Choose the belt architecture of a system from its star, age,
/// metallicity and planets. `planets` holds `(index, mass_earth, a_au)`.
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

/// A planetary system read from JSON (see `assets/presets/solar_system.json`).
pub(super) fn default_dust_boost() -> f64 {
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

pub(super) fn default_phase() -> f64 {
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

pub(super) fn default_e_sigma() -> f64 {
    0.10
}

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

pub(super) fn place_companion(b: &mut Bodies, c: &CompanionPreset, default_age: f64) {
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
    /// Body origin: `"comet"` makes it a comet. Absent → `Spawned`.
    #[serde(default)]
    pub origin: Option<String>,
    /// Mean anomaly at spawn in [0, 1). When absent (the common case),
    /// a random phase is drawn as before. Set to `0.0` to start at
    /// perihelion, `0.5` at aphelion.
    #[serde(default)]
    pub phase: Option<f64>,
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
    serde_json::from_str(include_str!("../../assets/presets/solar_system.json"))
        .expect("embedded solar_system.json is malformed")
}

/// The preset library: (slug, parsed preset) pairs, in display order.
/// Cached after the first call.
pub fn load_presets() -> &'static [(String, SystemPreset)] {
    static LIB: OnceLock<Vec<(String, SystemPreset)>> = OnceLock::new();
    LIB.get_or_init(|| {
        let sources: &[(&str, &str)] = &[
            (
                "2mass_j1207",
                include_str!("../../assets/presets/2mass_j1207.json"),
            ),
            (
                "hot_jupiter",
                include_str!("../../assets/presets/51_pegasi_b.json"),
            ),
            (
                "55_cancri",
                include_str!("../../assets/presets/55_cancri.json"),
            ),
            ("algol", include_str!("../../assets/presets/algol.json")),
            (
                "alpha_centauri",
                include_str!("../../assets/presets/alpha_centauri.json"),
            ),
            (
                "au_microscopii",
                include_str!("../../assets/presets/au_microscopii.json"),
            ),
            (
                "beta_pictoris",
                include_str!("../../assets/presets/beta_pictoris.json"),
            ),
            (
                "cygnus_x1",
                include_str!("../../assets/presets/cygnus_x1.json"),
            ),
            (
                "epsilon_eridani",
                include_str!("../../assets/presets/epsilon_eridani.json"),
            ),
            (
                "fomalhaut",
                include_str!("../../assets/presets/fomalhaut.json"),
            ),
            (
                "gaia_bh1",
                include_str!("../../assets/presets/gaia_bh1.json"),
            ),
            (
                "gaia_bh3",
                include_str!("../../assets/presets/gaia_bh3.json"),
            ),
            (
                "gamma_cephei",
                include_str!("../../assets/presets/gamma_cephei.json"),
            ),
            (
                "gliese_876",
                include_str!("../../assets/presets/gliese_876.json"),
            ),
            (
                "hd_40307",
                include_str!("../../assets/presets/hd_40307.json"),
            ),
            (
                "hd_80606",
                include_str!("../../assets/presets/hd_80606.json"),
            ),
            ("hl_tau", include_str!("../../assets/presets/hl_tau.json")),
            ("hr_8799", include_str!("../../assets/presets/hr_8799.json")),
            ("kelt_9", include_str!("../../assets/presets/kelt_9.json")),
            (
                "kepler_16",
                include_str!("../../assets/presets/kepler_16.json"),
            ),
            (
                "kepler_1625",
                include_str!("../../assets/presets/kepler_1625.json"),
            ),
            (
                "kepler_36",
                include_str!("../../assets/presets/kepler_36.json"),
            ),
            (
                "kepler_47",
                include_str!("../../assets/presets/kepler_47.json"),
            ),
            (
                "kepler_51",
                include_str!("../../assets/presets/kepler_51.json"),
            ),
            (
                "kepler_90",
                include_str!("../../assets/presets/kepler_90.json"),
            ),
            (
                "luhman_16",
                include_str!("../../assets/presets/luhman_16.json"),
            ),
            (
                "nice_model",
                include_str!("../../assets/presets/nice_model.json"),
            ),
            ("pds_70", include_str!("../../assets/presets/pds_70.json")),
            (
                "psr_b1257_plus_12",
                include_str!("../../assets/presets/psr_b1257_plus_12.json"),
            ),
            (
                "psr_b1620_26",
                include_str!("../../assets/presets/psr_b1620_26.json"),
            ),
            (
                "sgr_a_star",
                include_str!("../../assets/presets/sgr_a_star.json"),
            ),
            ("sirius", include_str!("../../assets/presets/sirius.json")),
            (
                "solar_system",
                include_str!("../../assets/presets/solar_system.json"),
            ),
            (
                "trappist_1",
                include_str!("../../assets/presets/trappist_1.json"),
            ),
            ("vega", include_str!("../../assets/presets/vega.json")),
            (
                "wasp_12b",
                include_str!("../../assets/presets/wasp_12b.json"),
            ),
            (
                "wd_1145_plus_017",
                include_str!("../../assets/presets/wd_1145_plus_017.json"),
            ),
            (
                "wd_1856_534",
                include_str!("../../assets/presets/wd_1856_534.json"),
            ),
        ];

        sources
            .iter()
            .map(|(name, json)| {
                let preset = serde_json::from_str(json)
                    .unwrap_or_else(|e| panic!("failed to parse preset {name}: {e}"));
                ((*name).to_string(), preset)
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
pub(super) const PRESET_BELT_BUDGET: usize = 200;

/// Seed of the planet phases of the single-system Solar System preset.
pub(super) const SOLAR_PHASE_SEED: u32 = 0xC0DE;

/// Spawn a preset system with its star at the origin (centre of mass at rest).
pub fn spawn_from_preset(b: &mut Bodies, preset: &SystemPreset) {
    spawn_from_preset_seeded(b, preset, SOLAR_PHASE_SEED);
}

/// Spawn a preset with its own phase seed (fresh phases each time).
pub fn spawn_preset_with_seed(b: &mut Bodies, preset: &SystemPreset, phase_seed: u32) {
    spawn_from_preset_seeded(b, preset, phase_seed);
}

/// Like `spawn_from_preset`, with the seed that picks each planet's phase.
/// Returns one warning for each preset flag that the masses contradict.
pub(super) fn spawn_from_preset_seeded(
    b: &mut Bodies,
    preset: &SystemPreset,
    phase_seed: u32,
) -> Vec<String> {
    let mut warnings: Vec<String> = Vec::new();
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
        // `phase: Some(..)` pins the mean anomaly without consuming an
        // RNG draw, so a preset can start a comet exactly at perihelion
        // without shifting the stream for any following body.
        let angle = if let Some(ph) = p.phase {
            ph * std::f64::consts::TAU
        } else {
            b.rng.next_f64() * std::f64::consts::TAU
        };
        let idx = if p.a_au > p_type_cutoff {
            push_circumbinary(b, p.a_au, p.e, p.m_earth, angle)
        } else {
            push_orbit(b, star, p.a_au, p.e, p.m_earth, angle)
        };
        b.name[idx] = Some(p.name.clone());
        if p.is_dwarf {
            apply_dwarf_flag(b, idx, &mut warnings);
        }
        let origin = match p.origin.as_deref() {
            Some("comet") => BodyOrigin::Comet,
            Some("rogue") => BodyOrigin::RoguePlanet,
            _ => BodyOrigin::Spawned,
        };
        b.origin[idx] = origin;
        // The body's class comes from where it sits relative to the
        // star's snow line, except comets which are always Icy.
        b.class[idx] = class_for(origin, p.a_au, snow_line);
        let cls = b.class[idx];
        let rho = p
            .density_gcc
            .unwrap_or_else(|| draw_density(&mut b.rng, cls));
        b.bulk_density[idx] = rho;
        b.r[idx] = b.radius_now(idx);

        for moon in &p.moons {
            let mangle = b.rng.next_f64() * std::f64::consts::TAU;
            let mi = push_orbit(b, idx, moon.a_au, moon.e, moon.m_earth, mangle);
            b.name[mi] = Some(moon.name.clone());
            if moon.is_dwarf {
                apply_dwarf_flag(b, mi, &mut warnings);
            }
            // Moons inherit the parent planet's class, so the Moon is
            // Rocky but Ganymede is Icy.
            b.class[mi] = b.class[idx];
            let mcls = b.class[mi];
            b.bulk_density[mi] = moon
                .density_gcc
                .unwrap_or_else(|| draw_density(&mut b.rng, mcls));
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
    warnings
}

/// Copy every body of `src` into `dst`, shifted by `(cx, cy)`. Returns the
/// index range the copies now occupy. Evolution state comes along.
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
pub(super) fn push_orbit(
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
pub(super) fn spawn_solar_systems_impl(b: &mut Bodies, count: usize) -> (f64, Vec<(usize, usize)>) {
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

/// Set the kind of a preset body flagged `is_dwarf` from its mass. A flag
/// that disagrees with the mass is overridden; the warning goes to the
/// event log and to `warnings`.
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
