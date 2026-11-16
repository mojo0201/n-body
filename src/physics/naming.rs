//! Classification and human-readable naming for every body in the simulation.
//!
//! Two naming layers are provided:
//!   * `class_name` — a coarse physical label ("rocky", "red giant", ...).
//!   * `body_class` — same, but promotes a spawned planet that orbits another
//!     spawned planet to "moon".
//!
//! Heavy callers (UI tables, tooltips) hit these functions every frame, so the
//! static-string variants return `Cow<'static, str>` to avoid heap allocation.

use std::borrow::Cow;
use std::fmt::Write;

use super::*;

// ---------------------------------------------------------------------------
// Mass -> class mappings (pure, allocation-free).
// ---------------------------------------------------------------------------

/// Bulk-composition class for a planet, from its mass in Earth masses.
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

/// Main-sequence spectral class (M..B) from mass, expressed in solar masses.
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

/// Spectral class for a brown dwarf from its *mass* (in Jupiter masses).
///
/// Note this is the mass-based approximation; use `bd_class_for_temp` when a
/// surface temperature is already available.
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

/// Promotes a small body in the dwarf-planet mass window to `DwarfPlanet`,
/// otherwise leaves it as `Planet`.
pub fn classify_small_body(m: f64) -> BodyKind {
    if (BELT_DWARF_M_MIN..=DWARF_M_MAX).contains(&m) {
        BodyKind::DwarfPlanet
    } else {
        BodyKind::Planet
    }
}

// ---------------------------------------------------------------------------
// Bodies -> human-readable class labels.
// ---------------------------------------------------------------------------

/// Coarse physical classification of a body, ignoring hierarchy.
///
/// Static labels are returned as borrowed `Cow`s so that the common callers
/// (table rendering, `Display` impls) don't allocate. Only the star and
/// brown-dwarf branches build a string with `format!`.
pub fn class_name(b: &Bodies, i: usize) -> Cow<'static, str> {
    // Hoist the discriminant; the `Spawned` arm needs it again.
    let kind = b.kind[i];
    match kind {
        BodyKind::BlackHole => "black hole".into(),
        BodyKind::NeutronStar => "neutron star".into(),
        BodyKind::WhiteDwarf => "white dwarf".into(),
        BodyKind::BrownDwarf => match bd_class_for_temp(bd_state(b.m[i], b.age_gyr[i]).temperature)
        {
            'M' => "M-type brown dwarf",
            'L' => "L-type brown dwarf",
            'T' => "T-type brown dwarf",
            _ => "Y-type brown dwarf",
        }
        .into(),
        BodyKind::Star => match phase_of(b, i) {
            Some(StellarPhase::RedGiant) => "red giant".into(),
            Some(StellarPhase::RedSupergiant) => "red supergiant".into(),
            _ => match star_class(b.m[i]) {
                'M' => "M-type star",
                'K' => "K-type star",
                'G' => "G-type star",
                'F' => "F-type star",
                'A' => "A-type star",
                _ => "B-type star",
            }
            .into(),
        },
        BodyKind::Planet | BodyKind::DwarfPlanet => match b.origin[i] {
            BodyOrigin::AsteroidBelt => "asteroid".into(),
            BodyOrigin::KuiperCold => "cold classical KBO".into(),
            BodyOrigin::KuiperHot => "hot classical KBO".into(),
            BodyOrigin::KuiperResonant => "resonant KBO".into(),
            BodyOrigin::KuiperScattered => "scattered KBO".into(),
            BodyOrigin::Trojan => "Trojan".into(),
            BodyOrigin::Hilda => "Hilda".into(),
            BodyOrigin::ImpactDebris => "fragment".into(),
            BodyOrigin::TidalDebris => "tidal fragment".into(),
            BodyOrigin::BackgroundDebris => "debris".into(),
            BodyOrigin::Comet => "comet".into(),
            BodyOrigin::Interstellar => "interstellar object".into(),
            BodyOrigin::RoguePlanet => "rogue planet".into(),
            BodyOrigin::Spawned => {
                if kind == BodyKind::Planet {
                    planet_class(b.m[i]).into()
                } else {
                    "dwarf planet".into()
                }
            }
        },
    }
}

/// `class_name`, but upgrades a spawned planet orbiting another spawned
/// planet to the label "moon".
pub fn body_class(b: &Bodies, i: usize) -> Cow<'static, str> {
    body_class_with_parent(b, i, find_parent(b, i))
}

/// Same as `body_class`, but accepts a pre-computed parent index so callers
/// that need the parent for other columns don't call `find_parent` twice.
pub fn body_class_with_parent(b: &Bodies, i: usize, parent: Option<usize>) -> Cow<'static, str> {
    let kind_i = b.kind[i];
    if matches!(kind_i, BodyKind::Planet | BodyKind::DwarfPlanet)
        && b.origin[i] == BodyOrigin::Spawned
    {
        if let Some(p) = parent {
            // Read each field once; `matches!` + the origin check would
            // otherwise re-bounds-check.
            let kind_p = b.kind[p];
            if matches!(kind_p, BodyKind::Planet | BodyKind::DwarfPlanet)
                && b.origin[p] == BodyOrigin::Spawned
            {
                return "moon".into();
            }
        }
    }
    class_name(b, i)
}

// ---------------------------------------------------------------------------
// Mass / tag formatting.
// ---------------------------------------------------------------------------

/// Appends the mass-with-unit string into `out`.
///
/// Split from `mass_str` so `tag` can build its final string in one buffer
/// instead of allocating a temporary per mass.
fn write_mass(out: &mut String, m: f64) {
    if m >= 1e4 {
        let _ = write!(out, "{:.2} M☉", m / M_SUN_ME);
    } else if m < 10.0 {
        let _ = write!(out, "{:.2} M⊕", m);
    } else {
        let _ = write!(out, "{:.0} M⊕", m);
    }
}

/// Standalone mass formatter. Equivalent to the old implementation, but
/// writes into a pre-sized buffer and avoids an extra allocation.
pub fn mass_str(m: f64) -> String {
    // Worst case "1234.56 M☉" ≈ 12 bytes; 24 is comfortable.
    let mut s = String::with_capacity(24);
    write_mass(&mut s, m);
    s
}

/// One-line display tag for a body: either "Name (mass)" or, for unnamed
/// bodies, "#id class (mass)".
///
/// Builds the result in a single pre-sized `String`; the old version
/// allocated once for `mass_str` and again for the join.
pub fn tag(b: &Bodies, i: usize) -> String {
    let mut s = String::with_capacity(48);
    match &b.name[i] {
        Some(n) => {
            s.push_str(n);
            s.push_str(" (");
            write_mass(&mut s, b.m[i]);
            s.push(')');
        }
        None => {
            let _ = write!(s, "#{} {} (", b.id[i], class_name(b, i));
            write_mass(&mut s, b.m[i]);
            s.push(')');
        }
    }
    s
}

/// Brown-dwarf spectral class from surface temperature (K).
///
/// Thresholds match the standard M/L/T/Y sequence; keep them in sync with
/// `bd_class` (which uses mass thresholds from the same temperature mapping).
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
