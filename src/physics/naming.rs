//! Class names and short tags for planets, stars and brown dwarfs.

use super::*;

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

pub fn classify_small_body(m: f64) -> BodyKind {
    if (BELT_DWARF_M_MIN..=DWARF_M_MAX).contains(&m) {
        BodyKind::DwarfPlanet
    } else {
        BodyKind::Planet
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
            BodyOrigin::Comet => "comet".to_string(),
            BodyOrigin::Interstellar => "interstellar object".to_string(),
            BodyOrigin::RoguePlanet => "rogue planet".to_string(),
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
