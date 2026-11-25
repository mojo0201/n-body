//! Surface temperature estimate: equilibrium temperature plus a greenhouse term.
//!
//!     T_surf = T_eq(A) · (1 + ¾ τ)^¼        (grey atmosphere)
//!
//! A rough one-zone model, not a climate model:
//! * Atmosphere: kept when the escape speed is high enough for the temperature
//!   (v_esc² / T_eq, calibrated so Venus and Earth hold one, Mars a thin one,
//!   and Mercury and the Moon none; Titan, being cold, keeps its own).
//! * Optical depth τ: 0.84 for an Earth-like atmosphere (gives 288 K at 1 S⊕),
//!   up to ×2.5 towards the cold edge of the habitable zone (the carbonate–
//!   silicate thermostat), and rising to ~33 beyond the moist-greenhouse limit
//!   (a runaway greenhouse; gives Venus's ~740 K).
//! * Albedo: 0.12 bare rock, 0.30 with an atmosphere, 0.55 when frozen over
//!   (ice–albedo feedback), 0.35 cloud tops for gas and ice giants.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Atmosphere {
    /// Gas or ice giant: no surface, the temperature is that of the cloud tops.
    Giant,
    None,
    Thin,
    Full,
    /// Full atmosphere past the moist-greenhouse limit.
    Runaway,
}

#[derive(Clone, Copy, Debug)]
pub struct SurfaceClimate {
    pub atmosphere: Atmosphere,
    pub albedo: f64,
    pub t_surface: f64,
    pub frozen: bool,
}

const R_EARTH_GM: f64 = 0.006371;
const V_ESC_EARTH_KMS: f64 = 11.19;
const TAU_EARTH: f64 = 0.84;
const TAU_RUNAWAY: f64 = 33.0;

/// Escape speed in km/s.
fn escape_speed_kms(m: f64, r: f64) -> f64 {
    V_ESC_EARTH_KMS * (m / (r / R_EARTH_GM)).sqrt()
}

/// Estimate the surface climate of planet `i` at stellar flux `s` (S⊕), with the
/// habitable-zone flux limits `(hz_in, hz_out)` of its star(s).
pub fn surface_climate(b: &Bodies, i: usize, s: f64, hz_in: f64, hz_out: f64) -> SurfaceClimate {
    let giant = b.m[i] > 10.0 && b.bulk_density[i] < 2.5;
    if giant {
        return SurfaceClimate {
            atmosphere: Atmosphere::Giant,
            albedo: 0.35,
            t_surface: equilibrium_temp_albedo(s, 0.35),
            frozen: false,
        };
    }
    let v = escape_speed_kms(b.m[i], b.r[i]);
    let t_bare = equilibrium_temp_albedo(s, 0.30);
    let hold = v * v / t_bare.max(20.0);
    let mut atm = if hold < 0.07 {
        Atmosphere::None
    } else if hold < 0.2 {
        Atmosphere::Thin
    } else {
        Atmosphere::Full
    };
    let mut tau = match atm {
        Atmosphere::None => 0.0,
        Atmosphere::Thin => 0.05,
        _ => {
            let cold = if hz_out > 0.0 && s < 1.0 { ((1.0 - s) / (1.0 - hz_out).max(0.1)).clamp(0.0, 1.0) } else { 0.0 };
            let mut t = TAU_EARTH * (1.0 + 1.5 * cold);
            if s > hz_in {
                let x = ((s / hz_in - 1.0) / 0.7).clamp(0.0, 1.0);
                t = (TAU_EARTH.ln() + (TAU_RUNAWAY / TAU_EARTH).ln() * x).exp();
                if x >= 1.0 {
                    atm = Atmosphere::Runaway;
                }
            }
            t
        }
    };
    let mut albedo = if atm == Atmosphere::None { 0.12 } else { 0.30 };
    let mut t = equilibrium_temp_albedo(s, albedo) * (1.0 + 0.75 * tau).powf(0.25);
    let mut frozen = false;
    if matches!(atm, Atmosphere::Full) && t < 250.0 && b.class[i] != BodyClass::Icy {
        albedo = 0.55;
        frozen = true;
        tau = tau.min(TAU_EARTH * 2.5);
        t = equilibrium_temp_albedo(s, albedo) * (1.0 + 0.75 * tau).powf(0.25);
    }
    SurfaceClimate { atmosphere: atm, albedo, t_surface: t, frozen }
}
