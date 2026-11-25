//! Split out of the former single-file `ui.rs`; see `mod.rs`.


use egui::Color32;

use crate::physics::{self as ph, Bodies, BodyKind, BodyOrigin};
use crate::theme::theme;
use super::*;

/// Rows for the selection grid. Split out from `selection_window` so that
/// the grid's `|ui| selection_rows(...)` closure stays readable.
///
/// Every section is conditional on the body kind:
///   * "Physical" always shows mass, radius, density, gravity, escape speed
///     (but for a black hole, the last three become the horizon/Schwarzschild
///     quantities, not the surface ones).
///   * "Stellar" only for stellar remnants and stars.
///   * "Climate" only for planets with a non-zero host flux.
///   * "Orbit" only if the body has a parent.
///   * "State" always (position, velocity, collision immunity).
pub(super) fn selection_rows(
    ui: &mut egui::Ui,
    si: bool,
    b: &Bodies,
    i: usize,
    parent: Option<usize>,
    sec: [bool; 6],
) {
    if sec[0] {
        physical_rows(ui, si, b, i);
    }
    match b.kind[i] {
        BodyKind::Star | BodyKind::BrownDwarf | BodyKind::WhiteDwarf | BodyKind::NeutronStar
            if sec[1] =>
        {
            stellar_rows(ui, si, b, i);
        }
        BodyKind::Planet | BodyKind::DwarfPlanet if sec[2] => climate_rows(ui, b, i, parent),
        _ => {}
    }
    if let (true, Some(p)) = (sec[3], parent) {
        orbit_rows(ui, si, b, i, p);
    }
    if sec[4] {
        state_rows(ui, si, b, i, parent);
    }
}

/// "Physical" section: mass, radius, density, gravity, escape speed (a black
/// hole shows its horizon and capture radius instead).
pub(super) fn physical_rows(ui: &mut egui::Ui, si: bool, b: &Bodies, i: usize) {
    let (m, r, kind) = (b.m[i], b.r[i], b.kind[i]);
    let t_unit = ph::time_unit_s();
    heading_row(ui, "Physical");
    kv(ui, "mass", fmt_mass(m, si));
    // "vs Earth" is shown only for sub-brown-dwarf masses; above that,
    // the M_sun or M_jup display is more meaningful. The 0.05 threshold
    // matches `fmt_mass`'s M_sun cutoff.
    if si && kind != BodyKind::BlackHole && m < 0.05 * ph::M_SUN_ME {
        kv(ui, "  vs Earth", format!("{:.3}×", m));
    }
    match kind {
        BodyKind::BlackHole => {
            // For a black hole, the "radius" is the Schwarzschild radius —
            // the event horizon, not a surface. Showing "surface gravity"
            // would be misleading; showing the horizon radius and capture
            // radius is what a black-hole user actually wants.
            kv(ui, "horizon radius", fmt_len(r, si));
            kv(ui, "capture radius", fmt_len(b.contact_r(i), si));
            kv(ui, "escape speed", "c (light)");
        }
        _ => {
            kv(ui, "radius", fmt_len(r, si));
            let rho = ph::density_gcc(b, i);
            // `density_gcc` returns g/cm³, which is the standard unit
            // for planetary densities. In non-SI mode, we show the
            // (dimensionless) m/r³ instead, which is the raw sim unit.
            kv(
                ui,
                // Planets and dwarf planets carry their own bulk density.
                if matches!(kind, BodyKind::Planet | BodyKind::DwarfPlanet) {
                    "bulk density"
                } else {
                    "density"
                },
                if si {
                    format!("{:.2} g/cm³", rho)
                } else {
                    format!("{:.3e} u", m / (r * r * r))
                },
            );
            // The surface gravity conversion is sim-units → m/s²:
            //   g_code = G * m / r²       (sim units)
            //   g_ms2  = g_code * LEN_m / TU_s²
            // where LEN_m = LEN_KM * 1000 and TU_s = time_unit_s().
            // Then divided by 9.80665 to get "× Earth gravity".
            let g_code = ph::G * m / (r * r);
            let g_ms2 = g_code * (ph::LEN_KM * 1000.0) / (t_unit * t_unit);
            kv(
                ui,
                "surface gravity",
                if si {
                    format!("{:.2} g⊕", g_ms2 / 9.80665)
                } else {
                    format!("{:.2} u", g_code)
                },
            );
            // Escape speed from a (spherical) surface: v_esc = sqrt(2GM/R).
            // Rendered via `fmt_speed` so the units match the rest of the
            // UI (km/s in SI mode).
            let ve = (2.0 * ph::G * m / r).sqrt();
            kv(ui, "escape speed", fmt_speed(ve, si));
        }
    }
}

/// "Stellar" section for stars and stellar remnants.
pub(super) fn stellar_rows(ui: &mut egui::Ui, si: bool, b: &Bodies, i: usize) {
    let (m, kind) = (b.m[i], b.kind[i]);
        heading_row(ui, "Stellar");
        // Evolution-aware accessors: they follow a star's age and the
        // cooling of white and brown dwarfs.
        let l = ph::luminosity_of(b, i);
        // Solar luminosities; neutron stars and white dwarfs get
        // very small values, which `{:e}` handles gracefully.
        kv(ui, "luminosity", format!("{:.3e} L☉", l));
        kv(
            ui,
            "temperature",
            format!("{:.0} K", ph::temperature_of(b, i)),
        );
        kv(
            ui,
            if kind == BodyKind::Star {
                "age"
            } else {
                "cooling age"
            },
            format!("{:.3} Gyr", b.age_gyr[i]),
        );
        if b.m_init[i] > 0.0 && ((b.m_init[i] - m) / b.m_init[i]).abs() > 1e-6 {
            kv(ui, "initial mass", fmt_mass(b.m_init[i], si));
            kv(
                ui,
                "mass lost",
                format!("{:.2} %", 100.0 * (b.m_init[i] - m) / b.m_init[i]),
            );
        }
        if let Some(phase) = ph::phase_of(b, i) {
            kv(
                ui,
                "phase",
                match phase {
                    ph::StellarPhase::MainSequence => "main sequence",
                    ph::StellarPhase::RedGiant => "red giant",
                    ph::StellarPhase::RedSupergiant => "red supergiant",
                },
            );
            let left = ph::star_time_left_gyr(b, i);
            kv(
                ui,
                "time to remnant",
                if left >= 0.1 {
                    format!("{:.2} Gyr", left)
                } else {
                    format!("{:.0} Myr", left * 1000.0)
                },
            );
            let (rk, rm) = ph::remnant_of(b.m_init[i]);
            let name = match rk {
                BodyKind::WhiteDwarf => "white dwarf",
                BodyKind::NeutronStar => "neutron star",
                _ => "black hole",
            };
            kv(
                ui,
                "will become",
                format!("{} ({})", name, ph::mass_str(rm)),
            );
        } else {
            kv(ui, "stage", "remnant");
        }
        // Habitable zone is only defined for bodies that have a
        // luminosity; `habitable_zone_of` returns `(NaN, NaN)` for
        // dark ones, which the `hz0.is_finite()` check filters out.
        let (hz0, hz1) = ph::habitable_zone_of(b, i);
        if hz0.is_finite() {
            kv(ui, "habitable zone", format!("{:.2} – {:.2} AU", hz0, hz1));
        }
}

/// "Climate" section (flux, temperature, zone, rotation, tidal heating) for
/// planets that receive stellar flux.
pub(super) fn climate_rows(ui: &mut egui::Ui, b: &Bodies, i: usize, parent: Option<usize>) {
        // Stellar flux at the body's current position, summed over all
        // luminous bodies in the system. A planet in a binary system
        // sees both stars, and this correctly accounts for that.
        let s = ph::stellar_flux(b, b.x[i], b.y[i], i);
        if s > 0.0 {
            heading_row(ui, "Climate");
            kv(ui, "stellar flux", format!("{:.2} S⊕", s));
            // Equilibrium temperature: T = 255 K * S^(1/4) (Bond albedo 0.3). No greenhouse — actual surface
            // temperatures are higher, but the user gets the right idea.
            kv(
                ui,
                "equilibrium temp",
                format!(
                    "{:.0} K ({:.0} °C)",
                    ph::equilibrium_temp(s),
                    ph::equilibrium_temp(s) - 273.15
                ),
            );
            // "Habitable zone" here is a very rough 0.36..1.1 S_earth
            // (the standard Kasting-ish bounds). Labeled "zone" not
            // "habitable" because the sim doesn't know about atmosphere.
            let (hz_in, hz_out) =
                ph::hz_flux_limits(ph::stellar_flux_teff(b, b.x[i], b.y[i], i));
            let (txt, col) = if s > hz_in {
                ("too hot", theme().colors.warn())
            } else if s >= hz_out {
                ("in habitable zone", theme().colors.good())
            } else {
                ("too cold", theme().colors.accent())
            };
            kv_color(ui, "zone", txt, col);
            if matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) && b.origin[i] == BodyOrigin::Spawned {
                let c = ph::surface_climate(b, i, s, hz_in, hz_out);
                let (label, what) = match c.atmosphere {
                    ph::Atmosphere::Giant => ("cloud-top temp", "giant planet"),
                    ph::Atmosphere::None => ("surface temp (est.)", "airless"),
                    ph::Atmosphere::Thin => ("surface temp (est.)", "thin atmosphere"),
                    ph::Atmosphere::Full if c.frozen => ("surface temp (est.)", "atmosphere, frozen over"),
                    ph::Atmosphere::Full => ("surface temp (est.)", "atmosphere"),
                    ph::Atmosphere::Runaway => ("surface temp (est.)", "runaway greenhouse"),
                };
                kv(ui, label, format!("{:.0} K ({:.0} °C), {}", c.t_surface, c.t_surface - 273.15, what));
            }
        }
        // Rotation and tidal heating (needs a parent to compare against).
        if let Some(p) = parent {
            let inertia = 0.3 * b.m[i] * b.r[i] * b.r[i];
            if inertia > 0.0 && b.spin[i] != 0.0 {
                let omega = b.spin[i] / inertia;
                let o = ph::orbit_info(b, i, p);
                let day_s = std::f64::consts::TAU / omega.abs() * ph::time_unit_s();
                let locked = o.bound
                    && o.period.is_finite()
                    && (omega.abs() * o.period / std::f64::consts::TAU - 1.0).abs() < 0.02;
                let txt = if locked {
                    "tidally locked".to_string()
                } else if day_s < 172_800.0 {
                    format!("{:.1} h day", day_s / 3600.0)
                } else {
                    format!("{:.1} d day", day_s / 86_400.0)
                };
                kv(ui, "rotation", txt);
            }
            let w = ph::tidal_heating_w(b, i, p);
            if w > 1e8 {
                let area = 4.0 * std::f64::consts::PI * (b.r[i] * ph::LEN_KM * 1e3).powi(2);
                kv(
                    ui,
                    "tidal heating",
                    format!("{:.1e} W ({:.2} W/m²)", w, w / area),
                );
            }
        }
}

/// "Orbit" section relative to the parent `p`.
pub(super) fn orbit_rows(ui: &mut egui::Ui, si: bool, b: &Bodies, i: usize, p: usize) {
    let (m, kind) = (b.m[i], b.kind[i]);
    let o = ph::orbit_info(b, i, p);
    heading_row(ui, "Orbit");
    kv(
        ui,
        "orbits",
        format!("#{} {}", b.id[p], ph::class_name(b, p)),
    );
    kv(ui, "distance", fmt_len(o.dist, si));
    kv(ui, "speed", fmt_speed(o.speed_rel, si));
    {
        let rx = b.x[i] - b.x[p];
        let ry = b.y[i] - b.y[p];
        let rvx = b.vx[i] - b.vx[p];
        let rvy = b.vy[i] - b.vy[p];
        let h = rx * rvy - ry * rvx;
        kv(
            ui,
            "direction",
            if h >= 0.0 {
                "counter-clockwise"
            } else {
                "clockwise"
            },
        );
        if o.dist > 0.0 {
            let vc = (ph::G * (b.m[p] + m) / o.dist).sqrt();
            if vc > 0.0 {
                kv(ui, "speed / circular", format!("{:.3}", o.speed_rel / vc));
            }
        }
        kv(ui, "specific ang. mom.", format!("{:.3e}", h.abs()));
        kv(ui, "mass / parent", format!("{:.3e}", m / b.m[p]));
    }
    if o.bound {
        kv(ui, "semi-major axis", fmt_len(o.semi_major, si));
        kv(ui, "eccentricity", format!("{:.4}", o.ecc));
        kv(ui, "period", fmt_time(o.period, si));
        kv(ui, "periapsis", fmt_len(o.peri, si));
        kv(ui, "apoapsis", fmt_len(o.apo, si));
        // The "periapsis inside parent" warning fires when the body's
        // closest approach passes through the parent's physical radius.
        // This is a collision-on-next-periapsis warning — the body is
        // going to merge or impact within one orbit.
        if o.peri < b.contact_r(p) + b.contact_r(i) {
            kv_color(
                ui,
                "warning",
                "periapsis inside parent",
                theme().colors.warn(),
            );
        }
    } else {
        // Hyperbolic / parabolic orbit. Not necessarily a problem — a
        // recently-created debris fragment has an escape trajectory
        // and is still "in the system" in any meaningful sense — but
        // the user needs to know the orbit isn't closed.
        kv_color(ui, "state", "unbound (escape orbit)", theme().colors.warn());
    }
    if matches!(kind, BodyKind::Planet | BodyKind::DwarfPlanet) {
        if matches!(b.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet) {
            // Moon case: show the *parent's* Hill sphere around its own
            // host (e.g. a moon of Jupiter shows Jupiter's Hill sphere
            // around the Sun). This is the relevant scale for moon
            // stability, not the moon's own tiny Hill sphere.
            kv(
                ui,
                "Hill sphere of parent",
                fmt_len(ph::hill_radius(b, p, ph::primary_index(b).unwrap_or(p)), si),
            );
        } else {
            // Planet-of-star case: show the planet's own Hill sphere
            // (the region within which it can host moons).
            kv(ui, "Hill radius", fmt_len(ph::hill_radius(b, i, p), si));
        }
        if b.m[p] > b.m[i] && b.origin[i] == BodyOrigin::Spawned {
            // Roche limit: within this distance, tidal forces exceed
            // self-gravity and the body will be torn apart. Shown in
            // orange within 1.5× the limit — that's the "you're about
            // to be disrupted" pre-warning.
            let roche = ph::roche_limit(b, i, p);
            let col = if o.dist < 1.5 * roche {
                theme().colors.warn()
            } else {
                Color32::GRAY
            };
            kv_color(ui, "Roche limit", fmt_len(roche, si), col);
        }
    }
}

/// "State" section: position, velocity-derived quantities, rank, immunity.
pub(super) fn state_rows(ui: &mut egui::Ui, si: bool, b: &Bodies, i: usize, parent: Option<usize>) {
    let m = b.m[i];
    heading_row(ui, "State");
    kv(
        ui,
        "position",
        format!("{}, {}", fmt_len(b.x[i], si), fmt_len(b.y[i], si)),
    );
    // "speed (frame)" is the body's speed in the simulation's inertial
    // frame — *not* relative to the primary. The orbit section shows
    // relative speed if the body has a parent.
    let v = (b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i]).sqrt();
    kv(ui, "speed (frame)", fmt_speed(v, si));
    kv(ui, "momentum", format!("{:.3e}", m * v));
    kv(ui, "kinetic energy", format!("{:.3e}", 0.5 * m * v * v));
    if b.spin[i] != 0.0 {
        kv(ui, "spin ang. momentum", format!("{:.3e}", b.spin[i]));
    }
    if let Some(prim) = ph::primary_index(b) {
        if prim != i && parent != Some(prim) {
            let dx = b.x[i] - b.x[prim];
            let dy = b.y[i] - b.y[prim];
            kv(
                ui,
                "distance from primary",
                fmt_len((dx * dx + dy * dy).sqrt(), si),
            );
        }
    }
    let heavier = (0..b.len()).filter(|&k| b.m[k] > m).count();
    kv(ui, "mass rank", format!("#{} of {}", heavier + 1, b.len()));
    // Collision immunity: a short post-collision grace period during which
    // the body can't hit anything (so a fragment doesn't immediately
    // re-impact the surviving body). Showing it makes it clear why two
    // overlapping bodies don't immediately merge.
    if b.time < b.imm[i] {
        kv(ui, "collision immunity", fmt_time(b.imm[i] - b.time, si));
    }
    if b.origin[i] != BodyOrigin::Spawned {
        kv(ui, "origin", ph::class_name(b, i));
    }
}

// ---------------------------------------------------------------------------
// Body list
// ---------------------------------------------------------------------------

