//! Split out of the former single-file `ui.rs`; see `mod.rs`.



use crate::physics::{self as ph, Bodies, BodyKind, BodyOrigin};

/// Snapshot of the whole-system statistics, recomputed every 250 ms.
/// `n`, `stars`, etc. are counts; `drift`, `dissipated`, `ang_mom` are
/// conservation diagnostics (see `physics::energy_drift_pct` and friends).
#[derive(Default, Clone)]
pub struct StatsSnapshot {
    pub n: usize,
    pub stars: usize,
    pub black_holes: usize,
    pub planets: usize,
    /// Small debris: belt bodies plus impact, tidal and background debris.
    pub rubble: usize,
    /// Belt bodies only (asteroids, Kuiper populations, Trojans, Hildas).
    /// Also counted in `rubble`.
    pub belt_bodies: usize,
    pub moons: Option<usize>,
    pub total_mass: f64,
    pub drift: Option<f64>,
    pub dissipated: Option<f64>,
    pub ang_mom: f64,
    pub brown_dwarfs: usize,
    pub white_dwarfs: usize,
    pub neutron_stars: usize,
    pub dwarf_planets: usize,
    /// Stars in the red giant or red supergiant phase (also counted in `stars`).
    pub giants: usize,
    /// Comets (origin == Comet), reported separately from planets.
    pub comets: usize,
    /// Interstellar objects (origin == Interstellar).
    pub interstellar: usize,
    /// Rogue planets (origin == RoguePlanet).
    pub rogues: usize,
    /// Mass held by stars and remnants, by planets and dwarfs, and by
    /// debris (belts, rubble, comets).
    pub mass_stellar: f64,
    pub mass_planets: f64,
    pub mass_debris: f64,
    /// Total kinetic energy, simulation units.
    pub kinetic: f64,
    /// Distance of the barycentre from the origin, and its speed.
    pub com_dist: f64,
    pub com_speed: f64,
    /// Heaviest star-like body: label and mass.
    pub primary: Option<(String, f64)>,
    /// Heaviest spawned planet: label and mass.
    pub heaviest_planet: Option<(String, f64)>,
    /// Bound body farthest from the primary: label and distance.
    pub farthest: Option<(String, f64)>,
    /// Bodies on escape orbits about the primary.
    pub unbound: usize,
    /// Spawned planets with a flux between 0.36 and 1.1 S⊕. `None` above
    /// 3000 bodies (one flux sum per planet is too slow there).
    pub hz_planets: Option<usize>,
    /// Age of the oldest star, in Gyr.
    pub oldest_star_gyr: Option<f64>,
}

/// Parent of body `i`: from the parent cache when it is valid, else `find_parent`.
pub(super) fn parent_of(b: &Bodies, i: usize) -> Option<usize> {
    if b.parent_cache_valid && b.parent_cache.len() == b.len() {
        b.parent_cache[i]
    } else {
        ph::find_parent(b, i)
    }
}

/// One pass over `Bodies` to count everything the stats panel needs.
///
/// Cost is O(n) for the class counts plus O(n²) for the moon count (each
/// `find_parent` is O(n)). The moon count is gated on `n <= 1000` — above
/// that, the panel shows "no data" for moons rather than paying 10⁶ lookups
/// per 250 ms. (Which means the "moons" cell in the UI is empty for big
/// simulations. This is a deliberate trade-off; if you want an O(n) version,
/// replace `find_parent` with a Hill-sphere sweep in `physics.rs`.)
pub fn compute_stats(b: &Bodies) -> StatsSnapshot {
    let mut s = StatsSnapshot {
        n: b.len(),
        ..Default::default()
    };
    count_classes(b, &mut s);
    if b.len() <= 1000 {
        count_moons(b, &mut s);
    }
    add_mass_and_motion(b, &mut s);
    if let Some(p) = ph::primary_index(b) {
        add_primary_stats(b, &mut s, p);
    }
    if b.len() <= 3000 {
        s.hz_planets = Some(count_habitable(b));
    }
    s.drift = ph::energy_drift_pct(b);
    s.dissipated = ph::energy_dissipated_pct(b);
    s.ang_mom = ph::total_angular_momentum(b);
    s
}

/// Bodies per class, and the total mass.
pub(super) fn count_classes(b: &Bodies, s: &mut StatsSnapshot) {
    for i in 0..b.len() {
        s.total_mass += b.m[i];
        match b.kind[i] {
            BodyKind::Star => {
                s.stars += 1;
                if matches!(
                    ph::phase_of(b, i),
                    Some(ph::StellarPhase::RedGiant | ph::StellarPhase::RedSupergiant)
                ) {
                    s.giants += 1;
                }
            }
            BodyKind::BrownDwarf => s.brown_dwarfs += 1,
            BodyKind::WhiteDwarf => s.white_dwarfs += 1,
            BodyKind::NeutronStar => s.neutron_stars += 1,
            BodyKind::BlackHole => s.black_holes += 1,
            BodyKind::Planet | BodyKind::DwarfPlanet => count_small_body(b, i, s),
        }
    }
}

/// Count a planet-kind body in the bucket its origin says.
pub(super) fn count_small_body(b: &Bodies, i: usize, s: &mut StatsSnapshot) {
    let o = b.origin[i];
    if o == BodyOrigin::Comet {
        s.comets += 1;
    } else if o == BodyOrigin::Interstellar {
        s.interstellar += 1;
    } else if o == BodyOrigin::RoguePlanet {
        s.rogues += 1;
    } else if o.is_belt() || o.is_rubble() {
        s.rubble += 1;
        if o.is_belt() {
            s.belt_bodies += 1;
        }
    } else if b.kind[i] == BodyKind::Planet {
        s.planets += 1;
    } else {
        s.dwarf_planets += 1;
    }
}

/// Moons: spawned planets whose parent is a spawned planet. Each moon is
/// taken out of the bucket it was counted in.
pub(super) fn count_moons(b: &Bodies, s: &mut StatsSnapshot) {
    let is_spawned_planet = |i: usize| {
        matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
            && b.origin[i] == BodyOrigin::Spawned
    };
    let (mut moons_planet, mut moons_dwarf) = (0, 0);
    for i in (0..b.len()).filter(|&i| is_spawned_planet(i)) {
        if let Some(p) = parent_of(b, i) {
            if is_spawned_planet(p) {
                match b.kind[i] {
                    BodyKind::Planet => moons_planet += 1,
                    _ => moons_dwarf += 1,
                }
            }
        }
    }
    s.moons = Some(moons_planet + moons_dwarf);
    s.planets -= moons_planet.min(s.planets);
    s.dwarf_planets -= moons_dwarf.min(s.dwarf_planets);
}

/// Mass by group, kinetic energy, barycentre motion and the oldest star.
pub(super) fn add_mass_and_motion(b: &Bodies, s: &mut StatsSnapshot) {
    let (mut mx, mut my, mut mvx, mut mvy) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..b.len() {
        let m = b.m[i];
        mx += m * b.x[i];
        my += m * b.y[i];
        mvx += m * b.vx[i];
        mvy += m * b.vy[i];
        s.kinetic += 0.5 * m * (b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i]);
        match b.kind[i] {
            BodyKind::Planet | BodyKind::DwarfPlanet => {
                let o = b.origin[i];
                if o.is_belt() || o.is_rubble() || o.is_comet() {
                    s.mass_debris += m;
                } else {
                    s.mass_planets += m;
                }
            }
            k => {
                s.mass_stellar += m;
                if k == BodyKind::Star {
                    let age = b.age_gyr[i];
                    if s.oldest_star_gyr.is_none_or(|a| age > a) {
                        s.oldest_star_gyr = Some(age);
                    }
                }
            }
        }
    }
    if s.total_mass > 0.0 {
        let cx = mx / s.total_mass;
        let cy = my / s.total_mass;
        s.com_dist = (cx * cx + cy * cy).sqrt();
        s.com_speed = (mvx * mvx + mvy * mvy).sqrt() / s.total_mass;
    }
}

/// Statistics relative to the primary: the unbound count, the heaviest
/// spawned planet and the farthest bound body.
pub(super) fn add_primary_stats(b: &Bodies, s: &mut StatsSnapshot, p: usize) {
    let label = |i: usize| -> String {
        b.name[i]
            .clone()
            .unwrap_or_else(|| format!("#{} {}", b.id[i], ph::class_name(b, i)))
    };
    s.primary = Some((label(p), b.m[p]));
    let mut heaviest: Option<usize> = None;
    let mut far: Option<(usize, f64)> = None;
    for i in (0..b.len()).filter(|&i| i != p) {
        let dx = b.x[i] - b.x[p];
        let dy = b.y[i] - b.y[p];
        let r = (dx * dx + dy * dy).sqrt().max(1e-12);
        let dvx = b.vx[i] - b.vx[p];
        let dvy = b.vy[i] - b.vy[p];
        let mu = ph::G * (b.m[p] + b.m[i]);
        if 0.5 * (dvx * dvx + dvy * dvy) - mu / r >= 0.0 {
            s.unbound += 1;
        } else if far.is_none_or(|(_, d)| r > d) {
            far = Some((i, r));
        }
        if b.kind[i] == BodyKind::Planet
            && b.origin[i] == BodyOrigin::Spawned
            && heaviest.is_none_or(|h| b.m[i] > b.m[h])
        {
            heaviest = Some(i);
        }
    }
    s.heaviest_planet = heaviest.map(|h| (label(h), b.m[h]));
    s.farthest = far.map(|(i, d)| (label(i), d));
}

/// Spawned planets whose stellar flux lies in the habitable-zone limits.
pub(super) fn count_habitable(b: &Bodies) -> usize {
    (0..b.len())
        .filter(|&i| b.kind[i] == BodyKind::Planet && b.origin[i] == BodyOrigin::Spawned)
        .filter(|&i| {
            let f = ph::stellar_flux(b, b.x[i], b.y[i], i);
            let (hi, ho) = ph::hz_flux_limits(ph::stellar_flux_teff(b, b.x[i], b.y[i], i));
            (ho..=hi).contains(&f)
        })
        .count()
}

