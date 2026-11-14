//! egui user interface: control dock, status bar, body panel, body list,
//! event log, help, and the on-canvas overlays (selection ring, labels, scale bar).
//!
//! Threading contract
//! ------------------
//! `draw` is called from the render thread **while it holds the bodies lock**.
//! That means:
//!   * this file may read `Bodies` freely but must not mutate it.
//!   * it must be fast — `find_parent` is O(n) and the stats snapshot re-runs
//!     it for every planet, so the body list and the selection panel are the
//!     two places where a large `n` can hurt.
//!   * it must never block. No locks, no I/O, no `sleep`.
//!
//! Every user action is accumulated into `Actions` and applied by the caller
//! (`main.rs`'s `apply_actions`), *not* performed here. This is what lets a
//! frame be assembled from a single lock acquisition.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use egui::{Align2, Color32, FontId, RichText, Stroke};

use crate::physics::{self as ph, Bodies, BodyKind, BodyOrigin, EventKind};
use crate::theme::{install_fonts, theme};

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------
//
// These helpers share a convention: the `si` argument selects physical units
// vs. simulation units. When `si == false`, everything is in raw sim units
// (Gm, Earth masses, time units). When `si == true`, values are formatted
// in the most natural scale — AU for distances, M_sun / M_jup / M_earth for
// masses, years for times. The `si` path is for the user who wants to know
// "what would this look like?", the non-`si` path is for cross-checking the
// integrator.

pub fn fmt_len(gm: f64, si: bool) -> String {
    if !gm.is_finite() {
        return "–".into();
    }
    if !si {
        return format!("{:.3} u", gm);
    }
    let a = gm.abs();
    let au = ph::au_in_gm();
    // The AU threshold is 0.1 AU, not 1 AU: a value of "0.5 AU" is much
    // easier to read at a glance than "75 Gm". Below 0.1 AU, Gm becomes
    // useful (planet-sized things); below 1 Gm, km becomes useful.
    if a >= 0.1 * au {
        format!("{:.3} AU", gm / au)
    } else if a >= 1.0 {
        format!("{:.2} Gm", gm)
    } else if a * 1e6 >= 1.0 {
        format!("{:.0} km", gm * 1e6)
    } else {
        // Sub-km bodies (the smallest debris) get scientific notation.
        format!("{:.2e} km", gm * 1e6)
    }
}

pub fn fmt_mass(me: f64, si: bool) -> String {
    if !si {
        return format!("{:.4} u", me);
    }
    // The M_sun threshold (5 % of a solar mass) is where "M_sun" becomes
    // more informative than "M_jup": at 5 % M_sun, the body is already a
    // brown dwarf or low-mass star, and M_jup numbers get unwieldy
    // (~523 M_jup for a 0.5 M_sun object).
    if me >= 0.05 * ph::M_SUN_ME {
        format!("{:.3} M☉", me / ph::M_SUN_ME)
    } else if me >= 30.0 {
        format!("{:.2} M♃", me / ph::JUPITER_MASS)
    } else if me >= 1e-3 {
        format!("{:.3} M⊕", me)
    } else {
        // Below 1e-3 M_earth (the smallest debris), kg is more intuitive
        // than "1.5e-3 M_earth".
        format!("{:.2e} kg", me * ph::MASS_KG)
    }
}

pub fn fmt_time(t: f64, si: bool) -> String {
    if !t.is_finite() {
        return "–".into();
    }
    if !si {
        return format!("{:.3} tu", t);
    }
    let s = t * ph::time_unit_s();
    let day = 86_400.0;
    let yr = 365.25 * day;
    // The unit thresholds here are tuned for the actual sim speeds:
    //   * < 2h: "minutes" is the natural scale for a close encounter.
    //   * < 2 days: "hours" for a fast moon.
    //   * < 1.5 yr: "days" for the inner planets.
    //   * < 1e4 yr: "years" for the outer planets.
    //   * < 1e7 yr: "kiloyears" for a long sim run.
    //   * else: "megayears" — the sim can't reach gigayears at 100× speed,
    //     so Myr is the terminal unit.
    if s < 2.0 * 3600.0 {
        format!("{:.0} min", s / 60.0)
    } else if s < 2.0 * day {
        format!("{:.1} h", s / 3600.0)
    } else if s < 1.5 * yr {
        format!("{:.1} d", s / day)
    } else if s < 1e4 * yr {
        format!("{:.2} yr", s / yr)
    } else if s < 1e7 * yr {
        format!("{:.2} kyr", s / (1e3 * yr))
    } else {
        format!("{:.2} Myr", s / (1e6 * yr))
    }
}

pub fn fmt_speed(v: f64, si: bool) -> String {
    if !si {
        return format!("{:.2} u/t", v);
    }
    // The conversion: 1 sim length unit = 1e6 km, 1 sim time unit ≈ 50 days.
    // So `v * LEN_KM / time_unit_s()` gives km/s. (LEN_KM = 1e6, so
    // `v * 1e6 / 4.3e6 ≈ v * 0.23` — a body moving at `v = 1` sim unit per
    // sim time unit is moving at ~0.2(≈390 km/s).)
    let kms = v * ph::LEN_KM / ph::time_unit_s();
    if kms >= 1000.0 {
        format!("{:.0} km/s", kms)
    } else {
        format!("{:.2} km/s", kms)
    }
}

fn fmt_pct(p: Option<f64>) -> String {
    match p {
        // The 0.001 % floor switches to scientific notation so a
        // "perfectly conserving" run doesn't display as "0.000 %" and make
        // the user wonder if the test is broken. Showing "1.2e-7 %" makes
        // it clear that *something* is being measured.
        Some(v) if v.abs() < 0.001 => format!("{:.1e} %", v),
        Some(v) => format!("{:.3} %", v),
        None => "n/a".into(),
    }
}

fn fmt_count(n: u64) -> String {
    // The 10k threshold is chosen so the widest possible count
    // ("9999") still fits in the stats grid column. "10.0k" is more compact
    // and conveys the same magnitude.
    if n >= 10_000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

// ---------------------------------------------------------------------------
// State shared with main
// ---------------------------------------------------------------------------
//
// `SortKey` and `ListRow` are the body-list cache. `ListRow` is a *denormalized*
// snapshot: it's computed once per ~300 ms so the list window can render
// without touching `Bodies` (which would need the parent lookup — O(n) per
// row). If you add a column, add the field to `ListRow` and fill it in
// `refresh_rows`, not in the render loop.

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SortKey {
    Id,
    Mass,
    Radius,
    Dist,
    Speed,
    Period,
}

pub struct ListRow {
    pub id: u64,
    pub label: String,
    pub class: String,
    pub m: f64,
    pub r: f64,
    pub dist: f64,
    pub speed: f64,
    pub period: f64,
}

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
            BodyKind::Planet | BodyKind::DwarfPlanet => {
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
        }
    }
    if b.len() <= 1000 {
        let mut moons_planet = 0;
        let mut moons_dwarf = 0;
        for i in 0..b.len() {
            if matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
                && b.origin[i] == BodyOrigin::Spawned
            {
                if let Some(p) = ph::find_parent(b, i) {
                    if matches!(b.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet)
                        && b.origin[p] == BodyOrigin::Spawned
                    {
                        match b.kind[i] {
                            BodyKind::Planet => moons_planet += 1,
                            _ => moons_dwarf += 1,
                        }
                    }
                }
            }
        }
        s.moons = Some(moons_planet + moons_dwarf);
        // Subtract from the bucket each moon was counted in.
        s.planets -= moons_planet.min(s.planets);
        s.dwarf_planets -= moons_dwarf.min(s.dwarf_planets);
    }
    // ---- extra statistics, one O(n) pass ---------------------------------
    let n = b.len();
    let label = |i: usize| -> String {
        b.name[i]
            .clone()
            .unwrap_or_else(|| format!("#{} {}", b.id[i], ph::class_name(b, i)))
    };
    let (mut mx, mut my, mut mvx, mut mvy) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
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
    if let Some(p) = ph::primary_index(b) {
        s.primary = Some((label(p), b.m[p]));
        let mut heaviest: Option<usize> = None;
        let mut far: Option<(usize, f64)> = None;
        for i in 0..n {
            if i == p {
                continue;
            }
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
    if n <= 3000 {
        let mut hz = 0;
        for i in 0..n {
            if b.kind[i] == BodyKind::Planet && b.origin[i] == BodyOrigin::Spawned {
                let f = ph::stellar_flux(b, b.x[i], b.y[i], i);
                if (0.36..=1.1).contains(&f) {
                    hz += 1;
                }
            }
        }
        s.hz_planets = Some(hz);
    }
    s.drift = ph::energy_drift_pct(b);
    s.dissipated = ph::energy_dissipated_pct(b);
    s.ang_mom = ph::total_angular_momentum(b);
    s
}

/// One point in the diagnostics history. Sampled once per wall-clock second.
/// Percent values are `None` when they have no meaning (for example the
/// energy drift above `ENERGY_MAX_N` bodies).
pub struct DiagSample {
    pub t: f64,
    pub n: usize,
    /// Integrator energy drift, percent of the baseline energy.
    pub drift: Option<f64>,
    /// Energy removed or added by merges, break-ups and escapes, percent.
    pub dissipated: Option<f64>,
    /// Change of total angular momentum since the baseline, percent.
    pub l_drift: Option<f64>,
    /// Fine time steps used in the last step (close encounters raise this).
    pub nsub: u32,
}

pub struct DiagHistory {
    pub samples: std::collections::VecDeque<DiagSample>,
    pub max_samples: usize,
    pub last_sample: Instant,
    pub eccs: Vec<f64>,
    pub ecc_bins: Vec<usize>,
    pub ecc_bin_width: f64,
    pub ecc_bin_count: usize,
    pub visible: bool,
    /// Angular momentum baseline. Set at the first sample and again when the
    /// energy baseline changes (a user edit re-anchors both).
    l0: Option<f64>,
    e0_seen: f64,
}

impl DiagHistory {
    pub fn new() -> Self {
        Self {
            samples: std::collections::VecDeque::new(),
            max_samples: 300,
            // `now - 10s` forces the first sample on the first frame.
            last_sample: Instant::now() - Duration::from_secs(10),
            eccs: Vec::new(),
            ecc_bins: vec![0; 20],
            ecc_bin_width: 0.05,
            ecc_bin_count: 20,
            // Closed at launch. Open it with G or the View section.
            visible: false,
            l0: None,
            e0_seen: 0.0,
        }
    }

    /// Forget all samples and the angular-momentum baseline.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.eccs.clear();
        self.ecc_bins.fill(0);
        self.l0 = None;
    }

    pub fn maybe_sample(&mut self, b: &Bodies) {
        if self.last_sample.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.last_sample = Instant::now();

        if let Some(last) = self.samples.back() {
            if b.time < last.t {
                // Time went backwards: a new system replaced the old one.
                self.clear();
            } else if b.time == last.t {
                // Paused: a second sample at the same time adds nothing.
                return;
            }
        }

        let n = b.len();
        let drift = if n <= ph::ENERGY_MAX_N {
            ph::energy_drift_pct(b)
        } else {
            None
        };
        let dissipated = ph::energy_dissipated_pct(b);

        let l = ph::total_angular_momentum(b);
        if self.e0_seen != b.stats.energy0 {
            self.e0_seen = b.stats.energy0;
            self.l0 = None;
        }
        let l0 = *self.l0.get_or_insert(l);
        let l_drift = if l0.abs() > 1e-12 && l.is_finite() {
            Some((l - l0) / l0.abs() * 100.0)
        } else {
            None
        };

        self.samples.push_back(DiagSample {
            t: b.time,
            n,
            drift,
            dissipated,
            l_drift,
            nsub: b.stats.last_nsub,
        });
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }

        self.eccs.clear();
        for i in 0..b.len() {
            if self.eccs.len() >= 500 {
                break;
            }
            if b.origin[i] != BodyOrigin::Spawned {
                continue;
            }
            if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
                continue;
            }
            if let Some(p) = ph::find_parent(b, i) {
                let o = ph::orbit_info(b, i, p);
                // Hyperbolic bodies (e > 1) have no bin: only bound orbits count.
                if o.bound && o.ecc.is_finite() {
                    self.eccs.push(o.ecc);
                }
            }
        }
        // `min(count - 1)` puts e = 1.0 in the last bin.
        self.ecc_bins.fill(0);
        for &e in &self.eccs {
            let bin = ((e / self.ecc_bin_width) as usize).min(self.ecc_bin_count - 1);
            self.ecc_bins[bin] += 1;
        }
    }

    /// The history as CSV text (time in simulation units).
    pub fn to_csv(&self) -> String {
        let opt = |v: Option<f64>| v.map(|x| format!("{x:e}")).unwrap_or_default();
        let mut s = String::from(
            "t,n,energy_drift_pct,energy_dissipated_pct,ang_mom_drift_pct,fine_steps\n",
        );
        for p in &self.samples {
            s.push_str(&format!(
                "{:e},{},{},{},{},{}\n",
                p.t,
                p.n,
                opt(p.drift),
                opt(p.dissipated),
                opt(p.l_drift),
                p.nsub
            ));
        }
        s
    }
}

impl Default for DiagHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// All UI-owned state that isn't in `SimState` and isn't in `Bodies`.
///
/// `pending_count` / `pending_seed` are *pending* because they only take
/// effect when the user clicks Generate (or presses the corresponding key).
/// They can diverge from the live `view.body_count` / `view.seed` while the
/// user is typing a new count into the slider.
pub struct UiState {
    pub units_si: bool,
    pub show_labels: bool,
    pub show_list: bool,
    pub show_help: bool,
    pub show_dock: bool,
    pub show_trails: bool,
    /// Slug of the preset that is loaded now, if any.
    pub loaded_preset_slug: Option<String>,
    pub diag: DiagHistory,
    pub pending_count: usize,
    /// Seed text in base62 (6 chars max). Parse with `commit_seed_text`.
    pub pending_seed_text: String,
    /// The seed text is not valid base62 (shows a red hint).
    pub seed_text_invalid: bool,
    /// Value of the "systems" slider. Applies to both presets.
    pub pending_systems: usize,
    pub sort: SortKey,
    pub sort_desc: bool,
    pub list_limit: usize,
    /// Text typed in the body-list search box (matches name or class).
    pub list_query: String,
    /// Bodies ticked in the body list, by id. Edits apply to all of them.
    pub multi: std::collections::BTreeSet<u64>,
    /// Bit mask of event kinds shown in the Events list (see `event_bit`).
    pub event_mask: u16,
    pub list_rows: Vec<ListRow>,
    list_time: Instant,
    pub stats: StatsSnapshot,
    stats_time: Instant,
}

/// A fresh random seed that uses all `SEED_LEN` base62 digits, so the text
/// shown in the seed box never starts with a long run of zeros.
///
/// The source is the clock and the address of a stack value, mixed with an
/// LCG. It is not cryptographic. It only has to differ between launches.
pub fn random_seed() -> u128 {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let probe = 0u8;
    let mut x = (d.subsec_nanos() as u64)
        ^ d.as_secs().wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (&probe as *const u8 as u64).wrapping_mul(2_654_435_761);
    let mut v: u128 = 0;
    for _ in 0..ph::SEED_LEN {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        v = v * 62 + ((x >> 33) % 62) as u128;
    }
    v
}

/// Parse seed text as base62. Return `fallback` when the text is not valid.
pub fn commit_seed_text(text: &str, fallback: u128) -> u128 {
    ph::seed_from_base62(text.trim()).unwrap_or(fallback)
}

impl UiState {
    /// Show `seed` in the seed input box.
    pub fn set_pending_seed(&mut self, seed: u128) {
        self.pending_seed_text = ph::seed_to_base62(seed);
    }

    /// Seed value of the input box, or `fallback` if the text is invalid.
    pub fn pending_seed(&self, fallback: u128) -> u128 {
        commit_seed_text(&self.pending_seed_text, fallback)
    }

    pub fn new(body_count: usize, seed: u128) -> Self {
        let old = Instant::now() - Duration::from_secs(10);
        Self {
            units_si: true,
            show_labels: false,
            show_list: false,
            show_help: false,
            show_dock: true,
            show_trails: true,
            loaded_preset_slug: None,
            diag: DiagHistory::new(),
            pending_count: body_count,
            pending_seed_text: ph::seed_to_base62(seed),
            seed_text_invalid: false,
            pending_systems: 1,
            // Default sort is descending by mass, which puts the star and
            // giant planets at the top of the list — the bodies a new user
            // is most likely to want to click.
            sort: SortKey::Mass,
            sort_desc: true,
            list_limit: 100,
            list_query: String::new(),
            multi: std::collections::BTreeSet::new(),
            event_mask: 0x1FF,
            list_rows: Vec::new(),
            // `old` is 10 seconds ago so the first `stats_time.elapsed()`
            // is large and the first frame computes stats immediately.
            list_time: old,
            stats: StatsSnapshot::default(),
            stats_time: old,
        }
    }
}

/// Everything the UI needs to know about the view, read-only.
/// Built once per frame in `main.rs` from `SimState` + `Shared`. Keeping this
/// as a plain struct means `ui.rs` has no dependency on `main.rs`.
pub struct ViewInfo {
    pub paused: bool,
    pub speed: f64,
    pub fps: f32,
    pub steps_per_sec: f32,
    pub zoom: f32,
    pub cam: [f64; 2],
    pub viewport_px: [f32; 2],
    pub ppp: f32,
    pub seed: u128,
    /// Bodies per system (the "bodies" slider target). For Solar this is
    /// the spawned system's body count; for Random it's the user's `pending_count` at spawn time.
    pub body_count: usize,
    /// Number of systems currently spawned (the "systems" slider target).
    pub systems: usize,
    /// Current live body count across all systems (`b.len()`). Differs from
    /// `body_count * systems` after merges/escapes.
    pub total_bodies: usize,
    pub show_orbits: bool,
    pub show_debris_orbits: bool,
    pub follow: bool,
    pub selected: Option<u64>,
    pub follow_smooth: Option<(u64, [f64; 2])>,
    pub preset: crate::Preset,
    pub vsync: bool,
    pub behind: bool,
    /// Pause the simulation when a merger, impact or tidal break-up happens.
    pub pause_on_event: bool,
}

/// A single body edit requested by the UI. Applied by `SimState::apply_actions`
/// in `main.rs`. `MassMul` is a *factor* so the same enum variant can serve
/// both "double" and "halve."
pub enum Edit {
    MassMul(f64),
    SetKind(BodyKind),
    Circularize,
    Delete,
    SpawnMoon,
    AddRing,
}

/// Accumulated actions for one frame. `Option<T>` means "user requested T
/// this frame"; `None` means "user said nothing about T" (so the current
/// value should be preserved). `bool` means "event happened this frame."
///
/// This is deliberately a flat struct, not an enum-of-enums: the render
/// loop applies actions in order in `apply_actions`, and flatness keeps
/// the match arms simple.
#[derive(Default)]
pub struct Actions {
    pub reset: bool,
    pub toggle_pause: bool,
    pub step_once: bool,
    pub speed: Option<f64>,
    pub body_count: Option<usize>,
    pub seed: Option<u128>,
    pub orbits: Option<bool>,
    pub debris_orbits: Option<bool>,
    pub zoom_mul: Option<f32>,
    pub fit: bool,
    pub select: Option<Option<u64>>,
    pub center_on: Option<[f64; 2]>,
    pub follow: Option<bool>,
    pub cycle: i32,
    pub edit: Option<Edit>,
    pub preset: Option<crate::Preset>,
    /// Number of systems requested by the "systems" slider. Applies to
    /// both presets.
    pub systems: Option<usize>,
    pub vsync: Option<bool>,
    pub trails: Option<bool>,
    pub diag_visible: Option<bool>,
    pub pause_on_event: Option<bool>,
    /// One edit applied to many bodies (ids from the body list), committed once.
    pub multi_edit: Option<(Vec<u64>, Edit)>,
    /// Slug of a preset from the library to load.
    pub load_preset: Option<String>,
}

// ---------------------------------------------------------------------------
// Styling
// ---------------------------------------------------------------------------

// Colours come from `config/ui.json` through `theme()`.

/// Apply the app's look to the egui context. Called once at startup.
/// Overriding `Visuals::dark()` (rather than building from scratch) means
/// most widgets keep sensible defaults; only the ones we care about are
/// touched.
pub fn setup_style(ctx: &egui::Context) {
    // Fonts first.
    install_fonts(ctx);

    let t = theme();
    let mut v = egui::Visuals::dark();
    v.panel_fill = t.colors.panel_fill();
    v.window_fill = t.colors.window_fill();
    v.window_stroke = Stroke::new(1.0_f32, Color32::from_gray(48));
    v.window_rounding = egui::Rounding::same(t.spacing.window_rounding);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, Color32::from_gray(40));
    v.selection.bg_fill = t.colors.selection_bg();
    v.override_text_color = Some(t.colors.text());
    ctx.set_visuals(v);

    let mut style = (*ctx.style()).clone();
    // Tighter spacing than egui's default: the dock is dense, and the
    // default 8×4 px item spacing makes it 20 % taller for no benefit.
    style.spacing.item_spacing = egui::vec2(t.spacing.item_spacing[0], t.spacing.item_spacing[1]);
    style.spacing.button_padding =
        egui::vec2(t.spacing.button_padding[0], t.spacing.button_padding[1]);
    use egui::{FontFamily, FontId, TextStyle};
    style.text_styles = [
        (
            TextStyle::Body,
            FontId::new(t.fonts.body_size, FontFamily::Proportional),
        ),
        (
            TextStyle::Button,
            FontId::new(t.fonts.body_size, FontFamily::Proportional),
        ),
        (
            TextStyle::Small,
            FontId::new(t.fonts.small_size, FontFamily::Proportional),
        ),
        (
            TextStyle::Heading,
            FontId::new(t.fonts.heading_size, FontFamily::Proportional),
        ),
        (
            TextStyle::Monospace,
            FontId::new(t.fonts.mono_size, FontFamily::Monospace),
        ),
    ]
    .into();
    ctx.set_style(style);
}
fn panel_frame() -> egui::Frame {
    egui::Frame::none()
        .fill(theme().colors.panel_fill())
        .inner_margin(egui::Margin::same(theme().spacing.panel_margin))
        .stroke(Stroke::new(1.0_f32, Color32::from_gray(40)))
}

/// Key-value row in an `egui::Grid`. Assumes a 2-column grid. The caller
/// must call `ui.end_row()` after the pair — that's why this takes `ui`
/// and not a grid handle.
fn kv(ui: &mut egui::Ui, k: &str, v: impl Into<String>) {
    ui.label(RichText::new(k).color(theme().colors.dim()));
    ui.label(RichText::new(v.into()).monospace());
    ui.end_row();
}

/// Same as `kv` but colors the value. Used for pass/fail (green/red/orange).
fn kv_color(ui: &mut egui::Ui, k: &str, v: impl Into<String>, c: Color32) {
    ui.label(RichText::new(k).color(theme().colors.dim()));
    ui.label(RichText::new(v.into()).monospace().color(c));
    ui.end_row();
}

/// Section divider in the stats grid: strong, colored, spans one cell.
/// The empty second cell is filled with `ui.label("")` so the grid layout
/// doesn't collapse to a single column for the header row.
fn heading_row(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).strong().color(theme().colors.accent()));
    ui.label("");
    ui.end_row();
}

// ---------------------------------------------------------------------------
// Main entry
// ---------------------------------------------------------------------------

/// Draw everything for one frame and return the accumulated actions.
///
/// Order matters: overlays go under the panels so panels are readable on
/// top, and the status bar is drawn before the dock so the dock's scroll
/// area gets the remaining height.
pub fn draw(ctx: &egui::Context, ui_state: &mut UiState, view: &ViewInfo, b: &Bodies) -> Actions {
    let mut act = Actions::default();

    // Stats are O(n²) for moons — recompute only every 250 ms, not every
    // frame. This is the main reason `stats` is a cached snapshot rather
    // than a live query.
    if ui_state.stats_time.elapsed() > Duration::from_millis(250) {
        ui_state.stats = compute_stats(b);
        ui_state.stats_time = Instant::now();
    }

    draw_overlays(ctx, ui_state, view, b);
    status_bar(ctx, ui_state, view, b);
    if ui_state.show_dock {
        dock(ctx, ui_state, view, b, &mut act);
    }
    if ui_state.diag.visible {
        diag_window(ctx, ui_state, view, b, &mut act);
    }
    // Selection window only shows if the selected body still exists. This
    // is the "the selection body merged away last frame" case: `view.selected`
    // is Some(id) but `b.index_of_id(id)` is None. Clear via the `if let`
    // falling through, which is why the check lives here and not in
    // `main.rs`'s cleanup.
    if let Some(id) = view.selected {
        if let Some(i) = b.index_of_id(id) {
            selection_window(ctx, ui_state, view, b, i, &mut act);
        }
    }
    if ui_state.show_list {
        list_window(ctx, ui_state, view, b, &mut act);
    }
    if ui_state.show_help {
        help_window(ctx, ui_state);
    }
    act
}

// ---------------------------------------------------------------------------
// Overlays drawn on the canvas
// ---------------------------------------------------------------------------

/// Convert world → screen **points** (not pixels). egui works in points
/// (logical pixels), so the result is divided by `ppp`. The `viewport_px`
/// field name is a bit of a misnomer — it's pixels, but we're converting
/// to points here.
fn world_to_screen(view: &ViewInfo, x: f64, y: f64) -> egui::Pos2 {
    let px = view.viewport_px[0] as f64 * 0.5 + (x - view.cam[0]) * view.zoom as f64;
    let py = view.viewport_px[1] as f64 * 0.5 + (y - view.cam[1]) * view.zoom as f64;
    egui::pos2(px as f32 / view.ppp, py as f32 / view.ppp)
}

/// Snap a length to a "nice" value (1, 2, 5 × 10^k) for the scale bar.
/// This is the standard "human-readable number line" algorithm: take the
/// floor of the decade, then pick the closest of {1, 2, 5, 10} × decade.
fn nice_length(target: f64) -> f64 {
    let p = 10f64.powf(target.log10().floor());
    let f = target / p;
    let n = if f < 1.5 {
        1.0
    } else if f < 3.5 {
        2.0
    } else if f < 7.5 {
        5.0
    } else {
        10.0
    };
    n * p
}

/// World-space position to draw body `i` at. For the followed body this
/// returns the smoothed position from `ViewInfo::follow_smooth`; for
/// everything else it returns the raw physics position.
#[inline]
fn drawn_pos(view: &ViewInfo, b: &Bodies, i: usize) -> (f64, f64) {
    if let Some((id, pos)) = view.follow_smooth {
        if b.id[i] == id {
            return (pos[0], pos[1]);
        }
    }
    (b.x[i], b.y[i])
}

/// Draw the selection ring, labels, and scale bar. All three are drawn on
/// the `background` layer so they appear under egui's own layers (the dock,
/// the status bar) but over the wgpu scene.
fn draw_overlays(ctx: &egui::Context, st: &UiState, view: &ViewInfo, b: &Bodies) {
    let painter = ctx.layer_painter(egui::LayerId::background());
    let screen = ctx.screen_rect();

    // Selection ring.
    if let Some(id) = view.selected {
        if let Some(i) = b.index_of_id(id) {
            let (bx, by) = drawn_pos(view, b, i);
            let c = world_to_screen(view, bx, by);
            let r_px = (b.r[i] * view.zoom as f64) as f32 / view.ppp;
            // Ring is 5 points outside the body, or 5 points outside a
            // 5-point minimum — so a tiny body still gets a visible ring.
            let r = r_px.max(5.0) + 5.0;
            painter.circle_stroke(c, r, Stroke::new(1.5_f32, theme().colors.accent()));
            let label = b.name[i].clone().unwrap_or_else(|| format!("#{}", b.id[i]));
            painter.text(
                c + egui::vec2(r + 4.0, -r),
                Align2::LEFT_BOTTOM,
                format!("{} {}", label, ph::class_name(b, i)),
                FontId::monospace(11.0),
                theme().colors.accent(),
            );
        }
    }

    // Labels.
    // Gated on `b.len() <= 5000` because every label is a text shaping +
    // glyph layout call, and 5000 is roughly where that starts to be the
    // dominant cost in a frame.
    if st.show_labels && b.len() <= 5000 {
        for i in 0..b.len() {
            if b.origin[i] != BodyOrigin::Spawned {
                continue;
            }
            let (bx, by) = drawn_pos(view, b, i);
            let c = world_to_screen(view, bx, by);
            // Cull off-screen labels — `painter.text` still does layout
            // work even for off-screen glyphs on some backends.
            if !screen.contains(c) {
                continue;
            }
            let r_px = (b.r[i] * view.zoom as f64) as f32 / view.ppp;
            let text = b.name[i].clone().unwrap_or_else(|| format!("#{}", b.id[i]));
            painter.text(
                c + egui::vec2(r_px.max(2.0) + 3.0, 0.0),
                Align2::LEFT_CENTER,
                text,
                FontId::monospace(9.0),
                Color32::from_white_alpha(130),
            );
        }
    }

    // Scale bar.
    // Target screen length is 110 points; convert to world units at the
    // current zoom, snap to a nice number, then convert back to points.
    // `nice_length` rounds to the nearest 1/2/5 × 10^k, so the drawn
    // length can overshoot 110 points (up to ~157); that's fine for a
    // decorative bar but not a strict bound.
    let target_px = 110.0 * view.ppp as f64;
    let len_gm = nice_length(target_px / view.zoom as f64);
    let len_px = (len_gm * view.zoom as f64) as f32 / view.ppp;
    // y = bottom - 22 points, centered horizontally.
    let y = screen.bottom() - 22.0;
    let x1 = screen.center().x + len_px * 0.5;
    let x0 = x1 - len_px;
    let col = Color32::from_white_alpha(170);
    painter.line_segment(
        [egui::pos2(x0, y), egui::pos2(x1, y)],
        Stroke::new(1.5_f32, col),
    );
    // End caps: 4-point vertical ticks at each end.
    painter.line_segment(
        [egui::pos2(x0, y - 4.0), egui::pos2(x0, y + 4.0)],
        Stroke::new(1.5_f32, col),
    );
    painter.line_segment(
        [egui::pos2(x1, y - 4.0), egui::pos2(x1, y + 4.0)],
        Stroke::new(1.5_f32, col),
    );
    painter.text(
        egui::pos2((x0 + x1) * 0.5, y - 6.0),
        Align2::CENTER_BOTTOM,
        fmt_len(len_gm, st.units_si),
        FontId::monospace(11.0),
        col,
    );
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

/// Top-of-window status strip: running state, speed, time, body count, fps,
/// warnings, and a hint about the help key.
fn status_bar(ctx: &egui::Context, st: &mut UiState, view: &ViewInfo, b: &Bodies) {
    egui::TopBottomPanel::top("status")
        .frame(panel_frame().inner_margin(egui::Margin::symmetric(10.0, 4.0)))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                // If the dock is hidden, surface a way to bring it back. The
                // button label is "☰ panel" to match the hamburger menu
                // idiom; the tooltip shows the keyboard shortcut.
                if !st.show_dock {
                    if ui
                        .small_button("☰ panel")
                        .on_hover_text("show the control panel (`)")
                        .clicked()
                    {
                        st.show_dock = true;
                    }
                    ui.separator();
                }
                let (txt, col) = if view.paused {
                    ("⏸ paused", theme().colors.warn())
                } else {
                    ("▶ running", theme().colors.good())
                };
                ui.label(RichText::new(txt).color(col).strong());
                ui.separator();
                ui.label(RichText::new(format!("{:.2}×", view.speed)).monospace());
                ui.separator();
                ui.label(
                    RichText::new(format!("t = {}", fmt_time(b.time, st.units_si))).monospace(),
                );
                ui.separator();
                let count_txt = if view.systems > 1 {
                    format!("{} bodies ({} systems)", view.total_bodies, view.systems)
                } else {
                    format!("{} bodies", view.total_bodies)
                };
                ui.label(RichText::new(count_txt).monospace());
                ui.separator();
                ui.label(
                    RichText::new(format!("{:.0} fps", view.fps))
                        .monospace()
                        .color(theme().colors.dim()),
                );
                // Only show the "close encounter" warning when the block
                // step has actually been subdivided. `last_nsub = 1 << kmax`,
                // so `last_nsub > 4` means rung ≥ 3 (8× or more fine ticks).
                ui.separator();
                let drift = st.stats.drift;
                let drift_col = match drift {
                    Some(d) if d.abs() > 1.0 => theme().colors.warn(),
                    _ => theme().colors.dim(),
                };
                ui.label(
                    RichText::new(format!("ΔE {}", fmt_pct(drift)))
                        .monospace()
                        .color(drift_col),
                )
                .on_hover_text("integrator energy drift (n/a for very large systems)");
                if b.stats.last_nsub > 4 {
                    ui.separator();
                    ui.label(
                        RichText::new(format!("close encounter: {} fine ticks", b.stats.last_nsub))
                            .monospace()
                            .color(theme().colors.warn()),
                    );
                }
                if view.behind {
                    ui.separator();
                    ui.label(
                        RichText::new("⚠ simulation behind")
                            .monospace()
                            .color(theme().colors.warn()),
                    );
                }
                // Right-aligned "? help" so the top-right corner is a
                // consistent affordance regardless of what's left-aligned.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("? help")
                        .on_hover_text("keyboard shortcuts (H)")
                        .clicked()
                    {
                        st.show_help = !st.show_help;
                    }
                });
            });
        });
}

// ---------------------------------------------------------------------------
// Dock
// ---------------------------------------------------------------------------

/// The left control panel. Everything is a `CollapsingHeader` so the panel
/// can be scrolled to whatever the user cares about.
fn dock(ctx: &egui::Context, st: &mut UiState, view: &ViewInfo, b: &Bodies, act: &mut Actions) {
    egui::SidePanel::left("dock")
        .frame(panel_frame())
        .default_width(300.0)
        .min_width(240.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Orbital N-Body").strong().size(15.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("hide")
                        .on_hover_text("hide this panel (`)")
                        .clicked()
                    {
                        st.show_dock = false;
                    }
                });
            });
            ui.add_space(2.0);
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // Section order is deliberate: the controls a new user
                    // needs first are at the top (Simulation, Statistics),
                    // and the ones that mutate the system (Generate) are
                    // near the bottom so a misclick on "New system" doesn't
                    // immediately reset a system they just made.
                    section_simulation(ui, st, view, b, act);
                    section_stats(ui, st, view, b);
                    section_view(ui, st, view, act);
                    section_advanced(ui, st, view, act);
                    section_generate(ui, st, view, act);
                    section_events(ui, st, b, act);
                });
        });
}

fn section_simulation(
    ui: &mut egui::Ui,
    _st: &mut UiState,
    view: &ViewInfo,
    _b: &Bodies,
    act: &mut Actions,
) {
    egui::CollapsingHeader::new(RichText::new("Simulation").strong())
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .button("New system")
                    .on_hover_text("generate with the next seed (R)")
                    .clicked()
                {
                    act.reset = true;
                }
                let label = if view.paused { "Resume" } else { "Pause" };
                if ui.button(label).on_hover_text("Space").clicked() {
                    act.toggle_pause = true;
                }
                if ui.button("Step").on_hover_text("one step (.)").clicked() {
                    act.step_once = true;
                }
            });
            // Speed slider is logarithmic: the useful range spans 0.01× to
            // 100×, and a linear slider would spend 99 % of its length on
            // 0.01×–2×.
            let mut speed_log = view.speed.log10();
            let slider = egui::Slider::new(&mut speed_log, -2.0..=2.0)
                .text("speed")
                .custom_formatter(|v, _| format!("{:.2}×", 10f64.powf(v)));
            if ui.add(slider).changed() {
                // Only emit an action when the value actually changed, so
                // dragging the slider doesn't spam `act.speed` every frame.
                act.speed = Some(10f64.powf(speed_log));
            }
            // Quick-speed buttons for the common values. Clicking sets the
            // exact value (not a factor), so a user who wants "1×" gets
            // exactly 1× regardless of where the slider was.
            ui.horizontal(|ui| {
                for s in [0.1, 0.5, 1.0, 4.0, 16.0, 64.0] {
                    if ui.small_button(format!("{}×", s)).clicked() {
                        act.speed = Some(s);
                    }
                }
            });
            let mut pause_ev = view.pause_on_event;
            if ui
                .checkbox(&mut pause_ev, "pause on merger / impact")
                .on_hover_text("pause when a merger, impact or tidal break-up happens")
                .changed()
            {
                act.pause_on_event = Some(pause_ev);
            }
        });
}

fn section_stats(ui: &mut egui::Ui, st: &mut UiState, view: &ViewInfo, b: &Bodies) {
    let si = st.units_si;
    egui::CollapsingHeader::new(RichText::new("Statistics").strong())
        .default_open(true)
        .show(ui, |ui| {
            // `st.stats` is the cached snapshot from `draw` (recomputed every
            // 250 ms), not a live query.
            let s = &st.stats;
            egui::Grid::new("stats_grid")
                .num_columns(2)
                .spacing([10.0, 2.0])
                .striped(true)
                .show(ui, |ui| {
                    heading_row(ui, "System");
                    kv(ui, "elapsed", fmt_time(b.time, si));
                    kv(ui, "seed", ph::seed_to_base62(view.seed));
                    kv(
                        ui,
                        "systems",
                        format!("{} × {} bodies", view.systems, view.body_count),
                    );
                    // Composition string is built by hand (not via a
                    // template) because the icons for each kind are
                    // chosen to be visually distinct in monospace font.
                    let mut comp = format!("{}★", s.stars);
                    if s.brown_dwarfs > 0 {
                        comp += &format!(" {}☾", s.brown_dwarfs);
                    }
                    if s.white_dwarfs > 0 {
                        comp += &format!(" {}✦", s.white_dwarfs);
                    }
                    if s.neutron_stars > 0 {
                        comp += &format!(" {}✹", s.neutron_stars);
                    }
                    if s.black_holes > 0 {
                        comp += &format!(" {}●", s.black_holes);
                    }
                    comp += &format!(" {} planets", s.planets);
                    if s.dwarf_planets > 0 {
                        comp += &format!(" {} dwarfs", s.dwarf_planets);
                    }
                    if let Some(m) = s.moons {
                        comp += &format!(" {} moons", m);
                    }
                    if s.comets > 0 {
                        comp += &format!(" {} comets", s.comets);
                    }
                    if s.interstellar > 0 {
                        comp += &format!(" {} ISO", s.interstellar);
                    }
                    if s.rogues > 0 {
                        comp += &format!(" {} rogues", s.rogues);
                    }
                    comp += &format!(" {} debris", s.rubble);
                    if s.belt_bodies > 0 {
                        comp += &format!(" ({} in belts)", s.belt_bodies);
                    }
                    if let Some(slug) = &st.loaded_preset_slug {
                        kv(ui, "loaded", slug.clone());
                    }
                    kv(ui, "bodies", format!("{} ({})", s.n, comp));
                    kv(ui, "total mass", fmt_mass(s.total_mass, si));
                    kv(
                        ui,
                        "phase mix",
                        format!(
                            "{} MS, {} giant, {} WD, {} NS, {} BH",
                            s.stars.saturating_sub(s.giants),
                            s.giants,
                            s.white_dwarfs,
                            s.neutron_stars,
                            s.black_holes
                        ),
                    );

                    if s.total_mass > 0.0 {
                        let pc = |x: f64| 100.0 * x / s.total_mass;
                        kv(
                            ui,
                            "mass split",
                            format!(
                                "{:.2} % stellar, {:.3} % planets, {:.4} % debris",
                                pc(s.mass_stellar),
                                pc(s.mass_planets),
                                pc(s.mass_debris)
                            ),
                        );
                    }

                    heading_row(ui, "Population");
                    if let Some((name, m)) = &s.primary {
                        kv(ui, "primary", format!("{} ({})", name, fmt_mass(*m, si)));
                    }
                    if let Some((name, m)) = &s.heaviest_planet {
                        kv(
                            ui,
                            "heaviest planet",
                            format!("{} ({})", name, fmt_mass(*m, si)),
                        );
                    }
                    if let Some((name, d)) = &s.farthest {
                        kv(
                            ui,
                            "farthest bound",
                            format!("{} ({})", name, fmt_len(*d, si)),
                        );
                    }
                    kv(ui, "on escape orbits", format!("{}", s.unbound));
                    if let Some(hz) = s.hz_planets {
                        kv(ui, "in habitable zone", format!("{}", hz));
                    }
                    if let Some(age) = s.oldest_star_gyr {
                        kv(ui, "oldest star", format!("{:.3} Gyr", age));
                    }
                    let eccs = &st.diag.eccs;
                    if !eccs.is_empty() {
                        let mean = eccs.iter().sum::<f64>() / eccs.len() as f64;
                        let max = eccs.iter().copied().fold(0.0_f64, f64::max);
                        kv(
                            ui,
                            "planet eccentricity",
                            format!("mean {:.3}, max {:.3} (n={})", mean, max, eccs.len()),
                        );
                    }

                    heading_row(ui, "Events");
                    kv(ui, "merges", fmt_count(b.stats.merges));
                    kv(ui, "impacts", fmt_count(b.stats.impacts));
                    kv(ui, "torn apart", fmt_count(b.stats.tidal));
                    kv(ui, "escaped", fmt_count(b.stats.escaped));

                    heading_row(ui, "Conservation");
                    // Green if drift < 1 %, orange if > 1 %, gray if no data.
                    // 1 % is a rough "this is fine for a toy" threshold; at
                    // 4th-order accuracy and rung-0 integration, well-behaved
                    // sims stay under 0.1 %, so 1 % is the point where
                    // something is likely wrong.
                    let drift_col = match s.drift {
                        Some(d) if d.abs() > 1.0 => theme().colors.warn(),
                        Some(_) => theme().colors.good(),
                        None => theme().colors.dim(),
                    };
                    kv_color(ui, "integrator drift", fmt_pct(s.drift), drift_col);
                    kv(ui, "heat / escapes", fmt_pct(s.dissipated));
                    kv(ui, "ang. momentum", format!("{:.4e}", s.ang_mom));
                    kv(ui, "kinetic energy", format!("{:.4e}", s.kinetic));
                    kv(ui, "barycentre offset", fmt_len(s.com_dist, si));
                    kv(ui, "barycentre speed", fmt_speed(s.com_speed, si));

                    heading_row(ui, "Solver");
                    // The solver names are hardcoded to match `physics.rs`'s
                    // thresholds (PAR_DIRECT_MIN, BH_THRESHOLD). If those
                    // constants change, this display lies. It's cheap enough
                    // not to bother with a single source of truth.
                    let solver = if b.len() < ph::PAR_DIRECT_MIN {
                        "direct pairs".to_string()
                    } else if b.len() < ph::BH_THRESHOLD {
                        "direct, parallel".to_string()
                    } else {
                        format!("Barnes-Hut θ={:.2}", ph::BH_THETA)
                    };
                    kv(ui, "gravity", solver);
                    let dt_s = ph::DT * ph::time_unit_s();
                    kv(ui, "base step", format!("{:.1} h", dt_s / 3600.0));
                    let integ = if b.stats.max_rung == 0 && b.len() < ph::YOSHIDA_MAX_N {
                        "Yoshida 4th order"
                    } else {
                        "block leapfrog"
                    };
                    kv(ui, "integrator", integ);
                    kv(
                        ui,
                        "fine ticks / step",
                        format!("{} (rung {})", b.stats.last_nsub, b.stats.max_rung),
                    );
                    kv(ui, "steps/s", format!("{:.0}", view.steps_per_sec));
                    kv(ui, "zoom", format!("{:.4} px/Gm", view.zoom));
                });
        });
}

fn section_view(ui: &mut egui::Ui, st: &mut UiState, view: &ViewInfo, act: &mut Actions) {
    egui::CollapsingHeader::new(RichText::new("View").strong())
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("−").on_hover_text("zoom out (-)").clicked() {
                    act.zoom_mul = Some(1.0 / 1.4);
                }
                if ui.button("+").on_hover_text("zoom in (+)").clicked() {
                    act.zoom_mul = Some(1.4);
                }
                if ui.button("Fit").on_hover_text("fit the system (F)").clicked() {
                    act.fit = true;
                }
                if ui.button("‹").on_hover_text("previous body (P)").clicked() {
                    act.cycle = -1;
                }
                if ui.button("›").on_hover_text("next body (N)").clicked() {
                    act.cycle = 1;
                }
            });
            let mut orbits = view.show_orbits;
            if ui.checkbox(&mut orbits, "orbit lines (O)").changed() {
                act.orbits = Some(orbits);
            }
            let mut follow = view.follow;
            if ui.checkbox(&mut follow, "follow selected (C)").changed() {
                act.follow = Some(follow);
            }
            ui.checkbox(&mut st.show_labels, "labels (L)");
            ui.checkbox(&mut st.units_si, "physical units (U)");
            let mut diag_vis = st.diag.visible;
            if ui.checkbox(&mut diag_vis, "diagnostics (G)").changed() {
                st.diag.visible = diag_vis;
                act.diag_visible = Some(diag_vis);
            }
            ui.toggle_value(&mut st.show_list, "Body list (B)");
            ui.label(
                RichText::new("Bodies are drawn at true size. Tiny ones get a minimum of a few pixels so they stay visible.")
                    .small()
                    .color(theme().colors.dim()),
            );
        });
}

fn section_advanced(ui: &mut egui::Ui, st: &mut UiState, view: &ViewInfo, act: &mut Actions) {
    egui::CollapsingHeader::new(RichText::new("Advanced").strong())
        .default_open(false)
        .show(ui, |ui| {
            let mut debris = view.show_debris_orbits;
            if ui.checkbox(&mut debris, "debris orbits (;)").changed() {
                act.debris_orbits = Some(debris);
            }
            let mut trails = st.show_trails;
            if ui.checkbox(&mut trails, "trails").changed() {
                st.show_trails = trails;
                act.trails = Some(trails);
            }
            let mut vsync = view.vsync;
            if ui
                .checkbox(&mut vsync, "vsync (V)")
                .on_hover_text("disabled = faster rendering, screen tearing")
                .changed()
            {
                act.vsync = Some(vsync);
            }
        });
}

fn section_generate(ui: &mut egui::Ui, st: &mut UiState, view: &ViewInfo, act: &mut Actions) {
    egui::CollapsingHeader::new(RichText::new("Generate").strong())
        .default_open(true)
        .show(ui, |ui| {
            // Random-mode is "preset == Random AND not currently showing a
            // library preset". Library presets set `loaded_preset_slug`,
            // which keeps the sliders hidden even though `preset` is
            // still `Random` under the hood.
            let random_selected =
                view.preset == crate::Preset::Random && st.loaded_preset_slug.is_none();

            let current_name = if let Some(slug) = &st.loaded_preset_slug {
                ph::load_presets()
                    .iter()
                    .find(|(s, _)| s == slug)
                    .map(|(_, p)| p.name.clone())
                    .unwrap_or_else(|| slug.clone())
            } else {
                match view.preset {
                    crate::Preset::SolarSystem => "Solar System".to_string(),
                    crate::Preset::Random => "Random".to_string(),
                }
            };

            enum Choice {
                Random,
                SolarSystem,
                Library(String),
            }
            let mut choice: Option<Choice> = None;

            ui.horizontal(|ui| {
                ui.label(RichText::new("preset").color(theme().colors.dim()));
                egui::ComboBox::from_id_salt("preset_picker")
                    .selected_text(current_name)
                    .width(200.0)
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(random_selected, "Random").clicked() {
                            choice = Some(Choice::Random);
                        }
                        for (slug, preset) in ph::load_presets() {
                            let selected = if slug == "solar_system" {
                                view.preset == crate::Preset::SolarSystem
                                    && st.loaded_preset_slug.is_none()
                            } else {
                                st.loaded_preset_slug.as_deref() == Some(slug.as_str())
                            };
                            if ui.selectable_label(selected, &preset.name).clicked() {
                                choice = Some(if slug == "solar_system" {
                                    Choice::SolarSystem
                                } else {
                                    Choice::Library(slug.clone())
                                });
                            }
                        }
                    });
            });
            match choice {
                Some(Choice::Random) => act.preset = Some(crate::Preset::Random),
                Some(Choice::SolarSystem) => act.preset = Some(crate::Preset::SolarSystem),
                Some(Choice::Library(slug)) => act.load_preset = Some(slug),
                None => {}
            }

            if random_selected {
                // Body count — the per-system target. With realistic
                // belts, this is a *ceiling*, not the actual spawn count:
                // a typical system has ~15 planets + a handful of belt
                // tracers, far below the slider value.
                ui.add(
                    egui::Slider::new(&mut st.pending_count, 10..=50_000)
                        .logarithmic(true)
                        .text("bodies"),
                );
                // Systems — Random only. Committed on drag-release so
                // dragging doesn't rebuild the sim every frame.
                let r = ui.add(
                    egui::Slider::new(&mut st.pending_systems, 1..=ph::MAX_SOLAR_SYSTEMS)
                        .logarithmic(true)
                        .text("systems"),
                );
                if r.drag_stopped() || (r.changed() && !r.dragged()) {
                    act.systems = Some(st.pending_systems);
                }
                // Warn if the requested combination exceeds MAX_BODIES.
                // `spawn_random_systems_impl` clamps `body_count`
                // per-system to fit, but the user should see that
                // happening rather than discovering it from the live
                // count.
                let n_sys = st.pending_systems;
                let n_bodies = st.pending_count;
                let requested = n_bodies.saturating_mul(n_sys);
                if requested > ph::MAX_BODIES {
                    let effective_per_system = ph::MAX_BODIES / n_sys.max(1);
                    ui.label(
                        RichText::new(format!(
                            "⚠ {n_sys} × {n_bodies} > {}: each system capped at {effective_per_system}",
                            ph::MAX_BODIES
                        ))
                        .small()
                        .color(theme().colors.warn()),
                    );
                }
                // Seed.
                ui.horizontal(|ui| {
                    ui.label("seed");
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut st.pending_seed_text)
                            .char_limit(ph::SEED_LEN)
                            .desired_width(150.0),
                    );
                    if r.changed() && ph::seed_from_base62(&st.pending_seed_text).is_none() {
                        // Hold the old value: `commit_seed_text` uses
                        // the fallback.
                        st.seed_text_invalid = true;
                    } else if r.changed() {
                        st.seed_text_invalid = false;
                    }
                    if st.seed_text_invalid {
                        r.on_hover_text(
                            RichText::new("invalid base62 seed")
                                .color(Color32::from_rgb(220, 60, 60)),
                        );
                    }
                    if ui
                        .button("Random")
                        .on_hover_text("roll a new random seed")
                        .clicked()
                    {
                        st.pending_seed_text = ph::seed_to_base62(random_seed());
                        st.seed_text_invalid = false;
                    }
                    if ui
                        .button("Copy")
                        .on_hover_text("copy the seed of the current system")
                        .clicked()
                    {
                        ui.ctx().copy_text(ph::seed_to_base62(view.seed));
                    }
                });
                // Status line. `body_count` is the per-system *target*;
                // the live count shows what actually spawned. They differ
                // because realistic belts are sparse — a typical 4.6-Gyr
                // G dwarf has ~15 planets and ~5 debris tracers, not the
                // thousands a full belt would hold.
                let live_per_system = view.total_bodies.checked_div(view.systems).unwrap_or(0);
                ui.label(
                    RichText::new(format!(
                        "{} system{} × {} target • {} live ({} per system)",
                        view.systems,
                        if view.systems == 1 { "" } else { "s" },
                        view.body_count,
                        view.total_bodies,
                        live_per_system,
                    ))
                    .small()
                    .color(theme().colors.dim()),
                );
                if ui.button("Generate").clicked() {
                    act.body_count = Some(st.pending_count);
                    act.seed = Some(commit_seed_text(&st.pending_seed_text, view.seed));
                }
            } else {
                // Named preset: no sliders, just info + reload.
                ui.label(
                    RichText::new(format!("{} bodies (fixed by preset)", view.total_bodies))
                        .small()
                        .color(theme().colors.dim()),
                );
            }
        });
}

/// Map an event kind to a text color. Matches the color-coding in the
/// 3D scene (merges are white-ish, impacts orange, tidal purple, escapes
/// gray, user edits green).
fn event_color(k: EventKind) -> Color32 {
    match k {
        EventKind::Merge => Color32::from_rgb(160, 200, 255),
        EventKind::Impact => Color32::from_rgb(255, 170, 90),
        EventKind::Tidal => Color32::from_rgb(220, 130, 255),
        EventKind::Escape => Color32::from_gray(150),
        EventKind::User => Color32::from_rgb(130, 220, 150),
        EventKind::Evolve => Color32::from_rgb(255, 220, 130),
        EventKind::Comet => Color32::from_rgb(150, 220, 255),
        EventKind::Interstellar => Color32::from_rgb(220, 170, 255),
        EventKind::Eject => Color32::from_rgb(255, 140, 90),
    }
}

/// Bit for one event kind in `UiState::event_mask`.
fn event_bit(k: EventKind) -> u16 {
    match k {
        EventKind::Merge => 1 << 0,
        EventKind::Impact => 1 << 1,
        EventKind::Tidal => 1 << 2,
        EventKind::Escape => 1 << 3,
        EventKind::User => 1 << 4,
        EventKind::Evolve => 1 << 5,
        EventKind::Comet => 1 << 6,
        EventKind::Interstellar => 1 << 7,
        EventKind::Eject => 1 << 8,
    }
}

const EVENT_KINDS: [(EventKind, &str); 9] = [
    (EventKind::Merge, "merge"),
    (EventKind::Impact, "impact"),
    (EventKind::Tidal, "tidal"),
    (EventKind::Escape, "escape"),
    (EventKind::Eject, "eject"),
    (EventKind::Evolve, "evolve"),
    (EventKind::Comet, "comet"),
    (EventKind::Interstellar, "interstellar"),
    (EventKind::User, "edit"),
];

fn section_events(ui: &mut egui::Ui, st: &mut UiState, b: &Bodies, act: &mut Actions) {
    egui::CollapsingHeader::new(RichText::new(format!("Events ({})", b.events.len())).strong())
        .id_salt("events_header")
        .default_open(true)
        .show(ui, |ui| {
            if b.events.is_empty() {
                ui.label(RichText::new("nothing has happened yet").color(theme().colors.dim()));
                return;
            }
            ui.horizontal_wrapped(|ui| {
                for (k, name) in EVENT_KINDS {
                    let bit = event_bit(k);
                    let on = st.event_mask & bit != 0;
                    let txt = RichText::new(name).small().color(if on {
                        event_color(k)
                    } else {
                        theme().colors.dim()
                    });
                    if ui.selectable_label(on, txt).clicked() {
                        st.event_mask ^= bit;
                    }
                }
            });
            let mask = st.event_mask;
            egui::ScrollArea::vertical()
                .max_height(220.0)
                .id_salt("events_scroll")
                .show(ui, |ui| {
                    for e in b
                        .events
                        .iter()
                        .rev()
                        .filter(|e| mask & event_bit(e.kind) != 0)
                        .take(60)
                    {
                        let resp = ui
                            .horizontal_wrapped(|ui| {
                                ui.label(
                                    RichText::new(fmt_time(e.time, st.units_si))
                                        .monospace()
                                        .small()
                                        .color(theme().colors.dim()),
                                );
                                ui.add(
                                    egui::Label::new(
                                        RichText::new(&e.text).small().color(event_color(e.kind)),
                                    )
                                    .sense(egui::Sense::click()),
                                )
                            })
                            .inner;
                        if resp
                            .on_hover_text("click to centre the view here")
                            .clicked()
                        {
                            act.center_on = Some([e.x, e.y]);
                        }
                    }
                });
        });
}

// ---------------------------------------------------------------------------
// Selected body
// ---------------------------------------------------------------------------

/// The floating window shown when a body is selected. Everything here is
/// read-only except the buttons at the bottom, which emit `Edit` actions.
fn selection_window(
    ctx: &egui::Context,
    st: &UiState,
    view: &ViewInfo,
    b: &Bodies,
    i: usize,
    act: &mut Actions,
) {
    let si = st.units_si;
    let mut open = true;
    // One O(n) `find_parent` per frame, reused for the title (moon vs
    // planet) and the Orbit section below.
    let parent = ph::find_parent(b, i);
    let label = b.name[i].clone().unwrap_or_else(|| format!("#{}", b.id[i]));
    let title = format!("{}  {}", label, ph::body_class_with_parent(b, i, parent));
    egui::Window::new(title)
        .id(egui::Id::new("selection"))
        .open(&mut open)
        // Anchored top-right with a 10-point margin, so it sits over the
        // 3D scene but doesn't collide with the dock.
        .anchor(Align2::RIGHT_TOP, [-10.0, 40.0])
        .resizable(false)
        .collapsible(true)
        .default_width(280.0)
        .show(ctx, |ui| {
            // Height is capped to "screen height minus 190 points", so the
            // window never overflows on small screens but always leaves
            // room for the physical/stellar/orbit/state sections.
            egui::ScrollArea::vertical()
                .max_height((ctx.screen_rect().height() - 190.0).max(120.0))
                .show(ui, |ui| {
                    egui::Grid::new("sel_grid")
                        .num_columns(2)
                        .spacing([10.0, 2.0])
                        .striped(true)
                        .show(ui, |ui| {
                            selection_rows(ui, si, b, i, parent);
                        });
                });
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                // Mass edits are *factors* (0.5× and 2×), not absolute
                // values. This matches the `Edit::MassMul(f64)` API and
                // makes repeated clicks compose (four halvings = /16).
                if ui.small_button("mass ÷2").clicked() {
                    act.edit = Some(Edit::MassMul(0.5));
                }
                if ui.small_button("mass ×2").clicked() {
                    act.edit = Some(Edit::MassMul(2.0));
                }
                // "circularize" is only meaningful for bodies with a parent
                // (planets and dwarf planets). Showing it for stars would
                // produce a no-op (no parent → `find_parent` returns None).
                if matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
                    && ui.small_button("circularize").clicked()
                {
                    act.edit = Some(Edit::Circularize);
                }
                let mut follow = view.follow;
                if ui.checkbox(&mut follow, "follow").changed() {
                    act.follow = Some(follow);
                }
            });
            ui.horizontal_wrapped(|ui| {
                if matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
                    && b.m[i] >= 0.1
                    && ui.small_button("add moon").clicked()
                {
                    act.edit = Some(Edit::SpawnMoon);
                }
                if matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
                    && b.m[i] >= 10.0
                    && ui.small_button("add ring").clicked()
                {
                    act.edit = Some(Edit::AddRing);
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("type").color(theme().colors.dim()));
                // Type conversion buttons. Clicking a type the body already
                // has is a no-op (the `&&` short-circuits), so the current
                // type isn't re-set (which would reset `origin` and fire a
                // spurious "changed type" event).
                for (k, name) in [
                    (BodyKind::Planet, "planet"),
                    (BodyKind::DwarfPlanet, "dwarf planet"),
                    (BodyKind::BrownDwarf, "brown dwarf"),
                    (BodyKind::Star, "star"),
                    (BodyKind::WhiteDwarf, "white dwarf"),
                    (BodyKind::NeutronStar, "neutron star"),
                    (BodyKind::BlackHole, "black hole"),
                ] {
                    if ui.selectable_label(b.kind[i] == k, name).clicked() && b.kind[i] != k {
                        act.edit = Some(Edit::SetKind(k));
                    }
                }
                if ui.small_button("delete").clicked() {
                    act.edit = Some(Edit::Delete);
                }
            });
            ui.label(
                RichText::new("Edits start a new energy baseline.")
                    .small()
                    .color(theme().colors.dim()),
            );
        });
    // If the user closed the window (the little X), clear the selection.
    // Otherwise closing the panel would leave the body selected with no
    // way to see its details.
    if !open {
        act.select = Some(None);
    }
}

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
fn selection_rows(ui: &mut egui::Ui, si: bool, b: &Bodies, i: usize, parent: Option<usize>) {
    let m = b.m[i];
    let r = b.r[i];
    let kind = b.kind[i];
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

    match kind {
        BodyKind::Star | BodyKind::BrownDwarf | BodyKind::WhiteDwarf | BodyKind::NeutronStar => {
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
        BodyKind::Planet | BodyKind::DwarfPlanet => {
            // Stellar flux at the body's current position, summed over all
            // luminous bodies in the system. A planet in a binary system
            // sees both stars, and this correctly accounts for that.
            let s = ph::stellar_flux(b, b.x[i], b.y[i], i);
            if s > 0.0 {
                heading_row(ui, "Climate");
                kv(ui, "stellar flux", format!("{:.2} S⊕", s));
                // Equilibrium temperature: T = 255 K * S^(1/4). That's the
                // blackbody temperature with no greenhouse — actual surface
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
                let (txt, col) = if s > 1.1 {
                    ("too hot", theme().colors.warn())
                } else if s >= 0.36 {
                    ("in habitable zone", theme().colors.good())
                } else {
                    ("too cold", theme().colors.accent())
                };
                kv_color(ui, "zone", txt, col);
            }
        }
        BodyKind::BlackHole => {}
    }

    if let Some(p) = parent {
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
    // Collision immunity: a short post-collision grace period during which
    // the body can't hit anything (so a fragment doesn't immediately
    // re-impact the surviving body). Showing it makes it clear why two
    // overlapping bodies don't immediately merge.
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
    // find_parent is O(n), so counting children is O(n²): small systems only.
    if b.len() <= 400 {
        let kids = (0..b.len())
            .filter(|&k| k != i && ph::find_parent(b, k) == Some(i))
            .count();
        if kids > 0 {
            kv(ui, "bodies orbiting it", format!("{}", kids));
        }
    }
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

/// Rebuild the body-list snapshot. Called every ~300 ms or after a sort.
///
/// Cost: O(n log n) for the sort, plus O(n) `find_parent` calls (each O(n))
/// for the distance/speed/period columns. The latter is why `body_class` is
/// gated on `n <= 3000` below — above that, the O(n²) `find_parent` cost
/// dominates the sort. The list rows still show distance/speed/period for
/// large n because those *do* need `find_parent` — there's no way around
/// it without a cache. So the list is effectively "O(n²) every 300 ms" for
/// large n, and `list_limit` (max 1000) bounds the *display* cost, not the
/// compute cost.
///
/// If you want this to scale, the fix is to make `find_parent` O(1) via a
/// precomputed Hill-sphere census, and use that here. Until then, don't
/// open the body list on a 10k-body sim.
fn refresh_rows(st: &mut UiState, b: &Bodies) {
    let n = b.len();
    let prim = ph::primary_index(b);
    let key = st.sort;

    // Cache `find_parent` once: sort and display share it, so belt bodies
    // get real values (orbit about the primary) instead of 0 / ∞.
    let parents: Vec<Option<usize>> = (0..n).map(|i| ph::find_parent(b, i)).collect();
    let orbital = |i: usize| -> (f64, f64, f64) {
        match parents[i].or_else(|| prim.filter(|&p| p != i)) {
            Some(p) => {
                let o = ph::orbit_info(b, i, p);
                (o.dist, o.speed_rel, o.period)
            }
            None => (0.0, 0.0, f64::INFINITY),
        }
    };

    let mut idx: Vec<usize> = (0..n).collect();
    let val = |i: usize| -> f64 {
        match key {
            SortKey::Id => b.id[i] as f64,
            SortKey::Mass => b.m[i],
            SortKey::Radius => b.r[i],
            SortKey::Dist => orbital(i).0,
            SortKey::Speed => orbital(i).1,
            SortKey::Period => orbital(i).2,
        }
    };
    idx.sort_by(|&a, &c| {
        val(a)
            .partial_cmp(&val(c))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if st.sort_desc {
        idx.reverse();
    }
    let q = st.list_query.trim().to_lowercase();
    if !q.is_empty() {
        idx.retain(|&i| {
            let label = b.name[i]
                .clone()
                .unwrap_or_else(|| format!("#{}", b.id[i]))
                .to_lowercase();
            label.contains(&q) || ph::class_name(b, i).to_lowercase().contains(&q)
        });
    }
    idx.truncate(st.list_limit);

    st.list_rows = idx
        .into_iter()
        .map(|i| {
            let (dist, speed, period) = orbital(i);
            let label = b.name[i].clone().unwrap_or_else(|| format!("#{}", b.id[i]));
            ListRow {
                id: b.id[i],
                label,
                class: if n <= 3000 {
                    ph::body_class_with_parent(b, i, parents[i]).into_owned()
                } else {
                    ph::class_name(b, i).into_owned()
                },
                m: b.m[i],
                r: b.r[i],
                dist,
                speed,
                period,
            }
        })
        .collect();
}

fn list_window(
    ctx: &egui::Context,
    st: &mut UiState,
    view: &ViewInfo,
    b: &Bodies,
    act: &mut Actions,
) {
    if st.list_time.elapsed() > Duration::from_millis(300) || st.list_rows.is_empty() {
        refresh_rows(st, b);
        st.list_time = Instant::now();
    }
    // Bodies that merged or escaped since the last frame leave the ticked set.
    st.multi.retain(|id| b.index_of_id(*id).is_some());
    let si = st.units_si;
    let mut open = st.show_list;
    egui::Window::new("Bodies")
        .open(&mut open)
        .default_pos([320.0, 60.0])
        .default_width(600.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{} bodies", b.len())).color(theme().colors.dim()));
                let r = ui.add(
                    egui::TextEdit::singleline(&mut st.list_query)
                        .hint_text("search name or class")
                        .desired_width(170.0),
                );
                if r.changed() {
                    st.list_time = Instant::now() - Duration::from_secs(5);
                }
                if !st.list_query.is_empty() && ui.small_button("✕").clicked() {
                    st.list_query.clear();
                    st.list_time = Instant::now() - Duration::from_secs(5);
                }
                ui.label("show");
                let mut lim = st.list_limit;
                egui::ComboBox::from_id_salt("list_limit")
                    .selected_text(lim.to_string())
                    .show_ui(ui, |ui| {
                        for v in [50usize, 100, 250, 500, 1000] {
                            ui.selectable_value(&mut lim, v, v.to_string());
                        }
                    });
                if lim != st.list_limit {
                    st.list_limit = lim;
                    st.list_time = Instant::now() - Duration::from_secs(5);
                }
            });
            ui.horizontal_wrapped(|ui| {
                if ui.small_button("tick all shown").clicked() {
                    for r in &st.list_rows {
                        st.multi.insert(r.id);
                    }
                }
                if !st.multi.is_empty() {
                    let ids: Vec<u64> = st.multi.iter().copied().collect();
                    ui.label(
                        RichText::new(format!("{} ticked", ids.len()))
                            .color(theme().colors.accent()),
                    );
                    if ui.small_button("mass ÷2").clicked() {
                        act.multi_edit = Some((ids.clone(), Edit::MassMul(0.5)));
                    }
                    if ui.small_button("mass ×2").clicked() {
                        act.multi_edit = Some((ids.clone(), Edit::MassMul(2.0)));
                    }
                    if ui.small_button("circularize").clicked() {
                        act.multi_edit = Some((ids.clone(), Edit::Circularize));
                    }
                    egui::ComboBox::from_id_salt("multi_kind")
                        .selected_text("set type…")
                        .show_ui(ui, |ui| {
                            for (k, name) in [
                                (BodyKind::Planet, "planet"),
                                (BodyKind::DwarfPlanet, "dwarf planet"),
                                (BodyKind::BrownDwarf, "brown dwarf"),
                                (BodyKind::Star, "star"),
                                (BodyKind::WhiteDwarf, "white dwarf"),
                                (BodyKind::NeutronStar, "neutron star"),
                                (BodyKind::BlackHole, "black hole"),
                            ] {
                                if ui.selectable_label(false, name).clicked() {
                                    act.multi_edit = Some((ids.clone(), Edit::SetKind(k)));
                                }
                            }
                        });
                    if ui.small_button("delete").clicked() {
                        act.multi_edit = Some((ids.clone(), Edit::Delete));
                        st.multi.clear();
                    }
                    if ui.small_button("clear ticks").clicked() {
                        st.multi.clear();
                    }
                }
            });
            let headers = [
                ("name", SortKey::Id),
                ("class", SortKey::Id),
                ("mass", SortKey::Mass),
                ("radius", SortKey::Radius),
                ("dist", SortKey::Dist),
                ("speed", SortKey::Speed),
                ("period", SortKey::Period),
            ];
            egui::ScrollArea::vertical()
                .max_height(420.0)
                .show(ui, |ui| {
                    egui::Grid::new("list_grid")
                        .num_columns(8)
                        .spacing([12.0, 3.0])
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label("");
                            for (k, (name, key)) in headers.iter().enumerate() {
                                let active = st.sort == *key && k != 1;
                                let arrow = if active {
                                    if st.sort_desc {
                                        " ▼"
                                    } else {
                                        " ▲"
                                    }
                                } else {
                                    ""
                                };
                                let txt = RichText::new(format!("{}{}", name, arrow))
                                    .strong()
                                    .color(if active {
                                        theme().colors.accent()
                                    } else {
                                        Color32::WHITE
                                    });
                                if k == 1 {
                                    ui.label(txt);
                                } else if ui
                                    .add(egui::Label::new(txt).sense(egui::Sense::click()))
                                    .clicked()
                                {
                                    if st.sort == *key {
                                        st.sort_desc = !st.sort_desc;
                                    } else {
                                        st.sort = *key;
                                        st.sort_desc = *key != SortKey::Id;
                                    }
                                    st.list_time = Instant::now() - Duration::from_secs(5);
                                }
                            }
                            ui.end_row();
                            for row in &st.list_rows {
                                let mut ticked = st.multi.contains(&row.id);
                                if ui.checkbox(&mut ticked, "").changed() {
                                    if ticked {
                                        st.multi.insert(row.id);
                                    } else {
                                        st.multi.remove(&row.id);
                                    }
                                }
                                let selected = view.selected == Some(row.id);
                                if ui.selectable_label(selected, &row.label).clicked() {
                                    act.select = Some(Some(row.id));
                                }
                                ui.label(&row.class);
                                ui.label(RichText::new(fmt_mass(row.m, si)).monospace());
                                ui.label(RichText::new(fmt_len(row.r, si)).monospace());
                                ui.label(RichText::new(fmt_len(row.dist, si)).monospace());
                                ui.label(RichText::new(fmt_speed(row.speed, si)).monospace());
                                ui.label(RichText::new(fmt_time(row.period, si)).monospace());
                                ui.end_row();
                            }
                        });
                });
            ui.label(
                RichText::new(
                    "Click a name to select it. Tick boxes to edit many bodies at once. \
                     Distance, speed and period are relative to the body's parent, \
                     or to the primary star for bodies without one (belts, debris).",
                )
                .small()
                .color(theme().colors.dim()),
            );
        });
    st.show_list = open;
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

fn help_window(ctx: &egui::Context, st: &mut UiState) {
    let mut open = st.show_help;
    egui::Window::new("Help")
        .open(&mut open)
        .default_pos([340.0, 80.0])
        .resizable(false)
        .show(ctx, |ui| {
            egui::Grid::new("help_keys").num_columns(2).spacing([14.0, 2.0]).show(ui, |ui| {
                for (k, d) in [
                    ("Space", "pause / resume"),
                    (".", "advance one step"),
                    ("R", "new system (next seed)"),
                    ("drag", "pan the view"),
                    ("wheel, + / −", "zoom toward the cursor"),
                    ("W A S D, arrows", "pan"),
                    ("click", "select a body (click empty space to deselect)"),
                    ("N / P", "select next / previous planet by distance"),
                    ("C", "follow the selected body"),
                    ("F, 0", "fit the whole system"),
                    ("O", "orbit lines"),
                    ("L", "labels"),
                    ("U", "physical / simulation units"),
                    ("B", "body list"),
                    ("G", "diagnostics"),
                    ("V", "vsync"),
                    ("`", "show / hide the control panel"),
                    ("Delete", "delete the selected body"),
                    ("Esc", "deselect"),
                    ("? or H", "this help"),
                ] {
                    ui.label(RichText::new(k).monospace().color(theme().colors.accent()));
                    ui.label(d);
                    ui.end_row();
                }
            });
            ui.separator();
            ui.label(RichText::new("About the physics").strong());
            ui.label(
                RichText::new(
                    "Bodies collide at their true physical radius, checked along the whole path of each step, so fast bodies cannot pass through each other. \
                     Slow impacts merge. Fast impacts throw out debris. A planet inside its Roche limit breaks apart. \
                     The heaviest body, or the star or black hole, always survives a merge. \
                     Energy lost to heat, break-ups and escapes is tracked separately from integrator drift.",
                )
                .small(),
            );
        });
    st.show_help = open;
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// Allocate a plot rectangle of the full available width. The painter is
/// clipped to the rectangle so drawing cannot spill into other widgets.
fn plot_rect(ui: &mut egui::Ui, height: f32) -> (egui::Rect, egui::Painter, egui::Response) {
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, egui::Rounding::same(2.0), Color32::from_gray(20));
    (rect, painter, resp)
}

fn draw_axes(rect: egui::Rect, painter: &egui::Painter) {
    let s = Stroke::new(1.0_f32, Color32::from_gray(60));
    painter.line_segment([rect.left_bottom(), rect.right_bottom()], s);
    painter.line_segment([rect.left_bottom(), rect.left_top()], s);
}

/// Y range with 10 % padding. A flat series gets a range of at least 1.
fn padded_range(lo: f64, hi: f64) -> (f64, f64) {
    let pad = (hi - lo) * 0.1;
    let pad = if pad > 0.0 {
        pad
    } else {
        (hi.abs() * 0.05).max(1.0)
    };
    (lo - pad, hi + pad)
}

fn fmt_int(v: f64) -> String {
    format!("{}", v.round() as i64)
}

fn fmt_pct_value(v: f64) -> String {
    fmt_pct(Some(v))
}

/// Look of one time-series plot.
struct PlotSpec<'a> {
    title: &'a str,
    empty: &'a str,
    color: Color32,
    /// Use the warning colour when the latest value is larger than this.
    warn_at: f64,
    /// Centre the y axis on zero (for drift values).
    symmetric: bool,
    fmt: fn(f64) -> String,
}

/// Draw one line plot of `get` over the sample history. Hover shows the
/// time and value of the nearest sample.
fn plot_series(
    ui: &mut egui::Ui,
    diag: &DiagHistory,
    si: bool,
    spec: &PlotSpec,
    get: impl Fn(&DiagSample) -> Option<f64>,
) {
    let (rect, painter, resp) = plot_rect(ui, 80.0);
    draw_axes(rect, &painter);

    let vals: Vec<Option<f64>> = diag
        .samples
        .iter()
        .map(|s| get(s).filter(|v| v.is_finite()))
        .collect();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in vals.iter().flatten() {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    if !lo.is_finite() {
        let msg = if diag.samples.is_empty() {
            "collecting…"
        } else {
            spec.empty
        };
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            msg,
            FontId::monospace(10.0),
            theme().colors.dim(),
        );
        return;
    }

    let (y0, y1) = if spec.symmetric {
        let m = lo.abs().max(hi.abs()).max(1e-12) * 1.1;
        (-m, m)
    } else {
        padded_range(lo, hi)
    };
    let t0 = diag.samples.front().map(|s| s.t).unwrap_or(0.0);
    let t1 = diag.samples.back().map(|s| s.t).unwrap_or(0.0);
    let t_span = (t1 - t0).max(1e-9);
    let map = |t: f64, v: f64| -> egui::Pos2 {
        let fx = ((t - t0) / t_span).clamp(0.0, 1.0) as f32;
        let fy = ((v - y0) / (y1 - y0)).clamp(0.0, 1.0) as f32;
        egui::pos2(
            rect.left() + fx * rect.width(),
            rect.bottom() - fy * rect.height(),
        )
    };

    if spec.symmetric {
        let zy = rect.center().y;
        painter.line_segment(
            [egui::pos2(rect.left(), zy), egui::pos2(rect.right(), zy)],
            Stroke::new(1.0_f32, Color32::from_gray(50)),
        );
    }

    let current = vals.iter().rev().flatten().next().copied().unwrap_or(0.0);
    let line_col = if current.abs() > spec.warn_at {
        theme().colors.warn()
    } else {
        spec.color
    };

    let mut prev: Option<egui::Pos2> = None;
    for (s, v) in diag.samples.iter().zip(vals.iter()) {
        match v {
            Some(v) => {
                let p = map(s.t, *v);
                if let Some(q) = prev {
                    painter.line_segment([q, p], Stroke::new(1.0_f32, line_col));
                }
                prev = Some(p);
            }
            // A gap (value not available) breaks the line.
            None => prev = None,
        }
    }
    if let Some(p) = prev {
        painter.circle_filled(p, 2.0, line_col);
    }

    painter.text(
        rect.left_top() + egui::vec2(4.0, 2.0),
        Align2::LEFT_TOP,
        format!("{}  {}", spec.title, (spec.fmt)(current)),
        FontId::monospace(10.0),
        theme().colors.dim(),
    );
    painter.text(
        rect.left_bottom() + egui::vec2(2.0, -2.0),
        Align2::LEFT_BOTTOM,
        fmt_time(t0, si),
        FontId::monospace(9.0),
        theme().colors.dim(),
    );
    painter.text(
        rect.right_bottom() + egui::vec2(-2.0, -2.0),
        Align2::RIGHT_BOTTOM,
        fmt_time(t1, si),
        FontId::monospace(9.0),
        theme().colors.dim(),
    );

    if let Some(pos) = resp.hover_pos() {
        let fx = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0) as f64;
        let th = t0 + fx * t_span;
        let nearest = diag.samples.iter().enumerate().min_by(|p, q| {
            (p.1.t - th)
                .abs()
                .partial_cmp(&(q.1.t - th).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some((k, s)) = nearest {
            if let Some(v) = vals[k] {
                let p = map(s.t, v);
                painter.line_segment(
                    [egui::pos2(p.x, rect.top()), egui::pos2(p.x, rect.bottom())],
                    Stroke::new(1.0_f32, Color32::from_gray(90)),
                );
                painter.circle_filled(p, 3.0, spec.color);
                painter.text(
                    rect.right_top() + egui::vec2(-4.0, 2.0),
                    Align2::RIGHT_TOP,
                    format!("{}  {}", fmt_time(s.t, si), (spec.fmt)(v)),
                    FontId::monospace(10.0),
                    Color32::WHITE,
                );
            }
        }
    }
}

fn plot_ecc(ui: &mut egui::Ui, diag: &DiagHistory) {
    let (rect, painter, resp) = plot_rect(ui, 80.0);
    draw_axes(rect, &painter);

    if diag.eccs.is_empty() {
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "no planets",
            FontId::monospace(10.0),
            theme().colors.dim(),
        );
        return;
    }

    let max_count = diag.ecc_bins.iter().copied().max().unwrap_or(1).max(1);
    let n = diag.ecc_bin_count;
    let w = rect.width() / n as f32;
    for (k, &cnt) in diag.ecc_bins.iter().enumerate() {
        if cnt == 0 {
            continue;
        }
        let x0 = rect.left() + k as f32 * w;
        let x1 = x0 + w * 0.9;
        let h = (cnt as f32 / max_count as f32) * rect.height();
        painter.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(x0, rect.bottom() - h),
                egui::pos2(x1, rect.bottom()),
            ),
            egui::Rounding::ZERO,
            theme().colors.accent(),
        );
    }

    painter.text(
        rect.left_top() + egui::vec2(4.0, 2.0),
        Align2::LEFT_TOP,
        format!("eccentricity (n={}, peak={})", diag.eccs.len(), max_count),
        FontId::monospace(10.0),
        theme().colors.dim(),
    );
    painter.text(
        rect.left_bottom() + egui::vec2(2.0, -2.0),
        Align2::LEFT_BOTTOM,
        "0.0",
        FontId::monospace(9.0),
        theme().colors.dim(),
    );
    painter.text(
        rect.right_bottom() + egui::vec2(-2.0, -2.0),
        Align2::RIGHT_BOTTOM,
        "1.0",
        FontId::monospace(9.0),
        theme().colors.dim(),
    );
    if let Some(pos) = resp.hover_pos() {
        let k = (((pos.x - rect.left()) / w) as usize).min(n - 1);
        let lo = k as f64 * diag.ecc_bin_width;
        painter.text(
            rect.right_top() + egui::vec2(-4.0, 2.0),
            Align2::RIGHT_TOP,
            format!(
                "e {:.2}–{:.2}  {} planets",
                lo,
                lo + diag.ecc_bin_width,
                diag.ecc_bins[k]
            ),
            FontId::monospace(10.0),
            Color32::WHITE,
        );
    }
}

fn diag_window(
    ctx: &egui::Context,
    st: &mut UiState,
    view: &ViewInfo,
    b: &Bodies,
    _act: &mut Actions,
) {
    let si = st.units_si;
    let mut open = st.diag.visible;
    egui::Window::new("Diagnostics")
        .open(&mut open)
        .default_pos([640.0, 60.0])
        .default_width(380.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui
                    .button("clear")
                    .on_hover_text("forget the history and the angular-momentum baseline")
                    .clicked()
                {
                    st.diag.clear();
                }
                if ui
                    .button("copy CSV")
                    .on_hover_text("copy the sample history to the clipboard")
                    .clicked()
                {
                    ui.ctx().copy_text(st.diag.to_csv());
                }
                ui.label("history");
                let mut maxs = st.diag.max_samples;
                egui::ComboBox::from_id_salt("diag_history")
                    .selected_text(format!("{} min", maxs / 60))
                    .show_ui(ui, |ui| {
                        for (v, name) in [(300usize, "5 min"), (900, "15 min"), (1800, "30 min")] {
                            ui.selectable_value(&mut maxs, v, name);
                        }
                    });
                st.diag.max_samples = maxs;
            });

            let last = st.diag.samples.back();
            egui::Grid::new("diag_summary")
                .num_columns(2)
                .spacing([14.0, 2.0])
                .show(ui, |ui| {
                    kv(ui, "bodies", format!("{}", b.len()));
                    kv(ui, "energy drift", fmt_pct(last.and_then(|s| s.drift)));
                    kv(
                        ui,
                        "energy dissipated",
                        fmt_pct(last.and_then(|s| s.dissipated)),
                    );
                    kv(
                        ui,
                        "ang. momentum change",
                        fmt_pct(last.and_then(|s| s.l_drift)),
                    );
                    kv(ui, "steps / s", format!("{:.0}", view.steps_per_sec));
                    kv(ui, "fine steps", format!("{}", b.stats.last_nsub));
                    kv(ui, "events logged", format!("{}", b.events.len()));
                });
            if b.len() > ph::ENERGY_MAX_N {
                ui.label(
                    RichText::new(format!(
                        "Energy values are off above {} bodies (too slow to compute).",
                        ph::ENERGY_MAX_N
                    ))
                    .small()
                    .color(theme().colors.warn()),
                );
            }
            ui.separator();

            egui::ScrollArea::vertical()
                .max_height((ctx.screen_rect().height() - 220.0).max(160.0))
                .show(ui, |ui| {
                    let d = &st.diag;
                    plot_series(
                        ui,
                        d,
                        si,
                        &PlotSpec {
                            title: "bodies",
                            empty: "no data",
                            color: theme().colors.good(),
                            warn_at: f64::INFINITY,
                            symmetric: false,
                            fmt: fmt_int,
                        },
                        |s| Some(s.n as f64),
                    );
                    ui.add_space(4.0);
                    plot_series(
                        ui,
                        d,
                        si,
                        &PlotSpec {
                            title: "energy drift",
                            empty: "n/a (system too large)",
                            color: theme().colors.accent(),
                            warn_at: 1.0,
                            symmetric: true,
                            fmt: fmt_pct_value,
                        },
                        |s| s.drift,
                    );
                    ui.add_space(4.0);
                    plot_series(
                        ui,
                        d,
                        si,
                        &PlotSpec {
                            title: "energy dissipated",
                            empty: "n/a (system too large)",
                            color: Color32::from_rgb(255, 170, 90),
                            warn_at: f64::INFINITY,
                            symmetric: false,
                            fmt: fmt_pct_value,
                        },
                        |s| s.dissipated,
                    );
                    ui.add_space(4.0);
                    plot_series(
                        ui,
                        d,
                        si,
                        &PlotSpec {
                            title: "angular momentum change",
                            empty: "no baseline",
                            color: Color32::from_rgb(220, 130, 255),
                            warn_at: 1.0,
                            symmetric: true,
                            fmt: fmt_pct_value,
                        },
                        |s| s.l_drift,
                    );
                    ui.add_space(4.0);
                    plot_series(
                        ui,
                        d,
                        si,
                        &PlotSpec {
                            title: "fine steps (close encounters)",
                            empty: "no data",
                            color: Color32::from_rgb(150, 220, 255),
                            warn_at: f64::INFINITY,
                            symmetric: false,
                            fmt: fmt_int,
                        },
                        |s| Some(s.nsub as f64),
                    );
                    ui.add_space(4.0);
                    plot_ecc(ui, d);
                });
        });
    st.diag.visible = open;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_range_pads_both_sides() {
        let (y0, y1) = padded_range(100.0, 150.0);
        assert!(y0 < 100.0, "y0={y0} should pad below min 100");
        assert!(y1 > 150.0, "y1={y1} should pad above max 150");
        let (y0, y1) = padded_range(200.0, 200.0);
        assert!(y1 - y0 >= 1.0, "flat range too small: {y0}..{y1}");
    }

    #[test]
    fn random_seed_fits_the_seed_space() {
        let s = random_seed();
        assert!(s < ph::SEED_SPACE);
        assert_eq!(ph::seed_to_base62(s).len(), ph::SEED_LEN);
    }

    #[test]
    fn event_bits_are_distinct() {
        let mut mask = 0u16;
        for (k, _) in EVENT_KINDS {
            let bit = event_bit(k);
            assert_eq!(mask & bit, 0);
            mask |= bit;
        }
        assert_eq!(mask, 0x1FF);
    }
}
