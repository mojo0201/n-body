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

// Mirrors `ph::ENERGY_MAX_N` — `total_energy` returns `None` above this, so
// the energy-drift chart has no data to show. Re-declared here because the
// constant in `physics.rs` is `pub` but we want the intent ("this is the
// threshold above which the diagnostics panel shows 'no data'") to live in
// the UI file too.
const ENERGY_MAX_N: usize = 1500;

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
    // sim time unit is moving at ~0.23 km/s.)
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
                if o.is_belt() || o.is_rubble() {
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
    s.drift = ph::energy_drift_pct(b);
    s.dissipated = ph::energy_dissipated_pct(b);
    s.ang_mom = ph::total_angular_momentum(b);
    s
}

/// One diagnostic sample.
pub struct DiagSample {
    pub t: f64,
    pub n: usize,
    pub drift: Option<f64>,
}

/// Rolling history for the three diagnostic plots: N(t), energy drift, and
/// eccentricity histogram.
///
/// `samples` is capped at `max_samples = 300`, which is 5 minutes of history
/// at the 1 Hz sampling rate. `eccs` is recomputed fresh each sample (it's a
/// snapshot of the *current* eccentricity distribution, not a history).
pub struct DiagHistory {
    pub samples: std::collections::VecDeque<DiagSample>,
    pub max_samples: usize,
    pub last_sample: Instant,
    pub eccs: Vec<f64>,
    pub ecc_bins: Vec<usize>,
    pub ecc_bin_width: f64,
    pub ecc_bin_count: usize,
    pub visible: bool,
}

impl DiagHistory {
    pub fn new() -> Self {
        Self {
            samples: std::collections::VecDeque::new(),
            max_samples: 300,
            // `now - 10s` forces the first sample to fire on the first
            // frame, so the plots aren't blank while the user wonders if
            // they're broken.
            last_sample: Instant::now() - Duration::from_secs(10),
            eccs: Vec::new(),
            ecc_bins: vec![0; 20],
            ecc_bin_width: 0.05,
            ecc_bin_count: 20,
            visible: true,
        }
    }

    /// Called from the render thread at most once per second while holding the lock.
    ///
    /// Cost note: the eccentricity pass calls `find_parent` and `orbit_info`
    /// for every planet. For n = 5000, that's 5000 × O(n) = 2.5×10⁷ ops.
    /// That's why this is throttled to 1 Hz and `eccs` is capped at 500.
    /// If you want a live eccentricity histogram at large n, this needs
    /// to become O(n) (a coarse-grained Hill-sphere census).
    pub fn maybe_sample(&mut self, b: &Bodies) {
        if self.last_sample.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.last_sample = Instant::now();

        let n = b.len();
        let drift = if n <= ENERGY_MAX_N {
            ph::energy_drift_pct(b)
        } else {
            None
        };
        self.samples.push_back(DiagSample {
            t: b.time,
            n,
            drift,
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
                // `bound` filters out escape orbits: a hyperbolic body has
                // e > 1, which has no meaningful bin in the histogram. If
                // the sample is mostly escaping bodies, the histogram will
                // be biased toward the bound ones — that's correct, the
                // plot is titled "eccentricity of bound planets".
                if o.bound && o.ecc.is_finite() {
                    self.eccs.push(o.ecc);
                }
            }
        }

        // Rebuild the bins from scratch. `bin = min(e / width, count-1)`
        // puts e=1.0 into the last bin rather than overflowing.
        self.ecc_bins.fill(0);
        for &e in &self.eccs {
            let bin = ((e / self.ecc_bin_width) as usize).min(self.ecc_bin_count - 1);
            self.ecc_bins[bin] += 1;
        }
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
    /// The preset library window is open.
    pub preset_library_open: bool,
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
    pub list_rows: Vec<ListRow>,
    list_time: Instant,
    pub stats: StatsSnapshot,
    stats_time: Instant,
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
            preset_library_open: false,
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
    /// always 9; for Random it's the user's `pending_count` at spawn time.
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
    /// Stellar aging rate in Gyr per simulation time unit (0 = off).
    pub evolution_rate: f64,
    pub sn_kicks: bool,
    pub mass_transfer: bool,
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
    /// Spawn one interstellar object inbound from `0.9 × CULL_DIST`.
    /// Not part of any automatic spawn path, so it never affects RNG
    /// state for other tests.
    pub spawn_iso: bool,
    /// Spawn one rogue planet inbound from `0.9 × CULL_DIST` (gas-giant
    /// mass). Exempt from the `CULL_DIST` cull, so it persists through
    /// the outer system until `HARD_CULL_DIST`.
    pub spawn_rogue: bool,
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
    pub evolution_rate: Option<f64>,
    pub sn_kicks: Option<bool>,
    pub mass_transfer: Option<bool>,
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
    preset_window(ctx, ui_state, &mut act);
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
                    ui.label(RichText::new("? help").color(theme().colors.dim()));
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
                    section_evolution(ui, st, view, act);
                    section_view(ui, st, view, act);
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
                if ui
                    .button("Spawn ISO")
                    .on_hover_text("spawn an interstellar object inbound (I)")
                    .clicked()
                {
                    act.spawn_iso = true;
                }
                if ui
                    .button("Spawn rogue")
                    .on_hover_text("spawn a gas-giant rogue planet inbound (Shift+I)")
                    .clicked()
                {
                    act.spawn_rogue = true;
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
                        "evolution",
                        if view.evolution_rate == 0.0 {
                            "off".to_string()
                        } else {
                            format!("10^{:.1} Gyr/tu", view.evolution_rate.log10())
                        },
                    );
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

/// Stellar evolution controls: aging rate, supernova kicks, mass transfer.
fn section_evolution(ui: &mut egui::Ui, _st: &mut UiState, view: &ViewInfo, act: &mut Actions) {
    egui::CollapsingHeader::new(RichText::new("Evolution").strong())
        .default_open(false)
        .show(ui, |ui| {
            let off = view.evolution_rate <= 0.0;
            // Logarithmic slider: 10^-3 to 10^3 Gyr per time unit.
            let mut slider_val = if off { 0.0 } else { view.evolution_rate.log10() };
            let changed = ui
                .add(
                    egui::Slider::new(&mut slider_val, -3.0..=3.0)
                        .text("age rate (Gyr / time unit)")
                        .custom_formatter(|v, _| format!("10^{:.1}", v)),
                )
                .changed();
            let mut disable = off;
            if ui.checkbox(&mut disable, "off").changed() {
                act.evolution_rate = Some(if disable { 0.0 } else { 1.0 });
            } else if changed {
                act.evolution_rate = Some(10f64.powf(slider_val));
            }
            ui.label(
                RichText::new(
                    "A 1 M☉ star lives ~11.5 Gyr. At 1 Gyr/tu, evolution takes ~11 tu (~1.5 yr sim time at 1×).",
                )
                .small()
                .color(theme().colors.dim()),
            );
            let mut kicks = view.sn_kicks;
            if ui.checkbox(&mut kicks, "supernova natal kicks").changed() {
                act.sn_kicks = Some(kicks);
            }
            let mut mt = view.mass_transfer;
            if ui.checkbox(&mut mt, "mass transfer in binaries").changed() {
                act.mass_transfer = Some(mt);
            }
        });
}

fn section_view(ui: &mut egui::Ui, st: &mut UiState, view: &ViewInfo, act: &mut Actions) {
    egui::CollapsingHeader::new(RichText::new("View").strong())
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                // Zoom buttons: ×1/1.4 and ×1.4 match the mouse-wheel factor
                // (`exp(0.15 * dy)` is ~1.16 per notch; 1.4 is the "big step"
                // equivalent of ~3 notches).
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
            // Each toggle mirrors a `SimState` field. The checkbox reads
            // from `view` (the committed value) and writes to `act` (the
            // requested value); the render loop then applies it. This is
            // why dragging a checkbox doesn't immediately change the sim —
            // it takes one frame.
            let mut orbits = view.show_orbits;
            if ui.checkbox(&mut orbits, "orbit lines (O)").changed() {
                act.orbits = Some(orbits);
            }
            let mut debris = view.show_debris_orbits;
            if ui.checkbox(&mut debris, "debris orbits (;)").changed() {
                act.debris_orbits = Some(debris);
            }
            // Trails are the exception: `show_trails` lives in `UiState`
            // (UI-only) and is toggled directly, not via `act`. The sim
            // doesn't care.
            let mut trails = st.show_trails;
            if ui.checkbox(&mut trails, "trails").changed() {
                st.show_trails = trails;
                act.trails = Some(trails);
            }
            let mut diag_vis = st.diag.visible;
            if ui.checkbox(&mut diag_vis, "diagnostics (G)").changed() {
                st.diag.visible = diag_vis;
                act.diag_visible = Some(diag_vis);
            }
            // NOTE: the line above is a deliberate place to flag — `diag_vis`
            // is a direct field, so the action `diag_visible` is redundant
            // here. The current code does write `act.diag_visible`, which
            // `apply_actions` then writes back to `ui_state.diag.visible`.
            // No bug, just a round-trip that could be a one-liner.
            ui.checkbox(&mut st.show_labels, "labels (L)");
            ui.checkbox(&mut st.units_si, "physical units (U)");
            let mut vsync = view.vsync;
            if ui
                .checkbox(&mut vsync, "vsync (V)")
                .on_hover_text("disabled = faster rendering, screen tearing")
                .changed()
            {
                act.vsync = Some(vsync);
            }
            let mut follow = view.follow;
            if ui.checkbox(&mut follow, "follow selected (C)").changed() {
                act.follow = Some(follow);
            }
            ui.horizontal(|ui| {
                ui.toggle_value(&mut st.show_list, "Body list (B)");
                ui.toggle_value(&mut st.show_help, "Help (?)");
            });
            ui.label(
                RichText::new("Bodies are drawn at true size. Tiny ones get a minimum of a few pixels so they stay visible.")
                    .small()
                    .color(theme().colors.dim()),
            );
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
                            .char_limit(6)
                            .desired_width(60.0),
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
                    if ui.button("Random").clicked() {
                        let d = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default();
                        let mut x = (d.subsec_nanos() as u64)
                            ^ d.as_secs().wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            ^ (view.seed as u64).wrapping_mul(2_654_435_761);
                        st.pending_seed_text = (0..6)
                            .map(|_| {
                                x = x
                                    .wrapping_mul(6_364_136_223_846_793_005)
                                    .wrapping_add(1_442_695_040_888_963_407);
                                ph::SEED_ALPHABET[((x >> 33) % 62) as usize] as char
                            })
                            .collect();
                        st.seed_text_invalid = false;
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
                if ui
                    .button("Reload (R)")
                    .on_hover_text("reload this preset with fresh planet phases")
                    .clicked()
                {
                    act.reset = true;
                }
            }

            if ui.small_button("Browse presets…").clicked() {
                st.preset_library_open = true;
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

/// Event log: newest-first, capped at 60 entries. Clicking an entry sets
/// `act.center_on`, which the render loop turns into a camera snap.
fn section_events(ui: &mut egui::Ui, st: &UiState, b: &Bodies, act: &mut Actions) {
    // `id_salt` is needed because two collapsing headers with the same
    // title ("Events") would share state otherwise. `id_salt` gives them
    // distinct egui ids.
    egui::CollapsingHeader::new(RichText::new(format!("Events ({})", b.events.len())).strong())
        .id_salt("events_header")
        .default_open(true)
        .show(ui, |ui| {
            if b.events.is_empty() {
                ui.label(RichText::new("nothing has happened yet").color(theme().colors.dim()));
                return;
            }
            egui::ScrollArea::vertical()
                .max_height(220.0)
                .id_salt("events_scroll")
                .show(ui, |ui| {
                    // `.rev()` shows newest first; `.take(60)` caps the
                    // cost so a system that has been running for hours
                    // doesn't render 200 event rows every frame.
                    for e in b.events.iter().rev().take(60) {
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
    // Refresh cadence: 300 ms is a compromise between "the list is live"
    // and "we don't rebuild the O(n²) snapshot every frame." The user
    // can force a refresh by clicking a sort header.
    if st.list_time.elapsed() > Duration::from_millis(300) || st.list_rows.is_empty() {
        refresh_rows(st, b);
        st.list_time = Instant::now();
    }
    let si = st.units_si;
    let mut open = st.show_list;
    egui::Window::new("Bodies")
        .open(&mut open)
        .default_pos([320.0, 60.0])
        .default_width(560.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{} bodies", b.len())).color(theme().colors.dim()));
                ui.label("show");
                // The limit is a power-of-2-ish sequence: 50, 100, 250, 500,
                // 1000. Not a log slider because the exact values matter to
                // the user (they may want exactly 100 rows).
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
                    // Force a refresh on the next frame.
                    st.list_time = Instant::now() - Duration::from_secs(5);
                }
            });
            // Header row: (label, sort key). The "class" column is not
            // sortable; the `k == 1` branch below draws it as a plain
            // label without a click handler.
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
                        .num_columns(7)
                        .spacing([12.0, 3.0])
                        .striped(true)
                        .show(ui, |ui| {
                            for (k, (name, key)) in headers.iter().enumerate() {
                                // The `class` header has `k == 1`; it never
                                // shows an arrow because it's not sortable
                                // (well — it *is*, see the note above, but
                                // the arrow is suppressed to hide that fact).
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
                                // Headers are made clickable via `Sense::click()`
                                // on a plain label. This is more flexible than
                                // `selectable_label` because we want to show
                                // the sort arrow only on the active column.
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
                                        // New column starts descending, except
                                        // Id where ascending (smallest id
                                        // first) is the natural order.
                                        st.sort_desc = *key != SortKey::Id;
                                    }
                                    // Force a refresh next frame.
                                    st.list_time = Instant::now() - Duration::from_secs(5);
                                }
                            }
                            ui.end_row();
                            for row in &st.list_rows {
                                let selected = view.selected == Some(row.id);
                                // `selectable_label` shows a highlight if
                                // selected; clicking emits an action.
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
                    "Distance, speed and period are relative to the body's parent, \
                     or to the primary star for bodies without one (belts, debris).",
                )
                .small()
                .color(theme().colors.dim()),
            );
        });
    // Persist the open/close state through the window's X button.
    st.show_list = open;
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

/// Static keyboard/mouse reference. The key labels here are *duplicated* in
/// `main.rs`'s `WindowEvent::KeyboardInput` handler — if you add a shortcut,
/// add it in both places. (There's no test that catches a mismatch.)
/// The preset library: a grid of cards. A click asks `main` to load it.
fn preset_window(ctx: &egui::Context, st: &mut UiState, act: &mut Actions) {
    egui::Window::new("Preset library")
        .open(&mut st.preset_library_open)
        .default_width(600.0)
        .default_pos([280.0, 80.0])
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                egui::Grid::new("preset_cards")
                    .num_columns(2)
                    .spacing([10.0, 10.0])
                    .show(ui, |ui| {
                        let lib = ph::load_presets();
                        for (idx, (slug, preset)) in lib.iter().enumerate() {
                            let selected = st.loaded_preset_slug.as_deref() == Some(slug.as_str());
                            let fill = if selected {
                                theme().colors.accent().gamma_multiply(0.15)
                            } else {
                                egui::Color32::TRANSPARENT
                            };
                            let resp = ui.add(
                                egui::Button::new(preset_card_text(preset))
                                    .fill(fill)
                                    .min_size(egui::vec2(270.0, 110.0)),
                            );
                            if resp.clicked() {
                                if slug == "solar_system" {
                                    act.preset = Some(crate::Preset::SolarSystem);
                                } else {
                                    act.load_preset = Some(slug.clone());
                                }
                            }
                            if idx % 2 == 1 {
                                ui.end_row();
                            }
                        }
                    });
            });
            ui.label(
                RichText::new(
                    "Presets overwrite the current system. Press R to reload the \
                     same preset with fresh phases.",
                )
                .small()
                .color(theme().colors.dim()),
            );
        });
}

/// Four lines of text for one preset card.
fn preset_card_text(preset: &ph::SystemPreset) -> String {
    let m_sun = preset.star.mass_earth / ph::M_SUN_ME;
    let star_desc = match preset.star.kind.as_deref() {
        Some("white_dwarf") => format!("white dwarf ({:.2} M☉)", m_sun),
        Some("neutron_star") => format!("neutron star ({:.2} M☉)", m_sun),
        Some("black_hole") => format!("black hole ({:.2} M☉)", m_sun),
        Some("brown_dwarf") => format!("brown dwarf ({:.2} M☉)", m_sun),
        _ if m_sun < 0.5 => format!("M dwarf ({:.2} M☉)", m_sun),
        _ if m_sun < 0.8 => format!("K dwarf ({:.2} M☉)", m_sun),
        _ if m_sun < 1.2 => format!("G dwarf ({:.2} M☉)", m_sun),
        _ if m_sun < 1.5 => format!("F dwarf ({:.2} M☉)", m_sun),
        _ if m_sun < 2.5 => format!("A dwarf ({:.2} M☉)", m_sun),
        _ if m_sun < 8.0 => format!("B dwarf ({:.2} M☉)", m_sun),
        _ => format!("O supergiant ({:.2} M☉)", m_sun),
    };
    let n_comp = preset.companions.len();
    let n_moons: usize = preset.planets.iter().map(|p| p.moons.len()).sum();
    let mut inner = format!(
        "{} planet{}",
        preset.planets.len(),
        if preset.planets.len() == 1 { "" } else { "s" },
    );
    if n_moons > 0 {
        inner += &format!(", {} moon{}", n_moons, if n_moons == 1 { "" } else { "s" });
    }
    let multi = if n_comp > 0 {
        format!(
            "  +{} companion{}",
            n_comp,
            if n_comp == 1 { "" } else { "s" }
        )
    } else {
        String::new()
    };
    let belt_line = if preset.belts.is_some() {
        "with belts"
    } else {
        "no belts"
    };
    format!(
        "{}\n{}{}\n{}  •  {}\n{:.2} Gyr  •  [Fe/H] {:+.2}",
        preset.name,
        star_desc,
        multi,
        inner,
        belt_line,
        preset.star.age_gyr,
        preset.star.metallicity,
    )
}
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
                    ("`", "show / hide the control panel"),
                    ("Delete", "delete the selected body"),
                    ("I", "spawn an interstellar object inbound"),
                    ("T", "spawn a gas-giant rogue planet inbound"),
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

/// Allocate a plot rectangle of the full available width and a given height,
/// returning the rect and a painter scoped to it. The painter's clip rect is
/// set to the rect so drawing can't bleed into neighboring widgets.
fn plot_rect(ui: &mut egui::Ui, height: f32) -> (egui::Rect, egui::Painter) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    // Background fill so the plot is distinguishable from the panel fill.
    painter.rect_filled(rect, egui::Rounding::same(2.0), Color32::from_gray(20));
    (rect, painter)
}

/// L-shaped axes: left edge and bottom edge of the plot rect.
fn draw_axes(rect: egui::Rect, painter: &egui::Painter) {
    let s = Stroke::new(1.0_f32, Color32::from_gray(60));
    painter.line_segment([rect.left_bottom(), rect.right_bottom()], s);
    painter.line_segment([rect.left_bottom(), rect.left_top()], s);
}

/// Y-range for the N(t) plot. Pads around `[min_n, max_n]` so a flat line
/// isn't glued to the top of the plot. Public for testing.
fn n_plot_range(samples: &std::collections::VecDeque<DiagSample>) -> (f64, f64) {
    let min_n = samples.iter().map(|s| s.n).min().unwrap_or(1);
    let max_n = samples.iter().map(|s| s.n).max().unwrap_or(1);
    let span = (max_n as f64) - (min_n as f64);
    let pad = (span * 0.1).max(1.0);
    let y0 = (min_n as f64 - pad).max(0.0);
    let y1 = (max_n as f64 + pad).max(y0 + 1.0);
    (y0, y1)
}

/// N(t) plot: body count over time. The y-range is padded to at least 1
/// so a plot of a constant-count system doesn't divide by zero.
fn plot_n(ui: &mut egui::Ui, diag: &DiagHistory) {
    let (rect, painter) = plot_rect(ui, 100.0);
    draw_axes(rect, &painter);

    if diag.samples.is_empty() {
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "collecting…",
            FontId::monospace(10.0),
            theme().colors.dim(),
        );
        return;
    }

    let (y0, y1) = n_plot_range(&diag.samples);
    let min_n = diag.samples.iter().map(|s| s.n).min().unwrap_or(1);
    let max_n = diag.samples.iter().map(|s| s.n).max().unwrap_or(1);

    let t0 = diag.samples.front().unwrap().t;
    let t1 = diag.samples.back().unwrap().t;
    let t_span = (t1 - t0).max(1e-9);

    let map = |t: f64, n: usize| -> egui::Pos2 {
        let fx = ((t - t0) / t_span).clamp(0.0, 1.0) as f32;
        let fy = ((n as f64 - y0) / (y1 - y0)).clamp(0.0, 1.0) as f32;
        egui::pos2(
            rect.left() + fx * rect.width(),
            rect.bottom() - fy * rect.height(),
        )
    };

    // Connect consecutive samples with line segments. `prev` is reset to
    // `None` at any gap (there are none in this plot, but the pattern
    // matches `plot_drift`, which has gaps).
    let mut prev: Option<egui::Pos2> = None;
    for s in &diag.samples {
        let p = map(s.t, s.n);
        if let Some(q) = prev {
            painter.line_segment([q, p], Stroke::new(1.0_f32, theme().colors.good()));
        }
        prev = Some(p);
    }

    painter.text(
        rect.left_top() + egui::vec2(4.0, 2.0),
        Align2::LEFT_TOP,
        format!("N(t)  {}..{}", min_n, max_n),
        FontId::monospace(10.0),
        theme().colors.dim(),
    );
    painter.text(
        rect.left_bottom() + egui::vec2(2.0, -2.0),
        Align2::LEFT_BOTTOM,
        fmt_time(t0, true),
        FontId::monospace(9.0),
        theme().colors.dim(),
    );
    painter.text(
        rect.right_bottom() + egui::vec2(-2.0, -2.0),
        Align2::RIGHT_BOTTOM,
        fmt_time(t1, true),
        FontId::monospace(9.0),
        theme().colors.dim(),
    );
}

/// Energy drift plot: drift over time, centered on zero. The y-axis is
/// symmetric so a +2 % excursion is visually the same size as a −2 % one.
fn plot_drift(ui: &mut egui::Ui, diag: &DiagHistory) {
    let (rect, painter) = plot_rect(ui, 100.0);
    draw_axes(rect, &painter);

    let has_data = diag.samples.iter().any(|s| s.drift.is_some());
    if !has_data {
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "collecting…",
            FontId::monospace(10.0),
            theme().colors.dim(),
        );
        return;
    }

    // Symmetric y-range from the largest |drift|. The `.max(0.001)` floor
    // prevents division by zero on a perfectly-conserving sim, and keeps
    // the axis from being "0 to 0" visually.
    let max_abs = diag
        .samples
        .iter()
        .filter_map(|s| s.drift.map(|d| d.abs()))
        .fold(0.0_f64, f64::max)
        .max(0.001);

    let t0 = diag.samples.front().unwrap().t;
    let t1 = diag.samples.back().unwrap().t;
    let t_span = (t1 - t0).max(1e-9);

    let map = |t: f64, d: f64| -> egui::Pos2 {
        let fx = ((t - t0) / t_span).clamp(0.0, 1.0) as f32;
        let fy = ((d + max_abs) / (2.0 * max_abs)).clamp(0.0, 1.0) as f32;
        egui::pos2(
            rect.left() + fx * rect.width(),
            rect.bottom() - fy * rect.height(),
        )
    };

    // Zero line: the "perfect conservation" reference.
    let zero_y = rect.center().y;
    painter.line_segment(
        [
            egui::pos2(rect.left(), zero_y),
            egui::pos2(rect.right(), zero_y),
        ],
        Stroke::new(1.0_f32, Color32::from_gray(50)),
    );

    // Line color depends on the *current* drift: green-ish (theme().colors.accent()) for
    // good runs, orange for > 1 % drift. This makes the whole history
    // change color when the current state is bad, drawing the eye to it.
    let current = diag
        .samples
        .iter()
        .rev()
        .find_map(|s| s.drift)
        .unwrap_or(0.0);
    let line_col = if current.abs() > 1.0 {
        theme().colors.warn()
    } else {
        theme().colors.accent()
    };

    // Gaps in the data (drift is None above ENERGY_MAX_N) break the line,
    // so a system that grew past the threshold doesn't draw a misleading
    // line across the gap.
    let mut prev: Option<egui::Pos2> = None;
    for s in &diag.samples {
        match s.drift {
            Some(d) => {
                let p = map(s.t, d);
                if let Some(q) = prev {
                    painter.line_segment([q, p], Stroke::new(1.0_f32, line_col));
                }
                prev = Some(p);
            }
            None => prev = None,
        }
    }

    painter.text(
        rect.left_top() + egui::vec2(4.0, 2.0),
        Align2::LEFT_TOP,
        format!("E drift ±{:.3}%", max_abs),
        FontId::monospace(10.0),
        theme().colors.dim(),
    );
    painter.text(
        rect.left_bottom() + egui::vec2(2.0, -2.0),
        Align2::LEFT_BOTTOM,
        fmt_time(t0, true),
        FontId::monospace(9.0),
        theme().colors.dim(),
    );
    painter.text(
        rect.right_bottom() + egui::vec2(-2.0, -2.0),
        Align2::RIGHT_BOTTOM,
        fmt_time(t1, true),
        FontId::monospace(9.0),
        theme().colors.dim(),
    );
}

/// Eccentricity histogram: bin counts of the current eccentricity
/// distribution of bound planets. Only updated on `maybe_sample`, so
/// this is a "current snapshot" not a "history" — it doesn't scroll.
fn plot_ecc(ui: &mut egui::Ui, diag: &DiagHistory) {
    let (rect, painter) = plot_rect(ui, 100.0);
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

    // Bar chart: 20 bins of width 0.05 from e=0 to e=1. The peak count
    // determines the y-scale so a single outlier doesn't flatten the bars.
    let max_count = diag.ecc_bins.iter().copied().max().unwrap_or(1).max(1);
    let n = diag.ecc_bin_count;
    let w = rect.width() / n as f32;
    for (k, &cnt) in diag.ecc_bins.iter().enumerate() {
        if cnt == 0 {
            continue;
        }
        let x0 = rect.left() + k as f32 * w;
        // 0.9 leaves a 10 % gap between bars so the bins are visually distinct.
        let x1 = x0 + w * 0.9;
        let h = (cnt as f32 / max_count as f32) * rect.height();
        let y0 = rect.bottom() - h;
        painter.rect_filled(
            egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, rect.bottom())),
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
}

/// The diagnostics window: three stacked plots and a clear button.
fn diag_window(
    ctx: &egui::Context,
    st: &mut UiState,
    _view: &ViewInfo,
    _b: &Bodies,
    _act: &mut Actions,
) {
    let mut open = st.diag.visible;
    egui::Window::new("Diagnostics")
        .open(&mut open)
        .default_pos([640.0, 60.0])
        .default_width(360.0)
        .resizable(true)
        .show(ctx, |ui| {
            if ui.button("clear").clicked() {
                // Clear both the history and the eccentricity snapshot,
                // so the next sample starts from an empty plot instead of
                // a discontinuous jump.
                st.diag.samples.clear();
                st.diag.eccs.clear();
                st.diag.ecc_bins.fill(0);
            }
            plot_n(ui, &st.diag);
            ui.add_space(4.0);
            plot_drift(ui, &st.diag);
            ui.add_space(4.0);
            plot_ecc(ui, &st.diag);
        });
    st.diag.visible = open;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn sample(t: f64, n: usize) -> DiagSample {
        DiagSample { t, n, drift: None }
    }

    #[test]
    fn plot_n_range_tracks_min_max() {
        let mut samples = VecDeque::new();
        samples.push_back(sample(0.0, 100));
        samples.push_back(sample(1.0, 150));
        samples.push_back(sample(2.0, 120));
        let (y0, y1) = n_plot_range(&samples);
        assert!(y0 < 100.0, "y0={y0} should pad below min 100");
        assert!(y1 > 150.0, "y1={y1} should pad above max 150");
        assert!(y0 >= 0.0);
        // Flat signal: still non-degenerate.
        let mut flat = VecDeque::new();
        flat.push_back(sample(0.0, 200));
        flat.push_back(sample(1.0, 200));
        let (y0, y1) = n_plot_range(&flat);
        assert!(y1 - y0 >= 1.0, "flat range too small: {y0}..{y1}");
    }
}
