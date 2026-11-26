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

use crate::physics::{self as ph, Bodies, BodyKind, BodyOrigin};
use crate::view::View3;
use crate::theme::{bold_mono, install_fonts, medium, theme};

mod diagnostics;
use diagnostics::*;
mod edit;
use edit::*;
mod events;
use events::*;
mod history;
pub use history::*;
mod inspector;
use inspector::*;
mod list;
use list::*;
mod plots;
use plots::*;
mod saves;
pub use saves::*;
mod stats;
pub use stats::*;
mod systems;
use systems::*;

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
    // `v * 1e6 / 4.3e6 ≈ v * 0.23`: a body moving at `v = 1` sim length unit
    // per sim time unit moves at about 0.23 km/s.)
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

/// Sort key of the system list.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SysSort {
    Index,
    Name,
    Bodies,
    Mass,
    Radius,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SortKey {
    Id,
    Mass,
    Radius,
    Dist,
    Speed,
    Period,
}

/// One spawned system in a multi-system run, found by `refresh_systems`.
#[derive(Clone)]
pub struct SysEntry {
    /// Id of the system's main star (the heaviest host in it).
    pub root_id: u64,
    pub label: String,
    pub n: usize,
    pub mass: f64,
    pub x: f64,
    pub y: f64,
    /// Distance from the main star that holds 95 % of the bodies, Gm.
    pub radius: f64,
}

/// One body that orbits the selected body.
#[derive(Clone)]
pub struct KidRow {
    pub id: u64,
    pub label: String,
    pub class: String,
    pub m: f64,
    pub dist: f64,
    pub period: f64,
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
    /// Timeline bar along the bottom (J).
    pub show_timeline: bool,
    /// System list window (beside the body list).
    pub show_systems: bool,
    /// Event log window.
    pub show_events: bool,
    /// Which Statistics sections are shown: System, Population, Events,
    /// Conservation, Solver.
    pub stats_sections: [bool; 5],
    /// Active dock tab: 0 Stats, 1 View, 2 Generate, 3 Advanced.
    pub dock_tab: usize,
    /// Sections shown in the selected-body window: Physical, Stellar,
    /// Climate, Orbit, State, Orbiting it.
    pub sel_sections: [bool; 6],
    /// Edit menu sliders and text.
    pub edit_mass: f64,
    pub edit_vel: f64,
    pub edit_moons: usize,
    pub edit_ring: usize,
    pub edit_name: String,
    pub edit_name_for: Option<u64>,
    /// Delete needs a second click. Holds the body id and the time of the first click.
    pub edit_confirm_delete: Option<(u64, Instant)>,
    /// The Saves window is open.
    pub show_saves: bool,
    /// Text in the save-name box.
    pub save_name: String,
    /// Cached list of files in the save folder.
    pub save_list: Vec<ph::SaveEntry>,
    pub save_list_time: Instant,
    /// Last save or load result: text, success, time.
    pub save_msg: Option<(String, bool, Instant)>,
    /// First click of an overwrite or delete: name, action, time.
    pub save_confirm: Option<(String, SaveConfirm, Instant)>,
    /// Sort order of the system list.
    pub sys_sort: SysSort,
    pub sys_sort_desc: bool,
    /// Slug of the preset that is loaded now, if any.
    pub loaded_preset_slug: Option<String>,
    /// Cached (slug, display name) pairs of the preset library.
    preset_names: Vec<(String, String)>,
    preset_names_time: Instant,
    pub diag: DiagHistory,
    pub elem: ElemHistory,
    pub light: LightCurve,
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
    /// Main-star id of the system the body list is limited to, if any.
    pub system_filter: Option<u64>,
    pub systems: Vec<SysEntry>,
    systems_time: Instant,
    /// What orbits the selected body: (selected id, time, rows). `None`
    /// rows mean the system is too large to search.
    pub kids: Option<(u64, Instant, Option<Vec<KidRow>>)>,
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

    /// Show a result line in the Saves window and in the toast.
    pub fn set_save_msg(&mut self, text: impl Into<String>, ok: bool) {
        self.save_msg = Some((text.into(), ok, Instant::now()));
    }

    /// Preset library as (slug, name), re-read at most every 5 s instead of
    /// on every frame.
    fn preset_names(&mut self) -> &[(String, String)] {
        if self.preset_names.is_empty() || self.preset_names_time.elapsed() > Duration::from_secs(5)
        {
            self.preset_names = ph::load_presets()
                .iter()
                .map(|(slug, p)| (slug.clone(), p.name.clone()))
                .collect();
            self.preset_names_time = Instant::now();
        }
        &self.preset_names
    }

    /// Read the save folder again now.
    pub fn refresh_saves(&mut self) {
        self.save_list = ph::list_saves();
        self.save_list_time = Instant::now();
    }

    /// Clear every cache that belongs to the old system. Call after a load.
    pub fn reset_after_load(
        &mut self,
        body_count: usize,
        seed: u128,
        systems: usize,
        slug: Option<String>,
    ) {
        let old = Instant::now() - Duration::from_secs(10);
        self.pending_count = body_count;
        self.set_pending_seed(seed);
        self.seed_text_invalid = false;
        self.pending_systems = systems;
        self.loaded_preset_slug = slug;
        self.multi.clear();
        self.kids = None;
        self.edit_mass = 1.0;
        self.edit_vel = 1.0;
        self.edit_name_for = None;
        self.edit_confirm_delete = None;
        self.system_filter = None;
        self.systems.clear();
        self.stats_time = old;
        self.systems_time = old;
        self.list_time = old;
        let visible = self.diag.visible;
        self.diag = DiagHistory::new();
        self.diag.visible = visible;
        self.elem.clear();
        self.light.clear();
    }

    pub fn new(body_count: usize, seed: u128) -> Self {
        let old = Instant::now() - Duration::from_secs(10);
        Self {
            units_si: true,
            show_labels: false,
            show_list: false,
            show_help: false,
            show_dock: true,
            show_timeline: true,
            show_systems: false,
            show_events: false,
            stats_sections: [true, true, false, true, false],
            dock_tab: 0,
            sel_sections: [true, true, true, true, false, true],
            edit_mass: 1.0,
            edit_vel: 1.0,
            edit_moons: 1,
            edit_ring: 25,
            edit_name: String::new(),
            edit_name_for: None,
            edit_confirm_delete: None,
            show_saves: false,
            save_name: String::from("my save"),
            save_list: Vec::new(),
            save_list_time: Instant::now() - Duration::from_secs(10),
            save_msg: None,
            save_confirm: None,
            sys_sort: SysSort::Index,
            sys_sort_desc: false,
            loaded_preset_slug: None,
            preset_names: Vec::new(),
            preset_names_time: Instant::now() - Duration::from_secs(60),
            diag: DiagHistory::new(),
            elem: ElemHistory::new(),
            light: LightCurve::new(),
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
            system_filter: None,
            systems: Vec::new(),
            systems_time: old,
            kids: None,
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
    pub cam: [f64; 3],
    /// View angles in radians: yaw about the sim z axis, pitch away from top-down.
    pub yaw: f64,
    pub pitch: f64,
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
    /// A random population is being generated in the background.
    pub generating: bool,
    pub show_orbits: bool,
    pub show_debris_orbits: bool,
    pub follow: bool,
    pub selected: Option<u64>,
    pub follow_smooth: Option<(u64, [f64; 3])>,
    pub preset: crate::Preset,
    pub vsync: bool,
    pub behind: bool,
    /// Pause the simulation when a merger, impact or tidal break-up happens.
    pub pause_on_event: bool,
    pub trails_enabled: bool,
    pub trail_age: f64,
    pub trail_bodies: usize,
    pub trail_non_rubble: bool,
    pub sn_kicks: bool,
    pub mass_transfer: bool,
    /// Stellar aging speed relative to real time.
    pub aging_mult: f64,
    /// Recorded history, once there is any.
    pub timeline: Option<TimelineInfo>,
}

/// Body types offered by the single-body and multi-body edit menus.
const KIND_CHOICES: [(BodyKind, &str); 7] = [
    (BodyKind::Planet, "planet"),
    (BodyKind::DwarfPlanet, "dwarf planet"),
    (BodyKind::BrownDwarf, "brown dwarf"),
    (BodyKind::Star, "star"),
    (BodyKind::WhiteDwarf, "white dwarf"),
    (BodyKind::NeutronStar, "neutron star"),
    (BodyKind::BlackHole, "black hole"),
];

/// A single body edit requested by the UI. Applied by `SimState::apply_actions`
/// in `main.rs`. `MassMul` is a *factor* so the same enum variant can serve
/// both "double" and "halve."
pub enum Edit {
    /// Multiply the mass by a factor.
    MassMul(f64),
    SetKind(BodyKind),
    Circularize,
    Delete,
    /// Add this many moons.
    SpawnMoon(usize),
    /// Add a ring of this many particles.
    AddRing(usize),
    /// Scale the speed relative to the parent (or primary) by a factor.
    VelMul(f64),
    /// Give the body a name (empty text clears it).
    Rename(String),
}

/// Accumulated actions for one frame. `Option<T>` means "user requested T
/// this frame"; `None` means "user said nothing about T" (so the current
/// value should be preserved). `bool` means "event happened this frame."
///
/// This is deliberately a flat struct, not an enum-of-enums: the render
/// loop applies actions in order in `apply_actions`, and flatness keeps
/// the match arms simple.
/// One timeline interaction from the UI.
#[derive(Clone, Copy, Debug)]
pub enum TimelineAct {
    /// Show the recorded moment nearest this simulation time.
    Scrub(f64),
    /// Back to the live simulation.
    Live,
    Play(bool),
    /// One recorded frame back (-1) or forward (+1).
    Step(i32),
}

/// What the timeline bar needs to know, built once per frame.
#[derive(Clone, Copy, Debug)]
pub struct TimelineInfo {
    pub t_min: f64,
    pub t_max: f64,
    /// `Some` while replaying: the time being shown.
    pub t_now: Option<f64>,
    pub playing: bool,
    pub frames: usize,
    pub mb: f32,
}

#[derive(Default)]
pub struct Actions {
    pub timeline: Option<TimelineAct>,
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
    /// New view angles `(yaw, pitch)` in radians.
    pub view_angles: Option<(f64, f64)>,
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
    /// Save the whole state under this (clean) name.
    pub save: Option<String>,
    /// Load the save with this name.
    pub load: Option<String>,
    /// Move the camera to (x, y) and zoom to show `radius` (Gm): x, y, radius.
    pub focus: Option<[f64; 3]>,
    /// Slug of a preset from the library to load.
    pub load_preset: Option<String>,
    pub trail_age: Option<f64>,
    pub trail_bodies: Option<usize>,
    pub trail_non_rubble: Option<bool>,
    pub sn_kicks: Option<bool>,
    pub mass_transfer: Option<bool>,
    pub aging_mult: Option<f64>,
}

impl Actions {
    /// Whether applying these actions edits, steps or saves the bodies, so the
    /// physics step in flight must be finished first.
    pub fn changes_bodies(&self) -> bool {
        self.edit.is_some() || self.multi_edit.is_some() || self.step_once || self.save.is_some()
    }

    /// Only the actions that change how the scene is viewed, not the
    /// simulation. Used while a timeline replay is showing.
    pub fn view_only(self) -> Self {
        Self {
            toggle_pause: self.toggle_pause,
            speed: self.speed,
            orbits: self.orbits,
            debris_orbits: self.debris_orbits,
            zoom_mul: self.zoom_mul,
            fit: self.fit,
            view_angles: self.view_angles,
            select: self.select,
            center_on: self.center_on,
            follow: self.follow,
            cycle: self.cycle,
            vsync: self.vsync,
            trails: self.trails,
            diag_visible: self.diag_visible,
            focus: self.focus,
            trail_age: self.trail_age,
            trail_bodies: self.trail_bodies,
            trail_non_rubble: self.trail_non_rubble,
            ..Self::default()
        }
    }
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
    v.window_corner_radius = egui::CornerRadius::same(t.spacing.window_rounding as u8);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, Color32::from_gray(40));
    v.selection.bg_fill = t.colors.selection_bg();
    v.override_text_color = Some(t.colors.text());
    ctx.set_visuals(v);

    let mut style = (*ctx.global_style()).clone();
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
    // Hover tooltips (the "i" explanations) appear after a one second pause.
    style.interaction.tooltip_delay = 1.0;
    // Thin scroll bars that float over the content instead of taking a column.
    // Reserve a gutter for the bar so it never covers the numbers beside it.
    style.spacing.scroll = egui::style::ScrollStyle::floating();
    style.spacing.scroll.floating_allocated_width = style.spacing.scroll.bar_width + 4.0;
    ctx.set_global_style(style);
}
fn panel_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(theme().colors.panel_fill())
        .inner_margin(egui::Margin::same(theme().spacing.panel_margin as i8))
        .stroke(Stroke::new(1.0_f32, Color32::from_gray(40)))
}

/// Key-value row in an `egui::Grid`. Assumes a 2-column grid. The caller
/// must call `ui.end_row()` after the pair — that's why this takes `ui`
/// and not a grid handle.
fn kv(ui: &mut egui::Ui, k: &str, v: impl Into<String>) {
    key_cell(ui, k);
    value_cell(ui, v.into(), None);
    ui.end_row();
}

/// Same as `kv` but colors the value. Used for pass/fail (green/red/orange).
fn kv_color(ui: &mut egui::Ui, k: &str, v: impl Into<String>, c: Color32) {
    key_cell(ui, k);
    value_cell(ui, v.into(), Some(c));
    ui.end_row();
}

/// Sub-row indented under a header or primary row. Same grid layout as
/// `kv`; the two leading spaces indent the label visually and are
/// trimmed for the `info_text` lookup.
fn sub_kv(ui: &mut egui::Ui, k: &str, v: impl Into<String>) {
    key_cell(ui, &format!("  {}", k));
    value_cell(ui, v.into(), None);
    ui.end_row();
}

/// Value cell. Long text wraps so it cannot widen the dock.
fn value_cell(ui: &mut egui::Ui, v: String, c: Option<Color32>) {
    let mut t = RichText::new(&v).monospace();
    if let Some(c) = c {
        t = t.color(c);
    }
    if v.chars().count() > 18 {
        ui.scope(|ui| {
            ui.set_max_width(ui.available_width());
            ui.add(egui::Label::new(t).wrap_mode(egui::TextWrapMode::Wrap));
        });
    } else {
        // Right-align the text so it pushes against the right edge of the column
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(t);
        });
    }
}

/// Draw the circled "i" for `k` if `info_text(k)` has an entry, with
/// hover/click behaviour. Does nothing otherwise. Shared by `key_cell`
/// and `heading_row` so both styles get the tooltip.
fn info_icon(ui: &mut egui::Ui, k: &str) {
    let Some(text) = info_text(k) else {
        return;
    };
    // Salted by the widget position, so rows that share a label (for
    // example "white dwarfs" in two sections) keep separate open state.
    let id = ui.auto_id_with(("stat_info", k.trim()));
    let open = ui.data(|d| d.get_temp::<bool>(id)).unwrap_or(false);
    let (rect, mut r) = ui.allocate_exact_size(egui::vec2(13.0, 13.0), egui::Sense::click());
    if r.clicked() {
        ui.data_mut(|d| d.insert_temp(id, !open));
    }
    let col = if open || r.hovered() {
        theme().colors.accent()
    } else {
        theme().colors.dim()
    };
    ui.painter()
        .circle_stroke(rect.center(), 5.5, Stroke::new(1.0_f32, col));
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        "i",
        FontId::proportional(9.0),
        col,
    );
    if !open {
        r = r.on_hover_ui(|ui| {
            ui.set_max_width(240.0);
            ui.label(RichText::new(text).small());
        });
    }
    if open {
        let area = egui::Area::new(id.with("box"))
            .order(egui::Order::Foreground)
            .fixed_pos(r.rect.left_bottom() + egui::vec2(0.0, 4.0))
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_max_width(240.0);
                    ui.label(RichText::new(text).small());
                });
            });
        let outside = ui.input(|i| i.pointer.any_pressed())
            && !area.response.contains_pointer()
            && !r.contains_pointer();
        if outside {
            ui.data_mut(|d| d.insert_temp(id, false));
        }
    }
}

/// Label cell of a key-value row. Rows that have an explanation get a
/// circled "i": hover for a second, or click, to see a small box.
fn key_cell(ui: &mut egui::Ui, k: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(k).color(theme().colors.dim()));
        info_icon(ui, k);
    });
}

/// Plain-language explanation of each statistic, keyed by its row label.
fn info_text(k: &str) -> Option<&'static str> {
    Some(match k.trim() {
        // ---- System ----
        "elapsed" => "Simulated time since this system was created.",
        "seed" => "Base62 code that rebuilds this exact random system. Type it into the seed box and press Generate to get it again.",
        "systems" => "Number of separate planetary systems, times the target body count of each.",
        "loaded" => "Name of the library preset that is loaded.",
        "bodies" => "Live number of bodies, with a breakdown: stars (★), brown dwarfs (☾), white dwarfs (✦), neutron stars (✹), black holes (●), planets, moons, comets and debris.",
        "total mass" => "Sum of the masses of every body.",
        "phase mix" => "How many stars are on the main sequence or are giants, and how many remnants exist (white dwarfs, neutron stars, black holes).",
        "mass split" => "Share of the total mass held by stars and remnants, by planets, and by debris.",
        // ---- Population ----
        "primary" => "The most massive star-like body. Most distances and orbits are measured from it.",
        "heaviest planet" => "The most massive planet that was spawned with the system.",
        "farthest bound" => "The body farthest from the primary that is still gravitationally bound to it.",
        "on escape orbits" => "Bodies moving fast enough to leave the primary for good (kinetic energy above the gravitational pull).",
        "in habitable zone" => "Planets inside the habitable zone (Kopparapu limits, which depend on the star's colour; about 0.36 to 1.1 times Earth's sunlight for a Sun-like star). This ignores atmosphere, so it is a rough guide.",
        "oldest star" => "Age of the oldest star, in billions of years.",
        "planet eccentricity" => "Eccentricity is how stretched an orbit is: 0 is a circle, close to 1 is a long thin ellipse. Shows the mean and maximum over planets.",
        // ---- Events ----
        "merges" => "Collisions slow enough that two bodies joined into one.",
        "impacts" => "Fast collisions that threw out debris.",
        "torn apart" => "Bodies broken up by tides after passing inside the Roche limit of a heavier body.",
        "escaped" => "Bodies that left the system and were removed.",
        // ---- Conservation ----
        "integrator drift" => "How much the total energy has changed because of numerical error in the solver, as a percent of the starting energy. Near 0 is good; above 1 % means the integration is struggling.",
        "heat / escapes" => "Energy removed or added by physical events (mergers, break-ups, escapes) as a percent. This is real loss, not solver error.",
        "ang. momentum" => "Total angular momentum (how much the whole system spins around). It should stay nearly constant when nothing leaves.",
        "kinetic energy" => "Total energy of motion, half of mass times speed squared, summed over all bodies. In simulation units.",
        "barycentre offset" => "Distance of the system's centre of mass from the origin.",
        "barycentre speed" => "Speed of the centre of mass. It should be near zero unless a body was kicked or left.",
        // ---- Solver ----
        "gravity" => "Method used to compute gravity: direct pair sums (exact), the same in parallel, or the Barnes-Hut tree (approximate, fast for many bodies; theta sets the accuracy).",
        "base step" => "Length of one basic time step, in simulated hours.",
        "integrator" => "Numerical method that advances the bodies: Yoshida 4th order for small calm systems, block leapfrog with variable steps otherwise.",
        "fine ticks / step" => "Close encounters make the solver cut the step into smaller pieces. This is how many pieces were used, and the rung is the depth of subdivision.",
        "steps/s" => "How many base steps the simulation runs per real second.",
        "zoom" => "Screen pixels per gigametre (1 Gm = a million km).",
        // ---- Physical ----
        "mass" => "How much matter the body contains.",
        "vs Earth" => "Mass divided by the mass of Earth.",
        "horizon radius" => "Radius of the event horizon, the point of no return around a black hole.",
        "capture radius" => "How close another body must come to be swallowed.",
        "escape speed" => "Speed needed to leave the body's surface and never fall back. For a black hole it is the speed of light.",
        "radius" => "Distance from the centre to the surface.",
        "bulk density" | "density" => "Mass divided by volume, in grams per cubic centimetre. Water is 1, rock about 3 to 5, iron about 8.",
        "surface gravity" => "Strength of gravity at the surface, in multiples of Earth's.",
        // ---- Stellar ----
        "luminosity" => "Energy output in units of the Sun's (L☉).",
        "temperature" => "Surface temperature in kelvin.",
        "age" | "cooling age" => "Age of the star in billions of years. For a remnant it is the time since it cooled.",
        "initial mass" => "Mass at birth, before the star lost material.",
        "mass lost" => "Share of the birth mass lost through stellar winds and evolution.",
        "phase" => "Stage of the star's life: main sequence (fusing hydrogen), red giant or red supergiant.",
        "time to remnant" => "Time left until the star ends its life and leaves a remnant.",
        "will become" => "The remnant it will leave, with its mass: white dwarf, neutron star or black hole.",
        "stage" => "This body is a stellar remnant and no longer evolves as a star.",
        "habitable zone" => "Range of distances, in AU, where a planet could have liquid water on the surface.",
        // ---- Climate ----
        "stellar flux" => "Starlight received, as a multiple of what Earth gets from the Sun (S⊕). Summed over every luminous body.",
        "equilibrium temp" => "Temperature a bare rock would settle to from starlight alone, without a greenhouse effect. Real surfaces are usually warmer.",
        "rotation" => "How fast the body spins. A planet whose day equals its year always shows one face to its star (tidally locked).",
        "tidal heating" => "Heat from flexing as the body is squeezed along an eccentric orbit (Io is the Solar System's example). Earth's internal heat is about 0.09 W/m².",
        "zone" => "Whether the flux is too hot, in the habitable range (star-colour dependent, ~0.36 to 1.1 S⊕ for the Sun), or too cold.",
        // ---- Orbit ----
        "orbits" => "The body this one is gravitationally bound to (its parent).",
        "distance" => "Current distance to the parent.",
        "speed" => "Speed relative to the parent.",
        "direction" => "Whether it goes around its parent counter-clockwise or clockwise as seen on screen.",
        "speed / circular" => "Current speed divided by the speed of a circular orbit at this distance. 1 is circular, above 1 is faster, and about 1.41 or more means escape.",
        "specific ang. mom." => "Angular momentum per unit mass about the parent. It sets the shape and size of the orbit.",
        "mass / parent" => "Mass of this body divided by the mass of its parent.",
        "semi-major axis" => "Half the longest width of the orbit ellipse: the average orbit size.",
        "eccentricity" => "How stretched the orbit is: 0 is a circle, close to 1 is a long thin ellipse.",
        "period" => "Time for one full orbit.",
        "periapsis" => "Closest distance to the parent on the orbit.",
        "apoapsis" => "Farthest distance from the parent on the orbit.",
        "warning" => "Shown when the orbit will pass through the parent's surface, so a collision is coming.",
        "state" => "Shown when the orbit is unbound: the body is on an escape path and will not return.",
        "Hill sphere of parent" => "Region around the parent where its gravity beats that of the star. A moon must stay inside it to remain bound.",
        "Hill radius" => "Radius of the region around this planet where its own gravity wins over the star's. Moons must orbit inside it.",
        "Roche limit" => "Distance inside which tides from the parent are stronger than the body's own gravity and tear it apart. Orange when within 1.5 times the limit.",
        // ---- State ----
        "position" => "Coordinates in the simulation frame.",
        "speed (frame)" => "Speed in the simulation's own frame, not relative to the parent.",
        "momentum" => "Mass times speed.",
        "spin ang. momentum" => "Angular momentum of the body's own rotation.",
        "distance from primary" => "Distance to the most massive star-like body.",
        "mass rank" => "Position in the list of all bodies sorted by mass (1 is the heaviest).",
        "collision immunity" => "Time left of a short grace period after a collision, during which the body cannot collide again.",
        "origin" => "How the body came to exist: comet, belt body, debris from an impact, interstellar visitor or rogue planet.",
        // ---- Diagnostics ----
        "energy drift" => "Change of total energy because of solver error, in percent of the start value.",
        "energy dissipated" => "Energy removed or added by mergers, break-ups and escapes, in percent.",
        "ang. momentum change" => "Change of total angular momentum since the baseline, in percent.",
        "steps / s" => "Base simulation steps per real second.",
        "fine steps" => "Number of small sub-steps used in the last step. More means a close encounter.",
        "events logged" => "Number of entries in the event log.",
        // ---- Stats sub-rows: body counts ----
        "stars" => "Hydrogen-burning stars (giants are counted here too).",
        "brown dwarfs" => "Objects below the hydrogen-burning limit (~0.08 M☉).",
        "white dwarfs" => "Cooling white-dwarf remnants of low- and intermediate-mass stars.",
        "neutron stars" => "Neutron-star remnants of core-collapse supernovae.",
        "black holes" => "Stellar-mass black holes.",
        "planets" => "Full-sized planets orbiting the primary.",
        "dwarf planets" => "Dwarf planets (Pluto-like: massive enough to be round, not to clear their neighbourhood).",
        "moons" => "Moons orbiting planets and dwarf planets.",
        "comets" => "Comets (origin = Comet), reported separately from planets.",
        "interstellar" => "Interstellar objects currently passing through the system.",
        "rogues" => "Rogue planets currently passing through the system.",
        "debris" => "Belt tracers, rubble (impact and tidal fragments), and background debris.",
        // ---- Stats sub-rows: phase mix ----
        "main sequence" => "Stars burning hydrogen on the main sequence.",
        "giants" => "Stars in the red giant or red supergiant phase.",
        // ---- Stats sub-rows: mass split ----
        "by stars" => "Fraction of total mass held by stars and remnants.",
        "by planets" => "Fraction of total mass held by planets and dwarf planets.",
        "by debris" => "Fraction of total mass held by belts, rubble, comets, and other debris.",
        // ---- Stats sub-rows: farthest bound ----
        "body" => "Index and type of the farthest bound body.",
        "from primary" => "Current distance from the primary star.",
        // ---- Stats sub-rows: planet eccentricity ----
        "mean" => "Mean eccentricity across all planets.",
        "max" => "Maximum eccentricity among all planets.",
        "sampled" => "Number of planets sampled.",
        _ => return None,
    })
}

/// Section divider in the stats grid: strong, colored, spans one cell.
/// The empty second cell is filled with `ui.label("")` so the grid layout
/// doesn't collapse to a single column for the header row.
fn heading_row(ui: &mut egui::Ui, text: &str) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(text)
                .family(medium())
                .color(theme().colors.accent()),
        );
        info_icon(ui, text);
    });
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
pub fn draw(root: &mut egui::Ui, ui_state: &mut UiState, view: &ViewInfo, b: &Bodies) -> Actions {
    let ctx = &root.ctx().clone();
    let mut act = Actions::default();

    // Stats are O(n²) for moons — recompute only every 250 ms, not every
    // frame. This is the main reason `stats` is a cached snapshot rather
    // than a live query.
    if ui_state.stats_time.elapsed() > Duration::from_millis(250) {
        ui_state.stats = compute_stats(b);
        ui_state.stats_time = Instant::now();
    }

    if view.systems > 1 {
        if ui_state.systems_time.elapsed() > Duration::from_millis(1000)
            || ui_state.systems.is_empty()
        {
            refresh_systems(ui_state, b);
            ui_state.systems_time = Instant::now();
        }
    } else if !ui_state.systems.is_empty() || ui_state.system_filter.is_some() {
        ui_state.systems.clear();
        ui_state.system_filter = None;
    }

    draw_overlays(ctx, ui_state, view, b);
    status_bar(root, ui_state, view, b);
    timeline_bar(root, ui_state, view, &mut act);
    save_toast(ctx, ui_state);
    if ui_state.show_dock {
        dock(root, ui_state, view, b, &mut act);
    }
    if ui_state.diag.visible {
        diag_window(ctx, ui_state, view, b, &mut act);
    }
    let replaying = view.timeline.as_ref().is_some_and(|t| t.t_now.is_some());
    ui_state.elem.track(b, view.selected, replaying);
    if ui_state.elem.visible {
        elements_window(ctx, ui_state, view, b);
    }
    ui_state.light.track(b, view.selected, replaying);
    if ui_state.light.visible {
        light_curve_window(ctx, ui_state, view, b);
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
    if ui_state.show_systems {
        systems_window(ctx, ui_state, &mut act);
    }
    if ui_state.show_events {
        events_window(ctx, ui_state, b, &mut act);
    }
    if ui_state.show_saves {
        saves_window(ctx, ui_state, &mut act);
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
fn world_to_screen(view: &ViewInfo, x: f64, y: f64, z: f64) -> egui::Pos2 {
    let v = View3::new(view.cam, view.yaw, view.pitch, view.zoom as f64);
    let p = v.to_screen([x, y, z], view.viewport_px);
    egui::pos2(p[0] as f32 / view.ppp, p[1] as f32 / view.ppp)
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
fn drawn_pos(view: &ViewInfo, b: &Bodies, i: usize) -> (f64, f64, f64) {
    if let Some((id, pos)) = view.follow_smooth {
        if b.id[i] == id {
            return (pos[0], pos[1], pos[2]);
        }
    }
    (b.x[i], b.y[i], b.z[i])
}

/// Draw the selection ring, labels, and scale bar. All three are drawn on
/// the `background` layer so they appear under egui's own layers (the dock,
/// the status bar) but over the wgpu scene.
fn draw_overlays(ctx: &egui::Context, st: &UiState, view: &ViewInfo, b: &Bodies) {
    let painter = ctx.layer_painter(egui::LayerId::background());
    let screen = ctx.content_rect();

    // Selection ring.
    if let Some(id) = view.selected {
        if let Some(i) = b.index_of_id(id) {
            let (bx, by, bz) = drawn_pos(view, b, i);
            let c = world_to_screen(view, bx, by, bz);
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
            let (bx, by, bz) = drawn_pos(view, b, i);
            let c = world_to_screen(view, bx, by, bz);
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
fn status_bar(root: &mut egui::Ui, st: &mut UiState, view: &ViewInfo, b: &Bodies) {
    let ctx = &root.ctx().clone();
    let _ = ctx;
    egui::Panel::top("status")
        .frame(panel_frame().inner_margin(egui::Margin::symmetric(10, 4)))
        .show(root, |ui| {
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
                    ("■ paused", theme().colors.warn())
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
// Timeline
// ---------------------------------------------------------------------------

/// Simulation time as a readable span ("37 d", "4.2 yr", "1.3 Myr").
fn format_sim_time(t: f64) -> String {
    let days = t * crate::physics::time_unit_s() / 86_400.0;
    let yr = days / 365.25;
    let a = yr.abs();
    if a < 1.0 {
        format!("{:.0} d", days)
    } else if a < 1_000.0 {
        format!("{yr:.1} yr")
    } else if a < 1.0e6 {
        format!("{:.1} kyr", yr / 1e3)
    } else {
        format!("{:.2} Myr", yr / 1e6)
    }
}

/// Bottom bar: scrub back through recorded history. Replays are view-only;
/// "Live" returns to the running simulation.
fn timeline_bar(root: &mut egui::Ui, st: &mut UiState, view: &ViewInfo, act: &mut Actions) {
    let Some(tl) = view.timeline else { return };
    if !st.show_timeline {
        return;
    }
    let live = tl.t_now.is_none();
    egui::Panel::bottom("timeline")
        .frame(panel_frame().inner_margin(egui::Margin::symmetric(10, 4)))
        .show(root, |ui| {
            ui.horizontal(|ui| {
                if ui.small_button("⏮").on_hover_text("previous frame").clicked() {
                    act.timeline = Some(TimelineAct::Step(-1));
                }
                let play = if tl.playing { "⏸" } else { "▶" };
                if ui
                    .add_enabled(!live, egui::Button::new(play).small())
                    .on_hover_text("play / pause the replay")
                    .clicked()
                {
                    act.timeline = Some(TimelineAct::Play(!tl.playing));
                }
                if ui.small_button("⏭").on_hover_text("next frame").clicked() {
                    act.timeline = Some(TimelineAct::Step(1));
                }
                let now = tl.t_now.unwrap_or(tl.t_max);
                let label = if live {
                    format!("live  t = {}", format_sim_time(tl.t_max))
                } else {
                    format!(
                        "replay  t = {}  ({} ago)",
                        format_sim_time(now),
                        format_sim_time(tl.t_max - now)
                    )
                };
                let reserve = 330.0;
                let w = (ui.available_width() - reserve).max(80.0);
                if tl.t_max - tl.t_min > 1e-9 {
                    let mut t = now;
                    ui.spacing_mut().slider_width = w;
                    let r = ui.add(
                        egui::Slider::new(&mut t, tl.t_min..=tl.t_max)
                            .show_value(false)
                            .trailing_fill(true),
                    );
                    if r.changed() {
                        act.timeline = Some(TimelineAct::Scrub(t));
                    }
                } else {
                    ui.add_space(w);
                }
                ui.label(RichText::new(label).monospace());
                if ui
                    .selectable_label(live, "● Live")
                    .on_hover_text("return to the running simulation")
                    .clicked()
                {
                    act.timeline = Some(TimelineAct::Live);
                }
                ui.label(
                    RichText::new(format!("{} frames, {:.0} MB", tl.frames, tl.mb))
                        .weak()
                        .small(),
                );
            });
        });
}

// ---------------------------------------------------------------------------
// Dock
// ---------------------------------------------------------------------------

/// The left control panel. Everything is a `CollapsingHeader` so the panel
/// can be scrolled to whatever the user cares about.
fn dock(root: &mut egui::Ui, st: &mut UiState, view: &ViewInfo, b: &Bodies, act: &mut Actions) {
    let ctx = &root.ctx().clone();
    let _ = ctx;
    egui::Panel::left("dock")
        .frame(panel_frame())
        .default_size(235.0)
        .size_range(190.0..=320.0)
        .resizable(true)
        .show(root, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Orbital N-Body")
                        .family(bold_mono())
                        .size(15.0),
                );
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
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        for (k, name) in
                            ["Stats", "View", "Generate", "Advanced"].iter().enumerate()
                        {
                            if ui.selectable_label(st.dock_tab == k, *name).clicked() {
                                st.dock_tab = k;
                            }
                        }
                    });
                    ui.separator();
                    match st.dock_tab {
                        0 => section_stats(ui, st, view, b),
                        1 => section_view(ui, st, view, act),
                        2 => section_generate(ui, st, view, act),
                        _ => section_advanced(ui, st, view, act),
                    }
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
    ui.vertical(|ui| {
        ui.horizontal(|ui| {
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
    ui.vertical(|ui| {
        stats_section_tabs(ui, st);
        let sec = st.stats_sections;
        // `st.stats` is the cached snapshot from `draw` (recomputed every
        // 250 ms), not a live query.
        let s = &st.stats;

        let col_w = (ui.available_width() - 10.0) / 2.0;
        egui::Grid::new("stats_grid")
            .num_columns(2)
            .spacing([10.0, 2.0])
            .min_col_width(col_w)
            .max_col_width(col_w)
            .striped(true)
            .show(ui, |ui| {
                if sec[0] {
                    stats_system_rows(ui, st, s, view, b, si);
                }
                if sec[1] {
                    stats_population_rows(ui, st, s, si);
                }
                if sec[2] {
                    stats_event_rows(ui, b);
                }
                if sec[3] {
                    stats_conservation_rows(ui, s, si);
                }
                if sec[4] {
                    stats_solver_rows(ui, b, view);
                }
            });
    });
}

/// Toggle chips choosing which statistics sections are shown.
fn stats_section_tabs(ui: &mut egui::Ui, st: &mut UiState) {
    ui.horizontal_wrapped(|ui| {
        for (k, name) in ["System", "Population", "Events", "Conservation", "Solver"]
            .iter()
            .enumerate()
        {
            let on = st.stats_sections[k];
            let colour = if on { theme().colors.accent() } else { theme().colors.dim() };
            if ui.selectable_label(on, RichText::new(*name).small().color(colour)).clicked() {
                st.stats_sections[k] = !on;
            }
        }
    });
}

/// "System": elapsed time, seed, body counts by kind, mass totals and splits.
fn stats_system_rows(
    ui: &mut egui::Ui,
    st: &UiState,
    s: &StatsSnapshot,
    view: &ViewInfo,
    b: &Bodies,
    si: bool,
) {
                heading_row(ui, "System");
                kv(ui, "elapsed", fmt_time(b.time, si));
                kv(ui, "seed", ph::seed_to_base62(view.seed));
                kv(
                    ui,
                    "systems",
                    format!("{} × {} bodies", view.systems, view.body_count),
                );
                if let Some(slug) = &st.loaded_preset_slug {
                    kv(ui, "loaded", slug.clone());
                }
                kv(ui, "bodies", format!("{}", s.n));
                sub_kv(ui, "stars", format!("{}", s.stars));
                if s.brown_dwarfs > 0 {
                    sub_kv(ui, "brown dwarfs", format!("{}", s.brown_dwarfs));
                }
                if s.white_dwarfs > 0 {
                    sub_kv(ui, "white dwarfs", format!("{}", s.white_dwarfs));
                }
                if s.neutron_stars > 0 {
                    sub_kv(ui, "neutron stars", format!("{}", s.neutron_stars));
                }
                if s.black_holes > 0 {
                    sub_kv(ui, "black holes", format!("{}", s.black_holes));
                }
                sub_kv(ui, "planets", format!("{}", s.planets));
                if s.dwarf_planets > 0 {
                    sub_kv(ui, "dwarf planets", format!("{}", s.dwarf_planets));
                }
                if let Some(m) = s.moons {
                    sub_kv(ui, "moons", format!("{}", m));
                }
                if s.comets > 0 {
                    sub_kv(ui, "comets", format!("{}", s.comets));
                }
                if s.interstellar > 0 {
                    sub_kv(ui, "interstellar", format!("{}", s.interstellar));
                }
                if s.rogues > 0 {
                    sub_kv(ui, "rogues", format!("{}", s.rogues));
                }
                let debris_val = if s.belt_bodies > 0 && s.belt_bodies == s.rubble {
                    format!("{} (all belt)", s.rubble)
                } else if s.belt_bodies > 0 {
                    format!("{} ({} belt)", s.rubble, s.belt_bodies)
                } else {
                    format!("{}", s.rubble)
                };
                sub_kv(ui, "debris", debris_val);
                kv(ui, "total mass", fmt_mass(s.total_mass, si));
                heading_row(ui, "phase mix");
                sub_kv(
                    ui,
                    "main sequence",
                    format!("{}", s.stars.saturating_sub(s.giants)),
                );
                sub_kv(ui, "giants", format!("{}", s.giants));
                sub_kv(ui, "white dwarfs", format!("{}", s.white_dwarfs));
                sub_kv(ui, "neutron stars", format!("{}", s.neutron_stars));
                sub_kv(ui, "black holes", format!("{}", s.black_holes));

                if s.total_mass > 0.0 {
                    let pc = |x: f64| 100.0 * x / s.total_mass;
                    heading_row(ui, "mass split");
                    sub_kv(ui, "by stars", format!("{:.2} %", pc(s.mass_stellar)));
                    sub_kv(ui, "by planets", format!("{:.3} %", pc(s.mass_planets)));
                    sub_kv(ui, "by debris", format!("{:.4} %", pc(s.mass_debris)));
                }
}

/// "Population": primary, heaviest planet, farthest bound body, eccentricities.
fn stats_population_rows(ui: &mut egui::Ui, st: &UiState, s: &StatsSnapshot, si: bool) {
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
                    heading_row(ui, "farthest bound");
                    sub_kv(ui, "body", name.clone());
                    sub_kv(ui, "from primary", fmt_len(*d, si));
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
                    heading_row(ui, "planet eccentricity");
                    sub_kv(ui, "mean", format!("{:.3}", mean));
                    sub_kv(ui, "max", format!("{:.3}", max));
                    sub_kv(ui, "sampled", format!("{}", eccs.len()));
                }
}

/// "Events": cumulative collision and escape counters.
fn stats_event_rows(ui: &mut egui::Ui, b: &Bodies) {
                heading_row(ui, "Events");
                kv(ui, "merges", fmt_count(b.stats.merges));
                kv(ui, "impacts", fmt_count(b.stats.impacts));
                kv(ui, "torn apart", fmt_count(b.stats.tidal));
                kv(ui, "escaped", fmt_count(b.stats.escaped));
}

/// "Conservation": integrator drift, ang. momentum, barycentre motion.
fn stats_conservation_rows(ui: &mut egui::Ui, s: &StatsSnapshot, si: bool) {
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
}

/// "Solver": which gravity and integration scheme is in use right now.
fn stats_solver_rows(ui: &mut egui::Ui, b: &Bodies, view: &ViewInfo) {
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
}

fn section_view(ui: &mut egui::Ui, st: &mut UiState, view: &ViewInfo, act: &mut Actions) {
    ui.vertical(|ui| {
        ui.horizontal(|ui| {
            if ui
                .button("Fit")
                .on_hover_text("fit the system (F)")
                .clicked()
            {
                act.fit = true;
            }
            if ui
                .button("Top-down")
                .on_hover_text("look straight down the z axis (T)")
                .clicked()
            {
                act.view_angles = Some((0.0, 0.0));
            }
        });
        // Tilt the camera to see inclined orbits. Right-drag in the scene
        // does the same.
        let (mut yaw_deg, mut pitch_deg) = (view.yaw.to_degrees(), view.pitch.to_degrees());
        let c1 = ui
            .add(egui::Slider::new(&mut pitch_deg, -89.0..=89.0).text("tilt°"))
            .changed();
        let c2 = ui
            .add(egui::Slider::new(&mut yaw_deg, -180.0..=180.0).text("spin°"))
            .changed();
        if c1 || c2 {
            act.view_angles = Some((yaw_deg.to_radians(), pitch_deg.to_radians()));
        }
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
        ui.checkbox(&mut st.elem.visible, "orbit elements (X)");
        ui.checkbox(&mut st.light.visible, "light curve (I)");
        let mut hz = ph::hz_rings();
        if ui
            .checkbox(&mut hz, "habitable zone (Z)")
            .on_hover_text("draws each star's habitable zone as a green band; the limits follow its luminosity and colour")
            .changed()
        {
            ph::set_hz_rings(hz);
        }
        ui.horizontal_wrapped(|ui| {
            ui.toggle_value(&mut st.show_list, "Body list (B)");
            ui.toggle_value(&mut st.show_systems, "System list (Y)");
            ui.toggle_value(&mut st.show_events, "Events (E)");
            ui.toggle_value(&mut st.show_saves, "Saves (K)");
        });
    });
}

fn section_advanced(ui: &mut egui::Ui, st: &mut UiState, view: &ViewInfo, act: &mut Actions) {
    let si = st.units_si;
    ui.vertical(|ui| {
        ui.label(
            RichText::new("Display")
                .small()
                .color(theme().colors.accent()),
        );
        let mut debris = view.show_debris_orbits;
        if ui.checkbox(&mut debris, "debris orbits (;)").changed() {
            act.debris_orbits = Some(debris);
        }
        let mut comets = ph::comet_orbits();
        if ui
            .checkbox(&mut comets, "comet orbits (')")
            .on_hover_text("comets have their own orbit toggle, separate from planets and debris; off by default")
            .changed()
        {
            ph::set_comet_orbits(comets);
        }
        let mut vsync = view.vsync;
        if ui
            .checkbox(&mut vsync, "vsync (V)")
            .on_hover_text("disabled = faster rendering, screen tearing")
            .changed()
        {
            act.vsync = Some(vsync);
        }

        ui.add_space(4.0);
        ui.label(
            RichText::new("Trails")
                .small()
                .color(theme().colors.accent()),
        );
        let mut trails = view.trails_enabled;
        if ui.checkbox(&mut trails, "trails").changed() {
            act.trails = Some(trails);
        }
        let mut age = view.trail_age;
        if ui
            .add(
                egui::Slider::new(&mut age, 0.2..=30.0)
                    .logarithmic(true)
                    .custom_formatter(|v, _| fmt_time(v, si))
                    .text("length"),
            )
            .on_hover_text("longest time a trail is kept (never more than one orbit)")
            .changed()
        {
            act.trail_age = Some(age);
        }
        let mut tb = view.trail_bodies as f64;
        if ui
            .add(
                egui::Slider::new(&mut tb, 1.0..=256.0)
                    .integer()
                    .text("bodies"),
            )
            .on_hover_text("trails are drawn for at most this many bodies")
            .changed()
        {
            act.trail_bodies = Some(tb as usize);
        }
        let mut nr = view.trail_non_rubble;
        if ui
            .checkbox(&mut nr, "skip belts and debris")
            .on_hover_text("draw trails only for planets, moons and stars")
            .changed()
        {
            act.trail_non_rubble = Some(nr);
        }

        ui.add_space(4.0);
        ui.label(
            RichText::new("Physics")
                .small()
                .color(theme().colors.accent()),
        );
        let mut sn = view.sn_kicks;
        if ui
            .checkbox(&mut sn, "supernova kicks")
            .on_hover_text("neutron stars and black holes get a random velocity kick at birth")
            .changed()
        {
            act.sn_kicks = Some(sn);
        }
        let mut mt = view.mass_transfer;
        if ui
            .checkbox(&mut mt, "mass transfer")
            .on_hover_text("close binaries can move mass from one star to the other")
            .changed()
        {
            act.mass_transfer = Some(mt);
        }
        let mut disk = ph::gas_disk();
        if ui
            .checkbox(&mut disk, "force gas disk")
            .on_hover_text("young stars (under ~20 Myr) already have a fading gas disk; this forces one around every star. Planets feel it: they spiral in, their orbits circularize, and converging pairs lock into resonances. Sped up so it is visible.")
            .changed()
        {
            ph::set_gas_disk(disk);
        }
        let mut early = ph::early_evolution();
        if ui
            .checkbox(&mut early, "early evolution on generate")
            .on_hover_text("young systems (host under 0.5 Gyr, about 5% of generated ones) first run a time-compressed version of their youth: planets migrate in a fading gas disk, then some packed giant pairs scatter once the gas is gone, and crowded inner planets can merge. Older systems are taken as already settled. Costs up to ~2 s per generate; belts and dust are not part of it.")
            .changed()
        {
            ph::set_early_evolution(early);
        }
        let mut accel = !ph::physical_mode();
        if ui
            .checkbox(&mut accel, "accelerated tides and drag")
            .on_hover_text("speeds up tides, Poynting-Robertson drag and Yarkovsky drift so they show within a session; off uses the physical rates")
            .changed()
        {
            ph::set_physical_mode(!accel);
        }
        let mut age_mult = view.aging_mult;
        let r = ui
            .add(
                egui::Slider::new(&mut age_mult, 1.0..=1e10)
                    .logarithmic(true)
                    .text("star aging ×"),
            )
            .on_hover_text(
                "1 = real time (stars do not visibly change). Raise it to watch stars evolve.",
            );
        if r.changed() {
            act.aging_mult = Some(age_mult);
        }
    });
}

fn section_generate(ui: &mut egui::Ui, st: &mut UiState, view: &ViewInfo, act: &mut Actions) {
    ui.vertical(|ui| {
        // Random-mode is "preset == Random AND not currently showing a
        // library preset". Library presets set `loaded_preset_slug`,
        // which keeps the sliders hidden even though `preset` is
        // still `Random` under the hood.
        let random_selected =
            view.preset == crate::Preset::Random && st.loaded_preset_slug.is_none();
        preset_picker(ui, st, view, random_selected, act);
        if random_selected {
            random_system_controls(ui, st, view, act);
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

/// What the preset drop-down selected.
enum PresetChoice {
    Random,
    SolarSystem,
    Library(String),
}

/// The preset drop-down: Random, the built-in Solar System, and the library.
fn preset_picker(
    ui: &mut egui::Ui,
    st: &mut UiState,
    view: &ViewInfo,
    random_selected: bool,
    act: &mut Actions,
) {
    let names: Vec<(String, String)> = st.preset_names().to_vec();
    let current_name = match &st.loaded_preset_slug {
        Some(slug) => names
            .iter()
            .find(|(s, _)| s == slug)
            .map(|(_, n)| n.clone())
            .unwrap_or_else(|| slug.clone()),
        None => match view.preset {
            crate::Preset::SolarSystem => "Solar System".to_string(),
            crate::Preset::Random => "Random".to_string(),
        },
    };
    let mut choice: Option<PresetChoice> = None;
    ui.horizontal(|ui| {
        ui.label(RichText::new("preset").color(theme().colors.dim()));
        egui::ComboBox::from_id_salt("preset_picker")
            .selected_text(current_name)
            .width(200.0)
            .show_ui(ui, |ui| {
                if ui.selectable_label(random_selected, "Random").clicked() {
                    choice = Some(PresetChoice::Random);
                }
                for (slug, name) in &names {
                    let is_solar = slug == "solar_system";
                    let selected = if is_solar {
                        view.preset == crate::Preset::SolarSystem && st.loaded_preset_slug.is_none()
                    } else {
                        st.loaded_preset_slug.as_deref() == Some(slug.as_str())
                    };
                    if ui.selectable_label(selected, name).clicked() {
                        choice = Some(if is_solar {
                            PresetChoice::SolarSystem
                        } else {
                            PresetChoice::Library(slug.clone())
                        });
                    }
                }
            });
    });
    match choice {
        Some(PresetChoice::Random) => act.preset = Some(crate::Preset::Random),
        Some(PresetChoice::SolarSystem) => act.preset = Some(crate::Preset::SolarSystem),
        Some(PresetChoice::Library(slug)) => act.load_preset = Some(slug),
        None => {}
    }
}

/// Sliders, seed box, status line and the Generate button for random systems.
fn random_system_controls(ui: &mut egui::Ui, st: &mut UiState, view: &ViewInfo, act: &mut Actions) {
    // Body count: the per-system target. With realistic belts this is a
    // *ceiling*, not the actual spawn count: a typical system has ~15 planets
    // and a handful of belt tracers, far below the slider value.
    ui.add(
        egui::Slider::new(&mut st.pending_count, 10..=50_000)
            .logarithmic(true)
            .text("bodies"),
    );
    // Systems: committed on drag-release so dragging doesn't rebuild the sim
    // every frame.
    let r = ui.add(
        egui::Slider::new(&mut st.pending_systems, 1..=ph::MAX_SOLAR_SYSTEMS)
            .logarithmic(true)
            .text("systems"),
    );
    if r.drag_stopped() || (r.changed() && !r.dragged()) {
        act.systems = Some(st.pending_systems);
    }
    body_cap_warning(ui, st.pending_systems, st.pending_count);
    seed_row(ui, st, view);
    population_status_line(ui, view);
    if view.generating {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(RichText::new("generating… (the current scene keeps running)").small().color(theme().colors.dim()));
        });
    }
    if ui.button("Generate").clicked() {
        act.body_count = Some(st.pending_count);
        act.seed = Some(commit_seed_text(&st.pending_seed_text, view.seed));
    }
}

/// Warn when the requested systems × bodies exceeds `MAX_BODIES`.
/// `spawn_random_systems_impl` clamps the per-system count to fit, but the
/// user should see that happening rather than discover it from the live count.
fn body_cap_warning(ui: &mut egui::Ui, n_systems: usize, n_bodies: usize) {
    if n_bodies.saturating_mul(n_systems) <= ph::MAX_BODIES {
        return;
    }
    let effective_per_system = ph::MAX_BODIES / n_systems.max(1);
    ui.label(
        RichText::new(format!(
            "⚠ {n_systems} × {n_bodies} > {}: each system capped at {effective_per_system}",
            ph::MAX_BODIES
        ))
        .small()
        .color(theme().colors.warn()),
    );
}

/// Seed text box (base62), with Random and Copy buttons.
fn seed_row(ui: &mut egui::Ui, st: &mut UiState, view: &ViewInfo) {
    ui.horizontal(|ui| {
        ui.label("seed");
        let r = ui.add(
            egui::TextEdit::singleline(&mut st.pending_seed_text)
                .char_limit(ph::SEED_LEN)
                .desired_width(150.0),
        );
        if r.changed() {
            // An invalid value holds the old seed: `commit_seed_text` falls back.
            st.seed_text_invalid = ph::seed_from_base62(&st.pending_seed_text).is_none();
        }
        if st.seed_text_invalid {
            r.on_hover_text(
                RichText::new("invalid base62 seed").color(Color32::from_rgb(220, 60, 60)),
            );
        }
        if ui.button("Random").on_hover_text("roll a new random seed").clicked() {
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
}

/// `body_count` is the per-system *target*; the live count shows what
/// actually spawned. They differ because realistic belts are sparse: a typical
/// 4.6-Gyr G dwarf has ~15 planets and ~5 debris tracers, not the thousands a
/// full belt would hold.
fn population_status_line(ui: &mut egui::Ui, view: &ViewInfo) {
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
}

/// Most "orbiting it" rows listed in the selection window.
const KIDS_LISTED: usize = 60;

/// The floating window shown when a body is selected. Everything here is
/// read-only except the buttons at the bottom, which emit `Edit` actions.
fn selection_window(
    ctx: &egui::Context,
    st: &mut UiState,
    view: &ViewInfo,
    b: &Bodies,
    i: usize,
    act: &mut Actions,
) {
    let si = st.units_si;
    let mut open = true;
    // One O(n) `find_parent` per frame, reused for the title (moon vs
    // planet) and the Orbit section below.
    let parent = parent_of(b, i);
    refresh_kids_cache(st, b, i);
    sync_edit_fields_to_selection(st, b, i);
    let kind = b.kind[i];
    let stellar = matches!(
        kind,
        BodyKind::Star | BodyKind::BrownDwarf | BodyKind::WhiteDwarf | BodyKind::NeutronStar
    );
    let planetish = matches!(kind, BodyKind::Planet | BodyKind::DwarfPlanet);
    let label = b.name[i].clone().unwrap_or_else(|| format!("#{}", b.id[i]));
    let title = format!("{}  {}", label, ph::body_class_with_parent(b, i, parent));
    egui::Window::new(title)
        .id(egui::Id::unique("selection"))
        .open(&mut open)
        // Anchored top-right with a 10-point margin, so it sits over the
        // 3D scene but doesn't collide with the dock.
        .anchor(Align2::RIGHT_TOP, [-10.0, 40.0])
        .resizable(false)
        .collapsible(true)
        .default_width(260.0)
        .show(ctx, |ui| {
            selection_follow_row(ui, view, b, parent, act);
            let sec = section_toggles(ui, st, stellar, planetish, parent.is_some());
            // Height is capped to "screen height minus 300 points", so the
            // window never overflows on small screens.
            egui::ScrollArea::vertical()
                .id_salt(("sel_scroll", b.id[i]))
                .max_height((ctx.content_rect().height() - 300.0).max(120.0))
                .show(ui, |ui| {
                    if sec[0] || sec[1] || sec[2] || sec[3] || sec[4] {
                        // Half the available width, minus half the spacing (10.0 / 2).
                        let col_w = (ui.available_width() - 10.0) / 2.0;
                        egui::Grid::new("sel_grid")
                            .num_columns(2)
                            .spacing([10.0, 2.0])
                            .min_col_width(col_w) // Force columns to fill the space
                            .max_col_width(col_w)
                            .striped(true)
                            .show(ui, |ui| {
                                selection_rows(ui, si, b, i, parent, sec);
                            });
                    }
                    if sec[5] {
                        if let Some((_, _, kids)) = &st.kids {
                            kids_section(ui, kids, si, act);
                        }
                    }
                });
            // ---- Edit menu, pinned under the list, closed by default ----
            ui.separator();
            edit_menu(ui, st, b, i, parent, planetish, act);
        });
    // If the user closed the window (the little X), clear the selection.
    if !open {
        act.select = Some(None);
    }
}

/// Recompute the "what orbits this body" list when the selection changed or
/// the cached list is older than half a second.
fn refresh_kids_cache(st: &mut UiState, b: &Bodies, i: usize) {
    let stale = match &st.kids {
        Some((id, t, _)) => *id != b.id[i] || t.elapsed() > Duration::from_millis(500),
        None => true,
    };
    if stale {
        st.kids = Some((b.id[i], Instant::now(), compute_kids(b, i)));
    }
}

/// The rename / scale boxes follow the selection: reset them when it changes.
fn sync_edit_fields_to_selection(st: &mut UiState, b: &Bodies, i: usize) {
    if st.edit_name_for != Some(b.id[i]) {
        st.edit_name_for = Some(b.id[i]);
        st.edit_name = b.name[i].clone().unwrap_or_default();
        st.edit_mass = 1.0;
        st.edit_vel = 1.0;
        st.edit_confirm_delete = None;
    }
}

/// Follow checkbox and the "select parent" shortcut at the top of the window.
fn selection_follow_row(
    ui: &mut egui::Ui,
    view: &ViewInfo,
    b: &Bodies,
    parent: Option<usize>,
    act: &mut Actions,
) {
    ui.horizontal_wrapped(|ui| {
        let mut follow = view.follow;
        if ui.checkbox(&mut follow, "follow").changed() {
            act.follow = Some(follow);
        }
        if let Some(p) = parent {
            if ui
                .small_button("select parent")
                .on_hover_text(ph::tag(b, p))
                .clicked()
            {
                act.select = Some(Some(b.id[p]));
            }
        }
    });
}

/// Section toggles. Only the sections that exist for this body are offered.
/// Returns which of the six sections are shown.
fn section_toggles(
    ui: &mut egui::Ui,
    st: &mut UiState,
    stellar: bool,
    planetish: bool,
    has_parent: bool,
) -> [bool; 6] {
    const NAMES: [&str; 6] = ["Physical", "Stellar", "Climate", "Orbit", "State", "Orbiting it"];
    let applicable = [true, stellar, planetish, has_parent, true, true];
    ui.horizontal_wrapped(|ui| {
        for k in 0..6 {
            if !applicable[k] {
                continue;
            }
            let on = st.sel_sections[k];
            let colour = if on { theme().colors.accent() } else { theme().colors.dim() };
            if ui.selectable_label(on, RichText::new(NAMES[k]).small().color(colour)).clicked() {
                st.sel_sections[k] = !on;
            }
        }
    });
    [
        st.sel_sections[0],
        stellar && st.sel_sections[1],
        planetish && st.sel_sections[2],
        has_parent && st.sel_sections[3],
        st.sel_sections[4],
        st.sel_sections[5],
    ]
}

/// The "Orbiting this body" list, or a note when it is empty or too large.
fn kids_section(ui: &mut egui::Ui, kids: &Option<Vec<KidRow>>, si: bool, act: &mut Actions) {
    let dim = theme().colors.dim();
    match kids {
        None => {
            ui.label(
                RichText::new("Too many bodies to list what orbits this one.")
                    .small()
                    .color(dim),
            );
        }
        Some(rows) if rows.is_empty() => {
            ui.label(RichText::new("Nothing orbits this body.").small().color(dim));
        }
        Some(rows) => {
            ui.label(
                RichText::new(format!("Orbiting this body ({})", rows.len()))
                    .strong()
                    .color(theme().colors.accent()),
            );
            egui::ScrollArea::vertical()
                .max_height(170.0)
                .id_salt("kids_scroll")
                .show(ui, |ui| {
                    egui::Grid::new("kids_grid")
                        .num_columns(4)
                        .spacing([10.0, 2.0])
                        .striped(true)
                        .show(ui, |ui| {
                            for r in rows.iter().take(KIDS_LISTED) {
                                if ui.selectable_label(false, &r.label).clicked() {
                                    act.select = Some(Some(r.id));
                                }
                                ui.label(&r.class);
                                ui.label(RichText::new(fmt_len(r.dist, si)).monospace());
                                ui.label(RichText::new(fmt_time(r.period, si)).monospace());
                                ui.end_row();
                            }
                        });
                    if rows.len() > KIDS_LISTED {
                        ui.label(
                            RichText::new(format!("+{} more", rows.len() - KIDS_LISTED))
                                .small()
                                .color(dim),
                        );
                    }
                });
        }
    }
}

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
/// Which spawned system every body belongs to.
struct SystemMap {
    /// Index of each system's main star, in body-index order.
    roots: Vec<usize>,
    /// System number of each body (index into `roots`), `u32::MAX` if none.
    of_body: Vec<u32>,
}

/// Group bodies into systems. The array stores no system id, so systems are
/// rebuilt from the stars. A star that is bound to a heavier star joins that
/// star's system (binaries and triples stay together). Every other body goes
/// to the system of the nearest star. The nearest-star search sorts the stars
/// by x, so it stays fast for 100 000 bodies.
fn assign_systems(b: &Bodies) -> SystemMap {
    use std::cmp::Ordering;
    let n = b.len();
    let hosts: Vec<usize> = (0..n).filter(|&i| ph::is_host_kind(b.kind[i])).collect();
    if hosts.is_empty() {
        return SystemMap {
            roots: Vec::new(),
            of_body: vec![u32::MAX; n],
        };
    }
    let mut by_mass = hosts.clone();
    by_mass.sort_by(|&a, &c| b.m[c].partial_cmp(&b.m[a]).unwrap_or(Ordering::Equal));
    let mut root_of: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::with_capacity(hosts.len());
    if by_mass.len() <= 2000 {
        for (pos, &h) in by_mass.iter().enumerate() {
            let mut best: Option<(usize, f64)> = None;
            for &k in &by_mass[..pos] {
                let dx = b.x[k] - b.x[h];
                let dy = b.y[k] - b.y[h];
                let w = b.m[k] / (dx * dx + dy * dy).max(1e-12);
                if best.is_none_or(|(_, bw)| w > bw) {
                    best = Some((k, w));
                }
            }
            let mut root = h;
            if let Some((k, _)) = best {
                let dx = b.x[h] - b.x[k];
                let dy = b.y[h] - b.y[k];
                let d = (dx * dx + dy * dy).sqrt().max(1e-12);
                let dvx = b.vx[h] - b.vx[k];
                let dvy = b.vy[h] - b.vy[k];
                let mu = ph::G * (b.m[h] + b.m[k]);
                if 0.5 * (dvx * dvx + dvy * dvy) - mu / d < 0.0 {
                    root = root_of.get(&k).copied().unwrap_or(k);
                }
            }
            root_of.insert(h, root);
        }
    } else {
        for &h in &hosts {
            root_of.insert(h, h);
        }
    }
    let roots: Vec<usize> = hosts
        .iter()
        .copied()
        .filter(|h| root_of.get(h) == Some(h))
        .collect();
    let sys_num: std::collections::HashMap<usize, u32> = roots
        .iter()
        .enumerate()
        .map(|(k, &r)| (r, k as u32))
        .collect();
    let mut hx: Vec<(f64, f64, u32)> = hosts
        .iter()
        .map(|&h| {
            (
                b.x[h],
                b.y[h],
                sys_num.get(&root_of[&h]).copied().unwrap_or(0),
            )
        })
        .collect();
    hx.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(Ordering::Equal));
    let nearest = |x: f64, y: f64| -> u32 {
        let pos = hx.partition_point(|h| h.0 < x);
        let (mut l, mut r) = (pos, pos);
        let mut best = f64::INFINITY;
        let mut sys = u32::MAX;
        loop {
            let mut moved = false;
            if l > 0 {
                let h = hx[l - 1];
                let dx = x - h.0;
                if dx * dx < best {
                    let d2 = dx * dx + (y - h.1) * (y - h.1);
                    if d2 < best {
                        best = d2;
                        sys = h.2;
                    }
                    l -= 1;
                    moved = true;
                }
            }
            if r < hx.len() {
                let h = hx[r];
                let dx = h.0 - x;
                if dx * dx < best {
                    let d2 = dx * dx + (y - h.1) * (y - h.1);
                    if d2 < best {
                        best = d2;
                        sys = h.2;
                    }
                    r += 1;
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
        sys
    };
    let of_body: Vec<u32> = (0..n).map(|i| nearest(b.x[i], b.y[i])).collect();
    SystemMap { roots, of_body }
}

/// Rebuild the system list (call about once a second).
fn refresh_systems(st: &mut UiState, b: &Bodies) {
    use std::cmp::Ordering;
    let map = assign_systems(b);
    let k = map.roots.len();
    let mut counts = vec![0usize; k];
    let mut masses = vec![0.0f64; k];
    let mut dists: Vec<Vec<f64>> = vec![Vec::new(); k];
    for i in 0..b.len() {
        let s = map.of_body[i];
        if s == u32::MAX {
            continue;
        }
        let s = s as usize;
        let r = map.roots[s];
        let dx = b.x[i] - b.x[r];
        let dy = b.y[i] - b.y[r];
        counts[s] += 1;
        masses[s] += b.m[i];
        dists[s].push((dx * dx + dy * dy).sqrt());
    }
    let mut out = Vec::with_capacity(k);
    for (s, &root) in map.roots.iter().enumerate() {
        let mut d = std::mem::take(&mut dists[s]);
        let radius = if d.is_empty() {
            50.0
        } else {
            let q = ((d.len() as f64 * 0.95) as usize).min(d.len() - 1);
            d.select_nth_unstable_by(q, |a, c| a.partial_cmp(c).unwrap_or(Ordering::Equal));
            d[q]
        };
        out.push(SysEntry {
            root_id: b.id[root],
            label: ph::tag(b, root),
            n: counts[s],
            mass: masses[s],
            x: b.x[root],
            y: b.y[root],
            radius: radius.max(10.0),
        });
    }
    st.systems = out;
    if let Some(f) = st.system_filter {
        if !st.systems.iter().any(|s| s.root_id == f) {
            st.system_filter = None;
        }
    }
}

/// The heavier star that star `k` is bound to, if any. `hosts` lists every
/// host-kind body, so the search is O(hosts) instead of O(n).
fn host_parent(b: &Bodies, hosts: &[usize], k: usize) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for &h in hosts {
        if h == k {
            continue;
        }
        if b.m[h] < b.m[k] || (b.m[h] == b.m[k] && h > k) {
            continue;
        }
        let dx = b.x[h] - b.x[k];
        let dy = b.y[h] - b.y[k];
        let w = b.m[h] / (dx * dx + dy * dy).max(1e-12);
        if best.is_none_or(|(_, bw)| w > bw) {
            best = Some((h, w));
        }
    }
    let (h, _) = best?;
    let dx = b.x[k] - b.x[h];
    let dy = b.y[k] - b.y[h];
    let d = (dx * dx + dy * dy).sqrt().max(1e-12);
    let dvx = b.vx[k] - b.vx[h];
    let dvy = b.vy[k] - b.vy[h];
    let mu = ph::G * (b.m[h] + b.m[k]);
    (0.5 * (dvx * dvx + dvy * dvy) - mu / d < 0.0).then_some(h)
}

/// Every body that orbits body `i`, nearest first. `None` when the system is
/// too large to search without the parent cache.
fn compute_kids(b: &Bodies, i: usize) -> Option<Vec<KidRow>> {
    let n = b.len();
    let cached = b.parent_cache_valid && b.parent_cache.len() == n;
    if !cached && n > 2000 {
        return None;
    }
    let mut rows = Vec::new();
    let hosts: Vec<usize> = (0..n).filter(|&h| ph::is_host_kind(b.kind[h])).collect();
    // Binary search among stars is O(hosts²); skip it in huge clusters.
    let search_hosts = hosts.len() <= 2000;
    for k in 0..n {
        if k == i {
            continue;
        }
        let parent = if ph::is_host_kind(b.kind[k]) {
            if search_hosts {
                host_parent(b, &hosts, k)
            } else {
                None
            }
        } else if cached {
            b.parent_cache[k]
        } else {
            ph::find_parent(b, k)
        };
        if parent != Some(i) {
            continue;
        }
        let o = ph::orbit_info(b, k, i);
        rows.push(KidRow {
            id: b.id[k],
            label: b.name[k].clone().unwrap_or_else(|| format!("#{}", b.id[k])),
            class: ph::body_class_with_parent(b, k, Some(i)).into_owned(),
            m: b.m[k],
            dist: o.dist,
            period: o.period,
        });
    }
    rows.sort_by(|a, c| {
        a.dist
            .partial_cmp(&c.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Some(rows)
}

fn refresh_rows(st: &mut UiState, b: &Bodies) {
    let n = b.len();
    let prim = ph::primary_index(b);
    let key = st.sort;

    // Cache `find_parent` once: sort and display share it, so belt bodies
    // get real values (orbit about the primary) instead of 0 / ∞.
    let owned: Vec<Option<usize>>;
    let parents: &[Option<usize>] = if b.parent_cache_valid && b.parent_cache.len() == n {
        &b.parent_cache
    } else {
        owned = (0..n).map(|i| ph::find_parent(b, i)).collect();
        &owned
    };
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
    if let Some(root_id) = st.system_filter {
        let map = assign_systems(b);
        if let Some(sn) = map.roots.iter().position(|&r| b.id[r] == root_id) {
            idx.retain(|&i| map.of_body[i] == sn as u32);
        }
    }
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
    // Sort keys once per body: `val` can call `orbit_info`, which is too slow
    // to repeat inside the comparator.
    let mut keyed: Vec<(f64, usize)> = idx.iter().map(|&i| (val(i), i)).collect();
    keyed.sort_by(|a, c| a.0.partial_cmp(&c.0).unwrap_or(std::cmp::Ordering::Equal));
    idx = keyed.into_iter().map(|(_, i)| i).collect();
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
