//! Save and load of the full simulation state.
//!
//! A save is one JSON file in the `saves/` folder (see `SAVE_DIR`). It holds
//! every body, the dust fields, the event log, the RNG state, the energy
//! bookkeeping and a few app settings (camera, speed, selection).
//!
//! ## Design rules
//!
//! * **Exact resume.** Floating-point values are written in shortest
//!   round-trip form. `serde_json` needs its `float_roundtrip` feature to read
//!   them back bit for bit (see `Cargo.toml`). With that feature, a loaded
//!   system continues with the same numbers that were saved.
//! * **Derived data is not saved.** Accelerations (`ax`, `ay`), previous
//!   positions, the parent cache and the Barnes–Hut tree are rebuilt on load.
//!   Flashes are short visual effects and are dropped.
//! * **Dead bodies are not saved.** Indices change, so dust-field `primary`
//!   indices are remapped. A dust field whose primary is gone is dropped.
//! * **Load is all or nothing.** The file is checked and a new `Bodies` is
//!   built first. The live state is replaced only when that succeeds.
//! * **No new dependencies.** Enums are written as short text names, so the
//!   physics types need no `serde` derives.
//!
//! Non-finite numbers cannot be written in JSON. Fields that may be infinite
//! (`dtl`, `imm`, grain `ttl`) use `None` or `f64::MAX` as the marker. Every
//! other number is checked and `capture_save` refuses to save a NaN.

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Value of the `format` field. A file without it is not one of our saves.
pub const SAVE_FORMAT: &str = "nbody-rust-save";
/// Bump when the layout changes in a way old code cannot read.
pub const SAVE_VERSION: u32 = 2;
/// Folder for save files, relative to the working directory.
pub const SAVE_DIR: &str = "saves";
/// End of every save file name.
pub const SAVE_EXT: &str = ".nbody.json";
/// Upper limit on bodies in one save file: the engine cap, `MAX_BODIES`.
///
/// Defined *from* `MAX_BODIES` rather than as its own number, because a file
/// the engine cannot hold is not a loadable save: `restore_save` builds a
/// fresh `Bodies` and everything downstream (body budgeting, the tree, the
/// renderer) assumes the live count fits under the cap. The two constants were
/// once 2 000 000 and 256 000, so a 2 M-body file passed validation and then
/// filled the engine past its limit.
pub const SAVE_MAX_BODIES: usize = MAX_BODIES;
/// Longest save name, in characters.
const SAVE_NAME_MAX: usize = 48;

/// Gives each load a new dust-field version range, so the renderer sees the
/// loaded fields as changed even when the old and new version numbers match.
static LOAD_EPOCH: AtomicU64 = AtomicU64::new(1);

// ---------------------------------------------------------------------------
// File layout
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SaveFile {
    pub format: String,
    pub version: u32,
    /// Seconds since the Unix epoch when the file was made.
    pub saved_unix: u64,
    pub sim: SimSave,
    pub app: AppSave,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SimSave {
    pub time: f64,
    pub next_id: u64,
    pub metallicity: f64,
    pub rng_state: u32,
    pub sn_kicks: bool,
    pub mass_transfer: bool,
    pub evolution_rate: f64,
    pub stats: StatsSave,
    pub bodies: Vec<BodySave>,
    pub dust: Vec<DustSave>,
    pub events: Vec<EventSave>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StatsSave {
    pub merges: u64,
    pub impacts: u64,
    pub tidal: u64,
    pub escaped: u64,
    pub steps: u64,
    pub last_nsub: u32,
    pub max_rung: u8,
    pub energy0: f64,
    pub energy_offset: f64,
    pub coalesces: u64,
    pub bounces: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BodySave {
    pub id: u64,
    pub name: Option<String>,
    pub kind: String,
    pub origin: String,
    pub class: String,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    /// Out-of-plane position and velocity (version 2). Version 1 files have
    /// none and load as 0.
    #[serde(default)]
    pub z: f64,
    #[serde(default)]
    pub vz: f64,
    pub m: f64,
    pub r: f64,
    /// `None` means "not finite" (read back as infinity for `dtl`, 0 for `imm`).
    pub imm: Option<f64>,
    pub dtl: Option<f64>,
    pub bulk_density: f64,
    pub spin: f64,
    pub age_gyr: f64,
    pub m_init: f64,
    pub rung: u8,
    pub coalesce_counter: u32,
    pub comet_grains: u32,
    pub disrupting: bool,
    pub rlof: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DustSave {
    /// Index into `SimSave::bodies` (already remapped).
    pub primary: usize,
    pub extent_gm: f64,
    pub version: u64,
    /// One grain per row: `[x, y, vx, vy, mass, size_um, ttl]`.
    /// `ttl == f64::MAX` stands for "never expires".
    /// `[x, y, vx, vy, mass, size, ttl]`, plus `z, vz` appended in v2 saves.
    pub grains: Vec<Vec<f64>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EventSave {
    pub time: f64,
    pub kind: String,
    pub text: String,
    pub x: f64,
    pub y: f64,
}

/// App-level state: what the camera sees and how the run is controlled.
/// The binary builds this from `SimState` and applies it again after a load.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AppSave {
    pub cam: [f64; 2],
    /// Focus height and view angles; absent in older saves (top-down view).
    #[serde(default)]
    pub cam_z: f64,
    #[serde(default)]
    pub yaw: f64,
    #[serde(default)]
    pub pitch: f64,
    pub zoom: f32,
    pub speed: f64,
    pub paused: bool,
    /// Base62 text, same as the seed box in the UI.
    pub seed: String,
    pub body_count: usize,
    /// `"random"` or `"solar"`.
    pub preset: String,
    pub systems: usize,
    pub library_preset: Option<String>,
    pub selected: Option<u64>,
    pub follow: bool,
    pub show_orbits: bool,
    pub show_debris_orbits: bool,
    pub aging_mult: f64,
}

// ---------------------------------------------------------------------------
// Enum <-> text
// ---------------------------------------------------------------------------

/// Builds a `to_name` and a `from_name` function for one enum. The `match` in
/// `to_name` has no `_` arm, so a new variant is a compile error here and
/// cannot be forgotten.
macro_rules! name_table {
    ($to:ident, $from:ident, $ty:ty, { $($v:path => $s:literal),+ $(,)? }) => {
        fn $to(x: $ty) -> &'static str {
            match x { $($v => $s),+ }
        }
        fn $from(s: &str) -> Result<$ty, String> {
            match s {
                $($s => Ok($v),)+
                _ => Err(format!("unknown {} \"{}\"", stringify!($ty), s)),
            }
        }
    };
}

name_table!(kind_to_str, kind_from_str, BodyKind, {
    BodyKind::Star => "star",
    BodyKind::BrownDwarf => "brown_dwarf",
    BodyKind::WhiteDwarf => "white_dwarf",
    BodyKind::NeutronStar => "neutron_star",
    BodyKind::BlackHole => "black_hole",
    BodyKind::Planet => "planet",
    BodyKind::DwarfPlanet => "dwarf_planet",
});

name_table!(origin_to_str, origin_from_str, BodyOrigin, {
    BodyOrigin::Spawned => "spawned",
    BodyOrigin::AsteroidBelt => "asteroid_belt",
    BodyOrigin::KuiperCold => "kuiper_cold",
    BodyOrigin::KuiperHot => "kuiper_hot",
    BodyOrigin::KuiperResonant => "kuiper_resonant",
    BodyOrigin::KuiperScattered => "kuiper_scattered",
    BodyOrigin::Trojan => "trojan",
    BodyOrigin::Hilda => "hilda",
    BodyOrigin::ImpactDebris => "impact_debris",
    BodyOrigin::TidalDebris => "tidal_debris",
    BodyOrigin::BackgroundDebris => "background_debris",
    BodyOrigin::Comet => "comet",
    BodyOrigin::Interstellar => "interstellar",
    BodyOrigin::RoguePlanet => "rogue_planet",
});

name_table!(class_to_str, class_from_str, BodyClass, {
    BodyClass::C => "c",
    BodyClass::S => "s",
    BodyClass::M => "m",
    BodyClass::V => "v",
    BodyClass::Icy => "icy",
    BodyClass::Rocky => "rocky",
});

name_table!(event_to_str, event_from_str, EventKind, {
    EventKind::Merge => "merge",
    EventKind::Impact => "impact",
    EventKind::Tidal => "tidal",
    EventKind::Escape => "escape",
    EventKind::User => "user",
    EventKind::Evolve => "evolve",
    EventKind::Comet => "comet",
    EventKind::Interstellar => "interstellar",
    EventKind::Eject => "eject",
});

// ---------------------------------------------------------------------------
// Capture and restore
// ---------------------------------------------------------------------------

fn need_finite(what: &str, vals: &[f64]) -> Result<(), String> {
    if vals.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(format!("{what} has a value that is not a finite number"))
    }
}

fn finite_or(v: f64, fallback: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        fallback
    }
}

/// Copy the full simulation state into a `SaveFile`.
///
/// Reads only. Fails if a body holds a NaN or infinity in a field that JSON
/// cannot store; the error names the body, so the user can find the cause.
pub fn capture_save(b: &Bodies, app: AppSave) -> Result<SaveFile, String> {
    let (bodies, remap) = capture_bodies(b)?;
    let dust = capture_dust(b, &remap)?;
    let events = capture_events(b);
    need_finite(
        "the simulation",
        &[b.time, b.metallicity, b.evolution_rate, b.stats.energy0, b.stats.energy_offset],
    )?;
    Ok(SaveFile {
        format: SAVE_FORMAT.to_string(),
        version: SAVE_VERSION,
        saved_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        sim: SimSave {
            time: b.time,
            next_id: b.next_id,
            metallicity: b.metallicity,
            rng_state: b.rng.s,
            sn_kicks: b.sn_kicks,
            mass_transfer: b.mass_transfer,
            evolution_rate: b.evolution_rate,
            stats: capture_stats(&b.stats),
            bodies,
            dust,
            events,
        },
        app,
    })
}

/// The live bodies, plus a map from old index to new index (`usize::MAX` for dead bodies).
fn capture_bodies(b: &Bodies) -> Result<(Vec<BodySave>, Vec<usize>), String> {
    let n = b.len();
    let mut remap = vec![usize::MAX; n];
    let mut bodies: Vec<BodySave> = Vec::with_capacity(n);
    for i in (0..n).filter(|&i| !b.dead[i]) {
        need_finite(
            &format!("body #{}", b.id[i]),
            &[
                b.x[i],
                b.y[i],
                b.vx[i],
                b.vy[i],
                b.z[i],
                b.vz[i],
                b.m[i],
                b.r[i],
                b.bulk_density[i],
                b.spin[i],
                b.age_gyr[i],
                b.m_init[i],
            ],
        )?;
        remap[i] = bodies.len();
        bodies.push(capture_body(b, i));
    }
    Ok((bodies, remap))
}

fn capture_body(b: &Bodies, i: usize) -> BodySave {
    BodySave {
        id: b.id[i],
        name: b.name[i].clone(),
        kind: kind_to_str(b.kind[i]).to_string(),
        origin: origin_to_str(b.origin[i]).to_string(),
        class: class_to_str(b.class[i]).to_string(),
        x: b.x[i],
        y: b.y[i],
        vx: b.vx[i],
        vy: b.vy[i],
        z: b.z[i],
        vz: b.vz[i],
        m: b.m[i],
        r: b.r[i],
        imm: b.imm[i].is_finite().then_some(b.imm[i]),
        dtl: b.dtl[i].is_finite().then_some(b.dtl[i]),
        bulk_density: b.bulk_density[i],
        spin: b.spin[i],
        age_gyr: b.age_gyr[i],
        m_init: b.m_init[i],
        rung: b.rung[i],
        coalesce_counter: b.coalesce_counter[i],
        comet_grains: b.comet_grains[i],
        disrupting: b.disrupting[i],
        rlof: b.rlof[i],
    }
}

/// The dust fields whose primary survived, with the primary index remapped.
fn capture_dust(b: &Bodies, remap: &[usize]) -> Result<Vec<DustSave>, String> {
    let mut dust = Vec::with_capacity(b.dust_fields.len());
    for f in &b.dust_fields {
        let primary = match remap.get(f.primary) {
            Some(&p) if p != usize::MAX => p,
            _ => continue, // The primary is gone: the field has no anchor.
        };
        need_finite("a dust field", &[f.extent_gm])?;
        let grains = f.grains.iter().map(capture_grain).collect::<Result<Vec<_>, _>>()?;
        dust.push(DustSave { primary, extent_gm: f.extent_gm, version: f.version, grains });
    }
    Ok(dust)
}

/// One grain as `x y vx vy mass size ttl z vz`; an infinite lifetime becomes `f64::MAX`.
fn capture_grain(g: &Grain) -> Result<Vec<f64>, String> {
    need_finite("a dust grain", &[g.x, g.y, g.vx, g.vy, g.mass, g.size_um])?;
    let ttl = if g.ttl.is_finite() { g.ttl as f64 } else { f64::MAX };
    Ok(vec![g.x, g.y, g.vx, g.vy, g.mass, g.size_um, ttl, g.z, g.vz])
}

fn capture_events(b: &Bodies) -> Vec<EventSave> {
    b.events
        .iter()
        .map(|e| EventSave {
            time: finite_or(e.time, 0.0),
            kind: event_to_str(e.kind).to_string(),
            text: e.text.clone(),
            x: finite_or(e.x, 0.0),
            y: finite_or(e.y, 0.0),
        })
        .collect()
}

fn capture_stats(s: &Stats) -> StatsSave {
    StatsSave {
        merges: s.merges,
        impacts: s.impacts,
        tidal: s.tidal,
        escaped: s.escaped,
        steps: s.steps,
        last_nsub: s.last_nsub,
        max_rung: s.max_rung,
        energy0: s.energy0,
        energy_offset: s.energy_offset,
        coalesces: s.coalesces,
        bounces: s.bounces,
    }
}

/// Replace `b` with the state in `f`.
///
/// The file is checked while a new `Bodies` is built. If anything is wrong,
/// the function returns an error and `b` is **not** changed. After the swap,
/// accelerations are recomputed once. The energy baseline from the file is
/// kept, so the drift display continues where it stopped.
pub fn restore_save(b: &mut Bodies, f: &SaveFile) -> Result<(), String> {
    check_save_header(f)?;
    let s = &f.sim;
    let mut nb = Bodies::with_capacity(s.bodies.len());
    restore_bodies(&mut nb, s)?;
    restore_sim_settings(&mut nb, s);
    restore_events(&mut nb, s)?;
    restore_dust(&mut nb, s)?;
    nb.parent_cache_valid = false;
    compute_accel(&mut nb);
    *b = nb;
    Ok(())
}

/// Reject files that are not saves, are too new, or are empty or too large.
fn check_save_header(f: &SaveFile) -> Result<(), String> {
    if f.format != SAVE_FORMAT {
        return Err("this file is not an n-body save".to_string());
    }
    if f.version > SAVE_VERSION {
        return Err(format!(
            "this save has format version {} but this program reads up to {}",
            f.version, SAVE_VERSION
        ));
    }
    let s = &f.sim;
    if s.bodies.is_empty() {
        return Err("the save has no bodies".to_string());
    }
    check_body_count(s.bodies.len())?;
    need_finite(
        "the simulation",
        &[s.time, s.metallicity, s.stats.energy0, s.stats.energy_offset],
    )
}

/// Reject a body count this build cannot hold.
///
/// Kept separate from `check_save_header` so the boundary can be tested
/// without materialising a 256 000-entry save, and public so a caller can
/// reject an oversized file from its metadata before parsing it.
pub fn check_body_count(count: usize) -> Result<(), String> {
    if count > SAVE_MAX_BODIES {
        return Err(format!(
            "the save has {count} bodies but this build supports at most {SAVE_MAX_BODIES}"
        ));
    }
    Ok(())
}

/// Rebuild every body, checking values and id uniqueness as it goes.
fn restore_bodies(nb: &mut Bodies, s: &SimSave) -> Result<(), String> {
    let mut seen: HashSet<u64> = HashSet::with_capacity(s.bodies.len());
    let mut max_id = 0u64;
    for bs in &s.bodies {
        let what = format!("body #{}", bs.id);
        need_finite(
            &what,
            &[
                bs.x,
                bs.y,
                bs.vx,
                bs.vy,
                bs.z,
                bs.vz,
                bs.m,
                bs.r,
                bs.bulk_density,
                bs.spin,
                bs.age_gyr,
                bs.m_init,
            ],
        )?;
        if bs.m < 0.0 || bs.r < 0.0 {
            return Err(format!("{what} has a mass or radius that is not valid"));
        }
        if !seen.insert(bs.id) {
            return Err(format!("{what} appears twice"));
        }
        max_id = max_id.max(bs.id);
        restore_body(nb, bs)?;
    }
    nb.next_id = s.next_id.max(max_id.saturating_add(1));
    Ok(())
}

/// Push one saved body, defaulting optional fields that are missing or not finite.
fn restore_body(nb: &mut Bodies, bs: &BodySave) -> Result<(), String> {
    let kind = kind_from_str(&bs.kind)?;
    let i = nb.push(bs.x, bs.y, bs.vx, bs.vy, bs.m, kind);
    nb.z[i] = bs.z;
    nb.vz[i] = bs.vz;
    nb.id[i] = bs.id;
    nb.name[i] = bs.name.clone();
    nb.origin[i] = origin_from_str(&bs.origin)?;
    nb.class[i] = class_from_str(&bs.class)?;
    nb.r[i] = bs.r;
    nb.imm[i] = bs.imm.filter(|v| v.is_finite()).unwrap_or(0.0);
    nb.dtl[i] = bs.dtl.filter(|v| v.is_finite()).unwrap_or(f64::INFINITY);
    nb.bulk_density[i] = bs.bulk_density;
    nb.spin[i] = bs.spin;
    nb.age_gyr[i] = bs.age_gyr;
    nb.m_init[i] = bs.m_init;
    nb.rung[i] = bs.rung.min(MAX_RUNG);
    nb.coalesce_counter[i] = bs.coalesce_counter;
    nb.comet_grains[i] = bs.comet_grains;
    nb.disrupting[i] = bs.disrupting;
    nb.rlof[i] = bs.rlof;
    Ok(())
}

/// Clock, RNG, stellar-evolution options and the statistics counters.
fn restore_sim_settings(nb: &mut Bodies, s: &SimSave) {
    nb.time = s.time;
    nb.metallicity = s.metallicity;
    nb.rng = XorShift { s: s.rng_state.max(1) };
    nb.sn_kicks = s.sn_kicks;
    nb.mass_transfer = s.mass_transfer;
    nb.evolution_rate = if s.evolution_rate.is_finite() && s.evolution_rate > 0.0 {
        s.evolution_rate
    } else {
        0.0
    };
    nb.stats = Stats {
        merges: s.stats.merges,
        impacts: s.stats.impacts,
        tidal: s.stats.tidal,
        escaped: s.stats.escaped,
        steps: s.stats.steps,
        last_nsub: s.stats.last_nsub,
        max_rung: s.stats.max_rung,
        energy0: s.stats.energy0,
        energy_offset: s.stats.energy_offset,
        coalesces: s.stats.coalesces,
        bounces: s.stats.bounces,
    };
}

/// The event log, trimmed to its maximum length.
fn restore_events(nb: &mut Bodies, s: &SimSave) -> Result<(), String> {
    for e in &s.events {
        nb.events.push_back(Event {
            time: e.time,
            kind: event_from_str(&e.kind)?,
            text: e.text.clone(),
            x: e.x,
            y: e.y,
        });
    }
    while nb.events.len() > EVENT_LOG_MAX {
        nb.events.pop_front();
    }
    Ok(())
}

/// The dust fields. Each gets a version stamped with the load epoch, which the
/// renderer compares to find changed fields.
fn restore_dust(nb: &mut Bodies, s: &SimSave) -> Result<(), String> {
    let epoch = LOAD_EPOCH.fetch_add(1, AtomicOrdering::Relaxed);
    for d in &s.dust {
        if d.primary >= nb.len() {
            return Err("a dust field points to a body that is not in the save".to_string());
        }
        need_finite("a dust field", &[d.extent_gm])?;
        let grains = d.grains.iter().map(|g| restore_grain(g)).collect::<Result<Vec<_>, _>>()?;
        nb.dust_fields.push(DustField {
            primary: d.primary,
            extent_gm: d.extent_gm,
            grains,
            version: (epoch << 32) | (d.version & 0xFFFF_FFFF),
        });
    }
    Ok(())
}

/// One grain from its saved values: 7 (planar, older saves) or 9 numbers
/// `x y vx vy mass size ttl [z vz]`. An infinite lifetime is stored as a huge number.
fn restore_grain(g: &[f64]) -> Result<Grain, String> {
    if g.len() != 7 && g.len() != 9 {
        return Err("a dust grain has the wrong number of values".to_string());
    }
    need_finite("a dust grain", &g[..6])?;
    let (gz, gvz) = if g.len() == 9 { (g[7], g[8]) } else { (0.0, 0.0) };
    need_finite("a dust grain", &[gz, gvz])?;
    if !g[6].is_finite() {
        return Err("a dust grain has a lifetime that is not valid".to_string());
    }
    let ttl = if g[6] >= f32::MAX as f64 { f32::INFINITY } else { g[6] as f32 };
    Ok(Grain {
        x: g[0],
        y: g[1],
        z: gz,
        vx: g[2],
        vy: g[3],
        vz: gvz,
        mass: g[4],
        size_um: g[5],
        ttl,
    })
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

/// Turn user text into a safe file name stem, or `None` if nothing is left.
///
/// Keeps letters, digits, space, `-` and `_`. Everything else (path
/// separators, dots, control characters) is removed, so a name can never
/// leave the `saves/` folder.
pub fn sanitize_save_name(raw: &str) -> Option<String> {
    let s: String = raw
        .trim()
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | ' '))
        .take(SAVE_NAME_MAX)
        .collect();
    let s = s.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn save_path(clean_name: &str) -> PathBuf {
    PathBuf::from(SAVE_DIR).join(format!("{clean_name}{SAVE_EXT}"))
}

/// Does a save with this (already clean) name exist?
pub fn save_exists(clean_name: &str) -> bool {
    save_path(clean_name).is_file()
}

/// Write `file` to `saves/<name>.nbody.json`.
///
/// The data goes to a temporary file first and is renamed at the end, so a
/// crash or a full disk cannot destroy an older save with the same name.
pub fn write_save(name: &str, file: &SaveFile) -> Result<PathBuf, String> {
    let clean = sanitize_save_name(name).ok_or("the name is empty")?;
    std::fs::create_dir_all(SAVE_DIR).map_err(|e| format!("cannot make folder: {e}"))?;
    let path = save_path(&clean);
    let tmp = path.with_extension("json.tmp");
    {
        let out = std::fs::File::create(&tmp).map_err(|e| format!("cannot write file: {e}"))?;
        let mut out = std::io::BufWriter::new(out);
        serde_json::to_writer(&mut out, file).map_err(|e| format!("cannot encode save: {e}"))?;
        std::io::Write::flush(&mut out).map_err(|e| format!("cannot write file: {e}"))?;
    }
    std::fs::rename(&tmp, &path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("cannot finish file: {e}")
    })?;
    Ok(path)
}

/// Read and parse `saves/<name>.nbody.json`. Does not change any state.
pub fn read_save(name: &str) -> Result<SaveFile, String> {
    let clean = sanitize_save_name(name).ok_or("the name is empty")?;
    let bytes = std::fs::read(save_path(&clean)).map_err(|e| format!("cannot read file: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("the file is damaged: {e}"))
}

/// Remove `saves/<name>.nbody.json`.
pub fn delete_save(name: &str) -> Result<(), String> {
    let clean = sanitize_save_name(name).ok_or("the name is empty")?;
    std::fs::remove_file(save_path(&clean)).map_err(|e| format!("cannot delete file: {e}"))
}

/// One row of the save list. Only file metadata is read, so listing stays
/// fast even when a save is many megabytes.
#[derive(Clone, Debug)]
pub struct SaveEntry {
    pub name: String,
    pub bytes: u64,
    pub modified: Option<SystemTime>,
}

/// All saves in `SAVE_DIR`, newest first. A missing folder gives an empty list.
pub fn list_saves() -> Vec<SaveEntry> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(SAVE_DIR) else {
        return out;
    };
    for ent in rd.flatten() {
        let fname = ent.file_name().to_string_lossy().into_owned();
        let Some(stem) = fname.strip_suffix(SAVE_EXT) else {
            continue;
        };
        let Ok(meta) = ent.metadata() else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        out.push(SaveEntry {
            name: stem.to_string(),
            bytes: meta.len(),
            modified: meta.modified().ok(),
        });
    }
    out.sort_by(|a, b| {
        b.modified
            .cmp(&a.modified)
            .then_with(|| a.name.cmp(&b.name))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_app() -> AppSave {
        AppSave {
            cam: [1.5, -2.5],
            cam_z: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            zoom: 0.7,
            speed: 1.0,
            paused: false,
            seed: "0000000000000abc".to_string(),
            body_count: 0,
            preset: "solar".to_string(),
            systems: 1,
            library_preset: None,
            selected: None,
            follow: false,
            show_orbits: true,
            show_debris_orbits: false,
            aging_mult: 1.0,
        }
    }

    #[test]
    fn solar_system_round_trips_exactly() {
        let mut a = Bodies::with_capacity(64);
        spawn_solar_system(&mut a);
        let file = capture_save(&a, test_app()).unwrap();
        let json = serde_json::to_string(&file).unwrap();
        let back: SaveFile = serde_json::from_str(&json).unwrap();

        let mut c = Bodies::with_capacity(1);
        restore_save(&mut c, &back).unwrap();

        assert_eq!(a.len(), c.len());
        for i in 0..a.len() {
            assert_eq!(a.id[i], c.id[i]);
            assert_eq!(a.kind[i], c.kind[i]);
            assert_eq!(a.origin[i], c.origin[i]);
            assert_eq!(a.name[i], c.name[i]);
            assert_eq!(a.x[i].to_bits(), c.x[i].to_bits());
            assert_eq!(a.y[i].to_bits(), c.y[i].to_bits());
            assert_eq!(a.vx[i].to_bits(), c.vx[i].to_bits());
            assert_eq!(a.vy[i].to_bits(), c.vy[i].to_bits());
            assert_eq!(a.z[i].to_bits(), c.z[i].to_bits());
            assert_eq!(a.vz[i].to_bits(), c.vz[i].to_bits());
            assert_eq!(a.m[i].to_bits(), c.m[i].to_bits());
        }
        assert!(c.next_id >= a.next_id);
        assert_eq!(a.rng.s, c.rng.s);
        assert_eq!(a.stats.energy0.to_bits(), c.stats.energy0.to_bits());
    }

    #[test]
    fn newer_version_is_refused_and_state_is_kept() {
        let mut a = Bodies::with_capacity(64);
        spawn_solar_system(&mut a);
        let mut file = capture_save(&a, test_app()).unwrap();
        file.version = SAVE_VERSION + 1;
        let n = a.len();
        assert!(restore_save(&mut a, &file).is_err());
        assert_eq!(a.len(), n);
    }

    #[test]
    fn duplicate_ids_are_refused() {
        let mut a = Bodies::with_capacity(64);
        spawn_solar_system(&mut a);
        let mut file = capture_save(&a, test_app()).unwrap();
        let first = file.sim.bodies[0].id;
        file.sim.bodies[1].id = first;
        assert!(restore_save(&mut a, &file).is_err());
    }

    #[test]
    fn dead_bodies_are_not_saved() {
        let mut a = Bodies::with_capacity(64);
        spawn_solar_system(&mut a);
        let n = a.len();
        a.dead[1] = true;
        let file = capture_save(&a, test_app()).unwrap();
        assert_eq!(file.sim.bodies.len(), n - 1);
    }

    #[test]
    fn names_cannot_leave_the_save_folder() {
        assert_eq!(sanitize_save_name("  my save-1 "), Some("my save-1".into()));
        assert_eq!(
            sanitize_save_name("../../etc/passwd"),
            Some("etcpasswd".into())
        );
        assert_eq!(sanitize_save_name("a/b\\c.d"), Some("abcd".into()));
        assert_eq!(sanitize_save_name("..."), None);
        assert_eq!(sanitize_save_name("   "), None);
    }
}
