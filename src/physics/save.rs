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
pub const SAVE_VERSION: u32 = 1;
/// Folder for save files, relative to the working directory.
pub const SAVE_DIR: &str = "saves";
/// End of every save file name.
pub const SAVE_EXT: &str = ".nbody.json";
/// Upper limit on bodies in one file. Protects against a damaged file.
pub const SAVE_MAX_BODIES: usize = 2_000_000;
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
    pub grains: Vec<[f64; 7]>,
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
    let n = b.len();
    // old index -> new index (usize::MAX for dead bodies).
    let mut remap = vec![usize::MAX; n];
    let mut bodies: Vec<BodySave> = Vec::with_capacity(n);
    for i in 0..n {
        if b.dead[i] {
            continue;
        }
        let what = format!("body #{}", b.id[i]);
        need_finite(
            &what,
            &[
                b.x[i],
                b.y[i],
                b.vx[i],
                b.vy[i],
                b.m[i],
                b.r[i],
                b.bulk_density[i],
                b.spin[i],
                b.age_gyr[i],
                b.m_init[i],
            ],
        )?;
        remap[i] = bodies.len();
        bodies.push(BodySave {
            id: b.id[i],
            name: b.name[i].clone(),
            kind: kind_to_str(b.kind[i]).to_string(),
            origin: origin_to_str(b.origin[i]).to_string(),
            class: class_to_str(b.class[i]).to_string(),
            x: b.x[i],
            y: b.y[i],
            vx: b.vx[i],
            vy: b.vy[i],
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
        });
    }

    let mut dust = Vec::with_capacity(b.dust_fields.len());
    for f in &b.dust_fields {
        let primary = match remap.get(f.primary) {
            Some(&p) if p != usize::MAX => p,
            _ => continue, // The primary is gone: the field has no anchor.
        };
        need_finite("a dust field", &[f.extent_gm])?;
        let mut grains = Vec::with_capacity(f.grains.len());
        for g in &f.grains {
            need_finite("a dust grain", &[g.x, g.y, g.vx, g.vy, g.mass, g.size_um])?;
            let ttl = if g.ttl.is_finite() {
                g.ttl as f64
            } else {
                f64::MAX
            };
            grains.push([g.x, g.y, g.vx, g.vy, g.mass, g.size_um, ttl]);
        }
        dust.push(DustSave {
            primary,
            extent_gm: f.extent_gm,
            version: f.version,
            grains,
        });
    }

    let events = b
        .events
        .iter()
        .map(|e| EventSave {
            time: finite_or(e.time, 0.0),
            kind: event_to_str(e.kind).to_string(),
            text: e.text.clone(),
            x: finite_or(e.x, 0.0),
            y: finite_or(e.y, 0.0),
        })
        .collect();

    need_finite(
        "the simulation",
        &[
            b.time,
            b.metallicity,
            b.evolution_rate,
            b.stats.energy0,
            b.stats.energy_offset,
        ],
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
            stats: StatsSave {
                merges: b.stats.merges,
                impacts: b.stats.impacts,
                tidal: b.stats.tidal,
                escaped: b.stats.escaped,
                steps: b.stats.steps,
                last_nsub: b.stats.last_nsub,
                max_rung: b.stats.max_rung,
                energy0: b.stats.energy0,
                energy_offset: b.stats.energy_offset,
                coalesces: b.stats.coalesces,
                bounces: b.stats.bounces,
            },
            bodies,
            dust,
            events,
        },
        app,
    })
}

/// Replace `b` with the state in `f`.
///
/// The file is checked while a new `Bodies` is built. If anything is wrong,
/// the function returns an error and `b` is **not** changed. After the swap,
/// accelerations are recomputed once. The energy baseline from the file is
/// kept, so the drift display continues where it stopped.
pub fn restore_save(b: &mut Bodies, f: &SaveFile) -> Result<(), String> {
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
    if s.bodies.len() > SAVE_MAX_BODIES {
        return Err(format!("the save has too many bodies ({})", s.bodies.len()));
    }
    need_finite(
        "the simulation",
        &[
            s.time,
            s.metallicity,
            s.stats.energy0,
            s.stats.energy_offset,
        ],
    )?;

    let mut nb = Bodies::with_capacity(s.bodies.len());
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
        let kind = kind_from_str(&bs.kind)?;
        let i = nb.push(bs.x, bs.y, bs.vx, bs.vy, bs.m, kind);
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
    }

    nb.next_id = s.next_id.max(max_id.saturating_add(1));
    nb.time = s.time;
    nb.metallicity = s.metallicity;
    nb.rng = XorShift {
        s: s.rng_state.max(1),
    };
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

    let epoch = LOAD_EPOCH.fetch_add(1, AtomicOrdering::Relaxed);
    for d in &s.dust {
        if d.primary >= nb.len() {
            return Err("a dust field points to a body that is not in the save".to_string());
        }
        need_finite("a dust field", &[d.extent_gm])?;
        let mut grains = Vec::with_capacity(d.grains.len());
        for g in &d.grains {
            need_finite("a dust grain", &g[..6])?;
            if !g[6].is_finite() {
                return Err("a dust grain has a lifetime that is not valid".to_string());
            }
            let ttl = if g[6] >= f32::MAX as f64 {
                f32::INFINITY
            } else {
                g[6] as f32
            };
            grains.push(Grain {
                x: g[0],
                y: g[1],
                vx: g[2],
                vy: g[3],
                mass: g[4],
                size_um: g[5],
                ttl,
            });
        }
        nb.dust_fields.push(DustField {
            primary: d.primary,
            extent_gm: d.extent_gm,
            grains,
            // The renderer compares versions to find changed fields.
            version: (epoch << 32) | (d.version & 0xFFFF_FFFF),
        });
    }

    nb.parent_cache_valid = false;
    compute_accel(&mut nb);
    *b = nb;
    Ok(())
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
