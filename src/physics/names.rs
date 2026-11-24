//! Systematic names for the bodies of a generated system.
//!
//! Real objects are mostly known by designation, not by name:
//!   * stars by catalogue (`HD 40307`, `GJ 667`, `Kepler-452`),
//!   * their planets by a letter in order (`b`, `c`, `d`),
//!   * asteroids and Kuiper belt objects by year and sequence (`2007 OR10`), the
//!     larger ones numbered as well (`(4179) 1989 AC`),
//!   * comets by type and year (`C/2023 A3`),
//!   * moons by a proper name once they are well known.
//!
//! The pass runs once, after a system is built, and draws from its own random
//! stream, so it never changes the physics or the other random draws.

use super::*;
use std::collections::HashSet;

const NAME_SEED_SALT: u128 = 0x4E41_4D45_5359_5354;

/// A moon or belt body at least this massive gets a proper name (Ceres-scale
/// and up, M⊕); smaller ones keep a designation.
const PROPER_NAME_MIN_M: f64 = 1e-4;
/// A belt body at least this massive also gets a minor-planet number.
const NUMBERED_MIN_M: f64 = 1e-6;

fn pick(rng: &mut XorShift, n: usize) -> usize {
    ((rng.next_f64() * n as f64) as usize).min(n.saturating_sub(1))
}

fn range(rng: &mut XorShift, lo: usize, hi: usize) -> usize {
    lo + pick(rng, hi - lo + 1)
}

/// Upper-case Roman numeral of `n` (1 and up).
pub fn roman(mut n: usize) -> String {
    const TABLE: [(usize, &str); 13] = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
        (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ];
    let mut out = String::new();
    for (v, s) in TABLE {
        while n >= v {
            out.push_str(s);
            n -= v;
        }
    }
    out
}

/// Planet letter for the `k`-th planet of a star (0 gives `b`): b..z, then aa, ab..
fn planet_letter(k: usize) -> String {
    let c = |i: usize| (b'b' + i as u8) as char;
    if k < 25 {
        c(k).to_string()
    } else {
        format!("{}{}", c((k - 25) / 25 % 25), c((k - 25) % 25))
    }
}

/// Right ascension and declination of survey names: `HHMM` and a signed
/// declination of `digits` digits (2 for `PSR J1257+12`, 3 for `WD 1145+017`,
/// 4 for `2MASS J1207-3932`).
fn sky_position(rng: &mut XorShift, dec_digits: usize) -> String {
    let (h, m) = (range(rng, 0, 23), range(rng, 0, 59));
    let sign = if rng.next_f64() < 0.5 { '+' } else { '-' };
    let dec = match dec_digits {
        2 => format!("{:02}", range(rng, 1, 89)),
        3 => format!("{:03}", range(rng, 1, 899)),
        _ => format!("{:02}{:02}", range(rng, 1, 89), range(rng, 0, 59)),
    };
    format!("{h:02}{m:02}{sign}{dec}")
}

/// Catalogue name of a system's host, picked by the kind of star.
fn host_designation(kind: BodyKind, m_sun: f64, n_planets: usize, rng: &mut XorShift) -> String {
    match kind {
        BodyKind::WhiteDwarf => format!("WD {}", sky_position(rng, 3)),
        BodyKind::NeutronStar => format!("PSR J{}", sky_position(rng, 2)),
        BodyKind::BlackHole => format!("Gaia BH{}", range(rng, 1, 9)),
        BodyKind::BrownDwarf => format!("2MASS J{}", sky_position(rng, 4)),
        _ => {
            let r = rng.next_f64();
            if m_sun < 0.6 {
                match r {
                    r if r < 0.55 => format!("GJ {}", range(rng, 100, 9000)),
                    r if r < 0.80 => format!("LHS {}", range(rng, 1, 6000)),
                    _ => format!("TOI-{}", range(rng, 100, 7000)),
                }
            } else {
                match r {
                    r if r < 0.55 => format!("HD {}", range(rng, 1000, 360_000)),
                    r if r < 0.80 => format!("HIP {}", range(rng, 100, 120_000)),
                    _ if n_planets >= 2 => format!("Kepler-{}", range(rng, 10, 2000)),
                    _ => format!("TOI-{}", range(rng, 100, 7000)),
                }
            }
        }
    }
}

/// Year, half-month letter, second letter and, sometimes, a cycle count:
/// `2007 OR10`. The letter I is never used, as in the real scheme.
fn provisional(rng: &mut XorShift) -> String {
    const LETTERS: &[u8] = b"ABCDEFGHJKLMNOPQRSTUVWXYZ";
    let year = range(rng, 1990, 2035);
    let l1 = LETTERS[pick(rng, 25)] as char;
    let l2 = LETTERS[pick(rng, 25)] as char;
    if rng.next_f64() < 0.4 {
        format!("{year} {l1}{l2}{}", range(rng, 1, 99))
    } else {
        format!("{year} {l1}{l2}")
    }
}

/// Comet designation: `C/2023 A3` (long period) or `P/2021 K1` (short period).
fn comet_designation(periodic: bool, rng: &mut XorShift) -> String {
    const LETTERS: &[u8] = b"ABCDEFGHJKLMNOPQRSTUVWXY";
    let year = range(rng, 1995, 2035);
    let letter = LETTERS[pick(rng, LETTERS.len())] as char;
    format!("{}/{year} {letter}{}", if periodic { 'P' } else { 'C' }, range(rng, 1, 9))
}

// ---- Proper names ----------------------------------------------------------

/// Mythological names the generator learns the sound of. All are names from
/// old myth, so new names built from them stay in that style.
const MYTH_NAMES: &[&str] = &[
    "Io", "Europa", "Ganymede", "Callisto", "Amalthea", "Himalia", "Elara", "Pasiphae", "Sinope",
    "Lysithea", "Carme", "Ananke", "Leda", "Thebe", "Adrastea", "Metis", "Mimas", "Enceladus",
    "Tethys", "Dione", "Rhea", "Titan", "Hyperion", "Iapetus", "Phoebe", "Janus", "Epimetheus",
    "Helene", "Telesto", "Calypso", "Atlas", "Prometheus", "Pandora", "Pan", "Daphnis", "Ariel",
    "Umbriel", "Titania", "Oberon", "Miranda", "Puck", "Triton", "Nereid", "Naiad", "Thalassa",
    "Despina", "Galatea", "Larissa", "Proteus", "Charon", "Nix", "Hydra", "Kerberos", "Styx",
    "Phobos", "Deimos", "Ceres", "Vesta", "Pallas", "Juno", "Hygiea", "Eunomia", "Psyche",
    "Iris", "Flora", "Hebe", "Astraea", "Metis", "Aurora", "Selene", "Helios", "Eos", "Nyx",
    "Erebus", "Gaia", "Uranus", "Cronus", "Rhea", "Theia", "Phoebe", "Themis", "Mnemosyne",
    "Hyperion", "Oceanus", "Tethys", "Atlas", "Hestia", "Demeter", "Persephone", "Athena",
    "Artemis", "Apollo", "Hermes", "Hephaestus", "Ares", "Aphrodite", "Dionysus", "Poseidon",
    "Odin", "Thor", "Freya", "Frigg", "Baldr", "Loki", "Tyr", "Heimdall", "Skadi", "Njord",
    "Ymir", "Bestla", "Surtur", "Mundilfari", "Skoll", "Hati", "Fenrir", "Greip", "Narvi",
    "Bergelmir", "Aegir", "Kari", "Fornjot", "Suttungr", "Thrymr", "Ijiraq", "Kiviuq", "Paaliaq",
    "Siarnaq", "Tarqeq", "Albiorix", "Erriapus", "Tarvos", "Bebhionn", "Eirene", "Danu", "Brigid",
    "Lugh", "Morrigan", "Cernunnos", "Epona", "Rhiannon", "Arawn", "Ogma", "Maeve", "Aengus",
    "Quetzal", "Tlaloc", "Huitzilin", "Itzel", "Coatl", "Inti", "Mama", "Viracocha", "Pachamama",
    "Amaterasu", "Tsukuyomi", "Susanoo", "Izanami", "Inari", "Ra", "Isis", "Osiris", "Horus",
    "Thoth", "Anubis", "Bastet", "Sekhmet", "Hathor", "Nephthys", "Marduk", "Ishtar", "Tiamat",
    "Enlil", "Nabu", "Shamash", "Anu", "Sedna", "Makemake", "Haumea", "Eris", "Orcus", "Quaoar",
    "Varuna", "Ixion", "Gonggong", "Salacia", "Varda", "Huya",
];

/// Second-order letter chain over `MYTH_NAMES`: from the last two letters to
/// the letters that followed them, with `'$'` at the end of a name.
struct NameChain {
    next: std::collections::HashMap<(char, char), Vec<char>>,
    starts: Vec<(char, char)>,
    known: HashSet<String>,
}

impl NameChain {
    fn new() -> Self {
        let mut next: std::collections::HashMap<(char, char), Vec<char>> = Default::default();
        let mut starts = Vec::new();
        for name in MYTH_NAMES {
            let cs: Vec<char> = name.to_lowercase().chars().collect();
            if cs.len() < 3 {
                continue;
            }
            starts.push((cs[0], cs[1]));
            for w in cs.windows(3) {
                next.entry((w[0], w[1])).or_default().push(w[2]);
            }
            let n = cs.len();
            next.entry((cs[n - 2], cs[n - 1])).or_default().push('$');
        }
        let known = MYTH_NAMES.iter().map(|s| s.to_lowercase()).collect();
        Self { next, starts, known }
    }

    /// One new name of 4 to 9 letters that is neither a training name nor in `used`.
    fn generate(&self, rng: &mut XorShift, used: &HashSet<String>) -> Option<String> {
        for _ in 0..60 {
            let (a, b) = self.starts[pick(rng, self.starts.len())];
            let mut cs = vec![a, b];
            while cs.len() < 10 {
                let n = cs.len();
                let Some(options) = self.next.get(&(cs[n - 2], cs[n - 1])) else { break };
                let c = options[pick(rng, options.len())];
                if c == '$' {
                    break;
                }
                cs.push(c);
            }
            if !(4..=9).contains(&cs.len()) {
                continue;
            }
            let s: String = cs.iter().collect();
            if self.known.contains(&s) || used.contains(&s) {
                continue;
            }
            let mut chars = s.chars();
            let first = chars.next()?.to_uppercase().collect::<String>();
            return Some(first + chars.as_str());
        }
        None
    }
}

// ---- The pass --------------------------------------------------------------

/// Give a name to every unnamed body of a freshly generated system. Bodies that
/// already have one (presets, rogue planets) are left alone.
pub fn name_system(b: &mut Bodies, seed: u128) {
    let mut rng = XorShift::new(seed ^ NAME_SEED_SALT);
    let chain = NameChain::new();
    let mut used: HashSet<String> = b.name.iter().flatten().map(|s| s.to_lowercase()).collect();

    let mut hosts: Vec<usize> = (0..b.len()).filter(|&i| is_host_kind(b.kind[i])).collect();
    hosts.sort_by(|&x, &y| b.m[y].total_cmp(&b.m[x]));
    if hosts.is_empty() {
        return;
    }
    let is_planet = |b: &Bodies, i: usize| {
        matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) && b.origin[i] == BodyOrigin::Spawned
    };
    // Parent of each spawned planet: a host star (a planet) or another planet (a moon).
    let parents: Vec<Option<usize>> = (0..b.len())
        .map(|i| if is_planet(b, i) { find_parent(b, i) } else { None })
        .collect();
    let n_planets = (0..b.len()).filter(|&i| is_planet(b, i) && parents[i].is_some_and(|p| is_host_kind(b.kind[p]))).count();

    name_hosts(b, &hosts, n_planets, &mut rng, &mut used);
    name_planets_and_moons(b, &hosts, &parents, &chain, &mut rng, &mut used);
    name_small_bodies(b, &chain, &mut rng, &mut used);
}

fn name_hosts(b: &mut Bodies, hosts: &[usize], n_planets: usize, rng: &mut XorShift, used: &mut HashSet<String>) {
    let lead = hosts[0];
    let base = match &b.name[lead] {
        Some(n) => n.clone(),
        None => {
            let base = host_designation(b.kind[lead], b.m[lead] / M_SUN_ME, n_planets, rng);
            used.insert(base.to_lowercase());
            base
        }
    };
    if hosts.len() == 1 {
        b.name[lead].get_or_insert(base);
        return;
    }
    for (k, &h) in hosts.iter().enumerate() {
        if b.name[h].is_none() {
            let suffix = (b'A' + k.min(25) as u8) as char;
            b.name[h] = Some(format!("{base} {suffix}"));
        }
    }
}

fn name_planets_and_moons(
    b: &mut Bodies,
    hosts: &[usize],
    parents: &[Option<usize>],
    chain: &NameChain,
    rng: &mut XorShift,
    used: &mut HashSet<String>,
) {
    let host_dist = |b: &Bodies, i: usize, h: usize| {
        (b.x[i] - b.x[h]).hypot(b.y[i] - b.y[h]).hypot(b.z[i] - b.z[h])
    };
    // Planets: a letter per host, in order of distance.
    for &h in hosts {
        let mut mine: Vec<usize> = (0..b.len())
            .filter(|&i| parents[i] == Some(h) && b.name[i].is_none())
            .collect();
        mine.sort_by(|&x, &y| host_dist(b, x, h).total_cmp(&host_dist(b, y, h)));
        let host_name = b.name[h].clone().unwrap_or_default();
        for (k, &i) in mine.iter().enumerate() {
            let name = format!("{host_name} {}", planet_letter(k));
            used.insert(name.to_lowercase());
            b.name[i] = Some(name);
        }
    }
    // Moons: a Roman numeral per planet by distance; the larger ones also get
    // a proper name, in place of the numeral.
    let planets: Vec<usize> = (0..b.len()).filter(|&p| parents.iter().any(|&q| q == Some(p))).collect();
    for p in planets {
        let mut moons: Vec<usize> = (0..b.len()).filter(|&i| parents[i] == Some(p) && b.name[i].is_none()).collect();
        moons.sort_by(|&x, &y| host_dist(b, x, p).total_cmp(&host_dist(b, y, p)));
        let planet_name = b.name[p].clone().unwrap_or_default();
        for (k, &i) in moons.iter().enumerate() {
            let designation = format!("{planet_name} {}", roman(k + 1));
            let proper = (b.m[i] >= PROPER_NAME_MIN_M).then(|| chain.generate(rng, used)).flatten();
            let name = match proper {
                Some(n) => {
                    used.insert(n.to_lowercase());
                    n
                }
                None => designation,
            };
            b.name[i] = Some(name);
        }
    }
}

/// Belts, comets, rubble and interlopers: designations, with numbers and
/// proper names for the most massive.
fn name_small_bodies(b: &mut Bodies, chain: &NameChain, rng: &mut XorShift, used: &mut HashSet<String>) {
    let mut next_number = range(rng, 1000, 60_000);
    for i in 0..b.len() {
        if b.name[i].is_some() || !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
            continue;
        }
        let origin = b.origin[i];
        if origin == BodyOrigin::Spawned || origin == BodyOrigin::RoguePlanet {
            continue;
        }
        let name = match origin {
            BodyOrigin::Comet => {
                let periodic = orbit_period_hint(b, i);
                unique(used, || comet_designation(periodic, rng))
            }
            BodyOrigin::Interstellar => unique(used, || format!("A/{} U{}", range(rng, 2017, 2035), range(rng, 1, 9))),
            _ => {
                let desig = unique(used, || provisional(rng));
                if origin.is_belt() && b.m[i] >= PROPER_NAME_MIN_M {
                    chain.generate(rng, used).unwrap_or(desig)
                } else if origin.is_belt() && b.m[i] >= NUMBERED_MIN_M {
                    next_number += range(rng, 1, 40);
                    format!("({next_number}) {desig}")
                } else {
                    desig
                }
            }
        };
        used.insert(name.to_lowercase());
        b.name[i] = Some(name);
    }
}

/// Short-period comets get `P/`; this uses the speed to decide whether the orbit is tight.
fn orbit_period_hint(b: &Bodies, i: usize) -> bool {
    primary_index(b).map_or(false, |p| {
        let o = orbit_info(b, i, p);
        o.bound && o.period * time_unit_s() / 3.15576e7 < 200.0
    })
}

/// Call `make` until it gives a name that is not in `used`.
fn unique(used: &HashSet<String>, mut make: impl FnMut() -> String) -> String {
    let mut s = make();
    for _ in 0..50 {
        if !used.contains(&s.to_lowercase()) {
            return s;
        }
        s = make();
    }
    s
}
