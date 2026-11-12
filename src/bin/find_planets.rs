//! find_planets: search random-system seeds for the highest planet count.
//!
//! Searches seeds in parallel across all available cores. Reports the seed
//! that produces the most Spawned Planet + DwarfPlanet bodies across all
//! systems, alongside the total body count and binary-system count.
//!
//! This bypasses `spawn_random_systems` and calls `spawn_system_with_info`
//! directly for each system, using the same per-system seed derivation
//! (`seed + k * 0x9E37_79B9`). The result is bit-identical to spawning the
//! ring, but skips the ring geometry, `recenter_com`, `compute_accel`, and
//! `refresh_parent_cache` per seed — none of which affect the body count.

use std::env;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rayon::prelude::*;

use nbody_rust::physics::{self, Bodies, BodyKind, BodyOrigin};

const SEED_MULT: u32 = 0x9E37_79B9;

const USAGE: &str = "\
find_planets: search random-system seeds for the highest planet count

USAGE:
    find_planets [OPTIONS]

OPTIONS:
    --n N            max bodies per system (default 1000)
    --systems S      number of systems (default 1)
    --seeds K        number of seeds to try (default 1000)
    --start-seed M   first seed, base62 (1-6 chars, default 000000)
    --seed M         same as --start-seed
    --threads T      parallel workers (default = num_cpus)
    --min-planets P  stop early once a seed with >= P planets is found
    --quiet          suppress progress on stderr
    --json           emit one line of JSON on stdout
    -h, --help       show this help

REPORTS:
    seed             the winning seed
    planets          Spawned BodyKind::Planet bodies across all systems
    dwarfs           Spawned BodyKind::DwarfPlanet bodies
    total_bodies     actual body count after spawn (sums across systems)
    binaries         number of systems that formed a stellar binary

EXAMPLES:
    find_planets --n 1000 --systems 3 --seeds 1000
    find_planets --n 5000 --seeds 500 --json
    find_planets --n 500 --systems 1 --seeds 50000 --min-planets 8
";

struct Args {
    n: usize,
    systems: usize,
    seeds: u32,
    start_seed: u32,
    threads: usize,
    min_planets: Option<usize>,
    quiet: bool,
    json: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut out = Args {
            n: 1000,
            systems: 1,
            seeds: 1000,
            start_seed: 0,
            threads: num_cpus::get(),
            min_planets: None,
            quiet: false,
            json: false,
        };
        let mut it = env::args().skip(1);
        while let Some(a) = it.next() {
            match a.as_str() {
                "--n" => {
                    out.n = it
                        .next()
                        .ok_or("--n needs a value")?
                        .parse()
                        .map_err(|e| format!("--n: {e}"))?
                }
                "--systems" => {
                    out.systems = it
                        .next()
                        .ok_or("--systems needs a value")?
                        .parse()
                        .map_err(|e| format!("--systems: {e}"))?
                }
                "--seeds" => {
                    out.seeds = it
                        .next()
                        .ok_or("--seeds needs a value")?
                        .parse()
                        .map_err(|e| format!("--seeds: {e}"))?
                }
                "--start-seed" | "--seed" => {
                    let v = it.next().ok_or(format!("{a} needs a value"))?;
                    out.start_seed = physics::seed_from_base62(&v).ok_or(format!(
                        "{a}: invalid base62 seed '{v}' (1-6 chars of 0-9a-zA-Z)"
                    ))?
                }
                "--threads" => {
                    out.threads = it
                        .next()
                        .ok_or("--threads needs a value")?
                        .parse()
                        .map_err(|e| format!("--threads: {e}"))?
                }
                "--min-planets" => {
                    out.min_planets = Some(
                        it.next()
                            .ok_or("--min-planets needs a value")?
                            .parse()
                            .map_err(|e| format!("--min-planets: {e}"))?,
                    );
                }
                "--quiet" => out.quiet = true,
                "--json" => out.json = true,
                "-h" | "--help" => {
                    println!("{USAGE}");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument: {other}")),
            }
        }
        if out.n == 0 {
            return Err("--n must be >= 1".into());
        }
        if out.systems == 0 {
            return Err("--systems must be >= 1".into());
        }
        Ok(out)
    }
}

#[derive(Clone, Debug)]
struct Candidate {
    seed: u32,
    planets: usize,
    dwarfs: usize,
    total_bodies: usize,
    binaries: usize,
}

fn count_spawned(b: &Bodies) -> (usize, usize) {
    let mut planets = 0usize;
    let mut dwarfs = 0usize;
    for i in 0..b.len() {
        if b.origin[i] != BodyOrigin::Spawned {
            continue;
        }
        match b.kind[i] {
            BodyKind::Planet => planets += 1,
            BodyKind::DwarfPlanet => dwarfs += 1,
            _ => {}
        }
    }
    (planets, dwarfs)
}

/// Spawn one seed's systems and count. Returns the candidate.
/// Uses the exact per-system seed derivation from `spawn_random_systems`.
fn evaluate_seed(n: usize, systems: usize, seed: u32) -> Candidate {
    let mut planets = 0usize;
    let mut dwarfs = 0usize;
    let mut total_bodies = 0usize;
    let mut binaries = 0usize;
    let mut b = Bodies::with_capacity(0);
    for k in 0..systems {
        let sys_seed = seed.wrapping_add((k as u32).wrapping_mul(SEED_MULT));
        let info = physics::spawn_system_with_info(&mut b, n, sys_seed);
        let (p, d) = count_spawned(&b);
        planets += p;
        dwarfs += d;
        total_bodies += b.len();
        if info.binary.is_some() {
            binaries += 1;
        }
    }
    Candidate {
        seed,
        planets,
        dwarfs,
        total_bodies,
        binaries,
    }
}

/// Ranking key: fewer total planets is worse; more planets is better.
/// Ties broken by planets alone (planets beat dwarfs on a tie), then by
/// fewer total bodies (clearer signal).
fn better(a: &Candidate, b: &Candidate) -> bool {
    let a_total = a.planets + a.dwarfs;
    let b_total = b.planets + b.dwarfs;
    if a_total != b_total {
        return a_total > b_total;
    }
    if a.planets != b.planets {
        return a.planets > b.planets;
    }
    a.total_bodies < b.total_bodies
}

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };

    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global();

    let best: Mutex<Option<Candidate>> = Mutex::new(None);
    let done = AtomicU32::new(0);
    let found_min = AtomicU32::new(0);
    let start = Instant::now();
    let args_ref = &args;

    (0..args.seeds).into_par_iter().for_each(|k| {
        // Early exit if we already hit the --min-planets floor.
        if args_ref.min_planets.is_some() && found_min.load(Ordering::Relaxed) != 0 {
            return;
        }
        let seed = args_ref.start_seed.wrapping_add(k);
        let cand = evaluate_seed(args_ref.n, args_ref.systems, seed);

        // Hit the floor? Signal the others to stop.
        if let Some(min) = args_ref.min_planets {
            if cand.planets + cand.dwarfs >= min {
                found_min.store(1, Ordering::Relaxed);
            }
        }

        let mut guard = best.lock().unwrap();
        let is_better = guard.as_ref().is_none_or(|cur| better(&cand, cur));
        if is_better {
            *guard = Some(cand);
        }
        drop(guard);

        let count = done.fetch_add(1, Ordering::Relaxed) + 1;
        if !args_ref.quiet && (count.is_multiple_of(100) || count == args_ref.seeds) {
            let elapsed = start.elapsed().as_secs_f64().max(1e-9);
            let rate = count as f64 / elapsed;
            eprintln!("[{}/{}] {:.0} seeds/sec", count, args_ref.seeds, rate);
        }
    });

    let best = best.into_inner().unwrap();
    let Some(r) = best else {
        eprintln!("no seeds tried (check --seeds and --start-seed)");
        return ExitCode::from(1);
    };

    let elapsed = start.elapsed().as_secs_f64();
    if args.json {
        println!(
            "{{\"seed\":\"{}\",\"planets\":{},\"dwarfs\":{},\"total_bodies\":{},\"binaries\":{},\"systems\":{},\"n\":{},\"seeds_tried\":{},\"elapsed_secs\":{:.3}}}",
            physics::seed_to_base62(r.seed), r.planets, r.dwarfs, r.total_bodies, r.binaries,
            args.systems, args.n, done.load(Ordering::Relaxed), elapsed,
        );
    } else {
        println!("best seed:     {}", physics::seed_to_base62(r.seed));
        println!("  planets:     {}", r.planets);
        println!("  dwarfs:      {}", r.dwarfs);
        println!("  total bodies: {}", r.total_bodies);
        println!("  binaries:    {}/{}", r.binaries, args.systems);
        println!(
            "  searched:    {} seeds in {:.2}s ({:.0}/s)",
            done.load(Ordering::Relaxed),
            elapsed,
            done.load(Ordering::Relaxed) as f64 / elapsed.max(1e-9),
        );
    }
    ExitCode::SUCCESS
}
