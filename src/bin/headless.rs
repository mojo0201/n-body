//! Headless physics benchmark — no window, no GPU, no UI.
//!
//! Runs the same `physics::advance` loop as `main.rs`, but free-running
//! (no render thread, no throttling to 480·speed steps/sec). Reports the
//! raw throughput and derives the ratio the app actually cares about:
//!
//!   realtime_ratio = steps_per_sec / (480 * speed)
//!
//! At ratio >= 1.0 the app can run at real-time without the `(behind)`
//! flag ever turning on. Below 1.0 the sim falls behind by that factor —
//! e.g. ratio = 0.017 at n=5000 means the sim advances ~1.7 % of sim-time
//! per wall-second, which is exactly what `fps=8 physics_busy=100%` in the
//! GUI means.
//!
//! The reported `steps_per_sec` is the free-running throughput, not the
//! throttled rate. That is *deliberately different* from the GUI, whose
//! `steps_per_sec` equals min(480·speed, free-running). Free-running is
//! the right thing to measure: it tells you whether the app would keep up.
//!
//! Usage:
//!   cargo build --release --bin headless
//!   ./target/release/headless --n 1000 --seed 42 --secs 20
//!   ./target/release/headless --n 5000 --json
//!   ./target/release/headless --n 2000 --min-ratio 0.5 --quiet   # CI gate
//!
//! See scripts/sweep.sh for a table of runs.

use std::env;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use nbody_rust::physics::{self, Bodies, DT};
use nbody_rust::Preset;

const USAGE: &str = "\
headless physics benchmark — no window, no GPU, no UI

USAGE:
    headless [OPTIONS]

OPTIONS:
    --n N           max body count (default 1000)
    --seed S        RNG seed, base62, 1-6 chars of 0-9a-zA-Z (default 00000G)
    --secs T        wall-clock seconds to run (default 20)
    --speed X       sim speed multiplier (default 1.0)
    --solar         use Solar System preset
    --random        use Random preset (default)
    --quiet         suppress per-second progress lines
    --json          emit one line of JSON per second (for scripts)
    --min-ratio X   exit non-zero if realtime_ratio < X (CI gate)
    -h, --help      show this help

REPORTS:
    steps_per_sec    physics advances per wall-clock second
    realtime_ratio   steps_per_sec / (480 * speed); >= 1.0 = real-time
    avg_advance_us   mean time inside physics::advance, in microseconds

EXAMPLES:
    headless --n 1000 --secs 20
    headless --n 5000 --seed 00000G --secs 30 --json
    headless --solar --secs 10
    headless --n 2000 --secs 10 --min-ratio 0.5 --quiet
";

struct Args {
    n: usize,
    seed: u128,
    secs: f64,
    speed: f64,
    preset: Preset,
    quiet: bool,
    json: bool,
    /// If set, exit non-zero when the free-running realtime ratio falls
    /// below this threshold. Used by CI to catch physics regressions.
    min_ratio: Option<f64>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut out = Args {
            n: 1000,
            seed: 42,
            secs: 20.0,
            speed: 1.0,
            preset: Preset::Random,
            quiet: false,
            json: false,
            min_ratio: None,
        };
        let mut args = env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--n" => {
                    out.n = args
                        .next()
                        .ok_or("--n needs a value")?
                        .parse()
                        .map_err(|e| format!("--n: {}", e))?;
                }
                "--seed" => {
                    let v = args.next().ok_or("--seed needs a value")?;
                    out.seed = physics::seed_from_base62(&v)
                        .ok_or(format!("--seed: invalid base62 seed '{}'", v))?;
                }
                "--secs" => {
                    out.secs = args
                        .next()
                        .ok_or("--secs needs a value")?
                        .parse()
                        .map_err(|e| format!("--secs: {}", e))?;
                }
                "--speed" => {
                    out.speed = args
                        .next()
                        .ok_or("--speed needs a value")?
                        .parse()
                        .map_err(|e| format!("--speed: {}", e))?;
                }
                "--min-ratio" => {
                    out.min_ratio = Some(
                        args.next()
                            .ok_or("--min-ratio needs a value")?
                            .parse()
                            .map_err(|e| format!("--min-ratio: {}", e))?,
                    );
                }
                "--solar" => out.preset = Preset::SolarSystem,
                "--random" => out.preset = Preset::Random,
                "--quiet" => out.quiet = true,
                "--json" => out.json = true,
                "-h" | "--help" => {
                    println!("{}", USAGE);
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument: {}", other)),
            }
        }
        if out.n == 0 {
            return Err("--n must be >= 1".into());
        }
        Ok(out)
    }
}

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {}\n\n{}", e, USAGE);
            return ExitCode::from(2);
        }
    };

    // Match the app's thread-pool sizing so numbers are comparable,
    // including the `RAYON_NUM_THREADS` override that `main.rs` honours.
    let nthreads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(num_cpus::get_physical);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build_global();

    // --- spawn ---
    // `spawn_*` reassign `*b` internally, so the outer `Bodies::with_capacity`
    // is thrown away. Pass capacity 0 to make that obvious.
    let mut b = Bodies::with_capacity(0);
    match args.preset {
        Preset::Random => {
            let _ = physics::spawn_random_systems(&mut b, args.n, args.seed, 1);
        }
        Preset::SolarSystem => {
            physics::spawn_solar_systems(&mut b, 1);
        }
    }
    let n0 = b.len();

    // --- run (free-running) ---
    // No accumulator, no throttle, no sleep: call `advance` as fast as
    // wall-clock permits and report the raw throughput. The app's real-time
    // throttle (480 steps/sec × speed) is what makes GUI numbers unusable
    // for benchmarking — at n=100 the app has 88 % idle time and hides that
    // it could go 8× real-time. That headroom is exactly what this bench
    // is for.
    let target_steps_per_sec = 480.0 * args.speed;
    let start = Instant::now();
    let deadline = start + Duration::from_secs_f64(args.secs);
    let mut last_report = start;
    let mut steps = 0u64;
    let mut advance_ns = 0u128;

    while Instant::now() < deadline {
        let t0 = Instant::now();
        physics::advance(&mut b, DT);
        advance_ns += t0.elapsed().as_nanos();
        steps += 1;

        // One report line per wall-second. `--quiet` suppresses it
        // completely (both the JSON and the human-readable line).
        if !args.quiet && last_report.elapsed() >= Duration::from_secs(1) {
            let el = start.elapsed().as_secs_f64();
            let sps = steps as f64 / el;
            let ratio = sps / target_steps_per_sec;
            let avg_us = (advance_ns as f64 / steps.max(1) as f64) / 1000.0;
            if args.json {
                // Interim line: `"final":false`, so a script that
                // filters on final only keeps the summary.
                println!(
                    "{{\"final\":false,\"n\":{},\"n0\":{},\"seed\":\"{}\",\"speed\":{:.3},\"secs\":{:.3},\"steps\":{},\"steps_per_sec\":{:.2},\"realtime_ratio\":{:.4},\"avg_advance_us\":{:.2},\"max_rung\":{},\"last_nsub\":{},\"merges\":{},\"impacts\":{},\"tidal\":{},\"escaped\":{}}}",
                    b.len(), n0, physics::seed_to_base62(args.seed), args.speed, el, steps, sps, ratio, avg_us,
                    b.stats.max_rung,
                    b.stats.last_nsub,
                    b.stats.merges,
                    b.stats.impacts,
                    b.stats.tidal,
                    b.stats.escaped,
                );
            } else {
                println!(
                    "[progress] t={:.1}s n={} steps={} steps/sec={:.1} ratio={:.3} avg={:.1}µs",
                    el,
                    b.len(),
                    steps,
                    sps,
                    ratio,
                    avg_us
                );
            }
            last_report = Instant::now();
        }
    }

    // --- summary ---
    let el = start.elapsed().as_secs_f64();
    let sps = steps as f64 / el;
    let ratio = sps / target_steps_per_sec;
    let avg_us = (advance_ns as f64 / steps.max(1) as f64) / 1000.0;
    if args.json {
        println!(
            "{{\"final\":true,\"n\":{},\"n0\":{},\"seed\":\"{}\",\"speed\":{:.3},\"secs\":{:.3},\"steps\":{},\"steps_per_sec\":{:.2},\"realtime_ratio\":{:.4},\"avg_advance_us\":{:.2}}}",
            b.len(), n0, physics::seed_to_base62(args.seed), args.speed, el, steps, sps, ratio, avg_us
        );
    } else {
        println!(
            "[summary] n={} (from {}) seed={} speed={:.2} secs={:.1} steps={} steps/sec={:.1} ratio={:.3} avg={:.1}µs",
            b.len(), n0, physics::seed_to_base62(args.seed), args.speed, el, steps, sps, ratio, avg_us
        );
    }

    // --- CI gate ---
    if let Some(min) = args.min_ratio {
        if ratio < min {
            eprintln!("FAIL: realtime_ratio {:.3} < threshold {:.3}", ratio, min);
            return ExitCode::from(1);
        }
    }
    ExitCode::SUCCESS
}
