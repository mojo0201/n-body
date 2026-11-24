//! Long-run stability sweep over random systems.
//!
//! usage: cargo run --release --example stability_sweep -- [seeds] [years] [bodies] [first_seed]
//!   seeds       number of systems (default 200)
//!   years       simulated time per system (default 1e5)
//!   bodies      the "bodies" slider (default 14: planets plus a little debris)
//!   first_seed  first seed (default 1)
//!
//! Cost: about 0.5 to 20 CPU-seconds per simulated year for the busiest systems
//! (tight chains with day-long periods), so budget `seeds x years` accordingly;
//! 200 systems x 1000 yr is an hours-long run, 100 x 100 yr takes minutes. Short
//! runs find fast instabilities (planets too close for their masses); slow
//! secular ones need far longer runs. Progress goes to stderr, and every system
//! is written to `stability_report.csv`.
//!
//! Each system runs alone. A planet counts as lost when its id is gone: ejected,
//! merged into another body, or swallowed by the star. The report groups the
//! systems by architecture and lists the worst, with their seeds to replay.
use nbody_rust::physics::*;
use rayon::prelude::*;

const CHECKPOINTS: usize = 20;

struct Outcome {
    seed: u128,
    n_planets: usize,
    lost: usize,
    first_loss_frac: Option<f64>,
    escaped: u64,
    merges: u64,
    drift_pct: Option<f64>,
    tags: Vec<&'static str>,
    star_m_sun: f64,
}

fn planets(b: &Bodies) -> Vec<(u64, f64, f64)> {
    // (id, mass, distance to the primary in AU) of spawned planets of at least 0.1 M⊕.
    let Some(p) = primary_index(b) else { return vec![] };
    let au = au_in_gm();
    (0..b.len())
        .filter(|&i| b.kind[i] == BodyKind::Planet && b.origin[i] == BodyOrigin::Spawned && b.m[i] >= 0.1)
        .map(|i| {
            let d = (b.x[i] - b.x[p]).hypot(b.y[i] - b.y[p]).hypot(b.z[i] - b.z[p]);
            (b.id[i], b.m[i], d / au)
        })
        .collect()
}

fn run_one(seed: u128, bodies: usize, sim_time: f64) -> Outcome {
    let mut b = Bodies::with_capacity(bodies + 16);
    let info = spawn_system_with_info(&mut b, bodies, seed);
    let start = planets(&b);
    let star_m_sun = primary_index(&b).map_or(0.0, |p| b.m[p] / M_SUN_ME);
    let mut tags = Vec::new();
    if info.binary.is_some() {
        tags.push("binary");
    }
    if start.iter().any(|&(_, m, d)| m >= 50.0 && d < 0.1) {
        tags.push("hot Jupiter");
    }
    if start.iter().any(|&(_, m, d)| m >= 95.0 && d >= 0.1) {
        tags.push("cold giant");
    }
    if start.iter().filter(|&&(_, _, d)| d < 1.0).count() >= 3 {
        tags.push("compact chain");
    }
    if start.len() <= 1 {
        tags.push("0-1 planets");
    }
    if kind_of_primary(&b) != BodyKind::Star {
        tags.push("remnant host");
    }
    let ids: Vec<u64> = start.iter().map(|p| p.0).collect();
    let chunk = sim_time / CHECKPOINTS as f64;
    let mut first_loss = None;
    let mut t = 0.0;
    for c in 0..CHECKPOINTS {
        let end = (c + 1) as f64 * chunk;
        while t < end {
            advance(&mut b, DT);
            t += DT;
        }
        if first_loss.is_none() {
            let now = planets(&b);
            if ids.iter().any(|id| !now.iter().any(|p| p.0 == *id)) {
                first_loss = Some((c + 1) as f64 / CHECKPOINTS as f64);
            }
        }
    }
    let now = planets(&b);
    let lost = ids.iter().filter(|id| !now.iter().any(|p| p.0 == **id)).count();
    Outcome {
        seed,
        n_planets: ids.len(),
        lost,
        first_loss_frac: first_loss,
        escaped: b.stats.escaped,
        merges: b.stats.merges,
        drift_pct: energy_drift_pct(&b),
        tags,
        star_m_sun,
    }
}

fn write_csv(outs: &[Outcome]) {
    let mut csv = String::from("seed,star_m_sun,planets,lost,first_loss_frac,escaped,merges,drift_pct,tags\n");
    for o in outs {
        csv += &format!(
            "{},{:.3},{},{},{},{},{},{},\"{}\"\n",
            o.seed,
            o.star_m_sun,
            o.n_planets,
            o.lost,
            o.first_loss_frac.map_or(String::new(), |f| format!("{f:.2}")),
            o.escaped,
            o.merges,
            o.drift_pct.map_or(String::new(), |d| format!("{d:.3e}")),
            o.tags.join("; ")
        );
    }
    if let Err(e) = std::fs::write("stability_report.csv", csv) {
        eprintln!("could not write stability_report.csv: {e}");
    }
}

fn kind_of_primary(b: &Bodies) -> BodyKind {
    primary_index(b).map_or(BodyKind::Star, |p| b.kind[p])
}

fn main() {
    let arg = |k: usize, d: f64| std::env::args().nth(k).and_then(|s| s.parse::<f64>().ok()).unwrap_or(d);
    let n_seeds = arg(1, 200.0) as usize;
    let years = arg(2, 1e5);
    let bodies = arg(3, 14.0) as usize;
    let first = arg(4, 1.0) as u128;
    let sim_time = years * 3.15576e7 / time_unit_s();
    println!("{n_seeds} systems, {years:.0e} yr each ({sim_time:.0} time units), slider {bodies}");

    let done = std::sync::atomic::AtomicUsize::new(0);
    let outs: Vec<Outcome> = (first..first + n_seeds as u128)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&s| {
            let o = run_one(s, bodies, sim_time);
            let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            eprintln!("[{k}/{n_seeds}] seed {s}: {} planets, {} lost", o.n_planets, o.lost);
            o
        })
        .collect();
    write_csv(&outs);

    let with_planets: Vec<&Outcome> = outs.iter().filter(|o| o.n_planets > 0).collect();
    let unstable = with_planets.iter().filter(|o| o.lost > 0).count();
    let planets_total: usize = with_planets.iter().map(|o| o.n_planets).sum();
    let lost_total: usize = with_planets.iter().map(|o| o.lost).sum();
    println!("\nsystems with planets: {}", with_planets.len());
    println!("systems that lost a planet: {unstable} ({:.1}%)", 100.0 * unstable as f64 / with_planets.len().max(1) as f64);
    println!("planets lost: {lost_total} of {planets_total} ({:.1}%)", 100.0 * lost_total as f64 / planets_total.max(1) as f64);

    println!("\nby architecture (a system can be in several groups):");
    println!("{:<16} {:>8} {:>10} {:>12}", "group", "systems", "lost any", "planets lost");
    for tag in ["compact chain", "cold giant", "hot Jupiter", "binary", "remnant host", "0-1 planets"] {
        let g: Vec<&&Outcome> = with_planets.iter().filter(|o| o.tags.contains(&tag)).collect();
        if g.is_empty() {
            continue;
        }
        let any = g.iter().filter(|o| o.lost > 0).count();
        let pl: usize = g.iter().map(|o| o.n_planets).sum();
        let ls: usize = g.iter().map(|o| o.lost).sum();
        println!("{tag:<16} {:>8} {:>9.1}% {:>11.1}%", g.len(), 100.0 * any as f64 / g.len() as f64, 100.0 * ls as f64 / pl.max(1) as f64);
    }

    let mut drifts: Vec<f64> = outs.iter().filter_map(|o| o.drift_pct).map(f64::abs).collect();
    drifts.sort_by(|a, c| a.partial_cmp(c).unwrap());
    if !drifts.is_empty() {
        let q = |p: f64| drifts[((drifts.len() - 1) as f64 * p) as usize];
        println!("\n|energy drift| %: median {:.2e}, 90th {:.2e}, worst {:.2e}", q(0.5), q(0.9), drifts[drifts.len() - 1]);
    }

    let mut worst: Vec<&Outcome> = with_planets.iter().copied().filter(|o| o.lost > 0).collect();
    worst.sort_by(|a, c| (c.lost as f64 / c.n_planets as f64).partial_cmp(&(a.lost as f64 / a.n_planets as f64)).unwrap());
    println!("\nworst systems (replay with the same seed and slider):");
    println!("{:>8} {:>6} {:>5}/{:<5} {:>9} {:>8} {:>7}  {}", "seed", "M_sun", "lost", "n", "1st loss", "escaped", "merges", "tags");
    for o in worst.iter().take(12) {
        let fl = o.first_loss_frac.map_or("-".to_string(), |f| format!("{:.0}%", f * 100.0));
        println!("{:>8} {:>6.2} {:>5}/{:<5} {:>9} {:>8} {:>7}  {}", o.seed, o.star_m_sun, o.lost, o.n_planets, fl, o.escaped, o.merges, o.tags.join(", "));
    }
}
