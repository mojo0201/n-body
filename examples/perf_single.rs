//! usage: perf_single [bodies] [seed] [steps]  — ms/step on one generated system
use nbody_rust::physics::*;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bodies: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let seed: u128 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);
    let steps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000);
    set_early_evolution(false);
    let mut b = Bodies::with_capacity(bodies * 2);
    spawn_system_with_info(&mut b, bodies, seed);
    let grains: usize = b.dust_fields.iter().map(|f| f.grains.len()).sum();
    println!("{} bodies, {} grains", b.len(), grains);
    for _ in 0..20 { advance(&mut b, DT); }
    rebase_energy(&mut b);
    let l0 = total_angular_momentum(&b);
    let t = Instant::now();
    for _ in 0..steps { advance(&mut b, DT); }
    let mut rungs = [0usize; 12];
    for &r in &b.rung { rungs[(r as usize).min(11)] += 1; }
    println!("rung histogram {:?}, last_nsub {}, max_rung {}", rungs, b.stats.last_nsub, b.stats.max_rung);
    println!("energy drift {:?} %, L drift {:.2e} rel", energy_drift_pct(&b), (total_angular_momentum(&b) - l0) / l0);
    println!("{:.3} ms/step ({} steps)", t.elapsed().as_secs_f64() / steps as f64 * 1e3, steps);
}
