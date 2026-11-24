//! Body counts of random systems at a given slider value.
use nbody_rust::physics::*;
fn main() {
    nbody_rust::physics::set_early_evolution(false);
    let slider: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let mut ns = Vec::new();
    for seed in 1u128..=60 {
        let mut b = Bodies::with_capacity(slider + 16);
        spawn_system(&mut b, slider, seed);
        ns.push(b.len());
    }
    ns.sort();
    let med = ns[ns.len() / 2];
    let over200 = ns.iter().filter(|&&n| n >= 200).count();
    println!("slider={slider} min={} median={med} max={} >=200: {over200}/60", ns[0], ns[ns.len() - 1]);
}
