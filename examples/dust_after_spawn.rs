use nbody_rust::physics::*;
fn grains(b: &Bodies) -> usize { b.dust_fields.iter().map(|f| f.grains.len()).sum() }
fn main() {
    nbody_rust::physics::set_early_evolution(false);
    for seed in [3u128, 5, 7, 8, 21, 33] {
        let mut b = Bodies::with_capacity(1100);
        spawn_random_systems(&mut b, 1000, seed, 1);
        let mut line = format!("seed {seed:>2} n={:>4} fields={} grains {:>6}", b.len(), b.dust_fields.len(), grains(&b));
        for _ in 0..4 {
            for _ in 0..1500 {
                if let Some(mut job) = advance_begin(&mut b, DT) {
                    let mut calls = 0;
                    while !advance_run(&mut job, &mut b, || { calls += 1; calls % 3 != 0 }) {}
                }
            }
            line += &format!(" -> {}", grains(&b));
        }
        println!("{line} n={}", b.len());
    }
}
