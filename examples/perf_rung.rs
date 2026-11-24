use nbody_rust::physics::*;
use std::time::Instant;
fn main() {
    nbody_rust::physics::set_early_evolution(false);
    for (sys, seed) in [(5usize,5u128),(10,7),(20,11),(30,42)] {
        let mut b = Bodies::with_capacity(20000);
        spawn_random_systems(&mut b, 30, seed, sys);
        let n = b.len();
        let mut worst = 0f64; let mut tot = 0f64; let mut rm = 0u8;
        for _ in 0..200 { let t=Instant::now(); advance(&mut b, DT); let e=t.elapsed().as_secs_f64()*1e3; worst=worst.max(e); tot+=e; rm=rm.max(b.stats.max_rung);}
        println!("sys {sys} seed {seed}: n={n} mean {:.1} ms worst {:.1} ms max_rung {rm}", tot/200.0, worst);
    }
}
