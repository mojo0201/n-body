use nbody_rust::physics::*;
use std::time::Instant;
fn main() {
    for n_sys in [10usize, 50, 100, 200] {
        let t = Instant::now();
        let mut b = Bodies::with_capacity(20000);
        spawn_random_systems(&mut b, 100, 5, n_sys);
        let spawn = t.elapsed().as_secs_f64();
        let n = b.len();
        for _ in 0..20 { advance(&mut b, DT); } // warm
        let t = Instant::now();
        let steps = 100;
        for _ in 0..steps { advance(&mut b, DT); }
        let per = t.elapsed().as_secs_f64() / steps as f64 * 1e3;
        let t = Instant::now();
        let segs = compute_orbit_segments(&b, 0.05, true, true);
        let orb = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let c = b.clone();
        let clone = t.elapsed().as_secs_f64() * 1e3;
        drop(c);
        println!("{n_sys:>4} systems {n:>6} bodies: spawn {spawn:.2}s  step {per:.2} ms (x480/unit = {:.0} ms/unit)  orbits {orb:.1} ms ({} seg)  clone {clone:.1} ms  rungs max {}",
            per*480.0, segs.len(), b.stats.max_rung);
    }
}
