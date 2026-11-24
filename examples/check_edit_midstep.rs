//! Edits while a physics step is paused must not panic or kick the system.
use nbody_rust::physics::*;
fn momentum(b: &Bodies) -> (f64, f64) {
    (0..b.len()).fold((0.0, 0.0), |p, i| (p.0 + b.m[i] * b.vx[i], p.1 + b.m[i] * b.vy[i]))
}
fn main() {
    nbody_rust::physics::set_early_evolution(false);
    for (settle, seed) in [(true, 3u128), (false, 3), (true, 5), (false, 5)] {
        let mut b = Bodies::with_capacity(1200);
        spawn_random_systems(&mut b, 1000, seed, 1);
        let mut paused_mid = 0;
        for round in 0..40 {
            if let Some(mut job) = advance_begin(&mut b, DT) {
                let mut calls = 0;
                let done = advance_run(&mut job, &mut b, || { calls += 1; calls < 2 });
                if !done {
                    paused_mid += 1;
                    if settle { while !advance_run(&mut job, &mut b, || true) {} }
                    if round % 5 == 0 && b.len() > 50 { { let last = b.len() - 1; delete_body(&mut b, last); } }
                    if !settle { while !advance_run(&mut job, &mut b, || true) {} }
                }
            }
        }
        let (px, py) = momentum(&b);
        println!("settle={settle} seed={seed} n={} paused_mid={paused_mid} |p|={:.3e}", b.len(), px.hypot(py));
    }
}
