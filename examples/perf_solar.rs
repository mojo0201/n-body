use nbody_rust::physics::*;
use std::time::Instant;
fn main() {
    let p = solar_system_preset();
    let mut b = Bodies::with_capacity(4096);
    spawn_from_preset(&mut b, &p);
    let grains: usize = b.dust_fields.iter().map(|f| f.grains.len()).sum();
    println!("{} bodies, {} grains", b.len(), grains);
    for _ in 0..50 { advance(&mut b, DT); }
    let steps = 3000;
    let t = Instant::now();
    for _ in 0..steps { advance(&mut b, DT); }
    let ms = t.elapsed().as_secs_f64() / steps as f64 * 1e3;
    println!("{:.3} ms/step; {:.0} steps/s; sim-years/s at 1 step/frame-step: {:.2}", ms, 1e3/ms, (1e3/ms) * DT / 7.3);
    println!("max_rung {}, last_nsub {}", b.stats.max_rung, b.stats.last_nsub);
}
