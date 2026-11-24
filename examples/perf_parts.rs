use nbody_rust::physics::*;
use std::time::Instant;
fn time(b: &mut Bodies, steps: usize) -> f64 {
    for _ in 0..5 { advance(b, DT); }
    let t = Instant::now();
    for _ in 0..steps { advance(b, DT); }
    t.elapsed().as_secs_f64() / steps as f64 * 1e3
}
fn main() {
    nbody_rust::physics::set_early_evolution(false);
    let mut b = Bodies::with_capacity(20000);
    spawn_random_systems(&mut b, 100, 5, 100);
    let grains: usize = b.dust_fields.iter().map(|f| f.grains.len()).sum();
    println!("{} bodies, {} dust fields, {} grains", b.len(), b.dust_fields.len(), grains);
    let mut c = b.clone();
    println!("full:           {:.1} ms/step", time(&mut c, 60));
    let mut c = b.clone(); c.dust_fields.clear();
    println!("no dust:        {:.1} ms/step", time(&mut c, 60));
    let mut c = b.clone(); c.dust_fields.clear();
    let keep: Vec<usize> = (0..c.len()).filter(|&i| !(c.origin[i].is_belt() || c.origin[i].is_rubble())).collect();
    println!("non-belt bodies: {} of {}", keep.len(), c.len());
    println!("rung max {} / dtl min {:.2e}", c.stats.max_rung, c.dtl.iter().cloned().fold(f64::MAX, f64::min));
    let mut rungs = [0usize; 8];
    for &r in &c.rung { rungs[(r as usize).min(7)] += 1; }
    println!("bodies per rung: {:?}", rungs);
}
