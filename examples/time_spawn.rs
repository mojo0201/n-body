use nbody_rust::physics::*;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let c: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    if std::env::var("NOEARLY").is_ok() { set_early_evolution(false); }
    let mut b = Bodies::with_capacity(1024);
    let t = Instant::now();
    spawn_random_systems(&mut b, c, 5, n);
    let rss = std::fs::read_to_string("/proc/self/status").unwrap().lines().find(|l| l.starts_with("VmHWM")).unwrap().to_string();
    println!("{} systems x {}: {} bodies in {:.2}s  {}", n, c, b.len(), t.elapsed().as_secs_f64(), rss);
}
