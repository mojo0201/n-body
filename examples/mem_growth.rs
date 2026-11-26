use nbody_rust::physics::*;
fn rss() -> f64 { let s = std::fs::read_to_string("/proc/self/statm").unwrap(); s.split_whitespace().nth(1).unwrap().parse::<f64>().unwrap() * 4096.0 / 1048576.0 }
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let c: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let chunks: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let mut b = Bodies::with_capacity(1024);
    spawn_random_systems(&mut b, c, 5, n);
    println!("spawned {} bodies rss {:.0} MB", b.len(), rss());
    let t = std::time::Instant::now();
    for k in 0..chunks {
        for _ in 0..2000 { advance(&mut b, DT); }
        let grains: usize = b.dust_fields.iter().map(|f| f.grains.len()).sum();
        println!("t={:.1}yr step {} bodies {} grains {} events {} rss {:.0} MB  ({:.1}s)", b.time/7.3, (k+1)*2000, b.len(), grains, b.events.len(), rss(), t.elapsed().as_secs_f64());
    }
}
