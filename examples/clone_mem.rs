use nbody_rust::physics::*;
fn rss() -> f64 { let s = std::fs::read_to_string("/proc/self/statm").unwrap(); s.split_whitespace().nth(1).unwrap().parse::<f64>().unwrap() * 4096.0 / 1048576.0 }
fn main() {
    let mut b = Bodies::with_capacity(4096);
    spawn_from_preset(&mut b, &solar_system_preset());
    for _ in 0..200 { advance(&mut b, DT); }
    let grains: usize = b.dust_fields.iter().map(|f| f.grains.len()).sum();
    let est = b.len() * 520 + grains * 72 + b.events.len() * 120 + b.flashes.len() * 64 + 4096;
    let r0 = rss();
    let v: Vec<Bodies> = (0..50).map(|_| b.clone()).collect();
    let r1 = rss();
    println!("{} bodies {} grains: est {:.2} MB/snapshot, actual {:.2} MB/snapshot", b.len(), grains, est as f64/1048576.0, (r1-r0)/50.0);
    drop(v);
}
