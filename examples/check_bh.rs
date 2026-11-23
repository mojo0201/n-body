use nbody_rust::physics::*;
fn main() {
    let mut b = Bodies::with_capacity(20000);
    spawn_random_systems(&mut b, 100, 5, 60);
    for _ in 0..120 { advance(&mut b, DT); }
    let mut h: u64 = 1469598103934665603;
    for i in 0..b.len() { for v in [b.x[i], b.y[i], b.z[i], b.vx[i], b.vy[i], b.vz[i]] { h = (h ^ v.to_bits()).wrapping_mul(1099511628211); } }
    println!("n={} hash={h:016x}", b.len());
}
