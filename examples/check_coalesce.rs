use nbody_rust::physics::*;
fn main() {
    let mut h: u64 = 1469598103934665603;
    let (mut co, mut im, mut me, mut nb) = (0u64, 0u64, 0u64, 0usize);
    for (s, m2, off) in [(1.0, 4.0, 0.0), (2.0, 4.0, 0.5), (3.0, 1.0, 1.0), (4.0, 8.0, 0.3), (6.0, 2.0, 0.0), (2.5, 4.0, 1.5)] {
        let mut b = Bodies::with_capacity(400);
        let i = b.push(-5.0, 0.0, 0.0, 0.0, 4.0, BodyKind::Planet); b.vx[i] = s * 0.5;
        let j = b.push(5.0, off, 0.0, 0.0, m2, BodyKind::Planet); b.vx[j] = -s * 0.5;
        let hst = b.push(0.0, 60.0, 0.0, 0.0, 3000.0, BodyKind::Star); let _ = hst;
        for _ in 0..30000 { advance(&mut b, DT); }
        for k in 0..b.len() { for v in [b.x[k], b.y[k], b.z[k], b.vx[k], b.vy[k], b.m[k]] { h = (h ^ v.to_bits()).wrapping_mul(1099511628211); } }
        co += b.stats.coalesces; im += b.stats.impacts; me += b.stats.merges; nb += b.len();
    }
    println!("bodies={nb} coalesces={co} impacts={im} merges={me} hash={h:016x}");
}
