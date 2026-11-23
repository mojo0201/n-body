//! Drives tidal disruption (planar and 3D) and prints a state hash, to compare refactors.
use nbody_rust::physics::*;
fn main() {
    let mut h: u64 = 1469598103934665603;
    let (mut td, mut nb) = (0u64, 0usize);
    for (rp, z, vzf, m) in [(0.10, 0.0, 0.0, 1.0), (0.13, 0.0, 0.0, 3.0), (0.16, 0.1, 0.2, 2.0), (0.2, -0.1, -0.2, 5.0), (0.12, 0.0, 0.0, 0.5), (0.15, 0.1, 0.1, 10.0)] {
        let d = 0.4f64;
        let vp = (2.0 * G * 800.0 * (1.0 / rp - 1.0 / d)).sqrt();
        let vt = rp * vp / d;
        let vz = vt * vzf;
        let mut b = Bodies::with_capacity(400);
        let g = b.push(0.0, 0.0, 0.0, 0.0, 800.0, BodyKind::Planet);
        b.x[g] = 100.0;
        let i = b.push(100.0 + d, 0.0, 0.0, vt, m, BodyKind::Planet);
        b.z[i] = z; b.vz[i] = vz;
        for _ in 0..30000 { advance(&mut b, DT); }
        for k in 0..b.len() { for v in [b.x[k], b.y[k], b.z[k], b.vx[k], b.vy[k], b.vz[k], b.m[k], b.spin[k]] { h = (h ^ v.to_bits()).wrapping_mul(1099511628211); } }
        td += b.stats.tidal as u64; nb += b.len();
        eprintln!("case rp={rp} tidal={} n={} merges={}", b.stats.tidal, b.len(), b.stats.merges);
    }
    println!("tidal={td} bodies={nb} hash={h:016x}");
}
