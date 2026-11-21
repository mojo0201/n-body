//! Determinism baseline. Run on main, save the output, run again on the 3D
//! branch after each step, and diff. With z = 0 everywhere the lines must be
//! identical.
use nbody_rust::physics::*;

fn main() {
    // The baseline pins the planar (coplanar) physics.
    set_inclination_scale(0.0);
    for seed in [1u128, 2, 3, 7, 11, 42, 99, 1234] {
        let mut b = Bodies::with_capacity(400);
        spawn_system(&mut b, 400, seed);
        for _ in 0..2000 {
            advance(&mut b, DT);
        }
        let mut h: u64 = 0xcbf29ce484222325;
        for i in 0..b.len() {
            for v in [b.x[i], b.y[i], b.vx[i], b.vy[i], b.m[i]] {
                h ^= v.to_bits();
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        println!("seed {seed:>5}  n {:>4}  hash {h:016x}", b.len());
    }
}
