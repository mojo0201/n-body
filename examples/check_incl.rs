//! Step 5 check: inclination statistics of random systems, and that they run.
use nbody_rust::physics::*;

fn main() {
    println!("scale = {}", inclination_scale());
    let mut by_origin: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    for seed in [1u128, 2, 3, 7, 11, 42, 99, 1234, 5, 8, 13, 21] {
        let mut b = Bodies::with_capacity(400);
        spawn_system(&mut b, 400, seed);
        let Some(p) = primary_index(&b) else { continue };
        for i in 0..b.len() {
            if i == p || !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
                continue;
            }
            let (x, y, z) = (b.x[i] - b.x[p], b.y[i] - b.y[p], b.z[i] - b.z[p]);
            let (vx, vy, vz) = (b.vx[i] - b.vx[p], b.vy[i] - b.vy[p], b.vz[i] - b.vz[p]);
            let hz = x * vy - y * vx;
            let hx = y * vz - z * vy;
            let hy = z * vx - x * vz;
            let h = (hx * hx + hy * hy + hz * hz).sqrt();
            if h > 0.0 {
                let inc = (hz / h).clamp(-1.0, 1.0).acos().to_degrees();
                by_origin.entry(format!("{:?}", b.origin[i])).or_default().push(inc);
            }
        }
        // does it run?
        for _ in 0..500 {
            advance(&mut b, DT);
        }
        assert!((0..b.len()).all(|i| b.x[i].is_finite() && b.z[i].is_finite()), "NaN, seed {seed}");
    }
    for (k, v) in by_origin {
        let n = v.len() as f64;
        let rms = (v.iter().map(|x| x * x).sum::<f64>() / n).sqrt();
        let max = v.iter().cloned().fold(0.0, f64::max);
        println!("{k:>16}: n={:>4} rms inc {rms:6.2} deg  max {max:6.1}", v.len());
    }
}
