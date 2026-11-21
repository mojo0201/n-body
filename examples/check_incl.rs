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

    // Per-system spread: rms planet inclination for several single-star systems.
    let mut line = String::from("per-system rms (deg):");
    for seed in [1u128, 2, 3, 7, 11, 42, 99, 1234] {
        let mut b = Bodies::with_capacity(400);
        spawn_system(&mut b, 400, seed);
        let Some(p) = primary_index(&b) else { continue };
        let (mut s2, mut n) = (0.0, 0.0);
        for i in 0..b.len() {
            if i == p || b.kind[i] != BodyKind::Planet || b.origin[i] != BodyOrigin::Spawned { continue; }
            let (x, y, z) = (b.x[i] - b.x[p], b.y[i] - b.y[p], b.z[i] - b.z[p]);
            let (vx, vy, vz) = (b.vx[i] - b.vx[p], b.vy[i] - b.vy[p], b.vz[i] - b.vz[p]);
            let hz = x * vy - y * vx;
            let h = ((y * vz - z * vy).powi(2) + (z * vx - x * vz).powi(2) + hz * hz).sqrt();
            let inc = (hz / h).clamp(-1.0, 1.0).acos().to_degrees();
            s2 += inc * inc; n += 1.0;
        }
        if n > 0.0 { line += &format!(" {:.1}", (s2 / n).sqrt()); }
    }
    println!("{line}");
    // Several systems: orbit-plane normals of each system's biggest planet.
    let mut b = Bodies::with_capacity(2000);
    let infos = spawn_random_systems(&mut b, 100, 77, 5);
    println!("multi: {} systems, {} bodies", infos.len(), b.len());
    let mut normals = vec![];
    for i in 0..b.len() {
        if b.kind[i] == BodyKind::Planet && b.m[i] > 0.01 && b.origin[i] == BodyOrigin::Spawned {
            if let Some(h) = find_parent(&b, i) {
                let (x, y, z) = (b.x[i] - b.x[h], b.y[i] - b.y[h], b.z[i] - b.z[h]);
                let (vx, vy, vz) = (b.vx[i] - b.vx[h], b.vy[i] - b.vy[h], b.vz[i] - b.vz[h]);
                let n = [y * vz - z * vy, z * vx - x * vz, x * vy - y * vx];
                let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                normals.push((h, [n[0] / l, n[1] / l, n[2] / l]));
            }
        }
    }
    normals.sort_by_key(|n| n.0); normals.dedup_by_key(|n| n.0);
    for (h, n) in &normals { println!("  host {h}: plane normal ({:.2}, {:.2}, {:.2})", n[0], n[1], n[2]); }
    for _ in 0..300 { advance(&mut b, DT); }
    assert!((0..b.len()).all(|i| b.x[i].is_finite() && b.z[i].is_finite()));
}
