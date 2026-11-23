//! Hashes the state `spawn_from_preset` produces for every preset JSON, plus
//! the warnings, so a refactor of the spawn code can be checked for identity.
use nbody_rust::physics::*;
fn main() {
    let mut paths: Vec<_> = std::fs::read_dir(std::env::args().nth(1).unwrap_or("assets/presets".into())).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).collect();
    paths.sort();
    let (mut loaded, mut skipped) = (0, 0);
    for path in paths {
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let Ok(p) = serde_json::from_str::<SystemPreset>(&text) else { skipped += 1; continue };
        let mut b = Bodies::with_capacity(500);
        spawn_from_preset(&mut b, &p);
        let mut h: u64 = 1469598103934665603;
        for i in 0..b.len() {
            for v in [b.x[i], b.y[i], b.z[i], b.vx[i], b.vy[i], b.vz[i], b.m[i], b.r[i], b.bulk_density[i]] {
                h = (h ^ v.to_bits()).wrapping_mul(1099511628211);
            }
            h = (h ^ (b.class[i] as u64) ^ ((b.origin[i] as u64) << 8)).wrapping_mul(1099511628211);
            for byte in b.name[i].clone().unwrap_or_default().bytes() { h = (h ^ byte as u64).wrapping_mul(1099511628211); }
        }
        loaded += 1;
        println!("{:<28} n={:<4} {h:016x}", path.file_name().unwrap().to_string_lossy(), b.len());
    }
    println!("loaded {loaded}, skipped {skipped}");
}
