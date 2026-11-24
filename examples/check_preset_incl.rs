use nbody_rust::physics::*;
fn main() {
    for path in ["assets/presets/algol.json"] {
        let p: SystemPreset = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut b = Bodies::with_capacity(500);
        spawn_from_preset(&mut b, &p);
        let ok = (0..b.len()).all(|i| b.x[i].is_finite() && b.z[i].is_finite());
        let zmax = (0..b.len()).map(|i| b.z[i].abs()).fold(0.0, f64::max);
        println!("{path}: {} bodies, finite={ok}, max|z|={zmax:.2} Gm", b.len());
    }
}
