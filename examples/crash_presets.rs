use nbody_rust::physics::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let mut presets: Vec<(String, SystemPreset)> = vec![("solar_system".into(), solar_system_preset())];
    for s in ["algol", "hd_80606", "kepler_16"] { { let t = std::fs::read_to_string(format!("assets/presets/{s}.json")).unwrap(); presets.push((s.into(), serde_json::from_str(&t).unwrap())); } }
    for (name, p) in &presets {
        for seed in 1..6u128 {
            let r = catch_unwind(AssertUnwindSafe(|| {
                let mut b = Bodies::with_capacity(4096);
                spawn_preset_with_seed(&mut b, p, seed);
                for _ in 0..4000 { advance(&mut b, DT); }
                (0..b.len()).any(|i| !(b.x[i].is_finite() && b.vx[i].is_finite()))
            }));
            println!("{name} seed {seed}: {}", match r { Err(_) => "PANIC".into(), Ok(true) => "NAN".into(), Ok(false) => "ok".to_string() });
        }
    }
}
