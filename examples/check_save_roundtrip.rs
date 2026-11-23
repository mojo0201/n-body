use nbody_rust::physics::*;
fn hash(b: &Bodies) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for i in 0..b.len() {
        for v in [b.x[i], b.y[i], b.z[i], b.vx[i], b.vy[i], b.vz[i], b.m[i], b.r[i], b.spin[i], b.age_gyr[i]] { h = (h ^ v.to_bits()).wrapping_mul(1099511628211); }
        h = (h ^ b.id[i] ^ ((b.rung[i] as u64) << 40)).wrapping_mul(1099511628211);
    }
    for f in &b.dust_fields { for g in &f.grains { for v in [g.x, g.y, g.z, g.vx, g.mass] { h = (h ^ v.to_bits()).wrapping_mul(1099511628211); } } }
    h = (h ^ b.time.to_bits() ^ b.stats.steps).wrapping_mul(1099511628211);
    h ^ (b.events.len() as u64) ^ ((b.dust_fields.len() as u64) << 32)
}
fn main() {
    let mut b = Bodies::with_capacity(5000);
    spawn_random_systems(&mut b, 80, 9, 3);
    for _ in 0..300 { advance(&mut b, DT); }
    let app = AppSave { cam: [1.0, 2.0], cam_z: 0.5, yaw: 0.3, pitch: 0.4, zoom: 2.0, speed: 1.0, paused: false, seed: "abc".into(), body_count: 80, preset: "random".into(), systems: 3, library_preset: None, selected: None, follow: false, show_orbits: true, show_debris_orbits: false, aging_mult: 1.0 };
    let file = capture_save(&b, app).unwrap();
    let mut c = Bodies::with_capacity(10);
    restore_save(&mut c, &file).unwrap();
    println!("n={} grains={} before={:016x} after={:016x} same_state={}", b.len(), b.dust_fields.iter().map(|f| f.grains.len()).sum::<usize>(), hash(&b), hash(&c), hash(&b) == hash(&c));
    // Bad files must be refused and leave the target unchanged.
    let mut bad = file.clone(); bad.format = "nope".into();
    let before = hash(&c);
    println!("bad format refused={} unchanged={}", restore_save(&mut c, &bad).is_err(), hash(&c) == before);
    let mut bad2 = file.clone(); bad2.sim.bodies[0].x = f64::NAN;
    println!("nan refused={} unchanged={}", restore_save(&mut c, &bad2).is_err(), hash(&c) == before);
    let mut bad3 = file.clone(); if let Some(d) = bad3.sim.dust.get_mut(0) { d.primary = 1_000_000; }
    println!("dust primary refused={}", restore_save(&mut c, &bad3).is_err());
}
