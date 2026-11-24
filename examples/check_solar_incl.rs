use nbody_rust::physics::*;
fn main() {
    let p = solar_system_preset();
    let mut b = Bodies::with_capacity(2000);
    spawn_from_preset(&mut b, &p);
    let s = primary_index(&b).unwrap();
    for name in ["Mercury", "Earth", "Jupiter", "Eris", "1P/Halley", "Moon", "Titan", "Triton", "Charon"] {
        let Some(i) = (0..b.len()).find(|&i| b.name[i].as_deref() == Some(name)) else { println!("{name}: not found"); continue };
        let h = find_parent(&b, i).unwrap_or(s);
        let (x, y, z) = (b.x[i] - b.x[h], b.y[i] - b.y[h], b.z[i] - b.z[h]);
        let (vx, vy, vz) = (b.vx[i] - b.vx[h], b.vy[i] - b.vy[h], b.vz[i] - b.vz[h]);
        let hz = x * vy - y * vx;
        let hm = ((y * vz - z * vy).powi(2) + (z * vx - x * vz).powi(2) + hz * hz).sqrt();
        println!("{name:>10} about {:?}: incl to z-plane {:.2} deg", b.name[h], (hz / hm).acos().to_degrees());
    }
}
