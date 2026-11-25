use nbody_rust::physics::*;
fn main() {
    let p = solar_system_preset();
    let mut b = Bodies::with_capacity(2000);
    spawn_preset_with_seed(&mut b, &p, 12345);
    let f = |b: &Bodies, n: &str| (0..b.len()).find(|&i| b.name[i].as_deref() == Some(n));
    let (e, m) = (f(&b, "Earth").unwrap(), f(&b, "Moon").unwrap());
    println!("parent of moon: {:?}, earth {:?}", find_parent(&b, m).map(|h| b.name[h].clone()), find_parent(&b, e).map(|h| b.name[h].clone()));
    let mut ang_prev = 0.0f64; let mut turns = 0.0;
    for s in 0..4000 {
        let (e, m) = (f(&b, "Earth").unwrap(), f(&b, "Moon").unwrap());
        if s % 400 == 0 { let d = ((b.x[m]-b.x[e]).powi(2)+(b.y[m]-b.y[e]).powi(2)+(b.z[m]-b.z[e]).powi(2)).sqrt();
            println!("step {s}: sep {:.4} Gm rel v {:.4} rung {}", d, ((b.vx[m]-b.vx[e]).powi(2)+(b.vy[m]-b.vy[e]).powi(2)).sqrt(), b.rung[m]); }
        let a = (b.y[m]-b.y[e]).atan2(b.x[m]-b.x[e]);
        let mut d = a - ang_prev; while d > std::f64::consts::PI { d -= std::f64::consts::TAU } while d < -std::f64::consts::PI { d += std::f64::consts::TAU }
        if s > 0 { turns += d; } ang_prev = a;
        if let Some(mut job) = advance_begin(&mut b, DT) {
            loop {
                let mut c = 0;
                if advance_run(&mut job, &mut b, || { c += 1; c < 2 }) { break; }
            }
        }
    }
    println!("moon revolutions about Earth: {:.2} (expect ~{:.2})", turns / std::f64::consts::TAU, 4000.0 * DT / 7.3 * 13.4);
}
