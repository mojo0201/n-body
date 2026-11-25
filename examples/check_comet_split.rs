use nbody_rust::physics::*;
fn main() {
    set_early_evolution(false);
    let au = au_in_gm();
    let mut b = Bodies::with_capacity(900);
    b.push(0.0, 0.0, 0.0, 0.0, 333_000.0, BodyKind::Star);
    let (q, a) = (0.8 * au, 0.8 * au);
    let e = 1.0 - q / a;
    let vp = (G * 333_000.0 * (1.0 + e) / q).sqrt();
    for k in 0..300 {
        let ang = k as f64 * 0.02;
        let i = b.push(q * ang.cos(), q * ang.sin(), -vp * ang.sin(), vp * ang.cos(), 1e-9, BodyKind::Planet);
        b.origin[i] = BodyOrigin::Comet; b.class[i] = BodyClass::Icy; b.bulk_density[i] = 0.6; b.r[i] = b.radius_now(i);
    }
    compute_accel(&mut b); rebase_energy(&mut b);
    let m0: f64 = b.m.iter().sum();
    let mom = |b: &Bodies| { let mut p = [0.0;2]; for i in 0..b.len() { if !b.dead[i] { p[0]+=b.m[i]*b.vx[i]; p[1]+=b.m[i]*b.vy[i]; } } p };
    let n0 = b.len();
    for s in 0..6000 { advance(&mut b, DT); if s % 2000 == 0 { println!("step {s}: {} bodies", b.len()); } }
    let splits = b.events.iter().filter(|e| e.text.contains("split in two")).count();
    println!("comets {} -> {} bodies, {} splits logged, exhausted {}", n0 - 1, b.len() - 1, splits, b.events.iter().filter(|e| e.text.contains("exhausted")).count());
    println!("mass lost to sublimation: {:.3e} M⊕ of {:.3e}", m0 - b.m.iter().sum::<f64>() , 300e-9);
    if let Some(e) = b.events.iter().find(|e| e.text.contains("split")) { println!("e.g. {}", e.text); }
}
