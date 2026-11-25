use nbody_rust::physics::*;
fn main() {
    set_early_evolution(false);
    let au = au_in_gm();
    let mut b = Bodies::with_capacity(200);
    let s = b.push(0.0, 0.0, 0.0, 0.0, 333_000.0, BodyKind::Star);
    let a = 5.2 * au;
    let v = (G * 333_000.0 / a).sqrt();
    let j = b.push(a, 0.0, 0.0, v, 318.0, BodyKind::Planet);
    b.bulk_density[j] = 1.3; b.r[j] = b.radius_now(j);
    let m = b.push(a + 0.2, 0.0, 0.0, v, 0.05, BodyKind::Planet);
    b.class[m] = BodyClass::Icy; b.bulk_density[m] = 1.0; b.r[m] = b.radius_now(m);
    let roche = roche_limit(&b, m, j);
    println!("Roche limit {:.3} Gm, Jupiter radius {:.3}", roche, b.r[j]);
    // place at 0.9 Roche on a circular orbit about Jupiter
    let d = 0.9 * roche; let vc = (G * 318.0 / d).sqrt();
    b.x[m] = a + d; b.vy[m] = v + vc;
    compute_accel(&mut b); rebase_energy(&mut b);
    let mom = |b: &Bodies| { let mut p = [0.0;2]; for i in 0..b.len() { if !b.dead[i] { p[0]+=b.m[i]*b.vx[i]; p[1]+=b.m[i]*b.vy[i]; } } p };
    let m0 = b.m.iter().sum::<f64>(); let p0 = mom(&b);
    for _ in 0..20 { advance(&mut b, DT); }
    let ring = (0..b.len()).filter(|&i| !b.dead[i] && b.origin[i] == BodyOrigin::TidalDebris).count();
    let p1 = mom(&b);
    println!("bodies {} tidal debris {} events {}", b.len(), ring, b.stats.tidal);
    println!("mass {:.5} -> {:.5}; momentum dp = ({:.2e}, {:.2e}) vs |p| {:.2e}", m0, b.m.iter().sum::<f64>(), p1[0]-p0[0], p1[1]-p0[1], p0[1]);
    for e in b.events.iter().take(4) { println!("event: {}", e.text); }
    for _ in 0..4000 { advance(&mut b, DT); }
    let (mut inside, mut esc) = (0, 0);
    for i in 0..b.len() { if b.dead[i] || b.origin[i] != BodyOrigin::TidalDebris { continue; }
        let d = ((b.x[i]-b.x[j]).powi(2)+(b.y[i]-b.y[j]).powi(2)).sqrt(); if d < 5.0*roche { inside += 1 } else { esc += 1 } }
    println!("Jupiter mass {:.4}; merges {} coalesces {} impacts {}", b.m[j], b.stats.merges, b.stats.coalesces, b.stats.impacts);
    for i in 0..b.len() { if !b.dead[i] && i != s && i != j { println!("  body m={:.5} d={:.3} origin {:?}", b.m[i], ((b.x[i]-b.x[j]).powi(2)+(b.y[i]-b.y[j]).powi(2)).sqrt(), b.origin[i]); } }
    println!("after 4000 steps: bound near Jupiter {inside}, far {esc}");
}
