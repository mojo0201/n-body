//! usage: early_stats [seeds] [bodies] [first_seed]  — timing and outcomes of early evolution
use nbody_rust::physics::*;
use std::time::Instant;
fn elems(b: &Bodies) -> Vec<(u64, f64, f64, f64)> {
    let au = au_in_gm();
    let mut v = vec![];
    for i in 0..b.len() {
        if b.dead[i] || b.kind[i] != BodyKind::Planet || b.origin[i] != BodyOrigin::Spawned || b.m[i] < 0.05 { continue; }
        let Some(h) = find_parent(b, i) else { continue };
        if !is_host_kind(b.kind[h]) { continue; }
        let (dx, dy, dz) = (b.x[i]-b.x[h], b.y[i]-b.y[h], b.z[i]-b.z[h]);
        let (vx, vy, vz) = (b.vx[i]-b.vx[h], b.vy[i]-b.vy[h], b.vz[i]-b.vz[h]);
        let r = (dx*dx+dy*dy+dz*dz).sqrt();
        let mu = G * (b.m[h] + b.m[i]);
        let en = 0.5*(vx*vx+vy*vy+vz*vz) - mu/r;
        let a = -mu/(2.0*en);
        let (hx, hy, hz) = (dy*vz-dz*vy, dz*vx-dx*vz, dx*vy-dy*vx);
        let h2 = hx*hx+hy*hy+hz*hz;
        let e = (1.0 - h2/(mu*a)).max(0.0).sqrt();
        v.push((b.id[i], a/au, e, b.m[i]));
    }
    v
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: u128 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let bodies: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(14);
    let s0: u128 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    set_early_evolution(false);
    let (mut tb, mut ta, mut tm, mut tsec) = (0, 0, 0, 0.0);
    for seed in s0..s0 + n {
        let mut b = Bodies::with_capacity(bodies);
        spawn_system_with_info(&mut b, bodies, seed);
        let e0 = elems(&b);
        let t = Instant::now();
        let r = evolve_early(&mut b);
        let dt = t.elapsed().as_secs_f64();
        println!("seed {seed}: planets {}->{} merged {} ejected {} packed {} skipped {} {:.2}s", r.planets_before, r.planets_after, r.merged, r.ejected, r.packed, r.skipped, dt);
        let e1 = elems(&b);
        for (id, a, e, m) in &e0 { let n = e1.iter().find(|x| x.0 == *id); match n { Some(x) => println!("   m={:.1} a {:.3}->{:.3} e {:.3}->{:.3}", m, a, x.1, e, x.2), None => println!("   m={:.1} a {:.3} LOST", m, a) } }
        tb += r.planets_before; ta += r.planets_after; tm += r.merged; tsec += dt;
    }
    println!("total planets {tb}->{ta}, merged {tm}, {tsec:.1}s");
}
