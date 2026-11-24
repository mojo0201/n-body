//! Two-giant (Jupiter 5.2 AU, Saturn-like 9.5 AU) systems through early evolution.
use nbody_rust::physics::*;
fn main() {
    let au = au_in_gm();
    for k in 0..16u32 {
        let mut b = Bodies::with_capacity(32);
        b.push(0.0, 0.0, 0.0, 0.0, 333_000.0, BodyKind::Star);
        for _ in 0..k { let d = b.push(1e9, 1e9, 0.0, 0.0, 1e-9, BodyKind::Planet); b.dead[d] = true; }
        for (m, a) in [(318.0, 5.2), (95.0, 9.5), (14.0, 19.0)] {
            let r = a * au;
            let v = (G * (333_000.0 + m) / r).sqrt();
            let i = b.push(r, 0.0, 0.0, v, m, BodyKind::Planet);
            b.class[i] = BodyClass::Rocky;
        }
        b.compact();
        compute_accel(&mut b);
        rebase_energy(&mut b);
        let t = std::time::Instant::now();
        let r = evolve_early(&mut b);
        let mut out = vec![];
        for i in 1..b.len() { if b.dead[i] { continue; }
            let (dx,dy)=(b.x[i]-b.x[0], b.y[i]-b.y[0]); let (vx,vy,vz)=(b.vx[i]-b.vx[0], b.vy[i]-b.vy[0], b.vz[i]-b.vz[0]);
            let rr=(dx*dx+dy*dy+b.z[i].powi(2)).sqrt(); let mu=G*(b.m[0]+b.m[i]); let en=0.5*(vx*vx+vy*vy+vz*vz)-mu/rr;
            if en>=0.0 { out.push(format!("m{:.0} UNBOUND", b.m[i])); continue; }
            let a=-mu/(2.0*en); let (hx,hy,hz)=(dy*vz-b.z[i]*vy, b.z[i]*vx-dx*vz, dx*vy-dy*vx);
            let e=(1.0-(hx*hx+hy*hy+hz*hz)/(mu*a)).max(0.0).sqrt();
            out.push(format!("m{:.0} a={:.2} e={:.2}", b.m[i], a/au, e)); }
        println!("k={k} packed {} merged {} {:.1}s | {}", r.packed, r.merged, t.elapsed().as_secs_f64(), out.join("; "));
    }
}
