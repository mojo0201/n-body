//! usage: triage_stats [seeds] [bodies] [first_seed] — does a Hill-spacing check predict which old systems early evolution changes?
use nbody_rust::physics::*;
use std::time::Instant;
fn planets(b: &Bodies) -> (f64, Vec<(f64, f64, f64)>) {
    let mut v = vec![]; let mut mh = 0.0;
    for i in 0..b.len() {
        if b.dead[i] || b.kind[i] != BodyKind::Planet || b.origin[i] != BodyOrigin::Spawned || b.m[i] < 0.05 { continue; }
        let Some(h) = find_parent(b, i) else { continue };
        if !is_host_kind(b.kind[h]) { continue; }
        let (dx, dy, dz) = (b.x[i]-b.x[h], b.y[i]-b.y[h], b.z[i]-b.z[h]);
        let (vx, vy, vz) = (b.vx[i]-b.vx[h], b.vy[i]-b.vy[h], b.vz[i]-b.vz[h]);
        let r = (dx*dx+dy*dy+dz*dz).sqrt();
        let mu = G * (b.m[h] + b.m[i]);
        let a = -mu / (2.0 * (0.5*(vx*vx+vy*vy+vz*vz) - mu/r));
        let (hx, hy, hz) = (dy*vz-dz*vy, dz*vx-dx*vz, dx*vy-dy*vx);
        let e = (1.0 - (hx*hx+hy*hy+hz*hz)/(mu*a)).max(0.0).sqrt();
        mh = b.m[h]; v.push((a, e, b.m[i]));
    }
    v.sort_by(|x, y| x.0.total_cmp(&y.0));
    (mh, v)
}
/// Smallest mutual-Hill spacing between neighbours, using periapsis/apoapsis gap.
fn min_spacing(mh: f64, v: &[(f64, f64, f64)]) -> f64 {
    let mut best = f64::INFINITY;
    for w in v.windows(2) {
        let (a1, e1, m1) = w[0]; let (a2, e2, m2) = w[1];
        let rh = ((m1 + m2) / (3.0 * mh)).cbrt() * 0.5 * (a1 + a2);
        let gap = a2 * (1.0 - e2) - a1 * (1.0 + e1);
        best = best.min(gap / rh);
    }
    best
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: u128 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let bodies: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let s0: u128 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    set_early_evolution(false);
    // bucket edges on min spacing
    let edges = [0.0, 3.0, 6.0, 10.0, 15.0, f64::INFINITY];
    let mut cnt = [0usize; 5]; let mut chg = [0usize; 5]; let mut secs = [0.0f64; 5];
    let (mut few, mut total) = (0, 0);
    for seed in s0..s0 + n {
        let mut b = Bodies::with_capacity(bodies);
        spawn_system_with_info(&mut b, bodies, seed);
        if is_early_system(&b) { continue; }
        total += 1;
        let (mh, p0) = planets(&b);
        if p0.len() < 2 { few += 1; continue; }
        let sp = { let s = min_spacing(mh, &p0); if s.is_finite() { s.max(0.0) } else { 0.0 } };
        let t = Instant::now();
        let r = evolve_early(&mut b);
        let dt = t.elapsed().as_secs_f64();
        let (_, p1) = planets(&b);
        let moved = p0.iter().filter(|p| !p1.iter().any(|q| (q.0 / p.0 - 1.0).abs() < 0.20 && (q.2 / p.2 - 1.0).abs() < 0.2)).count();
        let changed = r.merged + r.ejected > 0 || r.planets_after < r.planets_before;
        let _ = moved;
        let k = (0..5).find(|&k| sp >= edges[k] && sp < edges[k + 1]).unwrap();
        cnt[k] += 1; secs[k] += dt; if changed { chg[k] += 1; }
    }
    println!("old systems {total}, with <2 planets {few}");
    for k in 0..5 {
        println!("min spacing {:>5.1}..{:<5.1} Hill radii: {:>4} systems, {:>5.1}% lost/merged planets, {:.2}s each", edges[k], edges[k+1], cnt[k], if cnt[k] > 0 { 100.0 * chg[k] as f64 / cnt[k] as f64 } else { 0.0 }, if cnt[k] > 0 { secs[k] / cnt[k] as f64 } else { 0.0 });
    }
}
