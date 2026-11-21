use nbody_rust::physics::*;
fn norm(v: [f64; 3]) -> f64 { (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt() }
fn lvec(b: &Bodies, i: usize) -> [f64; 3] {
    let (x,y,z,vx,vy,vz) = (b.x[i],b.y[i],b.z[i],b.vx[i],b.vy[i],b.vz[i]);
    [y*vz-z*vy, z*vx-x*vz, x*vy-y*vx]
}
// Close planet around a *young* star so gas-disk migration and tides are live.
fn migrate(incd: f64) {
    let mut b = Bodies::with_capacity(4);
    let (ms, a) = (300000.0, 20.0);
    let inc = incd.to_radians();
    b.push(0.0, 0.0, 0.0, 0.0, ms, BodyKind::Star);
    let v = (G * ms / a).sqrt();
    let p = b.push(a, 0.0, 0.0, v * inc.cos(), 3.0, BodyKind::Planet);
    b.vz[p] = v * inc.sin();
    compute_accel(&mut b); rebase_energy(&mut b);
    let l0 = lvec(&b, 1);
    for _ in 0..40000 { advance(&mut b, DT); }
    let l1 = lvec(&b, 1);
    let r = (b.x[1].powi(2) + b.y[1].powi(2) + b.z[1].powi(2)).sqrt();
    let cosang = (l0[0]*l1[0] + l0[1]*l1[1] + l0[2]*l1[2]) / (norm(l0)*norm(l1));
    println!("migrate inc {incd:>4}: n={} r={r:.9} |L| {:.6}->{:.6} plane-angle {:.2e}", b.len(), norm(l0), norm(l1), cosang.clamp(-1.0,1.0).acos());
}
// Head-on pair along `dir`, offset by `off` perpendicular; momentum and mass must be conserved.
fn collide(dir: [f64; 3]) {
    let mut b = Bodies::with_capacity(4);
    let n = norm(dir); let d = [dir[0]/n, dir[1]/n, dir[2]/n];
    let s = 1.0; let sep = 5.0;
    for k in [-1.0f64, 1.0] {
        let i = b.push(k*sep*d[0], k*sep*d[1], 0.0, -k*s*d[1].mul_add(0.0, d[0]), 4.0, BodyKind::Planet);
        b.z[i] = k*sep*d[2]; b.vy[i] = -k*s*d[1]; b.vz[i] = -k*s*d[2];
    }
    // bodies must be large enough to touch within a few steps
    let m0: f64 = b.m.iter().sum();
    let p0 = [(0..2).map(|i| b.m[i]*b.vx[i]).sum::<f64>(), (0..2).map(|i| b.m[i]*b.vy[i]).sum::<f64>(), (0..2).map(|i| b.m[i]*b.vz[i]).sum::<f64>()];
    for _ in 0..20000 { advance(&mut b, DT); if b.len() == 1 { break; } }
    let m1: f64 = b.m.iter().sum();
    let nb = b.len();
    let p1 = [(0..nb).map(|i| b.m[i]*b.vx[i]).sum::<f64>(), (0..nb).map(|i| b.m[i]*b.vy[i]).sum::<f64>(), (0..nb).map(|i| b.m[i]*b.vz[i]).sum::<f64>()];
    println!("collide dir {dir:?}: bodies {nb}, mass {m0}->{m1:.6}, |p| {:.3e}->{:.3e}", norm(p0), norm(p1));
}
fn main() {
    for i in [0.0, 40.0, 75.0, 120.0] { migrate(i); }
    collide([1.0, 0.0, 0.0]);
    collide([0.0, 0.0, 1.0]);
    collide([1.0, 1.0, 1.0]);
}
