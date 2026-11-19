use nbody_rust::physics::*;
fn rng(s:&mut u64)->f64{ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; (*s>>11) as f64/(1u64<<53) as f64 }
fn main(){
    // A: tilted circular orbit.
    let mut b=Bodies::with_capacity(4);
    let ms=300000.0; let a=300.0; let inc=std::env::var("INC").ok().and_then(|s|s.parse::<f64>().ok()).unwrap_or(40.0).to_radians();
    b.push(0.0,0.0,0.0,0.0,ms,BodyKind::Star);
    let v=(G*ms/a).sqrt();
    let p=b.push(a,0.0,0.0,v*inc.cos(),1.0,BodyKind::Planet); b.vz[p]=v*inc.sin();
    set_star_age(&mut b,0,5.0); compute_accel(&mut b); rebase_energy(&mut b);
    let l0=|b:&Bodies|{ let (x,y,z,vx,vy,vz)=(b.x[1],b.y[1],b.z[1],b.vx[1],b.vy[1],b.vz[1]); [y*vz-z*vy,z*vx-x*vz,x*vy-y*vx] };
    let la=l0(&b);
    let period=std::f64::consts::TAU*(a.powi(3)/(G*ms)).sqrt();
    let steps=(10.0*period/DT) as usize;
    for _ in 0..steps{ advance(&mut b,DT); }
    let lb=l0(&b); let r=(b.x[1].powi(2)+b.y[1].powi(2)+b.z[1].powi(2)).sqrt();
    let n0=(la[0].powi(2)+la[1].powi(2)+la[2].powi(2)).sqrt(); let n1=(lb[0].powi(2)+lb[1].powi(2)+lb[2].powi(2)).sqrt();
    let cosang=(la[0]*lb[0]+la[1]*lb[1]+la[2]*lb[2])/(n0*n1);
    println!("A: r={r:.6} (a=300) |L| {n0:.6}->{n1:.6} plane-angle {:.2e} rad  max z={:.3}  drift%={:?}",cosang.clamp(-1.0,1.0).acos(), a*inc.sin(), energy_drift_pct(&b));
    // B: 3D cluster, tree vs brute force.
    let n=1500; let mut b=Bodies::with_capacity(n); let mut s=88172645463325252u64;
    for _ in 0..n { let (x,y,z)=(rng(&mut s)*200.0-100.0,rng(&mut s)*200.0-100.0,rng(&mut s)*200.0-100.0);
        let k=b.push(x,y,0.0,0.0,1.0+rng(&mut s)*10.0,BodyKind::DwarfPlanet); b.z[k]=z; b.vx[k]=(rng(&mut s)-0.5)*0.1; b.vy[k]=(rng(&mut s)-0.5)*0.1; b.vz[k]=(rng(&mut s)-0.5)*0.1; }
    compute_accel(&mut b);
    let mut maxrel=0.0f64;
    for i in (0..n).step_by(37){ let (mut ax,mut ay,mut az)=(0.0,0.0,0.0);
        for j in 0..n{ if j==i{continue;} let dx=b.x[j]-b.x[i]; let dy=b.y[j]-b.y[i]; let dz=b.z[j]-b.z[i]; let d2=dx*dx+dy*dy+dz*dz+SOFT2; let f=G*b.m[j]/(d2*d2.sqrt()); ax+=f*dx; ay+=f*dy; az+=f*dz; }
        let e=((b.ax[i]-ax).powi(2)+(b.ay[i]-ay).powi(2)+(b.az[i]-az).powi(2)).sqrt(); let mag=(ax*ax+ay*ay+az*az).sqrt(); maxrel=maxrel.max(e/mag.max(1e-12)); }
    println!("B: tree vs direct max relative error (sampled, includes COM correction) = {maxrel:.3e}");
}
