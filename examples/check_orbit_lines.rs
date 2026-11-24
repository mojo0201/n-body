use nbody_rust::physics::*;
fn main() {
    nbody_rust::physics::set_early_evolution(false);
    for scale in [0.0, 1.0] {
    set_inclination_scale(scale);
    println!("scale {scale}");
    for seed in [3u128, 7, 42] {
        let mut b = Bodies::with_capacity(400);
        spawn_system(&mut b, 400, seed);
        let segs = compute_orbit_segments(&b, 20.0, true, false);
        let mut worst: f64 = 0.0;
        let mut worst_rel: f64 = 0.0;
        for i in 0..b.len() {
            let p = [b.x[i], b.y[i], b.z[i]];
            // nearest point on any segment endpoint set belonging to some ellipse
            let mut best = f64::MAX;
            for (p0, _, _) in &segs {
                let d = ((p0[0] as f64 - p[0]).powi(2) + (p0[1] as f64 - p[1]).powi(2) + (p0[2] as f64 - p[2]).powi(2)).sqrt();
                best = best.min(d);
            }
            let r = (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt().max(1e-9);
            if best.is_finite() && best < f64::MAX && b.kind[i] == BodyKind::Planet {
                worst = worst.max(best);
                worst_rel = worst_rel.max(best / r); if best>5.0 { /* count */ };
            }
        }
        println!("seed {seed}: segs {} worst planet-to-line {worst:.3} Gm, rel {worst_rel:.4}", segs.len());
    }
    }
}
