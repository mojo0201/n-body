use nbody_rust::physics::*;
fn main() {
    set_early_evolution(false);
    let mut b = Bodies::with_capacity(8);
    let s = b.push(0.0, 0.0, 0.0, 0.0, 333_000.0, BodyKind::Star);
    let p = b.push(150.0, 0.0, 0.0, 0.0, 318.0, BodyKind::Planet);
    let (rs, rp) = (b.r[s], b.r[p]);
    println!("Rs {rs:.3} Gm, Rp {rp:.4} Gm, expect depth ~{:.0} ppm (x limb-darkening ~1.2 at centre)", (rp/rs).powi(2)*1e6);
    let obs = observer_dir(0.0, 0.0);
    for y in [-1.5 * rs, -0.9 * rs, -0.5 * rs, 0.0, 0.5 * rs, 0.9 * rs, 1.5 * rs] {
        b.y[p] = y;
        println!("y={:+.2} Rs: flux {:.6}", y / rs, visible_fraction(&b, s, obs));
    }
    b.y[p] = 0.0;
    println!("planet behind star: {:.6}", visible_fraction(&b, s, observer_dir(180.0, 0.0)));
    println!("face-on: {:.6}", visible_fraction(&b, s, observer_dir(0.0, 90.0)));
}
