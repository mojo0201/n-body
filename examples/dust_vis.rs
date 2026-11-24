use nbody_rust::physics::*;
fn main() {
    nbody_rust::physics::set_early_evolution(false);
    for seed in [3u128, 5, 7, 8, 33] {
        let mut b = Bodies::with_capacity(1100);
        spawn_random_systems(&mut b, 1000, seed, 1);
        let prim = primary_index(&b).unwrap();
        let mut d: Vec<f64> = (0..b.len()).filter(|&i| i != prim).map(|i| ((b.x[i]-b.x[prim]).powi(2)+(b.y[i]-b.y[prim]).powi(2)+(b.z[i]-b.z[prim]).powi(2)).sqrt()).collect();
        d.sort_by(|a, c| a.partial_cmp(c).unwrap());
        let k = ((d.len() as f64 * 0.92) as usize).min(d.len() - 1);
        let radius = (d[k] * 1.15).max(5.0);
        let zoom = 500.0 / radius;
        let f = &b.dust_fields[0];
        println!("seed {seed}: n={} radius={radius:.0} zoom={zoom:.4} extent={:.0} ext*zoom={:.1} dmax={:.0} grains={}", b.len(), f.extent_gm, f.extent_gm * zoom, d[d.len()-1], f.grains.len());
    }
}
