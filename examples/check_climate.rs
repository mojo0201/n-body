use nbody_rust::physics::*;
fn main() {
    set_early_evolution(false);
    let p = solar_system_preset();
    let mut b = Bodies::with_capacity(2000);
    spawn_from_preset(&mut b, &p);
    println!("expected: Mercury 440 K airless, Venus ~737, Earth 288, Mars ~210 thin, Titan ~94, Moon ~250 airless");
    for n in ["Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Titan", "Europa", "Neptune"] {
        let Some(i) = (0..b.len()).find(|&i| b.name[i].as_deref() == Some(n)) else { println!("{n}: ?"); continue };
        let s = stellar_flux(&b, b.x[i], b.y[i], i);
        let (hi, ho) = hz_flux_limits(stellar_flux_teff(&b, b.x[i], b.y[i], i));
        let c = surface_climate(&b, i, s, hi, ho);
        println!("{n:>8}: S={s:.3} {:?} albedo {:.2} T={:.0} K frozen {}", c.atmosphere, c.albedo, c.t_surface, c.frozen);
    }
}
