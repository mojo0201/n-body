use nbody_rust::physics::*;
fn main() {
    set_early_evolution(false);
    let p = solar_system_preset();
    let mut b = Bodies::with_capacity(2000);
    spawn_from_preset(&mut b, &p);
    let s = primary_index(&b).unwrap();
    println!("Sun m {:.0} m_init {:.0} age {:.3} Gyr L {:.3} T {:.0}", b.m[s], b.m_init[s], b.age_gyr[s], luminosity_of(&b, s), temperature_of(&b, s));
}
