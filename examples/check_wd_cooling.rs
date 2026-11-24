use nbody_rust::physics::*;
fn main() {
    for t in [0.01, 0.1, 0.3, 1.0, 3.0, 10.0] {
        println!("t={t:>5} Gyr  L={:.2e} Lsun  Teff={:.0} K", wd_luminosity(0.6 * M_SUN_ME, t), wd_temperature(0.6 * M_SUN_ME, t));
    }
}
