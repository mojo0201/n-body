use nbody_rust::physics::*;
fn main() {
    nbody_rust::physics::set_early_evolution(false);
    let mut b = Bodies::with_capacity(20000);
    spawn_random_systems(&mut b, 100, 5, 60);
    for _ in 0..4 { advance(&mut b, DT); }
}
