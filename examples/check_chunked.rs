use nbody_rust::physics::*;
fn main() {
    nbody_rust::physics::set_early_evolution(false);
    let mk = || { let mut b = Bodies::with_capacity(20000); spawn_random_systems(&mut b, 30, 7, 10); b };
    let (mut a, mut c) = (mk(), mk());
    let mut resumes = 0;
    for _ in 0..20 {
        advance(&mut a, DT);
        if let Some(mut j) = advance_begin(&mut c, DT) {
            while !advance_run(&mut j, &mut c, || false) { resumes += 1; }
        }
    }
    let same = a.x == c.x && a.y == c.y && a.z == c.z && a.vx == c.vx && a.time == c.time && a.stats.steps == c.stats.steps;
    println!("n={} resumes={} identical={}", a.len(), resumes, same);
}
