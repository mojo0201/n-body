use nbody_rust::physics::*;
fn main() {
    set_early_evolution(false);
    let mut ages = vec![];
    for seed in 1..2001u128 {
        let mut b = Bodies::with_capacity(64);
        spawn_system_with_info(&mut b, 20, seed);
        let p = (0..b.len()).filter(|&i| is_host_kind(b.kind[i])).max_by(|&i,&j| b.m[i].total_cmp(&b.m[j])).unwrap();
        ages.push(b.age_gyr[p]);
    }
    ages.sort_by(|a,b| a.total_cmp(b));
    for t in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0] { println!("age<{t} Gyr: {:.1}%", 100.0*ages.iter().filter(|&&a| a<t).count() as f64/ages.len() as f64); }
    println!("median {:.2}", ages[1000]);
}
