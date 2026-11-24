use nbody_rust::physics::*;
fn main(){ let mut b=Bodies::with_capacity(20000); spawn_random_systems(&mut b,30,7,10); for _ in 0..15 {advance(&mut b,DT);} println!("{}",b.len()); }
