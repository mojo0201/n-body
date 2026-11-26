//! usage: crash_scan [first_seed] [count] [steps]  — panics / NaNs on small generated systems
use nbody_rust::physics::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let first: u128 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let count: u128 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let steps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1500);
    std::panic::set_hook(Box::new(|_| {}));
    let mut bad = 0;
    for seed in first..first + count {
        for &n in &[2usize, 3, 5, 8, 12, 20, 40] {
            let r = catch_unwind(AssertUnwindSafe(|| {
                let mut b = Bodies::with_capacity(n * 4 + 64);
                spawn_system_with_info(&mut b, n, seed);
                for _ in 0..steps { advance(&mut b, DT); }
                let nan = (0..b.len()).any(|i| !(b.x[i].is_finite() && b.y[i].is_finite() && b.vx[i].is_finite() && b.m[i].is_finite()));
                nan
            }));
            match r {
                Err(e) => { bad += 1; let m = e.downcast_ref::<String>().cloned().or(e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default(); println!("PANIC seed {seed} n {n}: {m}"); }
                Ok(true) => { bad += 1; println!("NAN seed {seed} n {n}"); }
                Ok(false) => {}
            }
        }
    }
    println!("done, {bad} bad");
}
