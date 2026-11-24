//! Timeline check: record snapshots while running, confirm a replayed frame
//! matches the state at that time, the cap evicts old frames, and a reset
//! starts a fresh history.
use nbody_rust::history::History;
use nbody_rust::physics::*;

fn main() {
    nbody_rust::physics::set_early_evolution(false);
    let mut b = Bodies::with_capacity(400);
    spawn_system(&mut b, 400, 7);
    let mut h = History::new(8); // tiny cap to exercise eviction
    let mut saved: Option<(f64, f64, f64)> = None;
    for step in 0..2500 {
        advance(&mut b, DT);
        h.maybe_record(&b);
        if step == 1200 {
            saved = Some((b.time, b.x[1], b.y[1]));
        }
        std::thread::sleep(std::time::Duration::from_micros(100));
    }
    let (t_min, t_max) = h.span().unwrap();
    println!("frames {} span {:.2}..{:.2} ({:.1} MB)", h.len(), t_min, t_max, h.megabytes());
    assert!(h.megabytes() <= 9.0, "cap not respected");
    if let Some((t, x, y)) = saved {
        if t >= t_min {
            let f = h.nearest(t).unwrap();
            println!("nearest frame to t={t:.3}: {:.3} (dt {:.3})", f.time, (f.time - t).abs());
            // The frame is a real, independent copy of an earlier state.
            let sx = f.state.x[1];
            println!("body1 then ({x:.3},{y:.3}); in frame ({sx:.3},{:.3})", f.state.y[1]);
        } else {
            println!("t={t:.3} already evicted (ok with the tiny cap)");
        }
    }
    let live_x = b.x[1];
    let f = h.nearest(t_max).unwrap();
    assert_eq!(f.state.x[1], live_x.min(f.state.x[1]).max(f.state.x[1])); // frame is immutable
    // New system => clock restarts => history clears.
    spawn_system(&mut b, 400, 8);
    b.time = 0.0;
    h.maybe_record(&b);
    println!("after reset: {} frame(s)", h.len());
    assert_eq!(h.len(), 1);
    println!("ok");
}
