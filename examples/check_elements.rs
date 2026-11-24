//! The orbit-elements history samples the selected planet and resets on a new selection.
use nbody_rust::physics::*;
use nbody_rust::ui::ElemHistory;
fn main() {
    let mut b = Bodies::with_capacity(100);
    spawn_solar_system(&mut b);
    b.refresh_parent_cache();
    let earth = (0..b.len()).find(|&i| b.name[i].as_deref() == Some("Earth")).expect("Earth");
    let (id, moon) = (b.id[earth], (0..b.len()).find(|&i| b.name[i].as_deref() == Some("Moon")));
    let mut h = ElemHistory::new();
    h.visible = true;
    for _ in 0..6 {
        for _ in 0..300 { advance(&mut b, DT); }
        std::thread::sleep(std::time::Duration::from_millis(260));
        h.track(&b, Some(id), false);
    }
    let s = h.samples.back().unwrap();
    println!("Earth: {} samples, a={:.3} AU e={:.3} inc={:.2} deg", h.samples.len(), s.a_au.unwrap(), s.ecc, s.inc_deg);
    h.track(&b, Some(id), true); // replaying: unchanged
    println!("replaying keeps {} samples", h.samples.len());
    if let Some(m) = moon {
        std::thread::sleep(std::time::Duration::from_millis(260));
        h.track(&b, Some(b.id[m]), false);
        println!("Moon selected: {} sample(s), a={:.5} AU (history restarted)", h.samples.len(), h.samples.back().and_then(|s| s.a_au).unwrap_or(f64::NAN));
    }
    h.track(&b, None, false);
    println!("deselected: {} samples", h.samples.len());
}
