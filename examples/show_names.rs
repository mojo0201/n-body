use nbody_rust::physics::*;
fn main() {
    let mut shown = 0;
    for seed in 1u128..=300 {
        let mut b = Bodies::with_capacity(400);
        spawn_system(&mut b, 120, seed);
        let unnamed = b.name.iter().filter(|n| n.is_none()).count();
        let mut set = std::collections::HashSet::new();
        let dup = b.name.iter().flatten().filter(|n| !set.insert(n.as_str())).count();
        if unnamed > 0 || dup > 0 { println!("seed {seed}: unnamed={unnamed} dup={dup}"); }
        // moons: spawned planets whose parent is a planet
        let moons: Vec<usize> = (0..b.len()).filter(|&i| b.origin[i] == BodyOrigin::Spawned && matches!(b.kind[i], BodyKind::Planet|BodyKind::DwarfPlanet) && find_parent(&b, i).is_some_and(|p| !is_host_kind(b.kind[p]))).collect();
        if !moons.is_empty() && shown < 3 {
            shown += 1;
            println!("seed {seed}: {} {} moons:", b.name[primary_index(&b).unwrap()].as_deref().unwrap(), moons.len());
            for &i in moons.iter().take(6) {
                let p = find_parent(&b, i).unwrap();
                println!("   {:<26} orbits {:<22} m={:.2e}", b.name[i].as_deref().unwrap(), b.name[p].as_deref().unwrap(), b.m[i]);
            }
        }
        if let Some(i) = (0..b.len()).find(|&i| b.origin[i] == BodyOrigin::Comet) { if seed < 40 { println!("seed {seed}: comet {}", b.name[i].as_deref().unwrap()); } }
        if let Some(h) = (0..b.len()).find(|&i| matches!(b.kind[i], BodyKind::WhiteDwarf | BodyKind::NeutronStar | BodyKind::BlackHole | BodyKind::BrownDwarf)) { if seed < 120 { println!("seed {seed}: remnant/BD host {}", b.name[h].as_deref().unwrap()); } }
    }
    println!("done");
}
