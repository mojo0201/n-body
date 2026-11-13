//! User edits: scale mass, change kind, delete, circularize.

use super::*;

pub fn user_modified(b: &mut Bodies) {
    b.parent_cache_valid = false;
    compute_accel(b);
    rebase_energy(b);
}

pub fn scale_mass(b: &mut Bodies, i: usize, factor: f64) {
    let m = b.m[i] * factor;
    if !m.is_finite() || m < 1e-4 {
        return;
    }
    if b.kind[i] == BodyKind::Star {
        // Keep the same fraction of the (new) lifetime.
        let f = b.age_gyr[i] / star_lifetime_gyr(BodyKind::Star, b.m_init[i]);
        b.m_init[i] *= factor;
        b.age_gyr[i] = f * star_lifetime_gyr(BodyKind::Star, b.m_init[i]);
    } else {
        b.m_init[i] = m;
    }
    b.set_mass(i, m);
    b.origin[i] = BodyOrigin::Spawned;
    let text = format!("{} mass scaled by {}", tag(b, i), factor);
    let (x, y) = (b.x[i], b.y[i]);
    b.log(EventKind::User, text, x, y);
    user_modified(b);
}

pub fn set_kind(b: &mut Bodies, i: usize, kind: BodyKind) {
    let m = b.m[i];
    let new_m = match kind {
        BodyKind::Star if m < STAR_M_MIN || m > 100.0 * M_SUN_ME => M_SUN_ME,
        BodyKind::BrownDwarf if m < BD_M_MIN || m > BD_M_MAX => m.clamp(BD_M_MIN, BD_M_MAX),
        BodyKind::WhiteDwarf if m < WD_M_MIN || m > WD_M_MAX => m.clamp(WD_M_MIN, WD_M_MAX),
        BodyKind::NeutronStar if m < NS_M_MIN || m > NS_M_MAX => m.clamp(NS_M_MIN, NS_M_MAX),
        BodyKind::BlackHole if m < BH_M_MIN => 10.0 * M_SUN_ME,
        BodyKind::DwarfPlanet if m > DWARF_M_MAX => DWARF_M_MAX,
        BodyKind::Planet if m >= 1000.0 => JUPITER_MASS,
        _ => m,
    };
    b.kind[i] = kind;
    b.origin[i] = BodyOrigin::Spawned;
    // `set_kind` takes no RNG; reset to a fixed class so the result is
    // deterministic.
    b.class[i] = BodyClass::Rocky;
    b.bulk_density[i] = default_bulk_density(kind, new_m);
    b.m_init[i] = new_m;
    b.age_gyr[i] = default_age_gyr(kind);
    b.set_mass(i, new_m);
    let text = format!("{} changed type", tag(b, i));
    let (x, y) = (b.x[i], b.y[i]);
    b.log(EventKind::User, text, x, y);
    user_modified(b);
}

pub fn delete_body(b: &mut Bodies, i: usize) {
    let text = format!("{} deleted", tag(b, i));
    let (x, y) = (b.x[i], b.y[i]);
    b.dead[i] = true;
    b.compact();
    b.log(EventKind::User, text, x, y);
    user_modified(b);
}

pub fn circularize(b: &mut Bodies, i: usize) -> bool {
    let Some(p) = find_parent(b, i) else {
        return false;
    };
    let dx = b.x[i] - b.x[p];
    let dy = b.y[i] - b.y[p];
    let r = (dx * dx + dy * dy).sqrt();
    if r < 1e-9 {
        return false;
    }
    let dvx = b.vx[i] - b.vx[p];
    let dvy = b.vy[i] - b.vy[p];
    let sign = if dx * dvy - dy * dvx >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let vc = (G * (b.m[p] + b.m[i]) / r).sqrt();
    b.vx[i] = b.vx[p] - sign * dy / r * vc;
    b.vy[i] = b.vy[p] + sign * dx / r * vc;
    let text = format!("{} circularized", tag(b, i));
    let (x, y) = (b.x[i], b.y[i]);
    b.log(EventKind::User, text, x, y);
    user_modified(b);
    true
}
