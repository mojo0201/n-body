//! Parents, binary hosts and orbital elements (used by the UI).

use super::*;

pub fn primary_index(b: &Bodies) -> Option<usize> {
    let mut best: Option<usize> = None;
    for i in 0..b.len() {
        if is_host_kind(b.kind[i]) && best.is_none_or(|k| b.m[i] > b.m[k]) {
            best = Some(i);
        }
    }
    best
}

pub(super) fn host_list(b: &Bodies) -> Vec<usize> {
    (0..b.len()).filter(|&k| is_host_kind(b.kind[k])).collect()
}

/// Nearest other host that `h` is gravitationally bound to, if any.
pub(super) fn binary_partner_in(b: &Bodies, hosts: &[usize], h: usize) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for &k in hosts {
        if k == h {
            continue;
        }
        let (dx, dy) = (b.x[k] - b.x[h], b.y[k] - b.y[h]);
        let d = (dx * dx + dy * dy).sqrt().max(1e-12);
        let (dvx, dvy) = (b.vx[k] - b.vx[h], b.vy[k] - b.vy[h]);
        if dvx * dvx + dvy * dvy >= 2.0 * G * (b.m[h] + b.m[k]) / d {
            continue;
        }
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((k, d));
        }
    }
    best.map(|p| p.0)
}

/// Bound stellar companion of host `h`, if any.
pub fn binary_partner(b: &Bodies, h: usize) -> Option<usize> {
    binary_partner_in(b, &host_list(b), h)
}

/// Barycentre of a host binary, seen from a body that orbits the pair as a whole.
pub(super) struct Barycentre {
    pub(super) x: f64,
    pub(super) y: f64,
    pub(super) vx: f64,
    pub(super) vy: f64,
    pub(super) m: f64,
    /// The heavier star (lower index on a tie): the parent reported for
    /// circumbinary bodies, so every body in a circumbinary disk agrees.
    pub(super) lead: usize,
}

/// If host `h` has a bound companion and body `i` is more than
/// `CIRCUMBINARY_FACTOR` separations from their barycentre, return that
/// barycentre.
pub(super) fn circumbinary_frame(
    b: &Bodies,
    hosts: &[usize],
    h: usize,
    i: usize,
) -> Option<Barycentre> {
    let q = binary_partner_in(b, hosts, h)?;
    let sep = (b.x[q] - b.x[h]).hypot(b.y[q] - b.y[h]);
    let m = b.m[h] + b.m[q];
    let x = (b.m[h] * b.x[h] + b.m[q] * b.x[q]) / m;
    let y = (b.m[h] * b.y[h] + b.m[q] * b.y[q]) / m;
    if (b.x[i] - x).hypot(b.y[i] - y) <= CIRCUMBINARY_FACTOR * sep {
        return None;
    }
    let vx = (b.m[h] * b.vx[h] + b.m[q] * b.vx[q]) / m;
    let vy = (b.m[h] * b.vy[h] + b.m[q] * b.vy[q]) / m;
    let lead = if b.m[h] > b.m[q] || (b.m[h] == b.m[q] && h < q) {
        h
    } else {
        q
    };
    Some(Barycentre {
        x,
        y,
        vx,
        vy,
        m,
        lead,
    })
}

/// Same as `circumbinary_frame` but takes a precomputed partner table.
pub(super) fn circumbinary_frame_cached(
    b: &Bodies,
    partners: &[Option<usize>],
    h: usize,
    i: usize,
) -> Option<Barycentre> {
    let q = partners.get(h).copied().flatten()?;
    let sep = (b.x[q] - b.x[h]).hypot(b.y[q] - b.y[h]);
    let m = b.m[h] + b.m[q];
    let x = (b.m[h] * b.x[h] + b.m[q] * b.x[q]) / m;
    let y = (b.m[h] * b.y[h] + b.m[q] * b.y[q]) / m;
    if (b.x[i] - x).hypot(b.y[i] - y) <= CIRCUMBINARY_FACTOR * sep {
        return None;
    }
    let vx = (b.m[h] * b.vx[h] + b.m[q] * b.vx[q]) / m;
    let vy = (b.m[h] * b.vy[h] + b.m[q] * b.vy[q]) / m;
    let lead = if b.m[h] > b.m[q] || (b.m[h] == b.m[q] && h < q) {
        h
    } else {
        q
    };
    Some(Barycentre {
        x,
        y,
        vx,
        vy,
        m,
        lead,
    })
}

/// Mass that governs the Hill sphere of body `i` about its parent host `p`:
/// the pair's total for circumbinary bodies, else the host's own mass.
pub(super) fn effective_host_mass(b: &Bodies, hosts: &[usize], i: usize, p: usize) -> f64 {
    circumbinary_frame(b, hosts, p, i).map_or(b.m[p], |bc| bc.m)
}

/// Reference frame (x, y, vx, vy, mass) for drawing the orbit of body `i`:
/// the barycentre for circumbinary bodies, its host star for S-type bodies,
/// otherwise `default` (the primary, as before).
pub(super) fn orbit_reference(
    b: &Bodies,
    hosts: &[usize],
    i: usize,
    default: (f64, f64, f64, f64, f64),
) -> (f64, f64, f64, f64, f64) {
    if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) || !b.parent_cache_valid {
        return default;
    }
    let p = match b.parent_cache.get(i).copied().flatten() {
        Some(p) => Some(p),
        None => {
            // Belt and rubble bodies have no `find_parent` entry (they are
            // deliberately excluded so they aren't labelled moons), but
            // they still orbit a host. Use the nearest one so multi-system
            // rings draw each body about its own system, not the heaviest
            // host globally.
            if b.origin[i].is_belt() || b.origin[i].is_rubble() {
                hosts.iter().copied().min_by(|&a, &c| {
                    let da = (b.x[a] - b.x[i]).powi(2) + (b.y[a] - b.y[i]).powi(2);
                    let dc = (b.x[c] - b.x[i]).powi(2) + (b.y[c] - b.y[i]).powi(2);
                    da.total_cmp(&dc)
                })
            } else {
                None
            }
        }
    };
    let Some(p) = p else {
        return default;
    };
    if p >= b.len() {
        return default;
    }
    // NEW: moons draw their orbit about the parent planet, not the star.
    if matches!(b.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet) {
        return (b.x[p], b.y[p], b.vx[p], b.vy[p], b.m[p]);
    }
    if !is_host_kind(b.kind[p]) {
        return default;
    }
    if let Some(bc) = circumbinary_frame(b, hosts, p, i) {
        return (bc.x, bc.y, bc.vx, bc.vy, bc.m);
    }
    (b.x[p], b.y[p], b.vx[p], b.vy[p], b.m[p])
}

// ---- Fix 4: cached parent finding. Direct callers (`body_class`,
// ---- `circularize`, UI) still use `find_parent`, which pays the full
// ---- cost. `refresh_parent_cache` builds a `ParentContext` once and
// ---- then uses `find_parent_cached` per body.
/// Cached context for `find_parent_cached`. Built once per refresh.
pub struct ParentContext {
    pub hosts: Vec<usize>,
    /// Indexed by body index. Only entries for hosts are populated.
    pub(super) partners: Vec<Option<usize>>,
}

pub fn build_parent_context(b: &Bodies) -> ParentContext {
    let n = b.len();
    let hosts = host_list(b);
    let mut partners = vec![None; n];
    for &h in &hosts {
        partners[h] = binary_partner_in(b, &hosts, h);
    }
    ParentContext { hosts, partners }
}

/// Find the parent of body `i` using a prebuilt context. Bit-identical
/// to `find_parent` (same candidate set, same order, same partner value).
pub(super) fn find_parent_cached(b: &Bodies, i: usize, ctx: &ParentContext) -> Option<usize> {
    let n = b.len();
    if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
        return None;
    }
    if b.origin[i].is_belt() {
        return None;
    }
    if matches!(
        b.origin[i],
        BodyOrigin::Interstellar | BodyOrigin::RoguePlanet
    ) {
        return None;
    }
    let mut host: Option<usize> = None;
    let mut best = f64::NEG_INFINITY;
    for &k in &ctx.hosts {
        if k == i {
            continue;
        }
        let d2 = (b.x[k] - b.x[i]).powi(2) + (b.y[k] - b.y[i]).powi(2) + SOFT2;
        let w = b.m[k] / d2;
        if w > best {
            best = w;
            host = Some(k);
        }
    }
    // Far outside a bound host pair, the pair acts as one parent: report the
    // heavier star so all bodies of a circumbinary disk agree.
    if let Some(h0) = host {
        if let Some(bc) = circumbinary_frame_cached(b, &ctx.partners, h0, i) {
            host = Some(bc.lead);
        }
    }
    let mut parent = host;
    let mut best_ratio = MOON_HILL_FRAC;
    if let Some(h) = host {
        for k in 0..n {
            if k == i
                || !matches!(b.kind[k], BodyKind::Planet | BodyKind::DwarfPlanet)
                || b.m[k] <= b.m[i]
            {
                continue;
            }
            let dh = ((b.x[k] - b.x[h]).powi(2) + (b.y[k] - b.y[h]).powi(2)).sqrt();
            let rh = dh * (b.m[k] / (3.0 * b.m[h])).cbrt();
            if rh <= 0.0 {
                continue;
            }
            let dk = ((b.x[k] - b.x[i]).powi(2) + (b.y[k] - b.y[i]).powi(2)).sqrt();
            let ratio = dk / rh;
            // `<=` (not `<`) so a moon spawned at exactly
            // MOON_HILL_FRAC * r_hill is recognised as a moon.
            if ratio <= best_ratio {
                best_ratio = ratio;
                parent = Some(k);
            }
        }
    }
    parent
}

pub fn find_parent(b: &Bodies, i: usize) -> Option<usize> {
    let n = b.len();
    if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
        return None;
    }
    if b.origin[i].is_belt() {
        return None;
    }
    // Unbound visitors have no parent — they're on hyperbolic
    // trajectories that pass through the system. Reporting the primary
    // as a "parent" would make the UI show them as orbiting the star,
    // which is misleading. `host_distance` falls back to the origin
    // distance, which is the same value for a centred primary.
    if matches!(
        b.origin[i],
        BodyOrigin::Interstellar | BodyOrigin::RoguePlanet
    ) {
        return None;
    }
    let mut host: Option<usize> = None;
    let mut best = f64::NEG_INFINITY;
    for k in 0..n {
        if k == i || !is_host_kind(b.kind[k]) {
            continue;
        }
        let d2 = (b.x[k] - b.x[i]).powi(2) + (b.y[k] - b.y[i]).powi(2) + SOFT2;
        let w = b.m[k] / d2;
        if w > best {
            best = w;
            host = Some(k);
        }
    }
    // Far outside a bound host pair, the pair acts as one parent: report the
    // heavier star so all bodies of a circumbinary disk agree.
    if let Some(h0) = host {
        let hosts = host_list(b);
        if let Some(bc) = circumbinary_frame(b, &hosts, h0, i) {
            host = Some(bc.lead);
        }
    }
    let mut parent = host;
    let mut best_ratio = MOON_HILL_FRAC;
    if let Some(h) = host {
        for k in 0..n {
            if k == i
                || !matches!(b.kind[k], BodyKind::Planet | BodyKind::DwarfPlanet)
                || b.m[k] <= b.m[i]
            {
                continue;
            }
            let dh = ((b.x[k] - b.x[h]).powi(2) + (b.y[k] - b.y[h]).powi(2)).sqrt();
            let rh = dh * (b.m[k] / (3.0 * b.m[h])).cbrt();
            if rh <= 0.0 {
                continue;
            }
            let dk = ((b.x[k] - b.x[i]).powi(2) + (b.y[k] - b.y[i]).powi(2)).sqrt();
            let ratio = dk / rh;
            if ratio <= best_ratio {
                best_ratio = ratio;
                parent = Some(k);
            }
        }
    }
    parent
}

pub struct OrbitInfo {
    pub parent: usize,
    pub dist: f64,
    pub speed_rel: f64,
    pub bound: bool,
    pub semi_major: f64,
    pub ecc: f64,
    pub period: f64,
    pub peri: f64,
    pub apo: f64,
}

pub fn orbit_info(b: &Bodies, i: usize, par: usize) -> OrbitInfo {
    let dx = b.x[i] - b.x[par];
    let dy = b.y[i] - b.y[par];
    let dvx = b.vx[i] - b.vx[par];
    let dvy = b.vy[i] - b.vy[par];
    let r = (dx * dx + dy * dy).sqrt().max(1e-12);
    let v2 = dvx * dvx + dvy * dvy;
    let mu = G * (b.m[par] + b.m[i]);
    let rv = dx * dvx + dy * dvy;
    let ex = ((v2 - mu / r) * dx - rv * dvx) / mu;
    let ey = ((v2 - mu / r) * dy - rv * dvy) / mu;
    let e = (ex * ex + ey * ey).sqrt();
    let inv = 2.0 / r - v2 / mu;
    let bound = inv > 1e-12;
    let a = if bound { 1.0 / inv } else { f64::INFINITY };
    let period = if bound {
        std::f64::consts::TAU * (a * a * a / mu).sqrt()
    } else {
        f64::INFINITY
    };
    OrbitInfo {
        parent: par,
        dist: r,
        speed_rel: v2.sqrt(),
        bound,
        semi_major: a,
        ecc: e,
        period,
        peri: if bound { a * (1.0 - e) } else { f64::NAN },
        apo: if bound { a * (1.0 + e) } else { f64::NAN },
    }
}

/// Distance from body `i` to the host star it orbits, in Gm. A moon uses the
/// host of its planet. Falls back to the distance from the origin when the
/// parent cache is stale. O(1), so the renderer can call it for every body.
pub fn host_distance(b: &Bodies, i: usize) -> f64 {
    let fallback = b.x[i].hypot(b.y[i]);
    if !b.parent_cache_valid {
        return fallback;
    }
    let mut p = b.parent_cache.get(i).copied().flatten();
    if let Some(q) = p {
        if q < b.len() && !is_host_kind(b.kind[q]) {
            p = b.parent_cache.get(q).copied().flatten();
        }
    }
    match p {
        Some(h) if h < b.len() && is_host_kind(b.kind[h]) => {
            (b.x[i] - b.x[h]).hypot(b.y[i] - b.y[h])
        }
        _ => fallback,
    }
}
