//! Parents, binary hosts and orbital elements (used by the UI).

use super::*;

/// Heaviest host in the system (lowest index on a tie). The default
/// reference body for anything that isn't orbiting something more specific.
pub fn primary_index(b: &Bodies) -> Option<usize> {
    let mut best: Option<usize> = None;
    for i in 0..b.len() {
        if !is_host_kind(b.kind[i]) {
            continue;
        }
        if best.is_none_or(|k| b.m[i] > b.m[k]) {
            best = Some(i);
        }
    }
    best
}

pub(super) fn host_list(b: &Bodies) -> Vec<usize> {
    (0..b.len()).filter(|&k| is_host_kind(b.kind[k])).collect()
}

/// Nearest other host that `h` is bound to, if any. Two bodies are bound
/// when their relative speed is below the two-body escape speed
/// `sqrt(2G(m_h + m_k) / d)`; among bound partners we return the closest.
pub(super) fn binary_partner_in(b: &Bodies, hosts: &[usize], h: usize) -> Option<usize> {
    let mh = b.m[h];
    let (xh, yh) = (b.x[h], b.y[h]);
    let (vxh, vyh) = (b.vx[h], b.vy[h]);
    let (zh, vzh) = (b.z[h], b.vz[h]);
    let mut best: Option<(usize, f64)> = None;
    for &k in hosts {
        if k == h {
            continue;
        }
        let (dx, dy, dz) = (b.x[k] - xh, b.y[k] - yh, b.z[k] - zh);
        let d = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);
        let (dvx, dvy, dvz) = (b.vx[k] - vxh, b.vy[k] - vyh, b.vz[k] - vzh);
        // v² >= escape² ⇒ unbound, skip.
        if dvx * dvx + dvy * dvy + dvz * dvz >= 2.0 * G * (mh + b.m[k]) / d {
            continue;
        }
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((k, d));
        }
    }
    best.map(|p| p.0)
}

/// Allocation-free `binary_partner_in` over the full body array. Used by
/// call sites that would otherwise build a temporary `host_list`.
pub fn binary_partner(b: &Bodies, h: usize) -> Option<usize> {
    let mh = b.m[h];
    let (xh, yh) = (b.x[h], b.y[h]);
    let (vxh, vyh) = (b.vx[h], b.vy[h]);
    let (zh, vzh) = (b.z[h], b.vz[h]);
    let mut best: Option<(usize, f64)> = None;
    for k in 0..b.len() {
        if k == h || !is_host_kind(b.kind[k]) {
            continue;
        }
        let (dx, dy, dz) = (b.x[k] - xh, b.y[k] - yh, b.z[k] - zh);
        let d = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);
        let (dvx, dvy, dvz) = (b.vx[k] - vxh, b.vy[k] - vyh, b.vz[k] - vzh);
        if dvx * dvx + dvy * dvy + dvz * dvz >= 2.0 * G * (mh + b.m[k]) / d {
            continue;
        }
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((k, d));
        }
    }
    best.map(|p| p.0)
}

/// Centre of mass and momentum of a bound host pair, plus the star that
/// circumbinary bodies should report as their parent.
pub(super) struct Barycentre {
    pub(super) x: f64,
    pub(super) y: f64,
    pub(super) z: f64,
    pub(super) vx: f64,
    pub(super) vy: f64,
    pub(super) vz: f64,
    pub(super) m: f64,
    /// Heavier star of the pair (lower index on a tie). Report this as the
    /// parent so every body in a circumbinary disk agrees on a single host.
    pub(super) lead: usize,
}

/// Shared core of `circumbinary_frame` and `circumbinary_frame_cached`:
/// builds the barycentre of `(h, q)` and returns it only if `i` lies
/// clearly outside the pair (more than `CIRCUMBINARY_FACTOR` separations
/// from the barycentre). Inside that radius the body orbits one star,
/// not the pair.
#[inline]
fn barycentre_from(b: &Bodies, h: usize, q: usize, i: usize) -> Option<Barycentre> {
    let (xh, yh, vxh, vyh, mh) = (b.x[h], b.y[h], b.vx[h], b.vy[h], b.m[h]);
    let (xq, yq, vxq, vyq, mq) = (b.x[q], b.y[q], b.vx[q], b.vy[q], b.m[q]);
    let (zh, zq, vzh, vzq) = (b.z[h], b.z[q], b.vz[h], b.vz[q]);
    let sep = (xq - xh).hypot(yq - yh).hypot(zq - zh);
    let m = mh + mq;
    let x = (mh * xh + mq * xq) / m;
    let y = (mh * yh + mq * yq) / m;
    let z = (mh * zh + mq * zq) / m;
    if (b.x[i] - x).hypot(b.y[i] - y).hypot(b.z[i] - z) <= CIRCUMBINARY_FACTOR * sep {
        return None;
    }
    let vx = (mh * vxh + mq * vxq) / m;
    let vy = (mh * vyh + mq * vyq) / m;
    let vz = (mh * vzh + mq * vzq) / m;
    let lead = if mh > mq || (mh == mq && h < q) { h } else { q };
    Some(Barycentre {
        x,
        y,
        z,
        vx,
        vy,
        vz,
        m,
        lead,
    })
}

/// If host `h` has a bound companion and body `i` orbits the pair as a
/// whole (rather than one star), return the pair's barycentre. Callers
/// that already hold `ParentContext::partners` should use
/// `circumbinary_frame_cached` instead.
pub(super) fn circumbinary_frame(
    b: &Bodies,
    hosts: &[usize],
    h: usize,
    i: usize,
) -> Option<Barycentre> {
    let q = binary_partner_in(b, hosts, h)?;
    barycentre_from(b, h, q, i)
}

/// Allocation-free `circumbinary_frame` for callers without a `hosts`
/// slice; trades the O(H) scan for an O(N) scan but skips the `Vec` alloc.
fn circumbinary_frame_noslice(b: &Bodies, h: usize, i: usize) -> Option<Barycentre> {
    let q = binary_partner(b, h)?;
    barycentre_from(b, h, q, i)
}

/// Cached `circumbinary_frame`: partner of `h` comes from the precomputed
/// table, making the lookup O(1).
pub(super) fn circumbinary_frame_cached(
    b: &Bodies,
    partners: &[Option<usize>],
    h: usize,
    i: usize,
) -> Option<Barycentre> {
    let q = partners.get(h).copied().flatten()?;
    barycentre_from(b, h, q, i)
}

/// Mass that governs the Hill sphere of body `i` about host `p`: the
/// pair's total for circumbinary bodies, else the host's own mass.
pub(super) fn effective_host_mass(b: &Bodies, hosts: &[usize], i: usize, p: usize) -> f64 {
    circumbinary_frame(b, hosts, p, i).map_or(b.m[p], |bc| bc.m)
}

/// Cached `effective_host_mass`; see `circumbinary_frame_cached`.
pub(super) fn effective_host_mass_cached(
    b: &Bodies,
    partners: &[Option<usize>],
    i: usize,
    p: usize,
) -> f64 {
    circumbinary_frame_cached(b, partners, p, i).map_or(b.m[p], |bc| bc.m)
}

/// Shared implementation of `orbit_reference` and its cached variant.
/// `cbf(p)` must return the barycentre of `p`'s binary partner iff `i`
/// orbits the pair, and `None` otherwise.
fn orbit_reference_impl<F>(
    b: &Bodies,
    hosts: &[usize],
    i: usize,
    default: (f64, f64, f64, f64, f64, f64, f64),
    cbf: F,
) -> (f64, f64, f64, f64, f64, f64, f64)
where
    F: FnOnce(usize) -> Option<Barycentre>,
{
    if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) || !b.parent_cache_valid {
        return default;
    }
    let p = match b.parent_cache.get(i).copied().flatten() {
        Some(p) => p,
        None => {
            // Belt and rubble bodies have no `find_parent` entry (that
            // would label them as moons), but they still orbit a host.
            // Attribute each to the nearest host so multi-system rings
            // draw about their own system, not the heaviest host globally.
            if b.origin[i].is_belt() || b.origin[i].is_rubble() {
                let mut best: Option<(usize, f64)> = None;
                for &k in hosts {
                    let d2 = (b.x[k] - b.x[i]).powi(2) + (b.y[k] - b.y[i]).powi(2) + (b.z[k] - b.z[i]).powi(2);
                    if best.is_none_or(|(_, bd)| d2 < bd) {
                        best = Some((k, d2));
                    }
                }
                match best {
                    Some((k, _)) => k,
                    None => return default,
                }
            } else {
                return default;
            }
        }
    };
    if p >= b.len() {
        return default;
    }
    // A moon draws its orbit about the parent planet, not the star.
    if matches!(b.kind[p], BodyKind::Planet | BodyKind::DwarfPlanet) {
        return (b.x[p], b.y[p], b.z[p], b.vx[p], b.vy[p], b.vz[p], b.m[p]);
    }
    if !is_host_kind(b.kind[p]) {
        return default;
    }
    if let Some(bc) = cbf(p) {
        return (bc.x, bc.y, bc.z, bc.vx, bc.vy, bc.vz, bc.m);
    }
    (b.x[p], b.y[p], b.z[p], b.vx[p], b.vy[p], b.vz[p], b.m[p])
}

/// Reference frame (x, y, z, vx, vy, vz, mass) for drawing the orbit of body `i`:
/// the pair's barycentre for circumbinary bodies, the host star for S-type
/// bodies, otherwise `default` (normally the primary).
pub(super) fn orbit_reference(
    b: &Bodies,
    hosts: &[usize],
    i: usize,
    default: (f64, f64, f64, f64, f64, f64, f64),
) -> (f64, f64, f64, f64, f64, f64, f64) {
    orbit_reference_impl(b, hosts, i, default, |p| circumbinary_frame(b, hosts, p, i))
}

/// Cached `orbit_reference`: same result, but the binary-partner lookup
/// is O(1) per body instead of O(H). Preferred by the renderer, which
/// calls this once per body per frame.
pub(super) fn orbit_reference_cached(
    b: &Bodies,
    hosts: &[usize],
    partners: &[Option<usize>],
    i: usize,
    default: (f64, f64, f64, f64, f64, f64, f64),
) -> (f64, f64, f64, f64, f64, f64, f64) {
    orbit_reference_impl(b, hosts, i, default, |p| {
        circumbinary_frame_cached(b, partners, p, i)
    })
}

/// Per-refresh cache for `find_parent_cached`. Building this once turns
/// parent finding for the whole system from O(N·H + N²) into O(H² + N·P),
/// where H = hosts, P = potential moon parents.
pub struct ParentContext {
    /// All host bodies, ascending.
    pub hosts: Vec<usize>,
    /// Binary partner of each host, indexed by body index. Entries for
    /// non-host bodies are always `None`.
    pub(super) partners: Vec<Option<usize>>,
    /// Bodies eligible as moon parents (`Planet | DwarfPlanet`), ascending.
    /// Ascending order is required: `find_parent_cached` tie-breaks on
    /// `<=`, so it must see candidates in the same order as `find_parent`.
    pub(super) moons: Vec<usize>,
}

pub fn build_parent_context(b: &Bodies) -> ParentContext {
    let n = b.len();
    let mut hosts = Vec::new();
    let mut moons = Vec::new();
    for i in 0..n {
        if is_host_kind(b.kind[i]) {
            hosts.push(i);
        }
        if matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
            moons.push(i);
        }
    }
    let mut partners = vec![None; n];
    for &h in &hosts {
        partners[h] = binary_partner_in(b, &hosts, h);
    }
    ParentContext {
        hosts,
        partners,
        moons,
    }
}

/// Parent of body `i`, using a prebuilt `ParentContext`. Contract: the
/// result must be bit-identical to `find_parent(b, i)` for the same
/// `Bodies` (same candidate set, same iteration order, same partner
/// value). Any change here must be mirrored there.
pub(super) fn find_parent_cached(b: &Bodies, i: usize, ctx: &ParentContext) -> Option<usize> {
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
        // `SOFT2` keeps the weight finite when a body coincides with a host.
        let d2 = (b.x[k] - b.x[i]).powi(2) + (b.y[k] - b.y[i]).powi(2) + (b.z[k] - b.z[i]).powi(2) + SOFT2;
        let w = b.m[k] / d2;
        if w > best {
            best = w;
            host = Some(k);
        }
    }
    // Well outside a bound host pair, the pair acts as one parent: report
    // the heavier star so a circumbinary disk agrees on one host.
    if let Some(h0) = host {
        if let Some(bc) = circumbinary_frame_cached(b, &ctx.partners, h0, i) {
            host = Some(bc.lead);
        }
    }
    let mut parent = host;
    let mut best_ratio = MOON_HILL_FRAC;
    if let Some(h) = host {
        for &k in &ctx.moons {
            if k == i || b.m[k] <= b.m[i] {
                continue;
            }
            let dh = ((b.x[k] - b.x[h]).powi(2) + (b.y[k] - b.y[h]).powi(2) + (b.z[k] - b.z[h]).powi(2)).sqrt();
            let rh = dh * (b.m[k] / (3.0 * b.m[h])).cbrt();
            if rh <= 0.0 {
                continue;
            }
            let dk = ((b.x[k] - b.x[i]).powi(2) + (b.y[k] - b.y[i]).powi(2) + (b.z[k] - b.z[i]).powi(2)).sqrt();
            let ratio = dk / rh;
            // `<=` (not `<`) so a moon spawned at exactly
            // MOON_HILL_FRAC * r_hill is classified as a moon.
            if ratio <= best_ratio {
                best_ratio = ratio;
                parent = Some(k);
            }
        }
    }
    parent
}

/// Uncached parent lookup. See `find_parent_cached` for the algorithm;
/// this version pays the full O(N·H) cost and should only be used by
/// callers that genuinely need a single ad-hoc lookup (`body_class`,
/// `circularize`, tests).
pub fn find_parent(b: &Bodies, i: usize) -> Option<usize> {
    let n = b.len();
    if !matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet) {
        return None;
    }
    if b.origin[i].is_belt() {
        return None;
    }
    // Interstellar / rogue bodies are unbound visitors on hyperbolic
    // trajectories. Reporting the primary as their parent would make the
    // UI draw them as orbiting the star; `host_distance` instead falls
    // back to distance from the origin, which is correct for a centred
    // primary.
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
        let d2 = (b.x[k] - b.x[i]).powi(2) + (b.y[k] - b.y[i]).powi(2) + (b.z[k] - b.z[i]).powi(2) + SOFT2;
        let w = b.m[k] / d2;
        if w > best {
            best = w;
            host = Some(k);
        }
    }
    if let Some(h0) = host {
        if let Some(bc) = circumbinary_frame_noslice(b, h0, i) {
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
            let dh = ((b.x[k] - b.x[h]).powi(2) + (b.y[k] - b.y[h]).powi(2) + (b.z[k] - b.z[h]).powi(2)).sqrt();
            let rh = dh * (b.m[k] / (3.0 * b.m[h])).cbrt();
            if rh <= 0.0 {
                continue;
            }
            let dk = ((b.x[k] - b.x[i]).powi(2) + (b.y[k] - b.y[i]).powi(2) + (b.z[k] - b.z[i]).powi(2)).sqrt();
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
    /// Inclination of the orbit to the xy plane, radians in [0, π].
    pub inc: f64,
}

/// Osculating two-body elements of body `i` relative to `par`. Hyperbolic
/// orbits report `semi_major = period = ∞`, `peri = apo = NaN`, and
/// `bound = false`.
pub fn orbit_info(b: &Bodies, i: usize, par: usize) -> OrbitInfo {
    let dx = b.x[i] - b.x[par];
    let dy = b.y[i] - b.y[par];
    let dz = b.z[i] - b.z[par];
    let dvx = b.vx[i] - b.vx[par];
    let dvy = b.vy[i] - b.vy[par];
    let dvz = b.vz[i] - b.vz[par];
    let r = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);
    let v2 = dvx * dvx + dvy * dvy + dvz * dvz;
    let mu = G * (b.m[par] + b.m[i]);
    let rv = dx * dvx + dy * dvy + dz * dvz;
    // Eccentricity vector e = ((v² - μ/r)·r⃗ - (r⃗·v⃗)·v⃗) / μ.
    let ex = ((v2 - mu / r) * dx - rv * dvx) / mu;
    let ey = ((v2 - mu / r) * dy - rv * dvy) / mu;
    let ez = ((v2 - mu / r) * dz - rv * dvz) / mu;
    let e = (ex * ex + ey * ey + ez * ez).sqrt();
    // Orbit normal h = r × v gives the inclination to the xy reference plane
    // (0 for a prograde planar orbit, π for a retrograde one).
    let (hx, hy, hz) = (dy * dvz - dz * dvy, dz * dvx - dx * dvz, dx * dvy - dy * dvx);
    let hn = (hx * hx + hy * hy + hz * hz).sqrt();
    let inc = if hn > 0.0 { (hz / hn).clamp(-1.0, 1.0).acos() } else { 0.0 };
    // Vis-viva: 1/a = 2/r - v²/μ. Positive ⇒ bound ellipse.
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
        inc,
    }
}

/// Distance from body `i` to the host star it orbits, in Gm. Moons are
/// resolved to their planet's host. O(1), so the renderer can call it for
/// every body per frame. Falls back to distance-from-origin when the
/// parent cache is stale or the body has no host (e.g. interstellar).
pub fn host_distance(b: &Bodies, i: usize) -> f64 {
    let fallback = b.x[i].hypot(b.y[i]).hypot(b.z[i]);
    if !b.parent_cache_valid {
        return fallback;
    }
    let mut p = b.parent_cache.get(i).copied().flatten();
    // One hop up: moon → planet → host star.
    if let Some(q) = p {
        if q < b.len() && !is_host_kind(b.kind[q]) {
            p = b.parent_cache.get(q).copied().flatten();
        }
    }
    match p {
        Some(h) if h < b.len() && is_host_kind(b.kind[h]) => {
            (b.x[i] - b.x[h]).hypot(b.y[i] - b.y[h]).hypot(b.z[i] - b.z[h])
        }
        _ => fallback,
    }
}
