//! Collisionless dust as tracer grains.
//!
//! Loaded as `physics::dust` via `mod dust;` in `physics.rs`, so
//! `use super::*;` sees the private helpers (`kepler_state`, `host_list`...).
//!
//! Each grain is a test particle with a mass weight and a size. It feels the
//! primary (reduced by radiation pressure beta), Poynting-Robertson drag, and
//! the gravity of the other hosts and the massive planets. Grains are removed
//! when they hit the star or a planet, or leave the cull radius (blow-out).
//!
//! Approximations (all documented, none hidden):
//!  * Perturbers move on straight lines inside one dust step (dt ~ 0.033 tu).
//!  * Only the primary radiates; a companion star only gravitates.
//!  * PR drag is sped up by `PR_DRAG_BOOST` (c_eff = c / boost), like before.
//!  * Tracers are not number-faithful: sizes are log-uniform, `mass` is the
//!    mass each tracer stands for. The masses follow a collisional cascade
//!    (dn/ds ~ s^-3.5, Dohnanyi 1969): a tracer of size s stands for
//!    s^0.5 of the mass, because the tracers are log-uniform in size.

use super::*;

/// Tracers released per belt/rubble body at start-up (capped by the budget).
pub const DUST_GRAINS_PER_BODY: usize = 60;
/// Hard cap per field.
pub const DUST_MAX_GRAINS: usize = 256_000;
/// Tracers emitted by one collision.
pub const DUST_EMIT_GRAINS: usize = 80;
/// Ejection speed of collision dust, as a fraction of local circular speed.
pub const DUST_EJECT_FRAC: f64 = 0.15;
/// Grain size range in micrometres (log-uniform). The blow-out size for the
/// Sun is ~0.5 um (beta = 0.57 L/M / (rho s)); for Vega it is ~9 um.
pub const DUST_SIZE_UM: (f64, f64) = (0.3, 1000.0);
pub const DUST_GRAIN_RHO: f64 = 2.5;
/// Cull radius as a multiple of the field extent.
const CULL_FACTOR: f64 = 4.0;
const SUB_PER_ORBIT: f64 = 48.0;
const MAX_SUBSTEPS: u32 = 64;
const MAX_PERTURBERS: usize = 24;
/// Below this many grains, plain sequential iteration is faster than the
/// rayon + `Vec<bool>` compaction path: for small fields the parallel
/// dispatch and the bitset allocation cost more than the per-grain work.
pub const SEQUENTIAL_GRAIN_THRESHOLD: usize = 2048;
/// Fraction of a belt body's mass ground to dust per orbit, for a belt of
/// mass `BELT_GRIND_M_REF`. Real belts lose far less per orbit; this is
/// boosted (like `PR_DRAG_BOOST`) so the grinding shows in a run. The rate
/// scales with the mass of the belt (collision rate ~ number density), clamped
/// to 0.01-100 times the reference.
pub const BELT_GRIND_PER_ORBIT: f64 = 1e-5;
/// Reference belt mass (Earth masses): the Kuiper belt.
pub const BELT_GRIND_M_REF: f64 = 0.03;
/// A grinding event removes this fraction of the body's mass.
const BELT_GRIND_CHUNK: f64 = 0.02;
/// Bodies lighter than this (Earth masses) are not ground further.
pub const BELT_GRIND_MIN_M: f64 = 1e-9;
/// Planets lighter than this (Earth masses) do not perturb grains.
const PERT_MIN_M: f64 = 0.01;
/// Mean lifetime of a comet-tail tracer, in time units (1 tu is about 50
/// days, so 12 tu is about 1.6 yr). Each grain draws 0.5-1.5 times this, so
/// the tail thins out gradually instead of vanishing at once.
pub const COMET_DUST_LIFE_TU: f64 = 12.0;
/// Lifetime of belt-grinding and collision dust. Infinite: the belts keep
/// feeding it, and it still leaves by hitting a body or crossing the cull radius.
pub const NONCOMET_DUST_LIFE_TU: f64 = f64::INFINITY;
/// Comets stop emitting tracers while a field already holds this many, which
/// bounds the per-step cost no matter how many comets are active.
pub const DUST_COMET_FIELD_CAP: usize = 100_000;

#[derive(Clone, Copy, Debug)]
pub struct Grain {
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    /// Earth masses this tracer represents.
    pub mass: f64,
    pub size_um: f64,
    /// Remaining lifetime in time units. `f32::INFINITY` never expires.
    pub ttl: f32,
}

#[derive(Clone, Debug)]
pub struct DustField {
    pub primary: usize,
    pub extent_gm: f64,
    pub grains: Vec<Grain>,
    /// Bumped on every change, for the renderer.
    pub version: u64,
}

/// One point for the renderer: camera-relative position and mass weight.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DustPoint {
    pub pos: [f32; 2],
    pub weight: f32,
}

#[derive(Clone, Copy)]
struct Pert {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    gm: f64,
    rc: f64,
}

#[derive(Clone, Copy)]
struct Star {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    gm: f64,
    r: f64,
    /// (L / L_sun) / (M / M_sun), for beta.
    l_over_m: f64,
}

/// Relative mass of a tracer of size `size_um`. In a collisional cascade
/// (dn/ds ~ s^-3.5) the mass per logarithmic size bin goes as s^0.5, and the
/// tracer sizes are log-uniform, so each tracer carries a weight of s^0.5.
fn size_weight(size_um: f64) -> f64 {
    size_um.sqrt()
}

fn draw_size(rng: &mut XorShift) -> f64 {
    let (lo, hi) = DUST_SIZE_UM;
    lo * (hi / lo).powf(rng.next_f64())
}

/// Radiation-pressure ratio beta = F_rad / F_grav (Q_pr = 1).
pub fn beta_of(l_over_m: f64, size_um: f64) -> f64 {
    0.57 * l_over_m / (DUST_GRAIN_RHO * size_um)
}

fn collect_perturbers(b: &Bodies, pri: usize, reach: f64) -> Vec<Pert> {
    let mut v: Vec<Pert> = (0..b.len())
        .filter(|&i| i != pri && !b.dead[i])
        .filter(|&i| {
            is_host_kind(b.kind[i])
                || (matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
                    && !b.origin[i].is_belt()
                    && !b.origin[i].is_rubble()
                    && b.m[i] >= PERT_MIN_M)
        })
        .filter(|&i| (b.x[i] - b.x[pri]).hypot(b.y[i] - b.y[pri]) < reach)
        .map(|i| Pert {
            x: b.x[i],
            y: b.y[i],
            vx: b.vx[i],
            vy: b.vy[i],
            gm: G * b.m[i],
            rc: b.contact_r(i),
        })
        .collect();
    v.sort_by(|a, c| c.gm.total_cmp(&a.gm));
    v.truncate(MAX_PERTURBERS);
    v
}

/// Acceleration of a grain at time `tau` into the step. Star term follows
/// Burns et al. (1979): a = beta GM/r^2 [(1 - rdot/c) rhat - v/c] - GM/r^2 rhat.
#[inline]
fn accel(g: &Grain, beta: f64, st: &Star, pert: &[Pert], tau: f64, c: f64) -> (f64, f64) {
    let (sx, sy) = (st.x + st.vx * tau, st.y + st.vy * tau);
    let (rx, ry) = (g.x - sx, g.y - sy);
    let r2 = rx * rx + ry * ry + SOFT2;
    let inv_r = r2.sqrt().recip();
    let (hx, hy) = (rx * inv_r, ry * inv_r); // one reciprocal, two muls
    let (dvx, dvy) = (g.vx - st.vx, g.vy - st.vy);
    let rdot = hx * dvx + hy * dvy;
    let f = st.gm / r2;
    let bf = beta * f; // shared by both components below
    let mut ax = -f * hx + bf * ((1.0 - rdot / c) * hx - dvx / c);
    let mut ay = -f * hy + bf * ((1.0 - rdot / c) * hy - dvy / c);
    for p in pert {
        let dx = g.x - (p.x + p.vx * tau);
        let dy = g.y - (p.y + p.vy * tau);
        let d2 = dx * dx + dy * dy + SOFT2;
        let k = p.gm / (d2 * d2.sqrt());
        ax -= k * dx;
        ay -= k * dy;
    }
    (ax, ay)
}

/// Advance one grain by `dt` with adaptive sub-steps (kick-drift-kick).
/// Returns false if the grain was removed.
fn step_grain(g: &mut Grain, st: &Star, pert: &[Pert], dt: f64, c: f64, cull2: f64) -> bool {
    g.ttl -= dt as f32; // INFINITY stays INFINITY
    if g.ttl <= 0.0 {
        return false;
    }
    let beta = beta_of(st.l_over_m, g.size_um);
    let r = (g.x - st.x).hypot(g.y - st.y).max(1e-6);
    let mut t_dyn = (r.powi(3) / st.gm).sqrt();
    for p in pert {
        let d = (g.x - p.x).hypot(g.y - p.y).max(p.rc);
        t_dyn = t_dyn.min((d.powi(3) / p.gm).sqrt());
    }
    let n = ((dt * SUB_PER_ORBIT / (std::f64::consts::TAU * t_dyn)).ceil() as u32)
        .clamp(1, MAX_SUBSTEPS);
    let h = dt / n as f64;

    // KDK leapfrog: the accel evaluated at the end of one sub-step is exactly
    // the accel the next sub-step would recompute at its start, so carry it
    // forward and halve the number of `accel` calls.
    let (mut ax, mut ay) = accel(g, beta, st, pert, 0.0, c);
    for k in 0..n {
        let t1 = (k + 1) as f64 * h;
        g.vx += 0.5 * h * ax;
        g.vy += 0.5 * h * ay;
        g.x += h * g.vx;
        g.y += h * g.vy;
        let (ax2, ay2) = accel(g, beta, st, pert, t1, c);
        g.vx += 0.5 * h * ax2;
        g.vy += 0.5 * h * ay2;
        ax = ax2;
        ay = ay2;

        let (rx, ry) = (g.x - (st.x + st.vx * t1), g.y - (st.y + st.vy * t1));
        let r2 = rx * rx + ry * ry;
        if r2 < st.r * st.r || r2 > cull2 {
            return false;
        }
        for p in pert {
            let (dx, dy) = (g.x - (p.x + p.vx * t1), g.y - (p.y + p.vy * t1));
            if dx * dx + dy * dy < p.rc * p.rc {
                return false;
            }
        }
    }
    true
}

impl DustField {
    /// Advance every grain by `dt` (time units).
    ///
    /// Small fields stay sequential: the rayon dispatch + the `Vec<bool>`
    /// compaction cost more than the work they parallelise. Large fields
    /// keep the parallel integrator and one compaction pass.
    pub fn step(&mut self, b: &Bodies, dt: f64) {
        let pri = self.primary;
        if pri >= b.len() {
            return;
        }
        let m_sun = (b.m[pri] / M_SUN_ME).max(1e-9);
        let st = Star {
            x: b.x[pri],
            y: b.y[pri],
            vx: b.vx[pri],
            vy: b.vy[pri],
            gm: G * b.m[pri],
            r: b.contact_r(pri),
            l_over_m: luminosity_of(b, pri) / m_sun,
        };
        let cull = CULL_FACTOR * self.extent_gm;
        let pert = collect_perturbers(b, pri, cull);
        let c = c_sim() / PR_DRAG_BOOST;
        let cull2 = cull * cull;

        if self.grains.len() < SEQUENTIAL_GRAIN_THRESHOLD {
            self.grains
                .retain_mut(|g| step_grain(g, &st, &pert, dt, c, cull2));
        } else {
            let alive: Vec<bool> = self
                .grains
                .par_iter_mut()
                .map(|g| step_grain(g, &st, &pert, dt, c, cull2))
                .collect();
            let mut it = alive.into_iter();
            self.grains.retain(|_| it.next().unwrap_or(true));
        }
        self.version = self.version.wrapping_add(1);
    }

    fn emit_dir(
        &mut self,
        p: (f64, f64, f64, f64),
        mass: f64,
        kick: f64,
        dir: Option<(f64, f64)>,
        rng: &mut XorShift,
    ) -> usize {
        // `n` is a const, so the scratch arrays live on the stack.
        let n = DUST_EMIT_GRAINS;
        if self.grains.len() + n > DUST_MAX_GRAINS || mass <= 0.0 {
            return 0;
        }
        // `dir` is set only for comet tails. Cap them lower than the rest.
        if dir.is_some() && self.grains.len() + n > DUST_COMET_FIELD_CAP {
            return 0;
        }
        let mut sizes = [0.0f64; DUST_EMIT_GRAINS];
        let mut weights = [0.0f64; DUST_EMIT_GRAINS];
        let mut wsum = 0.0f64;
        for k in 0..n {
            let s = draw_size(rng);
            let w = size_weight(s);
            sizes[k] = s;
            weights[k] = w;
            wsum += w;
        }

        for k in 0..n {
            let size = sizes[k];
            let weight = weights[k];
            let (dvx, dvy) = if let Some((ux, uy)) = dir {
                // 1. Unit vector along the comet's current velocity.
                let speed = p.2.hypot(p.3).max(1e-9);
                let (vx_hat, vy_hat) = (p.2 / speed, p.3 / speed);

                // 2. Blend the anti-sunward direction with the backward
                // velocity. A higher sweep strongly tilts the cone backward.
                let sweep = 1.5;
                let base_x = ux - vx_hat * sweep;
                let base_y = uy - vy_hat * sweep;
                let base_len = base_x.hypot(base_y).max(1e-9);
                let (bx, by) = (base_x / base_len, base_y / base_len);

                // 3. Random cone spread makes the tail fan out.
                let spread = 0.8;
                let ang = sample_normal(rng, 0.0, spread * 0.5);
                let (sa, ca) = ang.sin_cos();
                let tx = bx * ca - by * sa;
                let ty = bx * sa + by * ca;

                // 4. Small ejection speed: radiation pressure (beta) does
                // most of the work, so the dust lags behind and curves.
                let v_kick = kick * (0.01 + 0.05 * rng.next_f64());
                (tx * v_kick, ty * v_kick)
            } else {
                // Isotropic (collisions).
                (sample_normal(rng, 0.0, kick), sample_normal(rng, 0.0, kick))
            };

            // Offset the initial position along the tail direction.
            // Dust is only stepped every `DUST_STEP_INTERVAL`, so without
            // this offset the un-stepped dust forms a flat wall at the
            // source position; spreading it out along the tail makes the
            // leading edge pointy instead of squared-off.
            let (px, py) = if let Some((ux, uy)) = dir {
                let spread_dist = rng.next_f64() * 0.05;
                (p.0 + ux * spread_dist, p.1 + uy * spread_dist)
            } else {
                (p.0, p.1)
            };

            self.grains.push(Grain {
                x: px,
                y: py,
                vx: p.2 + dvx,
                vy: p.3 + dvy,
                mass: mass * weight / wsum,
                size_um: size,
                ttl: if dir.is_some() {
                    (COMET_DUST_LIFE_TU * (0.5 + rng.next_f64())) as f32
                } else {
                    NONCOMET_DUST_LIFE_TU as f32
                },
            });
        }
        self.version = self.version.wrapping_add(1);
        n
    }

    fn emit(&mut self, p: (f64, f64, f64, f64), mass: f64, kick: f64, rng: &mut XorShift) -> usize {
        self.emit_dir(p, mass, kick, None, rng)
    }
}

pub fn deposit(
    b: &mut Bodies,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    mass: f64,
    dir: Option<(f64, f64)>,
    kick_scale: f64,
) -> usize {
    let mut fields = std::mem::take(&mut b.dust_fields);
    let mut rng = std::mem::replace(&mut b.rng, XorShift::new(1));
    let mut added = 0usize;
    for f in fields.iter_mut() {
        let p = f.primary;
        if p >= b.len() {
            continue;
        }
        let r = (x - b.x[p]).hypot(y - b.y[p]);
        if r < f.extent_gm {
            let vc = (G * b.m[p] / r.max(1e-6)).sqrt();
            let kick = DUST_EJECT_FRAC * vc * kick_scale;
            added = f.emit_dir((x, y, vx, vy), mass, kick, dir, &mut rng);
            break;
        }
    }
    b.rng = rng;
    b.dust_fields = fields;
    added
}

/// (a, e, periapsis angle) of body `i` about `p`, or `None` if unbound.
fn elements(b: &Bodies, i: usize, p: usize) -> Option<(f64, f64, f64)> {
    let (dx, dy) = (b.x[i] - b.x[p], b.y[i] - b.y[p]);
    let (dvx, dvy) = (b.vx[i] - b.vx[p], b.vy[i] - b.vy[p]);
    let r = dx.hypot(dy).max(1e-12);
    let v2 = dvx * dvx + dvy * dvy;
    let mu = G * (b.m[p] + b.m[i]);
    let rv = dx * dvx + dy * dvy;
    let ex = ((v2 - mu / r) * dx - rv * dvx) / mu;
    let ey = ((v2 - mu / r) * dy - rv * dvy) / mu;
    let inv = 2.0 / r - v2 / mu;
    (inv > 1e-12).then(|| (1.0 / inv, ex.hypot(ey).min(0.95), ey.atan2(ex)))
}

/// Just the semi-major axis (no eccentricity / periapsis needed for grinding).
/// Returns `None` if unbound, matching `elements`.
fn semi_major_axis(b: &Bodies, i: usize, p: usize) -> Option<f64> {
    let dx = b.x[i] - b.x[p];
    let dy = b.y[i] - b.y[p];
    let dvx = b.vx[i] - b.vx[p];
    let dvy = b.vy[i] - b.vy[p];
    let r = dx.hypot(dy).max(1e-12);
    let v2 = dvx * dvx + dvy * dvy;
    let mu = G * (b.m[p] + b.m[i]);
    let inv = 2.0 / r - v2 / mu;
    (inv > 1e-12).then(|| 1.0 / inv)
}

/// One dust field per system from its belt and rubble bodies. Each source
/// body releases grains on its own orbit, with the phase uniform in mean
/// anomaly and small scatter in a, e and the periapsis angle (a steady ring).
/// `None` means one system covering all bodies.
pub fn init_fields(b: &mut Bodies, ranges: Option<&[(usize, usize)]>) {
    let ranges: Vec<(usize, usize)> = ranges
        .map(|r| r.to_vec())
        .unwrap_or_else(|| vec![(0, b.len())]);
    let mut rng = std::mem::replace(&mut b.rng, XorShift::new(1));
    for (start, end) in ranges {
        let Some(pri) = (start..end)
            .filter(|&i| is_host_kind(b.kind[i]))
            .max_by(|&i, &j| b.m[i].total_cmp(&b.m[j]))
        else {
            continue;
        };
        let mut extent = 100.0f64;
        for i in start..end {
            if i == pri {
                continue;
            }
            let d = (b.x[i] - b.x[pri]).hypot(b.y[i] - b.y[pri]);
            let apo = orbit_info(b, i, pri).apo;
            extent = extent.max(if apo.is_finite() { apo.min(d * 3.0) } else { d });
        }
        let extent = (extent * 1.2).max(200.0);

        let sources: Vec<usize> = (start..end)
            .filter(|&i| i != pri && (b.origin[i].is_belt() || b.origin[i].is_rubble()))
            .collect();
        let per = (DUST_MAX_GRAINS / sources.len().max(1)).clamp(4, DUST_GRAINS_PER_BODY);
        let mut grains = Vec::with_capacity(sources.len() * per);
        for &i in &sources {
            let Some((a0, e0, w0)) = elements(b, i, pri) else {
                continue;
            };
            let total = DUST_TO_BELT_MASS * b.m[i];
            let sizes: Vec<f64> = (0..per).map(|_| draw_size(&mut rng)).collect();
            let wsum: f64 = sizes.iter().map(|&s| size_weight(s)).sum();
            for &size in &sizes {
                let a = a0 * (1.0 + 0.03 * sample_normal(&mut rng, 0.0, 1.0)).max(0.5);
                let e = (e0 + 0.02 * sample_normal(&mut rng, 0.0, 1.0)).clamp(0.0, 0.95);
                let w = w0 + 0.05 * sample_normal(&mut rng, 0.0, 1.0);
                let ma = rng.next_f64() * std::f64::consts::TAU;
                let nu = true_anomaly(solve_kepler(ma, e), e);
                let (x, y, vx, vy) = kepler_state(G * b.m[pri], a, e, nu, w);
                grains.push(Grain {
                    x: b.x[pri] + x,
                    y: b.y[pri] + y,
                    vx: b.vx[pri] + vx,
                    vy: b.vy[pri] + vy,
                    mass: total * size_weight(size) / wsum,
                    size_um: size,
                    ttl: f32::INFINITY,
                });
            }
        }
        b.dust_fields.push(DustField {
            primary: pri,
            extent_gm: extent,
            grains,
            version: DUST_VERSION_COUNTER.fetch_add(1_000_000, AtomicOrdering::Relaxed),
        });
    }
    b.rng = rng;
}

/// Diagnostics for `NBODY_TIMING`: nanoseconds spent in `evolve` since the
/// last reset (the reader swaps it to 0), and the total grain count after the
/// most recent step. Relaxed atomics, so the cost is negligible.
pub static DUST_STAT_NS: AtomicU64 = AtomicU64::new(0);
pub static DUST_STAT_GRAINS: AtomicU64 = AtomicU64::new(0);

/// Advance every field. Runs every `DUST_STEP_INTERVAL` physics steps.
pub fn evolve(b: &mut Bodies, dt: f64) {
    if !b.stats.steps.is_multiple_of(DUST_STEP_INTERVAL) {
        return;
    }
    if b.dust_fields.is_empty() {
        return;
    }
    let t0 = std::time::Instant::now();
    let mut fields = std::mem::take(&mut b.dust_fields);
    let mut rng = std::mem::replace(&mut b.rng, XorShift::new(1));
    let dt_field = dt * DUST_STEP_INTERVAL as f64;
    for f in fields.iter_mut() {
        f.step(b, dt_field);
        grind_belts(b, f, dt_field, &mut rng);
    }
    b.rng = rng;
    b.dust_fields = fields;
    DUST_STAT_NS.fetch_add(t0.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    let total: u64 = b.dust_fields.iter().map(|f| f.grains.len() as u64).sum();
    DUST_STAT_GRAINS.store(total, AtomicOrdering::Relaxed);
}

/// Collisional evolution of the belt bodies about the field's primary.
///
/// Each belt body sheds mass to dust at a rate that follows the belt mass
/// and the orbital period: heavier belts collide more, inner bodies orbit
/// faster. A shedding event removes `BELT_GRIND_CHUNK` of the body's mass,
/// and the mass goes to the field as dust at the body's position and velocity
/// plus an ejection kick, so mass and momentum are conserved. The belt thins
/// and the dust field is fed for as long as the belt lasts.
pub fn grind_belts(b: &mut Bodies, f: &mut DustField, dt: f64, rng: &mut XorShift) {
    let pri = f.primary;
    if pri >= b.len() {
        return;
    }
    let mu = G * b.m[pri];
    let members: Vec<(usize, f64)> = (0..b.len())
        .filter(|&i| i != pri && !b.dead[i] && b.origin[i].is_belt())
        .filter(|&i| (b.x[i] - b.x[pri]).hypot(b.y[i] - b.y[pri]) < f.extent_gm)
        .filter_map(|i| semi_major_axis(b, i, pri).map(|a| (i, a)))
        .collect();
    let belt_mass: f64 = members.iter().map(|&(i, _)| b.m[i]).sum();
    let rate = BELT_GRIND_PER_ORBIT * (belt_mass / BELT_GRIND_M_REF).clamp(0.01, 100.0);
    for (i, a) in members {
        let m = b.m[i];
        if m < BELT_GRIND_MIN_M || f.grains.len() + DUST_EMIT_GRAINS > DUST_MAX_GRAINS {
            continue;
        }
        let period = std::f64::consts::TAU * (a.powi(3) / mu).sqrt();
        let p_event = (rate * dt / period / BELT_GRIND_CHUNK).min(1.0);
        if rng.next_f64() >= p_event {
            continue;
        }
        let dm = BELT_GRIND_CHUNK * m;
        let r = (b.x[i] - b.x[pri]).hypot(b.y[i] - b.y[pri]).max(1e-6);
        let vc = (mu / r).sqrt();
        mark_energy(b);
        b.set_mass(i, m - dm);
        f.emit(
            (b.x[i], b.y[i], b.vx[i], b.vy[i]),
            dm,
            DUST_EJECT_FRAC * vc,
            rng,
        );
    }
}
