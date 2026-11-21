//! Tidal dissipation via the equilibrium tide model.
//!
//! A close-in planet raises a tidal bulge on its host. Because the bulge
//! lags the planet's position (for a host rotating slower than the orbit),
//! the pair loses orbital energy to heat: the orbit shrinks, circularizes,
//! and the host's spin absorbs the shed orbital angular momentum. A host
//! rotating faster than the orbit reverses the sign and the planet spirals
//! outward instead.
//!
//! The friction rate for a star–planet pair is (Hut 1981)
//!
//!     1/t_f = (9/2) · (k₂/Q) · q(1+q) · (R/a)⁵ · n
//!
//! with q = m_planet/m_host and n the mean motion. On a circular orbit the
//! semimajor axis evolves as
//!
//!     da/dt = −a · (1/t_f) · (1 − Ω_host/n)
//!
//! which is inward decay when Ω_host ≪ n and outward migration when
//! Ω_host > n. Total angular momentum is conserved: whatever the orbit
//! loses is added to the host's spin.
//!
//! Tides act in both directions on each bound pair: the bulge a body raises
//! on its parent (host or planet) and the bulge its parent raises on it.
//! The second one despins planets and moons toward synchronous rotation
//! (tidal locking), circularizes orbits, and — for star–star pairs —
//! circularizes close binaries during the run. Planet and moon spin is the
//! `spin` angular momentum, initialized lazily to a ~10 h (giants) or 24 h
//! (others) day the first time the body is seen. Spin transfer is capped at
//! the synchronous value so a light planet cannot overshoot corotation.
//! `tidal_heating_w` gives the Peale (1979) dissipation of an eccentric,
//! synchronously rotating body, for display.

use super::*;

/// Tidal Love number of a main-sequence star.
pub const TIDAL_K2_STAR: f64 = 0.03;

/// Physical tidal quality factor of a main-sequence star. The effective
/// rate uses `TIDAL_Q_STAR / TIDAL_BOOST`.
pub const TIDAL_Q_STAR: f64 = 1e6;

/// Multiplier on the tidal dissipation rate, analogous to `PR_DRAG_BOOST`.
///
/// At the default value the effective quality factor is 1 — i.e. tides
/// run 10⁶× faster than the physical value. That puts a hot Jupiter at
/// 0.02 AU on a decay timescale of a few ×10⁴ time units, which is
/// visible within a simulation session instead of the physical few ×10⁹
/// tu (~Gyr). Lower for physical rates, raise for a faster demo.
pub const TIDAL_BOOST: f64 = 1e6;

/// Maximum semimajor axis, in units of the host radius, at which tides
/// are applied. The rate scales as a⁻⁵, so beyond ~50 R∗ it is negligible;
/// this cap keeps the loop focused on the hot inner orbits.
pub const TIDAL_MAX_RADII: f64 = 50.0;

/// Ceiling on the fractional velocity change per step, so a single step
/// on a very tight orbit cannot destabilize the integrator.
const TIDAL_MAX_FRAC: f64 = 0.05;

/// Moment-of-inertia coefficient `k` in `I = k·M·R²`. A uniform sphere is
/// 0.4; real main-sequence stars are closer to 0.1 (radiative core plus
/// convective envelope). The remaining variants are rough but plausible.
#[inline]
fn inertia_factor(kind: BodyKind) -> f64 {
    match kind {
        BodyKind::Star => 0.1,
        BodyKind::BrownDwarf => 0.15,
        BodyKind::WhiteDwarf => 0.2,
        BodyKind::NeutronStar => 0.3,
        BodyKind::BlackHole => 0.5, // unphysical, but avoids div-by-zero
        BodyKind::Planet | BodyKind::DwarfPlanet => 0.3,
    }
}

/// Love number over quality factor for rocky / icy planets and moons.
pub const TIDAL_K2_Q_ROCKY: f64 = 0.3 / 100.0;
/// Love number over quality factor for gas giants.
pub const TIDAL_K2_Q_GIANT: f64 = 0.5 / 1e5;
/// Extra boost, relative to `tidal_boost()`, for tides raised in planets
/// (1e-2 of the stellar boost: planets are far more dissipative than stars).
const PLANET_BOOST_REL: f64 = 1e-2;
/// Planets of at least this mass (M⊕) get the gas-giant k₂/Q.
const GIANT_M_MIN: f64 = 50.0;
/// Smallest body that is given a spin and tidally evolved.
const SPIN_M_MIN: f64 = 0.01;

/// Effective (boosted) `k₂/Q` of body `k`.
fn k2_over_q(b: &Bodies, k: usize) -> f64 {
    match b.kind[k] {
        BodyKind::Planet | BodyKind::DwarfPlanet => {
            let base = if b.m[k] >= GIANT_M_MIN {
                TIDAL_K2_Q_GIANT
            } else {
                TIDAL_K2_Q_ROCKY
            };
            base * if physical_mode() {
                1.0
            } else {
                TIDAL_BOOST * PLANET_BOOST_REL
            }
        }
        _ => TIDAL_K2_STAR / TIDAL_Q_STAR * tidal_boost(),
    }
}

/// Spin angular momentum of a planet with a day of `hours`, in sim units.
fn initial_planet_spin(m: f64, r: f64, hours: f64) -> f64 {
    let omega = std::f64::consts::TAU / (hours * 3600.0 / time_unit_s());
    inertia_factor(BodyKind::Planet) * m * r * r * omega
}

/// Tidal dissipation in watts of body `j` raised by its parent `host`
/// (Peale 1979, synchronous rotation): Ė = (21/2)(k₂/Q) G M_h² R⁵ n e² / a⁶.
/// Uses the *physical* `k₂/Q` (no boost) and SI units, independent of the
/// simulation's G. Returns 0 for unbound or circular orbits.
pub fn tidal_heating_w(b: &Bodies, j: usize, host: usize) -> f64 {
    if !matches!(b.kind[j], BodyKind::Planet | BodyKind::DwarfPlanet) {
        return 0.0;
    }
    let o = orbit_info(b, j, host);
    if !o.bound || !o.semi_major.is_finite() || o.semi_major <= 0.0 {
        return 0.0;
    }
    let k2q = if b.m[j] >= GIANT_M_MIN {
        TIDAL_K2_Q_GIANT
    } else {
        TIDAL_K2_Q_ROCKY
    };
    let a = o.semi_major * LEN_KM * 1e3;
    let r = b.r[j] * LEN_KM * 1e3;
    let mh = b.m[host] * MASS_KG;
    let n = (G_SI * (mh + b.m[j] * MASS_KG) / (a * a * a)).sqrt();
    let e = o.ecc.clamp(0.0, 0.99);
    10.5 * k2q * G_SI * mh * mh * r.powi(5) * n * e * e / a.powi(6)
}

/// Apply equilibrium tides to every bound pair (child `j`, parent `host`):
/// planets and moons about stars or planets, and companion stars or brown
/// dwarfs about their primaries.
pub(super) fn tidal_dissipation(b: &mut Bodies, dt: f64) {
    let n = b.len();
    if n < 2 {
        return;
    }
    // Give every planet a day the first time we see it.
    for j in 0..n {
        if b.spin[j] == 0.0
            && !b.dead[j]
            && matches!(b.kind[j], BodyKind::Planet | BodyKind::DwarfPlanet)
            && b.m[j] >= SPIN_M_MIN
            && b.r[j] > 0.0
        {
            let hours = if b.m[j] >= GIANT_M_MIN { 10.0 } else { 24.0 };
            b.spin[j] = initial_planet_spin(b.m[j], b.r[j], hours);
        }
    }

    // A bound orbit satisfies r ≤ 2a, hence a ≥ r/2. So if r exceeds
    // 2·TIDAL_MAX_RADII·R∗, the later `a > TIDAL_MAX_RADII·R∗` test is
    // guaranteed to fire. We compare squared radii to skip the sqrt and
    // vis-viva entirely for the (common) far-out planets.
    const R_EARLY_OUT: f64 = 2.0 * TIDAL_MAX_RADII;
    const R_EARLY_OUT2: f64 = R_EARLY_OUT * R_EARLY_OUT;

    for j in 0..n {
        if b.dead[j]
            || !matches!(
                b.kind[j],
                BodyKind::Planet | BodyKind::Star | BodyKind::BrownDwarf
            )
        {
            continue;
        }
        let Some(host) = b.parent_cache.get(j).copied().flatten() else {
            continue;
        };
        if host >= n || b.dead[host] || host == j {
            continue;
        }
        let host_ok = is_host_kind(b.kind[host])
            || (b.kind[host] == BodyKind::Planet && b.kind[j] == BodyKind::Planet);
        if !host_ok {
            continue;
        }
        let r_host = b.r[host];
        if !(r_host > 0.0) || !r_host.is_finite() {
            continue;
        }
        let dx = b.x[j] - b.x[host];
        let dy = b.y[j] - b.y[host];
        let dz = b.z[j] - b.z[host];
        if dx * dx + dy * dy + dz * dz > R_EARLY_OUT2 * r_host * r_host {
            continue;
        }
        // Tide raised on the parent by the child (the original model)...
        tide_pair(b, dt, j, host, host);
        // ...and the tide raised on the child by the parent: despins and
        // locks planets, circularizes binaries. Not for degenerate parents'
        // children that have no meaningful radius.
        if b.r[j] > 0.0 && b.m[j] >= SPIN_M_MIN {
            tide_pair(b, dt, j, host, j);
        }
    }
}

/// One tidal interaction for the pair (`j` child, `host` parent) where the
/// bulge is raised on body `raised` ∈ {j, host}.
fn tide_pair(b: &mut Bodies, dt: f64, j: usize, host: usize, raised: usize) {
    let r_host = b.r[host];
    let dx = b.x[j] - b.x[host];
    let dy = b.y[j] - b.y[host];
    let dz = b.z[j] - b.z[host];
    let r2 = dx * dx + dy * dy + dz * dz;
    let r = r2.sqrt();
    if r <= r_host {
        return; // already inside the host; collision logic owns this
    }
    let dvx = b.vx[j] - b.vx[host];
    let dvy = b.vy[j] - b.vy[host];
    let dvz = b.vz[j] - b.vz[host];
    let m_host = b.m[host];
    let mu = G * (m_host + b.m[j]);
    let v2 = dvx * dvx + dvy * dvy + dvz * dvz;

    // Tides on a fly-by are meaningless, so skip hyperbolic passes.
    let energy = 0.5 * v2 - mu / r;
    if energy >= 0.0 {
        return;
    }
    let a_sma = -mu / (2.0 * energy);
    if !a_sma.is_finite() || a_sma > TIDAL_MAX_RADII * r_host || a_sma < r_host {
        return;
    }
    let n_mean = (mu / (a_sma * a_sma * a_sma)).sqrt();
    if !n_mean.is_finite() || !(n_mean > 0.0) {
        return;
    }

    // Body carrying the bulge, and the perturbing body's mass ratio.
    let (r_raised, q) = if raised == host {
        (r_host, b.m[j] / m_host)
    } else {
        (b.r[j], m_host / b.m[j])
    };
    let k2q_45 = 4.5 * k2_over_q(b, raised);
    // 1/t_f = (9/2)·(k₂/Q)·q(1+q)·(R/a)⁵·n
    let ratio = r_raised / a_sma;
    let ratio5 = ratio * ratio * ratio * ratio * ratio;
    let rate = k2q_45 * q * (1.0 + q) * ratio5 * n_mean;
    if !(rate > 0.0) || !rate.is_finite() {
        return;
    }

    // Spin angular velocity of the raised body: Ω = L_spin / I.
    let inertia = inertia_factor(b.kind[raised]) * b.m[raised] * r_raised * r_raised;
    let omega = if inertia > 0.0 { b.spin[raised] / inertia } else { 0.0 };

    // `b.spin` is signed (positive = counter-clockwise), so the mean motion
    // must be signed too: the sign of the orbital angular momentum gives the
    // sense of the orbit.
    // Spin is measured about +z, so a tilted orbit's sense is that of its
    // angular momentum's z component.
    let n_signed = if dx * dvy - dy * dvx >= 0.0 { n_mean } else { -n_mean };
    // Decay factor 1 − Ω/n: slow spin → inward, fast spin → outward.
    let mut frac = (rate * (1.0 - omega / n_signed) * dt).clamp(-TIDAL_MAX_FRAC, TIDAL_MAX_FRAC);
    if frac == 0.0 {
        return;
    }

    let inv_r = 1.0 / r;
    let ux = dx * inv_r;
    let uy = dy * inv_r;
    let uz = dz * inv_r;
    let vr = dvx * ux + dvy * uy + dvz * uz;
    // Planar pairs keep the original 2D expressions so z = 0 runs stay
    // bit-identical; otherwise the tangential part is the remainder of the
    // relative velocity, signed by the orbit's sense about +z.
    let planar = dz == 0.0 && dvz == 0.0;
    let (tx, ty, tz) = (dvx - vr * ux, dvy - vr * uy, dvz - vr * uz);
    let vt = if planar {
        dvy * ux - dvx * uy
    } else {
        let tm = (tx * tx + ty * ty + tz * tz).sqrt();
        if dx * dvy - dy * dvx >= 0.0 { tm } else { -tm }
    };
    let mu_red = m_host * b.m[j] / (m_host + b.m[j]);

    // Spin takes the orbit's lost angular momentum, ΔL = ½·frac·μ_red·r·v_t.
    // Cap it so the spin stops at corotation (I·n) instead of overshooting;
    // a light planet has far less spin L than the orbit exchanges per step.
    let mut dl = 0.5 * frac * mu_red * r * vt;
    if inertia > 0.0 && vt.abs() > 1e-30 {
        let l_sync = inertia * n_signed;
        let gap = l_sync - b.spin[raised];
        if dl.abs() > gap.abs() || dl * gap < 0.0 {
            // Would pass (or move away from) corotation: stop at it.
            let allowed = if dl * gap > 0.0 { gap } else { 0.0 };
            dl = allowed;
            frac = dl / (0.5 * mu_red * r * vt);
        }
    }
    if frac == 0.0 {
        return;
    }

    // Split the relative velocity into radial and tangential pieces:
    // damping v_r circularizes, damping v_t changes the semimajor axis.
    let new_vr = vr * (1.0 - frac);
    let new_vt = vt * (1.0 - 0.5 * frac);
    let (new_dvx, new_dvy, new_dvz) = if planar {
        (new_vr * ux - new_vt * uy, new_vr * uy + new_vt * ux, 0.0)
    } else {
        let s = 1.0 - 0.5 * frac;
        (new_vr * ux + s * tx, new_vr * uy + s * ty, new_vr * uz + s * tz)
    };

    // The tide changes the RELATIVE velocity by (ddvx, ddvy). Share it by
    // mass so the pair's linear momentum does not change.
    let ddvx = new_dvx - dvx;
    let ddvy = new_dvy - dvy;
    let ddvz = new_dvz - dvz;
    let m_pair = m_host + b.m[j];
    let share_planet = m_host / m_pair;
    let share_host = b.m[j] / m_pair;
    let ke0 = 0.5 * b.m[j] * (b.vx[j] * b.vx[j] + b.vy[j] * b.vy[j] + b.vz[j] * b.vz[j])
        + 0.5 * m_host * (b.vx[host] * b.vx[host] + b.vy[host] * b.vy[host] + b.vz[host] * b.vz[host]);
    b.vx[j] += share_planet * ddvx;
    b.vy[j] += share_planet * ddvy;
    b.vz[j] += share_planet * ddvz;
    b.vx[host] -= share_host * ddvx;
    b.vy[host] -= share_host * ddvy;
    b.vz[host] -= share_host * ddvz;
    // Positions and masses do not change, so the energy change is the
    // kinetic-energy change of both bodies. If a snapshot is pending,
    // `settle_energy` already counts it. Do not count it twice.
    if b.e_pre.is_none() {
        let ke1 = 0.5 * b.m[j] * (b.vx[j] * b.vx[j] + b.vy[j] * b.vy[j] + b.vz[j] * b.vz[j])
            + 0.5 * m_host * (b.vx[host] * b.vx[host] + b.vy[host] * b.vy[host] + b.vz[host] * b.vz[host]);
        b.stats.energy_offset += ke1 - ke0;
    }
    b.spin[raised] += dl;
}
