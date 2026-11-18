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
//! Only host–planet pairs are treated here. Planet–satellite tides are
//! intentionally not modelled; moons are spawned on circular orbits.

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

/// Apply equilibrium-tide orbital decay and circularization to every
/// planet that orbits a host body.
pub(super) fn tidal_dissipation(b: &mut Bodies, dt: f64) {
    let n = b.len();
    if n < 2 {
        return;
    }

    // (9/2)·(k₂/Q_eff) with Q_eff = Q*/boost. The boost is 1 in physical mode.
    let k2q_45 = 4.5 * TIDAL_K2_STAR * tidal_boost() / TIDAL_Q_STAR;

    // A bound orbit satisfies r ≤ 2a, hence a ≥ r/2. So if r exceeds
    // 2·TIDAL_MAX_RADII·R∗, the later `a > TIDAL_MAX_RADII·R∗` test is
    // guaranteed to fire. We compare squared radii to skip the sqrt and
    // vis-viva entirely for the (common) far-out planets.
    const R_EARLY_OUT: f64 = 2.0 * TIDAL_MAX_RADII;
    const R_EARLY_OUT2: f64 = R_EARLY_OUT * R_EARLY_OUT;

    for j in 0..n {
        if b.dead[j] || b.kind[j] != BodyKind::Planet {
            continue;
        }
        let Some(host) = b.parent_cache.get(j).copied().flatten() else {
            continue;
        };
        if host >= n || b.dead[host] || !is_host_kind(b.kind[host]) {
            continue;
        }

        let r_host = b.r[host];
        if !(r_host > 0.0) || !r_host.is_finite() {
            continue;
        }

        let dx = b.x[j] - b.x[host];
        let dy = b.y[j] - b.y[host];
        let r2 = dx * dx + dy * dy;

        // Cheap squared-distance early-out: if r > 2·MAX·R∗, a > MAX·R∗ too.
        if r2 > R_EARLY_OUT2 * r_host * r_host {
            continue;
        }

        let r = r2.sqrt();
        if r <= r_host {
            continue; // already inside the host; collision logic owns this
        }

        let dvx = b.vx[j] - b.vx[host];
        let dvy = b.vy[j] - b.vy[host];
        let m_host = b.m[host];
        let mu = G * (m_host + b.m[j]);
        let v2 = dvx * dvx + dvy * dvy;

        // Vis-viva: negative specific energy means the pair is bound.
        // Tides on a fly-by are meaningless, so skip hyperbolic passes.
        let energy = 0.5 * v2 - mu / r;
        if energy >= 0.0 {
            continue;
        }
        let a_sma = -mu / (2.0 * energy);
        if !a_sma.is_finite() || a_sma > TIDAL_MAX_RADII * r_host || a_sma < r_host {
            continue;
        }

        let n_mean = (mu / (a_sma * a_sma * a_sma)).sqrt();
        if !n_mean.is_finite() || !(n_mean > 0.0) {
            continue;
        }

        // 1/t_f = (9/2)·(k₂/Q)·q(1+q)·(R/a)⁵·n
        let q = b.m[j] / m_host;
        let ratio = r_host / a_sma;
        let ratio5 = ratio * ratio * ratio * ratio * ratio;
        let rate = k2q_45 * q * (1.0 + q) * ratio5 * n_mean;
        if !(rate > 0.0) || !rate.is_finite() {
            continue;
        }

        // Host spin angular velocity: Ω = L_spin / I.
        let inertia_host = inertia_factor(b.kind[host]) * m_host * r_host * r_host;
        let omega_host = if inertia_host > 0.0 {
            b.spin[host] / inertia_host
        } else {
            0.0
        };

        // Decay factor 1 − Ω/n: slow host → inward, fast host → outward.
        // `b.spin` is signed (positive = counter-clockwise), so the mean
        // motion must be signed too. The sign of the orbital angular
        // momentum `dx·dvy − dy·dvx` gives the sense of the orbit. A
        // retrograde orbit has n < 0, so a host that spins prograde
        // pushes the planet inward faster.
        let n_signed = if dx * dvy - dy * dvx >= 0.0 {
            n_mean
        } else {
            -n_mean
        };
        // Clamped so a single step cannot destabilize the integrator.
        let frac =
            (rate * (1.0 - omega_host / n_signed) * dt).clamp(-TIDAL_MAX_FRAC, TIDAL_MAX_FRAC);
        if frac == 0.0 {
            continue;
        }

        // Split the relative velocity into radial and tangential pieces:
        // damping v_r circularizes, damping v_t changes the semimajor axis.
        let inv_r = 1.0 / r;
        let ux = dx * inv_r;
        let uy = dy * inv_r;
        let vr = dvx * ux + dvy * uy;
        let vt = dvy * ux - dvx * uy;

        let new_vr = vr * (1.0 - frac);
        let new_vt = vt * (1.0 - 0.5 * frac);

        let new_dvx = new_vr * ux - new_vt * uy;
        let new_dvy = new_vr * uy + new_vt * ux;

        // The tide changes the RELATIVE velocity by (ddvx, ddvy). Share it by
        // mass so the pair's linear momentum does not change: the planet
        // gets m_host/M of it and the host gets -m_planet/M of it.
        let ddvx = new_dvx - dvx;
        let ddvy = new_dvy - dvy;
        let m_pair = m_host + b.m[j];
        let share_planet = m_host / m_pair;
        let share_host = b.m[j] / m_pair;
        let ke0 = 0.5 * b.m[j] * (b.vx[j] * b.vx[j] + b.vy[j] * b.vy[j])
            + 0.5 * m_host * (b.vx[host] * b.vx[host] + b.vy[host] * b.vy[host]);
        b.vx[j] += share_planet * ddvx;
        b.vy[j] += share_planet * ddvy;
        b.vx[host] -= share_host * ddvx;
        b.vy[host] -= share_host * ddvy;
        // Positions and masses do not change, so the energy change is the
        // kinetic-energy change of both bodies. If a snapshot is pending,
        // `settle_energy` already counts it. Do not count it twice.
        if b.e_pre.is_none() {
            let ke1 = 0.5 * b.m[j] * (b.vx[j] * b.vx[j] + b.vy[j] * b.vy[j])
                + 0.5 * m_host * (b.vx[host] * b.vx[host] + b.vy[host] * b.vy[host]);
            b.stats.energy_offset += ke1 - ke0;
        }

        // Orbital angular momentum about the pair's COM is μ_red·r·v_t.
        // Its change is μ_red·r·(new_vt − vt) = −½·frac·μ_red·r·v_t,
        // computed directly here to avoid cancellation when frac is small.
        // Conservation of total L puts the opposite change into the host's
        // spin, so b.spin rises as the orbit decays.
        let mu_red = m_host * b.m[j] / (m_host + b.m[j]);
        b.spin[host] += 0.5 * frac * mu_red * r * vt;
    }
}
