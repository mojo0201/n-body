//! Planet migration in a gas disk, and the resonance capture it produces.
//!
//! A gas disk exists around young stars (see `disk_level`): it is full at birth and
//! disperses with a 2 Myr e-folding time, so systems younger than a few Myr migrate
//! by default and mature ones do not. The "gas disk" switch forces one on around
//! every host regardless of age. While a disk is present, every planet orbiting a host feels the
//! disk's torque as a drag on its orbit (Papaloizou & Larwood 2000; Cresswell
//! & Nelson 2008):
//!
//!     a_mig = −v / (2 τ_a)          (semimajor axis shrinks as ȧ/a = −1/τ_a)
//!     a_ecc = −2 (v·r̂) r̂ / τ_e      (eccentricity damps, τ_e = τ_a / 50)
//!
//! τ_a falls with planet mass (Type I: ∝ 1/m up to 100 M⊕), and giants above
//! that open a gap and migrate more slowly (Type II). The disk has an inner
//! edge at max(3 R_host, 0.05 AU) where the torque switches off, so planets
//! stack up there. Two planets whose periods converge are captured into mean
//! motion resonance when the eccentricity damping lets them keep a commensurate
//! ratio, so migrating systems end up in resonant chains.
//!
//! The real Type I time for 1 M⊕ at 1 AU is ~10⁵ yr, far too slow to watch,
//! so `MIGRATION_SPEEDUP` shortens every timescale by a fixed factor.

use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

static GAS_DISK: AtomicBool = AtomicBool::new(false);

/// Force a gas disk around every host regardless of age. Off by default; young
/// stars get their disk from their age alone.
pub fn set_gas_disk(on: bool) {
    GAS_DISK.store(on, Ordering::Relaxed);
}

pub fn gas_disk() -> bool {
    GAS_DISK.load(Ordering::Relaxed)
}

/// Disk e-folding time in Gyr (2 Myr) and the age beyond which none is left.
const DISK_TAU_GYR: f64 = 0.002;
const DISK_MAX_AGE_GYR: f64 = 0.02;

/// Gas-disk strength 0..1 around host `h`: 1 if forced, else exp(−age/2 Myr)
/// for a star younger than 20 Myr, else 0.
pub fn disk_level(b: &Bodies, h: usize) -> f64 {
    if gas_disk() {
        return 1.0;
    }
    if b.kind[h] == BodyKind::Star && b.age_gyr[h] < DISK_MAX_AGE_GYR {
        (-b.age_gyr[h] / DISK_TAU_GYR).exp()
    } else {
        0.0
    }
}

/// Type I migration time of a 1 M⊕ planet at 1 AU, in simulation time units
/// (~10⁵ yr).
const TAU_I_REF: f64 = 7e5;
/// Factor by which migration is sped up so it is visible.
pub const MIGRATION_SPEEDUP: f64 = 300.0;
/// τ_a / τ_e.
const ECC_DAMP_RATIO: f64 = 50.0;
/// Disk outer edge in AU; nothing migrates beyond it.
const DISK_OUTER_AU: f64 = 100.0;
/// Largest fractional velocity change per step.
const MAX_FRAC: f64 = 0.02;

#[inline]
fn smoothstep01(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Migration time (time units) for a planet of mass `m` (M⊕) at `a_au`.
fn tau_a(m: f64, a_au: f64) -> f64 {
    // Type I up to 100 M⊕; heavier planets open a gap and slow down.
    let m_eff = if m <= 100.0 { m } else { 100.0 * (100.0 / m) };
    TAU_I_REF / (MIGRATION_SPEEDUP * m_eff.max(1e-3)) * a_au
}

/// Apply disk torques to every planet orbiting a host. Non-conservative: the
/// kinetic-energy change goes into `energy_offset`, and the host takes the
/// opposite impulse so linear momentum is kept.
pub(super) fn gas_disk_migration(b: &mut Bodies, dt: f64) {
    let n = b.len();
    if n < 2 || !b.parent_cache_valid {
        return;
    }
    let au = au_in_gm();
    for j in 0..n {
        if b.dead[j] || b.kind[j] != BodyKind::Planet || !(b.m[j] >= 0.1) {
            continue;
        }
        let Some(h) = b.parent_cache.get(j).copied().flatten() else {
            continue;
        };
        if h >= n || b.dead[h] || !is_host_kind(b.kind[h]) {
            continue;
        }
        let g = disk_level(b, h);
        if g < 1e-3 {
            continue;
        }
        let dx = b.x[j] - b.x[h];
        let dy = b.y[j] - b.y[h];
        let dz = b.z[j] - b.z[h];
        let dvx = b.vx[j] - b.vx[h];
        let dvy = b.vy[j] - b.vy[h];
        let dvz = b.vz[j] - b.vz[h];
        let r2 = dx * dx + dy * dy + dz * dz;
        let r = r2.sqrt();
        let v2 = dvx * dvx + dvy * dvy + dvz * dvz;
        let mu = G * (b.m[h] + b.m[j]);
        if !(r > 0.0) || !(v2 > 0.0) {
            continue;
        }
        let energy = 0.5 * v2 - mu / r;
        if energy >= 0.0 {
            continue;
        }
        let a_sma = -mu / (2.0 * energy);
        let a_au = a_sma / au;
        if !(a_au.is_finite()) || a_au > DISK_OUTER_AU {
            continue;
        }
        // Torque switch-off at the inner edge (cavity).
        let a_in = (3.0 * b.r[h]).max(0.05 * au);
        let f_mig = smoothstep01((a_sma - a_in) / (0.3 * a_in));
        let tau = tau_a(b.m[j], a_au.max(1e-3)) / g;
        let tau_e = tau / ECC_DAMP_RATIO;
        let v = v2.sqrt();
        let vr_dot = dvx * dx + dvy * dy + dvz * dz; // v·r
        // Accelerations.
        let ax = -0.5 * f_mig * dvx / tau - 2.0 * vr_dot * dx / (r2 * tau_e);
        let ay = -0.5 * f_mig * dvy / tau - 2.0 * vr_dot * dy / (r2 * tau_e);
        let az = -0.5 * f_mig * dvz / tau - 2.0 * vr_dot * dz / (r2 * tau_e);
        let mut dvx_k = ax * dt;
        let mut dvy_k = ay * dt;
        let mut dvz_k = az * dt;
        let dv = (dvx_k * dvx_k + dvy_k * dvy_k + dvz_k * dvz_k).sqrt();
        if dv > MAX_FRAC * v {
            let s = MAX_FRAC * v / dv;
            dvx_k *= s;
            dvy_k *= s;
            dvz_k *= s;
        }
        let ke0 = 0.5 * b.m[j] * (b.vx[j] * b.vx[j] + b.vy[j] * b.vy[j] + b.vz[j] * b.vz[j])
            + 0.5 * b.m[h] * (b.vx[h] * b.vx[h] + b.vy[h] * b.vy[h] + b.vz[h] * b.vz[h]);
        let ratio = b.m[j] / b.m[h];
        b.vx[j] += dvx_k;
        b.vy[j] += dvy_k;
        b.vz[j] += dvz_k;
        b.vx[h] -= dvx_k * ratio;
        b.vy[h] -= dvy_k * ratio;
        b.vz[h] -= dvz_k * ratio;
        if b.e_pre.is_none() {
            let ke1 = 0.5 * b.m[j] * (b.vx[j] * b.vx[j] + b.vy[j] * b.vy[j] + b.vz[j] * b.vz[j])
                + 0.5 * b.m[h] * (b.vx[h] * b.vx[h] + b.vy[h] * b.vy[h] + b.vz[h] * b.vz[h]);
            b.stats.energy_offset += ke1 - ke0;
        }
    }
}
