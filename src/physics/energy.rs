//! Energy and angular-momentum bookkeeping.

use super::*;

pub fn total_energy(b: &Bodies) -> Option<f64> {
    let n = b.len();
    if n > ENERGY_MAX_N {
        return None;
    }
    let ke: f64 = (0..n)
        .map(|i| 0.5 * b.m[i] * (b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i]))
        .sum();
    let pe: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut s = 0.0;
            for j in (i + 1)..n {
                let dx = b.x[j] - b.x[i];
                let dy = b.y[j] - b.y[i];
                s -= G * b.m[i] * b.m[j] / (dx * dx + dy * dy + SOFT2).sqrt();
            }
            s
        })
        .sum();
    Some(ke + pe)
}

pub fn total_angular_momentum(b: &Bodies) -> f64 {
    let n = b.len();
    if n == 0 {
        return 0.0;
    }
    let (mut mt, mut cx, mut cy, mut vx, mut vy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        mt += b.m[i];
        cx += b.m[i] * b.x[i];
        cy += b.m[i] * b.y[i];
        vx += b.m[i] * b.vx[i];
        vy += b.m[i] * b.vy[i];
    }
    if mt <= 0.0 {
        return 0.0;
    }
    cx /= mt;
    cy /= mt;
    vx /= mt;
    vy /= mt;
    let mut l = 0.0;
    for i in 0..n {
        l += b.m[i] * ((b.x[i] - cx) * (b.vy[i] - vy) - (b.y[i] - cy) * (b.vx[i] - vx)) + b.spin[i];
    }
    l
}

pub(super) fn mark_energy(b: &mut Bodies) {
    if b.e_pre.is_none() {
        b.e_pre = total_energy(b);
    }
}

pub(super) fn settle_energy(b: &mut Bodies) {
    if let Some(e0) = b.e_pre.take() {
        if let Some(e1) = total_energy(b) {
            b.stats.energy_offset += e1 - e0;
        }
    }
}

pub fn rebase_energy(b: &mut Bodies) {
    b.stats.energy0 = total_energy(b).unwrap_or(0.0);
    b.stats.energy_offset = 0.0;
    b.e_pre = None;
}

pub fn energy_drift_pct(b: &Bodies) -> Option<f64> {
    let e = total_energy(b)?;
    let e0 = b.stats.energy0;
    if e0 == 0.0 {
        return None;
    }
    Some(((e - b.stats.energy_offset - e0) / e0.abs()) * 100.0)
}

pub fn energy_dissipated_pct(b: &Bodies) -> Option<f64> {
    let e0 = b.stats.energy0;
    if e0 == 0.0 || b.len() > ENERGY_MAX_N {
        return None;
    }
    Some(b.stats.energy_offset / e0.abs() * 100.0)
}
