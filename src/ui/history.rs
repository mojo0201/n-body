//! Split out of the former single-file `ui.rs`; see `mod.rs`.

use std::time::{Duration, Instant};


use crate::physics::{self as ph, Bodies, BodyKind, BodyOrigin};
use super::*;

/// One point in the diagnostics history. Sampled once per wall-clock second.
/// Percent values are `None` when they have no meaning (for example the
/// energy drift above `ENERGY_MAX_N` bodies).
/// A sample with a simulation time, for the time-series plots.
pub trait Timed {
    fn time(&self) -> f64;
}

impl Timed for DiagSample {
    fn time(&self) -> f64 {
        self.t
    }
}

pub struct DiagSample {
    pub t: f64,
    pub n: usize,
    /// Integrator energy drift, percent of the baseline energy.
    pub drift: Option<f64>,
    /// Energy removed or added by merges, break-ups and escapes, percent.
    pub dissipated: Option<f64>,
    /// Change of total angular momentum since the baseline, percent.
    pub l_drift: Option<f64>,
    /// Fine time steps used in the last step (close encounters raise this).
    pub nsub: u32,
}

/// One point of a star's light curve.
pub struct FluxSample {
    pub t: f64,
    pub flux: f64,
}

impl Timed for FluxSample {
    fn time(&self) -> f64 {
        self.t
    }
}

pub(super) const LIGHT_MAX_SAMPLES: usize = 3000;

/// Light curve of the selected star as seen from a chosen direction: one sample
/// per frame in which simulation time has moved. At high speed the samples are
/// far apart in time and short transits fall between them.
pub struct LightCurve {
    pub visible: bool,
    pub samples: std::collections::VecDeque<FluxSample>,
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    star: Option<u64>,
}

impl Default for LightCurve {
    fn default() -> Self {
        Self::new()
    }
}

impl LightCurve {
    pub fn new() -> Self {
        Self { visible: false, samples: std::collections::VecDeque::new(), azimuth_deg: 0.0, elevation_deg: 0.0, star: None }
    }

    pub fn clear(&mut self) {
        self.samples.clear();
        self.star = None;
    }

    pub fn track(&mut self, b: &Bodies, selected: Option<u64>, replaying: bool) {
        if replaying || !self.visible {
            return;
        }
        let star = selected.and_then(|id| b.index_of_id(id)).filter(|&i| b.kind[i] == BodyKind::Star);
        let star_id = star.map(|i| b.id[i]);
        if star_id != self.star {
            self.samples.clear();
            self.star = star_id;
        }
        let Some(s) = star else { return };
        if self.samples.back().is_some_and(|p| b.time < p.t) {
            self.samples.clear();
        }
        if self.samples.back().is_some_and(|p| p.t == b.time) {
            return; // paused
        }
        let obs = ph::observer_dir(self.azimuth_deg, self.elevation_deg);
        self.samples.push_back(FluxSample { t: b.time, flux: ph::visible_fraction(b, s, obs) });
        while self.samples.len() > LIGHT_MAX_SAMPLES {
            self.samples.pop_front();
        }
    }
}

/// One point in the orbit-elements history of the selected body.
pub struct ElemSample {
    pub t: f64,
    /// Semi-major axis in AU; `None` while the orbit is unbound.
    pub a_au: Option<f64>,
    pub ecc: f64,
    /// Inclination to the reference plane, degrees.
    pub inc_deg: f64,
    /// Periapsis distance in AU; `None` while unbound.
    pub peri_au: Option<f64>,
}

impl Timed for ElemSample {
    fn time(&self) -> f64 {
        self.t
    }
}

/// How the selected body's osculating orbit about its parent changes with
/// time. Sampled in the UI from the live bodies, so it costs nothing in the
/// physics step. Starts over when the selection or the parent changes.
pub struct ElemHistory {
    pub visible: bool,
    pub samples: std::collections::VecDeque<ElemSample>,
    body: Option<u64>,
    parent: Option<u64>,
    last_sample: Instant,
}

pub(super) const ELEM_MAX_SAMPLES: usize = 600;
pub(super) const ELEM_SAMPLE_EVERY: Duration = Duration::from_millis(250);

impl Default for ElemHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl ElemHistory {
    pub fn new() -> Self {
        Self {
            visible: false,
            samples: std::collections::VecDeque::new(),
            body: None,
            parent: None,
            last_sample: Instant::now(),
        }
    }

    pub fn clear(&mut self) {
        self.samples.clear();
        self.body = None;
        self.parent = None;
    }

    /// Take a sample of `selected` if the window is open and time has moved on.
    /// Does nothing while a timeline replay is shown, since `b` is then a
    /// recorded snapshot.
    pub fn track(&mut self, b: &Bodies, selected: Option<u64>, replaying: bool) {
        if replaying {
            return;
        }
        if selected != self.body {
            self.clear();
            self.body = selected;
        }
        let Some(i) = selected.and_then(|id| b.index_of_id(id)) else {
            return;
        };
        if !self.visible || self.last_sample.elapsed() < ELEM_SAMPLE_EVERY {
            return;
        }
        self.last_sample = Instant::now();
        let Some(par) = parent_of(b, i) else {
            self.samples.clear();
            self.parent = None;
            return;
        };
        let parent_id = Some(b.id[par]);
        if self.parent != parent_id || self.samples.back().is_some_and(|s| b.time < s.t) {
            // Captured by another body, or time went backwards (a new system).
            self.samples.clear();
            self.parent = parent_id;
        }
        if self.samples.back().is_some_and(|s| s.t == b.time) {
            return; // paused
        }
        let au = ph::au_in_gm();
        let o = ph::orbit_info(b, i, par);
        self.samples.push_back(ElemSample {
            t: b.time,
            a_au: o.bound.then(|| o.semi_major / au),
            ecc: o.ecc,
            inc_deg: o.inc.to_degrees(),
            peri_au: o.bound.then(|| o.peri / au),
        });
        while self.samples.len() > ELEM_MAX_SAMPLES {
            self.samples.pop_front();
        }
    }

    pub(super) fn to_csv(&self) -> String {
        let mut out = String::from("time,a_au,ecc,inc_deg,peri_au\n");
        for s in &self.samples {
            out += &format!(
                "{},{},{},{},{}\n",
                s.t,
                s.a_au.map_or(String::new(), |v| v.to_string()),
                s.ecc,
                s.inc_deg,
                s.peri_au.map_or(String::new(), |v| v.to_string())
            );
        }
        out
    }
}

pub struct DiagHistory {
    pub samples: std::collections::VecDeque<DiagSample>,
    pub max_samples: usize,
    pub last_sample: Instant,
    pub eccs: Vec<f64>,
    pub ecc_bins: Vec<usize>,
    pub ecc_bin_width: f64,
    pub ecc_bin_count: usize,
    pub visible: bool,
    /// Angular momentum baseline. Set at the first sample and again when the
    /// energy baseline changes (a user edit re-anchors both).
    l0: Option<f64>,
    e0_seen: f64,
}

impl DiagHistory {
    pub fn new() -> Self {
        Self {
            samples: std::collections::VecDeque::new(),
            max_samples: 300,
            // `now - 10s` forces the first sample on the first frame.
            last_sample: Instant::now() - Duration::from_secs(10),
            eccs: Vec::new(),
            ecc_bins: vec![0; 20],
            ecc_bin_width: 0.05,
            ecc_bin_count: 20,
            // Closed at launch. Open it with G or the View section.
            visible: false,
            l0: None,
            e0_seen: 0.0,
        }
    }

    /// Forget all samples and the angular-momentum baseline.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.eccs.clear();
        self.ecc_bins.fill(0);
        self.l0 = None;
    }

    pub fn maybe_sample(&mut self, b: &Bodies) {
        if self.last_sample.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.last_sample = Instant::now();

        if let Some(last) = self.samples.back() {
            if b.time < last.t {
                // Time went backwards: a new system replaced the old one.
                self.clear();
            } else if b.time == last.t {
                // Paused: a second sample at the same time adds nothing.
                return;
            }
        }

        let n = b.len();
        let drift = if n <= ph::ENERGY_MAX_N {
            ph::energy_drift_pct(b)
        } else {
            None
        };
        let dissipated = ph::energy_dissipated_pct(b);

        let l = ph::total_angular_momentum(b);
        if self.e0_seen != b.stats.energy0 {
            self.e0_seen = b.stats.energy0;
            self.l0 = None;
        }
        let l0 = *self.l0.get_or_insert(l);
        let l_drift = if l0.abs() > 1e-12 && l.is_finite() {
            Some((l - l0) / l0.abs() * 100.0)
        } else {
            None
        };

        self.samples.push_back(DiagSample {
            t: b.time,
            n,
            drift,
            dissipated,
            l_drift,
            nsub: b.stats.last_nsub,
        });
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }

        self.eccs.clear();
        // Every k-th candidate, so a large run is sampled across all systems
        // instead of only the first 500 bodies by index.
        let candidates: Vec<usize> = (0..b.len())
            .filter(|&i| {
                b.origin[i] == BodyOrigin::Spawned
                    && matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
            })
            .collect();
        let step = candidates.len().div_ceil(500).max(1);
        for &i in candidates.iter().step_by(step) {
            if let Some(p) = parent_of(b, i) {
                let o = ph::orbit_info(b, i, p);
                // Hyperbolic bodies (e > 1) have no bin: only bound orbits count.
                if o.bound && o.ecc.is_finite() {
                    self.eccs.push(o.ecc);
                }
            }
        }
        // `min(count - 1)` puts e = 1.0 in the last bin.
        self.ecc_bins.fill(0);
        for &e in &self.eccs {
            let bin = ((e / self.ecc_bin_width) as usize).min(self.ecc_bin_count - 1);
            self.ecc_bins[bin] += 1;
        }
    }

    /// The history as CSV text (time in simulation units).
    pub fn to_csv(&self) -> String {
        let opt = |v: Option<f64>| v.map(|x| format!("{x:e}")).unwrap_or_default();
        let mut s = String::from(
            "t,n,energy_drift_pct,energy_dissipated_pct,ang_mom_drift_pct,fine_steps\n",
        );
        for p in &self.samples {
            s.push_str(&format!(
                "{:e},{},{},{},{},{}\n",
                p.t,
                p.n,
                opt(p.drift),
                opt(p.dissipated),
                opt(p.l_drift),
                p.nsub
            ));
        }
        s
    }
}

impl Default for DiagHistory {
    fn default() -> Self {
        Self::new()
    }
}

