//! Split out of `main.rs`.

use super::*;

pub(super) struct TrailConfig {
    pub(super) max_age: f64,
    pub(super) max_bodies: usize,
    pub(super) non_rubble_only: bool,
    pub(super) enabled: bool,
}

impl Default for TrailConfig {
    fn default() -> Self {
        Self {
            max_age: 3.0,
            max_bodies: 64,
            non_rubble_only: true,
            enabled: true,
        }
    }
}

/// (id, sim_time, position, rgba, period, parent_pos). Positions stay f64 so
/// deep zoom does not jitter; they become camera-relative f32 only in
/// `build_vertices`.
pub(super) type TrailSample = (u64, f64, [f64; 3], [u8; 4], f64, Option<[f64; 3]>);

pub(super) struct Trail {
    pub(super) history: VecDeque<(f64, [f64; 3])>,
    pub(super) color: [u8; 4],
}

pub(super) const TRAIL_MAX_POINTS: usize = 4000;
pub(super) const TRAIL_MAX_VERTS: usize = 100_000;

#[inline]
pub(super) fn v3_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(super) fn v3_len(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Intermediate points between two trail samples. Around a parent, the
/// segment is a great-circle arc (radius interpolated) so trails stay round
/// even when the sample spacing is coarse; without a parent it is a straight
/// line. Returns the number of steps and a function giving the point at a
/// fraction in (0, 1).
pub(super) fn trail_arc(last: [f64; 3], now: [f64; 3], parent: Option<[f64; 3]>) -> (usize, Box<dyn Fn(f64) -> [f64; 3]>) {
    let lerp = move |f: f64| -> [f64; 3] {
        [
            last[0] + (now[0] - last[0]) * f,
            last[1] + (now[1] - last[1]) * f,
            last[2] + (now[2] - last[2]) * f,
        ]
    };
    let linear_steps = ((v3_len(v3_sub(now, last)) / 0.01).ceil() as usize).clamp(1, 256);
    if let Some(p) = parent {
        let (r0, r1) = (v3_sub(last, p), v3_sub(now, p));
        let (l0, l1) = (v3_len(r0), v3_len(r1));
        if l0 > 1e-6 && l1 > 1e-6 {
            let cosang = ((r0[0] * r1[0] + r0[1] * r1[1] + r0[2] * r1[2]) / (l0 * l1)).clamp(-1.0, 1.0);
            let ang = cosang.acos();
            let sin_a = ang.sin();
            if sin_a > 1e-6 {
                let steps = ((ang / 0.035).ceil() as usize).clamp(1, 256);
                return (
                    steps,
                    Box::new(move |f: f64| {
                        let w0 = ((1.0 - f) * ang).sin() / sin_a;
                        let w1 = (f * ang).sin() / sin_a;
                        let r = l0 + (l1 - l0) * f;
                        // Unit-vector slerp, scaled to the interpolated radius.
                        [
                            p[0] + r * (w0 * r0[0] / l0 + w1 * r1[0] / l1),
                            p[1] + r * (w0 * r0[1] / l0 + w1 * r1[1] / l1),
                            p[2] + r * (w0 * r0[2] / l0 + w1 * r1[2] / l1),
                        ]
                    }),
                );
            }
        }
    }
    (linear_steps, Box::new(lerp))
}

pub(super) struct TrailState {
    pub(super) history: HashMap<u64, Trail>,
    pub(super) cfg: TrailConfig,
}

impl TrailState {
    pub(super) fn new() -> Self {
        Self {
            history: HashMap::new(),
            cfg: TrailConfig::default(),
        }
    }

    pub(super) fn clear(&mut self) {
        self.history.clear();
    }

    /// O(n) under the lock. Returns (id, sim_time, pos, rgb, period, parent_pos).
    /// `parent_pos` is used for arc sub-sampling in `update`.
    pub(super) fn snapshot(&self, b: &Bodies, override_pos: Option<(u64, [f64; 3])>) -> Vec<TrailSample> {
        if !self.cfg.enabled {
            return Vec::new();
        }
        let cap = self.cfg.max_bodies.min(b.len());
        let mut out = Vec::with_capacity(cap);
        for i in 0..b.len() {
            if out.len() >= self.cfg.max_bodies {
                break;
            }
            if self.cfg.non_rubble_only && (b.origin[i].is_belt() || b.origin[i].is_rubble()) {
                continue;
            }
            let col = render::body_color(b, i);
            let color = [
                (col[0].clamp(0.0, 1.0) * 255.0) as u8,
                (col[1].clamp(0.0, 1.0) * 255.0) as u8,
                (col[2].clamp(0.0, 1.0) * 255.0) as u8,
                255,
            ];
            // Fetch orbital period and parent position for curve sub-sampling.
            let (period, parent_pos) = b
                .parent_cache
                .get(i)
                .copied()
                .flatten()
                .filter(|&p| p != i)
                .map(|p| {
                    let o = crate::physics::orbit_info(b, i, p);
                    (o.period, Some([b.x[p], b.y[p], b.z[p]]))
                })
                .unwrap_or((f64::INFINITY, None));
            let pos = match override_pos {
                Some((id, pos)) if b.id[i] == id => pos,
                _ => [b.x[i], b.y[i], b.z[i]],
            };
            out.push((b.id[i], b.time, pos, color, period, parent_pos));
        }
        out
    }

    /// Cheap; runs without the lock. Uses the snapshot to update per-body
    /// history.
    pub(super) fn update(&mut self, snapshot: &[TrailSample]) {
        let max_age = self.cfg.max_age;
        let now_ids: HashSet<u64> = snapshot.iter().map(|s| s.0).collect();
        self.history.retain(|id, _| now_ids.contains(id));

        for &(id, t, pos, color, period, parent_pos) in snapshot {
            let trail = self.history.entry(id).or_insert_with(|| Trail {
                history: VecDeque::new(),
                color,
            });
            trail.color = color;
            let is_new = trail
                .history
                .back()
                .is_none_or(|&(tl, _)| t > tl + 1e-12);

            if is_new {
                if let Some(&(t_last, last)) = trail.history.back() {
                    let dt = t - t_last;
                    let (steps, at) = trail_arc(last, pos, parent_pos);
                    for s in 1..steps {
                        let frac = s as f64 / steps as f64;
                        trail.history.push_back((t_last + dt * frac, at(frac)));
                    }
                }
                trail.history.push_back((t, pos));
            }

            // Cap trail length to exactly one orbital period (plus 5% margin)
            let effective_age = if period.is_finite() && period > 0.0 {
                max_age.min(period * 1.05)
            } else {
                max_age
            };
            let t_newest = trail.history.back().map(|v| v.0).unwrap_or(t);
            while let Some(&(t0, _)) = trail.history.front() {
                if t_newest - t0 > effective_age {
                    trail.history.pop_front();
                } else {
                    break;
                }
            }
            // Global hard cap
            while trail.history.len() > TRAIL_MAX_POINTS {
                trail.history.pop_front();
            }
        }
    }

    /// Heavy; runs without the lock. Turns the per-body history into
    /// line-list vertices with alpha fading toward the tail. Positions are
    /// made relative to `cam` in f64 before the cast to f32, and the renderer
    /// draws trails with the relative camera. Trails are visited in id order
    /// and, when the vertex budget is exceeded, every trail is decimated by
    /// the same stride, so the result does not depend on hash order.
    pub(super) fn build_vertices(&self, cam: [f64; 3]) -> Vec<OrbitVertex> {
        let mut ids: Vec<u64> = self.history.keys().copied().collect();
        ids.sort_unstable();
        let total_segs: usize = self
            .history
            .values()
            .map(|t| t.history.len().saturating_sub(1))
            .sum();
        let stride = (total_segs * 2).div_ceil(TRAIL_MAX_VERTS).max(1);
        let mut out = Vec::with_capacity((total_segs * 2 / stride).min(TRAIL_MAX_VERTS) + 2);
        'outer: for id in ids {
            let trail = &self.history[&id];
            let n = trail.history.len();
            if n < 2 {
                continue;
            }
            let t_last = trail.history[n - 1].0;
            let t_first = trail.history[0].0;

            // Fade over the actual span of this trail, not the config max_age.
            let span = (t_last - t_first).max(1e-9);
            let rgb = [trail.color[0], trail.color[1], trail.color[2]];
            let alpha = |t: f64| -> u8 {
                // 30% alpha floor at the tail so the trail doesn't vanish.
                ((((1.0 - (t_last - t) / span) * 0.7 + 0.3).clamp(0.0, 1.0)) * 200.0) as u8
            };
            let rel = |p: [f64; 3]| -> [f32; 3] {
                [(p[0] - cam[0]) as f32, (p[1] - cam[1]) as f32, (p[2] - cam[2]) as f32]
            };
            let mut k = 0;
            while k < n - 1 {
                let k1 = (k + stride).min(n - 1);
                let (t0, p0) = trail.history[k];
                let (t1, p1) = trail.history[k1];
                out.push(OrbitVertex {
                    pos: rel(p0),
                    color: [rgb[0], rgb[1], rgb[2], alpha(t0)],
                });
                out.push(OrbitVertex {
                    pos: rel(p1),
                    color: [rgb[0], rgb[1], rgb[2], alpha(t1)],
                });
                if out.len() >= TRAIL_MAX_VERTS {
                    break 'outer;
                }
                k = k1;
            }
        }
        out
    }
}

