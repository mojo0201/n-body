//! Methods of `SimState` (input), split out of `main.rs`.

use super::*;

impl SimState {
    /// Left button: press starts a canvas drag; a release without movement is
    /// a click that picks the body under the cursor (or clears the selection).
    pub(super) fn on_left_button(&mut self, pressed: bool, vp: [f32; 2], shared: &Shared) {
        if pressed {
            self.dragging = true;
            self.press_pos = self.cursor;
            self.moved_px = 0.0;
            return;
        }
        if !self.dragging {
            return;
        }
        self.dragging = false;
        if self.moved_px >= CLICK_SLOP_PX {
            return;
        }
        let hit = match &self.replay_cache {
            Some((_, rb)) if self.replay.is_some() => self.pick(rb, self.cursor, vp),
            _ => {
                let b = shared.lock_for_render();
                self.pick(&b, self.cursor, vp)
            }
        };
        self.selected = hit;
        if hit.is_none() {
            self.follow = false;
        }
    }

    /// Right button: press starts turning the camera about the point under
    /// the cursor, not the screen centre. Zoomed in on one system among many,
    /// the centre can be far from it and the system would sweep away.
    pub(super) fn on_right_button(&mut self, pressed: bool, vp: [f32; 2]) {
        if !pressed {
            self.rotating = false;
            self.rot_anchor = None;
            return;
        }
        self.rotating = true;
        let p = self.cursor;
        self.rot_anchor = if self.follow {
            None // following already pivots about the body
        } else {
            Some((self.view().screen_to_world(p, vp), p))
        };
    }

    /// Cursor motion: turns the camera while the right button is held, pans
    /// while the left is dragged past the click slop.
    pub(super) fn on_cursor_moved(&mut self, pos: [f32; 2], vp: [f32; 2]) {
        if self.rotating {
            let mut v = self.view();
            v.orbit_px((pos[0] - self.cursor[0]) as f64, (pos[1] - self.cursor[1]) as f64);
            if let Some((world, screen)) = self.rot_anchor {
                // Slide the focus so the anchor stays where it was.
                let s = v.to_screen(world, vp);
                v.pan_px(screen[0] as f64 - s[0], screen[1] as f64 - s[1]);
            }
            self.set_view(&v);
            self.orbit_buf_valid = false;
        }
        if self.dragging {
            let dx = pos[0] - self.cursor[0];
            let dy = pos[1] - self.cursor[1];
            self.moved_px += dx.abs() + dy.abs();
            if self.moved_px >= CLICK_SLOP_PX {
                self.pan_by_pixels(dx, dy);
            }
        }
        self.cursor = pos;
    }

    /// Pan by a screen-space drag. Always leaves follow mode and clears the
    /// stale smoothed position.
    pub(super) fn pan_by_pixels(&mut self, dx: f32, dy: f32) {
        self.pan_gm(-dx as f64 / self.zoom as f64, -dy as f64 / self.zoom as f64);
        self.follow = false;
        self.follow_smooth = None;
    }

    /// Scroll: a mouse wheel (or Ctrl/Cmd + scroll, a pinch on Windows and
    /// Linux) zooms at the cursor; a two-finger touchpad scroll pans.
    pub(super) fn on_scroll(&mut self, delta: MouseScrollDelta, zoom_modifier: bool, vp: [f32; 2]) {
        // Convert to pixels. A mouse wheel sends whole lines on the y axis
        // only. A touchpad sends fractions or sideways motion.
        let (dx, dy, touchpad) = match delta {
            MouseScrollDelta::LineDelta(x, y) => (x * 30.0, y * 30.0, x != 0.0 || y.fract() != 0.0),
            MouseScrollDelta::PixelDelta(p) => (p.x as f32, p.y as f32, true),
        };
        if zoom_modifier || !touchpad {
            let cursor = self.cursor;
            let dz = if dy.abs() >= dx.abs() { dy } else { dx };
            self.zoom_at((dz / 30.0 * 0.15).exp(), cursor, vp);
        } else {
            self.pan_by_pixels(dx, dy);
        }
    }

    /// The current camera as a `View3`.
    pub(super) fn view(&self) -> View3 {
        View3::new(
            [self.cam[0], self.cam[1], self.cam_z],
            self.yaw,
            self.pitch,
            self.zoom as f64,
        )
    }

    pub(super) fn set_view(&mut self, v: &View3) {
        self.cam = [v.cam[0], v.cam[1]];
        self.cam_z = v.cam[2];
        self.yaw = v.yaw;
        self.pitch = v.pitch;
    }

    /// Slide the focus along the screen axes by `(vx, vy)` Gm.
    pub(super) fn pan_gm(&mut self, vx: f64, vy: f64) {
        let d = self.view().unrotate([vx, vy, 0.0]);
        self.cam[0] += d[0];
        self.cam[1] += d[1];
        self.cam_z += d[2];
    }

    /// Zoom about a screen point: the world point under the cursor stays put.
    pub(super) fn zoom_at(&mut self, factor: f32, p: [f32; 2], vp: [f32; 2]) {
        let before = self.view().screen_to_world(p, vp);
        self.zoom = (self.zoom * factor).clamp(ZOOM_MIN, ZOOM_MAX);
        let mut v = self.view();
        let s = v.to_screen(before, vp);
        v.pan_px(p[0] as f64 - s[0], p[1] as f64 - s[1]);
        self.set_view(&v);
        self.orbit_buf_valid = false;
    }

    pub(super) fn fit(&mut self, b: &Bodies, vp: [f32; 2]) {
        // Multi-system ring: only Random, and not while a library preset is
        // active. Presets are always single-system.
        if self.preset == Preset::Random && self.last_loaded_preset.is_none() && self.systems > 1 {
            let mut max_r2: f64 = 0.0;
            for i in 0..b.len() {
                let r2 = b.x[i] * b.x[i] + b.y[i] * b.y[i] + b.z[i] * b.z[i];
                if r2 > max_r2 {
                    max_r2 = r2;
                }
            }
            let radius = (max_r2.sqrt() * 1.15).max(50.0);
            self.cam = [0.0, 0.0];
            self.cam_z = 0.0;
            let half = 0.5 * vp[0].min(vp[1]) as f64;
            self.zoom = ((half / radius) as f32).clamp(ZOOM_MIN, ZOOM_MAX);
            self.orbit_buf_valid = false;
            return;
        }

        // Single Solar System preset: frame the whole extent (Neptune at
        // ~30 AU ≈ 4500 Gm), matching "fit the whole system".
        if self.preset == Preset::SolarSystem && self.last_loaded_preset.is_none() {
            let radius = physics::solar_extent_gm(1);
            let half = 0.5 * vp[0].min(vp[1]) as f64;
            self.cam = [0.0, 0.0];
            self.cam_z = 0.0;
            self.zoom = ((half / radius) as f32).clamp(ZOOM_MIN, ZOOM_MAX);
            self.orbit_buf_valid = false;
            return;
        }

        // Single Random, or a library preset: fit to body distances from the
        // primary.
        let Some(prim) = physics::primary_index(b) else {
            return;
        };
        let mut d: Vec<f64> = (0..b.len())
            .filter(|&i| i != prim)
            .map(|i| {
                ((b.x[i] - b.x[prim]).powi(2)
                    + (b.y[i] - b.y[prim]).powi(2)
                    + (b.z[i] - b.z[prim]).powi(2))
                .sqrt()
            })
            .collect();
        let radius = if d.is_empty() {
            50.0
        } else {
            d.sort_by(|a, c| a.partial_cmp(c).unwrap_or(std::cmp::Ordering::Equal));
            let k = ((d.len() as f64 * 0.92) as usize).min(d.len() - 1);
            (d[k] * 1.15).max(5.0)
        };
        self.cam = [b.x[prim], b.y[prim]];
        self.cam_z = b.z[prim];
        let half = 0.5 * vp[0].min(vp[1]) as f64;
        self.zoom = ((half / radius) as f32).clamp(ZOOM_MIN, ZOOM_MAX);
        self.orbit_buf_valid = false;
    }

    pub(super) fn pick(&self, b: &Bodies, p: [f32; 2], vp: [f32; 2]) -> Option<u64> {
        let view = self.view();
        // Nearest to the cursor wins; on a near tie the body closer to the
        // viewer (smaller view-space depth) wins, as it is drawn on top.
        let mut best: Option<(f64, f64, usize)> = None;
        for i in 0..b.len() {
            // Match the exact position the renderer uses (follow smooth or raw).
            let pos = match self.follow_smooth {
                Some((id, pos)) if b.id[i] == id => pos,
                _ => [b.x[i], b.y[i], b.z[i]],
            };
            let v = view.project(pos);
            let sx = vp[0] as f64 * 0.5 + v[0] * self.zoom as f64;
            let sy = vp[1] as f64 * 0.5 + v[1] * self.zoom as f64;
            let dx = sx - p[0] as f64;
            let dy = sy - p[1] as f64;
            let d = (dx * dx + dy * dy).sqrt();

            // Black holes are drawn with an accretion disk 6x the event horizon.
            // The minimum pick radius must match the visual size (24px).
            let rpx = match b.kind[i] {
                BodyKind::BlackHole => (b.r[i] * self.zoom as f64).max(24.0),
                _ => (b.r[i] * self.zoom as f64).max(8.0),
            };

            if d <= rpx
                && best.is_none_or(|(s, depth, _)| d < s - 1.0 || (d <= s + 1.0 && v[2] < depth))
            {
                best = Some((d, v[2], i));
            }
        }
        best.map(|(_, _, i)| b.id[i])
    }

    pub(super) fn cycle_selection(&mut self, b: &Bodies, dir: i32) {
        let prim = physics::primary_index(b);
        let mut list: Vec<(f64, u64)> = (0..b.len())
            .filter(|&i| {
                matches!(b.kind[i], BodyKind::Planet | BodyKind::DwarfPlanet)
                    && b.origin[i] == BodyOrigin::Spawned
            })
            .map(|i| {
                let d = match prim {
                    Some(p) => ((b.x[i] - b.x[p]).powi(2)
                        + (b.y[i] - b.y[p]).powi(2)
                        + (b.z[i] - b.z[p]).powi(2))
                    .sqrt(),
                    None => 0.0,
                };
                (d, b.id[i])
            })
            .collect();
        if list.is_empty() {
            return;
        }
        list.sort_by(|a, c| a.0.partial_cmp(&c.0).unwrap_or(std::cmp::Ordering::Equal));
        let cur = self
            .selected
            .and_then(|id| list.iter().position(|e| e.1 == id));
        let n = list.len() as i32;
        let next = match cur {
            Some(k) => (k as i32 + dir).rem_euclid(n),
            None => {
                if dir >= 0 {
                    0
                } else {
                    n - 1
                }
            }
        };
        self.selected = Some(list[next as usize].1);
    }
}
