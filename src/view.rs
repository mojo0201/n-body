//! Orthographic 3D view: a focus point, a yaw about the sim's z axis and a
//! pitch (tilt) away from top-down, plus the zoom in pixels per Gm.
//!
//! With `yaw = pitch = 0` the rotation is the exact identity, so the picture
//! is the old top-down 2D view bit for bit. View space has x to the right,
//! y down the screen (as in the 2D renderer) and z *into* the screen, so a
//! larger view z is farther away.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View3 {
    pub cam: [f64; 3],
    pub yaw: f64,
    pub pitch: f64,
    pub zoom: f64,
}

pub const PITCH_LIMIT: f64 = std::f64::consts::FRAC_PI_2 - 0.01;

impl View3 {
    pub fn new(cam: [f64; 3], yaw: f64, pitch: f64, zoom: f64) -> Self {
        Self { cam, yaw, pitch: pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT), zoom }
    }

    /// True when the view looks straight down the z axis (yaw only spins it).
    pub fn is_top_down(&self) -> bool {
        self.pitch == 0.0
    }

    /// World → view rotation, row-major: `Rx(pitch) · Rz(yaw)`.
    pub fn rot(&self) -> [[f64; 3]; 3] {
        let (s, c) = if self.yaw == 0.0 { (0.0, 1.0) } else { self.yaw.sin_cos() };
        let (sp, cp) = if self.pitch == 0.0 { (0.0, 1.0) } else { self.pitch.sin_cos() };
        [[c, -s, 0.0], [cp * s, cp * c, -sp], [sp * s, sp * c, cp]]
    }

    /// Rotate a world-space offset into view space (no translation).
    #[inline]
    pub fn rotate(&self, d: [f64; 3]) -> [f64; 3] {
        let r = self.rot();
        [
            r[0][0] * d[0] + r[0][1] * d[1] + r[0][2] * d[2],
            r[1][0] * d[0] + r[1][1] * d[1] + r[1][2] * d[2],
            r[2][0] * d[0] + r[2][1] * d[1] + r[2][2] * d[2],
        ]
    }

    /// View-space offset (Gm) of world point `p` from the focus.
    #[inline]
    pub fn project(&self, p: [f64; 3]) -> [f64; 3] {
        self.rotate([p[0] - self.cam[0], p[1] - self.cam[1], p[2] - self.cam[2]])
    }

    /// Screen position in pixels of world point `p`, given the viewport size.
    #[inline]
    pub fn to_screen(&self, p: [f64; 3], vp: [f32; 2]) -> [f64; 2] {
        let v = self.project(p);
        [vp[0] as f64 * 0.5 + v[0] * self.zoom, vp[1] as f64 * 0.5 + v[1] * self.zoom]
    }

    /// Inverse rotation (view → world), applied to a view-space offset.
    #[inline]
    pub fn unrotate(&self, v: [f64; 3]) -> [f64; 3] {
        let r = self.rot();
        [
            r[0][0] * v[0] + r[1][0] * v[1] + r[2][0] * v[2],
            r[0][1] * v[0] + r[1][1] * v[1] + r[2][1] * v[2],
            r[0][2] * v[0] + r[1][2] * v[1] + r[2][2] * v[2],
        ]
    }

    /// World point under a screen pixel. The ray through the pixel is
    /// intersected with the horizontal plane through the focus (z = cam.z);
    /// when the view is nearly edge-on it falls back to the view plane
    /// through the focus. Top-down this is exactly the old 2D mapping.
    pub fn screen_to_world(&self, p: [f32; 2], vp: [f32; 2]) -> [f64; 3] {
        let v = [
            (p[0] as f64 - vp[0] as f64 * 0.5) / self.zoom,
            (p[1] as f64 - vp[1] as f64 * 0.5) / self.zoom,
            0.0,
        ];
        let o = self.unrotate(v);
        let dir = self.unrotate([0.0, 0.0, 1.0]);
        let t = if dir[2].abs() > 0.05 { -o[2] / dir[2] } else { 0.0 };
        [
            self.cam[0] + o[0] + t * dir[0],
            self.cam[1] + o[1] + t * dir[1],
            self.cam[2] + o[2] + t * dir[2],
        ]
    }

    /// Slide the focus so the picture follows a mouse drag of `(dx, dy)` px.
    pub fn pan_px(&mut self, dx: f64, dy: f64) {
        let d = self.unrotate([-dx / self.zoom, -dy / self.zoom, 0.0]);
        for k in 0..3 {
            self.cam[k] += d[k];
        }
    }

    /// Turn the view: a horizontal drag spins it about z, a vertical drag
    /// tilts it. Radians per pixel is fixed (0.005: a 600 px drag is about 170 degrees).
    pub fn orbit_px(&mut self, dx: f64, dy: f64) {
        const K: f64 = 0.005;
        self.yaw += dx * K;
        self.pitch = (self.pitch + dy * K).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    pub fn reset_angles(&mut self) {
        self.yaw = 0.0;
        self.pitch = 0.0;
    }

    /// Unit vector (world axes) pointing from the viewer into the scene.
    pub fn view_dir(&self) -> [f64; 3] {
        self.unrotate([0.0, 0.0, 1.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_down_is_identity() {
        let v = View3::new([1.0, 2.0, 0.0], 0.0, 0.0, 3.0);
        let p = v.project([5.0, 7.0, 0.0]);
        assert_eq!(p, [4.0, 5.0, 0.0]);
        let w = v.screen_to_world([400.0, 300.0], [800.0, 600.0]);
        assert_eq!(w, [1.0, 2.0, 0.0]);
    }

    #[test]
    fn round_trip_tilted() {
        let v = View3::new([10.0, -4.0, 2.0], 0.7, 0.9, 2.0);
        let w = [3.0, 5.0, 2.0];
        let s = v.to_screen(w, [800.0, 600.0]);
        let back = v.screen_to_world([s[0] as f32, s[1] as f32], [800.0, 600.0]);
        for k in 0..3 {
            assert!((back[k] - w[k]).abs() < 1e-3, "{back:?} vs {w:?}");
        }
    }
}
