//! Methods of `SimState` (actions), split out of `main.rs`.

use super::*;

impl SimState {
    /// Apply one frame's worth of UI actions to the simulation state.
    /// Order matters: spawning comes first so later actions see the new
    /// system, edits come after selection so they act on the new target.
    pub(super) fn apply_actions(
        &mut self,
        mut act: Actions,
        ui_state: &mut UiState,
        vp: [f32; 2],
        b: &mut Bodies,
        shared: &Shared,
    ) {
        self.apply_spawn_actions(&mut act, ui_state, vp, b, shared);
        self.apply_run_control(&mut act, ui_state, b);
        self.apply_view_actions(&act, vp, b);
        self.apply_edit_actions(&mut act, b);
        self.apply_display_settings(&act, ui_state, b);
    }

    /// Systems slider, preset loading, preset / seed / count changes, reset.
    pub(super) fn apply_spawn_actions(
        &mut self,
        act: &mut Actions,
        ui_state: &mut UiState,
        vp: [f32; 2],
        b: &mut Bodies,
        shared: &Shared,
    ) {
        // Systems slider: respawn immediately. Works for both presets, so
        // the user can go from 1 random system to N random systems in one
        // click without touching the preset selector.
        self.poll_spawn(b, shared, vp, ui_state);
        if let Some(n) = act.systems {
            // Only applies when Random is active (no library preset).
            if self.preset == Preset::Random && self.last_loaded_preset.is_none() {
                self.systems = n.clamp(1, physics::MAX_SOLAR_SYSTEMS);
                let (count, seed) = (self.body_count, self.seed);
                self.begin_spawn(count, seed);
            }
        }

        if let Some(slug) = act.load_preset.take() {
            if let Some(preset) = physics::load_preset_by_slug(&slug) {
                physics::spawn_from_preset(b, &preset);
                physics::set_evolution_rate(b, self.evolution_rate);
                b.sn_kicks = self.sn_kicks;
                b.mass_transfer = self.mass_transfer;
                self.body_count = b.len();
                // The preset is one system; the mode stays `Random`.
                self.preset = Preset::Random;
                self.last_loaded_preset = Some(slug.clone());
                ui_state.loaded_preset_slug = Some(slug);
                ui_state.pending_count = b.len();
                self.fit(b, vp);
                self.orbit_buf_valid = false;
                self.selected = None;
                self.follow = false;
                shared.world_replaced();
                self.trail.clear();
            }
        }

        if let Some(p) = act.preset.take() {
            self.preset = p;
            self.last_loaded_preset = None;
            ui_state.loaded_preset_slug = None;
            match p {
                Preset::Random => {
                    let (count, seed) = (ui_state.pending_count, ui_state.pending_seed(self.seed));
                    self.begin_spawn(count, seed);
                }
                Preset::SolarSystem => {
                    self.spawn_job = None;
                    self.queued_spawn = None;
                    let seed = self.seed;
                    self.reset_with(b, shared, 9, seed);
                    self.fit(b, vp);
                }
            }
        } else if let (Some(n), Some(seed)) = (act.body_count, act.seed) {
            self.begin_spawn(n, seed);
        }
        if act.reset {
            self.reset(b, shared);
            self.fit(b, vp); // ← added
            ui_state.set_pending_seed(self.seed);
        }
    }

    /// Save / load, pause, single step, speed, vsync and orbit-line toggles.
    pub(super) fn apply_run_control(&mut self, act: &mut Actions, ui_state: &mut UiState, b: &mut Bodies) {
        if let Some(name) = act.save.take() {
            self.save_game(b, &name, ui_state);
        }
        if let Some(name) = act.load.take() {
            // Read from disk after the lock is released (see `App::flush_io`).
            self.pending_load = Some(name);
        }
        if act.toggle_pause {
            self.paused = !self.paused;
        }
        if act.step_once {
            physics::advance(b, physics::DT);
        }
        if let Some(s) = act.speed {
            self.speed = s.clamp(0.01, 100.0);
        }
        if let Some(v) = act.vsync {
            self.vsync = v;
        }
        if let Some(o) = act.orbits {
            self.show_orbits = o;
            self.orbit_buf_valid = false;
        }
        if let Some(o) = act.debris_orbits {
            self.show_debris_orbits = o;
            self.orbit_buf_valid = false;
        }
    }

    /// Camera, zoom, selection and follow.
    pub(super) fn apply_view_actions(&mut self, act: &Actions, vp: [f32; 2], b: &mut Bodies) {
        if let Some((yaw, pitch)) = act.view_angles {
            self.yaw = yaw;
            self.pitch = pitch.clamp(-view::PITCH_LIMIT, view::PITCH_LIMIT);
        }
        if let Some(f) = act.zoom_mul {
            self.zoom_at(f, [vp[0] * 0.5, vp[1] * 0.5], vp);
        }
        if act.fit {
            self.fit(b, vp);
        }
        if let Some(sel) = act.select {
            self.selected = sel;
            if sel.is_none() {
                self.follow = false;
            }
        }
        if let Some(c) = act.center_on {
            self.cam = c;
            self.follow = false;
        }
        if let Some([fx, fy, fr]) = act.focus {
            self.cam = [fx, fy];
            let half = 0.5 * vp[0].min(vp[1]) as f64;
            self.zoom = ((half / (fr * 1.15).max(1.0)) as f32).clamp(ZOOM_MIN, ZOOM_MAX);
            self.follow = false;
            self.follow_smooth = None;
            self.orbit_buf_valid = false;
        }
        if let Some(f) = act.follow {
            self.follow = f && self.selected.is_some();
        }
        if act.cycle != 0 {
            self.cycle_selection(b, act.cycle);
        }
    }

    /// Edits to the selected body, or to a multi-selection.
    pub(super) fn apply_edit_actions(&mut self, act: &mut Actions, b: &mut Bodies) {
        if let Some(edit) = act.edit.take() {
            if let Some(i) = self.selected.and_then(|id| b.index_of_id(id)) {
                self.apply_edit(b, i, edit);
                self.orbit_buf_valid = false;
            }
        }
        if let Some((ids, edit)) = act.multi_edit.take() {
            self.apply_multi_edit(b, &ids, edit);
        }
    }

    /// Apply one edit to body `i`.
    pub(super) fn apply_edit(&mut self, b: &mut Bodies, i: usize, edit: Edit) {
        match edit {
            Edit::MassMul(f) => physics::scale_mass(b, i, f),
            Edit::SetKind(k) => physics::set_kind(b, i, k),
            Edit::Circularize => {
                physics::circularize(b, i);
            }
            Edit::Delete => {
                physics::delete_body(b, i);
                self.selected = None;
                self.follow = false;
            }
            Edit::SpawnMoon(n) => {
                let mut rng = std::mem::replace(&mut b.rng, physics::XorShift::new(1));
                for _ in 0..n.clamp(1, 8) {
                    physics::spawn_moon(b, i, &mut rng);
                }
                b.rng = rng;
                self.orbit_buf_valid = false;
            }
            Edit::VelMul(f) => {
                // Scale the speed relative to the parent (or the
                // primary); with neither, scale the absolute speed.
                let refb = physics::find_parent(b, i)
                    .or_else(|| physics::primary_index(b).filter(|&p| p != i));
                let (rvx, rvy) = match refb {
                    Some(p) => (b.vx[p], b.vy[p]),
                    None => (0.0, 0.0),
                };
                b.vx[i] = rvx + (b.vx[i] - rvx) * f;
                b.vy[i] = rvy + (b.vy[i] - rvy) * f;
                b.parent_cache_valid = false;
                self.orbit_buf_valid = false;
            }
            Edit::Rename(name) => {
                let t = name.trim();
                b.name[i] = if t.is_empty() {
                    None
                } else {
                    Some(t.to_string())
                };
            }
            Edit::AddRing(n) => {
                let mut rng = std::mem::replace(&mut b.rng, physics::XorShift::new(1));
                physics::add_ring(b, i, n.clamp(5, 200), &mut rng);
                b.rng = rng;
                self.orbit_buf_valid = false;
            }
        }
    }

    /// Apply one edit to every body of a multi-selection, in batched calls
    /// so the O(N^2) physics commit runs once for the whole selection.
    pub(super) fn apply_multi_edit(&mut self, b: &mut Bodies, ids: &[u64], edit: Edit) {
        let idx: Vec<usize> = ids.iter().filter_map(|&id| b.index_of_id(id)).collect();
        if !idx.is_empty() {
            // One batched call each: the O(N^2) physics commit runs once
            // for the whole selection, not once per body.
            match edit {
                Edit::MassMul(f) => {
                    let e: Vec<(usize, f64)> = idx.iter().map(|&i| (i, f)).collect();
                    physics::scale_masses(b, &e);
                }
                Edit::SetKind(k) => {
                    let e: Vec<(usize, BodyKind)> = idx.iter().map(|&i| (i, k)).collect();
                    physics::set_kinds(b, &e);
                }
                Edit::Circularize => {
                    physics::circularize_many(b, &idx);
                }
                Edit::Delete => {
                    if self.selected.is_some_and(|s| ids.contains(&s)) {
                        self.selected = None;
                        self.follow = false;
                    }
                    physics::delete_bodies(b, idx);
                }
                Edit::SpawnMoon(_) | Edit::AddRing(_) | Edit::VelMul(_) | Edit::Rename(_) => {}
            }
            self.orbit_buf_valid = false;
        }
    }

    /// Trails, diagnostics, stellar-evolution options, pause-on-event.
    pub(super) fn apply_display_settings(&mut self, act: &Actions, ui_state: &mut UiState, b: &mut Bodies) {
        if let Some(t) = act.trails {
            self.trail.cfg.enabled = t;
        }
        if let Some(v) = act.diag_visible {
            ui_state.diag.visible = v;
        }
        if let Some(v) = act.trail_age {
            self.trail.cfg.max_age = v;
        }
        if let Some(v) = act.trail_bodies {
            self.trail.cfg.max_bodies = v.max(1);
        }
        if let Some(v) = act.trail_non_rubble {
            self.trail.cfg.non_rubble_only = v;
        }
        if let Some(v) = act.sn_kicks {
            self.sn_kicks = v;
            b.sn_kicks = v;
        }
        if let Some(v) = act.mass_transfer {
            self.mass_transfer = v;
            b.mass_transfer = v;
        }
        if let Some(v) = act.aging_mult {
            self.aging_mult = v.max(1.0);
            self.evolution_rate = physics::time_unit_s() / SECONDS_PER_GYR * self.aging_mult;
            physics::set_evolution_rate(b, self.evolution_rate);
        }
        if let Some(p) = act.pause_on_event {
            self.pause_on_event = p;
        }
    }
}

impl SimState {
    /// Generate a random population on a worker thread and install it from
    /// `poll_spawn` when it is ready. Generating many systems takes seconds
    /// (each one runs its early evolution), which would otherwise freeze the
    /// window while the bodies lock is held. Only one worker runs at a time:
    /// a request made meanwhile is queued, and a newer one replaces it, so
    /// dragging sliders cannot pile up heavy jobs.
    pub(super) fn begin_spawn(&mut self, count: usize, seed: u128) {
        let systems = self.systems.clamp(1, physics::MAX_SOLAR_SYSTEMS);
        if self.spawn_job.is_some() {
            self.queued_spawn = Some((count, seed, systems));
        } else {
            self.start_spawn(count, seed, systems);
        }
    }

    fn start_spawn(&mut self, count: usize, seed: u128, systems: usize) {
        let body_count = count.max(1);
        let handle = std::thread::spawn(move || {
            let mut world = Bodies::with_capacity(16);
            physics::spawn_random_systems(&mut world, body_count, seed, systems);
            world
        });
        self.spawn_job = Some(SpawnJob { handle, count, seed, systems });
    }

    /// Install a finished background spawn, if there is one.
    pub(super) fn poll_spawn(&mut self, b: &mut Bodies, shared: &Shared, vp: [f32; 2], _ui_state: &mut UiState) {
        if !self.spawn_job.as_ref().is_some_and(|j| j.handle.is_finished()) {
            return;
        }
        let Some(job) = self.spawn_job.take() else { return };
        let Ok(world) = job.handle.join() else {
            self.queued_spawn = None;
            return;
        };
        // A newer request supersedes this result: start it instead.
        if let Some((count, seed, systems)) = self.queued_spawn.take() {
            if self.preset == Preset::Random && self.last_loaded_preset.is_none() {
                self.start_spawn(count, seed, systems);
            }
            return;
        }
        if self.preset != Preset::Random || self.last_loaded_preset.is_some() {
            return;
        }
        self.systems = job.systems;
        self.prebuilt = Some(world);
        self.reset_with(b, shared, job.count, job.seed);
        self.fit(b, vp);
    }
}
