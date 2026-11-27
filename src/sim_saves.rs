//! Methods of `SimState` (saves), split out of `main.rs`.

use super::*;

impl SimState {
    /// Leave replay and return to the live simulation.
    pub(super) fn exit_replay(&mut self) {
        if self.replay.take().is_some() {
            // The live bodies may be inside the cache this frame; `leave_replay`
            // swaps them back and drops the cache.
            if !self.replay_swapped {
                self.replay_cache = None;
            }
            self.orbit_buf_valid = false;
            self.trail.clear();
        }
    }

    /// While replaying, swap the recorded snapshot into `b` for this frame and
    /// return its id. The caller must call `leave_replay` before releasing the
    /// lock, so the physics thread never sees the snapshot.
    pub(super) fn enter_replay(&mut self, shared: &Shared, b: &mut Bodies, frame_dt: f64) -> Option<u64> {
        let (id, arc) = {
            let rp = self.replay.as_mut()?;
            let hist = lock_ok(&shared.history);
            let Some((t_min, t_max)) = hist.span() else {
                drop(hist);
                self.exit_replay();
                return None;
            };
            if rp.playing {
                rp.time += frame_dt * self.speed * SPEED_SCALE;
                if rp.time >= t_max {
                    drop(hist);
                    self.exit_replay();
                    return None;
                }
            }
            rp.time = rp.time.clamp(t_min, t_max);
            let f = hist.nearest(rp.time)?;
            (f.id, f.state.clone())
        };
        if self.replay_cache.as_ref().map(|c| c.0) != Some(id) {
            if self.replay_swapped {
                // A frame is still swapped into `b`, so the cached copy holds
                // the live state and must not be overwritten in place.
                self.replay_cache = Some((id, (*arc).clone()));
            } else if let Some(c) = self.replay_cache.as_mut() {
                // Reuse the cached copy's allocations: `copy_from` refills the
                // column vectors in place, so scrubbing the timeline does not
                // reallocate a multi-megabyte snapshot per frame.
                c.0 = id;
                c.1.copy_from(&arc);
            } else {
                self.replay_cache = Some((id, (*arc).clone()));
            }
        }
        let cache = self.replay_cache.as_mut()?;
        std::mem::swap(b, &mut cache.1);
        self.replay_swapped = true;
        Some(id)
    }

    pub(super) fn leave_replay(&mut self, b: &mut Bodies) {
        if let Some(cache) = self.replay_cache.as_mut() {
            std::mem::swap(b, &mut cache.1);
        }
        self.replay_swapped = false;
        // The replay may have been ended while it was shown (the Live button).
        if self.replay.is_none() {
            self.replay_cache = None;
        }
    }

    pub(super) fn handle_timeline(&mut self, act: TimelineAct, shared: &Shared) {
        let hist = lock_ok(&shared.history);
        let Some((t_min, t_max)) = hist.span() else {
            return;
        };
        match act {
            TimelineAct::Live => {
                drop(hist);
                self.exit_replay();
            }
            TimelineAct::Scrub(t) => {
                let t = t.clamp(t_min, t_max);
                match self.replay.as_mut() {
                    Some(rp) => {
                        rp.time = t;
                        rp.playing = false;
                    }
                    None => {
                        self.replay = Some(Replay { time: t, playing: false });
                        self.follow_smooth = None;
                    }
                }
            }
            TimelineAct::Play(on) => {
                if let Some(rp) = self.replay.as_mut() {
                    rp.playing = on;
                }
            }
            TimelineAct::Step(dir) => {
                let now = self.replay.as_ref().map_or(t_max, |r| r.time);
                if let Some(t) = hist.neighbour_time(now, dir) {
                    match self.replay.as_mut() {
                        Some(rp) => {
                            rp.time = t;
                            rp.playing = false;
                        }
                        None => self.replay = Some(Replay { time: t, playing: false }),
                    }
                }
            }
        }
    }

    pub(super) fn reset_with(&mut self, b: &mut Bodies, shared: &Shared, count: usize, seed: u128) {
        lock_ok(&shared.history).clear();
        self.replay = None;
        self.replay_cache = None;
        self.seed = seed;
        match self.preset {
            Preset::Random => {
                self.body_count = count.max(1);
                let n = self.systems.clamp(1, physics::MAX_SOLAR_SYSTEMS);
                match self.prebuilt.take() {
                    Some(world) => *b = world,
                    None => {
                        physics::spawn_random_systems(b, self.body_count, seed, n);
                    }
                }
            }
            Preset::SolarSystem => {
                // Solar System is a preset: always single-system now.
                // `self.systems` is left untouched so switching back to
                // Random restores the user's previous count.
                physics::spawn_solar_system_seeded(b, seed);
                self.body_count = b.len();
            }
        }
        physics::set_evolution_rate(b, self.evolution_rate);
        b.sn_kicks = self.sn_kicks;
        b.mass_transfer = self.mass_transfer;

        self.orbit_buf_valid = false;
        self.selected = None;
        self.follow = false;
        shared.world_replaced();
        self.trail.clear();
    }

    pub(super) fn reset(&mut self, b: &mut Bodies, shared: &Shared) {
        // A library preset reloads with fresh planet phases.
        if let Some(slug) = self.last_loaded_preset.clone() {
            if let Some(preset) = physics::load_preset_by_slug(&slug) {
                self.seed = self.seed.wrapping_add(1);
                lock_ok(&shared.history).clear();
                self.replay = None;
                self.replay_cache = None;
                physics::spawn_preset_with_seed(b, &preset, self.seed);
                physics::set_evolution_rate(b, self.evolution_rate);
                b.sn_kicks = self.sn_kicks;
                b.mass_transfer = self.mass_transfer;
                self.body_count = b.len();
                self.orbit_buf_valid = false;
                self.selected = None;
                self.follow = false;
                shared.world_replaced();
                self.trail.clear();
                return;
            }
        }
        let seed = self.seed.wrapping_add(1);
        let count = self.body_count;
        self.reset_with(b, shared, count, seed);
    }

    /// Camera, speed and selection, for a save file.
    pub(super) fn app_save(&self, body_count: usize) -> physics::AppSave {
        physics::AppSave {
            cam: self.cam,
            cam_z: self.cam_z,
            yaw: self.yaw,
            pitch: self.pitch,
            zoom: self.zoom,
            speed: self.speed,
            paused: self.paused,
            seed: physics::seed_to_base62(self.seed),
            body_count,
            preset: match self.preset {
                Preset::Random => "random".to_string(),
                Preset::SolarSystem => "solar".to_string(),
            },
            systems: self.systems,
            library_preset: self.last_loaded_preset.clone(),
            selected: self.selected,
            follow: self.follow,
            show_orbits: self.show_orbits,
            show_debris_orbits: self.show_debris_orbits,
            aging_mult: self.aging_mult,
        }
    }

    /// Capture the whole state for `saves/<name>.nbody.json`. Reads `b` only.
    /// The disk write is queued in `deferred_io` and runs after the bodies
    /// lock is released (see `App::flush_io`); its result goes to the UI
    /// message line.
    pub(super) fn save_game(&mut self, b: &Bodies, name: &str, ui_state: &mut UiState) {
        let Some(clean) = physics::sanitize_save_name(name) else {
            ui_state.set_save_msg("The name needs a letter or a digit.", false);
            return;
        };
        let captured = physics::capture_save(b, self.app_save(b.len()));
        self.deferred_io = Some(Box::new(move |ui_state: &mut UiState| {
            let result = captured.and_then(|file| physics::write_save(&clean, &file));
            match result {
                Ok(path) => ui_state.set_save_msg(format!("Saved to {}", path.display()), true),
                Err(e) => ui_state.set_save_msg(format!("Save failed: {e}"), false),
            }
            ui_state.refresh_saves();
        }));
    }

    /// Replace the running state with `saves/<name>.nbody.json`.
    /// If the file is bad, nothing changes and the UI shows the reason.
    ///
    /// The file is read and parsed *before* the bodies lock is taken, so a slow
    /// disk never stalls the physics thread. The caller must NOT hold the lock.
    pub(super) fn load_game(&mut self, name: &str, ui_state: &mut UiState, shared: &Shared) {
        // Only a plain save name may reach the file system.
        if physics::sanitize_save_name(name).as_deref() != Some(name) {
            ui_state.set_save_msg("Load failed: invalid save name.", false);
            return;
        }
        let file = match physics::read_save(name) {
            Ok(f) => f,
            Err(e) => {
                ui_state.set_save_msg(format!("Load failed: {e}"), false);
                return;
            }
        };
        let mut b = shared.lock_for_render();
        let b = &mut *b;
        if let Err(e) = physics::restore_save(b, &file) {
            ui_state.set_save_msg(format!("Load failed: {e}"), false);
            return;
        }
        // The load worked, so the old timeline is over. Cleared under the bodies
        // lock, so the physics thread cannot record the old world after it.
        lock_ok(&shared.history).clear();
        self.replay = None;
        self.replay_cache = None;
        let a = &file.app;
        self.cam = a.cam;
        self.cam_z = a.cam_z;
        self.yaw = a.yaw;
        self.pitch = a.pitch.clamp(-view::PITCH_LIMIT, view::PITCH_LIMIT);
        self.zoom = a.zoom.clamp(ZOOM_MIN, ZOOM_MAX);
        self.speed = a.speed.clamp(0.01, 100.0);
        self.paused = a.paused;
        if let Some(seed) = physics::seed_from_base62(&a.seed) {
            self.seed = seed;
        }
        self.body_count = b.len();
        self.preset = if a.preset == "solar" {
            Preset::SolarSystem
        } else {
            Preset::Random
        };
        self.systems = a.systems.clamp(1, physics::MAX_SOLAR_SYSTEMS);
        self.last_loaded_preset = a.library_preset.clone();
        self.show_orbits = a.show_orbits;
        self.show_debris_orbits = a.show_debris_orbits;
        self.aging_mult = a.aging_mult.max(1.0);
        // The bodies carry the true values for these three.
        self.evolution_rate = b.evolution_rate;
        self.sn_kicks = b.sn_kicks;
        self.mass_transfer = b.mass_transfer;
        self.selected = a.selected.filter(|&id| b.index_of_id(id).is_some());
        self.follow = a.follow && self.selected.is_some();
        self.follow_smooth = None;
        // Do not fire "pause on event" for events that are already counted.
        self.last_pause_events = b
            .stats
            .merges
            .wrapping_add(b.stats.impacts)
            .wrapping_add(b.stats.tidal);
        self.orbit_buf_valid = false;
        shared.world_replaced();
        self.trail.clear();
        ui_state.reset_after_load(
            b.len(),
            self.seed,
            self.systems,
            self.last_loaded_preset.clone(),
        );
        ui_state.set_save_msg(format!("Loaded {} ({} bodies)", name, b.len()), true);
    }
}
