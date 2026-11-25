//! Methods of `SimState` (frame), split out of `main.rs`.

use super::*;

impl SimState {
    /// Everything a redraw does while it holds the bodies lock: replay, UI,
    /// actions, camera, orbit lines and the render instances.
    pub(super) fn update_locked(
        &mut self,
        ui_state: &mut UiState,
        shared: &Arc<Shared>,
        gfx: &mut GfxState,
        vp: [f32; 2],
        frame_dt: f64,
        timing_log: bool,
    ) -> FrameData {
        let mut actions = Actions::default();
        let raw_input = gfx.egui_state.take_egui_input(&gfx.window);
        // Time spent waiting for the physics thread to release the lock, then
        // time spent holding it (NBODY_TIMING only).
        let t_wait0 = Instant::now();
        let mut b = shared.lock_for_render();
        let t_held0 = Instant::now();
        // Replay: show a recorded snapshot in place of the live state. It is
        // swapped back before this lock is released.
        let replaying = self.enter_replay(shared, &mut b, frame_dt);
        self.t_wait_accum += t_wait0.elapsed().as_secs_f64() * 1e6;

        let new_title = self.update_fps_title(&b, shared.failed.load(Ordering::Acquire));

        let selected_idx = self.selected.and_then(|id| b.index_of_id(id));
        self.update_follow(&b, selected_idx, frame_dt);
        if self.selected.is_some() && selected_idx.is_none() {
            self.selected = None;
            self.follow = false;
        }

        if replaying.is_none() && !shared.mid_step.load(Ordering::Acquire) {
            ui_state.diag.maybe_sample(&b);
        }

        let view_info = self.make_view_info(shared, vp, gfx.egui_ctx.pixels_per_point(), b.len());
        let full_output = gfx.egui_ctx.run_ui(raw_input, |ui| {
            actions = ui::draw(ui, ui_state, &view_info, &b);
        });
        if let Some(t) = actions.timeline.take() {
            self.handle_timeline(t, shared);
        }
        // A replay is view-only: edits, spawns, resets and loads are dropped;
        // camera, selection and display options stay.
        if replaying.is_some() {
            actions = actions.view_only();
        }
        if actions.changes_bodies() {
            shared.settle_step(&mut b);
        }
        self.apply_actions(actions, ui_state, vp, &mut b, shared);
        let trail_snapshot = self.trail.snapshot(&b, self.follow_smooth);

        self.pause_on_new_events(&b, replaying.is_some());

        shared.paused.store(self.paused, Ordering::Relaxed);
        shared
            .speed_bits
            .store((self.speed * SPEED_SCALE).to_bits(), Ordering::Relaxed);

        let orbit_out = self.orbit_lines_for_frame(&b, shared, replaying, vp);
        let comet_tail_verts = self.build_render_instances(&b, gfx, vp);
        let flashes_active = !b.flashes.is_empty();
        let n_bodies = b.len();
        if timing_log {
            self.last_comets = b.origin.iter().filter(|&&o| o == BodyOrigin::Comet).count();
            self.last_comet_verts = comet_tail_verts.len();
        }
        self.t_held_accum += t_held0.elapsed().as_secs_f64() * 1e6;
        if replaying.is_some() {
            self.leave_replay(&mut b);
        }
        FrameData {
            full_output,
            flashes_active,
            n_bodies,
            new_title,
            trail_snapshot,
            comet_tail_verts,
            orbit_out,
            replaying,
        }
    }

    /// Fill the renderer's instance buffers; returns the comet tail vertices.
    pub(super) fn build_render_instances(&mut self, b: &Bodies, gfx: &mut GfxState, vp: [f32; 2]) -> Vec<OrbitVertex> {
        let t_inst0 = Instant::now();
        let override_body = self
            .follow_smooth
            .and_then(|(id, pos)| b.index_of_id(id).map(|idx| (idx, pos)));
        let scene_view = self.view();
        gfx.renderer.build_instances(b, &scene_view, vp, override_body);
        self.t_inst_accum += t_inst0.elapsed().as_secs_f64() * 1e6;
        let t_comet0 = Instant::now();
        let verts = nbody_rust::render::build_comet_tails(b, &scene_view);
        self.t_comet_accum += t_comet0.elapsed().as_secs_f64() * 1e6;
        verts
    }

    /// Update the trails (live run only) and upload the scene to the GPU.
    pub(super) fn upload_frame(&mut self, gfx: &mut GfxState, data: &FrameData, vp: [f32; 2]) {
        let cam_view = self.view();
        // Trails belong to the live run: hidden while replaying, and only
        // rebuilt when enabled.
        let trail_verts = if data.replaying.is_none() && self.trail.cfg.enabled {
            self.trail.update(&data.trail_snapshot);
            Some(self.trail.build_vertices(cam_view.cam))
        } else {
            None
        };
        gfx.upload_scene(
            data.orbit_out.as_ref().map(|(_, v)| v.as_slice()),
            &cam_view,
            vp,
            &data.comet_tail_verts,
            trail_verts.as_deref(),
        );
    }

    /// Respawn the system for the current preset.
    ///
    /// `count` is the *per-system* body count and is only used by `Random`;
    /// the Solar preset takes the body count from the spawn (13 planets +
    /// ~200 belt bodies, not the old hard-coded 9). `seed` is used by both:
    /// `Random` uses it as the RNG seed, `Solar` as the planet-phase seed.
    /// Count a frame and, twice a second, refresh fps / steps-per-second.
    /// Returns the new window title when the numbers were refreshed.
    pub(super) fn update_fps_title(&mut self, b: &Bodies, physics_failed: bool) -> Option<String> {
        let mut new_title = None;
        self.fps_frames += 1;
        let win = self.fps_timer.elapsed().as_secs_f32();
        if win >= 0.5 {
            self.fps = self.fps_frames as f32 / win;
            let steps = b.stats.steps;
            self.steps_per_sec =
                (steps.saturating_sub(self.steps_at_fps)) as f32 / win;
            self.steps_at_fps = steps;
            self.fps_frames = 0;
            self.fps_timer = Instant::now();
            let stopped = if physics_failed {
                " — PHYSICS STOPPED: generate or load a system"
            } else {
                ""
            };
            new_title = Some(format!(
                "N-Body Rust — {} — {:.0} fps — {:.2}×{}",
                primary_label(b),
                self.fps,
                self.speed,
                stopped
            ));
        }

        new_title
    }

    /// Move the camera to the followed body (smoothed), or drop follow mode
    /// when that body no longer exists.
    pub(super) fn update_follow(&mut self, b: &Bodies, selected_idx: Option<usize>, frame_dt: f64) {
        if self.follow {
            match selected_idx {
                Some(i) => {
                    let id = b.id[i];
                    const TAU: f64 = 0.06;
                    let alpha = 1.0 - (-frame_dt / TAU).exp();
                    let prev = match self.follow_smooth {
                        Some((prev_id, pos)) if prev_id == id => pos,
                        _ => [b.x[i], b.y[i], b.z[i]],
                    };
                    let new_pos = [
                        prev[0] + (b.x[i] - prev[0]) * alpha,
                        prev[1] + (b.y[i] - prev[1]) * alpha,
                        prev[2] + (b.z[i] - prev[2]) * alpha,
                    ];
                    self.follow_smooth = Some((id, new_pos));
                    self.cam = [new_pos[0], new_pos[1]];
                    self.cam_z = new_pos[2];
                }
                None => {
                    self.follow = false;
                    self.follow_smooth = None;
                }
            }
        } else {
            self.follow_smooth = None;
        }
    }

    /// Everything the UI needs to know about the current state of the view.
    pub(super) fn make_view_info(&self, shared: &Shared, vp: [f32; 2], ppp: f32, total_bodies: usize) -> ViewInfo {
        let effective_systems = if self.preset == Preset::Random
            && self.last_loaded_preset.is_none()
        {
            self.systems
        } else {
            1
        };
        ViewInfo {
            paused: self.paused,
            speed: self.speed,
            fps: self.fps,
            steps_per_sec: self.steps_per_sec,
            zoom: self.zoom,
            cam: [self.cam[0], self.cam[1], self.cam_z],
            yaw: self.yaw,
            pitch: self.pitch,
            viewport_px: vp,
            ppp,
            seed: self.seed,
            body_count: self.body_count,
            systems: effective_systems,
            total_bodies,
            show_orbits: self.show_orbits,
            show_debris_orbits: self.show_debris_orbits,
            follow: self.follow,
            selected: self.selected,
            follow_smooth: self.follow_smooth,
            preset: self.preset,
            vsync: self.vsync,
            behind: shared.behind.load(Ordering::Relaxed),
            pause_on_event: self.pause_on_event,
            trails_enabled: self.trail.cfg.enabled,
            trail_age: self.trail.cfg.max_age,
            trail_bodies: self.trail.cfg.max_bodies,
            trail_non_rubble: self.trail.cfg.non_rubble_only,
            sn_kicks: self.sn_kicks,
            mass_transfer: self.mass_transfer,
            aging_mult: self.aging_mult,
            timeline: {
                let hist = lock_ok(&shared.history);
                hist.span().map(|(t_min, t_max)| TimelineInfo {
                    t_min,
                    t_max,
                    t_now: self.replay.as_ref().map(|r| r.time),
                    playing: self.replay.as_ref().is_some_and(|r| r.playing),
                    frames: hist.len(),
                    mb: hist.megabytes(),
                })
            },
        }
    }

    /// Pause when the merger / impact / tidal count grows. A drop means a new
    /// system replaced the old one: no pause then. Replays never pause.
    pub(super) fn pause_on_new_events(&mut self, b: &Bodies, replaying: bool) {
        // Pause when the merger / impact / tidal count grows. A drop
        // means a new system replaced the old one: no pause then.
        let pause_events = b
            .stats
            .merges
            .wrapping_add(b.stats.impacts)
            .wrapping_add(b.stats.tidal);
        if !replaying
            && self.pause_on_event
            && !self.paused
            && pause_events > self.last_pause_events
        {
            self.paused = true;
        }
        if !replaying {
            self.last_pause_events = pause_events;
        }
    }

    /// Orbit lines for this frame. Live: take what the physics thread built
    /// and ask it for a refresh when the view or the system changed. Replay:
    /// build them here from the shown snapshot.
    pub(super) fn orbit_lines_for_frame(
        &mut self,
        b: &Bodies,
        shared: &Shared,
        replaying: Option<u64>,
        vp: [f32; 2],
    ) -> Option<(u64, Vec<OrbitVertex>)> {
        let mut orbit_out_taken = lock_ok(&shared.orbit_out).take();
        if let Some(fid) = replaying {
            // Orbit lines for the recorded frame, built here from
            // the snapshot; the physics thread's live lines are dropped.
            orbit_out_taken = None;
            let key = (fid, self.zoom);
            if self.replay_orbit != key || !self.orbit_buf_valid {
                let segs = physics::compute_orbit_segments(
                    b,
                    self.zoom as f64,
                    self.show_orbits,
                    self.show_debris_orbits,
                );
                orbit_out_taken = Some((0, render::segments_to_vertices(&segs)));
                self.replay_orbit = key;
                self.orbit_buf_valid = true;
            }
        }
        let events_total = b
            .stats
            .merges
            .wrapping_add(b.stats.impacts)
            .wrapping_add(b.stats.tidal)
            .wrapping_add(b.stats.escaped);
        let hz_now = physics::hz_rings();
        let hz_changed = hz_now != self.last_hz;
        self.last_hz = hz_now;
        let orbits_visible = self.show_orbits || self.show_debris_orbits || hz_now;
        let zoom_ratio = self.zoom / self.last_orbit_zoom.max(1e-6);
        let zoom_changed = (zoom_ratio - 1.0).abs() > 0.3;
        let span_x = 0.35 * vp[0] as f64 / self.zoom as f64;
        let span_y = 0.35 * vp[1] as f64 / self.zoom as f64;
        let cam_moved = (self.cam[0] - self.last_orbit_cam[0]).abs() > span_x
            || (self.cam[1] - self.last_orbit_cam[1]).abs() > span_y;
        let events_changed = events_total != self.last_orbit_events;
        let stale = !self.paused
            && self.last_orbit_update.elapsed().as_millis() > ORBIT_REFRESH_MS;
        if replaying.is_none()
            && (!self.orbit_buf_valid
            || hz_changed
            || (orbits_visible
                && (zoom_changed || cam_moved || events_changed || stale)))
        {
            self.orbit_posted += 1;
            *lock_ok(&shared.orbit_req) = Some((
                self.orbit_posted,
                OrbitReq {
                    zoom: self.zoom as f64,
                    planets: self.show_orbits,
                    debris: self.show_debris_orbits,
                    cam: [self.cam[0], self.cam[1], self.cam_z],
                    viewport: vp,
                },
            ));
            self.last_orbit_update = Instant::now();
            self.last_orbit_zoom = self.zoom;
            self.last_orbit_cam = self.cam;
            self.last_orbit_events = events_total;
            self.orbit_buf_valid = true;
        }
        orbit_out_taken
    }

    /// Accumulate per-frame timings and, once a second, print the
    /// `NBODY_TIMING` summary line.
    pub(super) fn log_frame_timing(&mut self, shared: &Shared, t_frame: Instant, n_bodies: usize) {
        self.t_frame_accum += t_frame.elapsed().as_secs_f64() * 1e6;
        self.t_window_frames += 1;
        let secs = self.t_window_start.elapsed().as_secs_f64();
        if secs >= 1.0 {
            let frames = self.t_window_frames as f64;
            let phys_busy = shared.phys_ns.swap(0, Ordering::Relaxed) as f64
                / (secs * 1e9)
                * 100.0;
            let ms = |us: f64| us / frames / 1000.0;
            let dust_busy = physics::DUST_STAT_NS.swap(0, Ordering::Relaxed) as f64
                / (secs * 1e9)
                * 100.0;
            let dust_grains = physics::DUST_STAT_GRAINS.load(Ordering::Relaxed);
            println!(
                "[timing] n={} comets={} (tail verts {}) fps={:.0} frame={:.2}ms \
                 lock_wait={:.2}ms lock_held={:.2}ms (instances={:.2} comet_tails={:.2}) \
                 physics_busy={:.0}% (dust={:.0}%, {} grains){}",
                n_bodies,
                self.last_comets,
                self.last_comet_verts,
                frames / secs,
                ms(self.t_frame_accum),
                ms(self.t_wait_accum),
                ms(self.t_held_accum),
                ms(self.t_inst_accum),
                ms(self.t_comet_accum),
                phys_busy,
                dust_busy,
                dust_grains,
                if shared.behind.load(Ordering::Relaxed) {
                    " (behind)"
                } else {
                    ""
                }
            );
            self.t_window_start = Instant::now();
            self.t_window_frames = 0;
            self.t_frame_accum = 0.0;
            self.t_wait_accum = 0.0;
            self.t_held_accum = 0.0;
            self.t_inst_accum = 0.0;
            self.t_comet_accum = 0.0;
        }
    }
}
