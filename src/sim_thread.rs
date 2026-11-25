//! Split out of `main.rs`.

use super::*;

// ---------------------------------------------------------------------------
// State shared between the render thread and the physics thread
// ---------------------------------------------------------------------------

/// Simulation seconds per wall second at a displayed speed of 1.00x. Below 1
/// so "normal" speed is slower than the raw rate; every speed control, the
/// slider and the quick buttons are relative to this.
pub(super) const SPEED_SCALE: f64 = 0.5;

pub(super) fn lock_ok<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub(super) struct Shared {
    pub(super) bodies: Mutex<Bodies>,
    pub(super) render_wants: AtomicBool,
    pub(super) quit: AtomicBool,
    pub(super) paused: AtomicBool,
    pub(super) speed_bits: AtomicU64,
    pub(super) clear_accum: AtomicBool,
    pub(super) behind: AtomicBool,
    /// True while a physics step is paused mid-way (velocities of a rung
    /// block are staggered then, so energy must not be sampled).
    pub(super) mid_step: AtomicBool,
    /// The physics thread panicked and is waiting for a new system.
    pub(super) failed: AtomicBool,
    /// The physics step paused mid-way, if any. Lives here, not in the pacer,
    /// so the render thread can finish it before it edits the bodies. Lock
    /// order is `bodies`, then `job`.
    pub(super) job: Mutex<Option<physics::AdvanceJob>>,
    pub(super) phys_ns: AtomicU64,
    pub(super) orbit_req: Mutex<Option<(u64, OrbitReq)>>,
    pub(super) orbit_out: Mutex<Option<(u64, Vec<OrbitVertex>)>>,
    /// Timeline snapshots, recorded by the physics thread. Lock order is
    /// always `bodies` then `history`.
    pub(super) history: Mutex<History>,
}

impl Shared {
    pub(super) fn new(bodies: Bodies, paused: bool, speed: f64) -> Self {
        Self {
            bodies: Mutex::new(bodies),
            render_wants: AtomicBool::new(false),
            quit: AtomicBool::new(false),
            paused: AtomicBool::new(paused),
            speed_bits: AtomicU64::new((speed * SPEED_SCALE).to_bits()),
            clear_accum: AtomicBool::new(false),
            behind: AtomicBool::new(false),
            mid_step: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            job: Mutex::new(None),
            phys_ns: AtomicU64::new(0),
            orbit_req: Mutex::new(None),
            orbit_out: Mutex::new(None),
            history: Mutex::new(History::new(history::DEFAULT_CAP_MB)),
        }
    }

    pub(super) fn lock_for_render(&self) -> RenderGuard<'_> {
        self.render_wants.store(true, Ordering::Release);
        RenderGuard {
            guard: lock_ok(&self.bodies),
            wants: &self.render_wants,
        }
    }

    /// Run the paused physics step to its end. Call with the bodies lock held,
    /// before anything that edits, deletes, saves or single-steps the bodies:
    /// a step in flight keeps indices, a tree and a momentum baseline that an
    /// edit would make wrong.
    /// The bodies were replaced (Generate, Load, a preset): tell the physics thread
    /// to drop any step in flight, and drop orbit lines built for the old bodies so
    /// no frame draws them. Call with the bodies lock held.
    pub(super) fn world_replaced(&self) {
        self.clear_accum.store(true, Ordering::Release);
        *lock_ok(&self.orbit_req) = None;
        *lock_ok(&self.orbit_out) = None;
    }

    pub(super) fn settle_step(&self, b: &mut Bodies) {
        let mut slot = lock_ok(&self.job);
        if let Some(mut job) = slot.take() {
            while !physics::advance_run(&mut job, b, || true) {}
        }
        self.mid_step.store(false, Ordering::Release);
    }

    pub(super) fn yield_to_render(&self) {
        while self.render_wants.load(Ordering::Acquire) && !self.quit.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_micros(100));
        }
    }
}

pub(super) struct RenderGuard<'a> {
    pub(super) guard: MutexGuard<'a, Bodies>,
    pub(super) wants: &'a AtomicBool,
}

impl Deref for RenderGuard<'_> {
    type Target = Bodies;
    fn deref(&self) -> &Bodies {
        &self.guard
    }
}

impl DerefMut for RenderGuard<'_> {
    fn deref_mut(&mut self) -> &mut Bodies {
        &mut self.guard
    }
}

impl Drop for RenderGuard<'_> {
    fn drop(&mut self) {
        self.wants.store(false, Ordering::Release);
    }
}

/// Longest real time the physics thread will "owe" before it gives up
/// catching up and drops the backlog (scaled by the simulation speed).
pub(super) const MAX_BACKLOG_REAL_S: f64 = 0.25;
/// Backlog below this many physics steps is never dropped.
pub(super) const MIN_BACKLOG_STEPS: f64 = 8.0;
/// How long the "behind" indicator stays lit after a backlog was dropped.
pub(super) const BEHIND_INDICATOR: Duration = Duration::from_secs(1);
/// Longest real time a single frame's worth of owed simulation can be.
pub(super) const MAX_FRAME_REAL_S: f64 = 0.1;
/// Idle sleep bounds while waiting for the next step to become due.
pub(super) const IDLE_SLEEP_MIN_S: f64 = 2.0e-4;
pub(super) const IDLE_SLEEP_MAX_S: f64 = 2.0e-3;

/// The physics thread's pacing state: how much simulated time is owed, the
/// step currently in flight (if any), and the "behind" indicator.
pub(super) struct PhysicsPacer {
    pub(super) last_tick: Instant,
    /// Simulated time owed to the world, in time units.
    pub(super) owed: f64,
    pub(super) behind_until: Option<Instant>,
}

impl PhysicsPacer {
    pub(super) fn new() -> Self {
        Self { last_tick: Instant::now(), owed: 0.0, behind_until: None }
    }

    /// Forget owed time and any step in flight (reset / load / spawn).
    pub(super) fn discard_pending_work(&mut self, sh: &Shared) {
        self.owed = 0.0;
        *lock_ok(&sh.job) = None;
        sh.mid_step.store(false, Ordering::Release);
    }

    /// Add the real time since the previous tick, scaled by `speed`.
    pub(super) fn accrue(&mut self, speed: f64, paused: bool) {
        let now = Instant::now();
        let real_dt = now.duration_since(self.last_tick).as_secs_f64();
        self.last_tick = now;
        if !paused {
            self.owed += real_dt.min(MAX_FRAME_REAL_S) * speed;
        }
    }

    pub(super) fn step_is_due(&self, sh: &Shared) -> bool {
        lock_ok(&sh.job).is_some() || self.owed >= physics::DT
    }

    /// Advance the simulation while time is owed, holding the bodies lock
    /// only until the render thread asks for it or the slice runs out.
    pub(super) fn run_slice(&mut self, sh: &Shared) {
        sh.yield_to_render();
        let mut b = lock_ok(&sh.bodies);
        // The render thread raises `clear_accum` while it holds this lock, when it
        // replaces the bodies. Checking it here, under the lock, makes sure a step
        // started on the old bodies never resumes on the new ones.
        if sh.clear_accum.swap(false, Ordering::AcqRel) {
            self.discard_pending_work(sh);
            return;
        }
        let mut job = lock_ok(&sh.job);
        let started = Instant::now();
        let must_pause = || {
            sh.render_wants.load(Ordering::Acquire) || started.elapsed() >= PHYS_SLICE
        };
        // A step can span several lock holds: it pauses after a sub-tick when
        // `must_pause` and resumes on the next slice.
        loop {
            if job.is_none() {
                if self.owed < physics::DT {
                    break;
                }
                self.owed -= physics::DT;
                *job = physics::advance_begin(&mut b, physics::DT);
                if job.is_none() {
                    continue;
                }
            }
            let finished = job
                .as_mut()
                .is_none_or(|j| physics::advance_run(j, &mut b, || !must_pause()));
            if finished {
                *job = None;
            }
            if must_pause() {
                break;
            }
        }
        lock_ok(&sh.history).maybe_record(&b);
        sh.mid_step.store(job.is_some(), Ordering::Release);
        drop(job);
        drop(b);
        sh.phys_ns
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// Drop a hopeless backlog and light the "behind" indicator for a while.
    pub(super) fn drop_backlog_if_hopeless(&mut self, speed: f64) {
        let limit = (MAX_BACKLOG_REAL_S * speed).max(MIN_BACKLOG_STEPS * physics::DT);
        if self.owed > limit {
            self.owed = 0.0;
            self.behind_until = Some(Instant::now() + BEHIND_INDICATOR);
        }
    }

    pub(super) fn is_behind(&self, paused: bool) -> bool {
        !paused && self.behind_until.is_some_and(|t| Instant::now() < t)
    }

    pub(super) fn idle_sleep(&self, speed: f64, paused: bool) {
        let seconds = if paused {
            IDLE_SLEEP_MAX_S
        } else {
            ((physics::DT - self.owed).max(0.0) / speed).clamp(IDLE_SLEEP_MIN_S, IDLE_SLEEP_MAX_S)
        };
        std::thread::sleep(Duration::from_secs_f64(seconds));
    }
}

/// Record the starting state so the timeline has a frame even while paused.
pub(super) fn seed_timeline(sh: &Shared) {
    if lock_ok(&sh.history).is_empty() {
        let b = lock_ok(&sh.bodies);
        lock_ok(&sh.history).maybe_record(&b);
    }
}

/// Build the orbit lines the render thread asked for. Returns whether it did.
pub(super) fn serve_orbit_request(sh: &Shared) -> bool {
    let Some((id, req)) = lock_ok(&sh.orbit_req).take() else {
        return false;
    };
    sh.yield_to_render();
    // Publish while still holding the bodies lock: if the world is replaced
    // meanwhile, the replacement clears the slot and this result is never stored.
    let b = lock_ok(&sh.bodies);
    let segments = physics::compute_orbit_segments(&b, req.zoom, req.planets, req.debris);
    *lock_ok(&sh.orbit_out) = Some((id, render::segments_to_vertices(&segments)));
    drop(b);
    true
}

/// Run `physics_loop`, and survive a panic in it. The error stays in the
/// terminal, the title bar says the physics stopped, and the thread resumes
/// as soon as the system is replaced (Generate, Load, a preset), because the
/// bodies are then known to be whole again.
pub(super) fn physics_supervisor(sh: Arc<Shared>) {
    while !sh.quit.load(Ordering::Relaxed) {
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| physics_loop(sh.clone())));
        if run.is_ok() {
            break;
        }
        eprintln!("physics thread stopped after a panic; generate or load a system to continue");
        *lock_ok(&sh.job) = None;
        sh.mid_step.store(false, Ordering::Release);
        sh.failed.store(true, Ordering::Release);
        // `clear_accum` is raised under the bodies lock when the world is replaced.
        while !sh.quit.load(Ordering::Relaxed) && !sh.clear_accum.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(50));
        }
        sh.failed.store(false, Ordering::Release);
    }
}

pub(super) fn physics_loop(sh: Arc<Shared>) {
    let mut pacer = PhysicsPacer::new();
    while !sh.quit.load(Ordering::Relaxed) {
        if sh.clear_accum.swap(false, Ordering::AcqRel) {
            pacer.discard_pending_work(&sh);
        }
        let paused = sh.paused.load(Ordering::Relaxed);
        let speed = f64::from_bits(sh.speed_bits.load(Ordering::Relaxed)).max(1e-6);
        pacer.accrue(speed, paused);

        seed_timeline(&sh);
        let mut worked = serve_orbit_request(&sh);
        if !paused && pacer.step_is_due(&sh) {
            pacer.run_slice(&sh);
            worked = true;
        }
        pacer.drop_backlog_if_hopeless(speed);
        sh.behind.store(pacer.is_behind(paused), Ordering::Relaxed);
        if !worked {
            pacer.idle_sleep(speed, paused);
        }
    }
}

