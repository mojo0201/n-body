//! Timeline history: a memory-capped ring buffer of full simulation
//! snapshots, recorded by the physics thread and replayed (view-only) by the
//! renderer.
//!
//! A snapshot is a complete `Bodies` clone behind an `Arc`, so scrubbing never
//! touches the live simulation and a frame can be handed to the renderer
//! without holding any lock. Frames are spaced by simulation time, but never
//! closer than `MIN_WALL_GAP` of wall-clock time, so a fast-forwarding run
//! does not spend its time cloning state.

use crate::physics::Bodies;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Default memory cap for stored snapshots.
pub const DEFAULT_CAP_MB: usize = 512;
/// Minimum simulation time between frames (code time units).
const MIN_SIM_GAP: f64 = 0.25;
/// Minimum wall-clock time between recorded frames.
const MIN_WALL_GAP: Duration = Duration::from_millis(200);

pub struct Frame {
    pub id: u64,
    pub time: f64,
    bytes: usize,
    pub state: Arc<Bodies>,
}

pub struct History {
    frames: VecDeque<Frame>,
    next_id: u64,
    bytes: usize,
    cap_bytes: usize,
    last_time: f64,
    last_wall: Instant,
}

/// Rough heap size of a snapshot, for the memory cap.
fn approx_bytes(b: &Bodies) -> usize {
    let grains: usize = b.dust_fields.iter().map(|f| f.grains.len()).sum();
    b.len() * 520 + grains * 72 + b.events.len() * 120 + b.flashes.len() * 64 + 4096
}

impl History {
    pub fn new(cap_mb: usize) -> Self {
        Self {
            frames: VecDeque::new(),
            next_id: 0,
            bytes: 0,
            cap_bytes: cap_mb * 1024 * 1024,
            last_time: f64::NEG_INFINITY,
            last_wall: Instant::now(),
        }
    }

    pub fn clear(&mut self) {
        self.frames.clear();
        self.bytes = 0;
        self.last_time = f64::NEG_INFINITY;
    }

    /// Record `b` if enough simulation and wall time has passed since the
    /// last frame. A clock that runs backwards (new system, loaded save)
    /// starts a fresh history.
    pub fn maybe_record(&mut self, b: &Bodies) {
        if b.time < self.last_time - 1e-9 {
            self.clear();
        }
        let first = self.frames.is_empty();
        if !first
            && (b.time - self.last_time < MIN_SIM_GAP || self.last_wall.elapsed() < MIN_WALL_GAP)
        {
            return;
        }
        let bytes = approx_bytes(b);
        self.frames.push_back(Frame {
            id: self.next_id,
            time: b.time,
            bytes,
            state: Arc::new(b.clone()),
        });
        self.next_id += 1;
        self.bytes += bytes;
        self.last_time = b.time;
        self.last_wall = Instant::now();
        while self.bytes > self.cap_bytes && self.frames.len() > 2 {
            if let Some(f) = self.frames.pop_front() {
                self.bytes -= f.bytes;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn megabytes(&self) -> f32 {
        self.bytes as f32 / (1024.0 * 1024.0)
    }

    /// Oldest and newest recorded simulation times.
    pub fn span(&self) -> Option<(f64, f64)> {
        Some((self.frames.front()?.time, self.frames.back()?.time))
    }

    /// Index of the frame closest in time to `t`.
    fn nearest_index(&self, t: f64) -> Option<usize> {
        if self.frames.is_empty() {
            return None;
        }
        let i = self.frames.partition_point(|f| f.time < t);
        Some(if i == 0 {
            0
        } else if i == self.frames.len() {
            i - 1
        } else if t - self.frames[i - 1].time <= self.frames[i].time - t {
            i - 1
        } else {
            i
        })
    }

    pub fn nearest(&self, t: f64) -> Option<&Frame> {
        self.nearest_index(t).map(|i| &self.frames[i])
    }

    /// Time of the frame one step before (`dir < 0`) or after (`dir > 0`)
    /// the frame nearest `t`.
    pub fn neighbour_time(&self, t: f64, dir: i32) -> Option<f64> {
        let i = self.nearest_index(t)? as i64 + dir.signum() as i64;
        self.frames.get(usize::try_from(i).ok()?).map(|f| f.time)
    }
}
