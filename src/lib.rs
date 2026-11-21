//! Library surface for `main.rs` and the criterion benchmarks.
//!
//! `main.rs` is a thin binary that drives the winit event loop; the actual
//! simulation, renderer and UI live in modules declared here so they can
//! be imported by `benches/`.

pub mod physics;
pub mod render;
pub mod theme;
pub mod ui;
pub mod view;

/// Simulation preset. Moved here from `main.rs` so `ui.rs` (in the lib) can
/// name it via `crate::Preset` without a circular dependency on the binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Random,
    SolarSystem,
}
