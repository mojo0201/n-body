//! Library surface for `main.rs` and the criterion benchmarks.
//!
//! `main.rs` is a thin binary that drives the winit event loop; the actual
//! simulation, renderer and UI live in modules declared here so they can
//! be imported by `benches/`.

// Deliberate, crate-wide: `!(x > 0.0)` is the NaN-safe form of "not a positive
// number" the physics relies on; the hot loops index several parallel arrays
// at once (struct-of-arrays), and the integrators take their state as arguments.
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::doc_overindented_list_items,
    clippy::doc_lazy_continuation
)]

pub mod history;
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
