//! Physics tests.
//!
//! Loaded as `physics::tests` via `#[cfg(test)] mod tests;` in
//! `physics/mod.rs`; each file below uses `use super::super::*;` and sees every private item (`Tree`,
//! `collision_pass`, `find_collisions`, `pair_hit`, `interact_frac`,
//! `impact_offset`, `dt_limit`, `rung_for`, `disrupt`, `coalesce_debris`,
//! `belt_spec_for`, `spawn_belts`, `belt_mass_earth`, `kirkwood_gaps`,
//! `radius_from_density`, the IC samplers, and so on) without widening the
//! crate's public API.
//!
//! Test naming convention: a letter prefix groups tests by area; a
//! descriptive snake_case name follows. `cargo test -- --list` shows the
//! grouping at a glance.
//!
//!   a  conservation invariants (L, p, mass) across merges
//!   b  hit-and-run
//!   c  debris / moon-forming impacts
//!   d  grazing vs head-on
//!   e  tidal disruption timing
//!   f  radius laws
//!   g  initial conditions (statistical)
//!   h  rubble coalescence
//!   i  radius / survivor / primary / Chandrasekhar
//!   j  compaction invariants
//!   k  binaries and circumbinary stability
//!   l  Barnes-Hut vs direct
//!   m  collision detection (hash vs brute force, immunity)
//!   n  integrator (rungs, dt limits, energy conservation)
//!   o  orbit_info, circularize, host_distance
//!   p  escape culling
//!   q  energy diagnostics
//!   r  kind transitions via merge
//!   s  user edits
//!   t  multi-system spawn
//!   u  RNG / samplers
//!   v  geometry helpers (Hill, Roche, impact, interact_frac, dt_limit)
//!   w  naming helpers (class/threshold/format)
//!   x  star luminosity, temperature, lifetime, habitability
//!   y  samplers (Rayleigh, power-law, normal, IMF median)
//!   z  small utilities (time unit, index_of_id, log ring buffer, cell key)
//!   aa spawn variants
//!   ab fast (hot) collision path matches exhaustive scan
//!   ac rubble bounce (momentum, spin, immunity)
//!   ad fragmentation (frag_masses, density, rigidity, disrupt)
//!   ae integrator internals (rungs, flashes, Yoshida beats leapfrog)
//!   af silent merge (no log, no flash, conservation)
//!   ag orbit / anomaly helpers
//!   ah osculating-ellipse geometry
//!   ai accuracy (tight BH error, tight energy drift, element preservation)
//!   aj realism (Solar System reproduces real periods)
//!   ak fuzz smoke (invariants across many seeds; tree-rebuild stress)
//!   al stellar evolution (aging, remnants, adiabatic expansion, belts,
//!      bulk density, random spawn)
//!
//! Belts and bulk density are tested under `al_*` (they were introduced by
//! the evolution-then-belts prompt pair) and under `mean_anomaly_*`,
//! `ices_are_lighter_than_rocks`, `belt_*`, `kirkwood_*`, `trojans_*`,
//! `random_systems_never_exceed_body_count`, `default_density_*`.
//!
//! Files: shared helpers, then one file per area (`cargo test --lib
//! physics::tests::collisions` runs one area).

mod collisions;
mod conservation;
mod debris;
mod dust;
mod evolution;
mod forces;
mod fuzz;
mod kepler;
mod orbits;
mod presets;
mod shared;
mod spawn;
mod util;
