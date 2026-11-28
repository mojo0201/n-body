#!/usr/bin/env bash
# Runs the tests that `cargo test --lib` skips via `#[ignore]`.
#
# These are ignored by default because each takes a minute or more on its own
# and would dominate the default gate's wall time. Run this before a release,
# or when touching the spawner, the early-evolution integrator, or the tidal
# and moon code:
#
#   physics::tests::spawn      g_ic_primary_diversity                     ~115 s
#                              g_ic_hot_jupiter_rate                      ~148 s
#                              hierarchical_stars_respect_stability_limit ~209 s
#                              spawn_triples_and_quadruples_exist         ~250 s
#   physics::tests::presets    moon_stays_bound_to_earth                 ~137 s
#                              moon_parent_stable                        ~155 s
#   physics::tests::evolution  smoke_evolution_runs_without_panic         ~55 s
#
# The four spawn tests are slow because each draw runs the full early-evolution
# integrator: 1 500-4 000 draws at roughly 60-80 ms each. The dev profile is
# already opt-level 2, so a release build does not make them cheap.
#
# Total ~18 min for all seven.
#
# Deliberately NOT part of scripts/ci.sh: the default gate stays fast enough to
# run on every change. Exits non-zero if any ignored test fails.

set -euo pipefail
cd "$(dirname "$0")/.."

cargo test --lib --quiet -- --ignored
