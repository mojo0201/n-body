# nbody-rust

A real-time N-body planetary-system simulator in Rust: random system generation,
real-system presets, stellar evolution, binaries, tides, collisions, dust and
a GPU renderer (wgpu + egui + winit).

```
cargo run --release --bin nbody-rust
```

Requires egui/egui-wgpu/egui-winit 0.37, wgpu 30, winit 0.30 (see `Cargo.toml`).

The physics is exercised headlessly by `cargo test --lib` and gated by
`scripts/ci.sh`; see [Building and testing](#building-and-testing).

## Units

| Quantity | Unit |
|---|---|
| Length | 1 Gm (10⁶ km); 1 AU ≈ 149.6 Gm |
| Mass | 1 Earth mass |
| G | 7.4 (so one time unit ≈ 49.9 days) |
| Step | `DT = 1/480` time units |

`physics::time_unit_s()` gives seconds per time unit.

## Layout

```
src/main.rs            window, wgpu setup, render/physics thread split
src/lib.rs             library surface for the binaries and the tests
src/render.rs/.wgsl    instanced renderer; black-hole ray tracer; colours
src/sim_thread.rs      physics thread: owns the step loop, lock slices
src/sim_frame.rs       per-frame readout published to the UI
src/sim_actions.rs     UI actions → simulation changes
src/sim_saves.rs       loading, saving, timeline replay
src/history.rs         timeline snapshots and replay state
src/view.rs, trail.rs  camera; orbit and trail geometry
src/theme.rs, ui/      egui: dock, stats, plots, inspector, saves, systems, ...
src/bin/               headless.rs (throughput/energy bench), find_planets.rs
src/physics/
  mod.rs               Bodies (struct of arrays), constants, helpers
  integrator.rs        Yoshida 4th order / block leapfrog, step pipeline
  forces.rs, barnes_hut.rs   direct sums, octree, 1PN / 2.5PN
  evolution.rs         stellar evolution, remnants, mass transfer
  early.rs             sandboxed pass that resolves instabilities at spawn
  tides.rs             tides (both directions), spin, locking, heating
  yarkovsky.rs         Yarkovsky drift of small bodies
  migration.rs         gas-disk migration and resonance capture
  collisions.rs, debris.rs, dust.rs, orbits.rs, energy.rs, ...
  climate.rs, transit.rs, names.rs/naming.rs, orbit_lines.rs, edit.rs, rng.rs
  spawn/               mod.rs (primary, tilt, Kepler placement), systems.rs
                       (architecture, moons), belts.rs, binary.rs, cluster.rs
  presets.rs, save.rs  real systems (JSON in assets/presets) and save files
  tests/               headless unit tests, one module per area
examples/              ~44 ad-hoc diagnostics (check_moon, crash_scan, ...)
scripts/ci.sh, sweep.sh  the CI gate and the throughput sweep
```

The render thread and physics thread share `Bodies` behind a mutex. `ui::draw`
runs under that lock and must stay fast and read-only. `render.rs` and
`render.wgsl` share a layout contract (instance structs, attribute offsets).

## Determinism

Spawning is seeded. Code that adds random draws to an existing path changes every
system generated after it, so new features use a private RNG seeded from the object
(see the terrestrial-moon block in `spawn/systems.rs`) or a hash of the body id
(Yarkovsky spin sign) instead of the main stream. A draw that is no longer needed
is still *consumed* for the same reason: `spawn_trojan` throws away its node
longitude but keeps taking it, so adding and removing a feature does not renumber
every later draw.

## Realism model

**Gravity and integration**
- Exact O(n²) pair sums up to `BH_THRESHOLD` = 600 bodies (serial below
  `PAR_DIRECT_MIN` = 150, parallel above); a Barnes–Hut octree above that with
  θ = 0.5 (`BH_THETA`, ≈1 % force error). Softening is 100 km (`SOFT2`).
- Adaptive block stepping: a body on rung r advances `DT / 2^r`, and the step is
  Yoshida 4th order with a per-step velocity limiter, so close encounters stay
  stable without global sub-stepping.
- 1PN and 2.5PN post-Newtonian terms: periapsis precession, and the
  gravitational-wave inspiral of very tight pairs.

**Stars**
- Piecewise main-sequence mass–luminosity (0.23 M^2.3 below 0.43 M☉, M⁴ to 2 M☉,
  1.4 M^3.5 to 55 M☉, then linear) and mass–radius (M^0.8 below 1 M☉, M^0.57 above).
- Lifetime t = 10 Gyr · M / L, floored at 3 Myr for stars. Brown dwarfs,
  remnants and non-positive masses have no main sequence and return infinity,
  so `age / t_life` stays finite for degenerate input.
- Radius and luminosity grow along the main sequence (`EVO_MS_RADIUS_GROWTH`
  0.4, `EVO_MS_LUM_GROWTH` 0.9: the Sun runs 0.7 → 1.35 L☉), normalised so a star
  46 % through its main sequence (`EVO_MS_REF_FRAC`) has exactly the mass–radius
  and mass–luminosity value — a newborn Sun is *dimmer* than 1 L☉.
- Giant phases, supernova kicks, white dwarf / brown dwarf cooling, remnants.

**Climate**
- `equilibrium_temp(s)` assumes Bond albedo 0.3 (255 K at 1 S⊕);
  `equilibrium_temp_albedo(s, a)` takes any albedo.
- Habitable zone uses Kopparapu et al. (2013) flux limits that depend on the
  star's effective temperature (`hz_flux_limits`).

**Planets**
- Brown-dwarf primaries (15 % of systems) follow the substellar mass function dN/dm ∝ m^-0.3.
- Giant mass is capped at about 1 % of the host mass.
- Rocky density rises with mass (∝ M^0.19). Above ~2 M⊕ planets may keep a gas
  envelope (sub-Neptunes), stripped by photoevaporation at high flux (radius valley).
- Hot Jupiters are inflated up to ~25 % above ~150 S⊕.
- About 25 % of rocky planets of 0.3–50 M⊕ with room in their Hill sphere get an
  impact-formed moon of 0.3–3 % of the planet's mass.

**Inclinations**
- Every orbit is 3D. Each planetary system draws its own flatness (Rayleigh
  scale log-uniform 0.7–4.2°, so Kepler-flat ≈1° up to dynamically hot ≈4°);
  small-body populations use their own dispersions — asteroid belt 7°, Hilda 8°,
  Trojans 12°, cold Kuiper 2°, hot/resonant/scattered Kuiper 10–15°,
  short-period comets 12° — and long-period comets (a > 200 AU) are isotropic.
- `set_inclination_scale(s)` is a global multiplier on every draw: 1 is the
  default (realistic), 0 makes every orbit coplanar, which is what
  `examples/baseline.rs` pins. There is no UI control for it yet; the dock only
  *reports* the inclination of the selected body.
- The octree, osculating elements, moon orbits, orbit lines and the timeline are
  all 3D; nothing is projected onto the ecliptic.

**Tides and spin**
- Planets and moons have spin (day length lazily initialised: 10 h giants, 24 h
  others). Every bound pair runs a tide in *both* directions: the host's bulge
  takes angular momentum from the orbit into the host's spin (inward migration
  while the host rotates slower than the orbit, outward while it spins faster),
  and the planet's bulge despins it toward corotation — tidal locking — and
  circularises the orbit. Tides act inside `TIDAL_MAX_RADII` (50 host radii).
  Close binary stars circularise during the run.
- Planet-raised tides run at `PLANET_BOOST_REL` (10⁻² of the stellar rate):
  planets are far more dissipative than stars.
- Spin transfer stops at corotation so a light body cannot overshoot.
- `tidal_heating_w` (Peale 1979) is shown in the planet panel with the rotation state.

**Binaries**
- Roche-lobe mass transfer. If the donor is too heavy relative to its partner
  (q > 3 main-sequence donor, q > 1.4 giant donor) the transfer is unstable and the
  pair merges in a common envelope.

**Migration and resonances**
- Young stars (under ~20 Myr) have a gas disk that fades with a 2 Myr e-folding time,
  so migration happens by default in young systems. The "force gas disk" checkbox
  adds one around every star regardless of age. The disk applies torques to every
  planet: orbits shrink (Type I, ∝ 1/mass up to 100 M⊕, slower Type II above), eccentricities
  damp 50× faster, and the torque switches off at an inner edge of max(3 R_host, 0.05 AU).
  Planets stack at the edge and converging pairs are captured into resonances.
  Timescales are shortened 300× (`MIGRATION_SPEEDUP`) so it is visible.
- Spawn already places ~30 % of compact chains just wide of 3:2 or 2:1.

**Small bodies**
- Yarkovsky transverse acceleration for belt, Trojan, Hilda and debris bodies under
  100 km, with a hidden ± spin sense.

**Visuals**
- Planet colours follow equilibrium temperature: giants by Sudarsky class, rocky
  worlds from frozen to lava.
- Stars are limb-darkened (I = 1 − u(1 − μ), u from colour: red stars darken more than blue ones).
- Active comets (inside ~4 AU) show a blue ion tail pointing away from the host and a shorter, curved,
  yellowish dust tail (`render::build_comet_tails`).
- Black holes have a thin white outline on their shadow edge.
- Black-hole accretion disks are drawn only while the hole is fed (a donor
  overflowing its Roche lobe, or a very luminous star close by); otherwise only the
  shadow shows.

## Boosts and physical mode

Real tidal, Poynting–Robertson and Yarkovsky rates are far too slow to see, so they
are multiplied by boosts (`TIDAL_BOOST` 10⁶, `PR_DRAG_BOOST` 100, `YARKOVSKY_BOOST`
10⁴; planet-tide boost is 10⁻² of the stellar one). Physical mode sets all of them
to 1 and turns belt grinding off (`belt_grind_per_orbit()` returns 0). The app
starts in physical mode, so out of the box **nothing** dissipates within a session:
a hot Jupiter inspiral takes ~Gyr. The dock's "accelerated tides and drag"
checkbox switches the boosts on, which is what you want for a demo.

## Caps and limits

| Limit | Value |
|---|---|
| Bodies in one simulation (`MAX_BODIES`) | 256 000 |
| Grains per dust field (`DUST_MAX_GRAINS`) | 256 000 |
| Grains a comet field emits (`DUST_COMET_FIELD_CAP`) | 100 000 |
| Dust fields (`MAX_DUST_FIELDS`) | 256 |
| Dust points drawn per frame (`MAX_DUST_POINTS`) | 2 000 000 |
| Planets with orbit lines (`ORBIT_MAX_BODIES`) | 4 000 (debris: 256 rings) |
| Trail points / vertices | 4 000 / 100 000 |

`Bodies::push` refuses to grow past `MAX_BODIES`, and the UI warns when a spawn
request would be clamped to fit.

## Building and testing

```
cargo run --release --bin nbody-rust       # the app
cargo test --lib                           # ~630 headless physics tests
cargo clippy --all-targets -- -D warnings  # what CI enforces
scripts/ci.sh                              # the whole gate
```

- The tests live in `src/physics/tests/`, one module per area: forces and the
  octree, the integrator, conservation laws, tides, dust and drag, collisions,
  debris, orbital elements, Kepler placement, spawn statistics, presets, stellar
  evolution, utilities, and a fuzz smoke test.
- The physical/accelerated switch is process-global. Every test that flips it —
  and every test that depends on a rate — takes a `SimMode` guard, which
  serialises them and restores the previous mode on drop (including on panic).
- `scripts/ci.sh` runs clippy, the unit tests, a release build of `headless`, the
  headless CLI contract (`--n 0` → exit 2, `--min-ratio` fail/pass, `--json`,
  `--quiet --json` prints exactly one line, a `--solar` smoke run), builds every
  example, and checks `scripts/sweep.sh`'s TSV shape.
- `src/bin/headless.rs` is the throughput/energy harness (`--json`, `--quiet`);
  `scripts/sweep.sh` wraps it over n × seeds.
- `examples/` holds ~44 throwaway diagnostics (`check_moon`, `crash_scan`,
  `stability_sweep`, `check_save_roundtrip`, ...). They are not shipped, but CI
  compiles them because they are the first thing to rot.
- `.cargo/config.toml` builds with `-C target-cpu=native`, so a binary is tied to
  the machine that built it.

## Not yet modelled

- **Spin axes.** `spin` is a scalar angular momentum about +z: no obliquity,
  precession or YORP. Tides assume the spin axis is aligned with the orbit
  normal, which is exact for prograde, low-inclination orbits and off by cos i
  otherwise.
- **Live instabilities.** The spawned architecture is stable by design;
  encounters are resolved in the sandboxed `early_evolution` pass at spawn time
  rather than by letting a running system scatter.
- **Hydrodynamics.** Gas-disk migration is a 1D torque prescription (Type I/II),
  with no disk self-gravity or gap-opening feedback, and fragmentation is a
  debris model with a Roche limit, not a hydrocode.
- **Yarkovsky obliquity.** The spin sense is a hidden ± derived from the body id,
  so there is no seasonal Yarkovsky and no YORP spin-up.
- **Tidal heating feedback.** `tidal_heating_w` is displayed; it does not feed
  back into surface temperature, radius or evolution.
- **Trojans in tilted systems.** The cloud is placed in the reference plane, so
  in a strongly inclined system it sits 60° from the giant in projection rather
  than in the giant's own orbital plane.
