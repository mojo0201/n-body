# nbody-rust

A real-time N-body planetary-system simulator in Rust: random system generation,
real-system presets, stellar evolution, binaries, tides, collisions, dust and
a GPU renderer (wgpu + egui + winit).

```
cargo run --release --bin nbody-rust
```

Requires egui/egui-wgpu/egui-winit 0.37, wgpu 30, winit 0.30 (see `Cargo.toml`).

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
src/render.rs/.wgsl    instanced renderer; black-hole ray tracer; colours
src/ui.rs, theme.rs    egui panels, stats, info text
src/physics/
  mod.rs               Bodies (struct of arrays), constants, helpers
  integrator.rs        Yoshida 4th order / block leapfrog, step pipeline
  forces.rs, barnes_hut.rs   direct sums, tree, 1PN / 2.5PN
  evolution.rs         stellar evolution, remnants, mass transfer
  tides.rs             tides, spin, tidal locking, heating
  yarkovsky.rs         Yarkovsky drift of small bodies
  migration.rs         gas-disk migration and resonance capture
  collisions.rs, debris.rs, dust.rs, orbits.rs, energy.rs, ...
  spawn/               random systems: primary, binaries, planets, belts, moons
  presets.rs, save.rs  real systems (JSON) and save files
```

The render thread and physics thread share `Bodies` behind a mutex. `ui::draw`
runs under that lock and must stay fast and read-only. `render.rs` and
`render.wgsl` share a layout contract (instance structs, attribute offsets).

## Determinism

Spawning is seeded. Code that adds random draws to an existing path changes every
system generated after it, so new features use a private RNG seeded from the object
(see the terrestrial-moon block in `spawn/systems.rs`) or a hash of the body id
(Yarkovsky spin sign) instead of the main stream.

## Realism model

**Stars**
- Piecewise main-sequence mass–luminosity (0.23 M^2.3 below 0.43 M☉, M⁴ to 2 M☉,
  1.4 M^3.5 to 55 M☉, then linear) and mass–radius (M^0.8 below 1 M☉, M^0.57 above).
- Lifetime t = 10 Gyr · M / L, floored at 3 Myr.
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

**Tides and spin**
- Planets and moons have spin (day length lazily initialised: 10 h giants, 24 h
  others). Tides raised on a body by its parent despin it to corotation (locking),
  circularise orbits and push moons outward; tides on the parent act as before.
  Close binary stars circularise during the run.
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
to 1 and turns belt grinding off. The app starts in physical mode; the dock's
"accelerated tides and drag" checkbox switches the boosts on.

## Not yet modelled

Orbital inclinations (everything is coplanar).
