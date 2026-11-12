# nbody-rust — 10,000-body N-Body simulation

Native N-body simulation in Rust with wgpu rendering, rayon parallel
Barnes-Hut gravity, and a minimal egui UI.

## Build & run

    cargo run --release

The first build takes 3–5 minutes (wgpu + egui + dependencies). Subsequent
builds are ~10s. Debug builds work but are ~30× slower.

## Controls

    Space          pause / resume
    R              reset (new random seed)
    W / A / S / D  pan camera
    Arrow keys     pan camera
    +/-            zoom
    Mouse drag     pan camera
    Mouse wheel    zoom
    Slider         body count (100–50,000)

## Performance notes

At 10,000 bodies the CPU physics loop (Barnes-Hut + rayon) is the bottleneck,
at roughly ~5ms/step on a modern 8-core. Rendering 10k instanced quads is
~0.3ms. The app stabilizes at ~60fps on the reference hardware (Ryzen 7 5800X).

Above 20,000 bodies, expect frame drops. Two paths forward:

1. Move the BH tree traversal into a **compute shader**. At 10k bodies this
   gives roughly 10× on physics — 50k bodies at 60fps becomes feasible.
2. Use **SIMD** (`wide::f64x4`) in the BH inner loop. ~2× on the same CPU.

## What this v1 includes vs. the JS version

| Feature | JS | Rust v1 |
|---|---|---|
| 10k bodies | ✗ (slow) | ✓ |
| Barnes-Hut gravity | ✓ (≥200 bodies) | ✓ (always) |
| Yoshida 4th-order integrator | ✓ | ✓ |
| Merge on collision | ✓ | ✓ |
| Multi-system spawn | ✓ | ✗ |
| Moon formation | ✓ (recently fixed) | ✗ |
| Tidal breakup | ✓ | ✗ |
| Prediction / trace paths | ✓ | ✗ |
| Body list table | ✓ | ✗ |
| Minimap | ✓ | ✗ |
| URL sharing | ✓ | ✗ |
| Screenshots | ✓ | ✗ |

The physics constants (`G`, `SOFT2`, `DT`, Yoshida weights, mass/radius
relationships) match the JS exactly. Adding moons back is a matter of porting
`moonCollision` + `specificEnergyAbout` + `hillRadiusOfBody` — see the JS
version for reference.

## Files

- `src/physics.rs` — SoA bodies, Barnes-Hut quadtree, Yoshida integrator, spawn
- `src/render.rs` — instanced quad renderer, per-body color
- `src/render.wgsl` — vertex + fragment shaders
- `src/main.rs` — winit application loop, wgpu surface, egui UI

## Next upgrades

1. **Compute shader physics** — replace `compute_accel()` with a WGSL compute
   pipeline. ~200 more lines. Gets you to 50k+ bodies.
2. **Port moons/tidal/prediction from JS** — the physics is the same, just
   needs the extra helpers. ~500 lines.
3. **Body list + minimap** — egui tables + a second render pass.
4. **URL sharing** — `arboard` for clipboard, or a small file dialog.