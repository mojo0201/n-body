// wgsl shader for the N-body renderer.
//
// Layout contract with render.rs (field order and padding must match):
//   CameraUniform  <-> struct Camera
//   BodyInstance   <-> struct Instance
//   OrbitVertex    <-> struct OrbitIn
//   FlashInstance  <-> struct FlashIn
//   DustVertex     <-> struct DustIn
//   dust centres   <-> struct DustFields (128 x vec4)
//
// Bind groups: group 0 is one camera uniform. Bodies, flashes and dust use
// the "relative" camera (pos = 0, positions pre-subtracted on the CPU);
// orbits and trails use the "world" camera. Dust also reads group 1, the
// per-field centres (camera-relative, rewritten every frame).
 
struct Camera {
    pos: vec2<f32>,
    zoom: f32,
    _pad0: f32,
    viewport: vec2<f32>,
    _pad1: vec2<f32>,
};
 
@group(0) @binding(0) var<uniform> camera: Camera;
 
// -------- Bodies (instanced quads) --------
 
struct Instance {
    @location(0) pos: vec2<f32>,
    @location(1) radius: f32,
    @location(2) color: vec3<f32>,
};
 
struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>,
};
 
// draw(0..4, 0..n), triangle strip: vi 0..3 -> (-1,-1) (+1,-1) (-1,+1) (+1,+1).
@vertex
fn vs_main(@builtin(vertex_index) vi: u32, inst: Instance) -> VertexOut {
    let fx = f32(vi & 1u) * 2.0 - 1.0;
    let fy = f32((vi >> 1u) & 1u) * 2.0 - 1.0;
    let corner = vec2<f32>(fx, fy);
    let world = inst.pos + corner * inst.radius;
    let screen = (world - camera.pos) * camera.zoom;
    let ndc = screen / (camera.viewport * 0.5);
    var out: VertexOut;
    out.clip_pos = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    out.local = corner;
    out.color = inst.color;
    return out;
}
 
// Anti-aliased disc. Colour is multiplied by alpha and ALPHA_BLENDING
// multiplies again (alpha^2 edge): intentional, see render.rs.
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let d = length(in.local);
    if d > 1.0 {
        discard;
    }
    let alpha = smoothstep(1.0, 0.85, d);
    return vec4<f32>(in.color * alpha, alpha);
}
 
// -------- Orbit lines (LineList) --------
 
struct OrbitIn {
    @location(0) pos: vec2<f32>,
    @location(1) color: vec4<f32>,
};
 
struct OrbitOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};
 
@vertex
fn vs_orbit(v: OrbitIn) -> OrbitOut {
    let screen = (v.pos - camera.pos) * camera.zoom;
    let ndc = screen / (camera.viewport * 0.5);
    var out: OrbitOut;
    out.clip_pos = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    out.color = v.color;
    return out;
}
 
@fragment
fn fs_orbit(in: OrbitOut) -> @location(0) vec4<f32> {
    return in.color;
}
 
// -------- Flash rings (instanced quads) --------
 
struct FlashIn {
    @location(0) pos: vec2<f32>,
    @location(1) radius: f32,
    @location(2) progress: f32,
};
 
struct FlashOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) progress: f32,
};
 
@vertex
fn vs_flash(@builtin(vertex_index) vi: u32, inst: FlashIn) -> FlashOut {
    let fx = f32(vi & 1u) * 2.0 - 1.0;
    let fy = f32((vi >> 1u) & 1u) * 2.0 - 1.0;
    let corner = vec2<f32>(fx, fy);
    let world = inst.pos + corner * inst.radius;
    let screen = (world - camera.pos) * camera.zoom;
    let ndc = screen / (camera.viewport * 0.5);
    var out: FlashOut;
    out.clip_pos = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    out.local = corner;
    out.progress = inst.progress;
    return out;
}
 
// Expanding, fading ring: starts at 10 % of the radius, grows to 95 %.
@fragment
fn fs_flash(in: FlashOut) -> @location(0) vec4<f32> {
    let d = length(in.local);
    if d > 1.0 {
        discard;
    }
    let ring_pos = 0.1 + 0.85 * in.progress;
    let ring_w = 0.18;
    let ring_alpha = smoothstep(ring_w, 0.0, abs(d - ring_pos));
    let alpha = ring_alpha * (1.0 - in.progress) * 0.9;
    if alpha < 0.01 {
        discard;
    }
    return vec4<f32>(1.0, 0.95, 0.7, alpha);
}
 
// -------- Dust grains (instanced soft points, additive) --------
//
// One instance per tracer grain. `pos` is relative to the field's primary at
// upload time; `field` indexes the per-frame, camera-relative centre of that
// primary. The quad is ~1 px wide at any zoom, so a grain is a soft dot and
// density shows up as overlap. The pipeline blends additively (src_alpha, one).
 
struct DustFields {
    c: array<vec4<f32>, 128>,
};
 
@group(1) @binding(0) var<uniform> dust_fields: DustFields;
 
struct DustIn {
    @location(0) pos: vec2<f32>,
    @location(1) weight: f32,
    @location(2) field: u32,
};
 
struct DustOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) intensity: f32,
};
 
@vertex
fn vs_dust(@builtin(vertex_index) vi: u32, inst: DustIn) -> DustOut {
    let fx = f32(vi & 1u) * 2.0 - 1.0;
    let fy = f32((vi >> 1u) & 1u) * 2.0 - 1.0;
    let corner = vec2<f32>(fx, fy);
 
    let centre = dust_fields.c[inst.field].xy;
    // Half-width of 0.9 px, expressed in world units.
    let r = 0.9 / camera.zoom;
    let rel = centre + inst.pos + corner * r;
    let ndc = (rel * camera.zoom) / (camera.viewport * 0.5);
 
    var out: DustOut;
    out.clip_pos = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    out.local = corner;
    // Tracer mass is ~1e-8..1e-3 Earth masses: map log10 onto 0.15..1.
    out.intensity = clamp((log(inst.weight) / 2.302585 + 9.0) / 6.0, 0.15, 1.0);
    return out;
}
 
@fragment
fn fs_dust(in: DustOut) -> @location(0) vec4<f32> {
    let d = length(in.local);
    if d > 1.0 {
        discard;
    }
    let a = smoothstep(1.0, 0.0, d) * in.intensity * 0.10;
    // Warm grey; with (src_alpha, one) blending the result is colour * a.
    return vec4<f32>(0.62, 0.58, 0.54, a);
}