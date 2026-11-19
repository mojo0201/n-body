// wgsl shader for the N-body renderer.
//
// Layout contract with render.rs (field order and padding must match):
//   CameraUniform  <-> struct Camera
//   BodyInstance   <-> struct Instance
//   OrbitVertex    <-> struct OrbitIn
//   FlashInstance  <-> struct FlashIn
//   DustVertex     <-> struct DustIn
//   GlowInstance   <-> struct GlowIn
//   BlackHoleInstance <-> struct BlackHoleIn
//   dust centres   <-> struct DustFields (256 x vec4)
//
// Bind groups: group 0 is one camera uniform. Bodies, flashes, dust, glow,
// black holes and trails use the "relative" camera (pos = 0, positions
// pre-subtracted on the CPU in f64); orbit lines use the "world" camera. Dust also reads group 1, the
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
    // 1 = lit by a star (planets), 0 = self-luminous or unlit.
    @location(3) lit: f32,
    // Direction to the lighting star, as an angle in world axes (radians).
    @location(4) light_angle: f32,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>,
    // xy = unit vector toward the star (world axes), z = light strength 0..1.
    @location(2) light: vec3<f32>,
    // Limb-darkening strength: 1 for a self-luminous disc big enough to show
    // it (stars), 0 otherwise.
    @location(3) limb: f32,
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
    // A shaded sphere needs a few pixels to read. Below that size the body
    // stays a flat dot, so a distant planet does not lose half its light.
    let px_r = inst.radius * camera.zoom;
    let strength = inst.lit * smoothstep(2.5, 7.0, px_r);
    out.light = vec3<f32>(cos(inst.light_angle), sin(inst.light_angle), strength);
    out.limb = (1.0 - inst.lit) * smoothstep(3.0, 8.0, px_r);
    return out;
}

// Anti-aliased disc. Colour is multiplied by alpha and ALPHA_BLENDING
// multiplies again (alpha^2 edge): intentional, see render.rs.
//
// The sim is top-down, so the star always lights a planet from the side
// (phase angle 90 deg): exactly half of the disc is lit. The surface normal
// of the sphere is (local.x, local.y, sqrt(1 - d^2)). The light vector lies
// in the screen plane, so N.L reduces to dot(local, toward_star).
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let d = length(in.local);
    if d > 1.0 {
        discard;
    }
    let alpha = smoothstep(1.0, 0.85, d);
    let n_dot_l = max(dot(in.local, in.light.xy), 0.0);
    // Lambert term plus a faint floor for scattered light on the night side.
    let lit_shade = 0.04 + 1.15 * pow(n_dot_l, 0.8);
    let shade = mix(1.0, lit_shade, in.light.z);
    // Limb darkening of self-luminous discs: I = 1 - u (1 - mu), mu = cos of
    // the angle from the disc centre. Cool red stars darken more (u ~ 0.7)
    // than hot blue ones (u ~ 0.35); the blue/red ratio of the colour stands
    // in for temperature.
    let mu = sqrt(max(1.0 - d * d, 0.0));
    let u = clamp(0.85 - 0.5 * (in.color.b / max(in.color.r, 0.05)), 0.35, 0.75);
    let limb = mix(1.0, 1.0 - u * (1.0 - mu), in.limb);
    return vec4<f32>(in.color * shade * limb * alpha, alpha);
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
    c: array<vec4<f32>, 256>,
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
    out.intensity = clamp((log(inst.weight) / 2.302585 + 16.0) / 6.0, 0.15, 1.0);
    return out;
}
 
@fragment
fn fs_dust(in: DustOut) -> @location(0) vec4<f32> {
    let d = length(in.local);
    if d > 1.0 {
        discard;
    }
    let a = smoothstep(1.0, 0.0, d) * in.intensity * 0.15;
    // Warm grey; with (src_alpha, one) blending the result is colour * a.
    return vec4<f32>(0.62, 0.58, 0.54, a);
}

// ── Star glow ─────────────────────────────────────────────────────────
//
// Additive soft quad centred on a luminous body. The CPU side has already
// subtracted the camera, so the vertex shader only needs zoom + viewport
// (matching how vs_main places BodyInstance quads). `uv` runs −1..1
// across the quad, so |uv| = 1 is the halo's outer edge.

struct GlowIn {
    @location(0) pos: vec2<f32>,
    @location(1) radius: f32,
    @location(2) intensity: f32,
    @location(3) color: vec3<f32>,
};

struct GlowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) intensity: f32,
};

@vertex
fn vs_glow(@builtin(vertex_index) vi: u32, inst: GlowIn) -> GlowOut {
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    let c = corners[vi];
    let world = inst.pos + c * inst.radius;
    let screen = (world - camera.pos) * camera.zoom;
    let ndc = screen / (camera.viewport * 0.5);
    var out: GlowOut;
    out.clip = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    out.uv = c;
    out.color = inst.color;
    out.intensity = inst.intensity;
    return out;
}

@fragment
fn fs_glow(in: GlowOut) -> @location(0) vec4<f32> {
    let r = length(in.uv);
    if r > 1.0 {
        discard;
    }
    // Wide power-law halo. The 2.5 exponent gives a long, soft skirt
    // that reads as scattered light rather than a hard-edged disc.
    let halo = pow(max(0.0, 1.0 - r), 2.5);
    // Narrow exponential core so the center of the quad still brightens
    // when the halo alone would be too dim near the star's edge.
    let core = exp(-r * r * 6.0);
    let a = (halo * 0.75 + core * 0.25) * in.intensity;
    // Additive blending (SrcAlpha / One): result = color * a + dst.
    return vec4<f32>(in.color, a);
}

// ── Black Holes ───────────────────────────────────────────────────────
//
// Each fragment traces one light ray backward from the camera through a
// Schwarzschild spacetime and finds where it meets a thin accretion disk.
//
// Units: lengths are in Schwarzschild radii (Rs = 2GM/c^2), so the event
// horizon is r = 1. With u = 1/r, a light ray in its orbital plane obeys
//     d2u/dphi2 = -u + 1.5 u^2.
// A ray that arrives at impact parameter b starts with u = 0, du/dphi = 1/b.
// Rays with b < 3*sqrt(3)/2 = 2.598 Rs fall in: that is the black shadow.
//
// The disk is a Novikov-Thorne style thin disk from the innermost stable
// orbit (3 Rs) outward. Its temperature follows T^4 ~ r^-3 (1 - sqrt(r_in/r)).
// Brightness and colour use the Doppler shift of the orbiting gas and the
// gravitational redshift. Tilt, outer radius and peak temperature are
// artistic choices: see the constants.
//
// The picture shows: the shadow, the near side of the disk in front of it,
// the far side bent over the top and under the bottom, and the thin photon
// ring that comes from rays which circle the hole.

// Half-size of the quad, in Rs. Must be larger than BH_R_OUT.
const BH_QUAD: f32 = 10.0;
const BH_R_ISCO: f32 = 3.0;
const BH_R_OUT: f32 = 8.0;
const BH_B_CRIT: f32 = 2.598076;
// Disk tilt from face-on. 78 deg: nearly edge-on. A disk in the sim plane
// would be face-on (0 deg) for this top-down camera.
const BH_COS_I: f32 = 0.2079117;
const BH_SIN_I: f32 = 0.9781476;
// Colour temperature at the brightest part of the disk, in K.
const BH_T_PEAK: f32 = 5600.0;
const BH_STEPS: i32 = 170;
const BH_DPHI: f32 = 0.075;
const PI: f32 = 3.14159265;

struct BlackHoleIn {
    @location(0) pos: vec2<f32>,
    @location(1) radius: f32,   // event-horizon radius (Rs), world units
    @location(2) accretion: f32, // 0..1: how strongly the disk is fed; 0 hides it
};

struct BlackHoleOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local: vec2<f32>,  // position in Rs, centred on the hole
    @location(1) @interpolate(flat) accretion: f32,
};

@vertex
fn vs_blackhole(@builtin(vertex_index) vi: u32, inst: BlackHoleIn) -> BlackHoleOut {
    let fx = f32(vi & 1u) * 2.0 - 1.0;
    let fy = f32((vi >> 1u) & 1u) * 2.0 - 1.0;
    let corner = vec2<f32>(fx, fy);

    let world = inst.pos + corner * inst.radius * BH_QUAD;
    let screen = (world - camera.pos) * camera.zoom;
    let ndc = screen / (camera.viewport * 0.5);

    var out: BlackHoleOut;
    out.clip_pos = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    out.local = corner * BH_QUAD;
    out.accretion = inst.accretion;
    return out;
}

fn bh_accel(u: f32) -> f32 {
    return -u + 1.5 * u * u;
}

// sRGB-ish colour of a blackbody, max channel 1. Same fit as
// `blackbody_to_rgb` in render.rs.
fn bb_rgb(t_kelvin: f32) -> vec3<f32> {
    let t = clamp(t_kelvin / 100.0, 10.0, 400.0);
    var r: f32;
    var g: f32;
    var b: f32;
    if t <= 66.0 {
        r = 255.0;
        g = 99.4708025861 * log(t) - 161.1195681661;
        if t <= 19.0 {
            b = 0.0;
        } else {
            b = 138.5177312231 * log(t - 10.0) - 305.0447927307;
        }
    } else {
        r = 329.698727446 * pow(t - 60.0, -0.1332047592);
        g = 288.1221695283 * pow(t - 60.0, -0.0755148492);
        b = 255.0;
    }
    return clamp(vec3<f32>(r, g, b) / 255.0, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Light from the disk at radius `r` (Rs) as seen by the camera.
// Returns (premultiplied colour, opacity).
//   pos_dir : unit vector from the hole to the emission point (3D, camera on +z)
//   k       : unit direction of the photon, in the local static frame
fn bh_disk_emission(r: f32, pos_dir: vec3<f32>, k: vec3<f32>) -> vec4<f32> {
    // Orbit velocity as measured by a static observer: sqrt(M / (r - 2M)),
    // with M = Rs / 2.
    let beta = sqrt(0.5 / max(r - 1.0, 0.05));
    let gamma = 1.0 / sqrt(max(1.0 - beta * beta, 1e-4));
    // Gas orbits counter-clockwise about the disk normal n = (0, sin i, cos i).
    // For the point (x_d, y_d) in the disk, v_hat = (-y_d, x_d cos i, -x_d sin i) / r.
    let xd = pos_dir.x;
    let yd = pos_dir.y * BH_COS_I - pos_dir.z * BH_SIN_I;
    let vhat = vec3<f32>(-yd, xd * BH_COS_I, -xd * BH_SIN_I);
    let doppler = 1.0 / (gamma * (1.0 - beta * dot(vhat, k)));
    let g = sqrt(max(1.0 - 1.0 / r, 0.0)) * doppler;

    // Disk flux, normalised to 1 at its peak (r = 49/36 r_in).
    let x = r / BH_R_ISCO;
    let flux = pow(x, -3.0) * (1.0 - inverseSqrt(x)) / 0.05665;
    // Taper the outer edge so the disk fades out instead of ending abruptly.
    let taper = smoothstep(BH_R_OUT, BH_R_OUT - 1.5, r);
    let t_emit = BH_T_PEAK * pow(max(flux, 0.0), 0.25);

    // Bolometric intensity goes as g^4.
    let intensity = pow(g, 4.0) * flux * taper;
    let emitted = bb_rgb(g * t_emit) * (1.0 - exp(-3.0 * intensity));
    let alpha = 1.0 - exp(-12.0 * intensity);
    return vec4<f32>(emitted, alpha);
}

// Trace one ray. `p` is the position in the image plane in Rs (x right,
// y up). Returns the light and opacity collected from the disk.
// Premultiplied colour in .rgb, opacity in .a.
fn bh_trace(p: vec2<f32>) -> vec4<f32> {
    let b = length(p);
    if b < 1e-3 {
        return vec4<f32>(0.0);
    }
    let ph = p / b;

    // The ray meets the disk plane at phi_c + k*pi along its orbital plane.
    let phi_c = atan2(BH_COS_I, -ph.y * BH_SIN_I);

    var u = 0.0;
    var du = 1.0 / b;
    var phi = 0.0;
    var next_k = 0;
    var acc = vec3<f32>(0.0);   // premultiplied light collected so far
    var acc_a = 0.0;            // opacity collected so far

    for (var s = 0; s < BH_STEPS; s++) {
        // One RK4 step in phi.
        let h = BH_DPHI;
        let k1u = du;
        let k1d = bh_accel(u);
        let k2u = du + 0.5 * h * k1d;
        let k2d = bh_accel(u + 0.5 * h * k1u);
        let k3u = du + 0.5 * h * k2d;
        let k3d = bh_accel(u + 0.5 * h * k2u);
        let k4u = du + h * k3d;
        let k4d = bh_accel(u + h * k3u);
        let u_new = u + h / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
        let du_new = du + h / 6.0 * (k1d + 2.0 * k2d + 2.0 * k3d + k4d);
        let phi_new = phi + h;

        // Did the ray cross the disk plane during this step?
        let phi_k = phi_c + PI * f32(next_k);
        if phi_k <= phi_new {
            next_k += 1;
            let t = clamp((phi_k - phi) / h, 0.0, 1.0);
            let u_x = mix(u, u_new, t);
            let du_x = mix(du, du_new, t);
            if u_x > 1e-3 {
                let r = 1.0 / u_x;
                if r >= BH_R_ISCO && r <= BH_R_OUT {
                    let sphi = sin(phi_k);
                    let cphi = cos(phi_k);
                    // Radial and tangential unit vectors of the ray plane.
                    let rhat = vec3<f32>(sphi * ph.x, sphi * ph.y, cphi);
                    let that = vec3<f32>(-cphi * ph.x, -cphi * ph.y, sphi);
                    // Angle between the photon and the outward radial
                    // direction, in the frame of a static observer.
                    let sin_psi = clamp(b * sqrt(1.0 - 1.0 / r) / r, 0.0, 1.0);
                    let cos_psi = sqrt(1.0 - sin_psi * sin_psi);
                    // The photon travels toward decreasing phi. It moves
                    // outward when u is falling in that direction.
                    let radial_sign = select(-1.0, 1.0, du_x > 0.0);
                    let k = radial_sign * cos_psi * rhat + sin_psi * that;
                    let e = bh_disk_emission(r, rhat, k);
                    acc += (1.0 - acc_a) * e.rgb;
                    acc_a += (1.0 - acc_a) * e.a;
                    if acc_a > 0.995 {
                        break;
                    }
                }
            }
        }

        u = u_new;
        du = du_new;
        phi = phi_new;
        if u >= 1.0 || u <= 0.0 {
            break; // fell in, or escaped
        }
        // Moving outward and already past the disk edge: no more crossings.
        if du < 0.0 && u < 1.0 / BH_R_OUT {
            break;
        }
    }
    return vec4<f32>(acc, acc_a);
}

@fragment
fn fs_blackhole(in: BlackHoleOut) -> @location(0) vec4<f32> {
    // Pixel size in Rs. Computed first, in uniform control flow.
    let px = max(fwidth(in.local.x), 1e-4);

    // Image plane: x to the right, y up. Screen y runs down, so flip it.
    let p = vec2<f32>(in.local.x, -in.local.y);
    let b = length(p);

    var c = bh_trace(p);

    // Near the photon ring the disk images are thinner than a pixel and
    // would sparkle. Average 4 rays in a rotated grid there.
    if b > BH_B_CRIT - 2.0 * px && b < BH_B_CRIT + 0.5 {
        c = 0.25 * (bh_trace(p + px * vec2<f32>(0.125, 0.375))
                  + bh_trace(p + px * vec2<f32>(0.375, -0.125))
                  + bh_trace(p + px * vec2<f32>(-0.125, -0.375))
                  + bh_trace(p + px * vec2<f32>(-0.375, 0.125)));
    }

    // A hole with nothing falling in has no luminous disk: fade it by the
    // accretion rate (the shadow and lensing stay).
    c = c * clamp(in.accretion, 0.0, 1.0);

    // Shadow: whatever the disk does not hide is black inside b_crit
    // and clear outside it.
    let shadow = 1.0 - smoothstep(BH_B_CRIT - px, BH_B_CRIT + px, b);
    let a = c.a + (1.0 - c.a) * shadow;

    // White outline on the shadow edge, about 1.5 px wide at any zoom, drawn
    // over everything else so a hole stays visible against bright backgrounds.
    let ring = 1.0 - smoothstep(0.6 * px, 1.4 * px, abs(b - BH_B_CRIT));
    let a2 = a + (1.0 - a) * ring;
    if a2 < 1e-4 {
        discard;
    }
    let rgb = c.rgb * (1.0 - ring) + vec3<f32>(1.0) * ring;
    return vec4<f32>(rgb / a2, a2);
}
