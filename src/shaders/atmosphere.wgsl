// Constants
const PI: f32 = 3.14159265359;
const MAX: f32 = 10000.0;
const R_INNER: f32 = 1.0;
const R: f32 = R_INNER + 0.8;
const NUM_OUT_SCATTER: i32 = 8;
const NUM_IN_SCATTER: i32 = 80;
const WORKGROUP_SIZE: u32 = 16u;

// Input bindings
@group(0) @binding(0)
var texture_2d_instance: texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler_instance: sampler;

// Camera uniform
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

// Output texture
@group(2) @binding(0)
var output_texture: texture_storage_2d<rgba32float, write>;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_position: vec3f,
    time: f32,
    resolution: vec2f,
    _padding: vec2f,
};

// Ray intersects sphere
// e = -b +/- sqrt(b^2 - c)
fn ray_vs_sphere(p: vec3f, dir: vec3f, r: f32) -> vec2f {
    let b = dot(p, dir);
    let c = dot(p, p) - r * r;
    let d = b * b - c;
    if (d < 0.0) {
        return vec2f(MAX, -MAX);
    }
    let sq_d = sqrt(d);
    return vec2f(-b - sq_d, -b + sq_d);
}

// Mie scattering phase function
// g: (-0.75, -0.999)
fn phase_mie(g: f32, c: f32, cc: f32) -> f32 {
    let gg = g * g;
    let a = (1.0 - gg) * (1.0 + cc);
    var b = 1.0 + gg - 2.0 * g * c;
    b *= sqrt(b);
    b *= 2.0 + gg;
    return (3.0 / 8.0 / PI) * a / b;
}

// Rayleigh scattering phase function
// g: 0
fn phase_ray(cc: f32) -> f32 {
    return (3.0 / 16.0 / PI) * (1.0 + cc);
}

// Atmospheric density at point
fn density(p: vec3f, ph: f32) -> f32 {
    return exp(-max(length(p) - R_INNER, 0.0) / ph);
}

// Optical depth calculation
fn optic(p: vec3f, q: vec3f, ph: f32) -> f32 {
    let s = (q - p) / f32(NUM_OUT_SCATTER);
    var v = p + s * 0.5;
    var sum = 0.0;
    
    for (var i = 0; i < NUM_OUT_SCATTER; i++) {
        sum += density(v, ph);
        v += s;
    }
    
    sum *= length(s);
    return sum;
}

// In-scattering calculation
fn in_scatter(o: vec3f, dir: vec3f, e: vec2f, l: vec3f) -> vec3f {
    let ph_ray = 0.05;
    let ph_mie = 0.02;
    let k_ray = vec3f(3.8, 13.5, 33.1);
    let k_mie = vec3f(21.0);
    let k_mie_ex = 1.1;
    
    var sum_ray = vec3f(0.0);
    var sum_mie = vec3f(0.0);
    var n_ray0 = 0.0;
    var n_mie0 = 0.0;
    
    let len = (e.y - e.x) / f32(NUM_IN_SCATTER);
    let s = dir * len;
    var v = o + dir * (e.x + len * 0.5);
    
    for (var i = 0; i < NUM_IN_SCATTER; i++) {
        v += s;
        let d_ray = density(v, ph_ray) * len;
        let d_mie = density(v, ph_mie) * len;
        
        n_ray0 += d_ray;
        n_mie0 += d_mie;
        
        let f = ray_vs_sphere(v, l, R);
        let u = v + l * f.y;
        
        let n_ray1 = optic(v, u, ph_ray);
        let n_mie1 = optic(v, u, ph_mie);
        
        let att = exp(-(n_ray0 + n_ray1) * k_ray - (n_mie0 + n_mie1) * k_mie * k_mie_ex);
        
        sum_ray += d_ray * att;
        sum_mie += d_mie * att;
    }
    
    let c = dot(dir, -l);
    let cc = c * c;
    
    let scatter = sum_ray * k_ray * phase_ray(cc) + sum_mie * k_mie * phase_mie(-0.78, c, cc);
    return scatter;
}

// Compute shader entry point
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Get the dimensions of the output texture
    let dimensions = textureDimensions(output_texture);
    
    // Check if we're within bounds
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }
    
    // Calculate normalized device coordinates
    let uv = vec2f(
        (f32(global_id.x) + 0.5) / f32(dimensions.x),
        (f32(global_id.y) + 0.5) / f32(dimensions.y)
    );
    
    // Calculate ray direction from camera to the current fragment
    let ro = camera.camera_position;
    let ndc = vec4f(uv * 2.0 - 1.0, 1.0, 1.0);
    let world_pos = camera.inv_view_proj * ndc;
    let rd = normalize(world_pos.xyz / world_pos.w - ro);
    
    // Sun light direction (can be parametrized with time)
    let light_dir = normalize(vec3f(0.0, 0.0, 1.0));
    
    // Check if ray intersects atmosphere
    var e = ray_vs_sphere(ro, rd, R);
    
    var color = vec4f(0.0, 0.0, 0.0, 1.0);
    
    if (e.x <= e.y) {
        // Ray intersects atmosphere
        
        // Check if ray intersects planet
        let f = ray_vs_sphere(ro, rd, R_INNER);
        e.y = min(e.y, f.x);
        
        // Calculate atmospheric scattering
        let I = in_scatter(ro, rd, e, light_dir);
        color = vec4f(I * 10.0, 1.0);
    }
    
    // Write the result to the output texture
    textureStore(output_texture, global_id.xy, color);
}