// voxel.wgsl

const PI = 3.14159265359;
const betaR = vec3f(5.5e-6, 13.0e-6, 22.4e-6);
const betaM = vec3f(21e-6);
const hR = 7994.0;
const hM = 1200.0;
const earth_radius = 6360e3;
const atmosphere_radius = 6420e3;
const sun_power = 20.0;
const g = 0.76;
const k = 1.55 * g - 0.55 * (g * g * g);
const num_samples = 16;
const num_samples_light = 8;

struct Ray {
    origin: vec3f,
    direction: vec3f,
};

struct Sphere {
    origin: vec3f,
    radius: f32,
    material: i32,
};

struct IsectResult {
    hit: bool,
    t0: f32,
    t1: f32,
};

const atmosphere = Sphere(vec3f(0.0, 0.0, 0.0), atmosphere_radius, 0);

fn isect_sphere(ray: Ray, sphere: Sphere) -> IsectResult {
    let rc = sphere.origin - ray.origin;
    let radius2 = sphere.radius * sphere.radius;
    let tca = dot(rc, ray.direction);
    let d2 = dot(rc, rc) - tca * tca;
    if d2 > radius2 {
        return IsectResult(false, 0.0, 0.0);
    }
    let thc = sqrt(radius2 - d2);
    let t0 = tca - thc;
    let t1 = tca + thc;
    return IsectResult(true, t0, t1);
}

fn rayleigh_phase_func(mu: f32) -> f32 {
    return 3.0 * (1.0 + mu * mu) / (16.0 * PI);
}

fn henyey_greenstein_phase_func(mu: f32) -> f32 {
    return (1.0 - g * g) / (4.0 * PI * pow(1.0 + g * g - 2.0 * g * mu, 1.5));
}

fn get_sun_light(ray: Ray, optical_depthR: ptr<function, f32>, optical_depthM: ptr<function, f32>) -> bool {
    let isect = isect_sphere(ray, atmosphere);
    if !isect.hit {
        return false;
    }
    let t1 = isect.t1;
    let march_step = t1 / f32(num_samples_light);
    var march_pos = 0.0;
    for (var i = 0; i < num_samples_light; i = i + 1) {
        let s = ray.origin + ray.direction * (march_pos + 0.5 * march_step);
        let height = length(s) - earth_radius;
        if height < 0.0 {
            return false;
        }
        *optical_depthR += exp(-height / hR) * march_step;
        *optical_depthM += exp(-height / hM) * march_step;
        march_pos += march_step;
    }
    return true;
}

fn get_incident_light(ray: Ray, sun_dir: vec3f) -> vec3f {
    let isect = isect_sphere(ray, atmosphere);
    if !isect.hit {
        return vec3f(0.0);
    }
    let t1 = isect.t1;
    let march_step = t1 / f32(num_samples);
    let mu = dot(ray.direction, sun_dir);
    let phaseR = rayleigh_phase_func(mu);
    let phaseM = henyey_greenstein_phase_func(mu);
    var optical_depthR = 0.0;
    var optical_depthM = 0.0;
    var sumR = vec3f(0.0);
    var sumM = vec3f(0.0);
    var march_pos = 0.0;
    for (var i = 0; i < num_samples; i = i + 1) {
        let s = ray.origin + ray.direction * (march_pos + 0.5 * march_step);
        let height = length(s) - earth_radius;
        let hr = exp(-height / hR) * march_step;
        let hm = exp(-height / hM) * march_step;
        optical_depthR += hr;
        optical_depthM += hm;
        let light_ray = Ray(s, sun_dir);
        var optical_depth_lightR = 0.0;
        var optical_depth_lightM = 0.0;
        let overground = get_sun_light(light_ray, &optical_depth_lightR, &optical_depth_lightM);
        if overground {
            let tau = betaR * (optical_depthR + optical_depth_lightR) + betaM * 1.1 * (optical_depthM + optical_depth_lightM);
            let attenuation = exp(-tau);
            sumR += hr * attenuation;
            sumM += hm * attenuation;
        }
        march_pos += march_step;
    }
    return sun_power * (sumR * phaseR * betaR + sumM * phaseM * betaM);
}

struct Settings {
    max: f32,
    r_inner: f32,
    r: f32,
    max_height: f32,
    max_water_height: f32,
    water_height: f32,
    tunnel_radius: f32,
    surface_factor: f32,
    camera_speed: f32,
    camera_time_offset: f32,
    voxel_level: i32,
    voxel_size: f32,
    steps: i32,
    max_dist: f32,
    min_dist: f32,
    eps: f32,
    
    light_color: vec4f,
    light_direction: vec4f,
    
    show_normals: i32,
    show_steps: i32,
    visualize_distance_field: i32,

    _padding: i32,
    // _padding2: i32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var noise0_texture: texture_2d<f32>; // iChannel0
@group(1) @binding(1) var noise1_texture: texture_3d<f32>; // iChannel1
@group(1) @binding(2) var grain_texture: texture_2d<f32>;  // iChannel2
@group(1) @binding(3) var dirt_texture: texture_2d<f32>;   // iChannel3
@group(1) @binding(4) var terrain_sampler: sampler; // Must use repeat mode
@group(2) @binding(0) var<uniform> settings: Settings;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_position: vec3f,
    time: f32,
};

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) tex_uv: vec2f,
    @location(2) normal: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) tex_uv: vec2f,
    @location(1) normal: vec3f,
    @location(2) world_position: vec3f,
};

struct HitInfo {
    is_hit: bool,
    t: f32,
    n: vec3f,
    id: vec3f,
    i: i32,
};


@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec2f, 4>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, 1.0)
    );
    let tex_coords = array<vec2f, 4>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0),
        vec2f(0.0, 1.0), vec2f(1.0, 1.0)
    );
    var output: VertexOutput;
    output.position = vec4f(positions[vertex_index], 0.0, 1.0);
    output.tex_uv = tex_coords[vertex_index];
    output.normal = vec3f(0.0, 0.0, 1.0);
    output.world_position = vec3f(0.0);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let ro = camera.camera_position;
    let ndc = vec4f(input.tex_uv * 2.0 - 1.0, 1.0, 1.0);
    let world_pos = camera.inv_view_proj * ndc;
    let rd = normalize(world_pos.xyz / world_pos.w - ro);

    let sun_dir = normalize(settings.light_direction.xyz);

    var sky_ro = vec3f(0.0, earth_radius + 1.0, 0.0);
    let sky_color =  get_incident_light(Ray(sky_ro, rd), sun_dir);
    return vec4f(sky_color, 1.0);
}