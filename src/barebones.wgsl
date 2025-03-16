// shader.wgsl

const PI = 3.14159265359;
const earth_radius = 6360e3;
const atmosphere_radius = 6420e3;

struct Settings {

};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var rgb_noise_texture: texture_2d<f32>;
@group(1) @binding(1) var gray_noise_texture: texture_2d<f32>;
@group(1) @binding(2) var gray_cube_noise_texture: texture_3d<f32>; 
@group(1) @binding(3) var grain_texture: texture_2d<f32>; 
@group(1) @binding(4) var dirt_texture: texture_2d<f32>;  
@group(1) @binding(5) var pebble_texture: texture_2d<f32>;  
@group(1) @binding(6) var terrain_sampler: sampler; // Must use repeat mode
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
    var sky_ro = ro + vec3f(0.0, earth_radius + 1.0, 0.0);
    
    return vec4f(0.0, 0.0, 0.0, 1.0);
}