// src/shaders/sky.wgsl

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var depth_texture: texture_2d<f32>; // Read R32Float depth
@group(1) @binding(1) var depth_sampler: sampler;       // Needed even for textureLoad
@group(2) @binding(0) var rgb_noise_texture: texture_2d<f32>;
@group(2) @binding(1) var gray_noise_texture: texture_2d<f32>;
@group(2) @binding(2) var gray_cube_noise_texture: texture_3d<f32>; 
@group(2) @binding(3) var grain_texture: texture_2d<f32>; 
@group(2) @binding(4) var dirt_texture: texture_2d<f32>;  
@group(2) @binding(5) var pebble_texture: texture_2d<f32>;  
@group(2) @binding(6) var terrain_sampler: sampler; // Must use repeat mode

// Shared structs (could be moved to a common file if larger)
struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_position: vec3f,
    time: f32,
    resolution: vec2f, // Added resolution
    _padding: vec2f, // Added padding
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) world_pos: vec3f, // Pass world position for view direction calc
    @location(1) ndc: vec4f, // Pass world position for view direction calc
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generate a fullscreen triangle strip
    let positions = array<vec2f, 4>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, 1.0)
    );
    let pos = positions[vertex_index];

    // Calculate world position for view direction in fragment shader
    let ndc = vec4f(pos.x, -pos.y, 1.0, 1.0); // Use Z=1 for far plane
    let world = camera.inv_view_proj * ndc;
    let world_xyz = world.xyz / world.w;

    var output: VertexOutput;
    output.position = vec4f(pos, 0.0, 1.0); // Project to near plane for rasterizer
    output.world_pos = world_xyz;
    output.ndc = ndc;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    // Get pixel coordinates (integer for textureLoad)
    let frag_coord = vec2<i32>(floor(input.position.xy));

    // Load depth from the custom R32Float depth texture
    // textureLoad requires integer coordinates and mip level (0 for base)
    // The result is vec4f, but only .r component is valid for R32Float
    let depth = textureLoad(depth_texture, frag_coord, 0).r;

    // If depth is very large (meaning voxel ray missed), draw sky
    if depth >= 100000000.0 {
        return vec4f(0.1, 0.2, 1.0, 1.0);
    } else {
        // Voxel terrain is here, discard this fragment so terrain shows through
        discard;
    }
}