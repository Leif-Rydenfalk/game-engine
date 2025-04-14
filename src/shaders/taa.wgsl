// taa_resolve.wgsl

struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
    resolution: vec2<f32>,
    _padding: vec2<f32>,
};

struct TaaCameraUniform {
    prev_view_proj: mat4x4<f32>, // Unjittered VP from last frame
    prev_inv_view_proj: mat4x4<f32>,
    jitter_offset: vec2<f32>,
    _padding: vec2<f32>,
};


@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> taa_camera: TaaCameraUniform;

@group(1) @binding(0) var current_frame_tex: texture_2d<f32>; // Rgba32Float (HDR Scene Color + Sky)
@group(1) @binding(1) var history_tex: texture_2d<f32>;       // Rgba32Float (HDR TAA Result from prev frame)
@group(1) @binding(2) var depth_tex: texture_2d<f32>;         // R32Float (Linear distance or view Z)
@group(1) @binding(3) var texture_sampler: sampler;           // Sampler for history/current (linear filtering)
@group(1) @binding(4) var output_tex: texture_storage_2d<rgba32float, write>; // Write target (HDR)

// Constants for tuning
const TAA_BLEND_FACTOR: f32 = 0.98; // History weight (0.9 to 0.98 typical). Higher = smoother, more ghosting risk. Lower = sharper, more jitter.
const BACKGROUND_DEPTH_THRESHOLD: f32 = 99999.0; // Value indicating no hit (should match depth clear value)

// Reconstruct world position assuming depth_value is linear distance 'd' from camera
fn world_position_from_depth_distance(uv: vec2<f32>, distance: f32) -> vec3<f32> {
    // 1. Find the world-space direction vector for this pixel's view ray
    let clip_xy = uv * 2.0 - 1.0; // UV to NDC xy [-1, 1]
    // Use inv_view_proj: project point on far plane NDC back to world space
    // Z=1 is far plane in NDC (assuming depth range [-1, 1] after projection)
    let ndc_far = vec4(clip_xy.x, clip_xy.y, 1.0, 1.0);
    var world_far = camera.inv_view_proj * ndc_far;
    world_far /= world_far.w; // Perspective divide

    // Calculate the direction vector from the camera's eye position
    let dir_world = normalize(world_far.xyz - camera.position);

    // 2. Calculate the world position by moving along the direction by the distance
    let world_pos = camera.position + dir_world * distance;
    return world_pos;
}

// Clamps the history color sample within the AABB of the current frame's neighborhood
// Reduces ghosting artifacts by preventing stale history data from differing too much.
fn clamp_history_aabb(history_color: vec4<f32>, current_uv: vec2<f32>) -> vec4<f32> {
    let texel_size = 1.0 / camera.resolution;
    var min_color = vec4(10000.0); // Initialize high
    var max_color = vec4(-10000.0); // Initialize low

    // Sample 3x3 neighborhood in the *current* frame
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let offset = vec2(f32(x), f32(y)) * texel_size;
            // Use textureSampleLevel for explicit LOD 0 sampling
            let neighbor_color = textureSampleLevel(current_frame_tex, texture_sampler, current_uv + offset, 0.0);
            min_color = min(min_color, neighbor_color);
            max_color = max(max_color, neighbor_color);
        }
    }

    // Clamp the raw history sample to the min/max bounds found in the current frame's neighborhood
    return clamp(history_color, min_color, max_color);
}


@compute @workgroup_size(8, 8) // Match dispatch size in Rust code
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_coord = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let screen_dims = textureDimensions(output_tex);

    // Bounds check
    if (global_id.x >= screen_dims.x || global_id.y >= screen_dims.y) {
        return;
    }

    // Calculate UV coordinate for the center of the pixel
    let current_uv = (vec2<f32>(global_id.xy) + vec2(0.5)) / vec2<f32>(screen_dims);

    // 1. Sample Current Frame Color (jittered scene render)
    let current_color = textureSampleLevel(current_frame_tex, texture_sampler, current_uv, 0.0);

    // 2. Sample Depth (use textureLoad for non-filterable R32Float)
    // Assumes depth is in the red channel.
    let depth_value = textureLoad(depth_tex, pixel_coord, 0).r;

    // 3. Handle Background Pixels (where ray didn't hit anything)
    // If depth is very large (or sentinel value), no valid history exists. Output current color.
    if (depth_value >= BACKGROUND_DEPTH_THRESHOLD) {
        textureStore(output_tex, pixel_coord, current_color);
        return;
    }

    // 4. Reconstruct World Position from depth
    // !! CRITICAL STEP: Assumes depth_value is linear distance !!
    let world_pos = world_position_from_depth_distance(current_uv, depth_value);

    // 5. Reproject World Position to Previous Frame's Clip Space
    // Use the UNJITTERED previous view-projection matrix
    var prev_clip_pos = taa_camera.prev_view_proj * vec4(world_pos, 1.0);

    // 6. Perspective Divide to get Previous Frame's NDC [-1, 1]
    // Check for w <= 0 to avoid division by zero/negative (points behind camera)
     if (prev_clip_pos.w <= 0.0) {
        textureStore(output_tex, pixel_coord, current_color); // Fallback if reprojection fails
        return;
    }
    prev_clip_pos /= prev_clip_pos.w;

    // 7. Convert Previous Frame's NDC to Previous Frame's UV [0, 1]
    let prev_uv = prev_clip_pos.xy * 0.5 + 0.5;

    // 8. Check if Previous UV is within screen bounds [0, 1]
    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0) {
         // History is outside the screen (e.g., camera moved revealing new area).
         // Use current color only, no history blend.
        textureStore(output_tex, pixel_coord, current_color);
        return;
    }

    // 9. Sample History Buffer at the reprojected previous UV
    let history_color_raw = textureSampleLevel(history_tex, texture_sampler, prev_uv, 0.0);

    // 10. Clamp History Sample to Neighborhood AABB (Anti-Ghosting)
    let history_color_clamped = clamp_history_aabb(history_color_raw, current_uv);

    // 11. Blend Current and (Clamped) History
    // Simple linear interpolation. More advanced techniques might vary blend factor based on motion magnitude, depth differences etc.
    let final_color = mix(current_color, history_color_clamped, TAA_BLEND_FACTOR);

    // 12. Write Final TAA Result
    textureStore(output_tex, pixel_coord, final_color);
}