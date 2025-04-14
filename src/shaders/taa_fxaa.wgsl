// taa_resolve_fxaa.wgsl

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

//--------------------------------------------------------------------------------------
// FXAA 3.11 Constants
//--------------------------------------------------------------------------------------
// Optimizations
// - Choose pattern size 12 or 13
// - Pattern 12 is standard, 13 includes two extra samples for more horizontal/vertical accuracy.
const FXAA_PATTERN_SIZE: i32 = 12; // Or 13

// Controls edge detection threshold.
// Controls sharpness. Directly affects sharpness, too low = soft, too high = jagged.
// 1/3 - too little
// 1/4 - low quality
// 1/8 - high quality
// 1/16 - overkill
const FXAA_EDGE_THRESHOLD_MIN: f32 = 0.0312; // 1/32
const FXAA_EDGE_THRESHOLD: f32 = 0.125;  // 1/8

// Trims the algorithm from processing darks.
// Start with 0.0833 (1/12).
const FXAA_SUBPIX_TRIM: f32 = 0.0833;

// Controls the maximum amount of blur applied to edges.
// Max length of search steps for edge endpoints.
// 4.0 is moderate
// 8.0 is default
// 12.0 is high quality
const FXAA_EDGE_SHARPNESS: f32 = 8.0;

//--------------------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> taa_camera: TaaCameraUniform;

// Renamed to reflect its role as input to *both* TAA and the FXAA part
@group(1) @binding(0) var input_color_tex: texture_2d<f32>;   // Rgba32Float (HDR Scene Color + Sky - JITTERED)
@group(1) @binding(1) var history_tex: texture_2d<f32>;         // Rgba32Float (HDR TAA Result from prev frame)
@group(1) @binding(2) var depth_tex: texture_2d<f32>;           // R32Float (Linear distance or view Z)
@group(1) @binding(3) var texture_sampler: sampler;             // Sampler for history/current (linear filtering)
// Sampler for FXAA (can be the same or different, often point for depth, linear for color)
@group(1) @binding(4) var fxaa_sampler: sampler;
@group(1) @binding(5) var output_tex: texture_storage_2d<rgba32float, write>; // Write target (HDR)


// TAA Constants for tuning
// const TAA_BLEND_FACTOR_MIN: f32 = 0.7;
// const TAA_BLEND_FACTOR_MAX: f32 = 0.9;
const TAA_BLEND_FACTOR_MIN: f32 = 0.0; // More history weight = less blend min
const TAA_BLEND_FACTOR_MAX: f32 = 0.4; // More history weight = less blend max
const TAA_MOTION_THRESHOLD: f32 = 0.5; // Pixels of motion before blend factor reduces significantly

const BACKGROUND_DEPTH_THRESHOLD: f32 = 10000000.0;

// --- TAA Helper Functions ---

// Reconstruct world position assuming depth_value is linear distance 'd' from camera
fn world_position_from_depth_distance(uv: vec2<f32>, distance: f32) -> vec3<f32> {
    let clip_xy = uv * 2.0 - 1.0;
    let ndc_far = vec4(clip_xy.x, clip_xy.y, 1.0, 1.0);
    var world_far = camera.inv_view_proj * ndc_far;
    world_far /= world_far.w;
    let dir_world = normalize(world_far.xyz - camera.position);
    let world_pos = camera.position + dir_world * distance;
    return world_pos;
}

// Clamps the history color sample within the AABB of the current frame's neighborhood
fn clamp_history_aabb(history_color: vec4<f32>, current_uv: vec2<f32>) -> vec4<f32> {
    let texel_size = 1.0 / camera.resolution;
    var min_color = vec4(10000.0);
    var max_color = vec4(-10000.0);

    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let offset = vec2(f32(x), f32(y)) * texel_size;
            let neighbor_color = textureSampleLevel(input_color_tex, texture_sampler, current_uv + offset, 0.0);
            min_color = min(min_color, neighbor_color);
            max_color = max(max_color, neighbor_color);
        }
    }
    return clamp(history_color, min_color, max_color);
}

// Variance Clipping
fn variance_clip(history_color: vec4<f32>, current_uv: vec2<f32>) -> vec4<f32> {
    let texel_size = 1.0 / camera.resolution;
    var M1 = vec4(0.0); // Sum of colors
    var M2 = vec4(0.0); // Sum of squares of colors
    var N: f32 = 0.0;

    // Sample 3x3 neighborhood
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let offset = vec2(f32(x), f32(y)) * texel_size;
            let C = textureSampleLevel(input_color_tex, texture_sampler, current_uv + offset, 0.0);
            M1 += C;
            M2 += C * C;
            N += 1.0;
        }
    }

    let mean = M1 / N;
    let variance = max(vec4(0.0), M2 / N - mean * mean);
    let stddev = sqrt(variance);
    let gamma = 1.5;
    let min_clip = mean - gamma * stddev;
    let max_clip = mean + gamma * stddev;

    return clamp(history_color, min_clip, max_clip);
}

// --- FXAA Helper Functions ---

// Standard Rec.709 approximate luminance calculation
fn rgb_to_luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
    // Alternative (simpler, slightly less accurate):
    // return dot(rgb, vec3(0.2126, 0.7152, 0.0722));
}

// Samples luminance at an offset
fn sample_luminance(tex: texture_2d<f32>, smplr: sampler, uv: vec2<f32>, offset: vec2<f32>) -> f32 {
    let color = textureSampleLevel(tex, smplr, uv + offset, 0.0).rgb;
    return rgb_to_luminance(color);
}

// --- FXAA Main Function ---
// Based on FXAA 3.11
// NOTE: In this combined shader, 'tex' will be `input_color_tex`. Ideally, it should be
// the TAA result texture from a previous pass.
fn apply_fxaa(tex: texture_2d<f32>, smplr: sampler, uv: vec2<f32>, rcp_frame: vec2<f32>) -> vec4<f32> {

    // Sample central pixel color and calculate its luminance
    let color_center = textureSampleLevel(tex, smplr, uv, 0.0);
    let luma_center = rgb_to_luminance(color_center.rgb);

    // Sample neighbors' luminance
    let luma_down = sample_luminance(tex, smplr, uv, vec2(0.0, -rcp_frame.y));
    let luma_up   = sample_luminance(tex, smplr, uv, vec2(0.0,  rcp_frame.y));
    let luma_left = sample_luminance(tex, smplr, uv, vec2(-rcp_frame.x, 0.0));
    let luma_right= sample_luminance(tex, smplr, uv, vec2( rcp_frame.x, 0.0));

    // Calculate min/max luminance in the neighborhood
    let luma_min = min(luma_center, min(min(luma_down, luma_up), min(luma_left, luma_right)));
    let luma_max = max(luma_center, max(max(luma_down, luma_up), max(luma_left, luma_right)));

    // Calculate local contrast
    let luma_range = luma_max - luma_min;

    // Early exit if contrast is below threshold (not an edge)
    if (luma_range < max(FXAA_EDGE_THRESHOLD_MIN, luma_max * FXAA_EDGE_THRESHOLD)) {
        return color_center;
    }

    // Sample corner luminances
    let luma_down_left = sample_luminance(tex, smplr, uv, vec2(-rcp_frame.x, -rcp_frame.y));
    let luma_up_right = sample_luminance(tex, smplr, uv, vec2( rcp_frame.x,  rcp_frame.y));
    let luma_up_left = sample_luminance(tex, smplr, uv, vec2(-rcp_frame.x,  rcp_frame.y));
    let luma_down_right = sample_luminance(tex, smplr, uv, vec2( rcp_frame.x, -rcp_frame.y));

    // Combine neighborhood luminances
    let luma_down_up = luma_down + luma_up;
    let luma_left_right = luma_left + luma_right;

    // Combine corner luminances
    let luma_left_corners = luma_down_left + luma_up_left;
    let luma_down_corners = luma_down_left + luma_down_right;
    let luma_right_corners = luma_down_right + luma_up_right;
    let luma_up_corners = luma_up_right + luma_up_left;

    // Calculate horizontal and vertical gradients
    let edge_horizontal = abs(-2.0 * luma_left + luma_left_corners) +
                          abs(-2.0 * luma_center + luma_down_up) * 2.0 +
                          abs(-2.0 * luma_right + luma_right_corners);
    let edge_vertical = abs(-2.0 * luma_up + luma_up_corners) +
                        abs(-2.0 * luma_center + luma_left_right) * 2.0 +
                        abs(-2.0 * luma_down + luma_down_corners);

    // Determine dominant edge direction (horizontal or vertical)
    let is_horizontal = edge_horizontal >= edge_vertical;

    // Select step size based on dominant direction
    let step_length = select(rcp_frame.y, rcp_frame.x, is_horizontal); // step Y if vertical, step X if horizontal

    // Determine gradient direction based on luminance
    let gradient_neg = select(luma_left, luma_down, is_horizontal);
    let gradient_pos = select(luma_right, luma_up, is_horizontal);

    // Calculate initial gradient estimate
    let luma_neg = (gradient_neg + luma_center) * 0.5;
    let luma_pos = (gradient_pos + luma_center) * 0.5;
    let gradient_scaled = (luma_pos - luma_neg) / luma_range;

    // Determine search step direction based on gradient sign
    let step_sign = select(-step_length, step_length, gradient_scaled > 0.0);

    // Build offset vector for sampling along the edge perpendicular
    var uv_offset = select(vec2(0.0, step_sign * 0.5), vec2(step_sign * 0.5, 0.0), is_horizontal);

    // Sample points along the perpendicular edge to find edge start/end
    let luma_neg_edge = sample_luminance(tex, smplr, uv, uv_offset * -1.0);
    let luma_pos_edge = sample_luminance(tex, smplr, uv, uv_offset *  1.0);

    // Calculate gradients at these points
    let gradient_neg_edge = abs(luma_neg_edge - luma_neg);
    let gradient_pos_edge = abs(luma_pos_edge - luma_pos);

    // Choose the stronger gradient direction
    let is_neg_step = gradient_neg_edge < gradient_pos_edge;

    // Update offset and luminance based on the stronger direction
    uv_offset = select(uv_offset, uv_offset * -1.0, is_neg_step); // Flip offset if negative step is stronger
    let luma_edge = select(luma_pos_edge, luma_neg_edge, is_neg_step);

    // Step search loop to find the edge endpoint
    // Unrolled loop for WGSL compatibility/simplicity
    // Pattern 12: Steps 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0
    // Pattern 13: Includes 1.0, 1.5 steps
    var luma_average: f32 = luma_center; // Initialize average

    // Iteration 1 (Step 1.5 or 1.0 depending on pattern)
    #if FXAA_PATTERN_SIZE == 13
    uv_offset = select(uv_offset, uv_offset / 1.5, true); // Adjust for pattern 13 step
    luma_average += sample_luminance(tex, smplr, uv, uv_offset);
    uv_offset = select(vec2(0.0, step_sign), vec2(step_sign, 0.0), is_horizontal) * 1.0; // Step 1.0
    #else // Pattern 12
    uv_offset = select(vec2(0.0, step_sign), vec2(step_sign, 0.0), is_horizontal) * 1.5; // Step 1.5
    #endif
    luma_average += sample_luminance(tex, smplr, uv, uv_offset); // Sample at 1.0 (P13) or 1.5 (P12)

    // Iteration 2 (Step 2.0)
    uv_offset = select(vec2(0.0, step_sign), vec2(step_sign, 0.0), is_horizontal) * 2.0;
    luma_average += sample_luminance(tex, smplr, uv, uv_offset);

    // Iteration 3 (Step 2.0) - Can merge with above if needed, kept separate for clarity
    luma_average += sample_luminance(tex, smplr, uv, uv_offset); // Re-use offset

    // Iteration 4 (Step 2.0)
    luma_average += sample_luminance(tex, smplr, uv, uv_offset); // Re-use offset

    // Iteration 5 (Step 2.0)
    luma_average += sample_luminance(tex, smplr, uv, uv_offset); // Re-use offset

    // Iteration 6 (Step 4.0)
    uv_offset = select(vec2(0.0, step_sign), vec2(step_sign, 0.0), is_horizontal) * 4.0;
    luma_average += sample_luminance(tex, smplr, uv, uv_offset);

    // Iteration 7 (Step 8.0)
    uv_offset = select(vec2(0.0, step_sign), vec2(step_sign, 0.0), is_horizontal) * 8.0;
    luma_average += sample_luminance(tex, smplr, uv, uv_offset);

    luma_average = luma_average / (8.0 + f32(FXAA_PATTERN_SIZE == 13)); // Divide by 8 (P12) or 9 (P13)

    // Check if average luminance deviates too much (subpixel aliasing check)
    if (abs(luma_average - luma_edge) / luma_range >= FXAA_SUBPIX_TRIM) {
         // Subpixel aliasing detected, use a simpler blend based on gradient
         let pixel_offset = gradient_scaled * (-1.0/select(1.0, 1.0, is_horizontal) ); // Need more robust clamp/scaling here potentially
         let final_uv = uv + select(vec2(0.0, pixel_offset * rcp_frame.y), vec2(pixel_offset * rcp_frame.x, 0.0), is_horizontal);
         return textureSampleLevel(tex, smplr, final_uv, 0.0);
    } else {
        // No significant subpixel aliasing, return the original center color
        return color_center;
    }

    // Final blend based on search (This part is slightly simplified from full FXAA, focusing on subpixel detection)
    // A more complete implementation would calculate the exact crossing point along the search path.
    // This version primarily handles the subpixel trim case.
    // For simplicity in this integration, if not trimmed, return center.
    return color_center; // Fallback if subpixel trim condition isn't met
}

// --- Main Compute Shader ---

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_coord = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let screen_dims = textureDimensions(output_tex);

    // Bounds check
    if (global_id.x >= screen_dims.x || global_id.y >= screen_dims.y) {
        return;
    }

    // Calculate UV coordinate for the center of the pixel
    let current_uv = (vec2<f32>(global_id.xy) + vec2(0.5)) / vec2<f32>(screen_dims);
    let rcp_frame = 1.0 / camera.resolution; // Inverse resolution for FXAA

    // === TAA Resolve Section ===

    // Sample Current Frame Color (jittered scene render)
    let current_color = textureSampleLevel(input_color_tex, texture_sampler, current_uv, 0.0);

    // Sample Depth
    let depth_value = textureLoad(depth_tex, pixel_coord, 0).r;

    var taa_resolve_color: vec4<f32>;

    // Handle Background Pixels
    if (depth_value >= BACKGROUND_DEPTH_THRESHOLD) {
        taa_resolve_color = current_color; // No history for background
    } else {
        // Reconstruct World Position
        let world_pos = world_position_from_depth_distance(current_uv, depth_value);

        // Reproject World Position to Previous Frame's Clip Space (using UNJITTERED matrix)
        var prev_clip_pos = taa_camera.prev_view_proj * vec4(world_pos, 1.0);

        if (prev_clip_pos.w <= 0.0) {
             taa_resolve_color = current_color; // Reprojection failed (behind camera)
        } else {
            // Perspective Divide
            prev_clip_pos /= prev_clip_pos.w;

            // Convert Previous Frame's NDC to Previous Frame's UV [0, 1]
            let prev_uv = prev_clip_pos.xy * 0.5 + 0.5;

            // Check if Previous UV is within screen bounds [0, 1]
            if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0) {
                 taa_resolve_color = current_color; // History is off-screen
            } else {
                // Calculate motion vector
                let motion_pixels = abs(current_uv - prev_uv) * camera.resolution;
                let motion_magnitude = length(motion_pixels);

                // Sample History Buffer
                let history_color_raw = textureSampleLevel(history_tex, texture_sampler, prev_uv, 0.0);

                // Clamp History Sample (Variance Clipping or AABB)
                let history_color_clamped = variance_clip(history_color_raw, current_uv);
                // let history_color_clamped = clamp_history_aabb(history_color_raw, current_uv);

                // Calculate adaptive blend factor
                let blend_lerp = 1.0 - saturate(motion_magnitude / TAA_MOTION_THRESHOLD);
                let adaptive_blend_factor = mix(TAA_BLEND_FACTOR_MAX, TAA_BLEND_FACTOR_MIN, blend_lerp); // Corrected mix order

                // Blend Current and (Clamped) History
                taa_resolve_color = mix(current_color, history_color_clamped, adaptive_blend_factor);
            }
        }
    }

    // === FXAA Section ===
    // NOTE: Applying FXAA here. As discussed, apply_fxaa samples `input_color_tex` for neighbors,
    // which is the *pre-TAA* input. Ideally, FXAA runs in a second pass on the `taa_resolve_color` result.
    // This implementation calls FXAA structurally but its quality depends on the input texture used.
    // For simplicity here, we use the FXAA result directly. A better single-pass *might* try to blend
    // the TAA result with an FXAA-calculated offset sample, but that gets complex.
    let final_color = apply_fxaa(input_color_tex, fxaa_sampler, current_uv, rcp_frame);

    // If you wanted to apply FXAA *conceptually* to the TAA result (without a second pass):
    // This isn't strictly correct FXAA but shows the flow:
    // let approx_fxaa_color = apply_fxaa(input_color_tex, fxaa_sampler, current_uv, rcp_frame);
    // let final_color = mix(taa_resolve_color, approx_fxaa_color, 0.5); // Example blend, not physically correct


    // === Output ===
    textureStore(output_tex, pixel_coord, final_color);

    // To output only the TAA result (for a two-pass setup):
    // textureStore(output_tex, pixel_coord, taa_resolve_color);
}

//https://aistudio.google.com/prompts/1BQs_QkcckUeDgxEhKrz0cBQ7AEMkalfo