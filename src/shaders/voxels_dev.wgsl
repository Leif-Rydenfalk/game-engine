// File: advanced_voxel_renderer.wgsl
// Purpose: Renders a voxel-based landscape with atmospheric scattering,
//          volumetric clouds, water reflections/refraction, and procedural texturing.

//--------------------------------------------------------------------------------------
// Compile-Time Constants
//--------------------------------------------------------------------------------------

// Mathematical Constants
const PI: f32 = 3.14159265359;
const EPSILON: f32 = 1e-4; // Small value for offsets, comparisons

// Physical Constants & Scene Scale
const EARTH_RADIUS: f32 = 6360e3;       // Radius of the planet (meters)
const ATMOSPHERE_RADIUS: f32 = 6420e3;  // Radius of the atmosphere outer bound (meters)

// Atmospheric Scattering Parameters (Precomputed / Fitted)
// Rayleigh
const RAYLEIGH_SCATTERING_COEFF: vec3f = vec3f(5.5e-6, 13.0e-6, 22.4e-6); // Beta R
const RAYLEIGH_SCALE_HEIGHT: f32 = 7994.0; // hR (meters)
// Mie
const MIE_SCATTERING_COEFF: vec3f = vec3f(21e-6); // Beta M
const MIE_SCALE_HEIGHT: f32 = 1200.0; // hM (meters)
const MIE_ASYMMETRY_FACTOR_G: f32 = 0.76; // g (Forward scattering preference)

// Sun Properties
const SUN_INTENSITY_FACTOR: f32 = 60.0; // Scalar multiplier for sun power (adjusts overall brightness)
const SUN_RADIANCE: vec3f = vec3f(1.0, 0.9, 0.6) * 750.0; // Color and power of the sun for cloud lighting

// Cloud Parameters
const CLOUD_LAYER_ALTITUDE_START: f32 = 800.0; // Altitude where cloud layer begins (meters)
const CLOUD_LAYER_THICKNESS: f32 = 600.0;    // Thickness of the cloud layer volume (meters)
const CLOUD_LOW_SCATTERING_COLOR: vec3f = vec3f(1.0, 0.7, 0.5); // Used in simplified cloud lighting (unused in main path?)

// Ray Marching / Sampling Quality
const ATMOSPHERE_SUN_LIGHT_SAMPLES: i32 = 8;
// Note: Cloud sampling counts are defined within cloud functions (lightRay, skyRay) based on 'fast' flag.

//--------------------------------------------------------------------------------------
// Structures
//--------------------------------------------------------------------------------------

struct Ray {
    origin: vec3f,
    direction: vec3f,
};

struct Sphere {
    origin: vec3f,
    radius: f32,
    radius_sq: f32, // Precomputed radius squared
};

// Result of a ray-sphere intersection test
struct IntersectionResult {
    hit: bool,
    t0: f32, // Near intersection distance (or entrance)
    t1: f32, // Far intersection distance (or exit)
};

// Holds information about a ray marching hit
struct VoxelHitInfo {
    is_hit: bool,    // Did the ray hit the terrain?
    t: f32,          // Distance along the ray to the hit point
    normal: vec3f,   // Normal vector at the hit point (voxel face normal)
    voxel_pos: vec3f,// Center position of the hit voxel
    steps: i32,      // Number of steps taken during the trace
};

// Uniform buffer structure for camera data
struct CameraUniform {
    view_proj: mat4x4<f32>,      // Combined view * projection matrix
    inv_view_proj: mat4x4<f32>,  // Inverse of view_proj
    view: mat4x4<f32>,           // View matrix
    camera_position: vec3f,      // World space camera position
    time: f32,                   // Shader execution time (seconds)
};

// Uniform buffer structure for rendering settings
struct SettingsUniform {
    // Voxel Terrain Generation & Rendering
    surface_threshold: f32, // Density value defining the surface
    max_terrain_height: f32,// Maximum height clamp for terrain generation
    voxel_size: f32,        // Size of individual voxels
    max_ray_steps: i32,     // Maximum steps for ray marching
    max_distance: f32,      // Maximum ray travel distance
    min_distance: f32,      // Minimum ray travel distance (unused?)
    ray_march_epsilon: f32, // Small offset for ray marching steps

    // Water Rendering
    water_level: f32,           // Global water height (y-coordinate)
    max_water_influenced_height: f32, // Height above water level where terrain color blends

    // Lighting
    light_color: vec4f,         // Color and intensity of the primary light (e.g., sun)
    light_direction: vec4f,     // Direction towards the light source (world space, xyz)

    // Debugging Flags
    show_normals: i32,              // Visualize surface normals if non-zero
    show_ray_steps: i32,            // Visualize ray marching step count if non-zero
    visualize_distance_field: i32,  // Visualize SDF value if non-zero

    // Unused / Legacy Parameters (from original code, kept for struct layout compatibility)
    max: f32,                   // Unused? Related to old map function?
    r_inner: f32,               // Unused?
    r: f32,                     // Unused?
    max_water_height: f32,      // Unused? Replaced by max_water_influenced_height?
    tunnel_radius: f32,         // Unused?
    camera_speed: f32,          // Unused? (Handled externally)
    camera_time_offset: f32,    // Unused? (Handled externally)
    voxel_level: i32,           // Unused?
    _padding: vec3<f32>,        // Explicit padding for alignment if needed (WGSL often handles this)
};

// Input structure for the vertex shader (for full-screen quad)
struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
};

// Output structure from the vertex shader / Input structure for the fragment shader
struct VertexOutput {
    @builtin(position) clip_position: vec4f, // Position in clip space
    @location(0) tex_coord: vec2f,         // Texture coordinate (0-1) for screen space effects/sampling
    // No world position needed here as we calculate ray direction in fragment shader
};

//--------------------------------------------------------------------------------------
// Bindings (Uniforms, Textures, Samplers)
//--------------------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var noise_texture_rgb: texture_2d<f32>;          // Generic RGB noise
@group(1) @binding(1) var noise_texture_gray: texture_2d<f32>;         // Generic grayscale noise (unused?)
@group(1) @binding(2) var noise_texture_3d: texture_3d<f32>;           // 3D noise for terrain SDF
@group(1) @binding(3) var grain_texture: texture_2d<f32>;              // Surface detail texture (wood grain?)
@group(1) @binding(4) var dirt_texture: texture_2d<f32>;               // Surface detail/biome texture (dirt/rock?)
@group(1) @binding(5) var pebble_texture: texture_2d<f32>;             // Noise texture used for clouds/weather
@group(1) @binding(6) var texture_sampler: sampler;                    // Sampler for all textures
@group(2) @binding(0) var<uniform> settings: SettingsUniform;

//--------------------------------------------------------------------------------------
// Global Variables (used across multiple functions)
//--------------------------------------------------------------------------------------

// Define atmosphere sphere globally for easy access in scattering functions
const atmosphere_sphere: Sphere = Sphere(
    vec3f(0.0), // Centered at origin
    ATMOSPHERE_RADIUS,
    ATMOSPHERE_RADIUS * ATMOSPHERE_RADIUS
);

//--------------------------------------------------------------------------------------
// Utility Functions
//--------------------------------------------------------------------------------------

// Smooth minimum function (blends two distances)
fn smooth_min(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

// Smooth maximum function (blends two distances)
fn smooth_max(d1: f32, d2: f32, k: f32) -> f32 {
    // Note: Equivalent to -smooth_min(-d1, -d2, k)
    let h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

// Simple pseudo-random number generator based on input float
fn hash1(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

// Simple pseudo-random number generator based on input vec3
fn hash1_from_3(p: vec3f) -> f32 {
    var p3 = fract(p * 0.1031); // Fractional part to break pattern
    p3 += dot(p3, p3.yzx + 33.33); // Mix components
    return fract((p3.x + p3.y) * p3.z); // Final mix and fractional part
}

// Intersects a ray with a sphere. Returns distance to nearest hit or -1.0 if no hit.
// Uses the robust quadratic formula solution.
fn intersect_sphere(ray: Ray, sphere: Sphere) -> f32 {
    let oc = ray.origin - sphere.origin;
    let b = dot(oc, ray.direction);
    let c = dot(oc, oc) - sphere.radius_sq;
    let h = b*b - c; // Discriminant (optimized calculation a=1)
    if (h < 0.0) {
        return -1.0; // No intersection
    }
    let sqrt_h = sqrt(h);
    // Calculate intersection distances t0 and t1
    let t0 = -b - sqrt_h;
    let t1 = -b + sqrt_h;

    // Return nearest valid intersection distance (t >= 0)
    if (t0 >= 0.0) { return t0; }
    if (t1 >= 0.0) { return t1; }
    return -1.0; // Both intersections are behind the ray origin
}


// Intersects a ray with a sphere. Returns detailed hit info (hit, t0, t1).
fn intersect_sphere_detailed(ray: Ray, sphere: Sphere) -> IntersectionResult {
    let rc = sphere.origin - ray.origin;
    let radius2 = sphere.radius_sq; // Use precomputed radius squared
    let tca = dot(rc, ray.direction);
    let d2 = dot(rc, rc) - tca * tca;

    if (d2 > radius2) {
        // Ray misses the sphere
        return IntersectionResult(false, 0.0, 0.0);
    }

    let thc = sqrt(radius2 - d2);
    let t0 = tca - thc;
    let t1 = tca + thc;

    // Return valid intersection points (can be negative if origin is inside)
    return IntersectionResult(true, t0, t1);
}


//--------------------------------------------------------------------------------------
// Noise Functions (Used for terrain, clouds, texturing)
//--------------------------------------------------------------------------------------

// Samples 3D texture noise using tri-linear filtering (via textureSampleLevel).
// Assumes noise texture is tiled across 32 units.
fn sample_noise_3d(p: vec3f) -> f32 {
    let p_floor = floor(p);
    let p_fract = fract(p);
    // Apply smoothstep for smoother interpolation (f = f*f*(3-2*f))
    let p_smooth = p_fract * p_fract * (3.0 - 2.0 * p_fract);
    // Sample the 3D noise texture (adjust coordinates to texture space)
    let tex_coord = (p_floor + p_smooth + 0.5) / 32.0; // +0.5 to sample cell center
    return textureSampleLevel(noise_texture_3d, texture_sampler, tex_coord, 0.0).r; // Use red channel
}

// Samples 2D texture noise using bi-linear filtering.
// Assumes noise texture is tiled across 256 units.
fn sample_noise_2d(p: vec2f) -> f32 {
    let p_floor = floor(p);
    let p_fract = fract(p);
    let p_smooth = p_fract * p_fract * (3.0 - 2.0 * p_fract);
    let tex_coord = (p_floor + p_smooth + 0.5) / 256.0;
    return textureSampleLevel(noise_texture_rgb, texture_sampler, tex_coord, 0.0).r; // Use red channel
}

// Fractional Brownian Motion (FBM) using 3D noise.
// Creates more complex patterns by summing noise at different frequencies/amplitudes.
fn fbm_3d(p: vec3f) -> f32 {
    // Rotation matrix to vary noise orientation across octaves
    let rot_matrix = mat3x3<f32>(
        vec3f( 0.00,  0.80,  0.60),
        vec3f(-0.80,  0.36, -0.48),
        vec3f(-0.60, -0.48,  0.64)
    );

    var frequency_multiplier: f32 = 2.02; // Lacunarity (how much frequency increases per octave)
    var amplitude_multiplier: f32 = 0.5; // Gain (how much amplitude decreases per octave)

    var noise_sum: f32 = 0.0;
    var current_p = p;

    // 3 Octaves of noise
    noise_sum += amplitude_multiplier * sample_noise_3d(current_p);

    amplitude_multiplier *= 0.5;
    current_p = rot_matrix * current_p * frequency_multiplier;
    noise_sum += amplitude_multiplier * sample_noise_3d(current_p);

    // Slightly change frequency multiplier for variation
    frequency_multiplier = 2.03;
    amplitude_multiplier *= 0.5;
    current_p = rot_matrix * current_p * frequency_multiplier;
    noise_sum += amplitude_multiplier * sample_noise_3d(current_p);

    // Note: Original code sums to 0.5 + 0.25 + 0.125 = 0.875.
    // Normalization might be needed depending on desired output range.
    // return noise_sum / 0.875; // Example normalization
    return noise_sum;
}


//--------------------------------------------------------------------------------------
// Atmospheric Scattering Functions
//--------------------------------------------------------------------------------------

// Rayleigh phase function (approximates scattering by small molecules like N2, O2)
fn phase_rayleigh(cos_theta: f32) -> f32 {
    // cos_theta: cosine of the angle between viewing direction and light direction
    let factor = 3.0 / (16.0 * PI);
    return factor * (1.0 + cos_theta * cos_theta);
}

// Henyey-Greenstein phase function (approximates scattering by larger particles like aerosols, dust)
fn phase_henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    // g: asymmetry parameter (-1 = backscatter, 0 = isotropic, 1 = forward scatter)
    let g2 = g * g;
    let factor = (1.0 - g2) / (4.0 * PI);
    let denominator = pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
    return factor / denominator;
}

// A numerical fit for Mie scattering phase function based on precomputed data or simulation.
// Provides a more accurate representation than Henyey-Greenstein for specific particle types.
// Parameters derived from external fitting process.
fn phase_mie_fitted(cos_theta: f32) -> f32 {
    // These parameters are specific to the fitting data used.
    const FIT_PARAMS = array<f32, 10>(
        9.805233e-06, -6.500000e+01, -5.500000e+01, 8.194068e-01,
        1.388198e-01, -8.370334e+01, 7.810083e+00, 2.054747e-03,
        2.600563e-02, -4.552125e-12
    );

    let p1 = cos_theta + FIT_PARAMS[3];
    let exp_terms = vec4f(
        exp(FIT_PARAMS[1] * cos_theta + FIT_PARAMS[2]), // Term 1
        exp(FIT_PARAMS[5] * p1 * p1),                  // Term 2
        exp(FIT_PARAMS[6] * cos_theta),                  // Term 3
        exp(FIT_PARAMS[9] * cos_theta)                   // Term 4 (Note: param 9 is very small)
    );

    let weights = vec4f(FIT_PARAMS[0], FIT_PARAMS[4], FIT_PARAMS[7], FIT_PARAMS[8]);

    // Return the weighted sum of exponential terms
    return dot(exp_terms, weights);
}


// Calculates optical depth for Rayleigh and Mie scattering along a ray segment towards the sun.
// Returns true if the path is not obstructed by the planet.
fn get_optical_depth_to_sun(
    ray: Ray,
    segment_length: f32,
    out_optical_depth_R: ptr<function, f32>, // Rayleigh optical depth output
    out_optical_depth_M: ptr<function, f32>  // Mie optical depth output
) -> bool {

    // Check if the ray segment intersects the atmosphere at all
    let isect = intersect_sphere_detailed(ray, atmosphere_sphere);
    if (!isect.hit) {
        // Ray completely misses the atmosphere (shouldn't happen if origin is inside?)
        return false; // Or handle based on context, maybe return max depth?
    }

    // Determine the actual length to integrate over (clamp to intersection or segment_length)
    let integration_length = min(segment_length, max(0.0, isect.t1));
    if (integration_length <= 0.0) {
        return true; // No path through atmosphere along this segment
    }

    let num_steps = ATMOSPHERE_SUN_LIGHT_SAMPLES;
    let step_size = integration_length / f32(num_steps);

    var current_t: f32 = 0.0;
    var accumulated_depth_R: f32 = 0.0;
    var accumulated_depth_M: f32 = 0.0;

    for (var i = 0; i < num_steps; i = i + 1) {
        // Sample at the midpoint of the step
        let sample_pos = ray.origin + ray.direction * (current_t + 0.5 * step_size);
        let height = length(sample_pos) - EARTH_RADIUS;

        if (height < 0.0) {
            // Path goes below the Earth's surface - fully occluded
            // Return large optical depth to signify blockage
            *out_optical_depth_R = 1000.0;
            *out_optical_depth_M = 1000.0;
            return false;
        }

        // Accumulate optical depth based on height-dependent density
        // Density ~ exp(-height / scale_height)
        accumulated_depth_R += exp(-height / RAYLEIGH_SCALE_HEIGHT) * step_size;
        accumulated_depth_M += exp(-height / MIE_SCALE_HEIGHT) * step_size;

        current_t += step_size;
    }

    *out_optical_depth_R = accumulated_depth_R;
    *out_optical_depth_M = accumulated_depth_M;

    return true; // Path was clear of the planet body
}


//--------------------------------------------------------------------------------------
// Cloud Rendering Functions
//--------------------------------------------------------------------------------------

// Calculates cloud density at a given world position 'p'.
// Also outputs normalized height within the cloud layer.
// 'use_simplified_noise': If true, uses fewer FBM octaves for performance.
fn get_cloud_density(
    p: vec3f,
    world_time: f32,
    out_cloud_layer_height_norm: ptr<function, f32>, // Normalized height [0, 1] within cloud layer
    use_simplified_noise: bool
) -> f32 {

    // Calculate height above sea level (assuming planet center at origin)
    // Adjust Y coordinate to account for earth radius if necessary (depends on coordinate system)
    // Original code assumes p.y is height above a flat plane centered at origin,
    // adjusted by EARTH_RADIUS later. Let's follow that for consistency.
    let altitude = length(p - vec3f(0.0, -EARTH_RADIUS, 0.0)) - EARTH_RADIUS;

    // Calculate normalized height within the defined cloud layer boundaries
    let cloud_layer_progress = clamp(
        (altitude - CLOUD_LAYER_ALTITUDE_START) / CLOUD_LAYER_THICKNESS,
        0.0, 1.0
    );
    *out_cloud_layer_height_norm = cloud_layer_progress;

    // Fade density at the top and bottom of the layer
    let vertical_fade = smoothstep(0.0, 0.5, cloud_layer_progress) * smoothstep(1.0, 0.5, cloud_layer_progress);
    if (vertical_fade <= 0.0) {
        return 0.0; // Outside the effective vertical range
    }

    // --- Cloud Shape and Weather Pattern ---
    // Use pebble texture for large scale weather patterns (moves slowly)
    var weather_p = p;
    weather_p.z += world_time * 10.3; // Animate weather pattern Z
    // Sample large scale weather map - map texture values to cloud coverage/density multiplier
    // Adjust threshold (0.18) and scale (5.0) to control weather appearance
    let large_weather_factor = clamp((textureSampleLevel(pebble_texture, texture_sampler, -0.00005 * weather_p.zx, 0.0).r - 0.18) * 5.0, 0.0, 2.0);

    // Sample smaller scale weather details (moves slightly faster)
    weather_p.x += world_time * 8.3; // Animate weather pattern X (offset from Z animation)
    // Map texture values to smaller variations, modulate by large pattern
    // Adjust threshold (0.28) and scale (1/0.72)
    let weather_detail_factor = max(0.0, textureSampleLevel(pebble_texture, texture_sampler, 0.0002 * weather_p.zx, 0.0).r - 0.28) / 0.72;
    let combined_weather = large_weather_factor * weather_detail_factor;

    // Modulate weather by vertical position within the cloud layer
    let weather_adjusted = combined_weather * vertical_fade;

    // Basic cloud shape function - pow enhances contrast based on height
    let base_cloud_shape = pow(weather_adjusted, 0.3 + 1.5 * smoothstep(0.2, 0.5, cloud_layer_progress));
    if (base_cloud_shape <= 0.0) {
        return 0.0;
    }

    // --- Cloud Detail (FBM Noise) ---
    var detail_p = p;
    detail_p.x += world_time * 12.3; // Animate primary detail noise

    // Apply primary FBM noise to break up the base shape
    // Scale (0.01) controls the size of the noise features
    var density = max(0.0, base_cloud_shape - 0.7 * fbm_3d(detail_p * 0.01));
    if (density <= 0.0) {
        return 0.0;
    }

    // Optional: Add higher frequency noise for finer detail (if not simplified)
    if (!use_simplified_noise) {
        detail_p.y += world_time * 15.2; // Animate secondary detail noise (offset)
        // Scale (0.05) is higher frequency than primary noise
        density = max(0.0, density - 0.2 * fbm_3d(detail_p * 0.05));
        if (density <= 0.0) { return 0.0; }
    }

    // Final density modulation and scaling
    // 'large_weather_factor * 0.2' provides overall density control based on weather
    // 'min(1.0, 5.0 * density)' clamps and scales the detailed noise contribution
    let final_density = large_weather_factor * 0.2 * min(1.0, 5.0 * density);

    return final_density;
}


// Calculates incoming light contribution for a point within the clouds.
// Marches a ray towards the sun to estimate light attenuation/scattering.

fn calculate_cloud_lighting(
    p: vec3f,
    sun_direction: vec3f,
    phase_function_value: f32,
    cloud_density_at_p: f32,
    cloud_layer_height_norm: f32,
    world_time: f32,
    use_simplified_lighting: bool
) -> f32 { // Return the final *modulated* scattering contribution

    let num_light_samples = select(30, 7, use_simplified_lighting);
    let max_light_ray_dist = 600.0;
    let step_size = max_light_ray_dist / f32(num_light_samples);
    var accumulated_density_along_light_ray: f32 = 0.0;
    let random_offset = hash1(dot(p, vec3f(12.256, 2.646, 6.356)) + world_time);
    var current_pos_along_light_ray = p + sun_direction * step_size * random_offset;

    for (var j: i32 = 0; j < num_light_samples; j = j + 1) {
        var sample_cloud_layer_height: f32;
        accumulated_density_along_light_ray += get_cloud_density(
            current_pos_along_light_ray,
            world_time,
            &sample_cloud_layer_height,
            use_simplified_lighting
        );
        current_pos_along_light_ray += sun_direction * step_size;
    }

    let total_optical_depth = step_size * accumulated_density_along_light_ray;

    var light_scatter_contribution : f32; // Will hold the final result

    if (use_simplified_lighting) {
        // Original simplified logic:
        // light_intensity = exp(-0.4 * total_optical_depth); // Example simplified extinction
        // return light_intensity * phase_function_value; // Apply phase (no extra modulation)

        // Let's match the original `lightRay` simplified path structure more closely:
        // Note: Original used 'mu', which isn't available here easily. We stick to simplified extinction.
        // The key is applying phase function at the end.
        let beers_simple = (0.5 * exp(-0.4 * total_optical_depth) + 0.5 * exp(-0.02 * total_optical_depth)); // Adapted from original, removed mu term
        light_scatter_contribution = beers_simple * phase_function_value;

    } else {
        // High quality path - Revert modulation order
        let view_sun_cosine = dot(normalize(p - camera.camera_position), sun_direction); // Approx mu needed for scatter_amount
        let scatter_amount = mix(0.008, 1.0, smoothstep(0.96, 0.0, view_sun_cosine));

        let beers_law_term = exp(-total_optical_depth);
        let scatter_term_1 = 0.5 * scatter_amount * exp(-0.1 * total_optical_depth);
        let scatter_term_2 = 0.4 * scatter_amount * exp(-0.02 * total_optical_depth);

        // Calculate combined intensity based on Beer's law + scattering terms
        let base_light_intensity = beers_law_term + scatter_term_1 + scatter_term_2;

        // Calculate result *before* final modulation (Intensity * Phase)
        let scattered_light_before_modulation = base_light_intensity * phase_function_value;

        // Apply the final modulation based on density/height profile (matching original structure)
        let density_height_factor = mix(
            0.05 + 1.5 * pow(min(1.0, cloud_density_at_p * 8.5), 0.3 + 5.5 * cloud_layer_height_norm),
            1.0,
            // Original used lighRayDen * 0.4 => total_optical_depth / step_size * 0.4
            // Let's use total_optical_depth directly for simplicity, adjusting the multiplier maybe?
            // Or stick closer to original: clamp(accumulated_density_along_light_ray * 0.4, 0.0, 1.0) ???
            // Let's try using total_optical_depth with adjusted scale:
             clamp(total_optical_depth * 0.1, 0.0, 1.0) // Keep the factor from the refactored version for now, but applied at the end
        );

        light_scatter_contribution = scattered_light_before_modulation * density_height_factor;
    }

    return light_scatter_contribution;
}


// Integrates scattering and extinction through the cloud layer along the view ray.
fn render_clouds_along_ray(
    ray: Ray,
    sun_direction: vec3f,
    world_time: f32,
    use_simplified_sampling: bool
) -> vec3f {

    // Calculate intersection points with the cloud layer spheres
    // Define spheres slightly offset from the actual start/end altitudes
    // Using simplified intersection assuming origin at (0, -EARTH_RADIUS, 0)
    let cloud_sphere_origin = vec3f(0.0, -EARTH_RADIUS, 0.0);
    let atm_start_sphere = Sphere(cloud_sphere_origin, EARTH_RADIUS + CLOUD_LAYER_ALTITUDE_START, (EARTH_RADIUS + CLOUD_LAYER_ALTITUDE_START)*(EARTH_RADIUS + CLOUD_LAYER_ALTITUDE_START));
    let atm_end_sphere = Sphere(cloud_sphere_origin, EARTH_RADIUS + CLOUD_LAYER_ALTITUDE_START + CLOUD_LAYER_THICKNESS, (EARTH_RADIUS + CLOUD_LAYER_ALTITUDE_START + CLOUD_LAYER_THICKNESS)*(EARTH_RADIUS + CLOUD_LAYER_ALTITUDE_START + CLOUD_LAYER_THICKNESS));

    let dist_to_atm_start = intersect_sphere(ray, atm_start_sphere);
    let dist_to_atm_end = intersect_sphere(ray, atm_end_sphere);

    // If the ray misses the outer bound of the cloud layer, return black
    if (dist_to_atm_end < 0.0) {
        return vec3f(0.0);
    }

    // Determine the segment of the ray that lies within the cloud layer
    let march_start_distance = max(0.0, dist_to_atm_start); // Start at ray origin or entry point
    let march_end_distance = dist_to_atm_end;              // End at exit point
    let segment_length = march_end_distance - march_start_distance;

    if (segment_length <= 0.0) {
        return vec3f(0.0); // Ray passes outside or tangentially
    }

    // Determine number of steps based on quality
    let num_view_samples = select(35, 13, use_simplified_sampling); // More steps for high quality
    let step_size = segment_length / f32(num_view_samples);

    var accumulated_color = vec3f(0.0);
    var accumulated_transmittance = 1.0; // Start with full transmittance

    // Calculate view-sun cosine and phase function once for the ray
    let view_sun_cosine = dot(ray.direction, sun_direction);
    // Use the more accurate fitted Mie phase function for clouds
    let phase_function_value = phase_mie_fitted(view_sun_cosine);
    // let phase_function_value = phase_henyey_greenstein(view_sun_cosine, MIE_ASYMMETRY_FACTOR_G); // Alternative

    // Offset start position slightly using hash to reduce banding
    let random_offset = hash1(dot(ray.direction, vec3f(12.256, 2.646, 6.356)) + world_time);
    var current_pos = ray.origin + ray.direction * (march_start_distance + step_size * random_offset);

    // Ray march through the cloud layer
    for (var i: i32 = 0; i < num_view_samples; i = i + 1) {
        var cloud_layer_height_norm: f32;
        let density = get_cloud_density(
            current_pos,
            world_time,
            &cloud_layer_height_norm,
            use_simplified_sampling // Use simplified noise if using simplified sampling
        );

        if (density > EPSILON) { // Process step only if density is significant
            // Calculate incoming light at this point
            let light_intensity = calculate_cloud_lighting(
                current_pos,
                sun_direction,
                phase_function_value,
                density,
                cloud_layer_height_norm,
                world_time,
                use_simplified_sampling // Match quality setting for lighting calculation
            );

            // Estimate ambient light term (empirical, based on height)
            // Blueish tint higher up, whitish lower down
            let ambient_sky_color = mix(vec3f(0.8), vec3f(0.2, 0.5, 1.0) * 6.5, cloud_layer_height_norm);
            let ambient_term = (0.5 + 0.6 * cloud_layer_height_norm) * ambient_sky_color;


            // Combine direct sun light and ambient light
            let total_radiance = ambient_term + SUN_RADIANCE * light_intensity;

            // Calculate radiance scattered towards the camera in this step (using volume rendering integral approximation)
            // Integral[0, L] T(0,s) * sigma_s * L_in(s) ds
            // Approx: T(step start) * (1 - exp(-density * step_size)) * L_in(step middle)
            let step_extinction = density * step_size;
            let transmittance_step = exp(-step_extinction);

            // Color contribution from this step (In-scattering)
            // The original code used a slightly different formulation:
            // color += T * (radianceDensity - radianceDensity * exp(-density * stepS)) / density;
            // which simplifies to: T * radiance * (1 - exp(-density * stepS))
            accumulated_color += accumulated_transmittance * total_radiance * (1.0 - transmittance_step);

            // Update transmittance for the next step
            accumulated_transmittance *= transmittance_step;

            // Early exit if transmittance is very low
            if (accumulated_transmittance < 0.01) { // Adjusted threshold from 0.05
                break;
            }
        } // end if density > EPSILON

        // Move to the next sample point
        current_pos += ray.direction * step_size;

    } // end for loop

    // Add background sky color, attenuated by cloud transmittance
    // Background color depends on view angle relative to sun (brighter near sun)
    let background_sky_color = mix(
        vec3f(0.2, 0.52, 1.0), // Horizon/away from sun
        vec3f(0.8, 0.95, 1.0), // Near sun
        pow(max(0.0, view_sun_cosine) * 0.5 + 0.5, 15.0) // Blend based on angle
    );
    accumulated_color += background_sky_color * accumulated_transmittance;

    return accumulated_color;
}

//--------------------------------------------------------------------------------------
// Voxel Terrain Signed Distance Field (SDF) and Ray Marching
//--------------------------------------------------------------------------------------

// Defines the terrain shape using a Signed Distance Function (SDF).
// Returns estimated distance to the surface (positive outside, negative inside).
fn terrain_sdf(p: vec3f) -> f32 {
    // Base density field using FBM noise (3 octaves from 3D texture)
    let noise_scale: f32 = 0.3; // Controls overall scale of noise features
    let noise_coord_base = p * noise_scale / 32.0 - vec3f(0.003, -0.006, 0.0); // Apply scaling and small offset

    // Sum multiple octaves of noise from the 3D texture
    var density: f32 = 0.0;
    density += textureSampleLevel(noise_texture_3d, texture_sampler, noise_coord_base * 1.0, 0.0).r * 0.5; // Octave 1
    density += textureSampleLevel(noise_texture_3d, texture_sampler, noise_coord_base * 2.0 + vec3f(0.3, 0.3, 0.3), 0.0).r * 0.25; // Octave 2 (offset coords)
    density += textureSampleLevel(noise_texture_3d, texture_sampler, noise_coord_base * 4.0 + vec3f(0.7, 0.7, 0.7), 0.0).r * 0.125; // Octave 3 (offset coords)

    // Normalize density (sum of weights = 0.5 + 0.25 + 0.125 = 0.875)
    let normalized_density = density / 0.875;

    // Convert density to distance: distance = (density - threshold) / scale
    // Lower threshold = more terrain, Higher threshold = less terrain
    // The 'noise_scale' factor here effectively scales the gradient.
    var distance = (normalized_density - settings.surface_threshold) / noise_scale;

    // Add vertical constraint: Smoothly blend with a plane at max_terrain_height
    // Use smooth_max to ensure the terrain doesn't exceed this height.
    // The smoothness parameter (0.6) controls how sharp the transition is.
    distance = smooth_max(distance, p.y - settings.max_terrain_height, 0.6);

    // TODO: Could add more features here (caves, overhangs, etc.)
    // Example: distance = smooth_min(distance, length(p.xz) - 100.0, 10.0); // Add a cylinder cut

    return distance;
}

// Calculates the gradient of the terrain SDF at point 'p'.
// The gradient points outwards from the surface and its length approximates 1.
fn terrain_sdf_gradient(p: vec3f) -> vec3f {
    // Use central differences for gradient calculation
    let eps = 0.1; // Small offset for sampling (adjust based on feature size)
    let grad_x = terrain_sdf(vec3f(p.x + eps, p.y, p.z)) - terrain_sdf(vec3f(p.x - eps, p.y, p.z));
    let grad_y = terrain_sdf(vec3f(p.x, p.y + eps, p.z)) - terrain_sdf(vec3f(p.x, p.y - eps, p.z));
    let grad_z = terrain_sdf(vec3f(p.x, p.y, p.z + eps)) - terrain_sdf(vec3f(p.x, p.y, p.z - eps));

    // Normalize the gradient vector (direction is important for normals)
    return normalize(vec3f(grad_x, grad_y, grad_z));

    // Original code used one-sided differences, which can be less accurate:
    // let e = vec2f(0.0, 0.1); // Note: y component is epsilon
    // return (terrain_sdf(p) - vec3f(
    //     terrain_sdf(p - e.yxx), // Sample p.x - eps
    //     terrain_sdf(p - e.xyx), // Sample p.y - eps
    //     terrain_sdf(p - e.xxy)  // Sample p.z - eps
    // )) / e.y; // Divide by epsilon (gradient approximation)
    // This should be normalized too.
}

// Calculates the center position of the voxel containing point 'p'.
fn get_voxel_center(p: vec3f, voxel_s: f32) -> vec3f {
    return (floor(p / voxel_s) + 0.5) * voxel_s;
}

// Ray marches through the voxel grid / SDF to find the first intersection with the terrain.
// Uses voxel grid traversal (DDA-like) as an acceleration structure for the SDF sampling.
fn trace_voxel_terrain(ray: Ray, max_dist: f32) -> VoxelHitInfo {
    let voxel_s = settings.voxel_size;
    let voxel_diag = voxel_s * sqrt(3.0); // Diagonal length of a voxel
    let ray_march_safety_margin = 1.0; // How close to zero SDF needs to be before switching to voxel mode

    // Precompute ray properties for grid traversal
    let inv_dir = 1.0 / ray.direction;
    let sign_dir = sign(ray.direction);
    let abs_inv_dir = abs(inv_dir);

    var t: f32 = 0.0; // Current distance along the ray
    var current_voxel_pos = get_voxel_center(ray.origin, voxel_s); // Voxel containing ray origin

    var in_voxel_mode = false;  // Are we currently stepping through individual voxels?
    var steps_in_current_voxel = 0; // Counter to detect escaping empty voxels
    var last_voxel_face_normal = vec3f(0.0); // Normal of the face we entered the current voxel through

    for (var i: i32 = 0; i < settings.max_ray_steps; i = i + 1) {
        let current_pos = ray.origin + ray.direction * t;

        // Determine the position to sample the SDF
        // If in voxel mode, sample at the *center* of the current voxel for stability.
        // If in SDF mode, sample at the current ray position.
        let sdf_sample_pos = select(current_pos, current_voxel_pos, in_voxel_mode);
        let dist = terrain_sdf(sdf_sample_pos);

        // Check termination conditions
        if (t >= max_dist || (ray.direction.y > 0.0 && current_pos.y > settings.max_terrain_height + 10.0)) {
           // Ray exceeded max distance or went too high above terrain limit
            return VoxelHitInfo(false, max_dist, vec3f(0.0), vec3f(0.0), i);
        }

        // --- State Machine: SDF Marching vs Voxel Traversal ---
        if (!in_voxel_mode) {
            // --- SDF Marching Mode ---
            if (dist < voxel_diag * ray_march_safety_margin) {
                // Close enough to potentially hit a voxel boundary soon.
                // Switch to voxel mode. Find the voxel containing the point slightly *before*
                // the estimated hit distance to ensure we don't skip the hit voxel.
                let potential_hit_pos = ray.origin + ray.direction * max(0.0, t + dist - voxel_diag * 0.5);
                current_voxel_pos = get_voxel_center(potential_hit_pos, voxel_s);
                in_voxel_mode = true;
                steps_in_current_voxel = 0;
                // Keep the current 't' value, the voxel step will advance it.
                continue; // Immediately switch to voxel logic in the next iteration
            } else {
                // Step forward by the SDF distance (Sphere tracing)
                t += dist;
                // Optional: Add a small step if dist is very small to avoid getting stuck
                // t += max(dist, settings.ray_march_epsilon * 0.1);
            }
        } else {
            // --- Voxel Traversal Mode ---

            // Calculate distances to the next voxel boundaries along each axis
            // Formula: t = (voxel_boundary - ray_origin) / ray_direction
            // Voxel boundary = current_voxel_pos.axis +/- voxel_s * 0.5
            // Relative pos = current_voxel_pos - current_pos = current_voxel_pos - (ray.origin + ray.direction * t)
            //                = (current_voxel_pos - ray.origin) - ray.direction * t
            // t_to_boundary = ((current_voxel_pos +/- 0.5*voxel_s) - ray.origin) / ray_direction
            // This can be simplified using precomputed values:
            let t_to_boundaries = (current_voxel_pos - ray.origin + sign_dir * voxel_s * 0.5) * inv_dir;

            // Find the smallest positive distance to the next boundary
            var t_next = min(min(t_to_boundaries.x, t_to_boundaries.y), t_to_boundaries.z);

            // Determine which axis boundary was hit first
            var step_normal = vec3f(0.0);
            if (abs(t_next - t_to_boundaries.x) < EPSILON) {
                step_normal = vec3f(-sign_dir.x, 0.0, 0.0);
            } else if (abs(t_next - t_to_boundaries.y) < EPSILON) {
                step_normal = vec3f(0.0, -sign_dir.y, 0.0);
            } else {
                step_normal = vec3f(0.0, 0.0, -sign_dir.z);
            }

            // Check if the SDF indicates a hit *within* the current voxel
            if (dist < 0.0) {
                // Hit detected! The surface is inside this voxel.
                // The normal is the normal of the face we *entered* through.
                // The hit position 't' is the distance to the *entry* of this voxel.
                // This provides a voxelized hit result.
                return VoxelHitInfo(true, t, last_voxel_face_normal, current_voxel_pos, i);
            }

            // Check if we are likely exiting the region of interest
            // If SDF is large and we've taken a few steps inside voxel mode,
            // it's safe to assume we missed the surface here. Switch back to SDF mode.
            let escape_threshold = voxel_diag * 1.5; // If SDF is larger than ~1.5 voxels
            if (dist > escape_threshold && steps_in_current_voxel > 2) {
                in_voxel_mode = false;
                // Advance 't' past the current voxel to avoid immediately re-entering it
                t = t_next + settings.ray_march_epsilon;
                continue; // Switch to SDF logic in next iteration
            }

            // No hit yet, step to the next voxel
            current_voxel_pos += sign_dir * step_normal * voxel_s; // This logic seems wrong, should use sign(step_normal) or just step_normal?
                                                                 // Original: vpos += nrd * s; where nrd = sign_dir on one axis.
                                                                 // Corrected: Use the axis determined by t_next
            if (abs(t_next - t_to_boundaries.x) < EPSILON) {
                 current_voxel_pos.x += sign_dir.x * voxel_s;
            } else if (abs(t_next - t_to_boundaries.y) < EPSILON) {
                 current_voxel_pos.y += sign_dir.y * voxel_s;
            } else {
                 current_voxel_pos.z += sign_dir.z * voxel_s;
            }

            last_voxel_face_normal = step_normal; // Store the normal of the face we just crossed
            t = t_next + settings.ray_march_epsilon; // Advance t slightly past the boundary
            steps_in_current_voxel += 1;

        } // end if (!in_voxel_mode) else
    } // end for loop

    // Max steps reached without a hit
    return VoxelHitInfo(false, max_dist, vec3f(0.0), vec3f(0.0), settings.max_ray_steps);
}

//--------------------------------------------------------------------------------------
// Texturing and Shading
//--------------------------------------------------------------------------------------

// Applies triplanar texture mapping using weighted contributions from XY, YZ, XZ planes.
// 'tex_index' selects which texture to sample (hardcoded for now).
// 'lod' specifies the mipmap level.
fn sample_triplanar_lod(
    world_pos: vec3f,   // Position to sample texture at
    world_normal: vec3f,// Normal vector at the position
    blend_sharpness: f32,// Controls how quickly the blend transitions between axes (higher = sharper)
    tex_index: i32,     // 0: noise_rgb, 2: grain, 3: dirt (based on original code)
    lod: f32            // Mipmap level of detail
) -> vec3f {

    // Calculate blend weights based on the normal vector components
    // Power emphasizes the dominant axis for sharper transitions
    let weights = pow(abs(world_normal), vec3f(blend_sharpness));
    // Normalize weights so they sum to 1
    let blend_weights = weights / dot(weights, vec3f(1.0));

    // Sample textures from the three planes
    var color = vec3f(0.0);
    if (tex_index == 0) { // noise_texture_rgb
        color += textureSampleLevel(noise_texture_rgb, texture_sampler, world_pos.yz, lod).rgb * blend_weights.x; // YZ plane (X normal)
        color += textureSampleLevel(noise_texture_rgb, texture_sampler, world_pos.xz, lod).rgb * blend_weights.y; // XZ plane (Y normal)
        color += textureSampleLevel(noise_texture_rgb, texture_sampler, world_pos.xy, lod).rgb * blend_weights.z; // XY plane (Z normal)
    } else if (tex_index == 2) { // grain_texture
        color += textureSampleLevel(grain_texture, texture_sampler, world_pos.yz, lod).rgb * blend_weights.x;
        color += textureSampleLevel(grain_texture, texture_sampler, world_pos.xz, lod).rgb * blend_weights.y;
        color += textureSampleLevel(grain_texture, texture_sampler, world_pos.xy, lod).rgb * blend_weights.z;
    } else if (tex_index == 3) { // dirt_texture
        color += textureSampleLevel(dirt_texture, texture_sampler, world_pos.yz, lod).rgb * blend_weights.x;
        color += textureSampleLevel(dirt_texture, texture_sampler, world_pos.xz, lod).rgb * blend_weights.y;
        color += textureSampleLevel(dirt_texture, texture_sampler, world_pos.xy, lod).rgb * blend_weights.z;
    }
    // Can add more texture indices here...

    return color;
}

// Samples triplanar texture mapping at LOD 0 (base mipmap level).
// Convenience function calling sample_triplanar_lod with lod = 0.0.
fn sample_triplanar(
    world_pos: vec3f, world_normal: vec3f, blend_sharpness: f32, tex_index: i32
) -> vec3f {
    // Calculate blend weights
    let weights = pow(abs(world_normal), vec3f(blend_sharpness));
    let blend_weights = weights / dot(weights, vec3f(1.0));

    // Sample textures (using textureSample for LOD 0)
    var color = vec3f(0.0);
     if (tex_index == 0) { // noise_texture_rgb
        color += textureSample(noise_texture_rgb, texture_sampler, world_pos.yz).rgb * blend_weights.x;
        color += textureSample(noise_texture_rgb, texture_sampler, world_pos.xz).rgb * blend_weights.y;
        color += textureSample(noise_texture_rgb, texture_sampler, world_pos.xy).rgb * blend_weights.z;
    } else if (tex_index == 2) { // grain_texture
        color += textureSample(grain_texture, texture_sampler, world_pos.yz).rgb * blend_weights.x;
        color += textureSample(grain_texture, texture_sampler, world_pos.xz).rgb * blend_weights.y;
        color += textureSample(grain_texture, texture_sampler, world_pos.xy).rgb * blend_weights.z;
    } else if (tex_index == 3) { // dirt_texture
        color += textureSample(dirt_texture, texture_sampler, world_pos.yz).rgb * blend_weights.x;
        color += textureSample(dirt_texture, texture_sampler, world_pos.xz).rgb * blend_weights.y;
        color += textureSample(dirt_texture, texture_sampler, world_pos.xy).rgb * blend_weights.z;
    }
    return color;
}


// Determines biome weights (e.g., snow, desert) based on position using texture lookups.
// Returns vec2f(desert_weight, snow_weight).
fn get_biome_weights(world_pos: vec3f) -> vec2f {
    // Sample biome maps (using channels of dirt_texture in the original code)
    // Scales control the size/frequency of biome regions
    let snow_map_value = textureSample(dirt_texture, texture_sampler, world_pos.xz * 0.00015).r; // Red channel for snow?
    let desert_map_value = textureSample(dirt_texture, texture_sampler, vec2f(0.55) - world_pos.zx * 0.00008).g; // Green channel for desert? (Note: uses .zx)

    // Apply smoothstep to create transition zones between biomes
    // Adjust thresholds (0.67, 0.672, 0.695, 0.7) to control biome boundaries and blend width
    let desert_weight = smoothstep(0.67, 0.672, desert_map_value);
    let snow_weight = smoothstep(0.695, 0.7, snow_map_value);

    return vec2f(desert_weight, snow_weight); // x = desert, y = snow
}

// Calculates the final albedo (base color) of the terrain at a given position and normal.
// Blends different textures and colors based on height, normal, and biome.
fn get_terrain_albedo(
    world_pos: vec3f,       // Position on the terrain surface
    surface_gradient: vec3f,// Gradient of the SDF (approximates normal)
    lod: f32                // Level of detail for texture sampling
) -> vec3f {

    let blend_sharpness: f32 = 4.0; // Sharpness for triplanar blending

    // Base rock/dirt texture (using 'grain' texture index 2)
    // Apply (1 - tex) * (1 - tex) = (1-tex)^2 for color transformation (common technique)
    var albedo = vec3f(1.0) - sample_triplanar_lod(world_pos * 0.08, surface_gradient, blend_sharpness, 2, lod);
    albedo *= albedo; // Square the result (adjusts color range/contrast)

    // Secondary dirt/grass texture? (using 'dirt' texture index 3)
    var albedo2 = vec3f(1.0) - sample_triplanar_lod(world_pos * 0.08, surface_gradient, blend_sharpness, 3, lod);
    albedo2 *= albedo2;

    // Large scale variation/mask (using 'noise_rgb' texture index 0)
    // Scale (0.0005) defines very large features
    let large_scale_mask = sample_triplanar_lod(world_pos * 0.0005, surface_gradient, blend_sharpness, 0, 0.0).r; // Use red channel

    // --- Blending based on Height and Slope ---
    // Weight based on how close to the max water influenced height (for blending near water)
    let water_influence_weight = smoothstep(settings.max_water_influenced_height, settings.max_water_influenced_height + 0.5, world_pos.y);
    // Weight based on slope (gradient.y close to 1 means flat)
    let flatness_weight = smoothstep(0.3, 0.7, surface_gradient.y);

    // Base albedo color tint
    albedo = albedo * 0.95 * vec3f(1.0, 0.7, 0.65) + 0.05; // Apply tint and lift blacks slightly

    // Blend towards albedo2 (grass?) on flatter areas near water level
    albedo = mix(albedo, albedo2 * vec3f(0.55, 1.0, 0.1), flatness_weight * water_influence_weight);

    // Blend towards albedo2 (using smoothstep for contrast) on steeper slopes based on large scale mask
    albedo = mix(albedo, smoothstep(vec3f(0.0), vec3f(1.0), albedo2), smoothstep(0.3, 0.25, large_scale_mask) * (1.0 - flatness_weight));

    // --- Biome Blending ---
    let biome = get_biome_weights(world_pos); // biome.x = desert, biome.y = snow

    // Define snow color (slightly blueish base, brighter on flat areas near water)
    var snow_color = albedo2 * 0.8 + 0.2 * vec3f(0.25, 0.5, 1.0);
    snow_color = mix(snow_color, vec3f(0.85, 0.95, 1.0), flatness_weight * water_influence_weight * 0.5);

    // Blend towards desert color (brighter rock/sand) based on desert weight
    let desert_color = clamp(vec3f(1.0, 0.95, 0.9) - albedo2 * 0.65, vec3f(0.0), vec3f(1.0));
    albedo = mix(albedo, desert_color, biome.x);

    // Blend towards snow color based on snow weight (applied after desert)
    albedo = mix(albedo, snow_color * 2.0, biome.y); // Snow brightens the result significantly

    // --- Darkening based on Height/Mask (Wetness/Soil effect?) ---
    // Define base darkening color, slightly adjusted by biome
    var dark_color_tint = vec3f(0.8, 0.55, 0.35); // Default brown/earthy
    dark_color_tint = mix(dark_color_tint, vec3f(0.8, 0.65, 0.4), biome.x); // More yellow/orange in desert
    dark_color_tint = mix(dark_color_tint, vec3f(0.2, 0.6, 0.8), biome.y); // More blue/cool in snow

    // Apply darkening effect below water influence height, modulated by large scale mask and biome
    let darkening_factor = (1.0 - water_influence_weight) * mix(1.0 - smoothstep(0.3, 0.25, large_scale_mask), 1.0, max(biome.x, biome.y));
    albedo = mix(albedo, albedo * dark_color_tint, darkening_factor);

    return albedo;
}

// Calculates the shaded color of the terrain surface.
// Includes diffuse lighting, simple shadowing, ambient occlusion approximation, and height-based darkening.
fn shade_terrain_surface(
    hit_pos: vec3f,       // World position of the hit point on the terrain
    voxel_face_normal: vec3f, // Normal of the voxel face hit (from trace)
    hit_info: VoxelHitInfo, // Full hit information from trace
    light_direction: vec3f, // Direction towards the main light source
    lod: f32              // Level of detail for texturing
) -> vec3f {

    // Use the center of the hit voxel for gradient/albedo calculation for stability
    let voxel_center_pos = hit_info.voxel_pos;
    // Calculate the surface gradient at the voxel center
    let surface_gradient = terrain_sdf_gradient(voxel_center_pos); // More accurate normal

    // Use the voxel face normal for diffuse lighting calculation (gives faceted look)
    let lighting_normal = voxel_face_normal;

    // Calculate diffuse lighting term (Lambertian)
    var diffuse_term = max(dot(lighting_normal, light_direction), 0.0);

    // Simple shadow check: trace a short ray towards the light source
    if (diffuse_term > EPSILON) {
        let shadow_ray = Ray(hit_pos + lighting_normal * settings.ray_march_epsilon * 10.0, light_direction);
        // Check for hit within a short distance (e.g., 12 units)
        let shadow_hit = trace_voxel_terrain(shadow_ray, 12.0);
        if (shadow_hit.is_hit) {
            diffuse_term = 0.0; // In shadow
        }
    }

    // Get the base albedo color for this point
    let albedo = get_terrain_albedo(voxel_center_pos, surface_gradient, lod);

    // Ambient Occlusion approximation based on SDF value relative to gradient magnitude
    // Small SDF value relative to gradient length suggests proximity to other surfaces.
    let sdf_value = terrain_sdf(hit_pos); // Sample SDF at the actual hit position
    let gradient_magnitude = length(terrain_sdf_gradient(hit_pos)); // Gradient magnitude should ideally be ~1
    let ao_factor = smoothstep(-0.08, 0.04, sdf_value / max(gradient_magnitude, EPSILON));

    // Height-based Ambient Occlusion / Darkening (for areas near/below water level)
    let height_ao_factor = smoothstep(settings.water_level - 12.0, settings.water_level, hit_pos.y);

    // Combine lighting components
    // Modulate light color by diffuse term and ambient light approximation (0.6 * diffuse + 0.4)
    var final_color = albedo * (diffuse_term * 0.6 + 0.4) * settings.light_color.rgb;

    // Apply AO terms
    // Modulate by AO factor (0.6 * ao + 0.4 adds some ambient light even in occluded areas)
    final_color *= (ao_factor * 0.6 + 0.4);
    // Modulate by height-based AO
    final_color *= (height_ao_factor * 0.6 + 0.4);

    return final_color;
}

// Simplified terrain shading specifically for water reflections.
// Uses absolute normal dot product for a slightly different look.
fn shade_terrain_for_water_reflection(
    hit_pos: vec3f,
    voxel_face_normal: vec3f,
    hit_info: VoxelHitInfo,
    light_direction: vec3f,
    lod: f32
) -> vec3f {
    let voxel_center_pos = hit_info.voxel_pos;
    let surface_gradient = terrain_sdf_gradient(voxel_center_pos);
    let lighting_normal = voxel_face_normal;

    let diffuse_term = max(dot(lighting_normal, light_direction), 0.0);
    // No shadow check for reflections (can be expensive / less important)

    let albedo = get_terrain_albedo(voxel_center_pos, surface_gradient, lod);

    let sdf_value = terrain_sdf(hit_pos);
    let gradient_magnitude = length(terrain_sdf_gradient(hit_pos));
    let ao_factor = smoothstep(-0.08, 0.04, sdf_value / max(gradient_magnitude, EPSILON));
    let height_ao_factor = smoothstep(settings.water_level - 12.0, settings.water_level, hit_pos.y);

    // --- Differences from shade_terrain_surface ---
    // 1. Modulate albedo by dot(abs(normal), ...) - tends to brighten glancing angles slightly?
    var final_color = albedo * dot(abs(lighting_normal), vec3f(0.8, 1.0, 0.9));
    // 2. No shadow check is performed.

    final_color *= (diffuse_term * 0.6 + 0.4) * settings.light_color.rgb;
    final_color *= (ao_factor * 0.6 + 0.4);
    final_color *= (height_ao_factor * 0.6 + 0.4);

    return final_color;
}


//--------------------------------------------------------------------------------------
// Water Rendering Functions
//--------------------------------------------------------------------------------------

// Calculates water surface height displacement based on noise.
// Currently returns 0, needs implementation if wave displacement is desired.
fn get_water_wave_displacement(uv: vec2f) -> f32 {
    // Placeholder: No wave displacement
    // To implement: Sample a noise texture (e.g., grain_texture) with time animation
    // let wave1 = textureSample(grain_texture, texture_sampler, uv + camera.time * 0.01).r;
    // let wave2 = textureSample(grain_texture, texture_sampler, uv * 0.5 - camera.time * 0.005).r;
    // return (wave1 + wave2) * 0.5 - 0.25; // Example blending and offset
    return 0.0;
}

// Calculates the normal vector of the water surface, including wave displacement.
fn get_water_normal(world_pos: vec3f, world_time: f32) -> vec3f {
    let tangent_epsilon = 0.01; // Small offset for calculating tangents
    let normal_strength = 1500.0; // Multiplier for wave normal strength

    // Calculate UV coordinates for wave sampling (XZ plane, animated)
    let wave_uv_offset = vec2f(1.0, 0.8) * world_time * 0.01; // Time-based animation offset
    let wave_uv_scale = 0.08; // Scale of wave features
    let wave_uv = world_pos.xz * wave_uv_scale + wave_uv_offset;

    // Sample height at current point and slightly offset points for tangents
    let height_center = get_water_wave_displacement(wave_uv);
    let height_dx = get_water_wave_displacement(wave_uv + vec2f(tangent_epsilon, 0.0));
    let height_dy = get_water_wave_displacement(wave_uv + vec2f(0.0, tangent_epsilon));

    // Calculate normal using cross product of tangents (approximated)
    // TangentX = (epsilon, 0, height_dx - height_center)
    // TangentZ = (0, epsilon, height_dy - height_center)
    // Normal = cross(TangentZ, TangentX) - simplified assuming small epsilon
    let normal = vec3f(
        (height_center - height_dx) * normal_strength, // dH/dX scaled
        tangent_epsilon,                               // Up vector component
        (height_center - height_dy) * normal_strength  // dH/dZ scaled
    );

    return normalize(normal);
}

// Calculates foam visibility based on proximity to terrain and wave shape.
fn get_water_foam_factor(
    water_surface_pos: vec3f, // Position on the water surface
    water_normal: vec3f,      // Normal of the water surface at that point
    world_time: f32
) -> f32 {
    // Sample terrain SDF near the water surface, slightly perturbed by water normal
    // This simulates waves interacting with the shore
    let foam_sample_offset = water_normal * vec3f(1.0, 0.0, 1.0) * 0.2; // Offset in XZ based on normal
    let foam_sample_pos = water_surface_pos + foam_sample_offset;

    // Get distance to terrain at the perturbed position
    let terrain_dist = terrain_sdf(foam_sample_pos);
    // Optionally normalize by gradient magnitude for more consistent results
    // let terrain_grad_mag = length(terrain_sdf_gradient(foam_sample_pos));
    // let normalized_terrain_dist = terrain_dist / max(terrain_grad_mag, EPSILON);

    // Use sine wave based on distance and time to create animated foam patterns
    let foam_wave = sin((terrain_dist - world_time * 0.03) * 60.0);

    // Sample water wave height again (or pass it in) - use for foam modulation
    let wave_uv_offset = vec2f(1.0, 0.8) * world_time * 0.01;
    let wave_uv_scale = 0.08;
    let wave_uv = water_surface_pos.xz * wave_uv_scale + wave_uv_offset;
    let wave_height = get_water_wave_displacement(wave_uv); // Re-calculating, could optimize

    // Combine factors: Foam appears where terrain is close (small terrain_dist),
    // modulated by the sine wave and water wave height.
    // Adjust thresholds (0.22, 0.0) and multipliers (0.03, 0.12) to control foam appearance.
    let foam_mask = smoothstep(
        0.22, 0.0, // Fade out foam as distance increases
        terrain_dist + foam_wave * 0.03 + (wave_height - 0.5) * 0.12 // Combined distance metric
    );

    return foam_mask;
}

//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    // Generate vertices for a full-screen triangle (or quad)
    // Outputs clip space position and texture coordinates spanning the screen.

    // Using 2 triangles (4 vertices) approach
    // Map vertex index to (-1,-1) to (+1,+1) clip space coordinates
    let positions = array<vec2f, 4>(
        vec2f(-1.0, -1.0), // Bottom Left
        vec2f( 1.0, -1.0), // Bottom Right
        vec2f(-1.0,  1.0), // Top Left
        vec2f( 1.0,  1.0)  // Top Right
    );
     // Corresponding texture coordinates (0,0) to (1,1)
    let tex_coords = array<vec2f, 4>(
        vec2f(0.0, 1.0), // Bottom Left UV (adjust Y if needed)
        vec2f(1.0, 1.0), // Bottom Right UV
        vec2f(0.0, 0.0), // Top Left UV
        vec2f(1.0, 0.0)  // Top Right UV
    );

    // Using triangle strip (requires index buffer or logic for 3 vertices)
    // Example for full screen triangle:
    // let x = f32((input.vertex_index & 1u) << 2u) - 1.0; // -1, 3, -1
    // let y = f32((input.vertex_index & 2u) << 1u) - 1.0; // -1, -1, 3
    // out.clip_position = vec4f(x, y, 0.0, 1.0);
    // out.tex_coord = vec2f(x * 0.5 + 0.5, y * -0.5 + 0.5); // Adjust for UV space

    var output: VertexOutput;
    let vertex_id = input.vertex_index % 4u; // Ensure index is within bounds for array lookup
    output.clip_position = vec4f(positions[vertex_id], 0.0, 1.0); // Z=0, W=1 for near plane
    output.tex_coord = tex_coords[vertex_id];

    return output;
}

//--------------------------------------------------------------------------------------
// Fragment Shader
//--------------------------------------------------------------------------------------

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {

    //--- 1. Ray Setup ---
    let ray_origin = camera.camera_position;

    // Calculate world space ray direction from screen coordinates
    // Convert texture coordinate (0-1) to Normalized Device Coordinates (NDC) (-1 to 1)
    var ndc = vec4f(input.tex_coord * 2.0 - 1.0, 1.0, 1.0); // Z=1 for far plane point
    // Flip Y if necessary (common difference between screen coords and NDC)
    ndc.y = -ndc.y;

    // Unproject NDC coordinate to world space
    let world_pos_far = camera.inv_view_proj * ndc;
    // Direction is from camera towards the far plane point
    let ray_direction = normalize(world_pos_far.xyz / world_pos_far.w - ray_origin);

    let primary_ray = Ray(ray_origin, ray_direction);

    // Get normalized sun direction from settings
    let sun_direction = normalize(settings.light_direction.xyz);

    //--- 2. Debug Visualizations (Optional Early Exit) ---
    if (settings.visualize_distance_field != 0) {
        let sample_pos = ray_origin + ray_direction * 10.0; // Sample SDF 10 units away
        let dist = terrain_sdf(sample_pos);
        // Map distance to color (e.g., gray scale)
        return vec4f(vec3f(dist * 0.1 + 0.5), 1.0);
    }

    //--- 3. Primary Ray Intersection (Terrain or Sky/Clouds) ---
    let hit_info = trace_voxel_terrain(primary_ray, settings.max_distance);

    var final_color = vec3f(0.0);
    var distance_to_surface = settings.max_distance; // Initialize with max distance

    if (hit_info.is_hit) {
        //--- 4a. Terrain Hit: Calculate Surface Shading ---
        distance_to_surface = hit_info.t;
        let hit_pos = ray_origin + ray_direction * distance_to_surface;

        // Calculate LOD based on distance to the *voxel center* (more stable)
        let distance_for_lod = distance(ray_origin, hit_info.voxel_pos);
        // Adjust LOD calculation: log2(dist) - bias. Clamp to valid range (e.g., 0-6)
        let lod = clamp(log2(max(distance_for_lod, 1.0)) - 2.0, 0.0, 6.0);

        final_color = shade_terrain_surface(
            hit_pos,
            hit_info.normal, // Voxel face normal
            hit_info,
            sun_direction,
            lod
        );

        // Debug Overrides for Hit
        if (settings.show_normals != 0) {
             // Visualize voxel face normal (remap from -1..1 to 0..1)
            final_color = hit_info.normal * 0.5 + 0.5;
        }
        if (settings.show_ray_steps != 0) {
             // Visualize number of ray steps taken (normalized)
            final_color = vec3f(f32(hit_info.steps) / f32(settings.max_ray_steps));
        }

    } else {
        //--- 4b. No Terrain Hit: Render Sky and Clouds ---
        distance_to_surface = settings.max_distance; // Ray hit the skybox distance

        // Render volumetric clouds along the ray
        // Use 'false' for high quality clouds/sampling
        final_color = render_clouds_along_ray(primary_ray, sun_direction, camera.time, false);

         // Debug Overrides for Miss
         if (settings.show_normals != 0) { final_color = vec3f(0.0, 0.0, 0.5); } // Blue for sky normals
         if (settings.show_ray_steps != 0) { final_color = vec3f(1.0); } // White for max steps

    } // End if (hit_info.is_hit)

    //--- 5. Water Plane Intersection and Shading ---
    // Check if the ray intersects the water plane *before* hitting terrain/skybox,
    // or if the camera starts below water.
    let water_plane_y = settings.water_level;
    let ray_origin_y = ray_origin.y;
    let ray_dir_y = ray_direction.y;

    var intersects_water_plane = false;
    var dist_to_water_plane: f32 = -1.0;

    // Calculate distance to water plane (t = -(origin.y - plane_y) / direction.y)
    if (abs(ray_dir_y) > EPSILON) { // Avoid division by zero for horizontal rays
        dist_to_water_plane = -(ray_origin_y - water_plane_y) / ray_dir_y;
        // Check if intersection is valid (in front of ray, before terrain hit)
        if (dist_to_water_plane > 0.0 && dist_to_water_plane < distance_to_surface) {
            intersects_water_plane = true;
        }
    }

    // Handle cases where camera is below water
    let camera_below_water = (ray_origin_y < water_plane_y);
    if (camera_below_water && dist_to_water_plane < 0.0) {
        // Ray starts below water and points away from the surface - treat as infinite distance
         dist_to_water_plane = settings.max_distance;
    }

    // If the ray hits the water surface (from above or below before hitting terrain)
    if (intersects_water_plane || camera_below_water) {

        let water_hit_dist = select(dist_to_water_plane, 0.0, camera_below_water); // Use 0 dist if starting below water
        let water_hit_pos = ray_origin + ray_direction * water_hit_dist;

        // Determine water color based on biome underneath
        let biome_at_water = get_biome_weights(water_hit_pos);
        var water_color = vec3f(0.3, 0.8, 1.0); // Base blue water
        water_color = mix(water_color, vec3f(0.4, 0.9, 0.8), biome_at_water.x); // Greener in desert?
        water_color = mix(water_color, vec3f(0.1, 0.7, 0.9), biome_at_water.y); // Darker/cyan in snow?
        // Define water absorption coefficients (approximated by 1 - color)
        let water_absorption_coeff = vec3f(0.1, 0.7, 0.9); // Example absorption

        // Calculate water surface normal with waves
        let water_normal = get_water_normal(water_hit_pos, camera.time);

        // --- Calculate Reflection ---
        var reflection_color = vec3f(0.0);
        // Only calculate reflection if viewing from above water
        if (!camera_below_water) {
            let reflection_vector = reflect(ray_direction, water_normal);
            let reflection_ray = Ray(water_hit_pos + water_normal * EPSILON * 10.0, reflection_vector); // Offset start slightly

            // Trace reflection ray against terrain
            let reflect_hit_info = trace_voxel_terrain(reflection_ray, 15.0); // Shorter max distance for reflections

            if (reflect_hit_info.is_hit) {
                // Shade the reflected terrain point (using simplified shader)
                 let reflect_hit_pos = reflection_ray.origin + reflection_ray.direction * reflect_hit_info.t;
                 let distance_for_lod = distance(water_hit_pos, reflect_hit_info.voxel_pos); // LOD based on reflection path
                 let lod = clamp(log2(max(distance_for_lod, 1.0)) - 2.0, 0.0, 6.0);

                reflection_color = shade_terrain_for_water_reflection(
                    reflect_hit_pos,
                    reflect_hit_info.normal,
                    reflect_hit_info,
                    sun_direction,
                    lod
                );
            } else {
                // Reflected ray hit the sky/clouds
                reflection_color = render_clouds_along_ray(reflection_ray, sun_direction, camera.time, true); // Use simplified clouds for reflection
            }
        } // end if (!camera_below_water) for reflection

        // --- Calculate Fresnel ---
        // Fresnel factor determines blend between reflection and refraction/absorption
        let fresnel_r0 = 0.03; // Reflectance at normal incidence (air-water interface ~0.02-0.04)
        // Use Schlick's approximation: R = R0 + (1 - R0) * (1 - cos(theta))^5
        // cos(theta) is dot product between view direction and surface normal
        var fresnel_factor = fresnel_r0 + (1.0 - fresnel_r0) * pow(1.0 - abs(dot(ray_direction, water_normal)), 5.0);

        // Force fresnel to 0 if viewing from underwater towards surface (no reflection in this case)
        if (camera_below_water) {
             fresnel_factor = 0.0;
        }

        // --- Calculate Refraction / Absorption ---
        // Calculate distance traveled underwater
        // If hit terrain/sky after water: distance = surface_hit_t - water_hit_t
        // If camera below water: distance = surface_hit_t
        let distance_underwater = select(
            distance_to_surface - water_hit_dist, // Camera above water
            distance_to_surface,                  // Camera below water
            camera_below_water
        );

        // Apply Beer's law for light absorption through water
        // Transmittance = exp(-absorption_coeff * distance)
        let water_transmittance = exp(-distance_underwater * (vec3f(1.0) - water_absorption_coeff) * 0.1); // Adjust absorption strength (0.1)

        // Color seen through water is the original surface/sky color attenuated by absorption
        let refracted_color = final_color * water_transmittance;


        // --- Combine Reflection and Refraction using Fresnel ---
        // If hit water before terrain/sky: Blend reflection and refraction
        if (intersects_water_plane) {
            // Add specular highlight on water surface
            let reflected_sun = reflect(-sun_direction, water_normal);
            let specular_term = pow(max(dot(reflected_sun, -ray_direction), 0.0), 50.0) * settings.light_color.rgb;

            final_color = mix(refracted_color, reflection_color + specular_term, fresnel_factor);

            // --- Add Foam ---
            let foam_factor = get_water_foam_factor(water_hit_pos, water_normal, camera.time);
            // Blend towards white based on foam factor
            final_color = mix(final_color, vec3f(1.0), foam_factor * 0.4); // Adjust foam intensity (0.4)

        } else if (camera_below_water) {
            // If camera is below water, only show attenuated surface/sky color
            final_color = refracted_color;
            // Optionally add volumetric water scattering effect here
        }


    } // end if (intersects_water_plane || camera_below_water)

    //--- 6. Final Output ---
    // TODO: Add post-processing effects (fog, bloom, tone mapping) if needed

    return vec4f(final_color, 1.0); // Output final color with alpha = 1.0
}