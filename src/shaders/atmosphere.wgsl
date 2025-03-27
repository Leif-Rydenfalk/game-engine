// Constants
const PI: f32 = 3.14159265359;
const MAX: f32 = 10000.0;
const WORKGROUP_SIZE: u32 = 16u;

// Planet and atmosphere constants
const PLANET_POS: vec3f = vec3f(0.0, 0.0, 0.0);
const PLANET_RADIUS: f32 = 6371e3;
const ATMOS_RADIUS: f32 = 6471e3;

// Scattering coefficients
const RAY_BETA: vec3f = vec3f(5.5e-6, 13.0e-6, 22.4e-6);
const MIE_BETA: vec3f = vec3f(21e-6);
const AMBIENT_BETA: vec3f = vec3f(0.0);
const ABSORPTION_BETA: vec3f = vec3f(2.04e-5, 4.97e-5, 1.95e-6);
const G: f32 = 0.7;

// Height constants
const HEIGHT_RAY: f32 = 8e3;
const HEIGHT_MIE: f32 = 1.2e3;
const HEIGHT_ABSORPTION: f32 = 30e3;
const ABSORPTION_FALLOFF: f32 = 4e3;

// Sampling steps
const PRIMARY_STEPS: i32 = 32;
const LIGHT_STEPS: i32 = 8;

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

// Ray-sphere intersection
fn ray_sphere_intersect(start: vec3f, dir: vec3f, radius: f32) -> vec2f {
    let a = dot(dir, dir);
    let b = 2.0 * dot(dir, start);
    let c = dot(start, start) - (radius * radius);
    let d = (b * b) - 4.0 * a * c;
    
    if (d < 0.0) {
        return vec2f(1e5, -1e5);
    }
    
    return vec2f(
        (-b - sqrt(d)) / (2.0 * a),
        (-b + sqrt(d)) / (2.0 * a)
    );
}

// Atmospheric scattering calculation
fn calculate_scattering(
    start: vec3f,              // the start of the ray (camera position)
    dir: vec3f,                // the direction of the ray
    max_dist: f32,             // the maximum distance the ray can travel
    scene_color: vec3f,        // the color of the scene
    light_dir: vec3f,          // the direction of the light
    light_intensity: vec3f,    // brightness of the light
    planet_position: vec3f,    // position of the planet
    planet_radius: f32,        // radius of the planet
    atmo_radius: f32,          // radius of the atmosphere
    beta_ray: vec3f,           // rayleigh scattering coefficient
    beta_mie: vec3f,           // mie scattering coefficient
    beta_absorption: vec3f,    // absorption coefficient
    beta_ambient: vec3f,       // ambient scattering
    g: f32,                    // mie direction
    height_ray: f32,           // rayleigh scale height
    height_mie: f32,           // mie scale height
    height_absorption: f32,    // absorption height
    absorption_falloff: f32,   // absorption falloff
    steps_i: i32,              // primary steps
    steps_l: i32               // light steps
) -> vec3f {
    // Offset the camera position
    let start_offset = start - planet_position;
    
    // Calculate ray-sphere intersection with atmosphere
    let a = dot(dir, dir);
    let b = 2.0 * dot(dir, start_offset);
    let c = dot(start_offset, start_offset) - (atmo_radius * atmo_radius);
    let d = (b * b) - 4.0 * a * c;
    
    // If no intersection with atmosphere, return scene color
    if (d < 0.0) {
        return scene_color;
    }
    
    // Calculate ray length inside atmosphere
    let ray_length = vec2f(
        max((-b - sqrt(d)) / (2.0 * a), 0.0),
        min((-b + sqrt(d)) / (2.0 * a), max_dist)
    );
    
    // If the ray didn't hit the atmosphere, return scene color
    if (ray_length.x > ray_length.y) {
        return scene_color;
    }
    
    // Check if Mie glow should be allowed
    let allow_mie = max_dist > ray_length.y;
    
    // Adjust ray length based on max distance
    let adjusted_ray_length_y = min(ray_length.y, max_dist);
    let adjusted_ray_length_x = max(ray_length.x, 0.0);
    
    // Get step size
    let step_size_i = (adjusted_ray_length_y - adjusted_ray_length_x) / f32(steps_i);
    
    // Initial position along the ray
    var ray_pos_i = adjusted_ray_length_x + step_size_i * 0.5;
    
    // Initialize accumulators
    var total_ray = vec3f(0.0);
    var total_mie = vec3f(0.0);
    
    // Initialize optical depth
    var opt_i = vec3f(0.0);
    
    // Scale heights for rayleigh and mie
    let scale_height = vec2f(height_ray, height_mie);
    
    // Calculate phase functions
    let mu = dot(dir, light_dir);
    let mumu = mu * mu;
    let gg = g * g;
    let phase_ray = 3.0 / (50.2654824574) * (1.0 + mumu);
    
    // Use select function instead of ternary operator for allow_mie condition
    let phase_mie = select(
        0.0, 
        3.0 / (25.1327412287) * ((1.0 - gg) * (mumu + 1.0)) / 
        (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg)),
        allow_mie
    );
    
    // Sample the primary ray
    for (var i = 0; i < steps_i; i++) {
        // Calculate position along the ray
        let pos_i = start_offset + dir * ray_pos_i;
        
        // Calculate height above planet surface
        let height_i = length(pos_i) - planet_radius;
        
        // Calculate particle densities
        var density = vec3f(exp(-height_i / scale_height.x), exp(-height_i / scale_height.y), 0.0);
        
        // Calculate absorption - uses 1/(x²+1) function from the updated shader
        let denom = (height_absorption - height_i) / absorption_falloff;
        density.z = (1.0 / (denom * denom + 1.0)) * density.x;
        
        // Multiply by step size
        density *= step_size_i;
        
        // Add to optical depth
        opt_i += density;
        
        // Calculate ray to sun
        let a_light = dot(light_dir, light_dir);
        let b_light = 2.0 * dot(light_dir, pos_i);
        let c_light = dot(pos_i, pos_i) - (atmo_radius * atmo_radius);
        let d_light = (b_light * b_light) - 4.0 * a_light * c_light;
        
        // Calculate sun ray step size
        let step_size_l = (-b_light + sqrt(d_light)) / (2.0 * a_light * f32(steps_l));
        
        // Initial position along sun ray
        var ray_pos_l = step_size_l * 0.5;
        
        // Initialize optical depth for sun ray
        var opt_l = vec3f(0.0);
        
        // Sample the light ray
        for (var l = 0; l < steps_l; l++) {
            // Calculate position along sun ray
            let pos_l = pos_i + light_dir * ray_pos_l;
            
            // Calculate height above planet surface
            let height_l = length(pos_l) - planet_radius;
            
            // Calculate densities
            var density_l = vec3f(exp(-height_l / scale_height.x), exp(-height_l / scale_height.y), 0.0);
            
            // Calculate absorption
            let denom_l = (height_absorption - height_l) / absorption_falloff;
            density_l.z = (1.0 / (denom_l * denom_l + 1.0)) * density_l.x;
            
            // Multiply by step size
            density_l *= step_size_l;
            
            // Add to optical depth
            opt_l += density_l;
            
            // Increment position
            ray_pos_l += step_size_l;
        }
        
        // Calculate attenuation
        let attn = exp(
            -beta_ray * (opt_i.x + opt_l.x) - 
            beta_mie * (opt_i.y + opt_l.y) - 
            beta_absorption * (opt_i.z + opt_l.z)
        );
        
        // Accumulate scattered light
        total_ray += density.x * attn;
        total_mie += density.y * attn;
        
        // Increment position
        ray_pos_i += step_size_i;
    }
    
    // Calculate atmosphere opacity
    let opacity = exp(-(beta_mie * opt_i.y + beta_ray * opt_i.x + beta_absorption * opt_i.z));
    
    // Calculate final color
    return (
        phase_ray * beta_ray * total_ray +          // Rayleigh
        phase_mie * beta_mie * total_mie +          // Mie
        opt_i.x * beta_ambient                      // Ambient
    ) * light_intensity + scene_color * opacity;    // Apply to scene
}

// Render the scene (planet and sky)
fn render_scene(pos: vec3f, dir: vec3f, light_dir: vec3f) -> vec4f {
    // Initialize color and depth
    var color = vec4f(0.0, 0.0, 0.0, 1e12);
    
    // Add a sun
    let sun_dot = dot(dir, light_dir);
    let sun_color = select(vec3f(0.0), vec3f(3.0), sun_dot > 0.9998);
    color = vec4f(sun_color, color.w);
    
    // Intersect with planet
    let planet_intersect = ray_sphere_intersect(pos - PLANET_POS, dir, PLANET_RADIUS);
    
    // If ray hits planet
    if (0.0 < planet_intersect.y) {
        color.w = max(planet_intersect.x, 0.0);
        
        // Sample position
        let sample_pos = pos + (dir * planet_intersect.x) - PLANET_POS;
        
        // Surface normal
        let surface_normal = normalize(sample_pos);
        
        // Planet color (dark blue-green)
        color.xyz = vec3f(0.05, 0.1, 0.15);
        
        // Simple lighting using Lommel-Seeliger law as in the reference shader
        let N = surface_normal;
        let V = -dir;
        let L = light_dir;
        let dotNV = max(1e-6, dot(N, V));
        let dotNL = max(1e-6, dot(N, L));
        let shadow = dotNL / (dotNL + dotNV);
        
        // Apply shadow
        let shadowed_color = color.xyz * shadow;
        color = vec4f(shadowed_color, color.w);
    }
    
    return color;
}

// Compute shader entry point
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Get output dimensions
    let dimensions = textureDimensions(output_texture);
    
    // Check if within bounds
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }
    
    // Calculate normalized device coordinates
    let uv = vec2f(
        (f32(global_id.x) + 0.5) / f32(dimensions.x),
        (f32(global_id.y) + 0.5) / f32(dimensions.y)
    );
    
    // Calculate ray direction using camera (keeping original camera system)
    let ro = camera.camera_position;
    let ndc = vec4f(uv * 2.0 - 1.0, 1.0, 1.0);
    let world_pos = camera.inv_view_proj * ndc;
    let rd = normalize(world_pos.xyz / world_pos.w - ro);
    
    // Light direction (time-based, similar to the reference shader)
    let light_dir = normalize(vec3f(0.0, cos(camera.time * -0.125), sin(camera.time * -0.125)));
    
    // Get scene color and depth
    let scene = render_scene(ro, rd, light_dir);
    
    // Calculate atmospheric scattering
    let color = calculate_scattering(
        ro,                    // camera position
        rd,                    // ray direction
        scene.w,               // max distance
        scene.xyz,             // scene color
        light_dir,             // light direction
        vec3f(40.0),           // light intensity
        PLANET_POS,            // planet position
        PLANET_RADIUS,         // planet radius
        ATMOS_RADIUS,          // atmosphere radius
        RAY_BETA,              // rayleigh coefficient
        MIE_BETA,              // mie coefficient
        ABSORPTION_BETA,       // absorption coefficient
        AMBIENT_BETA,          // ambient coefficient
        G,                     // mie direction
        HEIGHT_RAY,            // rayleigh height
        HEIGHT_MIE,            // mie height
        HEIGHT_ABSORPTION,     // absorption height
        ABSORPTION_FALLOFF,    // absorption falloff
        PRIMARY_STEPS,         // primary steps
        LIGHT_STEPS            // light steps
    );
    
    // Store result
    textureStore(output_texture, global_id.xy, vec4f(color, 1.0));
}