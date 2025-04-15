// src/shaders/sky.wgsl

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var depth_texture: texture_2d<f32>; // Read R32Float depth
@group(1) @binding(1) var depth_sampler: sampler;       // Needed even for textureLoad
@group(2) @binding(0) var rgb_noise_texture: texture_2d<f32>;  // Corresponds to GLSL iChannel 1
@group(2) @binding(1) var gray_noise_texture: texture_2d<f32>; // Corresponds to GLSL iChannel 3
@group(2) @binding(2) var gray_cube_noise_texture: texture_3d<f32>; // Corresponds to GLSL iChannel 2
@group(2) @binding(3) var grain_texture: texture_2d<f32>;
@group(2) @binding(4) var dirt_texture: texture_2d<f32>;
@group(2) @binding(5) var pebble_texture: texture_2d<f32>;    // Corresponds to GLSL iChannel 0
@group(2) @binding(6) var terrain_sampler: sampler;

// Shared structs
struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,         // World -> View (Camera)
    inv_view: mat4x4<f32>,      // View (Camera) -> World
    camera_position: vec3f,   // Still needed for world origin offset if required elsewhere, but not for ray origin
    time: f32,
    resolution: vec2f,
    _padding: vec2f,
};

// --- Ported Shadertoy Constants ---
const PI: f32 = 3.141592;
const EARTH_RADIUS: f32 = 6300000.0;
const CLOUD_START: f32 = 300.0;
const CLOUD_HEIGHT: f32 = 500.0;
const LOW_SCATTER: vec3f = vec3f(1.0, 0.7, 0.5);

// --- Ported Helper Functions ---

fn hash1f( n: f32 ) -> f32 { return fract(sin(n)*43758.5453); }
fn hash2f( p: vec2f ) -> f32 { return fract(sin(dot(p,vec2f(127.1,311.7)))*43758.5453123); }

// Noise functions remain the same internally
fn noise3d( x: vec3f ) -> f32 { // Expects world-space input from clouds()
    let p = floor(x);
    var f = fract(x);
    f = f*f*(3.0-2.0*f);
    let coord = (p + f + 0.5) / 32.0;
    return textureSampleLevel(gray_cube_noise_texture, terrain_sampler, coord, 0.0).r;
}

fn noise2d( p: vec2f ) -> f32 { // Expects world-space input if used similarly
    let i = floor( p );
    var f = fract( p );
    f = f*f*(3.0-2.0*f);
    return -1.0 + 2.0 * mix( mix( hash2f( i + vec2f(0.0,0.0) ),
                                  hash2f( i + vec2f(1.0,0.0) ), f.x),
                             mix( hash2f( i + vec2f(0.0,1.0) ),
                                  hash2f( i + vec2f(1.0,1.0) ), f.x), f.y);
}

fn fbm3d( p_in: vec3f ) -> f32 { // Expects world-space input from clouds()
    let m = mat3x3f( 0.00, -0.80, -0.60, 0.80,  0.36, -0.48, 0.60, -0.48,  0.64 );
    var p = p_in;
    var f: f32 = 0.0;
    f  = 0.5000 * noise3d( p ); p = m * p * 2.02;
    f += 0.2500 * noise3d( p ); p = m * p * 2.03;
    f += 0.1250 * noise3d( p );
    return f;
}

fn ACESFilm(x: vec3<f32>) -> vec3<f32> {
    let a: f32 = 2.51;
    let b: f32 = 0.03;
    let c: f32 = 2.43;
    let d: f32 = 0.59;
    let e: f32 = 0.14;

    return clamp((x * (a * x + vec3<f32>(b))) / (x * (c * x + vec3<f32>(d)) + vec3<f32>(e)), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Intersects a ray (origin_cs, dir_cs) with a sphere (spherePos_cs, sphereRad)
// All inputs are expected to be in the SAME coordinate space (camera space).
fn intersectSphere(origin_cs: vec3f, dir_cs: vec3f, spherePos_cs: vec3f, sphereRad: f32) -> f32 {
    let oc_cs = origin_cs - spherePos_cs;
    let b = 2.0 * dot(dir_cs, oc_cs);
    let c = dot(oc_cs, oc_cs) - sphereRad * sphereRad;
    let disc = b * b - 4.0 * c;
    if (disc < 0.0) { return -1.0; }
    let sgn_b = select(1.0, -1.0, b < 0.0);
    let q = -0.5 * (b + sgn_b * sqrt(disc));
    let t0 = q; // Assuming dir_cs is normalized, a=1
    let t1 = c / q;
    if (t0 < 0.0) {
        if (t1 < 0.0) { return -1.0; } else { return t1; }
    } else {
        if (t1 < 0.0) { return t0; } else { return min(t0, t1); }
    }
}

struct CloudInfo {
    density: f32,
    cloud_height_norm: f32,
}

// Calculates cloud density at a given point.
// p_cs: Position to sample clouds, in CAMERA SPACE.
// earth_center_cs: Position of the Earth's center, in CAMERA SPACE.
// fast: Boolean flag for quality/performance tradeoff.
fn clouds(p_cs: vec3f, earth_center_cs: vec3f, fast: bool) -> CloudInfo {
    // Calculate distance from Earth center using camera-space vectors
    // length(p_cs - earth_center_cs) is invariant to the coordinate system origin/rotation
    let atmoHeight = length(p_cs - earth_center_cs) - EARTH_RADIUS;
    let cloudHeightNorm = clamp((atmoHeight - CLOUD_START) / (CLOUD_HEIGHT), 0.0, 1.0);

    // *** CRITICAL: Transform sample point back to WORLD SPACE for noise lookups ***
    // This keeps noise patterns fixed relative to the world, not the camera view.
    let p_ws = (camera.inv_view * vec4f(p_cs, 1.0)).xyz;

    // Use world-space position for time-based offsets and texture lookups
    var p_noise = p_ws;
    //p_noise.z += camera.time * 10.3;
    let weather_uv1 = -0.00005 * p_noise.zx;
    let largeWeather = clamp((textureSampleLevel(pebble_texture, terrain_sampler, weather_uv1, 0.0).r - 0.18) * 5.0, 0.0, 2.0) + 0.0;

    p_noise = p_ws; // Reset for second lookup
    //p_noise.x += camera.time * 8.3;
    let weather_uv2 = 0.0002 * p_noise.zx;
    var weather = largeWeather * max(0.0, textureSampleLevel(pebble_texture, terrain_sampler, weather_uv2, 0.0).r - 0.28) / 0.72 + 0.08;
    weather *= smoothstep(0.0, 0.5, cloudHeightNorm) * smoothstep(1.0, 0.5, cloudHeightNorm);

    let cloudShape = pow(weather, 0.3 + 1.5 * smoothstep(0.2, 0.5, cloudHeightNorm));

    var info: CloudInfo;
    info.cloud_height_norm = cloudHeightNorm;

    if (cloudShape <= 0.0) {
        info.density = 0.0;
        return info;
    }

    p_noise = p_ws; // Reset for 3D noise
   // p_noise.x += camera.time * 12.3;
    // Use WORLD SPACE position for fbm3d (which uses noise3d)
    var den = max(0.0, cloudShape - 0.7 * fbm3d(p_noise * 0.01));
    if (den <= 0.0) {
        info.density = 0.0;
        return info;
    }

    if (fast) {
        info.density = largeWeather * 0.2 * min(1.0, 5.0 * den);
        return info;
    }

    p_noise = p_ws; // Reset for second 3D noise
    //p_noise.y += camera.time * 15.2;
    // Use WORLD SPACE position for fbm3d
    den = max(0.0, den - 0.2 * fbm3d(p_noise * 0.05));

    info.density = largeWeather * 0.2 * min(1.0, 5.0 * den);
    return info;
}


fn numericalMieFit(costh: f32) -> f32 {
    let bestParams = array<f32, 10>(
        9.805233e-06, -6.500000e+01, -5.500000e+01, 8.194068e-01,
        1.388198e-01, -8.370334e+01, 7.810083e+00, 2.054747e-03,
        2.600563e-02, -4.552125e-12
    );
    let p1 = costh + bestParams[3];
    let expValues = exp(vec4f(bestParams[1] * costh + bestParams[2],
                              bestParams[5] * p1 * p1,
                              bestParams[6] * costh,
                              bestParams[9] * costh));
    let expValWeight= vec4f(bestParams[0], bestParams[4], bestParams[7], bestParams[8]);
    return dot(expValues, expValWeight);
}


fn HenyeyGreenstein(mu: f32, inG: f32) -> f32 {
    let g2 = inG * inG;
    let denom_term = 1.0 + g2 - 2.0 * inG * mu;
    let denom = pow(max(0.0001, denom_term), 1.5) * 4.0 * PI;
    if (denom < 0.00001) { return 0.0; }
    return (1.0 - g2) / denom;
}


// Calculates light scattering along a ray towards the sun.
// p_cs_in: Current position on the view ray, in CAMERA SPACE.
// phaseFunction: Precomputed phase function value.
// dC: Density at p_cs_in.
// mu: Dot product of view direction and sun direction (already computed).
// sun_direction_cs: Sun direction vector, in CAMERA SPACE.
// earth_center_cs: Earth center position, in CAMERA SPACE.
// cloudHeightNorm: Precomputed cloud height norm at p_cs_in.
// fast: Boolean flag.
fn lightRay(p_cs_in: vec3f, phaseFunction: f32, dC: f32, mu: f32, sun_direction_cs: vec3f, earth_center_cs: vec3f, cloudHeightNorm: f32, fast: bool) -> f32 {
    var p_cs = p_cs_in; // Position along the light ray, starts at view ray sample point
    let nbSampleLight = select(20, 7, fast);
    let zMaxl = 600.0;
    let stepL = zMaxl / f32(nbSampleLight);
    var lighRayDen = 0.0;

    // Use camera-space position for hash input seeding
    p_cs += sun_direction_cs * stepL * hash1f(dot(p_cs, vec3f(12.256, 2.646, 6.356)) + camera.time*0.5);

    for (var j = 0; j < nbSampleLight; j = j + 1) {
        // March along the light ray in camera space
        let current_p_cs = p_cs + sun_direction_cs * f32(j) * stepL;
        // Call clouds with camera-space position and earth center
        let cloud_info = clouds(current_p_cs, earth_center_cs, fast);
        lighRayDen += cloud_info.density;
    }

    if (fast) {
        return (0.5 * exp(-0.4 * stepL * lighRayDen) + max(0.0, -mu * 0.6 + 0.3) * exp(-0.02 * stepL * lighRayDen)) * phaseFunction;
    }

    let scatterAmount = mix(0.008, 1.0, smoothstep(0.96, 0.0, mu));
    let beersLaw = exp(-stepL * lighRayDen) + 0.5 * scatterAmount * exp(-0.1 * stepL * lighRayDen) + scatterAmount * 0.4 * exp(-0.02 * stepL * lighRayDen);
    let density_term = 1.5 * pow(min(1.0, dC * 8.5), 0.3 + 5.5 * cloudHeightNorm);
    let mix_factor = clamp(lighRayDen * 0.4, 0.0, 1.0);
    let cloud_factor = mix(0.05 + density_term, 1.0, mix_factor);

    return beersLaw * phaseFunction * cloud_factor;
}

// Performs the main ray marching for the sky and clouds.
// org_cs: Ray origin, in CAMERA SPACE (always vec3f(0.0)).
// dir_cs: Ray direction, in CAMERA SPACE.
// sun_direction_cs: Sun direction vector, in CAMERA SPACE.
// earth_center_cs: Earth center position, in CAMERA SPACE.
// fast: Boolean flag.
fn skyRay(org_cs: vec3f, dir_cs: vec3f, sun_direction_cs: vec3f, sun_power: vec3f, earth_center_cs: vec3f, fast: bool) -> vec3f {
    let ATM_START = EARTH_RADIUS + CLOUD_START;
    let ATM_END = ATM_START + CLOUD_HEIGHT;

    let nbSample = select(35, 13, fast);
    var color = vec3f(0.0);

    // Intersect with atmosphere layers using camera-space inputs
    let distToAtmStart = intersectSphere(org_cs, dir_cs, earth_center_cs, ATM_START);
    let distToAtmEnd = intersectSphere(org_cs, dir_cs, earth_center_cs, ATM_END);

    let mu = dot(sun_direction_cs, dir_cs); // Use camera-space vectors

    // *** Calculate World Space Direction for Horizon Checks ***
    let dir_ws = normalize((camera.inv_view * vec4f(dir_cs, 0.0)).xyz);

    // *** Widen Sun Smoothstep slightly ***
    let sun_disk_intensity = 1e4 * smoothstep(0.9995, 1.0, mu); // Lowered 0.9998 threshold
    
    // Background sky color calculation
    //let background_base = mix(vec3f(0.2, 0.6, 1.0), vec3f(0.8, 0.95, 1.0), pow(0.5 + 0.5 * mu, 15.0));
    let background_base = vec3f(0.4, 0.6, 1.0);
    var background = 9.0 * background_base;

    if (distToAtmStart >= distToAtmEnd || distToAtmEnd < 0.0) {
        // Calculate background color directly (no cloud intersection)
        let background_base_bg = mix(vec3f(0.2, 0.52, 1.0), vec3f(0.8, 0.95, 1.0), pow(0.5 + 0.5 * mu, 15.0));
        // *** Use World Space Y for horizon mix ***
        let background_horizon_bg = mix(vec3f(3.5), vec3f(0.0), min(1.0, 2.3 * max(0.0, dir_ws.y)));
        var background_bg = 6.0 * background_base_bg + background_horizon_bg;
        return background_bg;
    } else {
        // Start ray marching from atmosphere entry point in camera space
        var p_cs = org_cs + max(0.0, distToAtmStart) * dir_cs;
        let totalDist = distToAtmEnd - max(0.0, distToAtmStart);
        let stepS = totalDist / f32(nbSample);
        var T = 1.0; // Transmittance
        let phaseFunction = numericalMieFit(mu);

        // Jitter starting position using camera-space direction for seeding
        p_cs += dir_cs * stepS * hash1f(dot(dir_cs, vec3f(12.256, 2.646, 6.356)) + camera.time);

        // *** Optimization: Use World Space Y direction (original intent) ***
        // Keep a small threshold to avoid marching when looking straight down. Adjust if needed.
        if (dir_ws.y > -0.1) { // Allow marching slightly below horizon
            for (var i = 0; i < nbSample; i = i + 1) {
                // Call clouds with camera-space position and earth center
                let cloud_info = clouds(p_cs, earth_center_cs, fast);
                let density = cloud_info.density;

                if (density > 0.001) {
                    // Call lightRay with camera-space inputs
                    let intensity = lightRay(p_cs, phaseFunction, density, mu, sun_direction_cs, earth_center_cs, cloud_info.cloud_height_norm, fast);

                    // let ambient_sky = vec3f(0.2, 0.5, 1.0) * 6.5;
                    let ambient_sky = background;
                    let ambient_cloud = vec3f(0.8);
                    let ambient_mix = cloud_info.cloud_height_norm;
                    let ambient = (0.5 + 0.6 * ambient_mix) * ambient_sky + ambient_cloud * max(0.0, 1.0 - 2.0 * ambient_mix);
                    let radiance = (ambient + sun_power * intensity);

                    let BeerTerm = exp(-density * stepS);
                    let Sint = select( stepS, (1.0 - BeerTerm) / density, density > 0.00001);
                    color += T * radiance * density * Sint;
                    T *= BeerTerm;

                    if ( T <= 0.05) { break; }
                }
                // Step along ray in camera space
                p_cs += dir_cs * stepS;
            }
        } // end if dir_ws.y > -0.1


        color += background * T; // Add background attenuated by clouds

        return color;
    }
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) view_dir_cs: vec3f, // Pass CAMERA SPACE View Direction
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // ... (Vertex shader remains the same, passes view_dir_cs) ...
    let positions = array<vec2f, 4>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, 1.0)
    );
    let pos = positions[vertex_index];
    let ndc_far = vec4f(pos.x, -pos.y, 1.0, 1.0); // Point on far plane NDC
    let world_far_plane = camera.inv_view_proj * ndc_far;
    let world_far_plane_xyz = world_far_plane.xyz / world_far_plane.w;
    let view_direction_ws = normalize(world_far_plane_xyz - camera.camera_position);
    let view_direction_cs = normalize((camera.view * vec4f(view_direction_ws, 0.0)).xyz);

    var output: VertexOutput;
    output.position = vec4f(pos, 0.0, 1.0);
    output.view_dir_cs = view_direction_cs;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let frag_coord = vec2<i32>(floor(input.position.xy));
    let texture_dims = textureDimensions(depth_texture, 0);
    let clamped_coord = clamp(frag_coord, vec2<i32>(0), vec2<i32>(texture_dims) - vec2<i32>(1));
    let depth = textureLoad(depth_texture, clamped_coord, 0).r;


    const SKY_DEPTH_THRESHOLD = 1000000.0; 
                     
    if depth >= SKY_DEPTH_THRESHOLD { // Sky pixel check potentially modified
        let org_cs = vec3f(0.0);
        let dir_cs = input.view_dir_cs;
        let sun_direction_ws = normalize( vec3f(0.6, 0.45, -0.8) );
        var sun_power = vec3f(1.0, 0.9, 0.6) * 300.0; //750.0 * (sin(camera.time) + 1.0);
        let earth_center_ws = vec3f(0.0, -EARTH_RADIUS, 0.0);
        let sun_direction_cs = normalize((camera.view * vec4f(sun_direction_ws, 0.0)).xyz);
        let earth_center_cs = (camera.view * vec4f(earth_center_ws, 1.0)).xyz;
        let mu = dot(sun_direction_cs, dir_cs); // mu for sun angle effects

        // Call skyRay - now it doesn't add the sun disk internally
        var scene_color = skyRay(org_cs, dir_cs, sun_direction_cs, sun_power, earth_center_cs, false);

        // --- Apply Exposure/Tonemapping ---
        var final_color = scene_color;

        // Adjust exposure - start with less reduction
        final_color *= 0.07;

        // // Reinhard Tonemapping (Example) - Often better for HDR
        // final_color = final_color / (final_color + vec3f(1.0));

        // Or keep power curve, but maybe adjust exponent
        // final_color = pow(final_color, vec3f(1.0 / 2.2)); // Gamma correction instead of squaring

        final_color = pow(final_color, vec3(2.0));
 
        final_color = ACESFilm(final_color);

        return vec4f(final_color, 1.0);

    } else {
        discard; // Pixel belongs to scene geometry
    }
}