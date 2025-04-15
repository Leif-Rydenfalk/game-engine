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
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
    resolution: vec2<f32>,
    _padding: vec2<f32>,
};

// --- Ported Shadertoy Constants ---
const PI: f32 = 3.141592;
// let EPSILON_NRM = (0.1 / camera.resolution.x); // Not used in sky code, careful if needed later

// Cloud parameters
const EARTH_RADIUS: f32 = 6300000.0;
const CLOUD_START: f32 = 800.0;
const CLOUD_HEIGHT: f32 = 600.0;
const SUN_POWER: vec3f = vec3f(1.0, 0.9, 0.6) * 750.0;
const LOW_SCATTER: vec3f = vec3f(1.0, 0.7, 0.5); // Used in original fog, not directly in skyRay

// --- Ported Helper Functions ---

// Noise generation functions (by iq) - Renamed for WGSL (no overloading)
fn hash1f( n: f32 ) -> f32 {
    return fract(sin(n)*43758.5453);
}

fn hash2f( p: vec2f ) -> f32 {
    // Use texture intrinsic if available and desired for better quality hash
    return fract(sin(dot(p,vec2f(127.1,311.7)))*43758.5453123);
}

// Using VOLUME_TEXTURES path (iChannel2 -> gray_cube_noise_texture)
fn noise3d( x: vec3f ) -> f32 {
    let p = floor(x);
    var f = fract(x);
    f = f*f*(3.0-2.0*f);
    // Sample from 3D noise texture (gray_cube_noise_texture)
    // Assuming gray_cube_noise_texture dimensions are 32x32x32 based on GLSL: /32.0
    let coord = (p + f + 0.5) / 32.0;
    // Use textureSampleLevel to specify LOD 0 explicitly
    return textureSampleLevel(gray_cube_noise_texture, terrain_sampler, coord, 0.0).r;
}

// Using procedural path (NOISE_TEXTURES not defined)
fn noise2d( p: vec2f ) -> f32 {
    let i = floor( p );
    var f = fract( p );
    f = f*f*(3.0-2.0*f);
    return -1.0 + 2.0 * mix( mix( hash2f( i + vec2f(0.0,0.0) ),
                                  hash2f( i + vec2f(1.0,0.0) ), f.x),
                             mix( hash2f( i + vec2f(0.0,1.0) ),
                                  hash2f( i + vec2f(1.0,1.0) ), f.x), f.y);
}

fn fbm3d( p_in: vec3f ) -> f32 {
     // WGSL matrices constructed column-major
    let m = mat3x3f( 0.00, -0.80, -0.60,  // col 0
                     0.80,  0.36, -0.48,  // col 1
                     0.60, -0.48,  0.64 ); // col 2
    var p = p_in;
    var f: f32 = 0.0;
    f  = 0.5000 * noise3d( p ); p = m * p * 2.02;
    f += 0.2500 * noise3d( p ); p = m * p * 2.03;
    f += 0.1250 * noise3d( p );
    return f;
}

// Renamed from fbm to avoid conflict if a 2D version existed
// This function wasn't directly used in the GLSL skyRay but kept for potential future use
fn fbm_texlookup( p_in: vec3f ) -> f32 {
     // WGSL matrices constructed column-major
    let m = mat3x3f( 0.00, -0.80, -0.60,  // col 0
                     0.80,  0.36, -0.48,  // col 1
                     0.60, -0.48,  0.64 ); // col 2
    var p = p_in;
    var f: f32;
    f  = 0.5000 * noise3d( p ); p = m * p * 2.02;
    f += 0.2500 * noise3d( p ); p = m * p * 2.03;
    f += 0.1250 * noise3d( p );
    return f;
}


fn intersectSphere(origin: vec3f, dir: vec3f, spherePos: vec3f, sphereRad: f32) -> f32 {
    let oc = origin - spherePos;
    let b = 2.0 * dot(dir, oc);
    let c = dot(oc, oc) - sphereRad * sphereRad;
    let disc = b * b - 4.0 * c;
    if (disc < 0.0) {
        return -1.0;
    }
    // Stable quadratic formula
    let sgn_b = select(1.0, -1.0, b < 0.0);
    let q = -0.5 * (b + sgn_b * sqrt(disc));
    let t0 = q; // Assuming dir is normalized, a=1
    let t1 = c / q;

    // Return the smallest non-negative intersection distance
    if (t0 < 0.0) {
        if (t1 < 0.0) { return -1.0; } // Both negative
        else { return t1; } // t0 negative, t1 non-negative
    } else {
        if (t1 < 0.0) { return t0; } // t0 non-negative, t1 negative
        else { return min(t0, t1); } // Both non-negative
    }
}


// Structure to return multiple values from clouds function
struct CloudInfo {
    density: f32,
    cloud_height_norm: f32, // Renamed from cloudHeight to avoid conflict
}

// WGSL cannot have 'out' parameters, return struct instead
fn clouds(p_in: vec3f, fast: bool) -> CloudInfo {
    var p = p_in;
    let atmoHeight = length(p - vec3f(0.0, -EARTH_RADIUS, 0.0)) - EARTH_RADIUS;
    let cloudHeightNorm = clamp((atmoHeight - CLOUD_START) / (CLOUD_HEIGHT), 0.0, 1.0);

    p.z += camera.time * 10.3;
    // Sample weather map (iChannel0 -> pebble_texture) <--- CORRECTED
    // Assuming pebble_texture dimensions are such that 0.00005 is appropriate scaling
    let weather_uv1 = -0.00005 * p.zx;
    let largeWeather = clamp((textureSampleLevel(pebble_texture, terrain_sampler, weather_uv1, 0.0).r - 0.18) * 5.0, 0.0, 2.0);

    p.x += camera.time * 8.3;
    // Sample weather map (iChannel0 -> pebble_texture) <--- CORRECTED
    let weather_uv2 = 0.0002 * p.zx;
    var weather = largeWeather * max(0.0, textureSampleLevel(pebble_texture, terrain_sampler, weather_uv2, 0.0).r - 0.28) / 0.72;
    weather *= smoothstep(0.0, 0.5, cloudHeightNorm) * smoothstep(1.0, 0.5, cloudHeightNorm);

    let cloudShape = pow(weather, 0.3 + 1.5 * smoothstep(0.2, 0.5, cloudHeightNorm));

    var info: CloudInfo;
    info.cloud_height_norm = cloudHeightNorm;

    if (cloudShape <= 0.0) {
        info.density = 0.0;
        return info;
    }

    p.x += camera.time * 12.3;
    // Uses 3D noise (fbm3d -> noise3d -> iChannel2 -> gray_cube_noise_texture)
    var den = max(0.0, cloudShape - 0.7 * fbm3d(p * 0.01));
    if (den <= 0.0) {
        info.density = 0.0;
        return info;
    }

    if (fast) {
        info.density = largeWeather * 0.2 * min(1.0, 5.0 * den);
        return info;
    }

    p.y += camera.time * 15.2;
    // Uses 3D noise (fbm3d -> noise3d -> iChannel2 -> gray_cube_noise_texture)
    den = max(0.0, den - 0.2 * fbm3d(p * 0.05));

    info.density = largeWeather * 0.2 * min(1.0, 5.0 * den);
    return info;
}

// From https://www.shadertoy.com/view/4sjBDG
fn numericalMieFit(costh: f32) -> f32 {
    // This function was optimized to minimize (delta*delta)/reference in order to capture
    // the low intensity behavior.
    // Use array instead of fixed size array bestParams[10];
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


fn lightRay(p_in: vec3f, phaseFunction: f32, dC: f32, mu: f32, sun_direction: vec3f, cloudHeightNorm: f32, fast: bool) -> f32 {
    var p = p_in;
    let nbSampleLight = select(20, 7, fast); // if fast use 7 else 20
    let zMaxl = 600.0;
    let stepL = zMaxl / f32(nbSampleLight);

    var lighRayDen = 0.0;
    // Use a different hash input as camera.time is already used in clouds
    p += sun_direction * stepL * hash1f(dot(p, vec3f(12.256, 2.646, 6.356)) + camera.time*0.5); // Offset hash time

    for (var j = 0; j < nbSampleLight; j = j + 1) {
        // Call clouds, but only need density here
        let current_p = p + sun_direction * f32(j) * stepL;
        let cloud_info = clouds(current_p, fast); // Need to call clouds again
        lighRayDen += cloud_info.density;
    }

    if (fast) {
        return (0.5 * exp(-0.4 * stepL * lighRayDen) + max(0.0, -mu * 0.6 + 0.3) * exp(-0.02 * stepL * lighRayDen)) * phaseFunction;
    }

    let scatterAmount = mix(0.008, 1.0, smoothstep(0.96, 0.0, mu));
    let beersLaw = exp(-stepL * lighRayDen) + 0.5 * scatterAmount * exp(-0.1 * stepL * lighRayDen) + scatterAmount * 0.4 * exp(-0.02 * stepL * lighRayDen);

    // Use pow for 0.3 + 5.5 * cloudHeightNorm exponent
    let density_term = 1.5 * pow(min(1.0, dC * 8.5), 0.3 + 5.5 * cloudHeightNorm);
    let mix_factor = clamp(lighRayDen * 0.4, 0.0, 1.0);
    let cloud_factor = mix(0.05 + density_term, 1.0, mix_factor);

    return beersLaw * phaseFunction * cloud_factor;
}


fn skyRay(org: vec3f, dir: vec3f, sun_direction: vec3f, fast: bool) -> vec3f {
    let ATM_START = EARTH_RADIUS + CLOUD_START;
    let ATM_END = ATM_START + CLOUD_HEIGHT;

    let nbSample = select(35, 13, fast);
    var color = vec3f(0.0);

    let sphere_center = vec3f(0.0, -EARTH_RADIUS, 0.0);
    let distToAtmStart = intersectSphere(org, dir, sphere_center, ATM_START);
    let distToAtmEnd = intersectSphere(org, dir, sphere_center, ATM_END);

    if (distToAtmStart >= distToAtmEnd || distToAtmEnd < 0.0) {
       // Ray doesn't properly intersect the cloud layer from outside
       // Calculate background color directly
        let mu_bg = dot(sun_direction, dir);
        let background_base_bg = mix(vec3f(0.2, 0.52, 1.0), vec3f(0.8, 0.95, 1.0), pow(0.5 + 0.5 * mu_bg, 15.0));
        let background_horizon_bg = mix(vec3f(3.5), vec3f(0.0), min(1.0, 2.3 * max(0.0, dir.y))); // Clamp dir.y
        var background_bg = 6.0 * background_base_bg + background_horizon_bg;
         if (!fast) {
                 background_bg += vec3f(1e4 * smoothstep(0.9998, 1.0, mu_bg)); // Additive sun disk
         }
        return background_bg;

    } else {
        var p = org + max(0.0, distToAtmStart) * dir; // Start at entry point or camera pos if inside
        let totalDist = distToAtmEnd - max(0.0, distToAtmStart);
        let stepS = totalDist / f32(nbSample);
        var T = 1.0; // Transmittance
        let mu = dot(sun_direction, dir);
        let phaseFunction = numericalMieFit(mu);

        // Jitter starting position
        p += dir * stepS * hash1f(dot(dir, vec3f(12.256, 2.646, 6.356)) + camera.time);

        // Optimization from GLSL: Only trace if not looking too far down
        // Adjusted threshold slightly for safety; original was 0.015
        if (dir.y > 0.0) {
            for (var i = 0; i < nbSample; i = i + 1) {
                let cloud_info = clouds(p, fast);
                let density = cloud_info.density;

                if (density > 0.001) { // Performance: Skip steps with negligible density (adjust threshold if needed)
                    let intensity = lightRay(p, phaseFunction, density, mu, sun_direction, cloud_info.cloud_height_norm, fast);

                    // Ambient term based on height
                    let ambient_sky = vec3f(0.2, 0.5, 1.0) * 6.5;
                    let ambient_cloud = vec3f(0.8);
                    let ambient_mix = cloud_info.cloud_height_norm;
                    // Original GLSL: (0.5 + 0.6*cloudHeight)*ambient_sky + ambient_cloud * max(0.0, 1.0-2.0*cloudHeight)
                    let ambient = (0.5 + 0.6 * ambient_mix) * ambient_sky + ambient_cloud * max(0.0, 1.0 - 2.0 * ambient_mix);

                    let radiance = (ambient + SUN_POWER * intensity); // Density multiplication moved to integration step

                    // Volumetric integration (Seb Hillaire) - corrected approach
                    let BeerTerm = exp(-density * stepS);
                    // Integral of Beer's law over step: (1 - exp(-density * stepS)) / density
                    // Handle potential division by zero
                    let Sint = select( stepS, (1.0 - BeerTerm) / density, density > 0.00001);
                    color += T * radiance * density * Sint; // Add attenuated radiance scattered towards camera * multiplied by density
                    T *= BeerTerm; // Attenuate accumulated transmittance

                    if ( T <= 0.05) { // Early exit if occluded
                        break;
                    }
                }
                p += dir * stepS;
            }
        } // end if dir.y > 0.0

        // Add distant stars/noise if not fast rendering (and not fully occluded)
        if (!fast && T > 0.05) {
            // Intersect with a slightly larger sphere to get a point for background noise
            let distToBgSphere = intersectSphere(org, dir, sphere_center, ATM_END + 1000.0);
            if (distToBgSphere > 0.0) {
                let pC = org + distToBgSphere * dir;
                 // Use fbm3d for background variation
                // Original GLSL: vec3(3.0)*max(0.0, fbm(vec3(1.0, 1.0, 1.8)*pC*0.002)-0.4)
                // Corrected noise call assuming fbm3d expects unscaled input like GLSL noise()
                color += T * vec3f(3.0) * max(0.0, fbm3d(pC * 0.002 * vec3f(1.0, 1.0, 1.8)) - 0.4); // Adjusted scale
            }
        }

        // Background sky color calculation (based on sun direction and view direction)
        let background_base = mix(vec3f(0.2, 0.52, 1.0), vec3f(0.8, 0.95, 1.0), pow(0.5 + 0.5 * mu, 15.0));
        let background_horizon = mix(vec3f(3.5), vec3f(0.0), min(1.0, 2.3 * max(0.0, dir.y))); // Clamp dir.y
        var background = 6.0 * background_base + background_horizon;

        // Sun disk (only if not fast)
        if (!fast) {
             background += vec3f(1e4 * smoothstep(0.9998, 1.0, mu)); // Additive sun disk
        }
        color += background * T; // Add background attenuated by clouds
        return color;
    } // end else (ray intersects cloud layer)

}

fn HenyeyGreenstein(mu: f32, inG: f32) -> f32 {
    let g2 = inG * inG;
    let denom_term = 1.0 + g2 - 2.0 * inG * mu;
    // Add epsilon to denominator to avoid potential division by zero or instability near poles
    let denom = pow(max(0.0001, denom_term), 1.5) * 4.0 * PI;
    if (denom < 0.00001) { return 0.0; } // Avoid division by zero if input is pathological
    return (1.0 - g2) / denom;
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) view_dir_ws: vec3f, // Pass World Space View Direction
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec2f, 4>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, 1.0)
    );
    let pos = positions[vertex_index];
    let ndc = vec4f(pos.x, -pos.y, 1.0, 1.0); // Point on far plane NDC
    let world_far_plane = camera.inv_view_proj * ndc;
    let world_far_plane_xyz = world_far_plane.xyz / world_far_plane.w;

    // Calculate world space direction FROM camera TO far plane point
    let view_direction_world_space = normalize(world_far_plane_xyz - camera.position);

    var output: VertexOutput;
    // Use near plane for rasterization, but direction is calculated towards far plane
    output.position = vec4f(pos, 0.0, 1.0);
    output.view_dir_ws = view_direction_world_space;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let frag_coord = vec2<i32>(floor(input.position.xy));
    let texture_dims = textureDimensions(depth_texture, 0);
    let clamped_coord = clamp(frag_coord, vec2<i32>(0), vec2<i32>(texture_dims) - vec2<i32>(1));
    let depth = textureLoad(depth_texture, clamped_coord, 0).r;

    if depth >= 1000000.0 {
        // Ray origin IS the camera position
        let org = camera.position;
        // Ray direction is passed directly from vertex shader
        let dir = input.view_dir_ws; // Already normalized

        // --- Rest of your fragment shader ---
        let sun_direction = normalize( vec3f(0.6, 0.45, -0.8) );
        let mu = dot(sun_direction, dir);
        let earth_center = vec3f(0.0, -EARTH_RADIUS, 0.0);

        var fogDistance = intersectSphere(org, dir, earth_center, EARTH_RADIUS);

        var scene_color = vec3f(0.0, 1.0, 0.0); // Placeholder
        if (fogDistance < 0.0) {
             // Use WORLD space origin and direction for sky ray
            scene_color = skyRay(org, dir, sun_direction, false);
            fogDistance = intersectSphere(org, dir, earth_center, EARTH_RADIUS + 160.0);
            if (fogDistance < 0.0) {
                fogDistance = 1000000.0;
            }
        }

        // Calculate Fog Color & Phase (matching commented GLSL)
        let fogPhase = 0.5 * HenyeyGreenstein(mu, 0.7) + 0.5 * HenyeyGreenstein(mu, -0.6);
        // Original GLSL fog color components:
        // Directional scattering: fogPhase * 0.1 * LOW_SCATTER * SUN_POWER
        // Ambient fog term: 10.0 * vec3(0.55, 0.8, 1.0)
        let fog_scatter_color = fogPhase * 0.1 * LOW_SCATTER * SUN_POWER;
        let fog_ambient_color = 10.0 * vec3f(0.55, 0.8, 1.0);
        let fogColor = fog_scatter_color + fog_ambient_color;

        // Apply Fog using exponential falloff
        let fog_density = 0.00003;
        // Ensure distance is non-negative for exp calculation
        let fog_factor = exp(-fog_density * max(0.0, fogDistance));
        let final_color_with_fog = mix(fogColor, scene_color, fog_factor);

        // Apply final brightness factor (from original GLSL)
        let final_color = 0.06 * final_color_with_fog;

        return vec4f(final_color, 1.0);

    } else {
        discard;
    }
}
