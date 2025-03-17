struct Uniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_position: vec3f,
    time: f32,
    resolution: vec2f,
    _padding: vec2f
};

// @group(0) @binding(0) var<uniform> uniforms: Uniforms;
// @group(0) @binding(1) var pebble_texture: texture_2d<f32>;
// @group(0) @binding(2) var rgb_noise_texture: texture_2d<f32>;
// @group(0) @binding(3) var gray_cube_noise_texture: texture_3d<f32>;
// @group(0) @binding(4) var gray_noise_texture: texture_2d<f32>;
// @group(0) @binding(5) var texture_sampler: sampler;

@group(0) @binding(0) var<uniform> uniforms: Uniform;
@group(1) @binding(0) var rgb_noise_texture: texture_2d<f32>;
@group(1) @binding(1) var gray_noise_texture: texture_2d<f32>;
@group(1) @binding(2) var gray_cube_noise_texture: texture_3d<f32>; 
@group(1) @binding(3) var grain_texture: texture_2d<f32>; 
@group(1) @binding(4) var dirt_texture: texture_2d<f32>;  
@group(1) @binding(5) var pebble_texture: texture_2d<f32>;  
@group(1) @binding(6) var texture_sampler: sampler; // Must use repeat mode

// Constants
const PI: f32 = 3.141592;
const EPSILON_NRM: f32 = 0.1;  // Will divide by resolution.x in the shader

// Cloud parameters
const EARTH_RADIUS: f32 = 6300e3;
const CLOUD_START: f32 = 800.0;
const CLOUD_HEIGHT: f32 = 600.0;
const SUN_POWER: vec3f = vec3f(1.0, 0.9, 0.6) * 750.0;
const LOW_SCATTER: vec3f = vec3f(1.0, 0.7, 0.5);
const FOG_DIST: f32 = 160.0;

// Noise generation functions
fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

fn hash2(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453123);
}

fn noise3d(x: vec3f) -> f32 {
    let p = floor(x);
    var f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    
    // Volume texture sampling
    return textureSample(gray_cube_noise_texture, texture_sampler, (p + f + 0.5) / 32.0).x;
}

fn noise2d(p: vec2f) -> f32 {
    let i = floor(p);
    var f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    // Noise texture sampling
    return textureSample(gray_noise_texture, texture_sampler, (i + f + vec2f(0.5)) / 64.0).x * 2.0 - 1.0;
}

fn fbm(p: vec3f) -> f32 {
    var p_mod = p;
    let m = mat3x3(
        0.00, 0.80, 0.60,
        -0.80, 0.36, -0.48,
        -0.60, -0.48, 0.64
    );
    
    var f: f32 = 0.5000 * noise3d(p_mod); 
    p_mod = m * p_mod * 2.02;
    f += 0.2500 * noise3d(p_mod); 
    p_mod = m * p_mod * 2.03;
    f += 0.1250 * noise3d(p_mod);
    
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
    
    let q = (-b + select(-sqrt(disc), sqrt(disc), b < 0.0)) / 2.0;
    var t0 = q;
    var t1 = c / q;
    
    if (t0 > t1) {
        let temp = t0;
        t0 = t1;
        t1 = temp;
    }
    
    if (t1 < 0.0) {
        return -1.0;
    }
    
    return select(t0, t1, t0 < 0.0);
}

fn clouds(p: vec3f, fast: bool) -> vec2f {
    var atmoHeight = length(p - vec3f(0.0, -EARTH_RADIUS, 0.0)) - EARTH_RADIUS;
    let cloudHeight = clamp((atmoHeight - CLOUD_START) / CLOUD_HEIGHT, 0.0, 1.0);
    
    var p_mod = p;
    p_mod.z += uniforms.time * 10.3;
    
    let largeWeather = clamp(
        (textureSample(pebble_texture, texture_sampler, -0.00005 * p_mod.zx).x - 0.18) * 5.0, 
        0.0, 
        2.0
    );

    p_mod.x += uniforms.time * 8.3;
    var weather = largeWeather * max(0.0, textureSample(pebble_texture, texture_sampler, 0.0002 * p_mod.zx).x - 0.28) / 0.72;
    weather *= smoothstep(0.0, 0.5, cloudHeight) * smoothstep(1.0, 0.5, cloudHeight);
    
    let cloudShape = pow(weather, 0.3 + 1.5 * smoothstep(0.2, 0.5, cloudHeight));
    if (cloudShape <= 0.0) {
        return vec2f(0.0, cloudHeight);
    }
    
    p_mod.x += uniforms.time * 12.3;
    var den = max(0.0, cloudShape - 0.7 * fbm(p_mod * 0.01));
    
    if (den <= 0.0) {
        return vec2f(0.0, cloudHeight);
    }
    
    if (fast) {
        return vec2f(largeWeather * 0.2 * min(1.0, 5.0 * den), cloudHeight);
    }
    
    p_mod.y += uniforms.time * 15.2;
    den = max(0.0, den - 0.2 * fbm(p_mod * 0.05));
    return vec2f(largeWeather * 0.2 * min(1.0, 5.0 * den), cloudHeight);
}

// From https://www.shadertoy.com/view/4sjBDG
fn numericalMieFit(costh: f32) -> f32 {
    let bestParams = array<f32, 10>(
        9.805233e-06, -6.500000e+01, -5.500000e+01, 8.194068e-01,
        1.388198e-01, -8.370334e+01, 7.810083e+00, 2.054747e-03,
        2.600563e-02, -4.552125e-12
    );
    
    let p1 = costh + bestParams[3];
    let expValues = vec4f(
        exp(bestParams[1] * costh + bestParams[2]),
        exp(bestParams[5] * p1 * p1),
        exp(bestParams[6] * costh),
        exp(bestParams[9] * costh)
    );
    
    let expValWeight = vec4f(bestParams[0], bestParams[4], bestParams[7], bestParams[8]);
    return dot(expValues, expValWeight);
}

fn lightRay(p: vec3f, phaseFunction: f32, dC: f32, mu: f32, sun_direction: vec3f, cloudHeight: f32, fast: bool) -> f32 {
    let nbSampleLight = select(20, 7, fast);
    let zMaxl = 600.0;
    let stepL = zMaxl / f32(nbSampleLight);
    
    var lighRayDen = 0.0;
    var p_mod = p + sun_direction * stepL * hash(dot(p, vec3f(12.256, 2.646, 6.356)) + uniforms.time);
    
    for (var j = 0; j < nbSampleLight; j += 1) {
        let cloudData = clouds(p_mod + sun_direction * f32(j) * stepL, fast);
        lighRayDen += cloudData.x;
    }
    
    if (fast) {
        return (0.5 * exp(-0.4 * stepL * lighRayDen) + max(0.0, -mu * 0.6 + 0.3) * exp(-0.02 * stepL * lighRayDen)) * phaseFunction;
    }
    
    let scatterAmount = mix(0.008, 1.0, smoothstep(0.96, 0.0, mu));
    let beersLaw = exp(-stepL * lighRayDen) + 0.5 * scatterAmount * exp(-0.1 * stepL * lighRayDen) + scatterAmount * 0.4 * exp(-0.02 * stepL * lighRayDen);
    return beersLaw * phaseFunction * mix(0.05 + 1.5 * pow(min(1.0, dC * 8.5), 0.3 + 5.5 * cloudHeight), 1.0, clamp(lighRayDen * 0.4, 0.0, 1.0));
}

fn Schlick(f0: f32, VoH: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - VoH, 5.0);
}

fn skyRay(org: vec3f, dir: vec3f, sun_direction: vec3f, fast: bool) -> vec3f {
    const ATM_START = EARTH_RADIUS + CLOUD_START;
    const ATM_END = ATM_START + CLOUD_HEIGHT;
    
    let nbSample = select(35, 13, fast);
    var color = vec3f(0.0);
    
    let distToAtmStart = intersectSphere(org, dir, vec3f(0.0, -EARTH_RADIUS, 0.0), ATM_START);
    let distToAtmEnd = intersectSphere(org, dir, vec3f(0.0, -EARTH_RADIUS, 0.0), ATM_END);
    var p = org + distToAtmStart * dir;
    
    let stepS = (distToAtmEnd - distToAtmStart) / f32(nbSample);
    var T = 1.0;
    
    let mu = dot(sun_direction, dir);
    let phaseFunction = numericalMieFit(mu);
    p += dir * stepS * hash(dot(dir, vec3f(12.256, 2.646, 6.356)) + uniforms.time);
    
    if (dir.y > 0.015 || true) {
        for (var i = 0; i < nbSample; i += 1) {
            let cloudData = clouds(p, fast);
            let density = cloudData.x;
            let cloudHeight = cloudData.y;
            
            if (density > 0.0) {
                let intensity = lightRay(p, phaseFunction, density, mu, sun_direction, cloudHeight, fast);
                let ambient = (0.5 + 0.6 * cloudHeight) * vec3f(0.2, 0.5, 1.0) * 6.5 + vec3f(0.8) * max(0.0, 1.0 - 2.0 * cloudHeight);
                let radiance = ambient + SUN_POWER * intensity;
                let scaledRadiance = radiance * density;
                
                // By Seb Hillaire
                color += T * (scaledRadiance - scaledRadiance * exp(-density * stepS)) / density;
                T *= exp(-density * stepS);
                
                if (T <= 0.05) {
                    break;
                }
            }
            p += dir * stepS;
        }
    }
    
    if (!fast) {
        let pC = org + intersectSphere(org, dir, vec3f(0.0, -EARTH_RADIUS, 0.0), ATM_END + 1000.0) * dir;
        color += T * vec3f(3.0) * max(0.0, fbm(vec3f(1.0, 1.0, 1.8) * pC * 0.002) - 0.4);
    }
    
    var background = 6.0 * mix(vec3f(0.2, 0.52, 1.0), vec3f(0.8, 0.95, 1.0), pow(0.5 + 0.5 * mu, 15.0)) + mix(vec3f(3.5), vec3f(0.0), min(1.0, 2.3 * dir.y));
    
    if (!fast) {
        background += T * vec3f(1e4 * smoothstep(0.9998, 1.0, mu));
    }
    
    color += background * T;
    return color;
}

fn HenyeyGreenstein(mu: f32, inG: f32) -> f32 {
    return (1.0 - inG * inG) / (pow(1.0 + inG * inG - 2.0 * inG * mu, 1.5) * 4.0 * PI);
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) texCoord: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var output: VertexOutput;
    
    // Generate a full-screen triangle
    let positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0)
    );
    
    let texCoords = array<vec2f, 3>(
        vec2f(0.0, 0.0),
        vec2f(2.0, 0.0),
        vec2f(0.0, 2.0)
    );
    
    output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
    output.texCoord = texCoords[vertexIndex];
    
    return output;
}

@fragment
fn fs_main(@location(0) texCoord: vec2f) -> @location(0) vec4f {
    let fragCoord = texCoord * uniforms.resolution;
    
    // Calculate normalized device coordinates (NDC)
    // Convert from [0,1] to [-1,1] range
    let ndc = vec2f(2.0 * texCoord.x - 1.0, 2.0 * texCoord.y - 1.0);
    
    // Create a ray in clip space going from near to far plane
    let clipPos = vec4f(ndc, 1.0, 1.0);
    
    // Transform to world space using inverse view-projection matrix
    let rayWorld = uniforms.inv_view_proj * clipPos;
    
    // Perspective division to get the actual world position
    let worldPos = rayWorld.xyz / rayWorld.w;
    
    // Calculate ray direction from camera to this world position
    let dir = normalize(worldPos - uniforms.camera_position);
    
    // Use the original camera position
    let org = uniforms.camera_position;
    var color = vec3f(0.0);
    
    let sun_direction = normalize(vec3f(sin(uniforms.time * 0.5) * 2.0, 0.45, -0.8));
    var fogDistance = intersectSphere(org, dir, vec3f(0.0, -EARTH_RADIUS, 0.0), EARTH_RADIUS);
    let mu = dot(sun_direction, dir);

    // Sky
    // Always render it for now
    let always_render_sky = false;
    if (fogDistance == -1.0 || always_render_sky) {
        let fast = false;
        color = skyRay(org, dir, sun_direction, fast);
        var newFogDistance = intersectSphere(org, dir, vec3f(0.0, -EARTH_RADIUS, 0.0), EARTH_RADIUS + FOG_DIST);
        if (always_render_sky) {
            fogDistance = newFogDistance;
        } else {
            fogDistance = select(fogDistance, newFogDistance, fogDistance == -1.0);
        }
    }

    let fogPhase = 0.5 * HenyeyGreenstein(mu, 0.7) + 0.5 * HenyeyGreenstein(mu, -0.6);
    let finalColor = mix(
        fogPhase * 0.1 * LOW_SCATTER * SUN_POWER + 10.0 * vec3f(0.55, 0.8, 1.0), 
        color, 
        exp(-0.0003 * fogDistance)
    ) * 0.06;
    return vec4f(finalColor, 1.0);
    // return vec4f(color * 0.06, 1.0);
}