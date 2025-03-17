struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) tex_uv: vec2f,
    @location(2) normal: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) fragCoord: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let positions = array<vec2f, 4>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, 1.0)
    );
    let tex_coords = array<vec2f, 4>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0),
        vec2f(0.0, 1.0), vec2f(1.0, 1.0)
    );
    output.position = vec4f(positions[input.vertex_index], 0.0, 1.0);
    output.fragCoord = tex_coords[input.vertex_index] * uniforms.resolution;
    return output;
}

const PI: f32 = 3.141592;
const EPSILON_NRM: f32 = 0.1 / 1024.0; // Assuming resolution.x is 1024.0

// Cloud parameters
const EARTH_RADIUS: f32 = 6300e3;
const CLOUD_START: f32 = 800.0;
const CLOUD_HEIGHT: f32 = 600.0;
const SUN_POWER: vec3<f32> = vec3<f32>(1.0, 0.9, 0.6) * 750.0;
const LOW_SCATTER: vec3<f32> = vec3<f32>(1.0, 0.7, 0.5);

// Uniforms
struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_position: vec3f,
    time: f32,
    resolution: vec2f,
    _padding: vec2f
};

// Textures and texture_samplers
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var rgb_noise_texture: texture_2d<f32>;
@group(1) @binding(1) var gray_noise_texture: texture_2d<f32>;
@group(1) @binding(2) var gray_cube_noise_texture: texture_3d<f32>; 
@group(1) @binding(3) var grain_texture: texture_2d<f32>; 
@group(1) @binding(4) var dirt_texture: texture_2d<f32>;  
@group(1) @binding(5) var pebble_texture: texture_2d<f32>;  
@group(1) @binding(6) var texture_sampler: sampler; // Must use repeat mode

fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

fn hash_vec2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn noise_3d(p: vec3<f32>) -> f32 {
    let p_floor = floor(p);
    let f = fract(p);
    let f_smooth = f * f * (3.0 - 2.0 * f);
    let coord = (p_floor + f_smooth + 0.5) / 32.0;
    return textureSampleLevel(gray_cube_noise_texture, texture_sampler, coord, 0.0).x;
}

fn noise_2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let f_smooth = f * f * (3.0 - 2.0 * f);
    let coord = (i + f_smooth + 0.5) / 64.0;
    return textureSampleLevel(gray_noise_texture, texture_sampler, coord, 0.0).x * 2.0 - 1.0;
}

fn fbm(p: vec3<f32>) -> f32 {
    let m = mat3x3<f32>(
        vec3<f32>(0.00, -0.80, -0.60),
        vec3<f32>(0.80, 0.36, -0.48),
        vec3<f32>(0.60, -0.48, 0.64)
    );
    var f: f32 = 0.5 * noise_3d(p);
    var q = m * p * 2.02;
    f += 0.25 * noise_3d(q);
    q = m * q * 2.03;
    f += 0.125 * noise_3d(q);
    return f;
}

fn intersectSphere(origin: vec3<f32>, dir: vec3<f32>, spherePos: vec3<f32>, sphereRad: f32) -> f32 {
    let oc = origin - spherePos;
    let b = 2.0 * dot(dir, oc);
    let c = dot(oc, oc) - sphereRad * sphereRad;
    let disc = b * b - 4.0 * c;
    if (disc < 0.0) {
        return -1.0;
    }
    let sqrt_disc = sqrt(disc);
    let q = (-b + select(-sqrt_disc, sqrt_disc, b < 0.0)) / 2.0;
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
    return select(t1, t0, t0 < 0.0);
}

struct CloudResult {
    density: f32,
    cloudHeight: f32,
};

fn clouds(p: vec3<f32>, fast: bool) -> CloudResult {
    let atmoHeight = length(p - vec3<f32>(0.0, -EARTH_RADIUS, 0.0)) - EARTH_RADIUS;
    let cloudHeight = clamp((atmoHeight - CLOUD_START) / CLOUD_HEIGHT, 0.0, 1.0);
    var p_moved = p;
    p_moved.z += uniforms.time * 10.3;
    let largeWeatherTex = textureSampleLevel(pebble_texture, texture_sampler, -0.00005 * p_moved.zx, 0.0).x;
    let largeWeather = clamp((largeWeatherTex - 0.18) * 5.0, 0.0, 2.0);
    p_moved.x += uniforms.time * 8.3;
    let weatherTex = textureSampleLevel(pebble_texture, texture_sampler, 0.0002 * p_moved.zx, 0.0).x;
    let weather = largeWeather * max(0.0, (weatherTex - 0.28) / 0.72);
    let weather_scaled = weather * smoothstep(0.0, 0.5, cloudHeight) * smoothstep(1.0, 0.5, cloudHeight);
    let cloudShape = pow(weather_scaled, 0.3 + 1.5 * smoothstep(0.2, 0.5, cloudHeight));
    if (cloudShape <= 0.0) {
        return CloudResult(0.0, cloudHeight);
    }
    p_moved.x += uniforms.time * 12.3;
    let den = max(0.0, cloudShape - 0.7 * fbm(p_moved * 0.01));
    if (den <= 0.0) {
        return CloudResult(0.0, cloudHeight);
    }
    if (fast) {
        return CloudResult(largeWeather * 0.2 * min(1.0, 5.0 * den), cloudHeight);
    }
    p_moved.y += uniforms.time * 15.2;
    let den_final = max(0.0, den - 0.2 * fbm(p_moved * 0.05));
    return CloudResult(largeWeather * 0.2 * min(1.0, 5.0 * den_final), cloudHeight);
}

fn numericalMieFit(costh: f32) -> f32 {
    const bestParams: array<f32, 10> = array<f32, 10>(
        9.805233e-06, -65.0, -55.0, 0.8194068, 0.1388198,
        -83.70334, 7.810083, 0.002054747, 0.02600563, -4.552125e-12
    );
    let p1 = costh + bestParams[3];
    let expValues = vec4<f32>(
        exp(bestParams[1] * costh + bestParams[2]),
        exp(bestParams[5] * p1 * p1),
        exp(bestParams[6] * costh),
        exp(bestParams[9] * costh)
    );
    let expValWeight = vec4<f32>(
        bestParams[0], bestParams[4], bestParams[7], bestParams[8]
    );
    return dot(expValues, expValWeight);
}

fn lightRay(p: vec3<f32>, phaseFunction: f32, dC: f32, mu: f32, sun_dir: vec3<f32>, cloudHeight: f32, fast: bool) -> f32 {
    let zMaxl: f32 = 600.0;
    let samples = select(20, 7, fast);
    let stepL = zMaxl / f32(samples);
    var lightRayDen: f32 = 0.0;
    var j: i32 = 0;
    while (j < 20) {
        if (j >= samples) { break; }
        let current_p = p + sun_dir * (f32(j) * stepL);
        let cloud_res = clouds(current_p, fast);
        lightRayDen += cloud_res.density;
        j += 1;
    }
    if (fast) {
        return (0.5 * exp(-0.4 * stepL * lightRayDen) + max(0.0, -mu * 0.6 + 0.3) * exp(-0.02 * stepL * lightRayDen)) * phaseFunction;
    }
    let scatterAmount = mix(0.008, 1.0, smoothstep(0.96, 0.0, mu));
    let beersLaw = exp(-stepL * lightRayDen) + 0.5 * scatterAmount * exp(-0.1 * stepL * lightRayDen) + scatterAmount * 0.4 * exp(-0.02 * stepL * lightRayDen);
    return beersLaw * phaseFunction * mix(0.05 + 1.5 * pow(min(1.0, dC * 8.5), 0.3 + 5.5 * cloudHeight), 1.0, clamp(lightRayDen * 0.4, 0.0, 1.0));
}

fn Schlick(f0: f32, VoH: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - VoH, 5.0);
}

fn skyRay(org: vec3<f32>, dir: vec3<f32>, sun_dir: vec3<f32>, fast: bool) -> vec3<f32> {
    const ATM_START: f32 = EARTH_RADIUS + CLOUD_START;
    const ATM_END: f32 = ATM_START + CLOUD_HEIGHT;
    let nbSample = select(35, 13, fast);
    let distToAtmStart = intersectSphere(org, dir, vec3<f32>(0.0, -EARTH_RADIUS, 0.0), ATM_START);
    let distToAtmEnd = intersectSphere(org, dir, vec3<f32>(0.0, -EARTH_RADIUS, 0.0), ATM_END);
    var p = org + distToAtmStart * dir;
    let stepS = (distToAtmEnd - distToAtmStart) / f32(nbSample);
    var T: f32 = 1.0;
    var color = vec3<f32>(0.0);
    let mu = dot(sun_dir, dir);
    let phaseFunction = numericalMieFit(mu);
    p += dir * stepS * hash(dot(dir, vec3<f32>(12.256, 2.646, 6.356)) + uniforms.time);
    if (dir.y > 0.015) {
        var i: i32 = 0;
        while (i < 35) {
            if (i >= nbSample) { break; }
            let cloud_res = clouds(p, fast);
            if (cloud_res.density > 0.0) {
                let intensity = lightRay(p, phaseFunction, cloud_res.density, mu, sun_dir, cloud_res.cloudHeight, fast);
                let ambient = (0.5 + 0.6 * cloud_res.cloudHeight) * vec3<f32>(0.2, 0.5, 1.0) * 6.5 + vec3<f32>(0.8) * max(0.0, 1.0 - 2.0 * cloud_res.cloudHeight);
                let radiance = ambient + SUN_POWER * intensity;
                let contrib = T * (radiance - radiance * exp(-cloud_res.density * stepS)) / cloud_res.density;
                color += contrib;
                T *= exp(-cloud_res.density * stepS);
                if (T <= 0.05) { break; }
            }
            p += dir * stepS;
            i += 1;
        }
    }
    if (!fast) {
        let pC = org + intersectSphere(org, dir, vec3<f32>(0.0, -EARTH_RADIUS, 0.0), ATM_END + 1000.0) * dir;
        color += T * vec3<f32>(3.0) * max(0.0, fbm(vec3<f32>(1.0, 1.0, 1.8) * pC * 0.002) - 0.4);
    }
    let background = 6.0 * mix(vec3<f32>(0.2, 0.52, 1.0), vec3<f32>(0.8, 0.95, 1.0), pow(0.5 + 0.5 * mu, 15.0)) + mix(vec3<f32>(3.5), vec3<f32>(0.0), min(1.0, 2.3 * dir.y));
    color += background * T;
    return color;
}

fn HenyeyGreenstein(mu: f32, g: f32) -> f32 {
    return (1.0 - g * g) / (pow(1.0 + g * g - 2.0 * g * mu, 1.5) * 4.0 * PI);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let q = input.fragCoord.xy / uniforms.resolution;
    let v = -1.0 + 2.0 * q;
    // return  vec4<f32>(v.xy, 0.0, 1.0);
    // let v = (input.fragCoord.xy - uniforms.resolution.xy * 0.5) / uniforms.resolution.xy;
    // return  vec4<f32>(v.xy, 0.0, 1.0);
    let vx = v.x * (uniforms.resolution.x / uniforms.resolution.y);
    let org = vec3<f32>(0.0, 0.0, 0.0);
    let ta = vec3<f32>(0.0, 5.0, 3.0);
    let ww = normalize(ta - org);
    let uu = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), ww));
    let vv = normalize(cross(ww, uu));
    let dir = normalize(vec3<f32>(vx, v.y, 1.4) * mat3x3<f32>(uu, vv, ww));
    let sun_dir = normalize(vec3<f32>(sin(uniforms.time * 1.0) * 10.0, 0.45, -0.8));
    let fogDist = intersectSphere(org, dir, vec3<f32>(0.0, -EARTH_RADIUS, 0.0), EARTH_RADIUS);
    let mu = dot(sun_dir, dir);
    let color = skyRay(org, dir, sun_dir, false);
    let fogDist2 = intersectSphere(org, dir, vec3<f32>(0.0, -EARTH_RADIUS, 0.0), EARTH_RADIUS + 160.0);
    let fogPhase = 0.5 * HenyeyGreenstein(mu, 0.7) + 0.5 * HenyeyGreenstein(mu, -0.6);
    let finalColor = mix(fogPhase * 0.1 * LOW_SCATTER * SUN_POWER + 10.0 * vec3<f32>(0.55, 0.8, 1.0), color, exp(-0.0003 * fogDist2)) * 0.06;
    return vec4<f32>(finalColor, 1.0);
}
