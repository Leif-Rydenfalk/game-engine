// Constants
const PI: f32 = 3.141592;
const EARTH_RADIUS: f32 = 6300e3;
const CLOUD_START: f32 = 800.0;
const CLOUD_HEIGHT: f32 = 600.0;
const SUN_POWER: vec3<f32> = vec3<f32>(1.0, 0.9, 0.6) * 750.0;
const LOW_SCATTER: vec3<f32> = vec3<f32>(1.0, 0.7, 0.5);

// Ocean parameters (defined but not fully used in main)
const ITER_GEOMETRY: i32 = 3;
const ITER_FRAGMENT: i32 = 5;
const SEA_HEIGHT: f32 = 0.6;
const SEA_CHOPPY: f32 = 4.0;
const SEA_FREQ: f32 = 0.16;
const SEA_BASE: vec3<f32> = vec3<f32>(0.1, 0.21, 0.35) * 8.0;

// Cube parameters (not used in main but included for completeness)
const albedo: vec3<f32> = vec3<f32>(0.95, 0.16, 0.015);

// Uniforms
struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

// Textures and Samplers
@group(0) @binding(1)
var texture0: texture_2d<f32>;  // iChannel0 equivalent (weather texture)
@group(0) @binding(2)
var texture2: texture_3d<f32>;  // iChannel2 equivalent (volume texture for noise3)
@group(0) @binding(3)
var sampler0: sampler;

// Noise Functions
fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn noise3(x: vec3<f32>) -> f32 {
    let p = floor(x);
    let f = fract(x);
    let f_smooth = f * f * (3.0 - 2.0 * f);
    // VOLUME_TEXTURES is defined, so use texture sampling
    return textureSampleLevel(texture2, sampler0, (p + f_smooth + 0.5) / 32.0, 0.0).x;
}

fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let f_smooth = f * f * (3.0 - 2.0 * f);
    // NOISE_TEXTURES is not defined, so use procedural noise
    let a = hash2(i + vec2<f32>(0.0, 0.0));
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    let mix1 = mix(a, b, f_smooth.x);
    let mix2 = mix(c, d, f_smooth.x);
    return -1.0 + 2.0 * mix(mix1, mix2, f_smooth.y);
}

fn fbm(p: vec3<f32>) -> f32 {
    let m = mat3x3<f32>(
        vec3<f32>(0.00,  0.80,  0.60),
        vec3<f32>(-0.80,  0.36, -0.48),
        vec3<f32>(-0.60, -0.48,  0.64)
    );
    var f: f32 = 0.0;
    var local_p = p;
    f += 0.5000 * noise3(local_p);
    local_p = m * local_p * 2.02;
    f += 0.2500 * noise3(local_p);
    local_p = m * local_p * 2.03;
    f += 0.1250 * noise3(local_p);
    return f;
}

// Intersection and Scattering Functions
fn intersectSphere(origin: vec3<f32>, dir: vec3<f32>, spherePos: vec3<f32>, sphereRad: f32) -> f32 {
    let oc = origin - spherePos;
    let b = 2.0 * dot(dir, oc);
    let c = dot(oc, oc) - sphereRad * sphereRad;
    let disc = b * b - 4.0 * c;
    if (disc < 0.0) {
        return -1.0;
    }
    let q = (-b + select(sqrt(disc), -sqrt(disc), b < 0.0)) / 2.0;
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

fn numericalMieFit(costh: f32) -> f32 {
    let bestParams = array<f32, 10>(
        9.805233e-06,
        -6.500000e+01,
        -5.500000e+01,
        8.194068e-01,
        1.388198e-01,
        -8.370334e+01,
        7.810083e+00,
        2.054747e-03,
        2.600563e-02,
        -4.552125e-12
    );
    let p1 = costh + bestParams[3];
    let expValues = vec4<f32>(
        exp(bestParams[1] * costh + bestParams[2]),
        exp(bestParams[5] * p1 * p1),
        exp(bestParams[6] * costh),
        exp(bestParams[9] * costh)
    );
    let expValWeight = vec4<f32>(bestParams[0], bestParams[4], bestParams[7], bestParams[8]);
    return dot(expValues, expValWeight);
}

fn HenyeyGreenstein(mu: f32, inG: f32) -> f32 {
    return (1.0 - inG * inG) / (pow(1.0 + inG * inG - 2.0 * inG * mu, 1.5) * 4.0 * PI);
}

// Cloud and Lighting Functions
fn clouds(p: vec3<f32>, cloudHeight: ptr<function, f32>, fast: bool) -> f32 {
    let atmoHeight = length(p - vec3<f32>(0.0, -EARTH_RADIUS, 0.0)) - EARTH_RADIUS;
    *cloudHeight = clamp((atmoHeight - CLOUD_START) / CLOUD_HEIGHT, 0.0, 1.0);
    var local_p = p;
    local_p.z += uniforms.time * 10.3;
    let largeWeather = clamp((textureSampleLevel(texture0, sampler0, -0.00005 * local_p.zx, 0.0).x - 0.18) * 5.0, 0.0, 2.0);
    local_p.x += uniforms.time * 8.3;
    let weather = largeWeather * max(0.0, textureSampleLevel(texture0, sampler0, 0.0002 * local_p.zx, 0.0).x - 0.28) / 0.72;
    let weatherAdjusted = weather * smoothstep(0.0, 0.5, *cloudHeight) * smoothstep(1.0, 0.5, *cloudHeight);
    let cloudShape = pow(weatherAdjusted, 0.3 + 1.5 * smoothstep(0.2, 0.5, *cloudHeight));
    if (cloudShape <= 0.0) {
        return 0.0;
    }
    local_p.x += uniforms.time * 12.3;
    var den = max(0.0, cloudShape - 0.7 * fbm(local_p * 0.01));
    if (den <= 0.0) {
        return 0.0;
    }
    if (fast) {
        return largeWeather * 0.2 * min(1.0, 5.0 * den);
    }
    local_p.y += uniforms.time * 15.2;
    den = max(0.0, den - 0.2 * fbm(local_p * 0.05));
    return largeWeather * 0.2 * min(1.0, 5.0 * den);
}

fn lightRay(p: vec3<f32>, phaseFunction: f32, dC: f32, mu: f32, sun_direction: vec3<f32>, cloudHeight: f32, fast: bool) -> f32 {
    let nbSampleLight = select(7, 20, fast);
    let zMaxl = 600.0;
    let stepL = zMaxl / f32(nbSampleLight);
    var lighRayDen = 0.0;
    var local_p = p + sun_direction * stepL * hash(dot(p, vec3<f32>(12.256, 2.646, 6.356)) + uniforms.time);
    for (var j: i32 = 0; j < nbSampleLight; j = j + 1) {
        var ch: f32;
        lighRayDen += clouds(local_p + sun_direction * f32(j) * stepL, &ch, fast);
    }
    if (fast) {
        return (0.5 * exp(-0.4 * stepL * lighRayDen) + max(0.0, -mu * 0.6 + 0.3) * exp(-0.02 * stepL * lighRayDen)) * phaseFunction;
    }
    let scatterAmount = mix(0.008, 1.0, smoothstep(0.96, 0.0, mu));
    let beersLaw = exp(-stepL * lighRayDen) + 0.5 * scatterAmount * exp(-0.1 * stepL * lighRayDen) + scatterAmount * 0.4 * exp(-0.02 * stepL * lighRayDen);
    return beersLaw * phaseFunction * mix(0.05 + 1.5 * pow(min(1.0, dC * 8.5), 0.3 + 5.5 * cloudHeight), 1.0, clamp(lighRayDen * 0.4, 0.0, 1.0));
}

// Sky Rendering
fn skyRay(org: vec3<f32>, dir: vec3<f32>, sun_direction: vec3<f32>, fast: bool) -> vec3<f32> {
    let ATM_START = EARTH_RADIUS + CLOUD_START;
    let ATM_END = ATM_START + CLOUD_HEIGHT;
    let nbSample = select(13, 35, fast);
    var color = vec3<f32>(0.0);
    let distToAtmStart = intersectSphere(org, dir, vec3<f32>(0.0, -EARTH_RADIUS, 0.0), ATM_START);
    let distToAtmEnd = intersectSphere(org, dir, vec3<f32>(0.0, -EARTH_RADIUS, 0.0), ATM_END);
    var p = org + distToAtmStart * dir;
    let stepS = (distToAtmEnd - distToAtmStart) / f32(nbSample);
    var T = 1.0;
    let mu = dot(sun_direction, dir);
    let phaseFunction = numericalMieFit(mu);
    p += dir * stepS * hash(dot(dir, vec3<f32>(12.256, 2.646, 6.356)) + uniforms.time);
    if (dir.y > 0.015) {
        for (var i: i32 = 0; i < nbSample; i = i + 1) {
            var cloudHeight: f32;
            let density = clouds(p, &cloudHeight, fast);
            if (density > 0.0) {
                let intensity = lightRay(p, phaseFunction, density, mu, sun_direction, cloudHeight, fast);
                let ambient = (0.5 + 0.6 * cloudHeight) * vec3<f32>(0.2, 0.5, 1.0) * 6.5 + vec3<f32>(0.8) * max(0.0, 1.0 - 2.0 * cloudHeight);
                let radiance = ambient + SUN_POWER * intensity;
                let radianceDensity = radiance * density;
                color += T * (radianceDensity - radianceDensity * exp(-density * stepS)) / density;
                T *= exp(-density * stepS);
                if (T <= 0.05) {
                    break;
                }
            }
            p += dir * stepS;
        }
    }
    if (!fast) {
        let pC = org + intersectSphere(org, dir, vec3<f32>(0.0, -EARTH_RADIUS, 0.0), ATM_END + 1000.0) * dir;
        color += T * vec3<f32>(3.0) * max(0.0, fbm(vec3<f32>(1.0, 1.0, 1.8) * pC * 0.002) - 0.4);
    }
    var background = 6.0 * mix(vec3<f32>(0.2, 0.52, 1.0), vec3<f32>(0.8, 0.95, 1.0), pow(0.5 + 0.5 * mu, 15.0)) + mix(vec3<f32>(3.5), vec3<f32>(0.0), min(1.0, 2.3 * dir.y));
    if (!fast) {
        background += T * vec3<f32>(1e4 * smoothstep(0.9998, 1.0, mu));
    }
    color += background * T;
    return color;
}

// Main Fragment Shader
@fragment
fn main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    // Compute EPSILON_NRM dynamically since it depends on resolution
    let EPSILON_NRM = 0.1 / uniforms.resolution.x;

    let q = fragCoord.xy / uniforms.resolution;
    var v = -1.0 + 2.0 * q;
    v.x *= uniforms.resolution.x / uniforms.resolution.y;

    let camRot = -7.0;
    let org = vec3<f32>(6.0 * cos(camRot), mix(2.2, 10.0, 1.0), 6.0 * sin(camRot));
    let ta = vec3<f32>(0.0, mix(1.2, 12.0, 1.3), 0.0);
    let ww = normalize(ta - org);
    let uu = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), ww));
    let vv = normalize(cross(ww, uu));
    let dir = normalize(v.x * uu + v.y * vv + 1.4 * ww);

    var color = vec3<f32>(0.0);
    let sun_direction = normalize(vec3<f32>(0.6, 0.45, -0.8));
    var fogDistance = intersectSphere(org, dir, vec3<f32>(0.0, -EARTH_RADIUS, 0.0), EARTH_RADIUS);
    let mu = dot(sun_direction, dir);

    if (fogDistance == -1.0) {
        color = skyRay(org, dir, sun_direction, false);
        fogDistance = intersectSphere(org, dir, vec3<f32>(0.0, -EARTH_RADIUS, 0.0), EARTH_RADIUS + 160.0);
    }

    let fogPhase = 0.5 * HenyeyGreenstein(mu, 0.7) + 0.5 * HenyeyGreenstein(mu, -0.6);
    let fogColor = 0.06 * mix(fogPhase * 0.1 * LOW_SCATTER * SUN_POWER + 10.0 * vec3<f32>(0.55, 0.8, 1.0), color, exp(-0.0003 * fogDistance));
    return vec4<f32>(fogColor, 1.0);
}