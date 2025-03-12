// shader.wgsl

// Constants from Shadertoy
const PI = 3.14159265359;
const EARTH_RADIUS = 6300e3;
const CLOUD_START = 800.0;
const CLOUD_HEIGHT = 600.0;
const SUN_POWER = vec3f(1.0, 0.9, 0.6) * 750.0;
const LOW_SCATTER = vec3f(1.0, 0.7, 0.5);
const ITER_GEOMETRY = 3;
const ITER_FRAGMENT = 5;
const SEA_HEIGHT = 0.6;
const SEA_CHOPPY = 4.0;
const SEA_FREQ = 0.16;
const SEA_BASE = vec3f(0.1, 0.21, 0.35) * 8.0;
const albedo = vec3f(0.95, 0.16, 0.015);
const EPSILON_NRM =  0.1 / 800.0; // Hardcoded approximation of Shadertoy's (0.1 / iResolution.x)
const cubeForm = mat3x3(
    vec3f(1.0, 0.0, 0.0),
    vec3f(0.0, 1.0, 0.0),
    vec3f(0.0, 0.0, 1.0)
);
const Yelevation = 0.0;

// Uniforms and Bindings
struct Settings {
    max: f32,
    r_inner: f32,
    r: f32,
    max_height: f32,
    max_water_height: f32,
    water_height: f32,
    tunnel_radius: f32,
    surface_factor: f32,
    camera_speed: f32,
    camera_time_offset: f32,
    voxel_level: i32,
    voxel_size: f32,
    steps: i32,
    max_dist: f32,
    min_dist: f32,
    eps: f32,
    light_color: vec4f,
    light_direction: vec4f,
    show_normals: i32,
    show_steps: i32,
    visualize_distance_field: i32,
    _padding: i32,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_position: vec3f,
    time: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var rgb_noise_texture: texture_2d<f32>;         // iChannel1
@group(1) @binding(1) var gray_noise_texture: texture_2d<f32>;       // iChannel3 (gray Perlin noise)
@group(1) @binding(2) var gray_cube_noise_texture: texture_3d<f32>;  // iChannel2
@group(1) @binding(3) var grain_texture: texture_2d<f32>;
@group(1) @binding(4) var dirt_texture: texture_2d<f32>;
@group(1) @binding(5) var pebble_texture: texture_2d<f32>;           // iChannel0
@group(1) @binding(6) var terrain_sampler: sampler; // Must use repeat mode
@group(2) @binding(0) var<uniform> settings: Settings;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) tex_uv: vec2f,
    @location(2) normal: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) tex_uv: vec2f,
    @location(1) normal: vec3f,
    @location(2) world_position: vec3f,
};

// Vertex Shader (unchanged)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec2f, 4>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, 1.0)
    );
    let tex_coords = array<vec2f, 4>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0),
        vec2f(0.0, 1.0), vec2f(1.0, 1.0)
    );
    var output: VertexOutput;
    output.position = vec4f(positions[vertex_index], 0.0, 1.0);
    output.tex_uv = tex_coords[vertex_index];
    output.normal = vec3f(0.0, 0.0, 1.0);
    output.world_position = vec3f(0.0);
    return output;
}

// Helper Functions (in dependency order)

// **Hash Functions**
fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

fn hash2(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453123);
}

// **Noise Functions**
fn noise3(x: vec3f) -> f32 {
    let p = floor(x);
    let f = fract(x);
    let ff = f * f * (3.0 - 2.0 * f);
    let uv = (p + ff + 0.5) / 32.0;
    return textureSample(gray_cube_noise_texture, terrain_sampler, uv).x;
}

fn noise2(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let ff = f * f * (3.0 - 2.0 * f);
    let uv = (i + ff + vec2f(0.5)) / 64.0;
    return textureSample(gray_noise_texture, terrain_sampler, uv).x * 2.0 - 1.0;
}

// **FBM (Fractional Brownian Motion)**
fn fbm(p: vec3f) -> f32 {
    let m = mat3x3(
        vec3f(0.00, 0.80, 0.60),
        vec3f(-0.80, 0.36, -0.48),
        vec3f(-0.60, -0.48, 0.64)
    );
    var f = 0.5000 * noise3(p);
    var pp = m * p * 2.02;
    f += 0.2500 * noise3(pp);
    pp = m * pp * 2.03;
    f += 0.1250 * noise3(pp);
    return f;
}

// **Sphere Intersection**
fn intersectSphere(origin: vec3f, dir: vec3f, spherePos: vec3f, sphereRad: f32) -> f32 {
    let oc = origin - spherePos;
    let b = 2.0 * dot(dir, oc);
    let c = dot(oc, oc) - sphereRad * sphereRad;
    let disc = b * b - 4.0 * c;
    if (disc < 0.0) {
        return -1.0;
    }
    let sqrt_disc = sqrt(disc);
    let q = (-b + select(sqrt_disc, -sqrt_disc, b < 0.0)) / 2.0;
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

// **Clouds**
fn clouds(p: vec3f, cloudHeight: ptr<function, f32>, fast: bool) -> f32 {
    let atmoHeight = length(p - vec3f(0.0, -EARTH_RADIUS, 0.0)) - EARTH_RADIUS;
    *cloudHeight = clamp((atmoHeight - CLOUD_START) / CLOUD_HEIGHT, 0.0, 1.0);
    var pp = p;
    pp.z += camera.time * 10.3;
    let largeWeather = clamp((textureSample(pebble_texture, terrain_sampler, -0.00005 * pp.zx).x - 0.18) * 5.0, 0.0, 2.0);
    pp.x += camera.time * 8.3;
    let weather = largeWeather * max(0.0, textureSample(pebble_texture, terrain_sampler, 0.0002 * pp.zx).x - 0.28) / 0.72;
    let weatherSmooth = weather * smoothstep(0.0, 0.5, *cloudHeight) * smoothstep(1.0, 0.5, *cloudHeight);
    let cloudShape = pow(weatherSmooth, 0.3 + 1.5 * smoothstep(0.2, 0.5, *cloudHeight));
    if (cloudShape <= 0.0) {
        return 0.0;
    }
    pp.x += camera.time * 12.3;
    var den = max(0.0, cloudShape - 0.7 * fbm(pp * 0.01));
    if (den <= 0.0) {
        return 0.0;
    }
    if (fast) {
        return largeWeather * 0.2 * min(1.0, 5.0 * den);
    }
    pp.y += camera.time * 15.2;
    den = max(0.0, den - 0.2 * fbm(pp * 0.05));
    return largeWeather * 0.2 * min(1.0, 5.0 * den);
}

// **Numerical Mie Fit**
fn numericalMieFit(costh: f32) -> f32 {
    let bestParams = array<f32, 10>(
        9.805233e-06, -6.500000e+01, -5.500000e+01, 8.194068e-01, 1.388198e-01,
        -8.370334e+01, 7.810083e+00, 2.054747e-03, 2.600563e-02, -4.552125e-12
    );
    let p1 = costh + bestParams[3];
    let expValues = exp(vec4f(
        bestParams[1] * costh + bestParams[2],
        bestParams[5] * p1 * p1,
        bestParams[6] * costh,
        bestParams[9] * costh
    ));
    let expValWeight = vec4f(bestParams[0], bestParams[4], bestParams[7], bestParams[8]);
    return dot(expValues, expValWeight);
}

// **Light Ray**
fn lightRay(p: vec3f, phaseFunction: f32, dC: f32, mu: f32, sun_direction: vec3f, cloudHeight: f32, fast: bool) -> f32 {
    let nbSampleLight = select(20, 7, fast);
    let zMaxl = 600.0;
    let stepL = zMaxl / f32(nbSampleLight);
    var lighRayDen = 0.0;
    var pp = p + sun_direction * stepL * hash(dot(p, vec3f(12.256, 2.646, 6.356)) + camera.time);
    for (var j = 0; j < nbSampleLight; j = j + 1) {
        var cloudHeightLocal: f32;
        lighRayDen += clouds(pp + sun_direction * f32(j) * stepL, &cloudHeightLocal, fast);
    }
    if (fast) {
        return (0.5 * exp(-0.4 * stepL * lighRayDen) + max(0.0, -mu * 0.6 + 0.3) * exp(-0.02 * stepL * lighRayDen)) * phaseFunction;
    }
    let scatterAmount = mix(0.008, 1.0, smoothstep(0.96, 0.0, mu));
    let beersLaw = exp(-stepL * lighRayDen) + 0.5 * scatterAmount * exp(-0.1 * stepL * lighRayDen) + scatterAmount * 0.4 * exp(-0.02 * stepL * lighRayDen);
    return beersLaw * phaseFunction * mix(0.05 + 1.5 * pow(min(1.0, dC * 8.5), 0.3 + 5.5 * cloudHeight), 1.0, clamp(lighRayDen * 0.4, 0.0, 1.0));
}

// **Rounded Box Distance**
fn udRoundBox(p: vec3f, b: vec3f, r: f32) -> f32 {
    return length(max(abs(p) - b, vec3f(0.0))) - r;
}

// **Map (Cube)**
fn map(pos: vec3f) -> f32 {
    var p = cubeForm * pos;
    p.y += Yelevation;
    p.y += 3.66;
    p.z += 0.4;
    p *= 0.35;
    var res = udRoundBox(p - vec3f(0.0, 1.25, 0.0), vec3f(0.15), 0.01);
    res = min(res, udRoundBox(p - vec3f(0.33, 1.25, 0.0), vec3f(0.15), 0.01));
    res = min(res, udRoundBox(p - vec3f(0.33, 1.25, 0.33), vec3f(0.15), 0.01));
    res = min(res, udRoundBox(p - vec3f(0.0, 1.25, 0.33), vec3f(0.15), 0.01));
    res = min(res, udRoundBox(p - vec3f(0.0, 1.58, 0.0), vec3f(0.15), 0.01));
    res = min(res, udRoundBox(p - vec3f(0.0, 1.58, 0.33), vec3f(0.15), 0.01));
    res = min(res, udRoundBox(p - vec3f(0.33, 1.58, 0.0), vec3f(0.15), 0.01));
    return res;
}

// **Sea Octave**
fn sea_octave(uv: vec2f, choppy: f32) -> f32 {
    let uvNoise = uv + noise2(uv);
    let wv = 1.0 - abs(sin(uvNoise));
    let swv = abs(cos(uvNoise));
    let wvMix = mix(wv, swv, wv);
    return pow(1.0 - pow(wvMix.x * wvMix.y, 0.65), choppy);
}

// **Water Map**
fn mapWater(p: vec3f, steps: i32, cube: bool) -> f32 {
    var freq = SEA_FREQ;
    var amp = SEA_HEIGHT;
    var choppy = SEA_CHOPPY;
    var uv = p.xz;
    uv.x *= 0.75;
    var d: f32;
    var h = 0.0;
    let SEA_SPEED = 0.8;
    let octave_m = mat2x2(1.6, 1.2, -1.2, 1.6);
    let seaTime = 1.0 + camera.time * SEA_SPEED;
    for (var i = 0; i < steps; i = i + 1) {
        d = sea_octave((uv + seaTime) * freq, choppy);
        d += sea_octave((uv - seaTime) * freq, choppy);
        h += d * amp;
        uv = octave_m * uv;
        freq *= 1.9;
        amp *= 0.22;
        choppy = mix(choppy, 1.0, 0.2);
    }
    if (!cube) {
        return p.y - h;
    }
    return p.y - h - 0.2 * exp(-max(0.0, 23.0 * map(p)));
}

// **Schlick Approximation**
fn Schlick(f0: f32, VoH: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - VoH, 5.0);
}

// **GGX Distribution**
fn D_GGX(r: f32, NoH: f32, h: vec3f) -> f32 {
    let a = NoH * r;
    let k = r / ((1.0 - NoH * NoH) + a * a);
    return k * k * (1.0 / PI);
}

// **Ray Casting for Cube**
fn castRay(ro: vec3f, rd: vec3f, tmin: f32) -> f32 {
    var t_min = tmin;
    var tmax = 10.0;
    let maxY = 3.0;
    let minY = -1.0;
    let tp1 = (minY - ro.y) / rd.y;
    if (tp1 > 0.0) {
        tmax = min(tmax, tp1);
    }
    let tp2 = (maxY - ro.y) / rd.y;
    if (tp2 > 0.0) {
        if (ro.y > maxY) {
            t_min = max(tmin, tp2);
        } else {
            tmax = min(tmax, tp2);
        }
    }
    var t = t_min;
    for (var i = 0; i < 100; i = i + 1) {
        let precis = 0.0005 * t;
        let res = map(ro + rd * t);
        if (res < precis || t > tmax) {
            break;
        }
        t += res;
    }
    if (t > tmax) {
        return -1.0;
    }
    return t;
}

// **Soft Shadow**
fn softshadow(ro: vec3f, rd: vec3f, mint: f32, tmax: f32) -> f32 {
    var res = 1.0;
    var t = mint;
    for (var i = 0; i < 14; i = i + 1) {
        let h = map(ro + rd * t);
        res = min(res, 8.0 * h / t);
        t += clamp(h, 0.08, 0.25);
        if (res < 0.001 || t > tmax) {
            break;
        }
    }
    return max(0.0, res);
}

// **Normal Calculation**
fn calcNormal(pos: vec3f) -> vec3f {
    let e = vec2f(1.0, -1.0) * EPSILON_NRM;
    return normalize(
        e.xyy * map(pos + e.xyy) +
        e.yyx * map(pos + e.yyx) +
        e.yxy * map(pos + e.yxy) +
        e.xxx * map(pos + e.xxx)
    );
}

// **Ambient Occlusion**
fn calcAO(pos: vec3f, nor: vec3f) -> f32 {
    var occ = 0.0;
    var sca = 1.0;
    for (var i = 0; i < 5; i = i + 1) {
        let hr = 0.01 + 0.12 * f32(i) / 4.0;
        let aopos = nor * hr + pos;
        let dd = map(aopos);
        occ += -(dd - hr) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}

// **Henyey-Greenstein Phase Function**
fn HenyeyGreenstein(mu: f32, inG: f32) -> f32 {
    return (1.0 - inG * inG) / (pow(1.0 + inG * inG - 2.0 * inG * mu, 1.5) * 4.0 * PI);
}

// **Sky Ray**
fn skyRay(org: vec3f, dir: vec3f, sun_direction: vec3f, fast: bool) -> vec3f {
    let ATM_START = EARTH_RADIUS + CLOUD_START;
    let ATM_END = ATM_START + CLOUD_HEIGHT;
    let nbSample = select(35, 13, fast);
    var color = vec3f(0.0);
    let distToAtmStart = intersectSphere(org, dir, vec3f(0.0, -EARTH_RADIUS, 0.0), ATM_START);
    let distToAtmEnd = intersectSphere(org, dir, vec3f(0.0, -EARTH_RADIUS, 0.0), ATM_END);
    var p = org + distToAtmStart * dir;
    let stepS = (distToAtmEnd - distToAtmStart) / f32(nbSample);
    var T = 1.0;
    let mu = dot(sun_direction, dir);
    let phaseFunction = numericalMieFit(mu);
    p += dir * stepS * hash(dot(dir, vec3f(12.256, 2.646, 6.356)) + camera.time);
    if (dir.y > 0.015) {
        for (var i = 0; i < nbSample; i = i + 1) {
            var cloudHeight: f32;
            let density = clouds(p, &cloudHeight, fast);
            if (density > 0.0) {
                let intensity = lightRay(p, phaseFunction, density, mu, sun_direction, cloudHeight, fast);
                let ambient = (0.5 + 0.6 * cloudHeight) * vec3f(0.2, 0.5, 1.0) * 6.5 + vec3f(0.8) * max(0.0, 1.0 - 2.0 * cloudHeight);
                let radiance = ambient + SUN_POWER * intensity;
                let radDensity = radiance * density;
                color += T * (radDensity - radDensity * exp(-density * stepS)) / density;
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

fn renderCubeFast(p: vec3f, dir: vec3f, sun_direction: vec3f, res: f32) -> vec3f {
    let pos = p + res * dir;
    let nor = calcNormal(pos);
    let NoL = max(0.0, dot(sun_direction, nor));
    var color = 0.6 * NoL * SUN_POWER * albedo / PI;
    color += albedo * vec3f(0.3, 0.6, 1.0) * 35.0 * (0.75 + 0.25 * nor.y);
    return color;
}

// **Sea Color**
fn getSeaColor(p: vec3f, N: vec3f, sun_direction: vec3f, dir: vec3f, dist: vec3f, mu: f32, cloudShadow: f32) -> vec3f {
    let L = normalize(reflect(dir, N));
    let V = -dir;
    let NoV = clamp(abs(dot(N, V)) + 1e-5, 0.0, 1.0);
    let NoL = max(0.0, dot(N, L));
    let VoH = max(0.0, dot(V, normalize(V + L)));
    let fresnel = Schlick(0.02, NoV);
    let cubeRes = castRay(p, L, 0.0001);
    var reflection = skyRay(p, L, sun_direction, true);
    if (cubeRes != -1.0) {
        reflection = renderCubeFast(p, L, sun_direction, cubeRes);
    }
    var color = mix(cloudShadow * SEA_BASE, reflection, fresnel);
    let subsurfaceAmount = 12.0 * HenyeyGreenstein(mu, 0.5);
    let SEA_WATER_COLOR = 0.6 * vec3f(0.8, 0.9, 0.6);
    color += subsurfaceAmount * SEA_WATER_COLOR * max(0.0, 1.0 + p.y - 0.6 * SEA_HEIGHT);
    if (cubeRes == -1.0) {
        let H = normalize(V + sun_direction);
        let NoL_sun = max(0.0, dot(N, sun_direction));
        let roughness = 0.05;
        color += LOW_SCATTER * 0.4 * (NoL_sun / PI * fresnel * SUN_POWER * D_GGX(roughness, max(0.0, dot(N, H)), H));
    }
    color += 9.0 * max(0.0, smoothstep(0.35, 0.6, p.y - SEA_HEIGHT) * N.x);
    let foamShadow = max(0.0, dot(sun_direction.xz, normalize(p.xz)));
    color += foamShadow * 2.5 * smoothstep(0.06 + 0.06 * N.z, 0.0, map(p)) * max(0.0, N.y);
    return color;
}

// **Water Normal**
fn getNormalWater(p: vec3f, eps: f32) -> vec3f {
    var n: vec3f;
    n.y = mapWater(p, ITER_FRAGMENT, true);
    n.x = mapWater(vec3f(p.x + eps, p.y, p.z), ITER_FRAGMENT, true) - n.y;
    n.z = mapWater(vec3f(p.x, p.y, p.z + eps), ITER_FRAGMENT, true) - n.y;
    n.y = eps;
    return normalize(n);
}

// **Height Map Tracing**
fn heightMapTracing(ori: vec3f, dir: vec3f, p: ptr<function, vec3f>) -> f32 {
    var tm = 0.0;
    var tx = 1e8;
    var hx = mapWater(ori + dir * tx, ITER_GEOMETRY, true);
    if (hx > 0.0) {
        *p = ori + dir * tx;
        return tx;
    }
    var hm = mapWater(ori + dir * tm, ITER_GEOMETRY, true);
    var tmid = 0.0;
    for (var i = 0; i < 8; i = i + 1) {
        tmid = mix(tm, tx, hm / (hm - hx));
        *p = ori + dir * tmid;
        let hmid = mapWater(*p, ITER_GEOMETRY, true);
        if (hmid < 0.0) {
            tx = tmid;
            hx = hmid;
        } else {
            tm = tmid;
            hm = hmid;
        }
    }
    return tmid;
}

// **World Reflection**
fn worldReflection(org: vec3f, dir: vec3f, sun_direction: vec3f) -> vec3f {
    if (castRay(org, dir, 0.05) != -1.0 || dir.y < 0.0) {
        return vec3f(0.0);
    }
    return skyRay(org, dir, sun_direction, true);
}

// **Cube Rendering**
fn renderCube(p: vec3f, dir: vec3f, sun_direction: vec3f, res: f32) -> vec3f {
    let pos = p + res * dir;
    let nor = calcNormal(pos);
    let occ = calcAO(pos, nor);
    let NoL = max(0.0, dot(sun_direction, nor));
    let sunShadow = softshadow(pos, sun_direction, 0.001, 2.0);
    var color = 0.6 * NoL * SUN_POWER * albedo * sunShadow / PI;
    color += albedo * occ * vec3f(0.3, 0.6, 1.0) * 35.0 * (0.75 + 0.25 * nor.y);
    let refDir = reflect(dir, nor);
    let refP = worldReflection(pos, refDir, sun_direction);
    color += Schlick(0.04, max(0.0, dot(nor, -dir))) * refP * max(0.0, occ - 0.7) / 0.7;
    return color;
}

// **Fragment Shader**
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    // Ray origin and direction from camera
    let ro = camera.camera_position;
    let ndc = vec4f(input.tex_uv * 2.0 - 1.0, 1.0, 1.0);
    let world_pos = camera.inv_view_proj * ndc;
    let rd = normalize(world_pos.xyz / world_pos.w - ro);
    let sun_direction = normalize(vec3f(0.6, 0.45, -0.8)); // Fixed as in Shadertoy

    var fogDistance = intersectSphere(ro, rd, vec3f(0.0, -EARTH_RADIUS, 0.0), EARTH_RADIUS);
    let mu = dot(sun_direction, rd);
    let cubeRes = castRay(ro, rd, 2.0);

    var color: vec3f;
    if (fogDistance == -1.0 && cubeRes == -1.0) {
        color = skyRay(ro, rd, sun_direction, false);
        fogDistance = intersectSphere(ro, rd, vec3f(0.0, -EARTH_RADIUS, 0.0), EARTH_RADIUS + 160.0);
    } else if (fogDistance == -1.0 && cubeRes != -1.0) {
        color = renderCube(ro, rd, sun_direction, cubeRes);
        fogDistance = cubeRes;
    } else {
        var waterHitPoint: vec3f;
        heightMapTracing(ro, rd, &waterHitPoint);
        let dist = waterHitPoint - ro;
        let n = getNormalWater(waterHitPoint, dot(dist, dist) * EPSILON_NRM);
        let cloudShadow = 1.0 - textureSample(pebble_texture, terrain_sampler, waterHitPoint.xz * 0.008 - vec2f(0.0, 0.03 * camera.time)).x;
        color = getSeaColor(waterHitPoint, n, sun_direction, rd, dist, mu, cloudShadow);
        if (cubeRes != -1.0) {
            let distT = length(dist);
            if (cubeRes > distT) {
                let refr = refract(rd, n, 0.75);
                let cubeResUnder = castRay(waterHitPoint, refr, 0.001);
                if (cubeResUnder != -1.0) {
                    let cube = renderCube(waterHitPoint, refr, sun_direction, cubeResUnder);
                    let fresnel = 1.0 - Schlick(0.04, max(0.0, dot(n, -rd)));
                    color += fresnel * (cube * vec3f(0.7, 0.8, 0.9) * exp(-0.01 * vec3f(60.0, 15.0, 1.0) * max(0.0, cubeResUnder)) - SEA_BASE * cloudShadow);
                }
            } else {
                color = renderCube(ro, rd, sun_direction, cubeRes);
            }
            fogDistance = cubeRes;
        }
    }

    let fogPhase = 0.5 * HenyeyGreenstein(mu, 0.7) + 0.5 * HenyeyGreenstein(mu, -0.6);
    let finalColor = mix(fogPhase * 0.1 * LOW_SCATTER * SUN_POWER + 10.0 * vec3f(0.55, 0.8, 1.0), color, exp(-0.0003 * fogDistance)) * 0.06;
    return vec4f(finalColor, 1.0);
}