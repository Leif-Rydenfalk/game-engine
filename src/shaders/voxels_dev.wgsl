// voxel.wgsl

const PI = 3.14159265359;
const betaR = vec3f(5.5e-6, 13.0e-6, 22.4e-6);
const betaM = vec3f(21e-6);
const hR = 7994.0;
const hM = 1200.0;
const earth_radius = 6360e3;
const atmosphere_radius = 6420e3;
const sun_power = 60.0;
const g = 0.76;

const num_samples_light = 8;

const EARTH_RADIUS: f32 = 6300e3;
const CLOUD_START: f32 = 800.0;
const CLOUD_HEIGHT: f32 = 600.0;
const SUN_POWER: vec3f = vec3f(1.0, 0.9, 0.6) * 750.0;
const LOW_SCATTER: vec3f = vec3f(1.0, 0.7, 0.5);

// --- CameraUniform Struct (Matches Rust) ---
struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_position: vec3f,
    time: f32,
    // Implicit padding handled by WGSL layout rules
};

// --- SettingsUniform Struct (Matches Rust) ---
struct SettingsUniform {
    surface_threshold: f32,
    max_terrain_height: f32,
    voxel_size: f32,
    max_ray_steps: i32,
    // --- Auto Padding to 16 bytes --- (Implicit)
    max_distance: f32,
    min_distance: f32, // Unused in shader logic currently
    ray_march_epsilon: f32,
    water_level: f32,
    // --- Auto Padding to 32 bytes --- (Implicit)
    max_water_influenced_height: f32,
    // --- Auto Padding to 48 bytes --- (Implicit, vec4 alignment)
    light_color: vec4f,
    light_direction: vec4f,
    // --- Members after light_direction ---
    show_normals: i32,
    show_ray_steps: i32,
    visualize_distance_field: i32,
        // Implicit padding handled by WGSL layout rules
};

// --- Bindings ---
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var rgb_noise_texture: texture_2d<f32>;
@group(1) @binding(1) var gray_noise_texture: texture_2d<f32>;
@group(1) @binding(2) var gray_cube_noise_texture: texture_3d<f32>;
@group(1) @binding(3) var grain_texture: texture_2d<f32>;
@group(1) @binding(4) var dirt_texture: texture_2d<f32>;
@group(1) @binding(5) var pebble_texture: texture_2d<f32>;
@group(1) @binding(6) var texture_sampler: sampler;
@group(2) @binding(0) var<uniform> settings: SettingsUniform; // Use the new struct name


// --- Helper Structs and Functions (Atmosphere/Clouds) ---

struct Ray {
    origin: vec3f,
    direction: vec3f,
};

struct Sphere {
    origin: vec3f,
    radius: f32,
};

struct IsectResult {
    hit: bool,
    t0: f32,
    t1: f32,
};

const atmosphere = Sphere(vec3f(0.0, 0.0, 0.0), atmosphere_radius);

fn isect_sphere(ray: Ray, sphere: Sphere) -> IsectResult {
    let rc = sphere.origin - ray.origin;
    let radius2 = sphere.radius * sphere.radius;
    let tca = dot(rc, ray.direction);
    let d2 = dot(rc, rc) - tca * tca;
    if d2 > radius2 {
        return IsectResult(false, 0.0, 0.0);
    }
    let thc = sqrt(radius2 - d2);
    let t0 = tca - thc;
    let t1 = tca + thc;
    return IsectResult(true, t0, t1);
}

fn rayleigh_phase_func(mu: f32) -> f32 {
    return 3.0 * (1.0 + mu * mu) / (16.0 * PI);
}

fn henyey_greenstein_phase_func(mu: f32) -> f32 {
    return (1.0 - g * g) / (4.0 * PI * pow(1.0 + g * g - 2.0 * g * mu, 1.5));
}

fn get_sun_light(ray: Ray, optical_depthR: ptr<function, f32>, optical_depthM: ptr<function, f32>) -> bool {
    let isect = isect_sphere(ray, atmosphere);
    if !isect.hit {
        return false;
    }
    let t1 = isect.t1;
    let march_step = t1 / f32(num_samples_light);
    var march_pos = 0.0;
    for (var i = 0; i < num_samples_light; i = i + 1) {
        let s = ray.origin + ray.direction * (march_pos + 0.5 * march_step);
        let height = length(s) - earth_radius;
        if height < 0.0 {
            return false;
        }
        *optical_depthR += exp(-height / hR) * march_step;
        *optical_depthM += exp(-height / hM) * march_step;
        march_pos += march_step;
    }
    return true;
}

fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

fn noise3(x: vec3f) -> f32 {
    let p = floor(x);
    let f = fract(x);
    let f_smooth = f * f * (3.0 - 2.0 * f);
    return textureSampleLevel(gray_cube_noise_texture, texture_sampler, (p + f_smooth + 0.5) / 32.0, 0.0).x;
}

fn noise2(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let f_smooth = f * f * (3.0 - 2.0 * f);
    let tex_coord = (i + f_smooth + 0.5) / 256.0;
    let sample = textureSampleLevel(rgb_noise_texture, texture_sampler, tex_coord, 0.0).x;
    return sample;
}

fn fbm(p: vec3f) -> f32 {
    let m = mat3x3<f32>(
        vec3f(0.00,  0.80,  0.60),
        vec3f(-0.80,  0.36, -0.48),
        vec3f(-0.60, -0.48,  0.64)
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
    return select(t1, t0, t0 >= 0.0);
}

fn clouds(p: vec3f, cloudHeight: ptr<function, f32>, fast: bool) -> f32 {
    let atmoHeight = length(p - vec3f(0.0, -EARTH_RADIUS, 0.0)) - EARTH_RADIUS;
    *cloudHeight = clamp((atmoHeight - CLOUD_START) / CLOUD_HEIGHT, 0.0, 1.0);
    var local_p = p;
    local_p.z += camera.time * 10.3;
    let largeWeather = clamp((textureSampleLevel(pebble_texture, texture_sampler, -0.00005 * local_p.zx, 0.0).x - 0.18) * 5.0, 0.0, 2.0);
    local_p.x += camera.time * 8.3;
    let weather = largeWeather * max(0.0, textureSampleLevel(pebble_texture, texture_sampler, 0.0002 * local_p.zx, 0.0).x - 0.28) / 0.72;
    let weatherAdjusted = weather * smoothstep(0.0, 0.5, *cloudHeight) * smoothstep(1.0, 0.5, *cloudHeight);
    let cloudShape = pow(weatherAdjusted, 0.3 + 1.5 * smoothstep(0.2, 0.5, *cloudHeight));
    if (cloudShape <= 0.0) {
        return 0.0;
    }
    local_p.x += camera.time * 12.3;
    var den = max(0.0, cloudShape - 0.7 * fbm(local_p * 0.01));
    if (den <= 0.0) {
        return 0.0;
    }
    if (fast) {
        return largeWeather * 0.2 * min(1.0, 5.0 * den);
    }
    local_p.y += camera.time * 15.2;
    den = max(0.0, den - 0.2 * fbm(local_p * 0.05));
    return largeWeather * 0.2 * min(1.0, 5.0 * den);
}

fn lightRay(p: vec3f, phaseFunction: f32, dC: f32, mu: f32, sun_direction: vec3f, cloudHeight: f32, fast: bool) -> f32 {
    let nbSampleLight = select(7, 30, fast);
    let zMaxl = 600.0;
    let stepL = zMaxl / f32(nbSampleLight);
    var lighRayDen = 0.0;
    let hash_val = hash(dot(p, vec3f(12.256, 2.646, 6.356)) + camera.time);
    var local_p = p + sun_direction * stepL * hash_val;
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

fn skyRay(org: vec3f, dir: vec3f, sun_direction: vec3f, fast: bool) -> vec3f {
    let ATM_START = EARTH_RADIUS + CLOUD_START;
    let ATM_END = ATM_START + CLOUD_HEIGHT;
    let nbSample = select(13, 35, fast);
    var color = vec3f(0.0);
    let distToAtmStart = intersectSphere(org, dir, vec3f(0.0, -EARTH_RADIUS, 0.0), ATM_START);
    let distToAtmEnd = intersectSphere(org, dir, vec3f(0.0, -EARTH_RADIUS, 0.0), ATM_END);
    if (distToAtmEnd <= 0.0) {
        return vec3f(0.0, 0.0, 0.0);
    }
    let march_start = max(distToAtmStart, 0.0);
    let stepS = (distToAtmEnd - march_start) / f32(nbSample);
    var p = org + dir * march_start;
    var T = 1.0;
    let mu = dot(sun_direction, dir);
    let phaseFunction = numericalMieFit(mu);
    //let phaseFunction = henyey_greenstein_phase_func(mu);
    let hash_val = hash(dot(dir, vec3f(12.256, 2.646, 6.356)) + camera.time);
    p += dir * stepS * hash_val;
    if (dir.y > 0.015) {
        for (var i: i32 = 0; i < nbSample; i = i + 1) {
            var cloudHeight: f32;
            let density = clouds(p, &cloudHeight, fast);
            if (density > 0.0) {
                let intensity = lightRay(p, phaseFunction, density, mu, sun_direction, cloudHeight, fast);
                let ambient = (0.5 + 0.6 * cloudHeight) * vec3f(0.2, 0.5, 1.0) * 6.5 + vec3f(0.8) * max(0.0, 1.0 - 2.0 * cloudHeight);
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
    let background = mix(vec3f(0.2, 0.52, 1.0), vec3f(0.8, 0.95, 1.0), pow(0.5 + 0.5 * mu, 15.0));
    color += background * T;
    return color;
}


// --- Voxel Rendering ---

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

struct HitInfo {
    is_hit: bool,
    t: f32,
    n: vec3f,
    id: vec3f,
    i: i32,
};

fn smin(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

fn smax(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

fn hash13(p: vec3f) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn triplanarLod(p: vec3f, n: vec3f, k: f32, tex_index: i32, lod: f32) -> vec3f {
    let n_pow = pow(abs(n), vec3f(k));
    let n_norm = n_pow / dot(n_pow, vec3f(1.0));
    var col = vec3f(0.0);
    if tex_index == 0 { // rgb_noise_texture
        col = textureSampleLevel(rgb_noise_texture, texture_sampler, p.yz, lod).rgb * n_norm.x +
              textureSampleLevel(rgb_noise_texture, texture_sampler, p.xz, lod).rgb * n_norm.y +
              textureSampleLevel(rgb_noise_texture, texture_sampler, p.xy, lod).rgb * n_norm.z;
    } else if tex_index == 2 { // grain_texture
        col = textureSampleLevel(grain_texture, texture_sampler, p.yz, lod).rgb * n_norm.x +
              textureSampleLevel(grain_texture, texture_sampler, p.xz, lod).rgb * n_norm.y +
              textureSampleLevel(grain_texture, texture_sampler, p.xy, lod).rgb * n_norm.z;
    } else if tex_index == 3 { // dirt_texture
        col = textureSampleLevel(dirt_texture, texture_sampler, p.yz, lod).rgb * n_norm.x +
              textureSampleLevel(dirt_texture, texture_sampler, p.xz, lod).rgb * n_norm.y +
              textureSampleLevel(dirt_texture, texture_sampler, p.xy, lod).rgb * n_norm.z;
    }
    return col;
}

// Distance Function (SDF) for the terrain
fn map(p: vec3f) -> f32 {
    var d: f32 = settings.max_distance; // Use max_distance from settings
    let sc: f32 = 0.3;
    let q: vec3f = sc * p / 32.0 - vec3f(0.003, -0.006, 0.0);

    // Sample 3D noise texture for base terrain shape
    d = textureSample(gray_cube_noise_texture, texture_sampler, q * 1.0).r * 0.5;
    d += textureSample(gray_cube_noise_texture, texture_sampler, q * 2.0 + vec3f(0.3, 0.3, 0.3)).r * 0.25;
    d += textureSample(gray_cube_noise_texture, texture_sampler, q * 4.0 + vec3f(0.7, 0.7, 0.7)).r * 0.125;

    // Normalize the noise value and apply surface threshold
    // Apply vertical falloff (smoother transition near y=0)
    var tp = smoothstep(50.0, -6.0, p.y);
    tp = tp * tp; // Make falloff faster
    d = (d / 0.875 - settings.surface_threshold) / sc; // Apply threshold

    // Combine terrain SDF with a plane representing max height
    d = smax(d, p.y - settings.max_terrain_height, 0.6); // Use max_terrain_height

    return d;
}

// Calculate the normal vector (gradient) of the distance field
fn grad(p: vec3f) -> vec3f {
    let e = vec2f(0.0, 0.1); // Small offset for finite differencing
    // Sample map function at slightly offset points to estimate gradient
    return (map(p) - vec3f(
        map(p - e.yxx), // Offset in x
        map(p - e.xyx), // Offset in y
        map(p - e.xxy)  // Offset in z
    )) / e.y; // Normalize by offset amount
}

// Get the center position of the voxel containing point p
fn get_voxel_pos(p: vec3f, s: f32) -> vec3f {
    return (floor(p / s) + 0.5) * s;
}

// Ray marching function with voxel acceleration
fn trace(ro: vec3f, rd: vec3f, tmax: f32) -> HitInfo {
    let s = settings.voxel_size;      // Voxel size from settings
    let sd = s * sqrt(3.0);         // Voxel diagonal length
    let ird = 1.0 / rd;             // Inverse ray direction
    let srd = sign(ird);            // Sign of inverse direction components
    let ard = abs(ird);             // Absolute value of inverse direction components
    var t = 0.0;                    // Current distance along the ray
    var vpos = get_voxel_pos(ro, s); // Starting voxel position
    var voxel = false;              // Flag indicating if we are inside voxel traversal mode
    var vi = 0;                     // Counter for steps within voxel mode
    var prd = vec3f(0.0);           // Normal of the last voxel face crossed

    for (var i = 0; i < settings.max_ray_steps; i = i + 1) { // Use max_ray_steps
        let pos = ro + rd * t;        // Current position along the ray
        let d = map(select(pos, vpos, voxel)); // SDF value (use voxel pos if in voxel mode)

        if !voxel { // Sphere tracing mode
            t += d; // Advance ray by SDF distance
            if d < sd { // If close enough to potentially enter a voxel
                vpos = get_voxel_pos(ro + rd * max(t - sd, 0.0), s); // Calculate voxel pos
                voxel = true; // Switch to voxel traversal mode
                vi = 0;       // Reset voxel step counter
            }
        } else { // Voxel traversal mode
            // Calculate distance to next voxel boundary along each axis
            let n = (ro - vpos) * ird;
            let k = ard * s * 0.5;
            let t2 = -n + k;
            // Find minimum distance to the next voxel boundary
            let tF = min(min(t2.x, t2.y), t2.z);
            var nrd = vec3f(0.0); // Normal of the face we are crossing

            // Determine which face is being crossed
            if t2.x <= t2.y && t2.x <= t2.z {
                nrd = vec3f(srd.x, 0.0, 0.0);
            } else if t2.y <= t2.z {
                nrd = vec3f(0.0, srd.y, 0.0);
            } else {
                nrd = vec3f(0.0, 0.0, srd.z);
            }

            if d < 0.0 { // If SDF is negative (inside the surface)
                // Hit found! Return info, using the *previous* face normal as the surface normal
                return HitInfo(true, t, -prd, vpos, i);
            } else if d > sd && vi > 2 { // If SDF is large (far from surface) and we've taken steps
                voxel = false; // Switch back to sphere tracing
                t = tF + sd;   // Jump ahead slightly past the voxel boundary
                continue;      // Restart loop in sphere tracing mode
            }

            // Advance to the next voxel
            vpos += nrd * s;
            prd = nrd; // Store the normal of the face we just crossed
            t = tF + settings.ray_march_epsilon; // Advance time slightly past the boundary (use epsilon)
            vi += 1; // Increment voxel step counter
        }

        // Check termination conditions
        if t >= tmax || (rd.y > 0.0 && pos.y > settings.max_terrain_height) { // Use max_terrain_height
            return HitInfo(false, t, vec3f(0.0), vec3f(0.0), i); // Miss
        }
    }
    // Max steps reached
    return HitInfo(false, tmax, vec3f(0.0), vec3f(0.0), settings.max_ray_steps); // Use max_ray_steps
}

// Triplanar mapping function (without LOD for simpler shading)
fn triplanar(p: vec3f, n: vec3f, k: f32, tex_index: i32) -> vec3f {
    let n_pow = pow(abs(n), vec3f(k));
    let n_norm = n_pow / dot(n_pow, vec3f(1.0)); // Blend weights based on normal
    var col = vec3f(0.0);

    // Sample appropriate texture based on index and blend using weights
    if tex_index == 0 { // rgb_noise_texture
        col = textureSample(rgb_noise_texture, texture_sampler, p.yz).rgb * n_norm.x +
              textureSample(rgb_noise_texture, texture_sampler, p.xz).rgb * n_norm.y +
              textureSample(rgb_noise_texture, texture_sampler, p.xy).rgb * n_norm.z;
    } else if tex_index == 2 { // grain_texture
        col = textureSample(grain_texture, texture_sampler, p.yz).rgb * n_norm.x +
              textureSample(grain_texture, texture_sampler, p.xz).rgb * n_norm.y +
              textureSample(grain_texture, texture_sampler, p.xy).rgb * n_norm.z;
    } else if tex_index == 3 { // dirt_texture
        col = textureSample(dirt_texture, texture_sampler, p.yz).rgb * n_norm.x +
              textureSample(dirt_texture, texture_sampler, p.xz).rgb * n_norm.y +
              textureSample(dirt_texture, texture_sampler, p.xy).rgb * n_norm.z;
    }
    return col;
}

// Determine biome blend factors (snow, desert) based on position
fn getBiome(pos: vec3f) -> vec2f {
    // Sample dirt texture at different scales/offsets for biome variation
    let snow = textureSample(dirt_texture, texture_sampler, pos.xz * 0.00015).r;
    let desert = textureSample(dirt_texture, texture_sampler, vec2f(0.55) - pos.zx * 0.00008).g;
    // Use smoothstep for smooth transitions between biomes
    return vec2f(smoothstep(0.67, 0.672, desert), smoothstep(0.695, 0.7, snow));
}

// Calculate the surface albedo (color) based on position, normal, LOD, and biomes
fn getAlbedo(vpos: vec3f, gn: vec3f, lod: f32) -> vec3f {
    // Base rock/grain texture
    var alb = vec3f(1.0) - triplanarLod(vpos * 0.08, gn, 4.0, 2, lod); // Sample grain tex
    alb *= alb; // Square for contrast

    // Base dirt/soil texture
    var alb2 = vec3f(1.0) - triplanarLod(vpos * 0.08, gn, 4.0, 3, lod); // Sample dirt tex
    alb2 *= alb2; // Square for contrast

    // Blend factor based on large-scale noise (controls dirt patches)
    let k = triplanarLod(vpos * 0.0005, gn, 4.0, 0, 0.0).r; // Sample noise tex

    // Blend factor based on height relative to water influence level
    let wk = smoothstep(settings.max_water_influenced_height, settings.max_water_influenced_height + 0.5, vpos.y); // Use max_water_influenced_height

    // Blend factor based on surface slope (more grass/dirt on flatter areas)
    let top = smoothstep(0.3, 0.7, gn.y);

    // Initial rock color tint
    alb = alb * 0.95 * vec3f(1.0, 0.7, 0.65) + 0.05;

    // Mix in green grass/moss on flatter areas above water influence level
    alb = mix(alb, alb2 * vec3f(0.55, 1.0, 0.1), top * wk);

    // Mix in dirt patches based on large-scale noise (more prevalent on slopes)
    alb = mix(alb, smoothstep(vec3f(0.0), vec3f(1.0), alb2), smoothstep(0.3, 0.25, k) * (1.0 - top));

    // Get biome blend factors
    let biome = getBiome(vpos);

    // Snow color definition (mixes dirt texture with blue tint)
    var snow = alb2 * 0.8 + 0.2 * vec3f(0.25, 0.5, 1.0);
    snow = mix(snow, vec3f(0.85, 0.95, 1.0), top * wk * 0.5); // Lighter snow on flatter areas

    // Mix in desert color based on biome factor
    alb = mix(alb, clamp(vec3f(1.0, 0.95, 0.9) - alb2 * 0.65, vec3f(0.0), vec3f(1.0)), biome.x);

    // Mix in snow color based on biome factor
    alb = mix(alb, snow * 2.0, biome.y);

    // Darker soil color definition (varies slightly by biome)
    var dcol = vec3f(0.8, 0.55, 0.35);
    dcol = mix(dcol, vec3f(0.8, 0.65, 0.4), biome.x); // Desert soil tint
    dcol = mix(dcol, vec3f(0.2, 0.6, 0.8), biome.y); // Snow/cold soil tint

    // Mix in darker soil color below water influence level and based on noise/biomes
    alb = mix(alb, alb * dcol, (1.0 - wk) * mix(1.0 - smoothstep(0.3, 0.25, k), 1.0, max(biome.x, biome.y)));

    return alb;
}

// Calculate shading for the terrain surface
fn shade(pos: vec3f, ldir: vec3f, lod: f32, hit: HitInfo) -> vec3f {
    let vpos = hit.id;   // Voxel position (center) where hit occurred
    let g = grad(vpos); // Gradient at voxel center (approximates normal)
    let gn = g / length(g); // Normalized gradient
    let n = hit.n;      // Voxel face normal (used for lighting/shadows)

    // Basic diffuse lighting (Lambertian)
    var dif = max(dot(n, ldir), 0.0);

    // Simple shadow check (trace ray towards light source)
    if dif > 0.0 {
        let hitL = trace(pos + n * 1e-3, ldir, 12.0); // Trace from slightly offset pos
        if hitL.is_hit { dif = 0.0; } // If hit, point is in shadow
    }

    // Get base surface color
    var col = getAlbedo(vpos, gn, lod);

    // Ambient occlusion approximation (darker in crevices)
    let ao = smoothstep(-0.08, 0.04, map(pos) / length(grad(pos)));

    // Height-based ambient occlusion (darker near water level)
    let hao = smoothstep(settings.water_level - 12.0, settings.water_level, pos.y); // Use water_level

    // Combine lighting components
    col *= (dif * 0.6 + 0.4) * settings.light_color.rgb; // Apply diffuse + ambient light color
    col *= (ao * 0.6 + 0.4) * (hao * 0.6 + 0.4); // Apply AO factors

    return col;
}

// Alternate shading function (e.g., for reflections) - simpler version
fn shade2(pos: vec3f, ldir: vec3f, lod: f32, hit: HitInfo) -> vec3f {
    let vpos = hit.id;
    let g = grad(vpos);
    let gn = g / length(g);
    let n = hit.n;
    let dif = max(dot(n, ldir), 0.0);
    var col = getAlbedo(vpos, gn, lod); // Get base color

    // AO approximations
    let ao = smoothstep(-0.08, 0.04, map(pos) / length(grad(pos)));
    let hao = smoothstep(settings.water_level - 12.0, settings.water_level, pos.y); // Use water_level

    // Add slight normal-based variation (emphasizes voxel faces)
    col *= dot(abs(n), vec3f(0.8, 1.0, 0.9));

    // Combine lighting
    col *= (dif * 0.6 + 0.4) * settings.light_color.rgb;
    col *= ao * 0.6 + 0.4;
    col *= hao * 0.6 + 0.4;

    return col;
}

// Function to get water wave height (currently returns flat)
fn get_water_height(wuv: vec2f) -> f32 {
    // Placeholder: could sample a noise texture here for waves
    // return textureSample(grain_texture, texture_sampler, wuv).r;
    return 0.0; // Flat water
}

// --- Vertex Shader ---
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Simple fullscreen triangle/quad setup
    let positions = array<vec2f, 4>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, 1.0)
    );
    let tex_coords = array<vec2f, 4>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0),
        vec2f(0.0, 1.0), vec2f(1.0, 1.0)
    );

    var output: VertexOutput;
    // Output clip space position (covering the screen)
    output.position = vec4f(positions[vertex_index], 0.0, 1.0);
    // Pass texture coordinates to fragment shader
    output.tex_uv = tex_coords[vertex_index];
    // Unused normal/world_position for this simple vertex shader
    output.normal = vec3f(0.0, 0.0, 1.0);
    output.world_position = vec3f(0.0);
    return output;
}

// --- Fragment Shader ---
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    // Get camera origin (position)
    let ro = camera.camera_position;

    // Calculate world space ray direction from UV coordinates
    var ndc = vec4f(input.tex_uv * 2.0 - 1.0, 1.0, 1.0); // Normalized Device Coordinates
    let flip_y = true; // Flip Y if needed based on coordinate system
    if (flip_y) {
        ndc.y = -ndc.y;
    }
    // Unproject NDC to world space
    let world_pos = camera.inv_view_proj * ndc;
    // Calculate ray direction
    let rd = normalize(world_pos.xyz / world_pos.w - ro);

    // Get normalized sun direction from settings
    let sun_dir = normalize(settings.light_direction.xyz);

    // --- Debug Visualization: Distance Field ---
    if settings.visualize_distance_field != 0 {
        let pos = ro + rd * 10.0; // Sample point along ray
        let d = map(pos);         // Get SDF value
        return vec4f(vec3f(d * 0.1 + 0.5), 1.0); // Visualize SDF value as grayscale
    }

    // --- Primary Ray Marching ---
    let hit = trace(ro, rd, settings.max_distance); // Use max_distance
    var col = vec3f(0.0); // Final pixel color
    var t = hit.t;        // Distance to hit point (or max_distance if miss)

    if hit.is_hit {
        // --- Terrain Hit ---
        let pos = ro + rd * hit.t; // Calculate hit position
        // Calculate Level of Detail based on distance to voxel center
        let lod = clamp(log2(distance(ro, hit.id)) - 2.0, 0.0, 6.0);
        // Shade the terrain surface
        col = shade(pos, sun_dir, lod, hit);
    } else {
        // --- Sky Hit ---
        // Calculate sky/cloud color using atmospheric scattering simulation
        col = skyRay(ro, rd, sun_dir, false); // 'false' for higher quality clouds
        t = settings.max_distance; // Set t to max distance for water intersection logic
    }

    // --- Water Intersection and Shading ---
    // Calculate distance to intersection with the water plane (y = water_level)
    let pt = -(ro.y - settings.water_level) / rd.y; // Use water_level

    // Check if water plane is hit *before* the terrain/sky OR if camera is underwater
    if ((pt > 0.0 && pt < t) || ro.y < settings.water_level) {

        // If ray missed terrain initially (hit sky), determine water color based on biome
        if !hit.is_hit {
            let biome = getBiome(ro + rd * pt);
            // Basic water color, slightly tinted by desert biome
            col = mix(vec3f(0.5, 0.8, 1.0), vec3f(1.0, 0.85, 0.6), biome.x);
        }

        // Determine base water color based on biomes
        let biome = getBiome(ro + rd * pt);
        var wcol = vec3f(0.3, 0.8, 1.0); // Default blue water
        wcol = mix(wcol, vec3f(0.4, 0.9, 0.8), biome.x); // Desert biome tint (greener)
        wcol = mix(wcol, vec3f(0.1, 0.7, 0.9), biome.y); // Snow biome tint (darker blue)
        let wabs = vec3f(0.1, 0.7, 0.9); // Water absorption color (for underwater fog)

        // Adjust intersection distance if camera is underwater and looking up
        var adjusted_pt = pt;
        if ro.y < settings.water_level && pt < 0.0 {
            adjusted_pt = settings.max_distance; // Effectively no surface intersection visible
        }

        // Calculate water surface position
        let wpos = ro + rd * adjusted_pt;

        // Calculate water surface normal using finite differencing on wave height function
        let e = 0.001; // Small offset
        let wnstr = 1500.0; // Wave normal strength (exaggerates wave steepness)
        let wo = vec2f(1.0, 0.8) * camera.time * 0.01; // Time-based offset for animation
        let wuv = wpos.xz * 0.08 + wo; // UV coordinates for wave sampling
        let wh = get_water_height(wuv); // Base wave height
        let whdx = get_water_height(wuv + vec2f(e, 0.0)); // Height offset in x
        let whdy = get_water_height(wuv + vec2f(0.0, e)); // Height offset in y
        // Calculate normal from height differences
        let wn = normalize(vec3f(wh - whdx, e * wnstr, wh - whdy));

        // Calculate reflection vector
        let wref = reflect(rd, wn);

        // Calculate reflection color
        var rcol = vec3f(0.0);
        if ro.y > settings.water_level { // Only calculate reflection if camera is above water
            // Trace reflection ray
            let hitR = trace(wpos + vec3f(0.0, 0.01, 0.0), wref, 15.0); // Start slightly above surface
            let lod = clamp(log2(distance(ro, hitR.id)) - 2.0, 0.0, 6.0); // LOD for reflection
            if hitR.is_hit {
                // Shade reflected terrain
                rcol = shade2(wpos, sun_dir, lod, hitR); // Use simpler shading for reflection
            } else {
                // Calculate reflected sky color
                rcol = skyRay(wpos, wref, sun_dir, false); // High quality clouds in reflection
            }
        }

        // Calculate specular highlight
        let spec = pow(max(dot(wref, sun_dir), 0.0), 50.0);

        // Calculate Fresnel term (reflection amount based on view angle)
        let r0 = 0.35; // Base reflectivity
        var fre = r0 + (1.0 - r0) * pow(max(dot(rd, wn), 0.0), 5.0); // Schlick's approximation
        // No reflection if looking up from underwater
        if rd.y < 0.0 && ro.y < settings.water_level {
            fre = 0.0;
        }

        // Calculate distance traveled underwater (for absorption/fog)
        // Selects distance based on whether camera is above or below water
        let abt = select(t - pt, min(t, pt), ro.y < settings.water_level);
        // Apply underwater fog/absorption to the background color (terrain/sky)
        col *= exp(-abt * (1.0 - wabs) * 0.1);

        // If water surface is the closest thing hit
        if pt < t {
            // Blend background color with water color (reflection + specular) using Fresnel
            col = mix(col, wcol * (rcol + spec), fre);

            // Calculate shore foam effect
            let wp = wpos + wn * vec3f(1.0, 0.0, 1.0) * 0.2; // Offset position slightly by normal
            let wd = map(wp) / length(grad(wp)); // Sample SDF near water surface
            let foam = sin((wd - camera.time * 0.03) * 60.0); // Time-animated foam pattern
            // Mask for foam based on proximity to terrain and wave height
            let foam_mask = smoothstep(0.22, 0.0, wd + foam * 0.03 + (wh - 0.5) * 0.12);
            // Add white foam color based on mask
            col = mix(col, col + vec3f(1.0), foam_mask * 0.4);
        }
    }

    // --- Debug Visualizations ---
    if settings.show_normals != 0 {
        col = hit.n * 0.5 + 0.5; // Visualize voxel face normal
    }
    if settings.show_ray_steps != 0 {
        col = vec3f(f32(hit.i) / f32(settings.max_ray_steps)); // Visualize number of steps taken
    }

    // Final output color
    return vec4f(col, 1.0);
}