// Translated to WGSL by [Your Name]
// Note: Ensure proper texture and sampler bindings in the shader.

const PI: f32 = 3.141592;
const EPSILON_NRM: f32 = 0.1 / 1080.0; // Assuming resolution.x is 1920 or similar

struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_position: vec3f,
    time: f32,
    resolution: vec2<f32>,
    _padding: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var rgb_noise_texture: texture_2d<f32>;
@group(1) @binding(1) var gray_noise_texture: texture_2d<f32>;
@group(1) @binding(2) var gray_cube_noise_texture: texture_3d<f32>; 
@group(1) @binding(3) var grain_texture: texture_2d<f32>; 
@group(1) @binding(4) var dirt_texture: texture_2d<f32>;  
@group(1) @binding(5) var pebble_texture: texture_2d<f32>;  
@group(1) @binding(6) var samp: sampler; // Must use repeat mode

fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn noise3D(p: vec3<f32>) -> f32 {
    let p_floor = floor(p);
    let f = fract(p);
    let f_smooth = f * f * (3.0 - 2.0 * f);
    let coord = (p_floor + f_smooth + 0.5) / 32.0;
    return textureSampleLevel(gray_cube_noise_texture, samp, coord, 0.0).r;
}

fn noise2D(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let f_smooth = f * f * (3.0 - 2.0 * f);
    let coord = (i + f_smooth + 0.5) / 64.0;
    return textureSampleLevel(gray_noise_texture, samp, coord, 0.0).r * 2.0 - 1.0;
}

fn fbm(p: vec3<f32>) -> f32 {
    let m = mat3x3<f32>(
        0.00, -0.80, -0.60,
        0.80,  0.36, -0.48,
        0.60, -0.48,  0.64
    );
    var p_var = p;
    var f: f32 = 0.0;
    f += 0.5 * noise3D(p_var);
    p_var = m * p_var * 2.02;
    f += 0.25 * noise3D(p_var);
    p_var = m * p_var * 2.03;
    f += 0.125 * noise3D(p_var);
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
    let q = (-b + select(-sqrt(disc), sqrt(disc), b < 0.0)) / 2.0;
    let t0 = q;
    let t1 = c / q;
    let t_min = min(t0, t1);
    let t_max = max(t0, t1);
    if (t_max < 0.0) {
        return -1.0;
    }
    return select(t_max, t_min, t_min > 0.0);
}

fn clouds(p: vec3<f32>, cloudHeight: ptr<function, f32>, fast: bool) -> f32 {
    let earthCenter = vec3<f32>(0.0, -6300000.0, 0.0);
    let atmHeight = length(p - earthCenter) - 6300000.0;
    let ch = clamp((atmHeight - 800.0) / 600.0, 0.0, 1.0);
    *cloudHeight = ch;
    
    var p_moved = p;
    p_moved.z += uniforms.time * 10.3;
    let weatherCoord = -0.00005 * p_moved.zx;
    let largeWeather = clamp((textureSampleLevel(pebble_texture, samp, weatherCoord, 0.0).r - 0.18) * 5.0, 0.0, 2.0);
    
    p_moved.x += uniforms.time * 8.3;
    let weatherUV = 0.0002 * p_moved.zx;
    let weather = largeWeather * max(0.0, textureSampleLevel(pebble_texture, samp, weatherUV, 0.0).r - 0.28) / 0.72;
    let weatherShaped = weather * smoothstep(0.0, 0.5, ch) * smoothstep(1.0, 0.5, ch);
    let cloudShape = pow(weatherShaped, 0.3 + 1.5 * smoothstep(0.2, 0.5, ch));
    
    if (cloudShape <= 0.0) {
        return 0.0;
    }
    
    p_moved.x += uniforms.time * 12.3;
    var den = max(0.0, cloudShape - 0.7 * fbm(p_moved * 0.01));
    if (den <= 0.0) {
        return 0.0;
    }
    
    if (fast) {
        return largeWeather * 0.2 * min(1.0, 5.0 * den);
    }
    
    p_moved.y += uniforms.time * 15.2;
    den = max(0.0, den - 0.2 * fbm(p_moved * 0.05));
    return largeWeather * 0.2 * min(1.0, 5.0 * den);
}

fn numericalMieFit(costh: f32) -> f32 {
    let bestParams = array<f32, 10>(
        9.805233e-06, -65.0, -55.0, 0.8194068, 0.1388198, -83.70334, 7.810083, 0.002054747, 0.02600563, -4.552125e-12
    );
    let p1 = costh + bestParams[3];
    let expValues = vec4<f32>(
        exp(bestParams[1] * costh + bestParams[2]),
        exp(bestParams[5] * p1 * p1),
        exp(bestParams[6] * costh),
        exp(bestParams[9] * costh)
    );
    let weights = vec4<f32>(bestParams[0], bestParams[4], bestParams[7], bestParams[8]);
    return dot(expValues, weights);
}

fn lightRay(p: ptr<function, vec3<f32>>, phase: f32, dC: f32, mu: f32, sunDir: vec3<f32>, cloudHeight: f32, fast: bool) -> f32 {
    let nbSampleLight = select(20, 7, fast);
    let zMaxl = 600.0;
    let stepL = zMaxl / f32(nbSampleLight);
    
    var lighRayDen: f32 = 0.0;
    let seed = dot((*p), vec3<f32>(12.256, 2.646, 6.356)) + uniforms.time;
    (*p) += sunDir * stepL * hash(seed);
    
    for(var j: i32 = 0; j < nbSampleLight; j++) {
        var ch: f32;
        lighRayDen += clouds((*p) + sunDir * f32(j) * stepL, &ch, fast);
    }
    
    if (fast) {
        return (0.5 * exp(-0.4 * stepL * lighRayDen) + max(0.0, -mu * 0.6 + 0.3) * exp(-0.02 * stepL * lighRayDen)) * phase;
    }
    
    let scatterAmount = mix(0.008, 1.0, smoothstep(0.96, 0.0, mu));
    let beers = exp(-stepL * lighRayDen) + 0.5 * scatterAmount * exp(-0.1 * stepL * lighRayDen) + scatterAmount * 0.4 * exp(-0.02 * stepL * lighRayDen);
    let densityFactor = mix(0.05 + 1.5 * pow(min(1.0, dC * 8.5), 0.3 + 5.5 * cloudHeight), 1.0, clamp(lighRayDen * 0.4, 0.0, 1.0));
    return beers * phase * densityFactor;
}

fn skyRay(org: vec3<f32>, dir: vec3<f32>, sunDir: vec3<f32>, fast: bool) -> vec3<f32> {
    const EARTH_RADIUS: f32 = 6300000.0;
    const ATM_START: f32 = EARTH_RADIUS + 800.0;
    const ATM_END: f32 = ATM_START + 600.0;
    
    let nbSample = select(35, 13, fast);
    var color = vec3<f32>(0.0);
    let distToAtmStart = intersectSphere(org, dir, vec3<f32>(0.0, -EARTH_RADIUS, 0.0), ATM_START);
    let distToAtmEnd = intersectSphere(org, dir, vec3<f32>(0.0, -EARTH_RADIUS, 0.0), ATM_END);
    
    if (distToAtmStart < 0.0) {
        return vec3<f32>(0.0);
    }
    
    var p = org + dir * distToAtmStart;
    let stepS = (distToAtmEnd - distToAtmStart) / f32(nbSample);
    var T: f32 = 1.0;
    let mu = dot(sunDir, dir);
    let phase = numericalMieFit(mu);
    
    let seed = dot(dir, vec3<f32>(12.256, 2.646, 6.356)) + uniforms.time;
    p += dir * stepS * hash(seed);
    
    if (dir.y > 0.015) {
        for(var i: i32 = 0; i < nbSample; i++) {
            var cloudHeight: f32;
            let density = clouds(p, &cloudHeight, fast);
            if (density > 0.0) {
                let intensity = lightRay(&p, phase, density, mu, sunDir, cloudHeight, fast);
                let ambient = (0.5 + 0.6 * cloudHeight) * vec3<f32>(0.2, 0.5, 1.0) * 6.5 + vec3<f32>(0.8) * max(0.0, 1.0 - 2.0 * cloudHeight);
                let radiance = ambient + vec3<f32>(1.0, 0.9, 0.6) * 750.0 * intensity;
                let contrib = radiance * density;
                color += T * (contrib - contrib * exp(-density * stepS)) / density;
                T *= exp(-density * stepS);
                if (T <= 0.05) {
                    break;
                }
            }
            p += dir * stepS;
        }
    }
    
    if (!fast) {
        let pC = org + dir * intersectSphere(org, dir, vec3<f32>(0.0, -EARTH_RADIUS, 0.0), ATM_END + 1000.0);
        color += T * vec3<f32>(3.0) * max(0.0, fbm(pC * 0.002 * vec3<f32>(1.0, 1.0, 1.8)) - 0.4);
    }
    
    let background = 6.0 * mix(vec3<f32>(0.2, 0.52, 1.0), vec3<f32>(0.8, 0.95, 1.0), pow(0.5 + 0.5 * mu, 15.0)) + mix(vec3<f32>(3.5), vec3<f32>(0.0), min(1.0, 2.3 * dir.y));
    color += background * T;
    return color;
}

fn HenyeyGreenstein(mu: f32, g: f32) -> f32 {
    return (1.0 - g * g) / (4.0 * PI * pow(1.0 + g * g - 2.0 * g * mu, 1.5));
}

struct VertexOutput {
    @builtin(position) frag_coord: vec4f,
    @location(0) tex_uv: vec2f,
    @location(1) normal: vec3f,
    @location(2) world_position: vec3f,
    @location(3) coord: vec2f,
};

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
    output.coord = positions[vertex_index];
    output.tex_uv = tex_coords[vertex_index];
    output.normal = vec3f(0.0, 0.0, 1.0);
    output.world_position = vec3f(0.0);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let q = input.frag_coord.xy / uniforms.resolution;
    let v = -1.0 + 2.0 * q;
    let aspect = uniforms.resolution.x / uniforms.resolution.y;
    let v_uv = vec2<f32>(v.x * aspect, v.y);
    return vec4<f32>(v_uv.xy, 0.0,  1.0);
    // return vec4<f32>(input.coord.xy, 0.0,  1.0);
    
    // let org = vec3<f32>(6.0, 2.0, 6.0);
    // let ta = vec3<f32>(3.0, 5.0, 3.0);
    // let ww = normalize(ta - org);
    // let uu = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), ww));
    // let vv = normalize(cross(ww, uu));
    // let dir = normalize(v_uv.x * uu + v_uv.y * vv + 1.4 * ww);
    
    // let sunDir = normalize(vec3<f32>(sin(uniforms.time * 0.5) * 2.0, 0.45, -0.8));
    // let color = skyRay(org, dir, sunDir, false);
    
    // let fogDistance = intersectSphere(org, dir, vec3<f32>(0.0, -6300000.0, 0.0), 6300000.0 + 160.0);
    // let mu = dot(sunDir, dir);
    // let fogPhase = 0.5 * HenyeyGreenstein(mu, 0.7) + 0.5 * HenyeyGreenstein(mu, -0.6);
    // let fogColor = fogPhase * 0.1 * vec3<f32>(1.0, 0.7, 0.5) * vec3<f32>(1.0, 0.9, 0.6) * 750.0 + 10.0 * vec3<f32>(0.55, 0.8, 1.0);
    
    // let finalColor = mix(fogColor, color, exp(-0.0003 * fogDistance)) * 0.06;
    // return vec4<f32>(finalColor, 1.0);
}

// shader.wgsl

// const PI = 3.14159265359;
// const earth_radius = 6360e3;
// const atmosphere_radius = 6420e3;

// @group(0) @binding(0) var<uniform> camera: CameraUniform;
// @group(1) @binding(0) var rgb_noise_texture: texture_2d<f32>;
// @group(1) @binding(1) var gray_noise_texture: texture_2d<f32>;
// @group(1) @binding(2) var gray_cube_noise_texture: texture_3d<f32>; 
// @group(1) @binding(3) var grain_texture: texture_2d<f32>; 
// @group(1) @binding(4) var dirt_texture: texture_2d<f32>;  
// @group(1) @binding(5) var pebble_texture: texture_2d<f32>;  
// @group(1) @binding(6) var terrain_sampler: sampler; // Must use repeat mode

// struct CameraUniform {
//     view_proj: mat4x4<f32>,
//     inv_view_proj: mat4x4<f32>,
//     view: mat4x4<f32>,
//     camera_position: vec3f,
//     time: f32,
// };

// struct VertexInput {
//     @location(0) position: vec3f,
//     @location(1) tex_uv: vec2f,
//     @location(2) normal: vec3f,
// };

// struct VertexOutput {
//     @builtin(position) position: vec4f,
//     @location(0) tex_uv: vec2f,
//     @location(1) normal: vec3f,
//     @location(2) world_position: vec3f,
// };

// @vertex
// fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
//     let positions = array<vec2f, 4>(
//         vec2f(-1.0, -1.0), vec2f(1.0, -1.0),
//         vec2f(-1.0, 1.0), vec2f(1.0, 1.0)
//     );
//     let tex_coords = array<vec2f, 4>(
//         vec2f(0.0, 0.0), vec2f(1.0, 0.0),
//         vec2f(0.0, 1.0), vec2f(1.0, 1.0)
//     );
//     var output: VertexOutput;
//     output.position = vec4f(positions[vertex_index], 0.0, 1.0);
//     output.tex_uv = tex_coords[vertex_index];
//     output.normal = vec3f(0.0, 0.0, 1.0);
//     output.world_position = vec3f(0.0);
//     return output;
// }

// @fragment
// fn fs_main(input: VertexOutput) -> @location(0) vec4f {
//     let ro = camera.camera_position;
//     let ndc = vec4f(input.tex_uv * 2.0 - 1.0, 1.0, 1.0);
//     let world_pos = camera.inv_view_proj * ndc;
//     let rd = normalize(world_pos.xyz / world_pos.w - ro);

//     return vec4f(ndc.xy, 0.0, 1.0);
// }