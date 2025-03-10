const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const MAX: f32 = 10000.0;
const R_INNER: f32 = 1.0;
const R: f32 = R_INNER + 0.8;
const NUM_OUT_SCATTER: i32 = 8;
const NUM_IN_SCATTER: i32 = 80;

// Voxel renderer constants
const MAX_HEIGHT: f32 = 5.0;
const MAX_WATER_HEIGHT: f32 = -2.2;
const WATER_HEIGHT: f32 = MAX_WATER_HEIGHT;
const TUNNEL_RADIUS: f32 = 1.1;
const SURFACE_FACTOR: f32 = 0.42;
const CAMERA_SPEED: f32 = -1.5;
const CAMERA_TIME_OFFSET: f32 = 0.0;
const VOXEL_LEVEL: i32 = 6; 
const VOXEL_SIZE: f32 = exp2(-f32(VOXEL_LEVEL)); 
const STEPS: i32 = 512 * 2 * 2;
const MAX_DIST: f32 = 600.0;
const MIN_DIST: f32 = VOXEL_SIZE;
const EPS: f32 = 1e-5;

// Updated light color and direction
const lcol: vec3f = vec3f(1.0, 0.9, 0.75) * 2.0;
const ldir: vec3f = vec3f(0.507746, 0.716817, 0.477878); // Normalized light direction

// Debug flags
const SHOW_NORMALS: bool = false;
const SHOW_STEPS: bool = false;
const VISUALIZE_DISTANCE_FIELD: bool = false;

// Bindings
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var noise0_texture: texture_2d<f32>; 
@group(1) @binding(1) var noise1_texture: texture_3d<f32>; 
@group(1) @binding(2) var grain_texture: texture_2d<f32>;  
@group(1) @binding(3) var dirt_texture: texture_2d<f32>;   
@group(1) @binding(4) var terrain_sampler: sampler; 

// Structures
struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_position: vec3f,
    time: f32,
};

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

// Ray intersects sphere
// e = -b +/- sqrt(b^2 - c)
fn ray_vs_sphere(p: vec3f, dir: vec3f, r: f32) -> vec2f {
    let b = dot(p, dir);
    let c = dot(p, p) - r * r;
    let d = b * b - c;
    if (d < 0.0) {
        return vec2f(MAX, -MAX);
    }
    let sq_d = sqrt(d);
    return vec2f(-b - sq_d, -b + sq_d);
}

// Mie scattering phase function
// g: (-0.75, -0.999)
fn phase_mie(g: f32, c: f32, cc: f32) -> f32 {
    let gg = g * g;
    let a = (1.0 - gg) * (1.0 + cc);
    var b = 1.0 + gg - 2.0 * g * c;
    b *= sqrt(b);
    b *= 2.0 + gg;
    return (3.0 / 8.0 / PI) * a / b;
}

// Rayleigh scattering phase function
// g: 0
fn phase_ray(cc: f32) -> f32 {
    return (3.0 / 16.0 / PI) * (1.0 + cc);
}

// Atmospheric density at point
fn density(p: vec3f, ph: f32) -> f32 {
    return exp(-max(length(p) - R_INNER, 0.0) / ph);
}

// Optical depth calculation
fn optic(p: vec3f, q: vec3f, ph: f32) -> f32 {
    let s = (q - p) / f32(NUM_OUT_SCATTER);
    var v = p + s * 0.5;
    var sum = 0.0;
    
    for (var i = 0; i < NUM_OUT_SCATTER; i++) {
        sum += density(v, ph);
        v += s;
    }
    
    sum *= length(s);
    return sum;
}

// In-scattering calculation
fn in_scatter(o: vec3f, dir: vec3f, e: vec2f, l: vec3f) -> vec3f {
    let ph_ray = 0.05;
    let ph_mie = 0.02;
    let k_ray = vec3f(3.8, 13.5, 33.1);
    let k_mie = vec3f(21.0);
    let k_mie_ex = 1.1;
    
    var sum_ray = vec3f(0.0);
    var sum_mie = vec3f(0.0);
    var n_ray0 = 0.0;
    var n_mie0 = 0.0;
    
    let len = (e.y - e.x) / f32(NUM_IN_SCATTER);
    let s = dir * len;
    var v = o + dir * (e.x + len * 0.5);
    
    for (var i = 0; i < NUM_IN_SCATTER; i++) {
        v += s;
        let d_ray = density(v, ph_ray) * len;
        let d_mie = density(v, ph_mie) * len;
        
        n_ray0 += d_ray;
        n_mie0 += d_mie;
        
        let f = ray_vs_sphere(v, l, R);
        let u = v + l * f.y;
        
        let n_ray1 = optic(v, u, ph_ray);
        let n_mie1 = optic(v, u, ph_mie);
        
        let att = exp(-(n_ray0 + n_ray1) * k_ray - (n_mie0 + n_mie1) * k_mie * k_mie_ex);
        
        sum_ray += d_ray * att;
        sum_mie += d_mie * att;
    }
    
    let c = dot(dir, -l);
    let cc = c * c;
    
    let scatter = sum_ray * k_ray * phase_ray(cc) + sum_mie * k_mie * phase_mie(-0.78, c, cc);
    return scatter;
}

// Utility Functions
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

fn hash23(p: vec3f) -> vec2f {
    var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn triplanarLod(p: vec3f, n: vec3f, k: f32, tex_index: i32, lod: f32) -> vec3f {
    let n_pow = pow(abs(n), vec3f(k));
    let n_norm = n_pow / dot(n_pow, vec3f(1.0));
    var col = vec3f(0.0);
    if (tex_index == 0) {
        col = textureSampleLevel(noise0_texture, terrain_sampler, p.yz, lod).rgb * n_norm.x +
              textureSampleLevel(noise0_texture, terrain_sampler, p.xz, lod).rgb * n_norm.y +
              textureSampleLevel(noise0_texture, terrain_sampler, p.xy, lod).rgb * n_norm.z;
    } else if (tex_index == 2) {
        col = textureSampleLevel(grain_texture, terrain_sampler, p.yz, lod).rgb * n_norm.x +
              textureSampleLevel(grain_texture, terrain_sampler, p.xz, lod).rgb * n_norm.y +
              textureSampleLevel(grain_texture, terrain_sampler, p.xy, lod).rgb * n_norm.z;
    } else if (tex_index == 3) {
        col = textureSampleLevel(dirt_texture, terrain_sampler, p.yz, lod).rgb * n_norm.x +
              textureSampleLevel(dirt_texture, terrain_sampler, p.xz, lod).rgb * n_norm.y +
              textureSampleLevel(dirt_texture, terrain_sampler, p.xy, lod).rgb * n_norm.z;
    }
    return col;
}

// fn map(p: vec3f) -> f32 {
//     var d: f32 = MAX_DIST;
//     let sc: f32 = 0.3;
//     // Terrain generation
//     let q: vec3f = sc * p / 32.0 - vec3f(0.003, -0.006, 0.0);
//     d = textureSample(noise1_texture, terrain_sampler, q * 1.0).r * 0.5;
//     d += textureSample(noise1_texture, terrain_sampler, q * 2.0 + vec3f(0.3, 0.3, 0.3)).r * 0.25;
//     d += textureSample(noise1_texture, terrain_sampler, q * 4.0 + vec3f(0.7, 0.7, 0.7)).r * 0.125;
//     var tp = smoothstep(50.0, -6.0, p.y);
//     tp = tp * tp;
//     d = (d/0.875 - SURFACE_FACTOR) / sc;
//     d = smax(d, p.y - MAX_HEIGHT, 0.6);

//     let camera_pos = camera.camera_position;
//     let camera_distance = length(p - camera_pos);
    
//     // Remove terrain too close to the camera
//     d = smax(d, MIN_DIST - camera_distance, 0.3);
    
//     return d;
// }

// New constants for the planet
const PLANET_RADIUS: f32 = 50.0;
const TERRAIN_HEIGHT: f32 = 5.0;

fn map(p: vec3f) -> f32 {
    var d: f32 = MAX_DIST;
    let sc: f32 = 0.3;
    
    // Calculate distance to the planet's surface
    let planet_center = vec3f(0.0, 0.0, 0.0); // Center of the planet
    let planet_distance = length(p - planet_center) - PLANET_RADIUS;
    
    // Apply noise to create terrain on the planet surface
    let surface_dir = normalize(p - planet_center);
    let q: vec3f = sc * surface_dir * 32.0;
    
    // Sample noise textures for terrain details
    var noise_value: f32 = textureSample(noise1_texture, terrain_sampler, q * 1.0).r * 0.5;
    noise_value += textureSample(noise1_texture, terrain_sampler, q * 2.0 + vec3f(0.3, 0.3, 0.3)).r * 0.25;
    noise_value += textureSample(noise1_texture, terrain_sampler, q * 4.0 + vec3f(0.7, 0.7, 0.7)).r * 0.125;
    
    // Scale terrain height
    noise_value = (noise_value / 0.875 - SURFACE_FACTOR) / sc * TERRAIN_HEIGHT;
    
    // Combine planet sphere with terrain
    d = planet_distance + noise_value;
    
    // Make sure camera can see inside the planet if needed
    let camera_pos = camera.camera_position;
    let camera_distance = length(p - camera_pos);
    
    // Remove terrain too close to the camera
    d = smax(d, MIN_DIST - camera_distance, 0.3);
    
    return d;
}

fn grad(p: vec3f) -> vec3f {
    let e = vec2f(0.0, 0.1);
    return (map(p) - vec3f(
        map(p - e.yxx),
        map(p - e.xyx),
        map(p - e.xxy)
    )) / e.y;
}

fn get_voxel_pos(p: vec3f, s: f32) -> vec3f {
    return (floor(p / s) + 0.5) * s;
}

fn trace(ro: vec3f, rd: vec3f, tmax: f32) -> HitInfo {
    let s = VOXEL_SIZE;
    let sd = s * sqrt(3.0);
    let ird = 1.0 / rd;
    let srd = sign(ird);
    let ard = abs(ird);
    var t = 0.0;
    var vpos = get_voxel_pos(ro, s);
    var voxel = false;
    var vi = 0;
    var prd = vec3f(0.0);
    for (var i = 0; i < STEPS; i = i + 1) {
        let pos = ro + rd * t;
        let d = map(select(pos, vpos, voxel));
        if !voxel {
            t += d;
            if d < sd {
                vpos = get_voxel_pos(ro + rd * max(t - sd, 0.0), s);
                voxel = true;
                vi = 0;
            }
        } else {
            let n = (ro - vpos) * ird;
            let k = ard * s * 0.5;
            let t2 = -n + k;
            let tF = min(min(t2.x, t2.y), t2.z);
            var nrd = vec3f(0.0);
            if t2.x <= t2.y && t2.x <= t2.z {
                nrd = vec3f(srd.x, 0.0, 0.0);
            } else if t2.y <= t2.z {
                nrd = vec3f(0.0, srd.y, 0.0);
            } else {
                nrd = vec3f(0.0, 0.0, srd.z);
            }
            if d < 0.0 {
                return HitInfo(true, t, -prd, vpos, i);
            } else if d > sd && vi > 2 {
                voxel = false;
                t = tF + sd;
                continue;
            }
            vpos += nrd * s;
            prd = nrd;
            t = tF + EPS;
            vi += 1;
        }
        if t >= tmax || (rd.y > 0.0 && pos.y > MAX_HEIGHT) {
            return HitInfo(false, t, vec3f(0.0), vec3f(0.0), i);
        }
    }
    return HitInfo(false, tmax, vec3f(0.0), vec3f(0.0), STEPS);
}

fn triplanar(p: vec3f, n: vec3f, k: f32, tex_index: i32) -> vec3f {
    let n_pow = pow(abs(n), vec3f(k));
    let n_norm = n_pow / dot(n_pow, vec3f(1.0));
    var col = vec3f(0.0);
    if tex_index == 0 {
        col = textureSample(noise0_texture, terrain_sampler, p.yz).rgb * n_norm.x +
              textureSample(noise0_texture, terrain_sampler, p.xz).rgb * n_norm.y +
              textureSample(noise0_texture, terrain_sampler, p.xy).rgb * n_norm.z;
    } else if tex_index == 2 {
        col = textureSample(grain_texture, terrain_sampler, p.yz).rgb * n_norm.x +
              textureSample(grain_texture, terrain_sampler, p.xz).rgb * n_norm.y +
              textureSample(grain_texture, terrain_sampler, p.xy).rgb * n_norm.z;
    } else if tex_index == 3 {
        col = textureSample(dirt_texture, terrain_sampler, p.yz).rgb * n_norm.x +
              textureSample(dirt_texture, terrain_sampler, p.xz).rgb * n_norm.y +
              textureSample(dirt_texture, terrain_sampler, p.xy).rgb * n_norm.z;
    }
    return col;
}

fn getBiome(pos: vec3f) -> vec2f {
    let snow = textureSample(dirt_texture, terrain_sampler, pos.xz * 0.00015).r;
    let desert = textureSample(dirt_texture, terrain_sampler, vec2f(0.55) - pos.zx * 0.00008).g;
    return vec2f(smoothstep(0.67, 0.672, desert), smoothstep(0.695, 0.7, snow));
}

fn getAlbedo(vpos: vec3f, gn: vec3f, lod: f32) -> vec3f {
    var alb = vec3f(1.0) - triplanarLod(vpos * 0.08, gn, 4.0, 2, lod);
    alb *= alb;
    var alb2 = vec3f(1.0) - triplanarLod(vpos * 0.08, gn, 4.0, 3, lod);
    alb2 *= alb2;
    let k = triplanarLod(vpos * 0.0005, gn, 4.0, 0, 0.0).r;
    let wk = smoothstep(MAX_WATER_HEIGHT, MAX_WATER_HEIGHT + 0.5, vpos.y);
    let top = smoothstep(0.3, 0.7, gn.y);
    alb = alb * 0.95 * vec3f(1.0, 0.7, 0.65) + 0.05;
    alb = mix(alb, alb2 * vec3f(0.55, 1.0, 0.1), top * wk);
    alb = mix(alb, smoothstep(vec3f(0.0), vec3f(1.0), alb2), smoothstep(0.3, 0.25, k) * (1.0 - top));
    let biome = getBiome(vpos);
    var snow = alb2 * 0.8 + 0.2 * vec3f(0.25, 0.5, 1.0);
    snow = mix(snow, vec3f(0.85, 0.95, 1.0), top * wk * 0.5);
    alb = mix(alb, clamp(vec3f(1.0, 0.95, 0.9) - alb2 * 0.65, vec3f(0.0), vec3f(1.0)), biome.x);
    alb = mix(alb, snow * 2.0, biome.y);
    var dcol = vec3f(0.8, 0.55, 0.35);
    dcol = mix(dcol, vec3f(0.8, 0.65, 0.4), biome.x);
    dcol = mix(dcol, vec3f(0.2, 0.6, 0.8), biome.y);
    alb = mix(alb, alb * dcol, (1.0 - wk) * mix(1.0 - smoothstep(0.3, 0.25, k), 1.0, max(biome.x, biome.y)));
    return alb;
}

fn shade(pos: vec3f, ldir: vec3f, lod: f32, hit: HitInfo) -> vec3f {
    let vpos = hit.id;
    let g = grad(vpos);
    let gn = g / length(g);
    let n = hit.n;
    var dif = max(dot(n, ldir), 0.0);
    if dif > 0.0 {
        let hitL = trace(pos + n * 1e-3, ldir, 12.0);
        if hitL.is_hit { dif = 0.0; }
    }
    var col = getAlbedo(vpos, gn, lod);
    let ao = smoothstep(-0.08, 0.04, map(pos) / length(grad(pos)));
    let hao = smoothstep(WATER_HEIGHT - 12.0, WATER_HEIGHT, pos.y);
    col *= dot(abs(n), vec3f(0.8, 1.0, 0.9));
    col *= (dif * 0.6 + 0.4) * lcol;
    col *= (ao * 0.6 + 0.4) * (hao * 0.6 + 0.4);
    return col;
}

fn shade2(pos: vec3f, ldir: vec3f, lod: f32, hit: HitInfo) -> vec3f {
    let vpos = hit.id;
    let g = grad(vpos);
    let gn = g / length(g);
    let n = hit.n;
    let dif = max(dot(n, ldir), 0.0);
    
    var col = getAlbedo(vpos, gn, lod);
    let ao = smoothstep(-0.08, 0.04, map(pos) / length(grad(pos)));
    let hao = smoothstep(WATER_HEIGHT - 12.0, WATER_HEIGHT, pos.y);
    
    col *= dot(abs(n), vec3f(0.8, 1.0, 0.9));
    col *= (dif * 0.6 + 0.4) * lcol;
    col *= ao * 0.6 + 0.4;
    col *= hao * 0.6 + 0.4;
    
    return col;
}

// New atmospheric sky function that uses the scattering implementation
fn getSky(rd: vec3f, ro: vec3f) -> vec3f {
    // Scale up the world to atmospheric scale
    let atmosphere_scale = 1000.0; // Scale factor to make the atmosphere appear larger
    let scaled_ro = ro / atmosphere_scale;
    
    // Assume we're inside the atmosphere
    let p = vec3f(0.0, R_INNER, 0.0); // Planet center
    let o = scaled_ro + p; // Observer position relative to planet center
    
    // Get intersection with the outer atmosphere
    let e = ray_vs_sphere(o, rd, R);
    
    // If the ray hits the atmosphere, compute scattering
    if (e.y > 0.0) {
        // Sun light direction (use existing light direction)
        let l = -ldir; // Negate for light coming from the sun
        
        // Compute atmospheric scattering
        let scatter = in_scatter(o, rd, e, l);
        
        // Scale intensity for better visual
        let intensity_scale = 0.8;
        return scatter * intensity_scale;
    }

    return vec3f(0.0, 0.0, 0.0);
}

fn ACESFilm(x: vec3f) -> vec3f {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let ro = camera.camera_position;
    let ndc = vec4f(input.tex_uv * 2.0 - 1.0, 1.0, 1.0);
    let world_pos = camera.inv_view_proj * ndc;
    let rd = normalize(world_pos.xyz / world_pos.w - ro);
    
    if VISUALIZE_DISTANCE_FIELD {
        let pos = ro + rd * 10.0;
        let d = map(pos);
        return vec4f(vec3f(d * 0.1 + 0.5), 1.0);
    }
    
    let hit = trace(ro, rd, MAX_DIST);
    var col = vec3f(0.0);
    var t = hit.t;
    
    if hit.is_hit {
        let pos = ro + rd * hit.t;
        let lod = clamp(log2(distance(ro, hit.id)) - 2.0, 0.0, 6.0);
        col = shade(pos, ldir, lod, hit);
        
        // Apply atmospheric haze to distant objects
        let dist_factor = smoothstep(10.0, 200.0, hit.t);
        let atmosphere_scale = 100.0;
        let scaled_ro = ro / atmosphere_scale;
        let p = vec3f(0.0, R_INNER, 0.0);
        let o = scaled_ro + p;
        let e = ray_vs_sphere(o, rd, R);
        
        if (e.y > 0.0) {
            let l = -ldir;
            let scatter = in_scatter(o, rd, e, l);
            col = mix(col, scatter * 0.8, dist_factor * 0.7);
        }
    } else {
        col = getSky(rd, ro);
        t = MAX_DIST;
    }
    
    // Handle water rendering
    let pt = -(ro.y - WATER_HEIGHT) / rd.y;
    if ((pt > 0.0 && pt < t)) || ro.y < WATER_HEIGHT {
        if !hit.is_hit {
            let biome = getBiome(ro + rd * pt);
            col = mix(vec3f(0.5, 0.8, 1.0), vec3f(1.0, 0.85, 0.6), biome.x);
        }
        
        let biome = getBiome(ro + rd * pt);
        var wcol = vec3f(0.3, 0.8, 1.0);
        wcol = mix(wcol, vec3f(0.4, 0.9, 0.8), biome.x);
        wcol = mix(wcol, vec3f(0.1, 0.7, 0.9), biome.y);
        
        let wabs = vec3f(0.1, 0.7, 0.9);
        
        var adjusted_pt = pt;
        if (ro.y < WATER_HEIGHT && pt < 0.0) {
            adjusted_pt = MAX_DIST;
        }
        
        let wpos = ro + rd * adjusted_pt;
        
        let e = 0.001;
        let wnstr = 1500.0;
        let wo = vec2f(1.0, 0.8) * camera.time * 0.01;
        let wuv = wpos.xz * 0.08 + wo;
        let wh = textureSample(grain_texture, terrain_sampler, wuv).r;
        let whdx = textureSample(grain_texture, terrain_sampler, wuv + vec2f(e, 0.0)).r;
        let whdy = textureSample(grain_texture, terrain_sampler, wuv + vec2f(0.0, e)).r;
        let wn = normalize(vec3f(wh - whdx, e * wnstr, wh - whdy));
        let wref = reflect(rd, wn);
        
        var rcol = vec3f(0.0);
        if (ro.y > WATER_HEIGHT) {
            let hitR = trace(wpos + vec3f(0.0, 0.01, 0.0), wref, 15.0);
            let lod = clamp(log2(distance(ro, hitR.id)) - 2.0, 0.0, 6.0);
            
            if (hitR.is_hit) {
                rcol = shade2(wpos, ldir, lod, hitR);
            } else {
                rcol = getSky(wref, ro);
            }
        }
        

        // Specular highlight
        let spec = pow(max(dot(wref, ldir), 0.0), 50.0);
        
        // Fresnel reflection factor
        let r0 = 0.35;
        var fre = r0 + (1.0 - r0) * pow(max(dot(rd, wn), 0.0), 5.0);
        
        if (rd.y < 0.0 && ro.y < WATER_HEIGHT) {
            fre = 0.0;
        }
        
        // Water absorption
        let abt = select(t - pt, min(t, pt), ro.y < WATER_HEIGHT);
        col *= exp(-abt * (1.0 - wabs) * 0.1);
        
        if (pt < t) {
            col = mix(col, wcol * (rcol + spec), fre);
            
            // Foam effect
            let wp = wpos + wn * vec3f(1.0, 0.0, 1.0) * 0.2;
            let wd = map(wp) / length(grad(wp));
            let foam = sin((wd - camera.time * 0.03) * 60.0);
            let foam_mask = smoothstep(0.22, 0.0, wd + foam * 0.03 + (wh - 0.5) * 0.12);
            col = mix(col, col + vec3f(1.0), foam_mask * 0.4);
        }
    }
    
    if SHOW_NORMALS {
        col = hit.n;
    }
    
    if SHOW_STEPS {
        col = vec3f(f32(hit.i) / f32(STEPS));
    }
    
    return vec4f(col, 1.0);
}



