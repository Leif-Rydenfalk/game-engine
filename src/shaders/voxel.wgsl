// voxel.wgsl

fn sin_0_1(v: f32) -> f32 {
    return sin(v) * 2.0 + 1.0;
}

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

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var rgb_noise_texture: texture_2d<f32>;
@group(1) @binding(1) var gray_noise_texture: texture_2d<f32>;
@group(1) @binding(2) var gray_cube_noise_texture: texture_3d<f32>; 
@group(1) @binding(3) var grain_texture: texture_2d<f32>; 
@group(1) @binding(4) var dirt_texture: texture_2d<f32>;  
@group(1) @binding(5) var pebble_texture: texture_2d<f32>;  
@group(1) @binding(6) var terrain_sampler: sampler; // Must use repeat mode
@group(2) @binding(0) var<uniform> settings: Settings;

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

struct HitInfo {
    is_hit: bool,
    t: f32,
    n: vec3f,
    id: vec3f,
    i: i32,
};

fn smax(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

fn smin(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

fn triplanarLod(p: vec3f, n: vec3f, k: f32, tex_index: i32, lod: f32) -> vec3f {
    let n_pow = pow(abs(n), vec3f(k));
    let n_norm = n_pow / dot(n_pow, vec3f(1.0));
    var col = vec3f(0.0);
    if tex_index == 0 {
        col = textureSampleLevel(rgb_noise_texture, terrain_sampler, p.yz, lod).rgb * n_norm.x + textureSampleLevel(rgb_noise_texture, terrain_sampler, p.xz, lod).rgb * n_norm.y + textureSampleLevel(rgb_noise_texture, terrain_sampler, p.xy, lod).rgb * n_norm.z;
    } else if tex_index == 2 {
        col = textureSampleLevel(grain_texture, terrain_sampler, p.yz, lod).rgb * n_norm.x + textureSampleLevel(grain_texture, terrain_sampler, p.xz, lod).rgb * n_norm.y + textureSampleLevel(grain_texture, terrain_sampler, p.xy, lod).rgb * n_norm.z;
    } else if tex_index == 3 {
        col = textureSampleLevel(dirt_texture, terrain_sampler, p.yz, lod).rgb * n_norm.x + textureSampleLevel(dirt_texture, terrain_sampler, p.xz, lod).rgb * n_norm.y + textureSampleLevel(dirt_texture, terrain_sampler, p.xy, lod).rgb * n_norm.z;
    }
    return col;
}

fn map(p: vec3f) -> f32 {
    var d: f32 = 100000000000000.0;
    let sc: f32 = 0.3;
    // Terrain generation remains the same
    let q: vec3f = sc * p / 32.0 - vec3f(0.003, -0.006, 0.0);
    d = textureSample(gray_cube_noise_texture, terrain_sampler, q * 0.6).r * 0.4;
    d += textureSample(gray_cube_noise_texture, terrain_sampler, q * 2.0 + vec3f(0.3, 0.3, 0.3)).r * 0.15;
    d += textureSample(gray_cube_noise_texture, terrain_sampler, q * 4.0 + vec3f(0.7, 0.7, 0.7)).r * 0.125;
    var tp = smoothstep(50.0, -6.0, p.y);
    tp = tp * tp;
    //d = (d / 0.875 - settings.surface_factor) / sc;
    d = 1.0 - d * 5.0;

    //let n = textureSample(gray_cube_noise_texture, terrain_sampler, vec3(p.xz * 0.01, 0.1)).r * 5.0;
    let n = 0.0;
    d = smax(d, p.y - settings.max_height - n, 2.6);

    // var mask = textureSample(gray_cube_noise_texture, terrain_sampler, q * 0.05).r * 1.7;
    // mask = smax(mask, p.y - settings.max_height, 0.6);

    //var mask = textureSample(gray_cube_noise_texture, terrain_sampler, q * 0.05).r * 1.7;
    //d = smax(d, -mask, 6.0);


    let sphere_d = length(p) - 10.0;
    // d = smin(d, sphere_d, 6.0);
    d = smax(d, -sphere_d, 6.0);

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

fn trace(ro: vec3f, rd: vec3f, tmax: f32, s: f32) -> HitInfo {
    // let s = settings.voxel_size;
    let sd = s * sqrt(3.0);
    let ird = 1.0 / rd;
    let srd = sign(ird);
    let ard = abs(ird);
    var t = 0.0;
    var vpos = get_voxel_pos(ro, s);
    var voxel = false;
    var vi = 0;
    var prd = vec3f(0.0);
    for (var i = 0; i < settings.steps; i = i + 1) {
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
                // Smoothes out the normals
                // let hit_pos_approx = ro + rd * t;
                // let surface_normal = normalize(grad(vpos));
                // return HitInfo(true, t, surface_normal, vpos, i);
                return HitInfo(true, t, -prd, vpos, i);
            } else if d > sd && vi > 2 {
                voxel = false;
                t = tF + sd;
                continue;
            }

            vpos += nrd * s;
            prd = nrd;
            t = tF + settings.eps;
            vi += 1;
        }
        if t >= tmax || (rd.y > 0.0 && pos.y > 10.0) {
            return HitInfo(false, t, vec3f(0.0), vec3f(0.0), i);
        }
    }
    return HitInfo(false, tmax, vec3f(0.0), vec3f(0.0), settings.steps);
}

fn triplanar(p: vec3f, n: vec3f, k: f32, tex_index: i32) -> vec3f {
    let n_pow = pow(abs(n), vec3f(k));
    let n_norm = n_pow / dot(n_pow, vec3f(1.0));
    var col = vec3f(0.0);
    if tex_index == 0 {
        col = textureSample(rgb_noise_texture, terrain_sampler, p.yz).rgb * n_norm.x + textureSample(rgb_noise_texture, terrain_sampler, p.xz).rgb * n_norm.y + textureSample(rgb_noise_texture, terrain_sampler, p.xy).rgb * n_norm.z;
    } else if tex_index == 2 {
        col = textureSample(grain_texture, terrain_sampler, p.yz).rgb * n_norm.x + textureSample(grain_texture, terrain_sampler, p.xz).rgb * n_norm.y + textureSample(grain_texture, terrain_sampler, p.xy).rgb * n_norm.z;
    } else if tex_index == 3 {
        col = textureSample(dirt_texture, terrain_sampler, p.yz).rgb * n_norm.x + textureSample(dirt_texture, terrain_sampler, p.xz).rgb * n_norm.y + textureSample(dirt_texture, terrain_sampler, p.xy).rgb * n_norm.z;
    }
    return col;
}

fn get_biome(pos: vec3f) -> vec2f {
    let snow = textureSample(dirt_texture, terrain_sampler, pos.xz * 0.00015).r;
    let desert = textureSample(dirt_texture, terrain_sampler, vec2f(0.55) - pos.zx * 0.00008).g;
    return vec2f(smoothstep(0.67, 0.672, desert), smoothstep(0.695, 0.7, snow));
}

fn get_voxel_color(vpos: vec3f, gn: vec3f, lod: f32) -> vec3f {
    var alb = vec3f(1.0) - triplanarLod(vpos * 0.08, gn, 4.0, 2, lod);
    alb *= alb;
    var alb2 = vec3f(1.0) - triplanarLod(vpos * 0.08, gn, 4.0, 3, lod);
    alb2 *= alb2;
    let k = triplanarLod(vpos * 0.0005, gn, 4.0, 0, 0.0).r;

    let wk = smoothstep(settings.max_water_height, settings.max_water_height + 0.5, vpos.y);
    let top = smoothstep(0.3, 0.7, gn.y);
    alb = alb * 0.95 * vec3f(1.0, 0.7, 0.65) + 0.05;
    alb = mix(alb, alb2 * vec3f(0.55, 1.0, 0.1), top * wk);
    alb = mix(alb, smoothstep(vec3f(0.0), vec3f(1.0), alb2), smoothstep(0.3, 0.25, k) * (1.0 - top));
    let biome = get_biome(vpos);
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

    var col = get_voxel_color(vpos, gn, lod);

    var dif = max(dot(n, ldir), 0.0);
    if dif > 0.0 {
        let tmax = 100000000.0;
        let hitL = trace(pos + n * 1e-3, ldir, tmax, settings.voxel_size);
        if hitL.is_hit { dif = 0.0; }
    }

    let ao = smoothstep(-0.08, 0.04, map(pos) / length(grad(pos)));
   // let hao = smoothstep(settings.water_height - 12.0, settings.water_height, pos.y);
    col *= dot(abs(n), vec3f(0.8, 1.0, 0.9));
    col *= (dif * 0.6 + 0.4) * settings.light_color.rgb;
    col *= (ao * 0.6 + 0.5); //* (hao * 0.6 + 0.4);
    return col;
}

fn ACESFilm(x: vec3<f32>) -> vec3<f32> {
    let a: f32 = 2.51;
    let b: f32 = 0.03;
    let c: f32 = 2.43;
    let d: f32 = 0.59;
    let e: f32 = 0.14;

    return clamp((x * (a * x + vec3<f32>(b))) / (x * (c * x + vec3<f32>(d)) + vec3<f32>(e)), vec3<f32>(0.0), vec3<f32>(1.0));
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) world_pos: vec3f, // Pass world position for view direction calc
    @location(1) ndc: vec4f, // Pass world position for view direction calc
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generate a fullscreen triangle strip
    let positions = array<vec2f, 4>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, 1.0)
    );
    let pos = positions[vertex_index];

    // Calculate world position for view direction in fragment shader
    let ndc = vec4f(pos.x, -pos.y, 1.0, 1.0); // Use Z=1 for far plane
    let world = camera.inv_view_proj * ndc;
    let world_xyz = world.xyz / world.w;

    var output: VertexOutput;
    output.position = vec4f(pos, 0.0, 1.0); // Project to near plane for rasterizer
    output.world_pos = world_xyz;
    output.ndc = ndc;
    return output;
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @location(1) depth: f32, // Output for our custom depth buffer
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    let ro = camera.position;
    var ndc = input.ndc;
    let world_pos = camera.inv_view_proj * ndc;
    let rd = normalize(world_pos.xyz / world_pos.w - ro);
    let sun_dir = normalize(settings.light_direction.xyz);

    let hit = trace(ro, rd, 100000000000000.0, settings.voxel_size);
    var col = vec3f(0.0); // Initialize color calculation variable
    var t = 100000000000000.0;       // Use local 't' for clarity

    if hit.is_hit {
        let pos = ro + rd * hit.t;
        let lod = clamp(log2(distance(ro, hit.id)) - 2.0, 0.0, 6.0);
        col = shade(pos, sun_dir, lod, hit);
        t = hit.t; // Assign the hit distance
    } 

    // --- Debug visualizations ---
    if settings.show_normals != 0 && hit.is_hit {
        col = hit.n * 0.5 + 0.5; // Map normal to color range
    }

    if settings.show_steps != 0 {
        col = vec3f(f32(hit.i) / f32(settings.steps));
    }

    // --- Final color processing ---
    col = pow(col * 1.3, vec3f(1.8));
    col = ACESFilm(col);       

    // --- Assign final values to output struct ---
    
    // --- Prepare output struct ---
    var output: FragmentOutput;
    output.color = vec4f(col, 1.0);
    output.depth = t; // Assign the calculated depth (hit distance or max_dist)

    return output;
}