// voxel.wgsl

const PI = 3.14159265359;
const betaR = vec3f(5.5e-6, 13.0e-6, 22.4e-6);
const betaM = vec3f(21e-6);
const hR = 7994.0;
const hM = 1200.0;
const earth_radius = 6360e3;
const atmosphere_radius = 6420e3;
const sun_power = 10.0;
const g = 0.76;
const k = 1.55 * g - 0.55 * (g * g * g);
const num_samples = 16;
const num_samples_light = 8;

struct Ray {
    origin: vec3f,
    direction: vec3f,
};

struct Sphere {
    origin: vec3f,
    radius: f32,
    material: i32,
};

struct IsectResult {
    hit: bool,
    t0: f32,
    t1: f32,
};

const atmosphere = Sphere(vec3f(0.0, 0.0, 0.0), atmosphere_radius, 0);

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

fn get_incident_light(ray: Ray, sun_dir: vec3f) -> vec3f {
    let isect = isect_sphere(ray, atmosphere);
    if !isect.hit {
        return vec3f(0.0);
    }
    let t1 = isect.t1;
    let march_step = t1 / f32(num_samples);
    let mu = dot(ray.direction, sun_dir);
    let phaseR = rayleigh_phase_func(mu);
    let phaseM = henyey_greenstein_phase_func(mu);
    var optical_depthR = 0.0;
    var optical_depthM = 0.0;
    var sumR = vec3f(0.0);
    var sumM = vec3f(0.0);
    var march_pos = 0.0;
    for (var i = 0; i < num_samples; i = i + 1) {
        let s = ray.origin + ray.direction * (march_pos + 0.5 * march_step);
        let height = length(s) - earth_radius;
        let hr = exp(-height / hR) * march_step;
        let hm = exp(-height / hM) * march_step;
        optical_depthR += hr;
        optical_depthM += hm;
        let light_ray = Ray(s, sun_dir);
        var optical_depth_lightR = 0.0;
        var optical_depth_lightM = 0.0;
        let overground = get_sun_light(light_ray, &optical_depth_lightR, &optical_depth_lightM);
        if overground {
            let tau = betaR * (optical_depthR + optical_depth_lightR) + betaM * 1.1 * (optical_depthM + optical_depth_lightM);
            let attenuation = exp(-tau);
            sumR += hr * attenuation;
            sumM += hm * attenuation;
        }
        march_pos += march_step;
    }
    var c = sun_power * (sumR * phaseR * betaR + sumM * phaseM * betaM);
    c = pow(c, vec3<f32>(1.5));
    c = c / (1.0 + c);
    c = pow(c, vec3<f32>(1.0 / 1.5));
    c = mix(c, c * c * (3.0 - 2.0 * c), vec3<f32>(1.0));
    c = pow(c, vec3<f32>(1.3, 1.20, 1.0));
    c = pow(c, vec3<f32>(0.7 / 2.2));
    return c;
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

// Added hash23 function for film grain effect
fn hash23(p: vec3f) -> vec2f {
    var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
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
    var d: f32 = settings.max_dist;
    let sc: f32 = 0.3;
    // Terrain generation remains the same
    let q: vec3f = sc * p / 32.0 - vec3f(0.003, -0.006, 0.0);
    d = textureSample(gray_cube_noise_texture, terrain_sampler, q * 1.0).r * 0.5;
    d += textureSample(gray_cube_noise_texture, terrain_sampler, q * 2.0 + vec3f(0.3, 0.3, 0.3)).r * 0.25;
    d += textureSample(gray_cube_noise_texture, terrain_sampler, q * 4.0 + vec3f(0.7, 0.7, 0.7)).r * 0.125;
    var tp = smoothstep(50.0, -6.0, p.y);
    tp = tp * tp;
    d = (d / 0.875 - settings.surface_factor) / sc;
    d = smax(d, p.y - settings.max_height, 0.6);
    
    // let camera_pos = camera.camera_position;
    // let camera_distance = length(p - camera_pos);
    
    // // Remove terrain to close to the camera
    // d = smax(d, settings.min_dist - camera_distance, 0.3);

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
    let s = settings.voxel_size;
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
        if t >= tmax || (rd.y > 0.0 && pos.y > settings.max_height) {
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
    let wk = smoothstep(settings.max_water_height, settings.max_water_height + 0.5, vpos.y);
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
    let hao = smoothstep(settings.water_height - 12.0, settings.water_height, pos.y);
    col *= dot(abs(n), vec3f(0.8, 1.0, 0.9));
    // Fixed: Added missing * operators
    col *= (dif * 0.6 + 0.4) * settings.light_color.rgb;
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
    let hao = smoothstep(settings.water_height - 12.0, settings.water_height, pos.y);

    col *= dot(abs(n), vec3f(0.8, 1.0, 0.9));
    col *= (dif * 0.6 + 0.4) * settings.light_color.rgb;
    col *= ao * 0.6 + 0.4;
    col *= hao * 0.6 + 0.4;

    return col;
}

fn ACESFilm(x: vec3f) -> vec3f {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let tex_coords = array<vec2f, 4>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0),
        vec2f(0.0, 1.0), vec2f(1.0, 1.0)
    );
    let positions = array<vec2f, 4>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, 1.0)
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
    var ndc = vec4f(input.tex_uv * 2.0 - 1.0, 1.0, 1.0);
    let flip_y = true;
    if (flip_y) {
        ndc.y = -ndc.y;
    }
    let world_pos = camera.inv_view_proj * ndc;
    let rd = normalize(world_pos.xyz / world_pos.w - ro);

    let sun_dir = normalize(settings.light_direction.xyz);

    var sky_ro = ro + vec3f(0.0, earth_radius + 1.0, 0.0);

    if settings.visualize_distance_field != 0 {
        let pos = ro + rd * 10.0;
        let d = map(pos);
        return vec4f(vec3f(d * 0.1 + 0.5), 1.0);
    }

    let hit = trace(ro, rd, settings.max_dist);
    var col = vec3f(0.0);
    var t = hit.t;
    
    if hit.is_hit {
        let pos = ro + rd * hit.t;
        let lod = clamp(log2(distance(ro, hit.id)) - 2.0, 0.0, 6.0);
        col = shade(pos, sun_dir, lod, hit);
    } else {
        col = get_incident_light(Ray(sky_ro, rd), sun_dir);
        t = settings.max_dist;
    }

    let pt = -(ro.y - settings.water_height) / rd.y;
    if ((pt > 0.0 && pt < t)) || ro.y < settings.water_height {
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
        if ro.y < settings.water_height && pt < 0.0 {
            adjusted_pt = settings.max_dist;
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
        if ro.y > settings.water_height {
            let hitR = trace(wpos + vec3f(0.0, 0.01, 0.0), wref, 15.0);
            let lod = clamp(log2(distance(ro, hitR.id)) - 2.0, 0.0, 6.0);
            
            if hitR.is_hit {
                rcol = shade2(wpos, sun_dir, lod, hitR);
            } else {
                let w_sky_ro = wpos + vec3f(0.0, earth_radius, 0.0);
                rcol = get_incident_light(Ray(wpos, wref), sun_dir);
            }
        }
        
        // Specular highlight
        let spec = pow(max(dot(wref, sun_dir), 0.0), 50.0);
        
        // Fresnel reflection factor
        let r0 = 0.35;
        var fre = r0 + (1.0 - r0) * pow(max(dot(rd, wn), 0.0), 5.0);

        if rd.y < 0.0 && ro.y < settings.water_height {
            fre = 0.0;
        }
        
        // Water absorption
        let abt = select(t - pt, min(t, pt), ro.y < settings.water_height);
        col *= exp(-abt * (1.0 - wabs) * 0.1);

        if pt < t {
            col = mix(col, wcol * (rcol + spec), fre);
            
            // Foam effect
            let wp = wpos + wn * vec3f(1.0, 0.0, 1.0) * 0.2;
            let wd = map(wp) / length(grad(wp));
            let foam = sin((wd - camera.time * 0.03) * 60.0);
            let foam_mask = smoothstep(0.22, 0.0, wd + foam * 0.03 + (wh - 0.5) * 0.12);
            col = mix(col, col + vec3f(1.0), foam_mask * 0.4);
        }
    }
    
    if settings.show_normals != 0 {
        col = hit.n;
    }

    if settings.show_steps != 0 {
        col = vec3f(f32(hit.i) / f32(settings.steps));
    }

    return vec4f(col, 1.0);
}