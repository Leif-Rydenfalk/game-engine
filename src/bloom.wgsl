// Global constants for blur passes, matching Shadertoy
const WEIGHTS: array<f32, 5> = array<f32, 5>(0.19638062, 0.29675293, 0.09442139, 0.01037598, 0.00025940);
const OFFSETS: array<f32, 5> = array<f32, 5>(0.0, 1.41176471, 3.29411765, 5.17647059, 7.05882353);

// Bicubic sampling functions, translated from Shadertoy
fn cubic(x: f32) -> vec4<f32> {
    let x2 = x * x;
    let x3 = x2 * x;
    var w: vec4<f32>;
    w.x = -x3 + 3.0 * x2 - 3.0 * x + 1.0;
    w.y = 3.0 * x3 - 6.0 * x2 + 4.0;
    w.z = -3.0 * x3 + 3.0 * x2 + 3.0 * x + 1.0;
    w.w = x3;
    return w / 6.0;
}

fn bicubic_texture(tex: texture_2d<f32>, s: sampler, coord: vec2<f32>) -> vec4<f32> {
    let resolution = vec2<f32>(textureDimensions(tex));
    var coord = coord * resolution;
    let fx = fract(coord.x);
    let fy = fract(coord.y);
    coord.x -= fx;
    coord.y -= fy;
    let fx_adj = fx - 0.5;
    let fy_adj = fy - 0.5;
    let xcubic = cubic(fx_adj);
    let ycubic = cubic(fy_adj);
    let c = vec4<f32>(coord.x - 0.5, coord.x + 1.5, coord.y - 0.5, coord.y + 1.5);
    let s_vec = vec4<f32>(xcubic.x + xcubic.y, xcubic.z + xcubic.w, ycubic.x + ycubic.y, ycubic.z + ycubic.w);
    let offset = c + vec4<f32>(xcubic.y, xcubic.w, ycubic.y, ycubic.w) / s_vec;
    let sample0 = textureSample(tex, s, vec2<f32>(offset.x, offset.z) / resolution);
    let sample1 = textureSample(tex, s, vec2<f32>(offset.y, offset.z) / resolution);
    let sample2 = textureSample(tex, s, vec2<f32>(offset.x, offset.w) / resolution);
    let sample3 = textureSample(tex, s, vec2<f32>(offset.y, offset.w) / resolution);
    let sx = s_vec.x / (s_vec.x + s_vec.y);
    let sy = s_vec.z / (s_vec.z + s_vec.w);
    return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}

// **Vertical Blur Pass**
struct VerticalVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vertical_vs_main(@builtin(vertex_index) vertex_index: u32) -> VerticalVertexOutput {
    let positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );
    let uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0)
    );
    var output: VerticalVertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.uv = uvs[vertex_index];
    return output;
}

@group(0) @binding(0) var vertical_t_input: texture_2d<f32>;
@group(0) @binding(1) var vertical_s: sampler;

@fragment
fn vertical_fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let dims = textureDimensions(vertical_t_input);
    let texel_size = vec2<f32>(1.0 / f32(dims.x), 1.0 / f32(dims.y));
    var color = vec3<f32>(0.0);
    var weight_sum = 0.0;
    if (uv.x < 0.52) {  // Matches Shadertoy condition
        color += textureSample(vertical_t_input, vertical_s, uv).rgb * WEIGHTS[0];
        weight_sum += WEIGHTS[0];
        for (var i = 1u; i < 5u; i = i + 1u) {
            let offset = vec2<f32>(0.0, OFFSETS[i] * texel_size.y * 0.5);
            color += textureSample(vertical_t_input, vertical_s, uv + offset).rgb * WEIGHTS[i];
            color += textureSample(vertical_t_input, vertical_s, uv - offset).rgb * WEIGHTS[i];
            weight_sum += WEIGHTS[i] * 2.0;
        }
        color /= weight_sum;
    }
    return vec4<f32>(color, 1.0);
}

// **Horizontal Blur Pass**
struct HorizontalVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn horizontal_vs_main(@builtin(vertex_index) vertex_index: u32) -> HorizontalVertexOutput {
    let positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );
    let uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0)
    );
    var output: HorizontalVertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.uv = uvs[vertex_index];
    return output;
}

@group(0) @binding(0) var horizontal_t_input: texture_2d<f32>;
@group(0) @binding(1) var horizontal_s: sampler;

@fragment
fn horizontal_fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let dims = textureDimensions(horizontal_t_input);
    let texel_size = vec2<f32>(1.0 / f32(dims.x), 1.0 / f32(dims.y));
    var color = vec3<f32>(0.0);
    var weight_sum = 0.0;
    if (uv.x < 0.52) {  // Matches Shadertoy condition
        color += textureSample(horizontal_t_input, horizontal_s, uv).rgb * WEIGHTS[0];
        weight_sum += WEIGHTS[0];
        for (var i = 1u; i < 5u; i = i + 1u) {
            let offset = vec2<f32>(OFFSETS[i] * texel_size.x * 0.5, 0.0);
            color += textureSample(horizontal_t_input, horizontal_s, uv + offset).rgb * WEIGHTS[i];
            color += textureSample(horizontal_t_input, horizontal_s, uv - offset).rgb * WEIGHTS[i];
            weight_sum += WEIGHTS[i] * 2.0;
        }
        color /= weight_sum;
    }
    return vec4<f32>(color, 1.0);
}

// **Mipmap Tree Pass**
struct MipVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn mip_vs_main(@builtin(vertex_index) vertex_index: u32) -> MipVertexOutput {
    let positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );
    let uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0)
    );
    var output: MipVertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.uv = uvs[vertex_index];
    return output;
}

@group(0) @binding(0) var mip_t_input: texture_2d<f32>;
@group(0) @binding(1) var mip_s: sampler;

fn mip_calc_offset(octave: f32) -> vec2<f32> {
    let dims = vec2<f32>(textureDimensions(mip_t_input));
    let padding = vec2<f32>(10.0 / dims.x, 10.0 / dims.y);
    var offset = vec2<f32>(0.0);
    offset.x = -min(1.0, floor(octave / 3.0)) * (0.25 + padding.x);
    offset.y = -(1.0 - (1.0 / exp2(octave))) - padding.y * octave;
    offset.y += min(1.0, floor(octave / 3.0)) * 0.35;
    return offset;
}

fn get_oversampling(octave: f32) -> i32 {
    if (octave <= 1.0) {
        return 1;  // Grab1: 1x1
    } else if (octave <= 2.0) {
        return 4;  // Grab4: 4x4
    } else if (octave <= 3.0) {
        return 8;  // Grab8: 8x8
    } else {
        return 16; // Grab16: 16x16
    }
}

fn grab(coord: vec2<f32>, octave: f32, offset: vec2<f32>) -> vec3<f32> {
    let scale = exp2(octave);
    let sample_coord = (coord + offset) * scale;
    if (sample_coord.x < 0.0 || sample_coord.x > 1.0 || sample_coord.y < 0.0 || sample_coord.y > 1.0) {
        return vec3<f32>(0.0);
    }
    let oversampling = get_oversampling(octave);
    var color = vec3<f32>(0.0);
    let dims = vec2<f32>(textureDimensions(mip_t_input));
    let step = 1.0 / f32(oversampling);
    for (var i = 0; i < oversampling; i = i + 1) {
        for (var j = 0; j < oversampling; j = j + 1) {
            let off = vec2<f32>(f32(i) * step, f32(j) * step) / dims * scale;
            color += textureSample(mip_t_input, mip_s, sample_coord + off).rgb;
        }
    }
    color /= f32(oversampling * oversampling);
    return color;
}

@fragment
fn mip_fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var color = vec3<f32>(0.0);
    color += grab(uv, 1.0, vec2<f32>(0.0, 0.0));
    color += grab(uv, 2.0, mip_calc_offset(1.0));
    color += grab(uv, 3.0, mip_calc_offset(2.0));
    color += grab(uv, 4.0, mip_calc_offset(3.0));
    color += grab(uv, 5.0, mip_calc_offset(4.0));
    color += grab(uv, 6.0, mip_calc_offset(5.0));
    color += grab(uv, 7.0, mip_calc_offset(6.0));
    color += grab(uv, 8.0, mip_calc_offset(7.0));
    return vec4<f32>(color, 1.0);
}

// **Final Composition Pass**
struct ApplyVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn apply_vs_main(@builtin(vertex_index) vertex_index: u32) -> ApplyVertexOutput {
    let positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );
    let uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0)
    );
    var output: ApplyVertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.uv = uvs[vertex_index];
    return output;
}

@group(0) @binding(0) var t_scene: texture_2d<f32>;
@group(0) @binding(1) var t_bloom: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;

fn calc_offset(octave: f32) -> vec2<f32> {
    let dims = vec2<f32>(textureDimensions(t_scene));
    let padding = vec2<f32>(10.0 / dims.x, 10.0 / dims.y);
    var offset = vec2<f32>(0.0);
    offset.x = -min(1.0, floor(octave / 3.0)) * (0.25 + padding.x);
    offset.y = -(1.0 - (1.0 / exp2(octave))) - padding.y * octave;
    offset.y += min(1.0, floor(octave / 3.0)) * 0.35;
    return offset;
}

fn get_bloom(coord: vec2<f32>) -> vec3<f32> {
    var bloom = vec3<f32>(0.0);
    bloom += bicubic_texture(t_bloom, s, (coord / exp2(1.0)) - calc_offset(0.0)).rgb * 1.0;
    bloom += bicubic_texture(t_bloom, s, (coord / exp2(2.0)) - calc_offset(1.0)).rgb * 1.5;
    bloom += bicubic_texture(t_bloom, s, (coord / exp2(3.0)) - calc_offset(2.0)).rgb * 1.0;
    bloom += bicubic_texture(t_bloom, s, (coord / exp2(4.0)) - calc_offset(3.0)).rgb * 1.5;
    bloom += bicubic_texture(t_bloom, s, (coord / exp2(5.0)) - calc_offset(4.0)).rgb * 1.8;
    bloom += bicubic_texture(t_bloom, s, (coord / exp2(6.0)) - calc_offset(5.0)).rgb * 1.0;
    bloom += bicubic_texture(t_bloom, s, (coord / exp2(7.0)) - calc_offset(6.0)).rgb * 1.0;
    bloom += bicubic_texture(t_bloom, s, (coord / exp2(8.0)) - calc_offset(7.0)).rgb * 1.0;
    return bloom;
}

@fragment
fn apply_fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var color = textureSample(t_scene, s, uv).rgb;
    color += get_bloom(uv) * 0.05;
    color *= 200.0;
    color = pow(color, vec3<f32>(1.5));
    color = color / (1.0 + color);
    color = pow(color, vec3<f32>(1.0 / 1.5));
    color = mix(color, color * color * (3.0 - 2.0 * color), vec3<f32>(1.0));
    color = pow(color, vec3<f32>(1.3, 1.20, 1.0));
    color = clamp(color * 1.01, vec3<f32>(0.0), vec3<f32>(1.0));
    color = pow(color, vec3<f32>(0.7 / 2.2));
    return vec4<f32>(color, 1.0);
}