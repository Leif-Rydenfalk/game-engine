
// Downsample shader
@fragment
fn downsample_fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(mip_t_input));
    let texel_size = vec2<f32>(1.0 / dims.x, 1.0 / dims.y);
    var color = vec3<f32>(0.0);
    color += textureSample(mip_t_input, mip_s, uv + vec2<f32>(-0.5, -0.5) * texel_size).rgb;
    color += textureSample(mip_t_input, mip_s, uv + vec2<f32>( 0.5, -0.5) * texel_size).rgb;
    color += textureSample(mip_t_input, mip_s, uv + vec2<f32>(-0.5,  0.5) * texel_size).rgb;
    color += textureSample(mip_t_input, mip_s, uv + vec2<f32>( 0.5,  0.5) * texel_size).rgb;
    color *= 0.25; // Average
    return vec4<f32>(color, 1.0);
}


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

const WEIGHTS: array<f32, 5> = array<f32, 5>(0.19638062, 0.29675293, 0.09442139, 0.01037598, 0.00025940);
const OFFSETS: array<f32, 5> = array<f32, 5>(0.0, 1.41176471, 3.29411765, 5.17647059, 7.05882353);

@fragment
fn vertical_fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let dims = textureDimensions(vertical_t_input);
    let texel_size = vec2(1.0 / f32(dims.x), 1.0 / f32(dims.y));
    var color = vec3(0.0);
    var weight_sum = 0.0;
    color += textureSample(vertical_t_input, vertical_s, uv).rgb * WEIGHTS[0];
    weight_sum += WEIGHTS[0];
    for (var i = 1u; i < 5u; i = i + 1u) {
        let offset = vec2(0.0, OFFSETS[i] * texel_size.y);
        color += textureSample(vertical_t_input, vertical_s, uv + offset).rgb * WEIGHTS[i];
        color += textureSample(vertical_t_input, vertical_s, uv - offset).rgb * WEIGHTS[i];
        weight_sum += WEIGHTS[i] * 2.0;
    }
    color /= weight_sum;
    return vec4(color, 1.0);
}


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
    let texel_size = vec2(1.0 / f32(dims.x), 1.0 / f32(dims.y));
    var color = vec3(0.0);
    var weight_sum = 0.0;
    color += textureSample(horizontal_t_input, horizontal_s, uv).rgb * WEIGHTS[0];
    weight_sum += WEIGHTS[0];
    for (var i = 1u; i < 5u; i = i + 1u) {
        let offset = vec2(OFFSETS[i] * texel_size.x, 0.0);
        color += textureSample(horizontal_t_input, horizontal_s, uv + offset).rgb * WEIGHTS[i];
        color += textureSample(horizontal_t_input, horizontal_s, uv - offset).rgb * WEIGHTS[i];
        weight_sum += WEIGHTS[i] * 2.0;
    }
    color /= weight_sum;
    return vec4(color, 1.0);
}

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

// Calculate offset for each octave, matching the reference code
fn mip_calc_offset(octave: f32) -> vec2<f32> {
    let dims = vec2<f32>(textureDimensions(mip_t_input));
    let padding = vec2<f32>(10.0 / dims.x, 10.0 / dims.y);
    var offset = vec2<f32>(0.0);
    offset.x = -min(1.0, floor(octave / 3.0)) * (0.25 + padding.x);
    offset.y = -(1.0 - (1.0 / exp2(octave))) - padding.y * octave;
    offset.y += min(1.0, floor(octave / 3.0)) * 0.35;
    return offset;
}

// Determine oversampling factor based on octave, matching Grab1, Grab4, Grab8, Grab16
fn get_oversampling(octave: f32) -> i32 {
    if (octave <= 1.0) {
        return 1;  // Grab1: single sample
    } else if (octave <= 2.0) {
        return 4;  // Grab4: 4x4 samples
    } else if (octave <= 3.0) {
        return 8;  // Grab8: 8x8 samples
    } else {
        return 16; // Grab16: 16x16 samples for octaves 4+
    }
}

// Updated grab function with oversampling
fn grab(coord: vec2<f32>, octave: f32, offset: vec2<f32>) -> vec3<f32> {
    let scale = exp2(octave);
    let sample_coord = (coord + offset) * scale;

    // Return black if outside texture bounds
    if (sample_coord.x < 0.0 || sample_coord.x > 1.0 || sample_coord.y < 0.0 || sample_coord.y > 1.0) {
        return vec3<f32>(0.0);
    }

    let oversampling = get_oversampling(octave);
    var color = vec3<f32>(0.0);
    let dims = vec2<f32>(textureDimensions(mip_t_input));
    let step = 1.0 / f32(oversampling);

    // Sample multiple points around sample_coord
    for (var i = 0; i < oversampling; i = i + 1) {
        for (var j = 0; j < oversampling; j = j + 1) {
            // Offset in normalized texture coordinates, scaled appropriately
            let off = vec2<f32>(f32(i) * step, f32(j) * step) / dims * scale;
            color += textureSample(mip_t_input, mip_s, sample_coord + off).rgb;
        }
    }
    color /= f32(oversampling * oversampling); // Average the samples
    return color;
}

@fragment
fn mip_fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var color = vec3<f32>(0.0);
    // Sum contributions from all octaves as in the reference code
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
@group(0) @binding(1) var t_bloom0: texture_2d<f32>;
@group(0) @binding(2) var t_bloom1: texture_2d<f32>;
@group(0) @binding(3) var t_bloom2: texture_2d<f32>;
@group(0) @binding(4) var t_bloom3: texture_2d<f32>;
@group(0) @binding(5) var t_bloom4: texture_2d<f32>;
@group(0) @binding(6) var s: sampler;

fn get_bloom(uv: vec2<f32>) -> vec3<f32> {
    var bloom = vec3<f32>(0.0);
    // Sample each level with weights (adjust as needed)
    bloom += textureSample(t_bloom0, s, uv).rgb * 1.0;  // Full size
    bloom += textureSample(t_bloom1, s, uv).rgb * 1.5;  // 1/2
    bloom += textureSample(t_bloom2, s, uv).rgb * 1.0;  // 1/4
    bloom += textureSample(t_bloom3, s, uv).rgb * 1.5;  // 1/8
    bloom += textureSample(t_bloom4, s, uv).rgb * 1.8;  // 1/16
    return bloom;
}

@fragment
fn apply_fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var color = textureSample(t_scene, s, uv).rgb;
    // color += get_bloom(uv) * 1.0; // Adjust bloom intensity
    color = textureSample(t_bloom0, s, uv).rgb * 1.0;
    return vec4<f32>(color, 1.0);
}

fn calc_offset(octave: f32) -> vec2<f32> {
    let dims = vec2<f32>(textureDimensions(t_scene));
    let padding = vec2<f32>(10.0 / dims.x, 10.0 / dims.y);
    var offset = vec2<f32>(0.0);
    offset.x = -min(1.0, floor(octave / 3.0)) * (0.25 + padding.x);
    offset.y = -(1.0 - (1.0 / exp2(octave))) - padding.y * octave;
    offset.y += min(1.0, floor(octave / 3.0)) * 0.35;
    return offset;
}

// Bright extraction shader
@fragment
fn bright_extract_fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let color = textureSample(mip_t_input, mip_s, uv).rgb;
    // Simple threshold for bright parts
    let brightness = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    if (brightness > 1.0) {
        return vec4<f32>(color, 1.0);
    } else {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
}
