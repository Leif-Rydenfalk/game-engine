struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
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
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.uv = uvs[vertex_index];
    return output;
}

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

const WEIGHTS: array<f32, 5> = array<f32, 5>(0.19638062, 0.29675293, 0.09442139, 0.01037598, 0.00025940);
const OFFSETS: array<f32, 5> = array<f32, 5>(0.0, 1.41176471, 3.29411765, 5.17647059, 7.05882353);

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var color = vec3(0.0);
    if (uv.y < 0.52) { // Adjust this threshold based on your mipmap layout
        let dims = textureDimensions(t_input);
        let texel_size = vec2(1.0 / f32(dims.y), 1.0 / f32(dims.y));
        var weight_sum = 0.0;
        color += textureSample(t_input, s, uv).rgb * WEIGHTS[0];
        weight_sum += WEIGHTS[0];
        for (var i = 1u; i < 5u; i = i + 1u) {
            let offset = vec2(0.0, OFFSETS[i] * texel_size.y); // Vertical
            // or vec2(0.0, OFFSETS[i] * texel_size.y) for vertical
            color += textureSample(t_input, s, uv + offset).rgb * WEIGHTS[i];
            color += textureSample(t_input, s, uv - offset).rgb * WEIGHTS[i];
            weight_sum += WEIGHTS[i] * 2.0;
        }
        color /= weight_sum;
    } else {
        color = textureSample(t_input, s, uv).rgb;
    }
    return vec4(color, 1.0);
}
