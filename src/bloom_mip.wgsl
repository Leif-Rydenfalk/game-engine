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

fn calc_offset(octave: f32) -> vec2<f32> {
    let dims = vec2<f32>(textureDimensions(t_input));
    let padding = vec2<f32>(10.0 / dims.x, 10.0 / dims.y);
    var offset = vec2<f32>(0.0);
    offset.x = -min(1.0, floor(octave / 3.0)) * (0.25 + padding.x);
    offset.y = -(1.0 - (1.0 / exp2(octave))) - padding.y * octave;
    offset.y += min(1.0, floor(octave / 3.0)) * 0.35;
    return offset;
}

fn grab(coord: vec2<f32>, octave: f32, offset: vec2<f32>) -> vec3<f32> {
    let scale = exp2(octave);
    let sample_coord = (coord + offset) * scale;
    if (sample_coord.x < 0.0 || sample_coord.x > 1.0 || sample_coord.y < 0.0 || sample_coord.y > 1.0) {
        return vec3<f32>(0.0);
    }
    return textureSample(t_input, s, sample_coord).rgb;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var color = vec3<f32>(0.0);
    color += grab(uv, 1.0, vec2<f32>(0.0, 0.0));
    color += grab(uv, 2.0, calc_offset(1.0));
    color += grab(uv, 3.0, calc_offset(2.0));
    color += grab(uv, 4.0, calc_offset(3.0));
    return vec4<f32>(color, 1.0);
}