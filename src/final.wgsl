struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec2f, 4>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, 1.0)
    );
    // let tex_coords = array<vec2f, 4>(
    //     vec2f(0.0, 0.0), vec2f(1.0, 0.0),
    //     vec2f(0.0, 1.0), vec2f(1.0, 1.0)
    // );
    let tex_coords = array<vec2f, 4>(
        vec2f(0.0, 1.0), vec2f(1.0, 1.0),
        vec2f(0.0, 0.0), vec2f(1.0, 0.0)
    );
    var output: VertexOutput;
    output.clip_position = vec4f(positions[vertex_index], 0.0, 1.0);
    output.tex_coords = tex_coords[vertex_index];
    return output;
}
// Fragment shader
@group(0) @binding(0)
var t_texture: texture_2d<f32>;
@group(0) @binding(1)
var s_sampler: sampler;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_texture, s_sampler, input.tex_coords);
}