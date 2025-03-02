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

@group(0) @binding(0) var t_scene: texture_2d<f32>;
@group(0) @binding(1) var t_bloom: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;

fn tonemap(color: vec3<f32>) -> vec3<f32> {
    var c = color;
    c = pow(c, vec3<f32>(1.5));
    c = c / (1.0 + c);
    c = pow(c, vec3<f32>(1.0 / 1.5));
    c = mix(c, c * c * (3.0 - 2.0 * c), vec3<f32>(1.0));
    c = pow(c, vec3<f32>(1.3, 1.20, 1.0));
    c = clamp(c * 1.01, vec3<f32>(0.0), vec3<f32>(1.0));
    c = pow(c, vec3<f32>(0.7 / 2.2));
    return c;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var color = textureSample(t_scene, s, uv).rgb;
    let bloom = textureSample(t_bloom, s, uv).rgb;
    color += bloom * 0.4; // Adjust bloom intensity as needed
    color = tonemap(color);
    return vec4<f32>(color, 1.0);
}