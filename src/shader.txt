// of type rs because deepeek wont let me upload a .wgsl file

@group(0) @binding(0)
var texture_2d_instance: texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler_instance: sampler;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) tex_uv: vec2f,
    @location(2) normal: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) tex_uv: vec2f,
    @location(1) normal: vec3f,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = camera.view_proj * vec4f(input.position, 1.0);
    output.tex_uv = input.tex_uv;
    output.normal = input.normal;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let light_dir = normalize(vec3f(1.0, 1.0, 0.8));
    let ambient = 0.1;
    let diffuse = max(dot(input.normal, light_dir), 0.0);
    let color = textureSample(texture_2d_instance, texture_sampler_instance, input.tex_uv).rgb;
    let lit_color = (ambient + diffuse) * color;
    return vec4f(lit_color, 1.0);
}
