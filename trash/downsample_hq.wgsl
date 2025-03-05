// High-quality downsampling filter constants from Shadertoy
const OFFSETS: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2<f32>(-0.75777156, -0.75777156),
    vec2<f32>(0.75777156, -0.75777156),
    vec2<f32>(0.75777156, 0.75777156),
    vec2<f32>(-0.75777156, 0.75777156),
    vec2<f32>(-2.90709914, 0.0),
    vec2<f32>(2.90709914, 0.0),
    vec2<f32>(0.0, -2.90709914),
    vec2<f32>(0.0, 2.90709914)
);

const WEIGHTS: array<f32, 8> = array<f32, 8>(
    0.37487566, 0.37487566, 0.37487566, 0.37487566,
    -0.12487566, -0.12487566, -0.12487566, -0.12487566
);

// Bindings
@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var output_texture: texture_storage_2d<rgba32float, write>;
@group(1) @binding(2) var downsample_sampler: sampler;

@compute @workgroup_size(8, 8)
fn downsample_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }

    let input_dims = textureDimensions(input_texture);
    let input_dims_f = vec2<f32>(input_dims);
    let fragCoord = vec2<f32>(id.xy);
    let uv = 2.0 * (fragCoord + 0.5) / input_dims_f;

    var color = vec4<f32>(0.0);
    for (var i = 0u; i < 8u; i = i + 1u) {
        let offset = OFFSETS[i] / input_dims_f;
        let sample = textureSampleLevel(input_texture, downsample_sampler, uv + offset, 0.0);
        color += sample * WEIGHTS[i];
    }

    textureStore(output_texture, vec2<i32>(i32(id.x), i32(id.y)), color);
}