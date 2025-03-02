@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = global_id.xy;
    let dims = textureDimensions(input_texture);
    if (coord.x >= dims.x || coord.y >= dims.y) {
        return;
    }
    var color = textureLoad(input_texture, coord, 0);
    color.r += 0.1;
    textureStore(output_texture, coord, vec4<f32>(color.rgb, color.a));
}