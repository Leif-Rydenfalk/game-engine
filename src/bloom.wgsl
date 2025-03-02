struct BloomSettings {
    min_brightness: f32,
    max_brightness: f32,
    blur_radius: f32,
    blur_type: u32,
};

// Uniform buffer in group 0
@group(0) @binding(0) var<uniform> settings: BloomSettings;

// Prefilter Shader
@group(1) @binding(0) var scene: texture_2d<f32>;
@group(1) @binding(1) var output: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn prefilter_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    let scene_dims = textureDimensions(scene);
    let x = id.x * 2u;
    let y = id.y * 2u;
    var color = vec4<f32>(0.0);
    for (var dx = 0u; dx < 2u; dx = dx + 1u) {
        for (var dy = 0u; dy < 2u; dy = dy + 1u) {
            if (x + dx < scene_dims.x && y + dy < scene_dims.y) {
                let texel = textureLoad(scene, vec2<i32>(i32(x + dx), i32(y + dy)), 0);
                let brightness = dot(texel.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
                let factor = smoothstep(settings.min_brightness, settings.max_brightness, brightness);
                color += texel * factor;
            }
        }
    }
    color /= 4.0;
    textureStore(output, vec2<i32>(i32(id.x), i32(id.y)), color);
}

// Downsample Shader
@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var output_texture: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn downsample_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    let input_dims = textureDimensions(input_texture);
    let x = id.x * 2u;
    let y = id.y * 2u;
    var color = vec4<f32>(0.0);
    var count = 0.0;
    for (var dx = 0u; dx < 2u; dx = dx + 1u) {
        for (var dy = 0u; dy < 2u; dy = dy + 1u) {
            if (x + dx < input_dims.x && y + dy < input_dims.y) {
                color += textureLoad(input_texture, vec2<i32>(i32(x + dx), i32(y + dy)), 0);
                count += 1.0;
            }
        }
    }
    color /= count;
    textureStore(output_texture, vec2<i32>(i32(id.x), i32(id.y)), color);
}

// Blur Shaders (5-tap Gaussian)
const BLUR_WEIGHTS: array<f32, 5> = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

@compute @workgroup_size(8, 8)
fn horizontal_blur_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    var color = vec3<f32>(0.0);
    for (var i = -2; i <= 2; i = i + 1) {
        let offset = i32(i) * i32(settings.blur_radius);
        let coord = i32(id.x) + offset;
        if (coord >= 0 && coord < i32(dims.x)) {
            color += textureLoad(input_texture, vec2<i32>(coord, i32(id.y)), 0).rgb * BLUR_WEIGHTS[u32(abs(i))];
        }
    }
    textureStore(output_texture, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(color, 1.0));
}

@compute @workgroup_size(8, 8)
fn vertical_blur_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    var color = vec3<f32>(0.0);
    for (var i = -2; i <= 2; i = i + 1) {
        let offset = i32(i) * i32(settings.blur_radius);
        let coord = i32(id.y) + offset;
        if (coord >= 0 && coord < i32(dims.y)) {
            color += textureLoad(input_texture, vec2<i32>(i32(id.x), coord), 0).rgb * BLUR_WEIGHTS[u32(abs(i))];
        }
    }
    textureStore(output_texture, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(color, 1.0));
}

const COMPOSITION_WEIGHTS: array<f32, 8> = array<f32, 8>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216, 0.004054, 0.001216, 0.000316);

// Composite Shader
@group(1) @binding(0) var scene_tex: texture_2d<f32>;
@group(2) @binding(0) var bloom0: texture_2d<f32>;
@group(2) @binding(1) var bloom1: texture_2d<f32>;
@group(2) @binding(2) var bloom2: texture_2d<f32>;
@group(2) @binding(3) var bloom3: texture_2d<f32>;
@group(2) @binding(4) var bloom4: texture_2d<f32>;
@group(2) @binding(5) var bloom5: texture_2d<f32>;
@group(2) @binding(6) var bloom6: texture_2d<f32>;
@group(2) @binding(7) var bloom7: texture_2d<f32>;
@group(1) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn composite_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_tex);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    var color = textureLoad(scene_tex, vec2<i32>(i32(id.x), i32(id.y)), 0).rgb;

    // Bloom0 (mip level offset by 1)
    var mip_x = id.x >> 1u;
    var mip_y = id.y >> 1u;
    var mip_dims = textureDimensions(bloom0);
    if (mip_x < mip_dims.x && mip_y < mip_dims.y) {
        color += textureLoad(bloom0, vec2<i32>(i32(mip_x), i32(mip_y)), 0).rgb * COMPOSITION_WEIGHTS[0];
    }

    // Bloom1 (mip level offset by 2)
    mip_x = id.x >> 2u;
    mip_y = id.y >> 2u;
    mip_dims = textureDimensions(bloom1);
    if (mip_x < mip_dims.x && mip_y < mip_dims.y) {
        color += textureLoad(bloom1, vec2<i32>(i32(mip_x), i32(mip_y)), 0).rgb * COMPOSITION_WEIGHTS[1];
    }

    // Bloom2 (mip level offset by 3)
    mip_x = id.x >> 3u;
    mip_y = id.y >> 3u;
    mip_dims = textureDimensions(bloom2);
    if (mip_x < mip_dims.x && mip_y < mip_dims.y) {
        color += textureLoad(bloom2, vec2<i32>(i32(mip_x), i32(mip_y)), 0).rgb * COMPOSITION_WEIGHTS[2];
    }

    // Bloom3 (mip level offset by 4)
    mip_x = id.x >> 4u;
    mip_y = id.y >> 4u;
    mip_dims = textureDimensions(bloom3);
    if (mip_x < mip_dims.x && mip_y < mip_dims.y) {
        color += textureLoad(bloom3, vec2<i32>(i32(mip_x), i32(mip_y)), 0).rgb * COMPOSITION_WEIGHTS[3];
    }

    // Bloom4 (mip level offset by 5)
    mip_x = id.x >> 5u;
    mip_y = id.y >> 5u;
    mip_dims = textureDimensions(bloom4);
    if (mip_x < mip_dims.x && mip_y < mip_dims.y) {
        color += textureLoad(bloom4, vec2<i32>(i32(mip_x), i32(mip_y)), 0).rgb * COMPOSITION_WEIGHTS[4];
    }

    // Bloom5 (mip level offset by 6)
    mip_x = id.x >> 6u;
    mip_y = id.y >> 6u;
    mip_dims = textureDimensions(bloom5);
    if (mip_x < mip_dims.x && mip_y < mip_dims.y) {
        color += textureLoad(bloom5, vec2<i32>(i32(mip_x), i32(mip_y)), 0).rgb * COMPOSITION_WEIGHTS[5];
    }

    // Bloom6 (mip level offset by 7)
    mip_x = id.x >> 7u;
    mip_y = id.y >> 7u;
    mip_dims = textureDimensions(bloom6);
    if (mip_x < mip_dims.x && mip_y < mip_dims.y) {
        color += textureLoad(bloom6, vec2<i32>(i32(mip_x), i32(mip_y)), 0).rgb * COMPOSITION_WEIGHTS[6];
    }

    // Bloom7 (mip level offset by 8)
    mip_x = id.x >> 8u;
    mip_y = id.y >> 8u;
    mip_dims = textureDimensions(bloom7);
    if (mip_x < mip_dims.x && mip_y < mip_dims.y) {
        color += textureLoad(bloom7, vec2<i32>(i32(mip_x), i32(mip_y)), 0).rgb * COMPOSITION_WEIGHTS[7];
    }

    textureStore(output_tex, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(color, 1.0));
}