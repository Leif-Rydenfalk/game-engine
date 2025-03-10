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
                //  color += texel * brightness;
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
const BLUR_WEIGHTS: array<f32, 5> = array<f32, 5>(0.19638062, 0.29675293, 0.09442139, 0.01037598, 0.00025940);

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

// Scene and output textures
@group(1) @binding(0) var scene_tex: texture_2d<f32>;
@group(1) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;

// Bloom textures and sampler
@group(2) @binding(0) var bloom0: texture_2d<f32>;
@group(2) @binding(1) var bloom1: texture_2d<f32>;
@group(2) @binding(2) var bloom2: texture_2d<f32>;
@group(2) @binding(3) var bloom3: texture_2d<f32>;
@group(2) @binding(4) var bloom4: texture_2d<f32>;
@group(2) @binding(5) var bloom5: texture_2d<f32>;
@group(2) @binding(6) var bloom6: texture_2d<f32>;
@group(2) @binding(7) var bloom7: texture_2d<f32>;
@group(2) @binding(8) var bloom_sampler: sampler;


fn cubic(v: f32) -> vec4<f32> {
    let n = vec4<f32>(1.0, 2.0, 3.0, 4.0) - v;
    let s = n * n * n;
    let x = s.x;
    let y = s.y - 4.0 * s.x;
    let z = s.z - 4.0 * s.y + 6.0 * s.x;
    let w = 6.0 - x - y - z;
    return vec4<f32>(x, y, z, w) * (1.0 / 6.0);
}

fn textureSampleBicubic(tex: texture_2d<f32>, tex_sampler: sampler, texCoords: vec2<f32>) -> vec4<f32> {
    let texture_size = vec2<f32>(textureDimensions(tex).xy);
    let invTexSize = 1.0 / texture_size;
    var texCoordsAdjusted = texCoords * texture_size - 0.5;
    let fxy = fract(texCoordsAdjusted);
    texCoordsAdjusted = texCoordsAdjusted - fxy;
    let xcubic = cubic(fxy.x);
    let ycubic = cubic(fxy.y);
    let c = texCoordsAdjusted.xxyy + vec2<f32>(-0.5, 1.5).xyxy;
    let s = vec4<f32>(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    var offset = c + vec4<f32>(xcubic.yw, ycubic.yw) / s;
    offset = offset * invTexSize.xxyy;
    let sample0 = textureSampleLevel(tex, tex_sampler, offset.xz, 0.0);
    let sample1 = textureSampleLevel(tex, tex_sampler, offset.yz, 0.0);
    let sample2 = textureSampleLevel(tex, tex_sampler, offset.xw, 0.0);
    let sample3 = textureSampleLevel(tex, tex_sampler, offset.yw, 0.0);
    let sx = s.x / (s.x + s.y);
    let sy = s.z / (s.z + s.w);
    return mix(mix(sample3, sample2, vec4<f32>(sx)), mix(sample1, sample0, vec4<f32>(sx)), vec4<f32>(sy));
}

@compute @workgroup_size(8, 8)
fn composite_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_tex);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }

    // Calculate UV coordinates (normalized [0,1] space)
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(dims.xy);
    
    // Sample scene texture
    var color = textureLoad(scene_tex, vec2<i32>(i32(id.x), i32(id.y)), 0).rgb;
    color = ACESFilm(color);

    var bloom = vec3<f32>(0.0);

    // Sample bloom textures with bicubic filtering and add contributions
    bloom += textureSampleBicubic(bloom0, bloom_sampler, uv).rgb * 1.0;
    bloom += textureSampleBicubic(bloom1, bloom_sampler, uv).rgb * 1.5;
    bloom += textureSampleBicubic(bloom2, bloom_sampler, uv).rgb * 1.0;
    bloom += textureSampleBicubic(bloom3, bloom_sampler, uv).rgb * 1.5;
    bloom += textureSampleBicubic(bloom4, bloom_sampler, uv).rgb * 1.8;
    bloom += textureSampleBicubic(bloom5, bloom_sampler, uv).rgb * 1.0;
    bloom += textureSampleBicubic(bloom6, bloom_sampler, uv).rgb * 1.0;
    bloom += textureSampleBicubic(bloom7, bloom_sampler, uv).rgb * 1.0;

    // Add bloom to scene color
    // color += bloom * 0.3;

    // Write to output texture
    textureStore(output_tex, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(color, 1.0));
}

fn ACESFilm(x: vec3<f32>) -> vec3<f32> {
    let a: f32 = 2.51;
    let b: f32 = 0.03;
    let c: f32 = 2.43;
    let d: f32 = 0.59;
    let e: f32 = 0.14;

    return clamp((x * (a * x + vec3<f32>(b))) / (x * (c * x + vec3<f32>(d)) + vec3<f32>(e)), vec3<f32>(0.0), vec3<f32>(1.0));
}
