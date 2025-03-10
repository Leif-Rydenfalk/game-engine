// ACES tone mapping curve fit to go from HDR to LDR
//https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
fn ACESFilm(x: vec3<f32>) -> vec3<f32> {
    let a: f32 = 2.51;
    let b: f32 = 0.03;
    let c: f32 = 2.43;
    let d: f32 = 0.59;
    let e: f32 = 0.14;

    return clamp((x * (a * x + vec3<f32>(b))) / (x * (c * x + vec3<f32>(d)) + vec3<f32>(e)), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn lessThanEqual(a: vec3f, b: vec3f) -> vec3f {
    var one = 0.0;
    if a.x <= b.x { one = 1.0; } else { one = 0.0; };
    var two = 0.0;
    if a.y <= b.y { two = 1.0; } else { two = 0.0; };
    var three = 0.0;
    if a.z <= b.z { three = 1.0; } else { three = 0.0; };
    return vec3f(one, two, three);
}

fn linearTosRGB(col: vec3f) -> vec3f {
    return mix(
        1.055 * pow(col, vec3f(1.0 / 2.4)) - 0.055,
        col * 12.92,
        vec3f(lessThanEqual(col, vec3f(0.0031308)))
    );
}

fn tonemap(color: vec3<f32>) -> vec3<f32> {
    var c = color;
    // c = pow(c, vec3<f32>(1.5));
    // c = c / (1.0 + c);
    // c = pow(c, vec3<f32>(1.0 / 1.5));
    // c = mix(c, c * c * (3.0 - 2.0 * c), vec3<f32>(1.0));
    // c = pow(c, vec3<f32>(1.3, 1.20, 1.0));
    // c = pow(c, vec3<f32>(0.7 / 2.2));


    // c = ACESFilm(c * 0.3);
    // c = ACESFilm(c);
    // c = linearTosRGB(c);
    return c;
}

// Vertex Shader: Full-screen quad
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

// Fragment Shader: Color correction
struct ColorCorrectionUniform {
    brightness: f32,
    contrast: f32,
    saturation: f32,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var<uniform> cc_uniform: ColorCorrectionUniform;

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = textureDimensions(input_texture);
    let tex_coord = frag_coord.xy / vec2<f32>(f32(dims.x), f32(dims.y));
    var color = textureSample(input_texture, input_sampler, tex_coord);
    color = vec4(tonemap(color.rgb), 1.0);
    return color;
}
