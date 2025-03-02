fn tonemap(color: vec3<f32>) -> vec3<f32> {
    var c = color;
    c = pow(c, vec3<f32>(1.5));
    c = c / (1.0 + c);
    c = pow(c, vec3<f32>(1.0 / 1.5));
    c = mix(c, c * c * (3.0 - 2.0 * c), vec3<f32>(1.0));
    c = pow(c, vec3<f32>(1.3, 1.20, 1.0));
    c = pow(c, vec3<f32>(0.7 / 2.2));
    return c;
}
