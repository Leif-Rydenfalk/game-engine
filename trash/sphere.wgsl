
// //--------------------------------------------------------
// // Sphere related functions
// //--------------------------------------------------------

// fn sphIntersect(ro: vec3<f32>, rd: vec3<f32>, sph: vec4<f32>) -> f32 {
//     let oc: vec3<f32> = ro - sph.xyz;
//     let b: f32 = dot(oc, rd);
//     let c: f32 = dot(oc, oc) - sph.w * sph.w;
//     let h: f32 = b * b - c;
//     if (h < 0.0) {
//         return -1.0;
//     }
//     return -b - sqrt(h);
// }

// fn sphSoftShadow(ro: vec3<f32>, rd: vec3<f32>, sph: vec4<f32>, k: f32) -> f32 {
//     let oc: vec3<f32> = ro - sph.xyz;
//     let b: f32 = dot(oc, rd);
//     let c: f32 = dot(oc, oc) - sph.w * sph.w;
//     let h: f32 = b * b - c;
//     // Using the "cheap" alternative (not physically plausible)
//     if (b > 0.0) {
//         return step(-0.0001, c);
//     } else {
//         return smoothstep(0.0, 1.0, h * k / b);
//     }
// }

// fn sphOcclusion(pos: vec3<f32>, nor: vec3<f32>, sph: vec4<f32>) -> f32 {
//     let r: vec3<f32> = sph.xyz - pos;
//     let l: f32 = length(r);
//     return dot(nor, r) * (sph.w * sph.w) / (l * l * l);
// }

// fn sphNormal(pos: vec3<f32>, sph: vec4<f32>) -> vec3<f32> {
//     return normalize(pos - sph.xyz);
// }

// //--------------------------------------------------------
// // Plane intersection function
// //--------------------------------------------------------

// fn iPlane(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
//     return (-1.0 - ro.y) / rd.y;
// }


// // Fragment Shader
// @fragment
// fn fs_main(input: VertexOutput) -> @location(0) vec4f {
//     // remap pixel coordinates
//     let p: vec2<f32> = input.tex_uv.xy;

//     let ro = camera.camera_position;
//     let ndc = vec4f(input.tex_uv * 2.0 - 1.0, 1.0, 1.0);
//     let world_pos = camera.inv_view_proj * ndc;
//     let rd = normalize(world_pos.xyz / world_pos.w - ro);
    
//     // Sphere animation: animate position based on time and optionally mouse input
//     var sph: vec4<f32> = vec4<f32>(
//         0.0, 0.0, 0.0,
//         1.0
//     );

//     let lig: vec3<f32> = normalize(vec3<f32>(0.6, 0.3, 0.4));
//     var col: vec3<f32> = vec3<f32>(0.0);
//     var tmin: f32 = 1e10;
//     var nor: vec3<f32>;
//     var occ: f32 = 1.0;

//     // Sphere intersection
//     let t2: f32 = sphIntersect(ro, rd, sph);
//     if (t2 > 0.0 && t2 < tmin) {
//         tmin = t2;
//         let pos: vec3<f32> = ro + t2 * rd;
//         nor = sphNormal(pos, sph);
//         occ = 0.5 + 0.5 * nor.y;
//     }

//     if (tmin < 1000.0) {
//         let pos: vec3<f32> = ro + tmin * rd;
//         col = vec3<f32>(1.0);
//         col = col * clamp(dot(nor, lig), 0.0, 1.0);
//         col = col * sphSoftShadow(pos, lig, sph, 2.0);
//         col = col + 0.05 * occ;
//         col = col * exp(-0.05 * tmin);
//     }

//     // Gamma correction (approximate)
//     col = sqrt(col);
//     return vec4<f32>(col, 1.0);
// }




    // // Sphere animation: animate position based on time and optionally mouse input
    // var sph: vec4<f32> = vec4<f32>(
    //     0.0, 0.0, 0.0,
    //     10.1
    // );

    // let lig: vec3<f32> = normalize(vec3<f32>(0.6, 0.3, 0.4));
    // var c: vec3<f32> = vec3<f32>(0.0);
    // var tmin: f32 = 1e10;
    // var nor: vec3<f32>;
    // var occ: f32 = 1.0;

    // // Sphere intersection
    // let t2: f32 = sphIntersect(ro, rd, sph);
    // if (t2 > 0.0 && t2 < tmin) {
    //     tmin = t2;
    //     let pos: vec3<f32> = ro + t2 * rd;
    //     nor = sphNormal(pos, sph);
    //     occ = 0.5 + 0.5 * nor.y;
    // }

    // if (tmin < 1000.0) {
    //     let pos: vec3<f32> = ro + tmin * rd;
    //     c = vec3<f32>(1.0);
    //     c = c * clamp(dot(nor, lig), 0.0, 1.0);
    //     c = c * sphSoftShadow(pos, lig, sph, 2.0);
    //     c = c + 0.05 * occ;
    //     c = c * exp(-0.05 * tmin);
    // }

    // c = sqrt(c);
    // if (c.r>0.0) {
    //     c = nor;
    // }
