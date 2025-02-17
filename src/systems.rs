use crate::*;
use cgmath::{perspective, InnerSpace, Matrix4, Point3, Rad, Vector3, Zero};
use hecs::World;
use std::time::Duration;
use winit::event::MouseButton;

pub fn update_camera_system(world: &mut World, input: &Input, dt: Duration) {
    // Get all entities with both Transform and CameraController components
    for (_, (transform, controller)) in world.query_mut::<(&mut Transform, &CameraController)>() {
        let dt = dt.as_secs_f32();

        // Handle rotation
        if input.is_mouse_button_down(winit::event::MouseButton::Left) {
            let mouse_delta = input.mouse_delta();
            transform.rotation.1 += Rad(mouse_delta.0 as f32 * controller.look_speed); // yaw
            transform.rotation.0 += Rad(-mouse_delta.1 as f32 * controller.look_speed); // pitch

            // Clamp pitch to prevent camera flipping
            transform.rotation.0 .0 = transform.rotation.0 .0.clamp(
                -std::f32::consts::FRAC_PI_2 + 0.1,
                std::f32::consts::FRAC_PI_2 - 0.1,
            );
        }

        // Calculate movement vectors
        let (sin_pitch, cos_pitch) = transform.rotation.0 .0.sin_cos();
        let (sin_yaw, cos_yaw) = transform.rotation.1 .0.sin_cos();

        let forward = Vector3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize();

        let right = forward.cross(Vector3::unit_y()).normalize();
        let up = right.cross(forward).normalize();

        // Handle movement
        let mut movement = Vector3::zero();

        if input.is_key_down(winit::keyboard::KeyCode::KeyW) {
            movement += forward;
        }
        if input.is_key_down(winit::keyboard::KeyCode::KeyS) {
            movement -= forward;
        }
        if input.is_key_down(winit::keyboard::KeyCode::KeyA) {
            movement -= right;
        }
        if input.is_key_down(winit::keyboard::KeyCode::KeyD) {
            movement += right;
        }
        if input.is_key_down(winit::keyboard::KeyCode::Space) {
            movement += up;
        }
        if input.is_key_down(winit::keyboard::KeyCode::ShiftLeft) {
            movement -= up;
        }

        if movement != Vector3::zero() {
            movement = movement.normalize() * controller.move_speed * dt;
            transform.position += movement;
        }
    }
}

pub fn calculate_view_matrix(transform: &Transform) -> Matrix4<f32> {
    let (sin_pitch, cos_pitch) = transform.rotation.0 .0.sin_cos();
    let (sin_yaw, cos_yaw) = transform.rotation.1 .0.sin_cos();

    let forward = Vector3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize();

    Matrix4::look_to_rh(transform.position, forward, Vector3::unit_y())
}

pub fn calculate_view_projection(transform: &Transform, camera: &Camera) -> Matrix4<f32> {
    let view = calculate_view_matrix(transform);
    let proj = perspective(camera.fov, camera.aspect, camera.near, camera.far);
    proj * view
}
