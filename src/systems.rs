use crate::*;
use cgmath::Rotation3;
use cgmath::{perspective, InnerSpace, Matrix4, Quaternion, Rad, Vector3, Zero};
use hecs::World;
use std::time::Duration;

pub fn update_camera_system(world: &mut World, input: &Input, dt: Duration) {
    for (_, (transform, camera, controller)) in
        world.query_mut::<(&mut Transform, &mut Camera, &mut CameraController)>()
    {
        let dt = dt.as_secs_f32();

        // Update move speed multiplier with scroll
        controller.move_speed_mult +=
            (controller.move_speed_mult * input.scroll_delta() as f32 * dt * 5.0) as f32;

        // Handle rotation using separate pitch and yaw
        if input.is_mouse_button_down(winit::event::MouseButton::Left) {
            let mouse_delta = input.mouse_delta();

            // Update yaw and pitch, with pitch clamping to prevent camera flipping
            controller.yaw -= Rad(mouse_delta.0 as f32 * controller.look_speed);
            controller.pitch -= Rad(mouse_delta.1 as f32 * controller.look_speed);

            // Clamp pitch to prevent camera flipping
            controller.pitch = controller.pitch;

            // Recreate rotation from yaw and pitch
            transform.rotation = Quaternion::from_axis_angle(Vector3::unit_y(), controller.yaw)
                * Quaternion::from_axis_angle(Vector3::unit_x(), controller.pitch);
        }

        // Calculate movement vectors using current rotation
        let forward = transform.rotation * -Vector3::unit_z();
        let right = transform.rotation * Vector3::unit_x();
        let up = camera.up_vector;

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

        // Apply movement
        if movement != Vector3::zero() {
            movement =
                movement.normalize() * controller.move_speed * controller.move_speed_mult * dt;
            transform.position += movement;
        }
    }
}

pub fn calculate_view_matrix(transform: &Transform) -> Matrix4<f32> {
    let position = transform.position;
    let forward = transform.rotation * -Vector3::unit_z();
    let up = transform.rotation * Vector3::unit_y();
    let target = position + forward;

    Matrix4::look_at_rh(position, target, up)
}

pub fn calculate_view_projection(transform: &Transform, camera: &Camera) -> Matrix4<f32> {
    let view = calculate_view_matrix(transform);
    let proj = perspective(camera.fov, camera.aspect, camera.near, camera.far);
    proj * view
}

pub fn calculate_view(transform: &Transform) -> Matrix4<f32> {
    calculate_view_matrix(transform)
}
