use crate::*;
use cgmath::Rotation3;
use cgmath::{perspective, InnerSpace, Matrix4, Quaternion, Rad, Vector3, Zero};
use cgmath::One;
use hecs::World;
use std::time::Duration;

pub fn update_camera_system(world: &mut World, input: &Input, dt: Duration) {
    for (_, (transform, camera, controller)) in
        world.query_mut::<(&mut Transform, &mut Camera, &mut CameraController)>()
    {
        let dt = dt.as_secs_f32();

        // Update move speed multiplier with scroll - logarithmic scaling for better space navigation
        if input.scroll_delta() != 0.0 {
            let scroll_factor = if input.scroll_delta() > 0.0 { 1.2 } else { 0.8 };
            controller.move_speed_mult *= scroll_factor;
            
            // Clamp the speed multiplier to reasonable values for space navigation
            controller.move_speed_mult = controller.move_speed_mult.clamp(0.01, 1000.0);
        }

        // Calculate camera-relative vectors once
        let forward = transform.rotation * -Vector3::unit_z();
        let right = transform.rotation * Vector3::unit_x();
        let camera_up = forward.cross(right).normalize(); // True camera-relative up
        
        // Handle rotation - using camera-relative axes for consistent control
        if input.is_mouse_button_down(winit::event::MouseButton::Left) {
            let mouse_delta = input.mouse_delta();
            
            // Create rotation based on camera-relative axes for consistency
            let yaw_rotation = Quaternion::from_axis_angle(
                camera_up, 
                Rad(mouse_delta.0 as f32 * controller.look_speed)
            );
            
            let pitch_rotation = Quaternion::from_axis_angle(
                right.normalize(),
                Rad(-mouse_delta.1 as f32 * controller.look_speed)
            );
            
            // Apply rotations in the correct order
            transform.rotation = (yaw_rotation * pitch_rotation) * transform.rotation;
        }

        // Recalculate movement vectors after rotation
        let forward = transform.rotation * -Vector3::unit_z();
        let right = transform.rotation * Vector3::unit_x();
        
        // World up vector for Shift/Space movement
        let world_up = Vector3::unit_y();
        
        // Apply inertia to previous movement for space-like floating
        controller.velocity *= controller.inertia;
        
        // Calculate new movement input
        let mut movement_input = Vector3::zero();
        if input.is_key_down(winit::keyboard::KeyCode::KeyW) {
            movement_input += forward;
        }
        if input.is_key_down(winit::keyboard::KeyCode::KeyS) {
            movement_input -= forward;
        }
        if input.is_key_down(winit::keyboard::KeyCode::KeyA) {
            movement_input -= right;
        }
        if input.is_key_down(winit::keyboard::KeyCode::KeyD) {
            movement_input += right;
        }
        
        // Add vertical movement for Space and Shift
        if input.is_key_down(winit::keyboard::KeyCode::Space) {
            movement_input += world_up;
        }
        if input.is_key_down(winit::keyboard::KeyCode::ShiftLeft) || 
           input.is_key_down(winit::keyboard::KeyCode::ShiftRight) {
            movement_input -= world_up;
        }
        
        // Add thrust to velocity when there's input
        if movement_input != Vector3::zero() {
            let thrust = movement_input.normalize() * controller.move_speed * controller.move_speed_mult * dt;
            controller.velocity += thrust;
            
            // Optional: cap maximum velocity
            let max_speed = controller.move_speed * controller.move_speed_mult * 2.0;
            if controller.velocity.magnitude() > max_speed {
                controller.velocity = controller.velocity.normalize() * max_speed;
            }
        }
        
        // Apply velocity to position
        transform.position += controller.velocity * dt;
        
        // Quick stop if needed
        if input.is_key_down(winit::keyboard::KeyCode::KeyX) {
            controller.velocity *= 0.9; // Rapid deceleration
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
