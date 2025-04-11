use crate::*;
use cgmath::One;
use cgmath::Point3;
use cgmath::Rotation3;
use cgmath::{perspective, InnerSpace, Matrix4, Quaternion, Rad, Vector3, Zero};
use hecs::Entity;
use hecs::World;
use std::time::Duration;
use tracing::info;

#[derive(Debug)]
pub struct Transform {
    pub position: Point3<f32>,
    pub rotation: Quaternion<f32>,
    pub scale: Vector3<f32>,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::from_axis_angle(Vector3::unit_y(), Rad(0.0)),
            scale: Vector3::new(1.0, 1.0, 1.0),
        }
    }
}

#[derive(Debug)]
pub struct Camera {
    pub fov: Rad<f32>,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    // We'll keep the up_vector but it's only used for the view matrix calculation,
    // not for constraining movement
    pub up_vector: Vector3<f32>,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fov: Rad(std::f32::consts::FRAC_PI_4),
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 10000.0,
            up_vector: Vector3::unit_y(),
        }
    }
}

#[derive(Debug)]
pub struct ModelInstance {
    pub model: usize, // Index into the model registry
}

pub fn spawn_model_entity(
    world: &mut World,
    model_index: usize,
    position: Point3<f32>,
) -> hecs::Entity {
    world.spawn((
        Transform {
            position,
            ..Default::default()
        },
        ModelInstance { model: model_index },
    ))
}

// Add this struct to store euler angles for the FPS camera
#[derive(Debug, Default)]
pub struct FPSController {
    pub yaw: f32,   // Rotation around Y axis (left/right)
    pub pitch: f32, // Rotation around X axis (up/down)
}

// In world.rs, modify setup_camera_entity to include the FPSController component
pub fn setup_camera_entity(world: &mut World, window_size: Option<(u32, u32)>) -> hecs::Entity {
    // Calculate initial aspect ratio based on window size, or use default if not provided
    let aspect = if let Some((width, height)) = window_size {
        width as f32 / height as f32
    } else {
        16.0 / 9.0 // Default aspect ratio
    };

    world.spawn((
        Transform {
            position: Point3::new(0.0, 7.0, 0.0),
            ..Default::default()
        },
        Camera {
            aspect,
            ..Default::default()
        },
        FPSController::default(), // Add the FPS controller component
    ))
}

// Replace the update_world function with this FPS controller version
pub fn update_world(world: &mut World, input: &Input, dt: Duration) {
    // Skip if no controllers connected
    if input.connected_controllers_count() == 0 {
        return;
    }

    let controller_index = 0;

    // Use first controller if invalid index
    let controller_idx = if controller_index >= input.connected_controllers_count() {
        0
    } else {
        controller_index
    };

    let dt_seconds = dt.as_secs_f32();

    // Camera movement and look settings
    let move_speed = 5.0; // Units per second
    let look_speed = 1.5; // Radians per second
    let stick_deadzone = 0.15;

    // Query for camera entity with FPSController
    for (_, (transform, _, fps_controller)) in
        world.query_mut::<(&mut Transform, &Camera, &mut FPSController)>()
    {
        // Movement using left stick
        let (raw_move_x, raw_move_z) = input.left_stick_vector(controller_idx, stick_deadzone);

        // Calculate the magnitude of the move vector
        let move_magnitude = (raw_move_x * raw_move_x + raw_move_z * raw_move_z).sqrt();

        if move_magnitude > 0.0 {
            // Apply exponential response curve to the magnitude
            let scaled_magnitude = move_magnitude.powf(2.5);

            // Scale the original direction by the new magnitude
            let scale_factor = scaled_magnitude / move_magnitude;
            let move_x = raw_move_x * scale_factor;
            let move_z = raw_move_z * scale_factor;

            // Calculate forward and right vectors based on yaw only (for FPS-style movement)
            // This ensures movement is always in the XZ plane regardless of pitch
            let yaw_quat = Quaternion::from_axis_angle(Vector3::unit_y(), Rad(fps_controller.yaw));
            let forward = yaw_quat * -Vector3::unit_z(); // Forward vector in XZ plane
            let right = yaw_quat * Vector3::unit_x(); // Right vector in XZ plane

            // Apply movement based on stick input
            transform.position += forward * move_z * move_speed * dt_seconds;
            transform.position += right * move_x * move_speed * dt_seconds;
        }

        // Vertical movement using triggers
        let up_input = input.controller_trigger_value(controller_idx, false, 0.1); // Left trigger
        let down_input = input.controller_trigger_value(controller_idx, true, 0.1); // Right trigger
        transform.position.y += (up_input - down_input) * move_speed * dt_seconds;

        // Camera rotation using right stick
        let (raw_look_x, raw_look_y) = input.right_stick_vector(controller_idx, stick_deadzone);
        let raw_look_y = raw_look_y * -1.0;

        // Calculate the magnitude of the look vector
        let look_magnitude = (raw_look_x * raw_look_x + raw_look_y * raw_look_y).sqrt();

        if look_magnitude > 0.0 {
            // Apply exponential response curve to the magnitude
            let scaled_magnitude = look_magnitude.powf(2.5);

            // Scale the original direction by the new magnitude
            let scale_factor = scaled_magnitude / look_magnitude;
            let look_x = raw_look_x * scale_factor;
            let look_y = raw_look_y * scale_factor;

            // Update yaw (horizontal rotation) with pitch compensation
            let pitch_correction = 1.0 / fps_controller.pitch.cos().abs();
            info!("{}", pitch_correction);
            fps_controller.yaw -= look_x * look_speed * dt_seconds * pitch_correction;

            // Update pitch (vertical rotation) with constraints to prevent flipping
            fps_controller.pitch = (fps_controller.pitch - look_y * look_speed * dt_seconds).clamp(
                -std::f32::consts::FRAC_PI_2 + 0.1,
                std::f32::consts::FRAC_PI_2 - 0.1,
            );

            // Calculate the rotation quaternion from yaw and pitch
            let yaw_quat = Quaternion::from_axis_angle(Vector3::unit_y(), Rad(fps_controller.yaw));
            let pitch_quat =
                Quaternion::from_axis_angle(Vector3::unit_x(), Rad(fps_controller.pitch));

            // Combine the rotations: apply yaw, then pitch
            transform.rotation = yaw_quat * pitch_quat;

            // Normalize quaternion to prevent floating-point drift
            transform.rotation = transform.rotation.normalize();
        }
    }
}

// Update the view matrix calculation to ensure it uses the world up vector
pub fn calculate_view_matrix(transform: &Transform) -> Matrix4<f32> {
    let position = transform.position;
    let forward = transform.rotation * -Vector3::unit_z();

    // Always use world up for the camera's up vector
    let up = Vector3::unit_y();

    let target = position + forward;
    Matrix4::look_at_rh(position, target, up)
}

// pub fn calculate_view_matrix(transform: &Transform) -> Matrix4<f32> {
//     let position = transform.position;
//     let forward = transform.rotation * -Vector3::unit_z();
//     let up = transform.rotation * Vector3::unit_y();
//     let target = position + forward;

//     Matrix4::look_at_rh(position, target, up)
// }

pub fn calculate_view_projection(transform: &Transform, camera: &Camera) -> Matrix4<f32> {
    let view = calculate_view_matrix(transform);
    let proj = perspective(camera.fov, camera.aspect, camera.near, camera.far);
    proj * view
}

pub fn calculate_view(transform: &Transform) -> Matrix4<f32> {
    calculate_view_matrix(transform)
}

#[derive(Debug)]
pub struct Velocity {
    pub value: Vector3<f32>,
    pub forward_direction: Vector3<f32>,
}

impl Default for Velocity {
    fn default() -> Self {
        Self {
            value: Vector3::new(0.0, 0.0, 0.0),
            forward_direction: Vector3::new(0.0, 0.0, -1.0), // Forward is -Z
        }
    }
}
