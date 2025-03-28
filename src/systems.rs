use crate::*;
use cgmath::One;
use cgmath::Rotation3;
use cgmath::{perspective, InnerSpace, Matrix4, Point3, Quaternion, Rad, Vector3, Zero};
use hecs::World;
use std::time::Duration;

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
    pub up_vector: Vector3<f32>,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fov: Rad(std::f32::consts::FRAC_PI_4),
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 1000.0,
            up_vector: Vector3::unit_y(),
        }
    }
}

#[derive(Debug)]
pub struct CameraController {
    pub move_speed: f32,
    pub look_speed: f32,
    pub pitch: Rad<f32>,
    pub yaw: Rad<f32>,
    pub pitch_limit: Rad<f32>, // Limit vertical camera rotation
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            move_speed: 5.0,
            look_speed: 0.003,
            pitch: Rad(0.0),
            yaw: Rad(0.0),
            pitch_limit: Rad(std::f32::consts::FRAC_PI_2 - 0.1), // Prevent looking directly up/down
        }
    }
}

pub fn update_camera_system(world: &mut World, input: &Input, dt: Duration) {
    for (_, (transform, camera, controller)) in
        world.query_mut::<(&mut Transform, &mut Camera, &mut CameraController)>()
    {
        let dt = dt.as_secs_f32();

        // Update move speed multiplier with scroll - logarithmic scaling for better space navigation
        if input.scroll_delta() != 0.0 {
            let scroll_factor = if input.scroll_delta() > 0.0 { 1.2 } else { 0.8 };
            controller.move_speed *= scroll_factor;

            // Clamp the speed multiplier to reasonable values for space navigation
            controller.move_speed = controller.move_speed.clamp(0.01, 1000.0);
        }

        // Handle mouse look
        if input.is_mouse_button_down(winit::event::MouseButton::Left) {
            let mouse_delta = input.mouse_delta();

            // Update yaw and pitch based on mouse movement
            controller.yaw -= Rad(mouse_delta.0 as f32 * controller.look_speed);
            controller.pitch -= Rad(mouse_delta.1 as f32 * controller.look_speed);

            // Clamp pitch to prevent camera flipping
            controller.pitch = Rad(controller
                .pitch
                .0
                .max(-controller.pitch_limit.0)
                .min(controller.pitch_limit.0));
        }

        // Recreate rotation based on yaw and pitch
        transform.rotation = Quaternion::from_axis_angle(Vector3::unit_y(), controller.yaw)
            * Quaternion::from_axis_angle(Vector3::unit_x(), controller.pitch);

        // Calculate camera-relative movement vectors
        let forward = transform.rotation * -Vector3::unit_z();
        let right = transform.rotation * Vector3::unit_x();
        let world_up = Vector3::unit_y();

        // Movement input handling
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

        // Vertical movement (less common in FPS, but useful for spectator mode)
        if input.is_key_down(winit::keyboard::KeyCode::Space) {
            movement_input += world_up;
        }
        if input.is_key_down(winit::keyboard::KeyCode::ShiftLeft) {
            movement_input -= world_up;
        }

        // Normalize movement to prevent faster diagonal movement
        if movement_input != Vector3::zero() {
            let movement = movement_input.normalize() * controller.move_speed * dt;
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

#[derive(Debug)]
pub struct ModelInstance {
    pub model: usize, // Index into the model registry
}
