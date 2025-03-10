use cgmath::{InnerSpace, Matrix3, Point3, Quaternion, Rad, Rotation3, SquareMatrix, Vector3};
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
            far: 100.0,
            up_vector: Vector3::unit_y(),
        }
    }
}

#[derive(Debug)]
pub struct CameraController {
    pub move_speed: f32,
    pub move_speed_mult: f32,
    pub look_speed: f32,
    pub pitch: Rad<f32>,
    pub yaw: Rad<f32>,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            move_speed: 5.0,
            move_speed_mult: 1.0,
            look_speed: 0.003,
            pitch: Rad(0.0),
            yaw: Rad(0.0),
        }
    }
}

#[derive(Debug)]
pub struct ModelInstance {
    pub model: usize, // Index into the model registry
}
