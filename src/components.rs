use cgmath::{Point3, Rad, Vector3};

#[derive(Debug)]
pub struct Transform {
    pub position: Point3<f32>,
    pub rotation: (Rad<f32>, Rad<f32>), // (pitch, yaw)
    pub scale: Vector3<f32>,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 0.0),
            rotation: (Rad(0.0), Rad(0.0)),
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
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fov: Rad(std::f32::consts::FRAC_PI_4),
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

#[derive(Debug)]
pub struct CameraController {
    pub move_speed: f32,
    pub look_speed: f32,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            move_speed: 5.0,
            look_speed: 0.003,
        }
    }
}

#[derive(Debug)]
pub struct Player {
    pub health: f32,
    pub stamina: f32,
}

impl Default for Player {
    fn default() -> Self {
        Self {
            health: 100.0,
            stamina: 100.0,
        }
    }
}

#[derive(Debug)]
pub struct ModelInstance {
    pub model: usize, // Index into the model registry
}
