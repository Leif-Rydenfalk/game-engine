use cgmath::{InnerSpace, Matrix4, Point3, Rad, Vector3, Zero};

pub struct Camera {
    pub position: Point3<f32>,
    yaw: Rad<f32>,
    pitch: Rad<f32>,
}

impl Default for Camera {
    fn default() -> Self {
        Camera::new(Point3::new(10.0, 0.0, 0.0))
    }
}

impl Camera {
    pub fn new(position: Point3<f32>) -> Self {
        Self {
            position,
            yaw: Rad(0.0),
            pitch: Rad(0.0),
        }
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();

        Matrix4::look_to_rh(
            self.position,
            Vector3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize(),
            Vector3::unit_y(),
        )
    }

    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += Rad(delta_yaw);
        self.pitch += Rad(-delta_pitch);
        self.pitch.0 = self.pitch.0.clamp(-1.57, 1.57); // Clamp between -π/2 and π/2
    }
}

pub struct CameraController {
    speed: f32,
    sensitivity: f32,
}

impl Default for CameraController {
    fn default() -> Self {
        CameraController::new(10.0, 10.0)
    }
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self { speed, sensitivity }
    }

    pub fn update_camera(&self, input: &crate::input::Input, camera: &mut Camera, dt: f32) {
        // Keyboard movement
        let forward = Vector3::new(camera.yaw.0.cos(), 0.0, camera.yaw.0.sin());
        let right = Vector3::new(-forward.z, 0.0, forward.x);

        let mut move_dir = Vector3::zero();
        if input.is_key_down(winit::keyboard::KeyCode::KeyW) {
            move_dir += forward;
        }
        if input.is_key_down(winit::keyboard::KeyCode::KeyS) {
            move_dir -= forward;
        }
        if input.is_key_down(winit::keyboard::KeyCode::KeyA) {
            move_dir -= right;
        }
        if input.is_key_down(winit::keyboard::KeyCode::KeyD) {
            move_dir += right;
        }
        if input.is_key_down(winit::keyboard::KeyCode::Space) {
            move_dir.y += 1.0;
        }
        if input.is_key_down(winit::keyboard::KeyCode::ShiftLeft) {
            move_dir.y -= 1.0;
        }

        if move_dir != Vector3::zero() {
            camera.position += move_dir.normalize() * self.speed * dt;
        }

        // Mouse rotation
        let mouse_delta = input.mouse_delta();
        camera.rotate(
            mouse_delta.0 as f32 * self.sensitivity,
            mouse_delta.1 as f32 * self.sensitivity,
        );
    }
}
