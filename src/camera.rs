use cgmath::{perspective, InnerSpace, Matrix4, Point3, Rad, Vector3, Zero};

pub struct Camera {
    pub position: Point3<f32>,
    yaw: Rad<f32>,   // Rotation around Y axis
    pitch: Rad<f32>, // Rotation around X axis
}

impl Default for Camera {
    fn default() -> Self {
        Camera::new(Point3::new(0.0, 0.0, 5.0)) // Start a bit back from the origin
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

        // Calculate the camera's forward vector
        let forward = Vector3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize();

        // Create view matrix
        let view = Matrix4::look_to_rh(self.position, forward, Vector3::unit_y());

        // Create perspective projection matrix
        let projection = perspective(
            Rad(std::f32::consts::FRAC_PI_4), // 45 degrees FOV
            16.0 / 9.0,                       // Aspect ratio (adjust based on your window size)
            0.1,                              // Near plane (very close)
            100.0,                            // Far plane (very far)
        );

        // Combine projection and view matrices
        projection * view
    }

    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += Rad(delta_yaw);
        self.pitch += Rad(-delta_pitch); // Invert pitch for intuitive mouse control

        // Clamp pitch to prevent camera flipping
        self.pitch.0 = self.pitch.0.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.1,
            std::f32::consts::FRAC_PI_2 - 0.1,
        );
    }

    // Add these helper methods to get camera vectors
    pub fn forward_vector(&self) -> Vector3<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();
        Vector3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize()
    }

    pub fn right_vector(&self) -> Vector3<f32> {
        self.forward_vector().cross(Vector3::unit_y()).normalize()
    }

    pub fn up_vector(&self) -> Vector3<f32> {
        self.right_vector().cross(self.forward_vector()).normalize()
    }
}

pub struct CameraController {
    speed: f32,
    sensitivity: f32,
}

impl Default for CameraController {
    fn default() -> Self {
        CameraController::new(5.0, 0.003) // Reduced sensitivity for smoother control
    }
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self { speed, sensitivity }
    }

    pub fn update_camera(&self, input: &crate::input::Input, camera: &mut Camera, dt: f32) {
        // Handle mouse movement for rotation
        let mouse_delta = input.mouse_delta();
        if input.is_mouse_button_down(winit::event::MouseButton::Left) {
            camera.rotate(
                mouse_delta.0 as f32 * self.sensitivity,
                mouse_delta.1 as f32 * self.sensitivity,
            );
        }

        // Get camera vectors for movement
        let forward = camera.forward_vector();
        let right = camera.right_vector();
        let up = camera.up_vector();

        // Calculate movement direction based on input
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

        // Apply movement if any keys are pressed
        if movement != Vector3::zero() {
            let movement = movement.normalize() * self.speed * dt;
            camera.position += movement;
        }
    }
}
