// app.rs
use cgmath::Point3;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event::{DeviceEvent, DeviceId};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};

use crate::input::Input;
use crate::wgpu_ctx::WgpuCtx;
use crate::{Camera, CameraController};

#[derive(Default)]
pub struct App<'window> {
    window: Option<Arc<Window>>,
    wgpu_ctx: Option<WgpuCtx<'window>>,
    input_system: Input,
    camera: Camera,
    camera_controller: CameraController,
    last_frame_time: Option<Instant>,
}

impl<'window> ApplicationHandler for App<'window> {
    // In app.rs, update the resumed method:
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let win_attr = Window::default_attributes().with_title("wgpu winit example");
            let window = Arc::new(event_loop.create_window(win_attr).unwrap());
            self.window = Some(window.clone());
            self.wgpu_ctx = Some(WgpuCtx::new(window));

            // Position the camera to see the cube
            self.camera = Camera::new(Point3::new(0.0, 1.0, 3.0));
            self.camera_controller = CameraController::new(2.0, 0.003);
        }

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if let (Some(wgpu_ctx), Some(window)) =
                    (self.wgpu_ctx.as_mut(), self.window.as_ref())
                {
                    wgpu_ctx.resize((new_size.width, new_size.height));
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                // Update camera
                let now = Instant::now();
                let dt = self
                    .last_frame_time
                    .map_or(0.0, |t| now.duration_since(t).as_secs_f32());
                self.last_frame_time = Some(now);

                self.camera_controller
                    .update_camera(&self.input_system, &mut self.camera, dt);

                if let Some(wgpu_ctx) = self.wgpu_ctx.as_mut() {
                    wgpu_ctx.update_camera_uniform(self.camera.calc_matrix());
                    wgpu_ctx.draw();
                }
                self.input_system.update();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    self.input_system.handle_key_input(key, event.state);
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                self.input_system.handle_mouse_button(button, state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input_system.handle_cursor_moved(&position);
            }
            _ => (),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.input_system.handle_mouse_motion((delta.0, delta.1));
        }
    }
}
