use cgmath::{EuclideanSpace, Point3, SquareMatrix};
// app.rs
use hecs::World;
use winit::dpi::Size;
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
use crate::*;

#[derive(Default)]
pub struct App<'window> {
    window: Option<Arc<Window>>,
    wgpu_ctx: Option<WgpuCtx<'window>>,
    input_system: Input,
    world: World,
    camera_entity: Option<hecs::Entity>,
    last_frame_time: Option<Instant>,
}

impl<'window> ApplicationHandler for App<'window> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let win_attr = Window::default_attributes().with_title("ECS WGPU Example").with_inner_size(winit::dpi::PhysicalSize::new(800, 800)).with_min_inner_size(winit::dpi::PhysicalSize::new(200, 200));
            let window = Arc::new(event_loop.create_window(win_attr).unwrap());
            self.window = Some(window.clone());
            self.wgpu_ctx = Some(WgpuCtx::new(window.clone()));

            // Initialize ECS world
            self.world = World::new();

            // Get window size for initial aspect ratio
            let window_size = if let Some(window) = &self.window {
                let size = window.inner_size();
                Some((size.width, size.height))
            } else {
                None
            };

            // Setup camera with correct aspect ratio
            self.camera_entity = Some(crate::world::setup_camera_entity(
                &mut self.world,
                window_size,
            ));

            // if let Some(wgpu_ctx) = &mut self.wgpu_ctx {
            //     // Load a model
            //     if let Some(model_index) = wgpu_ctx.load_model("./assets/models/suzanne.gltf") {
            //         // Spawn a model entity
            //         crate::world::spawn_model_entity(
            //             &mut self.world,
            //             model_index,
            //             Point3::new(2.0, 0.0, 0.0), // Position to the right
            //         );
            //     }
            // }
        }

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
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
            // In app.rs, update the window_event handler for WindowEvent::Resized
            WindowEvent::Resized(new_size) => {
                if let (Some(wgpu_ctx), Some(window)) =
                    (self.wgpu_ctx.as_mut(), self.window.as_ref())
                {
                    wgpu_ctx.resize((new_size.width, new_size.height));

                    // Update camera aspect ratio
                    if let Some(camera_entity) = self.camera_entity {
                        if let Ok(camera) = self.world.query_one_mut::<&mut Camera>(camera_entity) {
                            camera.aspect = new_size.width as f32 / new_size.height as f32;
                        }
                    }

                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = self
                    .last_frame_time
                    .map(|t| now.duration_since(t))
                    .unwrap_or_default();
                self.last_frame_time = Some(now);

                // Update camera system
                update_camera_system(&mut self.world, &self.input_system, dt);

                if let (Some(wgpu_ctx), Some(camera_entity)) =
                    (&mut self.wgpu_ctx, self.camera_entity)
                {
                    if let Ok((transform, camera)) = self
                        .world
                        .query_one_mut::<(&Transform, &Camera)>(camera_entity)
                    {
                        let view_proj = calculate_view_projection(transform, camera);
                        let inv_view_proj = view_proj.invert().unwrap();
                        let view = calculate_view_matrix(transform);
                        wgpu_ctx.update_camera_uniform(view_proj,  inv_view_proj, view, transform.position.into());
                    }
                }

                if let Some(wgpu_ctx) = &mut self.wgpu_ctx {
                    wgpu_ctx.draw(&self.world);
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
