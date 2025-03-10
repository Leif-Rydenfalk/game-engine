use cgmath::{EuclideanSpace, Point3, SquareMatrix};
// app.rs
use hecs::World;
use winit::event::Event;
use winit::keyboard::Key;
use winit::keyboard::NamedKey;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::Size;
use winit::event::MouseScrollDelta::*;
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
            let win_attr = Window::default_attributes()
                .with_title("Voxel Renderer")
                .with_inner_size(winit::dpi::PhysicalSize::new(800, 800))
                .with_min_inner_size(winit::dpi::PhysicalSize::new(200, 200));
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

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match event.clone() {
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
            WindowEvent::KeyboardInput { event, .. } => {
                let imgui = &mut self.wgpu_ctx.as_mut().unwrap().imgui;
                let io = imgui.context.io();
                
                // Only process keyboard input if ImGui isn't capturing it
                if !io.want_capture_keyboard {
                    if let Key::Named(NamedKey::Escape) = event.logical_key {
                        if event.state.is_pressed() {
                            event_loop.exit();
                        }
                    }
            
                    if let PhysicalKey::Code(key) = event.physical_key {
                        self.input_system.handle_key_input(key, event.state);
                    }
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
                        wgpu_ctx.update_camera_uniform(
                            view_proj,
                            inv_view_proj,
                            view,
                            transform.position.into(),
                        );
                    }
                }

                if let Some(wgpu_ctx) = &mut self.wgpu_ctx {
                    wgpu_ctx.draw(&self.world, self.window.as_mut().unwrap());
                }

                self.input_system.update();
            }
            WindowEvent::MouseInput { button, state, .. } => {
                let imgui = &mut self.wgpu_ctx.as_mut().unwrap().imgui;
                let io = imgui.context.io();
                
                if !io.want_capture_mouse {
                    self.input_system.handle_mouse_button(button, state);
                }
            }
            
            WindowEvent::CursorMoved { position, .. } => {
                let imgui = &mut self.wgpu_ctx.as_mut().unwrap().imgui;
                let io = imgui.context.io();
                
                if !io.want_capture_mouse {
                    self.input_system.handle_cursor_moved(&position);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                LineDelta(_, y) => {
                    self.input_system.handle_mouse_scroll(y as f64);
                }
                PixelDelta(d) => {
                    self.input_system.handle_mouse_scroll(d.y);
                }
            },
            _ => (),
        }

        let window = self.window.as_mut().unwrap();
        let imgui = &mut self.wgpu_ctx.as_mut().unwrap().imgui;
        imgui.platform.handle_event::<()>(
            imgui.context.io_mut(),
            &window,
            &Event::WindowEvent { window_id, event },
        );
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.input_system.handle_mouse_motion((delta.0, delta.1));
        }

        let window = self.window.as_mut().unwrap();
        let imgui = &mut self.wgpu_ctx.as_mut().unwrap().imgui;
        imgui.platform.handle_event::<()>(
            imgui.context.io_mut(),
            &window,
            &Event::DeviceEvent { device_id, event },
        );
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: ()) {
        let window = self.window.as_mut().unwrap();
        let imgui = &mut self.wgpu_ctx.as_mut().unwrap().imgui;
        imgui.platform.handle_event::<()>(
            imgui.context.io_mut(),
            &window,
            &Event::UserEvent(event),
        );
    }


    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let window = self.window.as_mut().unwrap();
        let imgui = &mut self.wgpu_ctx.as_mut().unwrap().imgui;
        window.request_redraw();
        imgui.platform.handle_event::<()>(
            imgui.context.io_mut(),
            &window,
            &Event::AboutToWait,
        );
    }
}
