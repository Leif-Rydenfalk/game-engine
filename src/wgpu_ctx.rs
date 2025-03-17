use crate::vertex::{create_vertex_buffer_layout, INDICES_SQUARE, VERTICES_SQUARE};
use crate::{
    BloomEffect, Camera, CameraController, ColorCorrectionEffect, ColorCorrectionUniform, ImguiState, Model, ModelInstance, RgbaImg, Transform, VoxelRenderer
};
use cgmath::{Matrix4, Point3, SquareMatrix};
use hecs::World;
use std::borrow::Cow;
use std::{path::Path, sync::Arc, time::Instant};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{MemoryHints, SamplerDescriptor, ShaderSource};
use winit::window::Window;

use imgui::*;
use imgui_wgpu::{Renderer, RendererConfig};
use imgui_winit_support::WinitPlatform;

// --- CameraUniform Struct ---
#[repr(C)]
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    position: [f32; 3],
    time: f32,
    resolution: [f32; 2],
    _padding: [f32; 2],
}

// --- WgpuCtx Struct ---
pub struct WgpuCtx<'window> {
    surface: wgpu::Surface<'window>,
    surface_config: wgpu::SurfaceConfiguration,
    adapter: wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    camera_buffer: wgpu::Buffer,
    camera_bind_group_layout: wgpu::BindGroupLayout,
    camera_bind_group: wgpu::BindGroup,
    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,
    models: Vec<Model>,
    render_texture_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    render_texture: wgpu::Texture,
    render_texture_view: wgpu::TextureView,
    sampler: Arc<wgpu::Sampler>,
    bloom_effect: BloomEffect,
    post_process_texture: wgpu::Texture,
    post_process_texture_view: wgpu::TextureView,
    color_correction_effect: ColorCorrectionEffect,
    final_float_texture: wgpu::Texture,           // Rgba32Float for post-processing output
    final_float_texture_view: wgpu::TextureView,
    vertex_buffer: wgpu::Buffer,                 // For full-screen quad
    index_buffer: wgpu::Buffer,
    final_texture_bind_group: wgpu::BindGroup,   // Bind group for final_float_texture_view
    final_pipeline: wgpu::RenderPipeline,        // Pipeline to render to surface
    time: Instant,
    hidpi_factor: f64,
    pub imgui: ImguiState,
    voxel_renderer: VoxelRenderer,
}

impl<'window> WgpuCtx<'window> {
    /// Creates a depth texture for depth testing
    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        (depth_texture, depth_texture_view)
    }

    /// Asynchronous constructor
    pub async fn new_async(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::FLOAT32_FILTERABLE,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);
        let mut surface_config = surface.get_default_config(&adapter, width, height).unwrap();
        surface_config.present_mode = wgpu::PresentMode::Fifo;
        surface.configure(&device, &surface_config);

        // Sampler for render texture
        let sampler = Arc::new(device.create_sampler(&SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
    
        // Camera setup (unchanged)
        let camera_uniform = CameraUniform {
            view_proj: Matrix4::identity().into(),
            ..Default::default()
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });
    
        // VoxelRenderer, depth texture, render_texture_bind_group_layout (unchanged)
        let voxel_renderer = VoxelRenderer::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            &camera_bind_group_layout,
        );
        let (depth_texture, depth_texture_view) = Self::create_depth_texture(&device, &surface_config);
        let render_texture_bind_group_layout = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            },
        ));
    
        // Sampler, render_texture, bloom_effect, post_process_texture (unchanged)
        let sampler = Arc::new(device.create_sampler(&SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
        let render_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Texture"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let render_texture_view = render_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("bloom.wgsl"))),
        });
        let bloom_effect = BloomEffect::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            Arc::clone(&render_texture_bind_group_layout),
            Arc::clone(&sampler),
            surface_config.width,
            surface_config.height,
            &render_texture_view,
            &bloom_shader,
        );
        let post_process_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Post Process Texture"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let post_process_texture_view = post_process_texture.create_view(&wgpu::TextureViewDescriptor::default());
    
        // Color correction effect (unchanged)
        let color_correction_effect = ColorCorrectionEffect::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            &post_process_texture_view,
            Arc::clone(&sampler),
        );
    
        // Final float texture (simplified usage)
        let final_float_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Final Float Texture"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let final_float_texture_view = final_float_texture.create_view(&wgpu::TextureViewDescriptor::default());
    
        // Vertex and index buffers for full-screen quad
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES_SQUARE),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES_SQUARE),
            usage: wgpu::BufferUsages::INDEX,
        });
    
        // Final texture bind group
        let final_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &render_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&final_float_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("final_texture_bind_group"),
        });
    
        // Final render pipeline
        let final_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Final Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("final.wgsl"))),
        });
        let final_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Final Pipeline Layout"),
            bind_group_layouts: &[&render_texture_bind_group_layout],
            push_constant_ranges: &[],
        });
        let final_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Final Pipeline"),
            layout: Some(&final_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &final_shader,
                entry_point: Some("vs_main"),
                buffers: &[create_vertex_buffer_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &final_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(surface_config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
    
        // ImGui setup (unchanged)
        let hidpi_factor = window.scale_factor();
        let imgui = {
            let mut context = imgui::Context::create();
            let mut platform = WinitPlatform::new(&mut context);
            platform.attach_window(context.io_mut(), &window, imgui_winit_support::HiDpiMode::Default);
            context.set_ini_filename(None);
            let font_size = (13.0 * hidpi_factor) as f32;
            context.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
            context.fonts().add_font(&[FontSource::DefaultFontData {
                config: Some(imgui::FontConfig {
                    oversample_h: 1,
                    pixel_snap_h: true,
                    size_pixels: font_size,
                    ..Default::default()
                }),
            }]);
            let clear_color = wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 };
            let renderer_config = RendererConfig {
                texture_format: surface_config.format,
                ..Default::default()
            };
            let renderer = Renderer::new(&mut context, &device, &queue, renderer_config);
            ImguiState {
                context,
                platform,
                renderer,
                clear_color,
                demo_open: true,
                last_frame: Instant::now(),
                last_cursor: None,
            }
        };
    
        Self {
            surface,
            surface_config,
            adapter,
            device,
            queue,
            camera_buffer,
            camera_bind_group_layout,
            camera_bind_group,
            depth_texture,
            depth_texture_view,
            models: Vec::new(),
            render_texture_bind_group_layout,
            render_texture,
            render_texture_view,
            sampler,
            bloom_effect,
            post_process_texture,
            post_process_texture_view,
            color_correction_effect,
            final_float_texture,
            final_float_texture_view,
            vertex_buffer,
            index_buffer,
            final_texture_bind_group,
            final_pipeline,
            time: Instant::now(),
            hidpi_factor,
            imgui,
            voxel_renderer,
        }
    }

    /// Synchronous constructor
    pub fn new(window: Arc<Window>) -> Self {
        pollster::block_on(Self::new_async(window))
    }

    /// Updates the camera uniform buffer
    pub fn update_camera_uniform(
        &mut self,
        view_proj: Matrix4<f32>,
        inv_view_proj: Matrix4<f32>,
        view: Matrix4<f32>,
        position: [f32; 3],
    ) {
        let camera_uniform = CameraUniform {
            view_proj: view_proj.into(),
            inv_view_proj: inv_view_proj.into(),
            view: view.into(),
            position,
            time: self.time.elapsed().as_secs_f32(),
            resolution: [self.surface_config.width as f32, self.surface_config.height as f32],
            _padding: [0.0; 2],
        };
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );
    }

    /// Resizes rendering surfaces
    pub fn resize(&mut self, new_size: (u32, u32)) {
        let (width, height) = new_size;
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);
    
        let (depth_texture, depth_texture_view) = Self::create_depth_texture(&self.device, &self.surface_config);
        self.depth_texture = depth_texture;
        self.depth_texture_view = depth_texture_view;
    
        self.render_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Texture"),
            size: wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        self.render_texture_view = self.render_texture.create_view(&wgpu::TextureViewDescriptor::default());
    
        self.post_process_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Post Process Texture"),
            size: wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        self.post_process_texture_view = self.post_process_texture.create_view(&wgpu::TextureViewDescriptor::default());
    
        self.final_float_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Final Float Texture"),
            size: wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        self.final_float_texture_view = self.final_float_texture.create_view(&wgpu::TextureViewDescriptor::default());
    
        // Update final_texture_bind_group
        self.final_texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.render_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.final_float_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
            label: Some("final_texture_bind_group"),
        });
    
        self.bloom_effect.resize(self.surface_config.width, self.surface_config.height, &self.render_texture_view);
        self.color_correction_effect.resize(&self.post_process_texture_view);
    }

    /// Renders the scene
    pub fn draw(&mut self, world: &mut World, window: &Window) {
        let surface_texture = self.surface.get_current_texture().expect("Failed to acquire next swap chain texture");
        let surface_texture_view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    
        // Render voxel scene to render_texture_view (Rgba32Float)
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Scene Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.render_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_bind_group(0, &self.camera_bind_group, &[]);
            self.voxel_renderer.render(&mut rpass);
        }
    
        // Post-processing
        self.bloom_effect.render(&mut encoder, &self.render_texture_view);
        self.bloom_effect.apply(&mut encoder, &self.post_process_texture_view, &self.render_texture_view);
        self.color_correction_effect.update_uniform(ColorCorrectionUniform {
            brightness: 1.0,
            contrast: 1.0,
            saturation: 1.0,
        });
        self.color_correction_effect.apply(&mut encoder, &self.final_float_texture_view);
    
        // Final render pass to surface texture
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Final Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.final_pipeline);
            rpass.set_bind_group(0, &self.final_texture_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..INDICES_SQUARE.len() as u32, 0, 0..1);
        }
    
        // ImGui rendering (unchanged)
        let now = Instant::now();
        self.imgui.context.io_mut().update_delta_time(now - self.imgui.last_frame);
        self.imgui.last_frame = now;
        self.imgui.platform.prepare_frame(self.imgui.context.io_mut(), window).expect("Failed to prepare ImGui frame");
        let ui = self.imgui.context.frame();
    
        // ImGui UI (unchanged)
        let mut modified = false;
        ui.window("Settings")
            .size([300.0, 200.0], Condition::FirstUseEver)
            .always_auto_resize(true)
            .build(|| {
                if ui.button("Set Camera Position Origin") {
                    for (_, (transform, _, _)) in world.query_mut::<(&mut Transform, &mut Camera, &mut CameraController)>() {
                        transform.position = Point3::new(6.0, 2.2, 6.0);
                    }
                }
                for (_, (transform, _, _)) in world.query_mut::<(&mut Transform, &mut Camera, &mut CameraController)>() {
                    let mut pos: [f32; 3] = transform.position.into();
                    if ui.input_float3("Camera Transform", &mut pos).build() {
                        transform.position = pos.into();
                    }
                }
                if ui.slider("Voxel Level", 1, 8, &mut self.voxel_renderer.voxel_settings.voxel_level) {
                    self.voxel_renderer.voxel_settings.update_voxel_size();
                    modified = true;
                }
                if ui.input_float("Max Ray Distance", &mut self.voxel_renderer.voxel_settings.max_dist).build() {
                    modified = true;
                }
                if ui.input_float("Water Height", &mut self.voxel_renderer.voxel_settings.water_height).build() {
                    self.voxel_renderer.voxel_settings.max_water_height = self.voxel_renderer.voxel_settings.water_height;
                    modified = true;
                }
                if ui.input_float("Max Terrain Height", &mut self.voxel_renderer.voxel_settings.max_height).build() {
                    modified = true;
                }
                if ui.slider("Sun Incline", 0.0, 1.0, &mut self.voxel_renderer.sun_incline) {
                    self.voxel_renderer.voxel_settings.light_direction = calculate_sun_direction(
                        self.voxel_renderer.time_of_day,
                        self.voxel_renderer.sun_incline,
                    );
                    modified = true;
                }
                if ui.slider("Time of Day", 0.0, 24.0, &mut self.voxel_renderer.time_of_day) {
                    self.voxel_renderer.voxel_settings.light_direction = calculate_sun_direction(
                        self.voxel_renderer.time_of_day,
                        self.voxel_renderer.sun_incline,
                    );
                    modified = true;
                }
            });
    
        if modified {
            self.voxel_renderer.update_settings_buffer();
        }
    
        if self.imgui.last_cursor != ui.mouse_cursor() {
            self.imgui.last_cursor = ui.mouse_cursor();
            self.imgui.platform.prepare_render(ui, window);
        }
    
        self.imgui.renderer.render(
            self.imgui.context.render(),
            &self.queue,
            &self.device,
            &mut encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ImGui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            }),
        ).expect("ImGui rendering failed");
    
        self.queue.submit(Some(encoder.finish()));
        surface_texture.present();
    }
}

// --- Helper Functions ---
pub fn create_pipeline(
    device: &wgpu::Device,
    swap_chain_format: wgpu::TextureFormat,
    pipeline_layout: &wgpu::PipelineLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("voxels.wgsl"))),
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[create_vertex_buffer_layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(swap_chain_format.into())],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn calculate_sun_direction(time_of_day: f32, sun_incline: f32) -> [f32; 4] {
    let time_angle = (time_of_day / 24.0) * 2.0 * std::f32::consts::PI;
    let x = time_angle.cos();
    let z = time_angle.sin();
    let incline_angle = sun_incline * std::f32::consts::FRAC_PI_2;
    let y = incline_angle.sin();
    let horizontal_scale = incline_angle.cos();
    let x_adjusted = x * horizontal_scale;
    let z_adjusted = z * horizontal_scale;
    [x_adjusted, -z_adjusted, y, 0.0]
}