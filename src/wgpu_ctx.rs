use crate::{
    AtmosphereRenderer, BloomEffect, Camera, ColorCorrectionEffect, ColorCorrectionUniform,
    ImguiState, Model, Settings, Transform, VoxelRenderer,
};
use cgmath::{Matrix4, Point3, SquareMatrix};
use hecs::World;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{MemoryHints, PipelineCompilationOptions, SamplerDescriptor, ShaderSource};
use winit::window::Window;

use imgui::*;
use imgui_wgpu::{Renderer, RendererConfig};
use imgui_winit_support::WinitPlatform;

use tracing::{debug, error, info, trace, warn};

/// Enum for selecting the active rendering mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderMode {
    Voxel,
    Atmosphere,
}

// --- ShaderHotReload implementation ---
pub struct ShaderHotReload {
    device: Arc<wgpu::Device>,
    shader_dir: PathBuf,
    modules: Mutex<HashMap<String, (wgpu::ShaderModule, SystemTime)>>,
    last_check: Mutex<SystemTime>,
}

impl ShaderHotReload {
    pub fn new(device: Arc<wgpu::Device>, shader_dir: impl AsRef<Path>) -> Self {
        Self {
            device,
            shader_dir: shader_dir.as_ref().to_path_buf(),
            modules: Mutex::new(HashMap::new()),
            last_check: Mutex::new(SystemTime::now()),
        }
    }

    pub fn get_shader(&self, name: &str) -> wgpu::ShaderModule {
        let mut modules = self.modules.lock().unwrap();
        let path = self.shader_dir.join(name);

        // Try to get modification time, if file doesn't exist or has other issues,
        // fallback to embedded shader
        let metadata = match fs::metadata(&path) {
            Ok(meta) => meta,
            Err(e) => {
                info!(
                    "Warning: Couldn't read shader file {}: {}",
                    path.display(),
                    e
                );
                // Return embedded fallback shader
                return self.create_fallback_shader(name);
            }
        };

        let modified = match metadata.modified() {
            Ok(time) => time,
            Err(e) => {
                info!(
                    "Warning: Couldn't get modification time for {}: {}",
                    path.display(),
                    e
                );
                // Return embedded fallback shader
                return self.create_fallback_shader(name);
            }
        };

        if let Some((module, last_modified)) = modules.get(name) {
            if *last_modified >= modified {
                return module.clone();
            }
        }

        // Load shader from file
        let source = match fs::read_to_string(&path) {
            Ok(content) => content,
            Err(e) => {
                info!(
                    "Warning: Failed to read shader file {}: {}",
                    path.display(),
                    e
                );
                // Return embedded fallback shader
                return self.create_fallback_shader(name);
            }
        };

        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(source)),
            });

        modules.insert(name.to_string(), (shader_module.clone(), modified));
        info!("Loaded shader: {}", name);

        shader_module
    }

    fn create_fallback_shader(&self, name: &str) -> wgpu::ShaderModule {
        // For simplicity, we're using empty shaders as fallbacks
        // In a real implementation, you would embed the shaders here
        let fallback_source = match name {
            "bloom.wgsl" => include_str!("shaders/bloom.wgsl"),
            "color_correction.wgsl" => include_str!("shaders/color_correction.wgsl"),
            "final_conversion.wgsl" => include_str!("shaders/final_conversion.wgsl"),
            "atmosphere.wgsl" => include_str!("shaders/atmosphere.wgsl"),
            "voxel.wgsl" => include_str!("shaders/voxel.wgsl"),
            _ => {
                info!("Unknown shader: {}, using empty fallback", name);
                "// Empty fallback shader\n@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }\n@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 1.0, 1.0); }"
            }
        };

        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{}_fallback", name)),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(fallback_source)),
            })
    }

    pub fn check_for_updates(&self) -> Vec<String> {
        let mut last_check = self.last_check.lock().unwrap();
        let now = SystemTime::now();
        let mut updated_shaders = Vec::new();

        // Only check every 1 second to avoid excessive file reads
        if now
            .duration_since(*last_check)
            .unwrap_or(Duration::from_secs(0))
            < Duration::from_secs(1)
        {
            return updated_shaders;
        }

        *last_check = now;
        let modules = self.modules.lock().unwrap();

        for (name, (_, last_modified)) in modules.iter() {
            let path = self.shader_dir.join(name);
            if let Ok(metadata) = fs::metadata(&path) {
                if let Ok(modified) = metadata.modified() {
                    if modified > *last_modified {
                        updated_shaders.push(name.clone());
                    }
                }
            }
        }

        updated_shaders
    }
}

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

    // Shader hot reloading
    shader_hot_reload: Arc<ShaderHotReload>,

    // Camera resources
    camera_buffer: wgpu::Buffer,
    camera_bind_group_layout: wgpu::BindGroupLayout,
    camera_bind_group: wgpu::BindGroup,

    // Depth resources
    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,

    // Rendering resources
    models: Vec<Model>,
    render_texture_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    render_texture: wgpu::Texture,
    render_texture_view: wgpu::TextureView,
    sampler: Arc<wgpu::Sampler>,

    // Post-processing resources
    bloom_effect: BloomEffect,
    post_process_texture: wgpu::Texture,
    post_process_texture_view: wgpu::TextureView,
    color_correction_effect: ColorCorrectionEffect,

    // Final rendering resources (simplified conversion)
    final_render_pipeline: wgpu::RenderPipeline,
    final_vertex_buffer: wgpu::Buffer,
    final_texture_bind_group: wgpu::BindGroup,

    // State tracking
    time: Instant,
    hidpi_factor: f64,

    // Renderers and UI
    pub imgui: ImguiState,
    pub render_mode: RenderMode,
    voxel_renderer: VoxelRenderer,
    atmosphere_renderer: AtmosphereRenderer,
    atmosphere_output_bind_group: wgpu::BindGroup,
}

impl<'window> WgpuCtx<'window> {
    /// Synchronous constructor wrapper
    pub fn new(window: Arc<Window>) -> Self {
        pollster::block_on(Self::new_async(window))
    }

    /// Asynchronous constructor
    pub async fn new_async(window: Arc<Window>) -> Self {
        // Initialize WGPU core resources
        let (surface, surface_config, adapter, device, queue) =
            Self::init_wgpu_core(Arc::clone(&window)).await;

        // Create shader hot reloading manager
        let shader_hot_reload = Arc::new(ShaderHotReload::new(
            Arc::clone(&device),
            std::path::Path::new("src/shaders"), // Change this to your shader directory
        ));

        // Create camera resources
        let (camera_buffer, camera_bind_group_layout, camera_bind_group) =
            Self::create_camera_resources(&device);

        // Create depth resources
        let (depth_texture, depth_texture_view) =
            Self::create_depth_texture(&device, &surface_config);

        // Create render texture resources
        let render_texture_bind_group_layout = Self::create_texture_bind_group_layout(&device);
        let (render_texture, render_texture_view) = Self::create_color_texture(
            &device,
            &surface_config,
            "Render Texture",
            wgpu::TextureFormat::Rgba32Float,
        );

        // Create sampler
        let sampler = Arc::new(device.create_sampler(&SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        // Create post-processing resources - use shader hot reload
        let bloom_shader = shader_hot_reload.get_shader("bloom.wgsl");

        // Create post-processing textures (both are now Rgba32Float)
        let (post_process_texture, post_process_texture_view) = Self::create_color_texture(
            &device,
            &surface_config,
            "Post Process Texture",
            wgpu::TextureFormat::Rgba32Float,
        );

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

        let color_correction_shader = shader_hot_reload.get_shader("color_correction.wgsl");
        let color_correction_effect = ColorCorrectionEffect::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            &post_process_texture_view,
            Arc::clone(&sampler),
            Some(&color_correction_shader),
        );

        // Create final texture bind group
        let final_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &render_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&render_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("final_texture_bind_group"),
        });

        // Create final render pipeline and vertex buffer for direct conversion - use shader hot reload
        let final_shader = shader_hot_reload.get_shader("final_conversion.wgsl");
        let (final_render_pipeline, final_vertex_buffer) =
            Self::create_final_render_pipeline_with_shader(
                &device,
                &render_texture_bind_group_layout,
                surface_config.format,
                &final_shader,
            );

        // Initialize voxel renderer with hot reloading
        let voxel_renderer = VoxelRenderer::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            &camera_bind_group_layout,
            Arc::clone(&shader_hot_reload),
        );

        // Initialize atmosphere renderer with hot reloading
        let mut atmosphere_renderer = AtmosphereRenderer::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            &camera_bind_group_layout,
        );

        // Load atmosphere shader using hot reload
        let atmosphere_shader = shader_hot_reload.get_shader("atmosphere.wgsl");
        atmosphere_renderer.update_shader(&atmosphere_shader);

        // Create atmosphere output bind group
        let atmosphere_output_bind_group =
            atmosphere_renderer.create_output_bind_group(&device, &render_texture_view);

        // Initialize ImGui
        let imgui = Self::init_imgui(&device, &queue, &window, surface_config.format);

        Self {
            surface,
            surface_config,
            adapter,
            device,
            queue,
            shader_hot_reload,
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
            final_render_pipeline,
            final_vertex_buffer,
            final_texture_bind_group,
            time: Instant::now(),
            hidpi_factor: window.scale_factor(),
            imgui,
            render_mode: RenderMode::Voxel, // Default to atmosphere rendering
            voxel_renderer,
            atmosphere_renderer,
            atmosphere_output_bind_group,
        }
    }

    /// Initialize core WGPU components
    async fn init_wgpu_core(
        window: Arc<Window>,
    ) -> (
        wgpu::Surface<'window>,
        wgpu::SurfaceConfiguration,
        wgpu::Adapter,
        Arc<wgpu::Device>,
        Arc<wgpu::Queue>,
    ) {
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

        (surface, surface_config, adapter, device, queue)
    }

    /// Create camera uniform buffer and bind group
    fn create_camera_resources(
        device: &wgpu::Device,
    ) -> (wgpu::Buffer, wgpu::BindGroupLayout, wgpu::BindGroup) {
        let camera_uniform = CameraUniform {
            view_proj: Matrix4::identity().into(),
            ..Default::default()
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
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

        (camera_buffer, camera_bind_group_layout, camera_bind_group)
    }

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

    /// Creates a color texture with the specified format
    fn create_color_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        label: &str,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        (texture, texture_view)
    }

    /// Creates a texture bind group layout
    fn create_texture_bind_group_layout(device: &wgpu::Device) -> Arc<wgpu::BindGroupLayout> {
        Arc::new(
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            }),
        )
    }

    /// Creates a render pipeline for final conversion with a specific shader
    fn create_final_render_pipeline_with_shader(
        device: &wgpu::Device,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
        shader: &wgpu::ShaderModule,
    ) -> (wgpu::RenderPipeline, wgpu::Buffer) {
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Final Render Pipeline Layout"),
            bind_group_layouts: &[texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create vertex buffer with a fullscreen quad
        let vertices = [
            // x, y, u, v
            -1.0f32, -1.0, 0.0, 0.0, // Bottom-left corner, texture UV at (0,0)
            1.0, -1.0, 1.0, 0.0, // Bottom-right corner, texture UV at (1,0)
            1.0, 1.0, 1.0, 1.0, // Top-right corner, texture UV at (1,1)
            -1.0, -1.0, 0.0, 0.0, // Bottom-left corner again
            1.0, 1.0, 1.0, 1.0, // Top-right corner again
            -1.0, 1.0, 0.0, 1.0, // Top-left corner, texture UV at (0,1)
        ];

        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Final Render Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Final Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 4 * 4, // 4 floats, 4 bytes each
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        // Position
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        // UV
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 2 * 4, // 2 floats, 4 bytes each
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        (render_pipeline, vertex_buffer)
    }

    /// Initialize ImGui
    fn init_imgui(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        window: &Window,
        format: wgpu::TextureFormat,
    ) -> ImguiState {
        let hidpi_factor = window.scale_factor();
        let mut context = imgui::Context::create();
        let mut platform = WinitPlatform::new(&mut context);
        platform.attach_window(
            context.io_mut(),
            window,
            imgui_winit_support::HiDpiMode::Default,
        );

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

        let clear_color = wgpu::Color {
            r: 0.1,
            g: 0.2,
            b: 0.3,
            a: 1.0,
        };
        let renderer_config = RendererConfig {
            texture_format: format,
            ..Default::default()
        };

        let renderer = Renderer::new(&mut context, device, queue, renderer_config);

        ImguiState {
            context,
            platform,
            renderer,
            clear_color,
            demo_open: true,
            last_frame: Instant::now(),
            last_cursor: None,
        }
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
            resolution: [
                self.surface_config.width as f32,
                self.surface_config.height as f32,
            ],
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

        // Recreate depth texture
        let (depth_texture, depth_texture_view) =
            Self::create_depth_texture(&self.device, &self.surface_config);
        self.depth_texture = depth_texture;
        self.depth_texture_view = depth_texture_view;

        // Recreate render texture (Rgba32Float)
        let (render_texture, render_texture_view) = Self::create_color_texture(
            &self.device,
            &self.surface_config,
            "Render Texture",
            wgpu::TextureFormat::Rgba32Float,
        );
        self.render_texture = render_texture;
        self.render_texture_view = render_texture_view;

        // Recreate post-process texture (Rgba32Float)
        let (post_process_texture, post_process_texture_view) = Self::create_color_texture(
            &self.device,
            &self.surface_config,
            "Post Process Texture",
            wgpu::TextureFormat::Rgba32Float,
        );
        self.post_process_texture = post_process_texture;
        self.post_process_texture_view = post_process_texture_view;

        // Update final_texture_bind_group to point to the render texture
        self.final_texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.render_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.render_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
            label: Some("final_texture_bind_group"),
        });

        // Update atmosphere output bind group
        self.atmosphere_output_bind_group = self
            .atmosphere_renderer
            .create_output_bind_group(&self.device, &self.render_texture_view);

        // Update post-processing effects
        self.bloom_effect.resize(
            self.surface_config.width,
            self.surface_config.height,
            &self.render_texture_view,
        );
        self.color_correction_effect
            .resize(&self.post_process_texture_view);
    }

    // Add a method to check for shader updates and recreate pipelines
    pub fn check_shader_updates(&mut self) {
        let updated_shaders = self.shader_hot_reload.check_for_updates();

        for shader_name in &updated_shaders {
            match shader_name.as_str() {
                "bloom.wgsl" => {
                    // Reload bloom shader and recreate bloom effect
                    let bloom_shader = self.shader_hot_reload.get_shader("bloom.wgsl");
                    self.bloom_effect = BloomEffect::new(
                        Arc::clone(&self.device),
                        Arc::clone(&self.queue),
                        Arc::clone(&self.render_texture_bind_group_layout),
                        Arc::clone(&self.sampler),
                        self.surface_config.width,
                        self.surface_config.height,
                        &self.render_texture_view,
                        &bloom_shader,
                    );
                    info!("Reloaded bloom shader");
                }
                "color_correction.wgsl" => {
                    // Recreate color correction effect
                    let color_correction_shader =
                        self.shader_hot_reload.get_shader("color_correction.wgsl");
                    self.color_correction_effect = ColorCorrectionEffect::new(
                        Arc::clone(&self.device),
                        Arc::clone(&self.queue),
                        &self.post_process_texture_view,
                        Arc::clone(&self.sampler),
                        Some(&color_correction_shader),
                    );
                    info!("Reloaded color correction shader");
                }
                "final_conversion.wgsl" => {
                    // Reload final conversion shader
                    let final_shader = self.shader_hot_reload.get_shader("final_conversion.wgsl");
                    let (final_render_pipeline, _) = Self::create_final_render_pipeline_with_shader(
                        &self.device,
                        &self.render_texture_bind_group_layout,
                        self.surface_config.format,
                        &final_shader,
                    );
                    self.final_render_pipeline = final_render_pipeline;
                    info!("Reloaded final conversion shader");
                }
                "atmosphere.wgsl" => {
                    // Reload atmosphere compute shader
                    let atmosphere_shader = self.shader_hot_reload.get_shader("atmosphere.wgsl");
                    self.atmosphere_renderer.update_shader(&atmosphere_shader);
                    info!("Reloaded atmosphere compute shader");
                }
                _ => {}
            }
        }

        // Check for voxel shader updates in the VoxelRenderer
        self.voxel_renderer.check_shader_updates();
    }

    /// Renders the scene
    pub fn draw(&mut self, world: &mut World, window: &Window) {
        // Check for shader updates before rendering
        self.check_shader_updates();

        let surface_texture = self
            .surface
            .get_current_texture()
            .expect("Failed to acquire next swap chain texture");

        let surface_texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Render scene based on the selected rendering mode
        match self.render_mode {
            RenderMode::Voxel => {
                // Render voxel scene to render_texture
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Voxel Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.render_texture_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    // Set camera bind group (index 0)
                    render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

                    // Render voxels
                    self.voxel_renderer.render(&mut render_pass);
                }
            }
            RenderMode::Atmosphere => {
                // Render atmosphere directly to render_texture using compute shader
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Atmosphere Compute Pass"),
                            timestamp_writes: None,
                        });

                    compute_pass.set_bind_group(1, &self.camera_bind_group, &[]); // Camera is at binding 1
                    self.atmosphere_renderer.render(
                        &mut compute_pass,
                        &self.atmosphere_output_bind_group,
                        self.surface_config.width,
                        self.surface_config.height,
                    );
                }
            }
        }

        // // Apply post-processing effects (all on Rgba32Float textures)
        // self.bloom_effect
        //     .render(&mut encoder, &self.render_texture_view);
        // self.bloom_effect.apply(
        //     &mut encoder,
        //     &self.post_process_texture_view,
        //     &self.render_texture_view,
        // );

        // self.color_correction_effect
        //     .update_uniform(ColorCorrectionUniform {
        //         brightness: 1.0,
        //         contrast: 1.0,
        //         saturation: 1.0,
        //     });
        // self.color_correction_effect
        //     .apply(&mut encoder, &self.render_texture_view);

        // Final render pass: Convert from Rgba32Float directly to swapchain
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Final Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.final_render_pipeline);
            render_pass.set_bind_group(0, &self.final_texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.final_vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..1); // Draw two triangles (6 vertices)
        }

        // Update and render ImGui
        self.render_imgui(&mut encoder, &surface_texture_view, world, window);

        self.queue.submit(Some(encoder.finish()));
        surface_texture.present();
    }

    /// Update and render ImGui UI
    fn render_imgui(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        texture_view: &wgpu::TextureView,
        world: &mut World,
        window: &Window,
    ) {
        let now = Instant::now();
        self.imgui
            .context
            .io_mut()
            .update_delta_time(now - self.imgui.last_frame);
        self.imgui.last_frame = now;

        self.imgui
            .platform
            .prepare_frame(self.imgui.context.io_mut(), window)
            .expect("Failed to prepare ImGui frame");

        let ui = self.imgui.context.frame();

        // ImGui UI controls
        ui.window("Settings")
            .size([300.0, 400.0], Condition::FirstUseEver)
            .always_auto_resize(true)
            .build(|| {
                // // Camera controls
                // if ui.button("Set Camera Position Origin") {
                //     for (_, (transform, _, _)) in world.query_mut::<(&mut Transform, &mut Camera, &mut CameraController)>() {
                //         transform.position = Point3::new(6.0, 2.2, 6.0);
                //     }
                // }

                // for (_, (transform, _, _)) in world.query_mut::<(&mut Transform, &mut Camera, &mut CameraController)>() {
                //     let mut pos: [f32; 3] = transform.position.into();
                //     if ui.input_float3("Camera Transform", &mut pos).build() {
                //         transform.position = pos.into();
                //     }
                // }

                // Render mode selection
                ui.separator();
                ui.text("Render Mode");

                // let mut is_voxel = matches!(self.render_mode, RenderMode::Voxel);
                // if ui.radio_button("Voxel", &mut is_voxel, is_voxel) && !is_voxel {
                //     self.render_mode = RenderMode::Voxel;
                //     is_voxel = true;
                // }
                // if ui.radio_button("Atmosphere", &mut is_voxel, !is_voxel) && is_voxel {
                //     self.render_mode = RenderMode::Atmosphere;
                //     is_voxel = false;
                // }

                // Show settings based on the current rendering mode
                match self.render_mode {
                    RenderMode::Voxel => {
                        // Voxel settings controls
                        ui.separator();
                        ui.text("Voxel Settings");

                        // Visualization options
                        let mut show_normals = self.voxel_renderer.voxel_settings.show_normals != 0;
                        if ui.checkbox("Show Normals", &mut show_normals) {
                            self.voxel_renderer.voxel_settings.show_normals =
                                if show_normals { 1 } else { 0 };
                            self.voxel_renderer.update_settings_buffer();
                        }

                        let mut show_steps = self.voxel_renderer.voxel_settings.show_steps != 0;
                        if ui.checkbox("Show Ray Steps", &mut show_steps) {
                            self.voxel_renderer.voxel_settings.show_steps =
                                if show_steps { 1 } else { 0 };
                            self.voxel_renderer.update_settings_buffer();
                        }
                    }
                    RenderMode::Atmosphere => {
                        // Atmosphere settings controls
                        ui.separator();
                        ui.text("Atmosphere Settings");

                        // Add atmosphere-specific settings here when needed
                        ui.text("No configurable atmosphere settings available yet.");
                    }
                }

                // Add shader hot reload section to UI
                ui.separator();
                ui.text("Shader Hot Reloading");
                if ui.button("Force Reload All Shaders") {
                    // Force reload all known shaders
                    let bloom_shader = self.shader_hot_reload.get_shader("bloom.wgsl");
                    self.bloom_effect = BloomEffect::new(
                        Arc::clone(&self.device),
                        Arc::clone(&self.queue),
                        Arc::clone(&self.render_texture_bind_group_layout),
                        Arc::clone(&self.sampler),
                        self.surface_config.width,
                        self.surface_config.height,
                        &self.render_texture_view,
                        &bloom_shader,
                    );

                    let color_correction_shader =
                        self.shader_hot_reload.get_shader("color_correction.wgsl");
                    self.color_correction_effect = ColorCorrectionEffect::new(
                        Arc::clone(&self.device),
                        Arc::clone(&self.queue),
                        &self.post_process_texture_view,
                        Arc::clone(&self.sampler),
                        Some(&color_correction_shader),
                    );

                    let final_shader = self.shader_hot_reload.get_shader("final_conversion.wgsl");
                    let (final_render_pipeline, _) = Self::create_final_render_pipeline_with_shader(
                        &self.device,
                        &self.render_texture_bind_group_layout,
                        self.surface_config.format,
                        &final_shader,
                    );
                    self.final_render_pipeline = final_render_pipeline;

                    // Force reload voxel shader
                    let voxel_shader = self.shader_hot_reload.get_shader("voxel.wgsl");
                    let render_pipeline = VoxelRenderer::create_pipeline(
                        &self.device,
                        &self.voxel_renderer.pipeline_layout,
                        &voxel_shader,
                    );
                    self.voxel_renderer.render_pipeline = render_pipeline;

                    // Force reload atmosphere shader
                    let atmosphere_shader = self.shader_hot_reload.get_shader("atmosphere.wgsl");
                    self.atmosphere_renderer.update_shader(&atmosphere_shader);

                    info!("Forced reload of all shaders");
                }
            });

        if self.imgui.last_cursor != ui.mouse_cursor() {
            self.imgui.last_cursor = ui.mouse_cursor();
            self.imgui.platform.prepare_render(ui, window);
        }

        self.imgui
            .renderer
            .render(
                self.imgui.context.render(),
                &self.queue,
                &self.device,
                &mut encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ImGui Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: texture_view,
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
            )
            .expect("ImGui rendering failed");
    }
}
