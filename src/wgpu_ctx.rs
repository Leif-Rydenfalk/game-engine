use crate::{
    BloomEffect, Camera, ColorCorrectionEffect, ColorCorrectionUniform, ImguiState, Model,
    ShaderHotReload, SkyRenderer, StaticTextures, Transform, VoxelRenderer,
};
use cgmath::{Matrix4, Point3, SquareMatrix};
use hecs::World;
use rand::Rng;
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

// Need access to previous camera state for motion vectors
#[repr(C)]
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TAACameraUniform {
    // For motion calculation
    prev_view_proj: [[f32; 4]; 4],
    prev_inv_view_proj: [[f32; 4]; 4],
    // Jitter offset for the current frame
    jitter_offset: [f32; 2],
    _padding: [f32; 2], // Adjust padding if needed
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
    taa_camera_buffer: wgpu::Buffer,
    taa_camera_bind_group_layout: wgpu::BindGroupLayout,
    taa_camera_bind_group: wgpu::BindGroup,

    // Rendering resources
    models: Vec<Model>,
    render_texture_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    render_texture: wgpu::Texture, // Rgba32Float HDR target
    render_texture_view: wgpu::TextureView,
    sampler: Arc<wgpu::Sampler>,

    // Custom Voxel Depth Buffer
    voxel_depth_texture: wgpu::Texture, // R32Float Depth target
    voxel_depth_texture_view: wgpu::TextureView,

    // Post-processing resources
    bloom_effect: BloomEffect,
    post_process_texture: wgpu::Texture, // Rgba32Float for bloom intermediate/output
    post_process_texture_view: wgpu::TextureView,
    color_correction_effect: ColorCorrectionEffect,

    // --- TAA Resources ---
    history_textures: [wgpu::Texture; 2], // Ping-pong history buffers (HDR)
    history_texture_views: [wgpu::TextureView; 2],
    taa_pipeline_layout: wgpu::PipelineLayout,
    taa_resolve_bind_group_layout: wgpu::BindGroupLayout, // Binds current frame, history, depth, etc.
    taa_resolve_bind_groups: [wgpu::BindGroup; 2],        // Ping-pong bind groups
    taa_resolve_pipeline: wgpu::ComputePipeline,          // TAA is often done in Compute

    // --- TAA State ---
    frame_index: u64,                 // To drive jitter pattern
    needs_history_reset: bool,        // Flag to clear history (e.g., after resize/teleport)
    prev_view_proj: Matrix4<f32>,     // Store previous frame's matrices for motion vectors
    prev_inv_view_proj: Matrix4<f32>, // Store previous frame's matrices for motion vectors

    // Final rendering resources (simplified conversion)
    final_render_pipeline: wgpu::RenderPipeline,
    final_vertex_buffer: wgpu::Buffer,
    final_texture_bind_group: wgpu::BindGroup, // Binds render_texture for final pass

    // State tracking
    time: Instant,
    hidpi_factor: f64,

    // Renderers and UI
    pub imgui: ImguiState,
    voxel_renderer: VoxelRenderer,
    sky_renderer: SkyRenderer,
}

impl<'window> WgpuCtx<'window> {
    /// Synchronous constructor wrapper
    pub fn new(window: Arc<Window>) -> Self {
        pollster::block_on(Self::new_async(window))
    }

    pub async fn new_async(window: Arc<Window>) -> Self {
        // Initialize WGPU core resources
        let (surface, surface_config, adapter, device, queue) =
            Self::init_wgpu_core(Arc::clone(&window)).await;

        // Create shader hot reloading manager
        let shader_hot_reload = Arc::new(ShaderHotReload::new(
            Arc::clone(&device),
            std::path::Path::new("src/shaders"),
        ));

        // Create camera resources
        let (
            camera_buffer,
            camera_bind_group_layout,
            camera_bind_group,
            taa_camera_buffer,
            taa_camera_bind_group_layout,
            taa_camera_bind_group,
        ) = Self::create_camera_resources(&device);

        // --- Create TAA History Textures (HDR, same format/size as render_texture) ---
        let history_texture_desc = wgpu::TextureDescriptor {
            label: Some("TAA History Texture"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float, // Match render_texture
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST, // STORAGE for compute, COPY_DST for reset
            view_formats: &[],
        };
        let history_textures = [
            device.create_texture(&history_texture_desc),
            device.create_texture(&history_texture_desc),
        ];
        let history_texture_views = [
            history_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            history_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];

        // --- Create TAA Resolve Bind Group Layout ---
        // This layout defines what the TAA shader needs access to
        let taa_resolve_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("TAA Resolve Bind Group Layout"),
                entries: &[
                    // Input: Current Frame's Rendered Scene (HDR)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Input: Previous Frame's TAA Result (History)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Input: Custom Depth Texture (for motion vectors)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            // R32Float is non-filterable, use UnfilterableFloat
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Sampler for Input/History (Linear)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Output: Current Frame's TAA Result (Write Target)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE, // Only COMPUTE needs storage texture write
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        // --- Create TAA Compute Pipeline ---
        let taa_shader = shader_hot_reload.get_shader("taa.wgsl");
        let taa_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TAA Resolve Pipeline Layout"),
            bind_group_layouts: &[
                &taa_camera_bind_group_layout,
                &taa_resolve_bind_group_layout,
            ], // Needs camera uniforms too
            push_constant_ranges: &[],
        });
        let taa_resolve_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TAA Resolve Compute Pipeline"),
                layout: Some(&taa_pipeline_layout),
                module: &taa_shader,
                entry_point: Some("main"), // Compute shader entry point
                compilation_options: Default::default(),
                cache: None,
            });

        // --- Initialize TAA state ---
        let frame_index = 0;
        let needs_history_reset = true; // Start needing a reset
        let prev_view_proj = Matrix4::identity();
        let prev_inv_view_proj = Matrix4::identity();

        // --- Create render targets ---
        // Custom depth texture (R32Float)
        let (voxel_depth_texture, voxel_depth_texture_view) = Self::create_color_texture(
            &device,
            &surface_config,
            "Voxel Custom Depth Texture",
            wgpu::TextureFormat::R32Float, // Use R32Float format
        );
        // Main HDR color texture (Rgba32Float)
        let (render_texture, render_texture_view) = Self::create_color_texture(
            &device,
            &surface_config,
            "Render Texture",
            wgpu::TextureFormat::Rgba32Float,
        );
        // Post-process texture (Rgba32Float)
        let (post_process_texture, post_process_texture_view) = Self::create_color_texture(
            &device,
            &surface_config,
            "Post Process Texture",
            wgpu::TextureFormat::Rgba32Float,
        );
        // --- End render targets ---

        let render_texture_bind_group_layout = Self::create_texture_bind_group_layout(&device);
        let sampler = Arc::new(device.create_sampler(&SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        // Create post-processing resources
        let bloom_shader = shader_hot_reload.get_shader("bloom.wgsl");
        let bloom_effect = BloomEffect::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            Arc::clone(&render_texture_bind_group_layout),
            Arc::clone(&sampler),
            surface_config.width,
            surface_config.height,
            &render_texture_view, // Bloom reads from the main render texture initially
            &bloom_shader,
        );

        let color_correction_shader = shader_hot_reload.get_shader("color_correction.wgsl");
        let color_correction_effect = ColorCorrectionEffect::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            // Correct ColorCorrection to read from render_texture and write to post_process_texture?
            // Or read from post_process and write back to render_texture? Let's assume Bloom writes to post_process, CC reads post_process writes render_texture
            &post_process_texture_view, // Let's assume CC reads from where Bloom wrote
            Arc::clone(&sampler),
            Some(&color_correction_shader),
        );

        let final_shader = shader_hot_reload.get_shader("final_conversion.wgsl");
        let (final_render_pipeline, final_vertex_buffer) =
            Self::create_final_render_pipeline_with_shader(
                &device,
                &render_texture_bind_group_layout,
                surface_config.format, // Target swapchain format
                &final_shader,
            );

        let static_textures =
            Arc::new(StaticTextures::new(Arc::clone(&device), Arc::clone(&queue)));

        // Initialize voxel renderer
        let voxel_renderer = VoxelRenderer::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            &camera_bind_group_layout,
            Arc::clone(&shader_hot_reload),
            static_textures.clone(),
        );

        // --- Initialize SkyRenderer ---
        let mut sky_renderer = SkyRenderer::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            &camera_bind_group_layout,
            Arc::clone(&shader_hot_reload),
            wgpu::TextureFormat::Rgba32Float, // Sky renders to the same HDR target
            static_textures,
        );
        // Initialize its depth texture bind group immediately
        sky_renderer.update_depth_texture_bind_group(&voxel_depth_texture_view);
        // --- End SkyRenderer Init ---

        // Initialize ImGui
        let imgui = Self::init_imgui(&device, &queue, &window, surface_config.format);

        // --- (!!) Update `final_texture_bind_group` creation ---
        // It should now point to one of the TAA history textures initially.
        // We'll update which one it points to each frame *after* TAA runs.
        let final_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &render_texture_bind_group_layout, // Using the simple layout for final pass
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    // Initially bind the first history view, will be updated in draw()
                    resource: wgpu::BindingResource::TextureView(&history_texture_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("final_texture_bind_group"),
        });

        // --- Create the specific TAA bind groups now that views exist ---
        let taa_resolve_bind_groups = [
            // Bind Group 0: Reads History 0, Writes History 1
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("TAA Bind Group 0 (R:H0, W:H1)"),
                layout: &taa_resolve_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&render_texture_view),
                    }, // Current Scene Input
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&history_texture_views[0]),
                    }, // History Read Input
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&voxel_depth_texture_view),
                    }, // Depth Input
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    }, // Sampler
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&history_texture_views[1]),
                    }, // History Write Output
                ],
            }),
            // Bind Group 1: Reads History 1, Writes History 0
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("TAA Bind Group 1 (R:H1, W:H0)"),
                layout: &taa_resolve_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&render_texture_view),
                    }, // Current Scene Input
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&history_texture_views[1]),
                    }, // History Read Input
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&voxel_depth_texture_view),
                    }, // Depth Input
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    }, // Sampler
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&history_texture_views[0]),
                    }, // History Write Output
                ],
            }),
        ];

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
            models: Vec::new(),
            render_texture_bind_group_layout,
            render_texture, // Rgba32Float color
            render_texture_view,
            sampler,
            voxel_depth_texture, // R32Float depth
            voxel_depth_texture_view,
            bloom_effect,
            post_process_texture, // Rgba32Float intermediate
            post_process_texture_view,
            color_correction_effect,
            final_render_pipeline,
            final_vertex_buffer,
            history_textures,
            history_texture_views,
            taa_camera_buffer,
            taa_camera_bind_group,
            taa_camera_bind_group_layout,
            taa_resolve_bind_group_layout,
            taa_resolve_bind_groups,
            taa_pipeline_layout,
            taa_resolve_pipeline,
            frame_index,
            needs_history_reset,
            prev_view_proj,
            prev_inv_view_proj,
            final_texture_bind_group,
            time: Instant::now(),
            hidpi_factor: window.scale_factor(),
            imgui,
            voxel_renderer,
            sky_renderer,
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
        surface_config.present_mode = wgpu::PresentMode::Immediate;
        surface.configure(&device, &surface_config);

        (surface, surface_config, adapter, device, queue)
    }

    /// Create camera uniform buffer and bind group
    fn create_camera_resources(
        device: &wgpu::Device,
    ) -> (
        wgpu::Buffer,
        wgpu::BindGroupLayout,
        wgpu::BindGroup,
        wgpu::Buffer,
        wgpu::BindGroupLayout,
        wgpu::BindGroup,
    ) {
        let camera_uniform = CameraUniform {
            view_proj: Matrix4::identity().into(),
            ..Default::default()
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_buffer"),
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

        let taa_camera_uniform = TAACameraUniform {
            prev_view_proj: Matrix4::identity().into(),
            ..Default::default()
        };

        let taa_camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("taa_cameara_buffer"),
            contents: bytemuck::cast_slice(&[taa_camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let taa_camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
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
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX
                            | wgpu::ShaderStages::FRAGMENT
                            | wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("taa_camera_bind_group_layout"),
            });

        let taa_camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &taa_camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: taa_camera_buffer.as_entire_binding(),
                },
            ],
            label: Some("taa_camera_bind_group"),
        });

        (
            camera_buffer,
            camera_bind_group_layout,
            camera_bind_group,
            taa_camera_buffer,
            taa_camera_bind_group_layout,
            taa_camera_bind_group,
        )
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

    fn get_jitter_offset(frame_index: u64, width: u32, height: u32) -> [f32; 2] {
        // Use a Halton sequence or similar low-discrepancy sequence for sub-pixel offsets
        // Example using Halton(2, 3) - need a proper implementation
        fn halton(index: u64, base: u64) -> f32 {
            let mut result = 0.0;
            let mut f = 1.0 / base as f32;
            let mut i = index;
            while i > 0 {
                result += f * (i % base) as f32;
                i /= base;
                f /= base as f32;
            }
            result
        }

        // Sequence length, e.g., 8 or 16 frames
        const JITTER_SEQUENCE_LENGTH: u64 = 8;
        let index_in_sequence = frame_index % JITTER_SEQUENCE_LENGTH;

        // Generate offset in range [-0.5, 0.5] pixels
        let x = halton(index_in_sequence + 1, 2) - 0.5; // Start index > 0 for Halton
        let y = halton(index_in_sequence + 1, 3) - 0.5;

        // Convert pixel offset to clip space offset (NDC)
        [
            x * (2.0 / width as f32),
            y * (2.0 / height as f32), // Y might need negation depending on NDC convention
        ]
    }

    /// Updates the camera uniform buffer
    pub fn update_camera_uniform(
        &mut self,
        view_proj: Matrix4<f32>,
        inv_view_proj: Matrix4<f32>,
        view: Matrix4<f32>,
        position: [f32; 3],
    ) {
        // let jitter = Self::get_jitter_offset(
        //     self.frame_index,
        //     self.surface_config.width,
        //     self.surface_config.height,
        // );

        let jitter = [0.0, 0.0];

        let jitter_matrix =
            Matrix4::from_translation(cgmath::Vector3::new(jitter[0], jitter[1], 0.0));
        let jittered_view_proj = jitter_matrix * view_proj; // Apply jitter *after* projection
        let jittered_inv_view_proj = jittered_view_proj.invert().unwrap_or(Matrix4::identity());

        let taa_camera_uniform = TAACameraUniform {
            prev_view_proj: self.prev_view_proj.into(), // Send previous frame's VP
            prev_inv_view_proj: self.prev_inv_view_proj.into(), // Send previous frame's inverse VP
            jitter_offset: jitter, // Send current jitter (optional, might not be needed if VP is jittered)
            _padding: [0.0; 2],    // Adjust as needed
        };

        self.queue.write_buffer(
            &self.taa_camera_buffer,
            0,
            bytemuck::cast_slice(&[taa_camera_uniform]),
        );

        let camera_uniform = CameraUniform {
            view_proj: jittered_view_proj.into(), // Send CURRENT JITTERED VP
            inv_view_proj: jittered_inv_view_proj.into(), // Send CURRENT JITTERED INV VP
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

        // --- Store previous matrices for NEXT frame ---
        self.prev_view_proj = view_proj; // Store the UNJITTERED VP
        self.prev_inv_view_proj = inv_view_proj; // Store the UNJITTERED inverse
    }

    /// Resizes rendering surfaces
    pub fn resize(&mut self, new_size: (u32, u32)) {
        info!("Resizing to {}x{}", new_size.0, new_size.1);
        let (width, height) = new_size;
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);

        // Recreate custom depth texture (R32Float)
        let (voxel_depth_texture, voxel_depth_texture_view) = Self::create_color_texture(
            &self.device,
            &self.surface_config,
            "Voxel Custom Depth Texture",
            wgpu::TextureFormat::R32Float,
        );
        self.voxel_depth_texture = voxel_depth_texture;
        self.voxel_depth_texture_view = voxel_depth_texture_view;

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

        // Update final_texture_bind_group to point to the correct final texture view
        // Assuming post-processing ends in render_texture_view for now
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

        // --- Recreate TAA History Textures ---
        let history_texture_desc = wgpu::TextureDescriptor {
            label: Some("TAA History Texture"),
            size: wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST, // Add COPY_DST if clearing/copying
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            view_formats: &[],
        };
        self.history_textures = [
            self.device.create_texture(&history_texture_desc),
            self.device.create_texture(&history_texture_desc),
        ];
        self.history_texture_views = [
            self.history_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            self.history_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];

        // --- Recreate TAA Bind Groups ---
        self.taa_resolve_bind_groups = [
            // Bind Group 0: Reads History 0, Writes History 1
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("TAA Bind Group 0 (R:H0, W:H1)"),
                layout: &self.taa_resolve_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.render_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.history_texture_views[0],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &self.voxel_depth_texture_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &self.history_texture_views[1],
                        ),
                    },
                ],
            }),
            // Bind Group 1: Reads History 1, Writes History 0
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("TAA Bind Group 1 (R:H1, W:H0)"),
                layout: &self.taa_resolve_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.render_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.history_texture_views[1],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &self.voxel_depth_texture_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &self.history_texture_views[0],
                        ),
                    },
                ],
            }),
        ];

        // --- Recreate Final Bind Group (references initial history view) ---
        self.final_texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.render_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    // Reset to bind view 0 initially after resize
                    resource: wgpu::BindingResource::TextureView(&self.history_texture_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
            label: Some("final_texture_bind_group (resized)"),
        });

        // --- Trigger History Reset ---
        self.needs_history_reset = true;

        // Update post-processing effects
        self.bloom_effect.resize(
            self.surface_config.width,
            self.surface_config.height,
            &self.render_texture_view, // Input to bloom
        );
        self.color_correction_effect.resize(
            &self.post_process_texture_view, // Input to CC
        );

        // --- Update SkyRenderer's depth texture view ---
        self.sky_renderer
            .update_depth_texture_bind_group(&self.voxel_depth_texture_view);
        info!("Updated sky renderer depth texture bind group.");
        // --- End SkyRenderer Update ---
    }

    // Add a method to check for shader updates and recreate pipelines
    pub fn check_shader_updates(&mut self) {
        let updated_shaders = self.shader_hot_reload.check_for_updates();

        for shader_name in &updated_shaders {
            match shader_name.as_str() {
                "bloom.wgsl" => {
                    // TODO: Implement actual bloom reload logic here
                    // let bloom_shader = self.shader_hot_reload.get_shader("bloom.wgsl");
                    // self.bloom_effect.reload_pipeline(&bloom_shader); // Or similar
                    warn!("Bloom shader updated, but automatic reload not implemented yet.");
                }
                "color_correction.wgsl" => {
                    // TODO: Implement actual color correction reload logic here
                    // let cc_shader = self.shader_hot_reload.get_shader("color_correction.wgsl");
                    // self.color_correction_effect.reload_pipeline(&cc_shader); // Or similar
                    warn!("Color correction shader updated, but automatic reload not implemented yet.");
                }
                "final_conversion.wgsl" => {
                    let final_shader = self.shader_hot_reload.get_shader("final_conversion.wgsl");
                    let (final_render_pipeline, _) = Self::create_final_render_pipeline_with_shader(
                        &self.device,
                        &self.render_texture_bind_group_layout,
                        self.surface_config.format,
                        &final_shader,
                    );
                    self.final_render_pipeline = final_render_pipeline;
                    info!("Reloaded final_conversion.wgsl shader and pipeline.");
                }
                "voxel.wgsl" => {
                    // Directly call the reload method, don't delegate the check
                    self.voxel_renderer.reload_shader();
                    info!("Reloaded voxel.wgsl shader and pipeline."); // Add confirmation
                }
                "sky.wgsl" => {
                    // Directly call the reload method, don't delegate the check
                    // Make sure SkyRenderer has a public `reload_shader` method
                    self.sky_renderer
                        .reload_shader(wgpu::TextureFormat::Rgba32Float);
                    info!("Reloaded sky.wgsl shader and pipeline.");
                }
                "taa.wgsl" => {
                    let taa_shader = self.shader_hot_reload.get_shader("taa.wgsl");
                    self.taa_resolve_pipeline =
                        self.device
                            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                                label: Some("TAA Resolve Compute Pipeline (Reloaded)"),
                                layout: Some(&self.taa_pipeline_layout),
                                module: &taa_shader,
                                entry_point: Some("main"),
                                compilation_options: Default::default(),
                                cache: None,
                            });
                    info!("Reloaded taa.wgsl shader and compute pipeline.");
                    self.needs_history_reset = true;
                }
                _ => {
                    // Ignore other files
                }
            }
        }
    }

    /// Renders the scene
    pub fn draw(&mut self, world: &mut World, window: &Window) {
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
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Main Encoder"),
            });

        // --- Voxel Rendering ---
        {
            let mut voxel_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Voxel Render Pass"),
                color_attachments: &[
                    // Attachment 0: Main HDR Color (Rgba32Float)
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.render_texture_view, // Write color
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store, // Store color for sky pass
                        },
                    }),
                    // Attachment 1: Custom Depth (R32Float)
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.voxel_depth_texture_view, // Write depth
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // Clear depth to far value used in sky shader check
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 100000000000.0 as f64,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store, // Store depth for sky pass reading
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Set camera once for this pass
            voxel_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            // Render Voxels (writes color@0 and depth@1)
            self.voxel_renderer.render(&mut voxel_pass);
        } // End Voxel Render Pass Scope

        // --- Sky Rendering ---
        {
            // Now voxel_depth_texture is finished being written to and can be read.
            // This pass only targets the color buffer.
            let mut sky_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Sky Render Pass"),
                color_attachments: &[
                    // Attachment 0: Main HDR Color (Rgba32Float)
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.render_texture_view, // Target same color buffer
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,    // <<<< Load the voxel colors
                            store: wgpu::StoreOp::Store, // Store combined result for post-fx/final
                        },
                    }),
                    // NO depth attachment here
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Set camera again for this pass
            sky_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            // Render Sky (reads depth texture via bind group, writes color@0)
            self.sky_renderer.render(&mut sky_pass);
        } // End Sky Render Pass Scope

        // --- Post Processing ---
        // TODO: Review post-processing chain. Does Bloom write to post_process_texture?
        // Does CC read post_process_texture and write back to render_texture?
        // Example: Bloom (render -> post_process), CC (post_process -> render)
        // For now, let's assume we just apply bloom to render_texture and write back to it

        // Apply Bloom (reads render_texture, writes back to render_texture via internal ping-pong)
        // self.bloom_effect.render(&mut encoder, &self.render_texture_view, &self.post_process_texture_view);

        // Apply Color Correction (reads render_texture, writes to post_process_texture - adjust as needed)
        // self.color_correction_effect.apply(&mut encoder, &self.post_process_texture_view); // Source = render_texture? Target = post_process?

        // --- TAA History Reset (if needed) ---
        if self.needs_history_reset {
            // Option 1: Clear texture (might require COPY_DST usage)
            // encoder.clear_texture(&self.history_textures[0], &wgpu::ImageSubresourceRange{..});
            // encoder.clear_texture(&self.history_textures[1], &wgpu::ImageSubresourceRange{..});

            // Option 2: Copy the current frame into both history buffers
            // This provides a valid starting point immediately.
            let current_history_write_index = (self.frame_index % 2) as usize;
            let next_history_write_index = ((self.frame_index + 1) % 2) as usize;
            encoder.copy_texture_to_texture(
                self.render_texture.as_image_copy(), // Source: Current render
                self.history_textures[current_history_write_index].as_image_copy(), // Dest: Current history write target
                wgpu::Extent3d {
                    width: self.surface_config.width,
                    height: self.surface_config.height,
                    depth_or_array_layers: 1,
                },
            );
            encoder.copy_texture_to_texture(
                self.render_texture.as_image_copy(), // Source: Current render
                self.history_textures[next_history_write_index].as_image_copy(), // Dest: Next history write target (will be read next frame)
                wgpu::Extent3d {
                    width: self.surface_config.width,
                    height: self.surface_config.height,
                    depth_or_array_layers: 1,
                },
            );
            self.needs_history_reset = false;
            info!("TAA history reset.");
        }

        // --- TAA Resolve (Compute Pass) ---
        {
            let read_history_index = ((self.frame_index + 1) % 2) as usize; // History texture that was WRITTEN last frame
            let write_history_index = (self.frame_index % 2) as usize; // History texture to WRITE this frame

            let mut taa_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TAA Resolve Pass"),
                timestamp_writes: None,
            });
            taa_pass.set_pipeline(&self.taa_resolve_pipeline);
            taa_pass.set_bind_group(0, &self.taa_camera_bind_group, &[]); // Camera uniforms (incl. prev matrices)
            taa_pass.set_bind_group(1, &self.taa_resolve_bind_groups[write_history_index], &[]); // Use the group that reads correct history and writes to correct target

            // Dispatch compute shaders
            let workgroup_size_x = 8; // Match shader's workgroup size
            let workgroup_size_y = 8;
            let num_workgroups_x =
                (self.surface_config.width + workgroup_size_x - 1) / workgroup_size_x;
            let num_workgroups_y =
                (self.surface_config.height + workgroup_size_y - 1) / workgroup_size_y;
            taa_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);

            // --- Update Final Texture Bind Group to point to the *result* of TAA ---
            // This MUST be done *after* dispatching but conceptually "linked" to this pass
            // It needs to know which texture view the compute shader just wrote to.
            // Determine the ACTUAL index that was written to based on the bind group used
            let actual_output_texture_index = if write_history_index == 0 { 1 } else { 0 }; // BG0 writes H1, BG1 writes H0
            let taa_output_view = &self.history_texture_views[actual_output_texture_index];

            self.final_texture_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.render_texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(taa_output_view), // Bind the texture TAA just wrote to
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                    label: Some("final_texture_bind_group (updated after TAA dispatch)"),
                });
        } // End TAA Compute Pass Scope

        // --- Update Final Texture Bind Group to point to the *result* of TAA ---
        let taa_output_view = &self.history_texture_views[(self.frame_index % 2) as usize];
        self.final_texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.render_texture_bind_group_layout, // Final pass uses simple layout
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(taa_output_view), // Bind the texture TAA just wrote to
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler), // Use the appropriate sampler
                },
            ],
            label: Some("final_texture_bind_group (updated)"),
        });

        // --- Final Render Pass (Tonemapping, Gamma Correction) ---
        // Input is now the TAA result (one of the history buffers)
        {
            let mut final_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Final Conversion Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_texture_view, // Target: Swapchain
                    // ... ops: LoadOp::Load (or Clear if nothing else drawn before) ...
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    resolve_target: None,
                })],
                ..Default::default()
            });

            final_render_pass.set_pipeline(&self.final_render_pipeline);
            // Bind the *correctly updated* final_texture_bind_group containing the TAA result
            final_render_pass.set_bind_group(0, &self.final_texture_bind_group, &[]);
            final_render_pass.set_vertex_buffer(0, self.final_vertex_buffer.slice(..));
            final_render_pass.draw(0..6, 0..1); // Draw fullscreen quad
        }

        // // --- Final Render Pass ---
        // // Renders the result of scene + sky + post-processing (assumed to be in render_texture_view)
        // // to the swapchain texture (surface_texture_view) using a simple conversion shader.
        // {
        //     let mut final_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        //         label: Some("Final Conversion Pass"),
        //         color_attachments: &[Some(wgpu::RenderPassColorAttachment {
        //             view: &surface_texture_view, // Target: Swapchain
        //             resolve_target: None,
        //             ops: wgpu::Operations {
        //                 load: wgpu::LoadOp::Load, // Don't clear if post-processing happened
        //                 store: wgpu::StoreOp::Store,
        //             },
        //         })],
        //         depth_stencil_attachment: None,
        //         timestamp_writes: None,
        //         occlusion_query_set: None,
        //     });

        //     final_render_pass.set_pipeline(&self.final_render_pipeline);
        //     // Ensure final_texture_bind_group binds the correct texture (render_texture_view or post_process_texture_view)
        //     final_render_pass.set_bind_group(0, &self.final_texture_bind_group, &[]);
        //     final_render_pass.set_vertex_buffer(0, self.final_vertex_buffer.slice(..));
        //     final_render_pass.draw(0..6, 0..1); // Draw fullscreen quad
        // }

        // --- Render ImGui ---
        // Renders ImGui onto the swapchain texture *after* the final scene render.
        self.render_imgui(&mut encoder, &surface_texture_view, world, window);

        // --- Submit ---
        self.queue.submit(Some(encoder.finish()));
        surface_texture.present();

        // Increment frame index for next frame's jitter and TAA ping-pong
        self.frame_index += 1; // Crucial!
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
                // Camera controls
                if ui.button("Set Camera Position Origin") {
                    for (_, (transform, _)) in world.query_mut::<(&mut Transform, &mut Camera)>() {
                        transform.position = Point3::new(6.0, 2.2, 6.0);
                    }
                }

                for (_, (transform, _)) in world.query_mut::<(&mut Transform, &mut Camera)>() {
                    let mut pos: [f32; 3] = transform.position.into();
                    if ui.input_float3("Camera Transform", &mut pos).build() {
                        transform.position = pos.into();
                    }
                }

                ui.separator();
                ui.text("Time Controls");
                if ui.button("Reset Time") {
                    self.time = Instant::now(); // This resets the time to the current moment
                }

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
                    self.voxel_renderer.voxel_settings.show_steps = if show_steps { 1 } else { 0 };
                    self.voxel_renderer.update_settings_buffer();
                }

                if ui.slider(
                    "Voxel Size",
                    0.0,
                    1.0,
                    &mut self.voxel_renderer.voxel_settings.voxel_size,
                ) {
                    self.voxel_renderer.update_settings_buffer();
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

                    self.sky_renderer
                        .reload_shader(wgpu::TextureFormat::Rgba32Float);
                    self.voxel_renderer.reload_shader();

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
