use crate::vertex::{create_vertex_buffer_layout, INDICES_SQUARE, VERTICES_SQUARE};
use crate::{
    BloomEffect, ColorCorrectionEffect, ColorCorrectionUniform, Model, ModelInstance, RgbaImg,
    Transform,
};
use cgmath::{Matrix4, SquareMatrix};
use hecs::World;
use std::borrow::Cow;
use std::{path::Path, sync::Arc, time::Instant};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{MemoryHints, SamplerDescriptor, ShaderSource};
use winit::window::Window;

use imgui::*;
use imgui_wgpu::{Renderer, RendererConfig};
use imgui_winit_support::WinitPlatform;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelSettings {
    // Constants from shader
    pub max: f32,
    pub r_inner: f32,
    pub r: f32,
    pub max_height: f32,
    pub max_water_height: f32,
    pub water_height: f32,
    pub tunnel_radius: f32,
    pub surface_factor: f32,
    pub camera_speed: f32,
    pub camera_time_offset: f32,
    pub voxel_level: i32,
    pub voxel_size: f32,
    pub steps: i32,
    pub max_dist: f32,
    pub min_dist: f32,
    pub eps: f32,

    // Light settings
    pub light_color: [f32; 4],     // Using vec4 for alignment
    pub light_direction: [f32; 4], // Using vec4 for alignment

    // Debug flags (using i32 as bools for uniform compatibility)
    pub show_normals: i32,
    pub show_steps: i32,
    pub visualize_distance_field: i32,

    _padding: u32,
    // // Padding to ensure 16-byte alignment
    // _padding: [u8; 8],
    // _padding: u32,
    // _padding2: u8
}

impl Default for VoxelSettings {
    fn default() -> Self {
        // Calculate voxel_size based on voxel_level for consistency
        let voxel_level = 3;
        let voxel_size = 2.0f32.powf(-voxel_level as f32);

        Self {
            max: 10000.0,
            r_inner: 1.0,
            r: 1.0 + 0.8, // R_INNER + 0.8
            max_height: 5.0,
            max_water_height: -2.2,
            water_height: -2.2, // Same as MAX_WATER_HEIGHT
            tunnel_radius: 1.1,
            surface_factor: 0.42,
            camera_speed: -1.5,
            camera_time_offset: 0.0,
            voxel_level,
            voxel_size,
            steps: 512 * 2 * 2,
            max_dist: 600000.0,
            min_dist: 0.0001,
            eps: 1e-5,

            // Light settings - converted to arrays for uniform compatibility
            light_color: [1.0, 0.9, 0.75, 2.0], // vec3f(1.0, 0.9, 0.75) * 2.0
            light_direction: [0.507746, 0.716817, 0.477878, 0.0], // Normalized in shader

            // Debug flags
            show_normals: 0,             // false
            show_steps: 0,               // false
            visualize_distance_field: 0, // false

            // _padding: [0; 8],
            _padding: 0,
        }
    }
}

impl VoxelSettings {
    pub fn update_voxel_size(&mut self) {
        self.voxel_size = 2.0f32.powf(-self.voxel_level as f32);
    }

    // Create buffer from settings
    pub fn create_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Voxel Settings Buffer"),
            contents: bytemuck::cast_slice(&[*self]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }
}

pub struct ImguiState {
    pub context: imgui::Context,
    pub platform: WinitPlatform,
    pub renderer: Renderer,
    pub clear_color: wgpu::Color,
    pub demo_open: bool,
    pub last_frame: Instant,
    pub last_cursor: Option<MouseCursor>,
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    position: [f32; 3],
    time: f32,
}

pub struct WgpuCtx<'window> {
    surface: wgpu::Surface<'window>,
    surface_config: wgpu::SurfaceConfiguration,
    adapter: wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_index_buffer: wgpu::Buffer,
    texture: wgpu::Texture,
    texture_size: wgpu::Extent3d,
    sampler: Arc<wgpu::Sampler>,
    texture_sampler: Arc<wgpu::Sampler>,
    bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,
    models: Vec<Model>,
    texture_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    render_texture: wgpu::Texture,
    render_texture_view: wgpu::TextureView,
    bloom_effect: BloomEffect,
    post_process_texture: wgpu::Texture,
    post_process_texture_view: wgpu::TextureView,
    color_correction_effect: ColorCorrectionEffect,
    noise0_texture: wgpu::Texture,
    noise1_texture: wgpu::Texture,
    grain_texture: wgpu::Texture,
    dirt_texture: wgpu::Texture,
    terrain_bind_group_layout: wgpu::BindGroupLayout,
    terrain_bind_group: wgpu::BindGroup,
    time: Instant,
    hidpi_factor: f64,
    pub imgui: ImguiState,
    voxel_settings: VoxelSettings,
    voxel_settings_buffer: wgpu::Buffer,
    voxel_settings_bind_group: wgpu::BindGroup,
}

impl<'window> WgpuCtx<'window> {
    /// Creates a depth texture and its view for depth testing
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

    /// Asynchronous constructor for WgpuCtx
    pub async fn new_async(window: Arc<Window>) -> WgpuCtx<'window> {
        // Core WGPU setup
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
        let surface_config = surface.get_default_config(&adapter, width, height).unwrap();
        surface.configure(&device, &surface_config);

        // Vertex and index buffers for rendering a full-screen quad
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&VERTICES_SQUARE),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let vertex_index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&INDICES_SQUARE),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Non repeat sampler for render texture
        let sampler = Arc::new(device.create_sampler(&SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        // Shared sampler for texture sampling
        let texture_sampler = Arc::new(device.create_sampler(&SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        // Load multiple textures (emulating Shadertoy iChannels)
        // Noise0 texture
        let noise0_img = RgbaImg::new("./assets/images/textures/rgbnoise.png").unwrap();
        let noise0_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Noise0 Texture"),
            size: wgpu::Extent3d {
                width: noise0_img.width,
                height: noise0_img.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &noise0_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &noise0_img.bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * noise0_img.width),
                rows_per_image: Some(noise0_img.height),
            },
            wgpu::Extent3d {
                width: noise0_img.width,
                height: noise0_img.height,
                depth_or_array_layers: 1,
            },
        );
        let noise0_texture_view =
            noise0_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Noise1 texture (3D)
        let noise1_data_full =
            std::fs::read("./assets/images/textures/graynoise_32x32x32_cube.bin")
                .expect("Failed to read noise1 binary file");
        let noise1_data = &noise1_data_full[20..20 + 32 * 32 * 32];
        assert_eq!(
            noise1_data.len(),
            32 * 32 * 32,
            "Noise1 data size mismatch; expected 32768 bytes"
        );

        let noise1_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Noise1 Texture"),
            size: wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &noise1_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            noise1_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(32), // 32 texels * 1 byte per texel
                rows_per_image: Some(32),
            },
            wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 32,
            },
        );

        let noise1_texture_view =
            noise1_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Grain texture
        let grain_img = RgbaImg::new("./assets/images/textures/stone.png").unwrap();
        let grain_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Grain Texture"),
            size: wgpu::Extent3d {
                width: grain_img.width,
                height: grain_img.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &grain_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &grain_img.bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * grain_img.width),
                rows_per_image: Some(grain_img.height),
            },
            wgpu::Extent3d {
                width: grain_img.width,
                height: grain_img.height,
                depth_or_array_layers: 1,
            },
        );
        let grain_texture_view = grain_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Dirt texture
        let dirt_img = RgbaImg::new("./assets/images/textures/mud.png").unwrap();
        let dirt_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dirt Texture"),
            size: wgpu::Extent3d {
                width: dirt_img.width,
                height: dirt_img.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &dirt_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &dirt_img.bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dirt_img.width),
                rows_per_image: Some(dirt_img.height),
            },
            wgpu::Extent3d {
                width: dirt_img.width,
                height: dirt_img.height,
                depth_or_array_layers: 1,
            },
        );
        let dirt_texture_view = dirt_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Terrain bind group layout for multiple textures
        let terrain_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2, // noise0_texture is 2D
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D3, // noise1_texture is 3D
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2, // grain_texture is 2D
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2, // dirt_texture is 2D
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("terrain_bind_group_layout"),
            });

        // Terrain bind group to bind textures and sampler
        let terrain_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &terrain_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&noise0_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&noise1_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&grain_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&dirt_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
            ],
            label: Some("terrain_bind_group"),
        });

        // Create the bind group layout
        let voxel_settings_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Voxel Settings Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT, //  | wgpu::ShaderStages::VERTEX  | wgpu::ShaderStages::COMPUTE
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Create the default settings
        let voxel_settings = VoxelSettings::default();

        // Create the buffer
        let voxel_settings_buffer = voxel_settings.create_buffer(&device);

        // Create the bind group
        let voxel_settings_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Voxel Settings Bind Group"),
            layout: &voxel_settings_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: voxel_settings_buffer.as_entire_binding(),
            }],
        });

        // Camera uniform and bind group
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

        // Render pipeline setup
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &terrain_bind_group_layout,
                    &voxel_settings_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let render_pipeline = create_pipeline(
            &device,
            wgpu::TextureFormat::Rgba32Float,
            &render_pipeline_layout,
        );

        // Depth texture
        let (depth_texture, depth_texture_view) =
            Self::create_depth_texture(&device, &surface_config);

        // Texture bind group layout for post-processing
        let texture_bind_group_layout = Arc::new(device.create_bind_group_layout(
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

        // Render texture for intermediate rendering
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let render_texture_view =
            render_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Bloom effect setup
        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("bloom.wgsl"))),
        });
        let bloom_effect = BloomEffect::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            Arc::clone(&texture_bind_group_layout),
            Arc::clone(&sampler),
            surface_config.width,
            surface_config.height,
            &render_texture_view,
            &bloom_shader,
        );

        // Post-process texture
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let post_process_texture_view =
            post_process_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Color correction effect
        let color_correction_effect = ColorCorrectionEffect::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            &post_process_texture_view,
            Arc::clone(&sampler),
            surface_config.format,
        );

        let hidpi_factor = window.scale_factor();

        let imgui = {
            let mut context = imgui::Context::create();
            let mut platform = imgui_winit_support::WinitPlatform::new(&mut context);
            platform.attach_window(
                context.io_mut(),
                &window,
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

            //
            // Set up dear imgui wgpu renderer
            //
            let clear_color = wgpu::Color {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 1.0,
            };

            let renderer_config = RendererConfig {
                texture_format: surface_config.format,
                ..Default::default()
            };

            let renderer = Renderer::new(&mut context, &device, &queue, renderer_config);
            let last_frame = Instant::now();
            let last_cursor = None;
            let demo_open = true;

            ImguiState {
                context,
                platform,
                renderer,
                clear_color,
                demo_open,
                last_frame,
                last_cursor,
            }
        };

        WgpuCtx {
            surface,
            surface_config,
            adapter,
            device,
            queue,
            render_pipeline,
            vertex_buffer,
            vertex_index_buffer,
            texture: noise1_texture.clone(), // Primary texture for compatibility
            texture_size: wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 32,
            },
            texture_sampler,
            sampler,
            bind_group: terrain_bind_group.clone(),
            camera_buffer,
            camera_bind_group,
            depth_texture,
            depth_texture_view,
            models: Vec::new(),
            texture_bind_group_layout,
            render_texture,
            render_texture_view,
            bloom_effect,
            post_process_texture,
            post_process_texture_view,
            color_correction_effect,
            noise0_texture,
            noise1_texture,
            grain_texture,
            dirt_texture,
            terrain_bind_group_layout,
            terrain_bind_group,
            time: Instant::now(),
            imgui,
            hidpi_factor,
            voxel_settings,
            voxel_settings_buffer,
            voxel_settings_bind_group,
        }
    }

    pub fn load_model<P: AsRef<Path>>(&mut self, path: P) -> Option<usize> {
        if let Some(mut model) = Model::load(&self.device, &self.queue, path) {
            model.create_bind_groups(&self.device, &self.texture_bind_group_layout);
            model.upload_textures(&self.queue);
            let index = self.models.len();
            self.models.push(model);
            Some(index)
        } else {
            None
        }
    }

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
        };
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );
    }
    /// Synchronous constructor that blocks on async initialization
    pub fn new(window: Arc<Window>) -> WgpuCtx<'window> {
        pollster::block_on(WgpuCtx::new_async(window))
    }

    /// Resizes the rendering surfaces and updates related resources
    pub fn resize(&mut self, new_size: (u32, u32)) {
        let (width, height) = new_size;
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);

        let (depth_texture, depth_texture_view) =
            Self::create_depth_texture(&self.device, &self.surface_config);
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        self.render_texture_view = self
            .render_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        self.post_process_texture_view = self
            .post_process_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.bloom_effect.resize(
            self.surface_config.width,
            self.surface_config.height,
            &self.render_texture_view,
        );
        self.color_correction_effect
            .resize(&self.post_process_texture_view);
    }

    /// Renders the scene with post-processing effects
    pub fn draw(&mut self, world: &World, window: &Window) {
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

        // Render the scene to an intermediate texture
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Scene Render Pass"),
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

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.camera_bind_group, &[]);
            rpass.set_bind_group(1, &self.terrain_bind_group, &[]);
            rpass.set_bind_group(2, &self.voxel_settings_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_index_buffer(
                self.vertex_index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            rpass.draw_indexed(0..INDICES_SQUARE.len() as u32, 0, 0..1);
        }

        // Apply post-processing effects
        self.bloom_effect
            .render(&mut encoder, &self.render_texture_view);
        self.bloom_effect.apply(
            &mut encoder,
            &self.post_process_texture_view,
            &self.render_texture_view,
        );
        self.color_correction_effect
            .update_uniform(ColorCorrectionUniform {
                brightness: 1.0,
                contrast: 1.0,
                saturation: 1.0,
            });
        self.color_correction_effect
            .apply(&mut encoder, &surface_texture_view);

        // Setup UI first
        // Update time delta
        let now = Instant::now();
        self.imgui
            .context
            .io_mut()
            .update_delta_time(now - self.imgui.last_frame);
        self.imgui.last_frame = now;

        // Prepare frame
        self.imgui
            .platform
            .prepare_frame(self.imgui.context.io_mut(), window)
            .expect("Failed to prepare ImGui frame");
        let ui = self.imgui.context.frame();

        // Build your UI here
        {
            let mut modified = false;
            let window = ui.window("Voxel Settings");
            window
                .size([300.0, 200.0], Condition::FirstUseEver)
                .build(|| {
                    if ui.slider("Voxel Level", 1, 7, &mut self.voxel_settings.voxel_level) {
                        self.voxel_settings.update_voxel_size();
                        modified = true;
                    }
                    // // Add buttons to test mouse capture
                    // if ui.button("Test Button") {
                    //     println!("ImGui button clicked!");
                    // }

                    // // Add input field to test keyboard capture
                    // let mut test_input = String::new();
                    // ui.input_text("Test Input", &mut test_input).build();

                    // // Debug info
                    // ui.text(format!(
                    //     "ImGui wants capture: Mouse={}, Keyboard={}",
                    //     ui.io().want_capture_mouse,
                    //     ui.io().want_capture_keyboard
                    // ));
                    // ui.separator();
                });
            // // Show demo window (useful while developing)
            // ui.show_demo_window(&mut imgui.demo_open);

            if modified {
                self.queue.write_buffer(
                    &self.voxel_settings_buffer,
                    0,
                    bytemuck::cast_slice(&[self.voxel_settings]),
                );
            }
        }

        // Update cursor if changed
        if self.imgui.last_cursor != ui.mouse_cursor() {
            self.imgui.last_cursor = ui.mouse_cursor();
            self.imgui.platform.prepare_render(ui, window);
        }

        // Render ImGui UI on top of the scene
        self.imgui
            .renderer
            .render(
                self.imgui.context.render(),
                &self.queue,
                &self.device,
                &mut encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ImGui Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &surface_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Important: Load existing content
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                }),
            )
            .expect("ImGui rendering failed");

        self.queue.submit(Some(encoder.finish()));
        surface_texture.present();
    }
}

fn create_pipeline(
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
