use crate::vertex::{create_vertex_buffer_layout, INDICES_SQUARE, VERTICES_SQUARE};
use crate::{
    BloomEffect, Camera, CameraController, ColorCorrectionEffect, ColorCorrectionUniform, Model,
    ModelInstance, RgbaImg, ShaderHotReload, Transform,
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

// --- Settings Struct (Unchanged) ---
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SettingsUniform {
    // Renamed struct for clarity
    // --- Fields Used by Shader Logic (Matching WGSL Order) ---
    pub surface_threshold: f32,  // offset 0
    pub max_terrain_height: f32, // offset 4
    pub voxel_size: f32,         // offset 8
    pub max_ray_steps: i32,      // offset 12
    // --- Auto Padding to 16 bytes ---
    pub max_distance: f32,      // offset 16
    pub min_distance: f32,      // offset 20 (unused in shader logic)
    pub ray_march_epsilon: f32, // offset 24
    pub water_level: f32,       // offset 28
    // --- Auto Padding to 32 bytes ---
    pub max_water_influenced_height: f32, // offset 32
    /// Padding to align `light_color` to 16 bytes.
    _padding0: [u8; 12], // offset 36, size 12 -> Next field at 48

    pub light_color: [f32; 4],     // offset 48, size 16
    pub light_direction: [f32; 4], // offset 64, size 16

    pub show_normals: i32,             // offset 80
    pub show_ray_steps: i32,           // offset 84
    pub visualize_distance_field: i32, // offset 88

    // --- Unused / Legacy Parameters (Matching WGSL Order for layout stability) ---
    pub max: f32, // offset 92 (unused)
    // --- Auto Padding to 96 bytes ---
    pub r_inner: f32,          // offset 96 (unused)
    pub r: f32,                // offset 100 (unused)
    pub max_water_height: f32, // offset 104 (unused) - Different from max_water_influenced_height!
    pub tunnel_radius: f32,    // offset 108 (unused)
    // --- Auto Padding to 112 bytes ---
    pub camera_speed: f32,       // offset 112 (unused)
    pub camera_time_offset: f32, // offset 116 (unused)
    pub voxel_level: i32,        // offset 120 (unused)

    // --- Explicit WGSL Padding Field Alignment & Representation ---
    /// Padding to align the representation of WGSL's `_padding: vec3<f32>` field.
    _padding1: [u8; 4], // offset 124, size 4 -> Next field at 128
    /// Represents the explicit `_padding: vec3<f32>` field in the WGSL struct.
    _wgsl_explicit_padding: [f32; 3], // offset 128, size 12

    // --- Final Struct Padding ---
    /// Padding to ensure the total struct size is a multiple of 16.
    _padding2: [u8; 4], // offset 140, size 4 -> Total size 144
}

const _: () = assert!(std::mem::size_of::<SettingsUniform>() == 144);

impl Default for SettingsUniform {
    fn default() -> Self {
        // Normalize the default light direction
        let dir = [-0.5f32, -0.7f32, -0.5f32];
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        let norm_dir = [dir[0] / len, dir[1] / len, dir[2] / len];

        Self {
            // --- Meaningful Defaults ---
            surface_threshold: 0.4,   // Controls how much terrain is generated
            max_terrain_height: 10.0, // Max height of terrain features
            voxel_size: 2.0f32.powf(-5.0), // Size of voxels, impacts detail/perf
            max_ray_steps: 128,       // Max steps for ray marching
            max_distance: 1000.0,     // Max view distance
            ray_march_epsilon: 0.001, // Small offset for ray marching
            water_level: 5.0,         // Y-coordinate for water plane
            max_water_influenced_height: 1.0, // Terrain height above water where color blends
            light_color: [1.0, 0.95, 0.9, 1.0], // Slightly warm white light
            light_direction: [norm_dir[0], norm_dir[1], norm_dir[2], 0.0], // From above-side

            // --- Debug Flags (Default Off) ---
            show_normals: 0,
            show_ray_steps: 0,
            visualize_distance_field: 0,

            // --- Unused/Padding/Legacy Fields (Zeroed) ---
            min_distance: 0.1, // Often unused, small positive value
            max: 0.0,
            r_inner: 0.0,
            r: 0.0,
            max_water_height: 0.0, // Unused field from old struct
            tunnel_radius: 0.0,
            camera_speed: 0.0,
            camera_time_offset: 0.0,
            voxel_level: 5,
            _padding0: [0; 12],
            _padding1: [0; 4],
            _wgsl_explicit_padding: [0.0; 3],
            _padding2: [0; 4],
        }
    }
}

impl SettingsUniform {
    pub fn update_voxel_size(&mut self) {
        self.voxel_size = 2.0f32.powf(-self.voxel_level as f32);
    }

    pub fn create_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Voxel Settings Buffer"),
            contents: bytemuck::cast_slice(&[*self]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }
}

// --- VoxelRenderer Struct ---
pub struct VoxelRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pub render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_index_buffer: wgpu::Buffer,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group: wgpu::BindGroup,
    pub voxel_settings: SettingsUniform,
    voxel_settings_buffer: wgpu::Buffer,
    voxel_settings_bind_group: wgpu::BindGroup,
    pub sun_incline: f32,
    pub time_of_day: f32,
    rgb_noise_texture: wgpu::Texture,
    gray_noise_texture: wgpu::Texture,
    gray_noise_cube_texture: wgpu::Texture,
    grain_texture: wgpu::Texture,
    dirt_texture: wgpu::Texture,
    pebble_texture: wgpu::Texture,
    texture_sampler: Arc<wgpu::Sampler>,
    shader_hot_reload: Arc<ShaderHotReload>,
    pub pipeline_layout: wgpu::PipelineLayout,
}

impl VoxelRenderer {
    /// Creates a new VoxelRenderer with voxel-specific resources
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        shader_hot_reload: Arc<ShaderHotReload>,
    ) -> Self {
        // Texture sampler for voxel textures
        let texture_sampler = Arc::new(device.create_sampler(&SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        // Load voxel textures
        macro_rules! load_texture_2d {
            ($path:expr, $label:expr) => {{
                let img = RgbaImg::new($path).unwrap();
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some($label),
                    size: wgpu::Extent3d {
                        width: img.width,
                        height: img.height,
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
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &img.bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * img.width),
                        rows_per_image: Some(img.height),
                    },
                    wgpu::Extent3d {
                        width: img.width,
                        height: img.height,
                        depth_or_array_layers: 1,
                    },
                );
                (
                    texture.clone(),
                    texture.create_view(&wgpu::TextureViewDescriptor::default()),
                )
            }};
        }

        let (rgb_noise_texture, rgb_noise_texture_view) =
            load_texture_2d!("./assets/images/textures/rgbnoise.png", "Rgb noise Texture");
        let (gray_noise_texture, gray_noise_texture_view) = load_texture_2d!(
            "./assets/images/textures/graynoise.png",
            "Gray noise Texture"
        );
        let (grain_texture, grain_texture_view) =
            load_texture_2d!("./assets/images/textures/grain.png", "Grain Texture");
        let (dirt_texture, dirt_texture_view) =
            load_texture_2d!("./assets/images/textures/mud.png", "Dirt Texture");
        let (pebble_texture, pebble_texture_view) =
            load_texture_2d!("./assets/images/textures/pebble.png", "Pebble Texture");

        let gray_noise_cube_data =
            std::fs::read("./assets/images/textures/graynoise_32x32x32_cube.bin")
                .expect("Failed to read gray noise cube")[20..20 + 32 * 32 * 32]
                .to_vec();
        let gray_noise_cube_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Gray Noise Cube Texture"),
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
                texture: &gray_noise_cube_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &gray_noise_cube_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(32),
                rows_per_image: Some(32),
            },
            wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 32,
            },
        );
        let gray_noise_cube_texture_view =
            gray_noise_cube_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Texture bind group layout
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D3,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        // Texture bind group
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&rgb_noise_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&gray_noise_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&gray_noise_cube_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&grain_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&dirt_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&pebble_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
            ],
            label: Some("texture_bind_group"),
        });

        // Voxel settings bind group layout
        let voxel_settings_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Voxel Settings Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Voxel settings
        let voxel_settings = SettingsUniform::default();
        let voxel_settings_buffer = voxel_settings.create_buffer(&device);
        let voxel_settings_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Voxel Settings Bind Group"),
            layout: &voxel_settings_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: voxel_settings_buffer.as_entire_binding(),
            }],
        });

        // Vertex and index buffers for full-screen quad
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

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Voxel Render Pipeline Layout"),
            bind_group_layouts: &[
                camera_bind_group_layout,
                &texture_bind_group_layout,
                &voxel_settings_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Get the voxel shader using hot reload
        let voxel_shader = shader_hot_reload.get_shader("voxel.wgsl");

        // Create render pipeline with hot reloaded shader
        let render_pipeline = Self::create_pipeline(&device, &pipeline_layout, &voxel_shader);

        Self {
            device,
            queue,
            render_pipeline,
            vertex_buffer,
            vertex_index_buffer,
            texture_bind_group_layout,
            texture_bind_group,
            voxel_settings,
            voxel_settings_buffer,
            voxel_settings_bind_group,
            sun_incline: 0.5,
            time_of_day: 12.0,
            rgb_noise_texture,
            gray_noise_texture,
            gray_noise_cube_texture,
            grain_texture,
            dirt_texture,
            pebble_texture,
            texture_sampler,
            shader_hot_reload,
            pipeline_layout,
        }
    }

    /// Create pipeline with the provided shader
    pub fn create_pipeline(
        device: &wgpu::Device,
        pipeline_layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Voxel Render Pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[create_vertex_buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba32Float,
                    blend: None, // Some(wgpu::BlendState::REPLACE)
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
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
        })
    }

    /// Check for shader updates and recreate pipeline if needed
    pub fn check_shader_updates(&mut self) {
        let updated_shaders = self.shader_hot_reload.check_for_updates();

        if updated_shaders.contains(&"voxel.wgsl".to_string()) {
            println!("Reloading voxel shader");
            let voxel_shader = self.shader_hot_reload.get_shader("voxel.wgsl");
            self.render_pipeline =
                Self::create_pipeline(&self.device, &self.pipeline_layout, &voxel_shader);
        }
    }

    /// Renders the voxel scene within a render pass
    pub fn render(&self, rpass: &mut wgpu::RenderPass) {
        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(1, &self.texture_bind_group, &[]);
        rpass.set_bind_group(2, &self.voxel_settings_bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.set_index_buffer(
            self.vertex_index_buffer.slice(..),
            wgpu::IndexFormat::Uint16,
        );
        rpass.draw_indexed(0..INDICES_SQUARE.len() as u32, 0, 0..1);
    }

    /// Updates the voxel settings buffer when settings change
    pub fn update_settings_buffer(&self) {
        self.queue.write_buffer(
            &self.voxel_settings_buffer,
            0,
            bytemuck::cast_slice(&[self.voxel_settings]),
        );
    }

    /// Updates light direction based on time of day and sun incline
    pub fn update_light_direction(&mut self) {
        let time_angle = (self.time_of_day / 24.0) * 2.0 * std::f32::consts::PI;
        let x = time_angle.cos();
        let z = time_angle.sin();
        let incline_angle = self.sun_incline * std::f32::consts::FRAC_PI_2;
        let y = incline_angle.sin();
        let horizontal_scale = incline_angle.cos();

        self.voxel_settings.light_direction = [x * horizontal_scale, y, z * horizontal_scale, 0.0];

        self.update_settings_buffer();
    }
}
