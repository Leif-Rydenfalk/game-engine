use crate::vertex::{create_vertex_buffer_layout, INDICES_SQUARE, VERTICES_SQUARE};
use crate::{
    BloomEffect, Camera, CameraController, ColorCorrectionEffect, ColorCorrectionUniform, Model,
    ModelInstance, RgbaImg, Transform,
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
pub struct Settings {
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
    pub light_color: [f32; 4],
    pub light_direction: [f32; 4],
    pub show_normals: i32,
    pub show_steps: i32,
    pub visualize_distance_field: i32,
    _padding: u32,
}

impl Default for Settings {
    fn default() -> Self {
        let voxel_level = 3;
        let voxel_size = 2.0f32.powf(-voxel_level as f32);
        Self {
            max: 10000.0,
            r_inner: 1.0,
            r: 1.0 + 0.8,
            max_height: 5.0,
            max_water_height: -2.2,
            water_height: -2.2,
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
            light_color: [1.0, 0.9, 0.75, 2.0],
            light_direction: [0.507746, 0.716817, 0.477878, 0.0],
            show_normals: 0,
            show_steps: 0,
            visualize_distance_field: 0,
            _padding: 0,
        }
    }
}

impl Settings {
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
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_index_buffer: wgpu::Buffer,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group: wgpu::BindGroup,
    pub voxel_settings: Settings,
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
}

impl VoxelRenderer {
    /// Creates a new VoxelRenderer with voxel-specific resources
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
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
                (texture.clone(), texture.create_view(&wgpu::TextureViewDescriptor::default()))
            }};
        }

        let (rgb_noise_texture, rgb_noise_texture_view) =
            load_texture_2d!("./assets/images/textures/rgbnoise.png", "Rgb noise Texture");
        let (gray_noise_texture, gray_noise_texture_view) =
            load_texture_2d!("./assets/images/textures/graynoise.png", "Gray noise Texture");
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
        let voxel_settings = Settings::default();
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

        // Render pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    camera_bind_group_layout,
                    &texture_bind_group_layout,
                    &voxel_settings_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let render_pipeline = crate::create_pipeline(
            &device,
            wgpu::TextureFormat::Rgba32Float,
            &render_pipeline_layout,
        );

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
        }
    }

    /// Renders the voxel scene within a render pass
    pub fn render(&self, rpass: &mut wgpu::RenderPass) {
        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(1, &self.texture_bind_group, &[]);
        rpass.set_bind_group(2, &self.voxel_settings_bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.set_index_buffer(self.vertex_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
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
}