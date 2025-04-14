use crate::vertex::{create_vertex_buffer_layout, INDICES_SQUARE, VERTICES_SQUARE};
use crate::*;
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

use tracing::{debug, error, info, trace, warn};

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
        let voxel_level = 5;
        let voxel_size = 2.0f32.powf(-voxel_level as f32);
        Self {
            max: 10000.0, // not used
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
            //steps: 512 * 2 * 2,
            steps: 512, // Too low values causes artifact around edges
            max_dist: 100000.0,
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
    pub render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_index_buffer: wgpu::Buffer,
    pub voxel_settings: Settings,
    voxel_settings_buffer: wgpu::Buffer,
    voxel_settings_bind_group: wgpu::BindGroup,
    static_textures: Arc<StaticTextures>,
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
        static_textures: Arc<StaticTextures>,
    ) -> Self {
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

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Voxel Render Pipeline Layout"),
            bind_group_layouts: &[
                camera_bind_group_layout,
                &static_textures.texture_bind_group_layout,
                &voxel_settings_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Get the voxel shader using hot reload
        let voxel_shader = shader_hot_reload.get_shader("voxel.wgsl");

        // Create render pipeline with hot reloaded shader
        let render_pipeline = Self::create_pipeline(
            &device,
            &pipeline_layout,
            &voxel_shader,
            wgpu::TextureFormat::Rgba32Float, // Main color format
            wgpu::TextureFormat::R32Float,    // Custom depth format
        );

        Self {
            device,
            queue,
            render_pipeline,
            vertex_buffer,
            vertex_index_buffer,
            voxel_settings,
            voxel_settings_buffer,
            voxel_settings_bind_group,
            static_textures,
            shader_hot_reload,
            pipeline_layout,
        }
    }

    /// Create pipeline with the provided shader and target formats
    pub fn create_pipeline(
        device: &wgpu::Device,
        pipeline_layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        color_format: wgpu::TextureFormat, // Format for @location(0)
        depth_format: wgpu::TextureFormat, // Format for @location(1)
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Voxel Render Pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"), // Ensure this matches your vertex shader entry point
                // --- FIX: Use an empty slice for buffers ---
                // This tells the pipeline not to expect any vertex buffers,
                // which is correct when generating vertices in the shader.
                buffers: &[],
                // -----------------------------------------
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"), // Ensure this matches your fragment shader entry point
                targets: &[
                    // Target @location(0) - Color
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: None, // Or Some(wgpu::BlendState::REPLACE)
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Target @location(1) - Custom Depth
                    Some(wgpu::ColorTargetState {
                        format: depth_format,
                        blend: None, // No blending needed for depth
                        write_mask: wgpu::ColorWrites::ALL, // Write the single float channel
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip, // Use TriangleStrip for 4 vertices
                strip_index_format: None, // None for TriangleStrip with 4 vertices
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling needed for a fullscreen quad
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None, // We are not using the traditional depth buffer here
            multisample: wgpu::MultisampleState::default(), // No MSAA
            multiview: None,
            cache: None,
        })
    }

    /// Check for shader updates and recreate pipeline if needed
    pub fn check_shader_updates(&mut self) {
        let updated_shaders = self.shader_hot_reload.check_for_updates();

        if updated_shaders.contains(&"voxel.wgsl".to_string()) {
            self.reload_shader();
        }
    }

    pub fn reload_shader(&mut self) {
        info!("Reloading voxel shader");
        let voxel_shader = self.shader_hot_reload.get_shader("voxel.wgsl");
        // Recreate pipeline with both formats
        self.render_pipeline = Self::create_pipeline(
            &self.device,
            &self.pipeline_layout, // Use the stored layout
            &voxel_shader,
            wgpu::TextureFormat::Rgba32Float, // Main color format
            wgpu::TextureFormat::R32Float,    // Custom depth format
        );
    }

    /// Renders the voxel scene within a render pass, writing to color and custom depth views
    pub fn render(
        &self,
        rpass: &mut wgpu::RenderPass,
        // No need to pass views here, they are set in WgpuCtx::draw's begin_render_pass
    ) {
        rpass.set_pipeline(&self.render_pipeline);
        // Bind groups remain the same (camera @ 0 is set outside, textures @ 1, settings @ 2)
        rpass.set_bind_group(1, &self.static_textures.texture_bind_group, &[]);
        rpass.set_bind_group(2, &self.voxel_settings_bind_group, &[]);
        rpass.draw(0..4, 0..1); // 4 vertices for TriangleStrip
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
