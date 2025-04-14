// src/sky_renderer.rs
use crate::{Settings, StaticTextures}; // Assuming Settings is accessible
use std::sync::Arc;
use tracing::info;
use wgpu::util::DeviceExt;

use crate::ShaderHotReload; // Use the alias if needed

pub struct SkyRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pub render_pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout,
    shader_hot_reload: Arc<ShaderHotReload>, // To reload sky shader
    depth_texture_bind_group_layout: wgpu::BindGroupLayout,
    depth_sampler: Arc<wgpu::Sampler>,
    pub depth_texture_bind_group: Option<wgpu::BindGroup>, // Option<> because view changes on resize
    static_textures: Arc<StaticTextures>,
}

impl SkyRenderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        shader_hot_reload: Arc<ShaderHotReload>,
        output_format: wgpu::TextureFormat, // The format of the texture we render to (Rgba32Float)
        static_textures: Arc<StaticTextures>,
    ) -> Self {
        // Layout for binding the custom depth texture (R32Float) and a sampler
        let depth_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sky Depth Texture Bind Group Layout"),
                entries: &[
                    // Depth Texture (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false }, // R32Float isn't filterable
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Sampler (required even for textureLoad sometimes)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // Use NonFiltering if you only use textureLoad
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        // Create a simple sampler
        let depth_sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sky Depth Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        // Create pipeline layout: Camera (0), Depth+Sampler (1), Settings (2)
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky Render Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, &depth_texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Get the initial sky shader
        let sky_shader = shader_hot_reload.get_shader("sky.wgsl");

        // Create the render pipeline
        let render_pipeline =
            Self::create_pipeline(&device, &pipeline_layout, &sky_shader, output_format);

        Self {
            device,
            queue,
            render_pipeline,
            pipeline_layout,
            shader_hot_reload,
            depth_texture_bind_group_layout,
            depth_sampler,
            depth_texture_bind_group: None, // Will be created on first resize/draw
            static_textures,
        }
    }

    /// Creates the graphics pipeline for sky rendering
    fn create_pipeline(
        device: &wgpu::Device,
        pipeline_layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        output_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Render Pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[], // Vertex positions generated in shader
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    // Blend state might be needed if you want transparency later,
                    // but for opaque sky replacing background, None/Replace is fine.
                    // Use AlphaBlending if sky can be semi-transparent.
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip, // Fullscreen quad
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Render front face of the quad
                ..Default::default()
            },
            depth_stencil: None, // No depth/stencil testing in this pass
            multisample: wgpu::MultisampleState::default(), // No MSAA
            multiview: None,
            cache: None,
        })
    }

    /// Updates the bind group containing the depth texture view. Call this on resize.
    pub fn update_depth_texture_bind_group(&mut self, depth_texture_view: &wgpu::TextureView) {
        self.depth_texture_bind_group =
            Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sky Depth Texture Bind Group"),
                layout: &self.depth_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(depth_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                    },
                ],
            }));
    }

    /// Checks for shader updates and recreates the pipeline if necessary.
    pub fn check_shader_updates(&mut self, output_format: wgpu::TextureFormat) {
        if self
            .shader_hot_reload
            .check_for_updates()
            .contains(&"sky.wgsl".to_string())
        {
            info!("Reloading sky shader");
            let sky_shader = self.shader_hot_reload.get_shader("sky.wgsl");
            self.render_pipeline = Self::create_pipeline(
                &self.device,
                &self.pipeline_layout,
                &sky_shader,
                output_format,
            );
        }
    }

    /// Add this renderer's pass to draw the sky
    pub fn render<'rpass>(&'rpass self, rpass: &mut wgpu::RenderPass<'rpass>) {
        if let Some(depth_bind_group) = &self.depth_texture_bind_group {
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(1, depth_bind_group, &[]);
            // Draw the fullscreen quad (4 vertices for triangle strip)
            rpass.draw(0..4, 0..1);
        } else {
            // This shouldn't happen after the first frame/resize, but good to check.
            tracing::warn!("SkyRenderer depth_texture_bind_group not set, skipping render.");
        }
    }
}
