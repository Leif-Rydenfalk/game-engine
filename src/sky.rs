// src/sky_renderer.rs
use crate::{Settings, ShaderHotReload, StaticTextures}; // Added ShaderHotReload
use std::sync::Arc;
use tracing::info;
use wgpu::util::DeviceExt;

pub struct SkyRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pub render_pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout, // <-- Store the layout
    shader_hot_reload: Arc<ShaderHotReload>, // <-- Store the hot reloader
    depth_texture_bind_group_layout: wgpu::BindGroupLayout,
    depth_sampler: Arc<wgpu::Sampler>,
    pub depth_texture_bind_group: Option<wgpu::BindGroup>,
    static_textures: Arc<StaticTextures>,
}

impl SkyRenderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        camera_bind_group_layout: &wgpu::BindGroupLayout, // Group 0
        shader_hot_reload: Arc<ShaderHotReload>,          // <-- Add parameter
        output_format: wgpu::TextureFormat,
        static_textures: Arc<StaticTextures>, // Group 2
    ) -> Self {
        // Layout for binding the custom depth texture (R32Float) and a sampler (Group 1)
        let depth_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sky Depth Texture Bind Group Layout"),
                entries: &[
                    // Depth Texture (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

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

        // Create pipeline layout: Camera (0), Depth+Sampler (1), Static Textures (2)
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky Render Pipeline Layout"),
            bind_group_layouts: &[
                camera_bind_group_layout,                   // Group 0
                &depth_texture_bind_group_layout,           // Group 1
                &static_textures.texture_bind_group_layout, // Group 2
            ],
            push_constant_ranges: &[],
        });

        // --- Use ShaderHotReload to get the shader ---
        let sky_shader = shader_hot_reload.get_shader("sky.wgsl");
        // -------------------------------------------

        let render_pipeline =
            Self::create_pipeline(&device, &pipeline_layout, &sky_shader, output_format);

        Self {
            device,
            queue,
            render_pipeline,
            pipeline_layout,   // <-- Store layout
            shader_hot_reload, // <-- Store hot reloader
            depth_texture_bind_group_layout,
            depth_sampler,
            depth_texture_bind_group: None,
            static_textures,
        }
    }

    // create_pipeline remains the same
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
                    blend: None, // Keep as None/Replace for opaque sky
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

    // update_depth_texture_bind_group remains the same
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

    /// Check for shader updates and recreate pipeline if needed.
    /// `output_format` is needed because the pipeline depends on the target format.
    pub fn check_shader_updates(&mut self, output_format: wgpu::TextureFormat) {
        let updated_shaders = self.shader_hot_reload.check_for_updates();

        if updated_shaders.contains(&"sky.wgsl".to_string()) {
            self.reload_shader(output_format);
        }
    }

    pub fn reload_shader(&mut self, output_format: wgpu::TextureFormat) {
        info!("Reloading sky shader");
        // Get the updated shader module
        let sky_shader = self.shader_hot_reload.get_shader("sky.wgsl");
        // Recreate the pipeline using the stored layout and the current output format
        self.render_pipeline = Self::create_pipeline(
            &self.device,
            &self.pipeline_layout, // Use the stored layout
            &sky_shader,
            output_format,
        );
    }

    // render remains the same
    pub fn render<'rpass>(&'rpass self, rpass: &mut wgpu::RenderPass<'rpass>) {
        if let Some(depth_bind_group) = &self.depth_texture_bind_group {
            rpass.set_pipeline(&self.render_pipeline);
            // Group 0 (Camera) is assumed to be set by the caller
            rpass.set_bind_group(1, depth_bind_group, &[]); // Bind depth texture + sampler
            rpass.set_bind_group(2, &self.static_textures.texture_bind_group, &[]); // Bind static textures
            rpass.draw(0..4, 0..1);
        } else {
            tracing::warn!("SkyRenderer depth_texture_bind_group not set, skipping render.");
        }
    }
}
