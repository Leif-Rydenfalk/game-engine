use std::sync::Arc;

pub struct BloomEffect {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    texture_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    sampler: Arc<wgpu::Sampler>,
    bloom_mip_texture: wgpu::Texture,
    bloom_mip_texture_view: wgpu::TextureView,
    bloom_horizontal_texture: wgpu::Texture,
    bloom_horizontal_texture_view: wgpu::TextureView,
    bloom_vertical_texture: wgpu::Texture,
    bloom_vertical_texture_view: wgpu::TextureView,
    bloom_mip_bind_group: wgpu::BindGroup,
    bloom_horizontal_bind_group: wgpu::BindGroup,
    bloom_vertical_bind_group: wgpu::BindGroup,
    bloom_mip_pipeline: wgpu::RenderPipeline,
    bloom_horizontal_pipeline: wgpu::RenderPipeline,
    bloom_vertical_pipeline: wgpu::RenderPipeline,
}

impl BloomEffect {
    /// Creates a new BloomEffect instance.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        texture_bind_group_layout: Arc<wgpu::BindGroupLayout>,
        sampler: Arc<wgpu::Sampler>,
        width: u32,
        height: u32,
        render_texture_view: &wgpu::TextureView,
        bloom_shader: &wgpu::ShaderModule,
    ) -> Self {
        // Create bloom textures
        let bloom_mip_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Mip Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bloom_mip_texture_view =
            bloom_mip_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bloom_horizontal_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Horizontal Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bloom_horizontal_texture_view =
            bloom_horizontal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bloom_vertical_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Vertical Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bloom_vertical_texture_view =
            bloom_vertical_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create bloom bind groups
        let bloom_mip_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(render_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("bloom_mip_bind_group"),
        });

        let bloom_horizontal_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bloom_mip_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("bloom_horizontal_bind_group"),
        });

        let bloom_vertical_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bloom_horizontal_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("bloom_vertical_bind_group"),
        });

        // Create pipeline layouts
        let bloom_mip_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Mip Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let bloom_horizontal_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Horizontal Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let bloom_vertical_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Vertical Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create bloom pipelines
        let bloom_mip_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bloom Mip Pipeline"),
            layout: Some(&bloom_mip_pipeline_layout),
            vertex: wgpu::VertexState {
                module: bloom_shader,
                entry_point: Some("mip_vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: bloom_shader,
                entry_point: Some("mip_fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
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

        let bloom_horizontal_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Bloom Horizontal Pipeline"),
                layout: Some(&bloom_horizontal_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: bloom_shader,
                    entry_point: Some("horizontal_vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: bloom_shader,
                    entry_point: Some("horizontal_fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
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

        let bloom_vertical_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Bloom Vertical Pipeline"),
                layout: Some(&bloom_vertical_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: bloom_shader,
                    entry_point: Some("vertical_vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: bloom_shader,
                    entry_point: Some("vertical_fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
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

        Self {
            device,
            queue,
            texture_bind_group_layout,
            sampler,
            bloom_mip_texture,
            bloom_mip_texture_view,
            bloom_horizontal_texture,
            bloom_horizontal_texture_view,
            bloom_vertical_texture,
            bloom_vertical_texture_view,
            bloom_mip_bind_group,
            bloom_horizontal_bind_group,
            bloom_vertical_bind_group,
            bloom_mip_pipeline,
            bloom_horizontal_pipeline,
            bloom_vertical_pipeline,
        }
    }

    /// Updates bloom textures and bind groups when the window is resized.
    pub fn update_textures(
        &mut self,
        width: u32,
        height: u32,
        render_texture_view: &wgpu::TextureView,
    ) {
        self.bloom_mip_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Mip Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.bloom_mip_texture_view = self
            .bloom_mip_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.bloom_horizontal_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Horizontal Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.bloom_horizontal_texture_view = self
            .bloom_horizontal_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.bloom_vertical_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Vertical Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.bloom_vertical_texture_view = self
            .bloom_vertical_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.bloom_mip_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(render_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
            label: Some("bloom_mip_bind_group"),
        });

        self.bloom_horizontal_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.bloom_mip_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
                label: Some("bloom_horizontal_bind_group"),
            });

        self.bloom_vertical_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.bloom_horizontal_texture_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
                label: Some("bloom_vertical_bind_group"),
            });
    }

    /// Renders the bloom effect passes.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder) {
        // Generate bloom_mip_texture
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Mip Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_mip_texture_view,
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
            rpass.set_pipeline(&self.bloom_mip_pipeline);
            rpass.set_bind_group(0, &self.bloom_mip_bind_group, &[]);
            rpass.draw(0..4, 0..1);
        }

        // Apply horizontal blur
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Horizontal Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_horizontal_texture_view,
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
            rpass.set_pipeline(&self.bloom_horizontal_pipeline);
            rpass.set_bind_group(0, &self.bloom_horizontal_bind_group, &[]);
            rpass.draw(0..4, 0..1);
        }

        // Apply vertical blur
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Vertical Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_vertical_texture_view,
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
            rpass.set_pipeline(&self.bloom_vertical_pipeline);
            rpass.set_bind_group(0, &self.bloom_vertical_bind_group, &[]);
            rpass.draw(0..4, 0..1);
        }
    }

    /// Returns the final bloom texture view for post-processing.
    pub fn get_bloom_texture_view(&self) -> &wgpu::TextureView {
        &self.bloom_vertical_texture_view
    }
}
