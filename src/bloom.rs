use std::sync::Arc;

pub struct BloomEffect {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    texture_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    sampler: Arc<wgpu::Sampler>,
    max_level: u32,
    bloom_downsample_textures: Vec<wgpu::Texture>,
    bloom_downsample_views: Vec<wgpu::TextureView>,
    bloom_horizontal_textures: Vec<wgpu::Texture>,
    bloom_horizontal_views: Vec<wgpu::TextureView>,
    bloom_vertical_textures: Vec<wgpu::Texture>,
    bloom_vertical_views: Vec<wgpu::TextureView>,
    bloom_downsample_bind_groups: Vec<wgpu::BindGroup>,
    bloom_horizontal_bind_groups: Vec<wgpu::BindGroup>,
    bloom_vertical_bind_groups: Vec<wgpu::BindGroup>,
    bright_extract_pipeline: wgpu::RenderPipeline,
    downsample_pipeline: wgpu::RenderPipeline,
    bloom_horizontal_pipeline: wgpu::RenderPipeline,
    bloom_vertical_pipeline: wgpu::RenderPipeline,
    // New fields for post-processing
    apply_bloom_bind_group_layout: wgpu::BindGroupLayout,
    apply_bloom_pipeline: wgpu::RenderPipeline,
    apply_bloom_bind_group: wgpu::BindGroup,
}

impl BloomEffect {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        texture_bind_group_layout: Arc<wgpu::BindGroupLayout>,
        sampler: Arc<wgpu::Sampler>,
        width: u32,
        height: u32,
        render_texture_view: &wgpu::TextureView,
        bloom_shader: &wgpu::ShaderModule,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let max_level = 5; // Levels: full, 1/2, 1/4, 1/8, 1/16

        let mut bloom_downsample_textures = Vec::new();
        let mut bloom_downsample_views = Vec::new();
        let mut bloom_horizontal_textures = Vec::new();
        let mut bloom_horizontal_views = Vec::new();
        let mut bloom_vertical_textures = Vec::new();
        let mut bloom_vertical_views = Vec::new();
        let mut bloom_downsample_bind_groups = Vec::new();
        let mut bloom_horizontal_bind_groups = Vec::new();
        let mut bloom_vertical_bind_groups = Vec::new();

        // Create bloom textures and bind groups
        for i in 0..max_level {
            let level_width = width >> i;
            let level_height = height >> i;
            let size = wgpu::Extent3d {
                width: level_width.max(1),
                height: level_height.max(1),
                depth_or_array_layers: 1,
            };

            // Downsample texture
            let downsample_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Downsample Texture Level {}", i)),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let downsample_view =
                downsample_tex.create_view(&wgpu::TextureViewDescriptor::default());
            bloom_downsample_textures.push(downsample_tex);
            bloom_downsample_views.push(downsample_view);

            // Horizontal blur texture
            let horizontal_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Horizontal Texture Level {}", i)),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let horizontal_view =
                horizontal_tex.create_view(&wgpu::TextureViewDescriptor::default());
            bloom_horizontal_textures.push(horizontal_tex);
            bloom_horizontal_views.push(horizontal_view);

            // Vertical blur texture
            let vertical_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Vertical Texture Level {}", i)),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let vertical_view = vertical_tex.create_view(&wgpu::TextureViewDescriptor::default());
            bloom_vertical_textures.push(vertical_tex);
            bloom_vertical_views.push(vertical_view);

            // Bind groups
            let input_view = if i == 0 {
                render_texture_view
            } else {
                &bloom_downsample_views[(i - 1) as usize]
            };
            let downsample_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
                label: Some(&format!("Bloom Downsample Bind Group Level {}", i)),
            });
            bloom_downsample_bind_groups.push(downsample_bind_group);

            let horizontal_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &bloom_downsample_views[i as usize],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
                label: Some(&format!("Bloom Horizontal Bind Group Level {}", i)),
            });
            bloom_horizontal_bind_groups.push(horizontal_bind_group);

            let vertical_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &bloom_horizontal_views[i as usize],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
                label: Some(&format!("Bloom Vertical Bind Group Level {}", i)),
            });
            bloom_vertical_bind_groups.push(vertical_bind_group);
        }

        // Create pipelines for bloom passes
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bloom Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let bright_extract_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Bright Extract Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: bloom_shader,
                    entry_point: Some("mip_vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: bloom_shader,
                    entry_point: Some("bright_extract_fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::TextureFormat::Rgba32Float.into())],
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

        let downsample_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Downsample Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: bloom_shader,
                entry_point: Some("mip_vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: bloom_shader,
                entry_point: Some("downsample_fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::TextureFormat::Rgba32Float.into())],
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
                layout: Some(&pipeline_layout),
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
                    targets: &[Some(wgpu::TextureFormat::Rgba32Float.into())],
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
                layout: Some(&pipeline_layout),
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
                    targets: &[Some(wgpu::TextureFormat::Rgba32Float.into())],
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

        // Create apply bloom (post-processing) resources
        let apply_bloom_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Apply Bloom Bind Group Layout"),
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
                            view_dimension: wgpu::TextureViewDimension::D2,
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
            });

        let apply_bloom_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Apply Bloom Pipeline Layout"),
                bind_group_layouts: &[&apply_bloom_bind_group_layout],
                push_constant_ranges: &[],
            });

        let apply_bloom_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Apply Bloom Pipeline"),
            layout: Some(&apply_bloom_pipeline_layout),
            vertex: wgpu::VertexState {
                module: bloom_shader,
                entry_point: Some("apply_vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: bloom_shader,
                entry_point: Some("apply_fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(surface_format.into())],
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

        let apply_bloom_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &apply_bloom_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(render_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bloom_vertical_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&bloom_vertical_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&bloom_vertical_views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&bloom_vertical_views[3]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&bloom_vertical_views[4]),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("apply_bloom_bind_group"),
        });

        Self {
            device,
            queue,
            texture_bind_group_layout,
            sampler,
            max_level,
            bloom_downsample_textures,
            bloom_downsample_views,
            bloom_horizontal_textures,
            bloom_horizontal_views,
            bloom_vertical_textures,
            bloom_vertical_views,
            bloom_downsample_bind_groups,
            bloom_horizontal_bind_groups,
            bloom_vertical_bind_groups,
            bright_extract_pipeline,
            downsample_pipeline,
            bloom_horizontal_pipeline,
            bloom_vertical_pipeline,
            apply_bloom_bind_group_layout,
            apply_bloom_pipeline,
            apply_bloom_bind_group,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32, render_texture_view: &wgpu::TextureView) {
        // Clear existing resources
        self.bloom_downsample_textures.clear();
        self.bloom_downsample_views.clear();
        self.bloom_horizontal_textures.clear();
        self.bloom_horizontal_views.clear();
        self.bloom_vertical_textures.clear();
        self.bloom_vertical_views.clear();
        self.bloom_downsample_bind_groups.clear();
        self.bloom_horizontal_bind_groups.clear();
        self.bloom_vertical_bind_groups.clear();

        // Recreate bloom textures and bind groups
        for i in 0..self.max_level {
            let level_width = width >> i;
            let level_height = height >> i;
            let size = wgpu::Extent3d {
                width: level_width.max(1),
                height: level_height.max(1),
                depth_or_array_layers: 1,
            };

            let downsample_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Downsample Texture Level {}", i)),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let downsample_view =
                downsample_tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.bloom_downsample_textures.push(downsample_tex);
            self.bloom_downsample_views.push(downsample_view);

            let horizontal_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Horizontal Texture Level {}", i)),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let horizontal_view =
                horizontal_tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.bloom_horizontal_textures.push(horizontal_tex);
            self.bloom_horizontal_views.push(horizontal_view);

            let vertical_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Vertical Texture Level {}", i)),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let vertical_view = vertical_tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.bloom_vertical_textures.push(vertical_tex);
            self.bloom_vertical_views.push(vertical_view);

            let input_view = if i == 0 {
                render_texture_view
            } else {
                &self.bloom_downsample_views[(i - 1) as usize]
            };
            let downsample_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
                label: Some(&format!("Bloom Downsample Bind Group Level {}", i)),
            });
            self.bloom_downsample_bind_groups
                .push(downsample_bind_group);

            let horizontal_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.bloom_downsample_views[i as usize],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
                label: Some(&format!("Bloom Horizontal Bind Group Level {}", i)),
            });
            self.bloom_horizontal_bind_groups
                .push(horizontal_bind_group);

            let vertical_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.bloom_horizontal_views[i as usize],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
                label: Some(&format!("Bloom Vertical Bind Group Level {}", i)),
            });
            self.bloom_vertical_bind_groups.push(vertical_bind_group);
        }

        // Recreate apply_bloom_bind_group
        self.apply_bloom_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.apply_bloom_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(render_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_vertical_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_vertical_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_vertical_views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_vertical_views[3]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_vertical_views[4]),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
            label: Some("apply_bloom_bind_group"),
        });
    }

    pub fn render(&self, encoder: &mut wgpu::CommandEncoder) {
        // Level 0: Extract bright parts
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bright Extract Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_downsample_views[0],
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
            rpass.set_pipeline(&self.bright_extract_pipeline);
            rpass.set_bind_group(0, &self.bloom_downsample_bind_groups[0], &[]);
            rpass.draw(0..4, 0..1);
        }

        // Downsample subsequent levels
        for i in 1..self.max_level {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("Bloom Downsample Render Pass Level {}", i)),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_downsample_views[i as usize],
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
            rpass.set_pipeline(&self.downsample_pipeline);
            rpass.set_bind_group(0, &self.bloom_downsample_bind_groups[i as usize], &[]);
            rpass.draw(0..4, 0..1);
        }

        // Apply blur to each level
        for i in 0..self.max_level {
            // Horizontal blur
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(&format!("Bloom Horizontal Render Pass Level {}", i)),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.bloom_horizontal_views[i as usize],
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
                rpass.set_bind_group(0, &self.bloom_horizontal_bind_groups[i as usize], &[]);
                rpass.draw(0..4, 0..1);
            }

            // Vertical blur
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(&format!("Bloom Vertical Render Pass Level {}", i)),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.bloom_vertical_views[i as usize],
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
                rpass.set_bind_group(0, &self.bloom_vertical_bind_groups[i as usize], &[]);
                rpass.draw(0..4, 0..1);
            }
        }
    }

    pub fn apply(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        surface_texture_view: &wgpu::TextureView,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Apply Bloom Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface_texture_view,
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
        rpass.set_pipeline(&self.apply_bloom_pipeline);
        rpass.set_bind_group(0, &self.apply_bloom_bind_group, &[]);
        rpass.draw(0..4, 0..1);
    }
}
