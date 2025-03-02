use std::sync::Arc;

pub struct BloomEffect {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    texture_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    sampler: Arc<wgpu::Sampler>,
    // Textures for each bloom level
    max_level: u32,
    bloom_downsample_textures: Vec<wgpu::Texture>,
    bloom_downsample_views: Vec<wgpu::TextureView>,
    bloom_horizontal_textures: Vec<wgpu::Texture>,
    bloom_horizontal_views: Vec<wgpu::TextureView>,
    bloom_vertical_textures: Vec<wgpu::Texture>,
    bloom_vertical_views: Vec<wgpu::TextureView>,
    // Bind groups for each pass per level
    bloom_downsample_bind_groups: Vec<wgpu::BindGroup>,
    bloom_horizontal_bind_groups: Vec<wgpu::BindGroup>,
    bloom_vertical_bind_groups: Vec<wgpu::BindGroup>,
    // Pipelines (reused across levels)
    bright_extract_pipeline: wgpu::RenderPipeline, // New pipeline to extract bright parts
    downsample_pipeline: wgpu::RenderPipeline,
    bloom_horizontal_pipeline: wgpu::RenderPipeline,
    bloom_vertical_pipeline: wgpu::RenderPipeline,
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
    ) -> Self {
        let max_level = 5; // Levels: full, 1/2, 1/4, 1/8, 1/16

        // Create textures for each level
        let mut bloom_downsample_textures = Vec::new();
        let mut bloom_downsample_views = Vec::new();
        let mut bloom_horizontal_textures = Vec::new();
        let mut bloom_horizontal_views = Vec::new();
        let mut bloom_vertical_textures = Vec::new();
        let mut bloom_vertical_views = Vec::new();
        let mut bloom_downsample_bind_groups = Vec::new();
        let mut bloom_horizontal_bind_groups = Vec::new();
        let mut bloom_vertical_bind_groups = Vec::new();

        for i in 0..max_level {
            let level_width = width >> i; // Divide by 2^i
            let level_height = height >> i;
            let size = wgpu::Extent3d {
                width: level_width.max(1), // Ensure at least 1x1
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
                render_texture_view // Level 0 uses the scene texture
            } else {
                &bloom_downsample_views[(i - 1) as usize] // Downsample from previous level
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

        // Create pipelines (assume shader entries exist or are added)
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
                    entry_point: Some("mip_vs_main"), // Reuse vertex shader
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: bloom_shader,
                    entry_point: Some("bright_extract_fs_main"), // New shader entry
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
                entry_point: Some("downsample_fs_main"), // New shader entry
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
                // Same as original, reused across levels
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
                // Same as original, reused across levels
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
        }
    }

    /// Updates bloom textures and bind groups when the window is resized.
    pub fn update_textures(
        &mut self,
        width: u32,
        height: u32,
        render_texture_view: &wgpu::TextureView,
    ) {
        // Clear existing vectors to drop old resources
        self.bloom_downsample_textures.clear();
        self.bloom_downsample_views.clear();
        self.bloom_horizontal_textures.clear();
        self.bloom_horizontal_views.clear();
        self.bloom_vertical_textures.clear();
        self.bloom_vertical_views.clear();
        self.bloom_downsample_bind_groups.clear();
        self.bloom_horizontal_bind_groups.clear();
        self.bloom_vertical_bind_groups.clear();

        // Recreate textures, views, and bind groups for each level
        for i in 0..self.max_level {
            let level_width = width >> i; // Downsample by 2^i
            let level_height = height >> i;
            let size = wgpu::Extent3d {
                width: level_width.max(1), // Ensure at least 1x1
                height: level_height.max(1),
                depth_or_array_layers: 1,
            };

            // Downsample texture and view
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

            // Horizontal blur texture and view
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

            // Vertical blur texture and view
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

            // Create bind groups
            let input_view = if i == 0 {
                render_texture_view // Level 0 uses the render texture
            } else {
                &self.bloom_downsample_views[(i - 1) as usize] // Downsample from previous level
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
    }

    /// Renders the bloom effect passes.
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

        // Downsample to subsequent levels
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

    // Return all vertical texture views for use in the final pass
    pub fn get_bloom_texture_views(&self) -> &[wgpu::TextureView] {
        &self.bloom_vertical_views
    }
}
