use std::sync::Arc;
use wgpu::{util::DeviceExt, PipelineCompilationOptions};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomSettings {
    min_brightness: f32,
    max_brightness: f32,
    blur_radius: f32,
    blur_type: u32, // 0 = Gaussian, 1 = Box, etc.
}

pub struct BloomEffect {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    texture_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    sampler: Arc<wgpu::Sampler>,
    max_level: u32,
    downsample_texture: wgpu::Texture,
    downsample_views: Vec<wgpu::TextureView>,
    horizontal_blur_texture: wgpu::Texture,
    horizontal_blur_views: Vec<wgpu::TextureView>,
    vertical_blur_texture: wgpu::Texture,
    vertical_blur_views: Vec<wgpu::TextureView>,
    settings_buffer: wgpu::Buffer,
    downsample_bind_groups: Vec<wgpu::BindGroup>,
    horizontal_blur_bind_groups: Vec<wgpu::BindGroup>,
    vertical_blur_bind_groups: Vec<wgpu::BindGroup>,
    prefilter_pipeline: wgpu::ComputePipeline,
    downsample_pipeline: wgpu::ComputePipeline,
    horizontal_blur_pipeline: wgpu::ComputePipeline,
    vertical_blur_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::ComputePipeline,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    full_width: u32,
    full_height: u32,
    half_width: u32,
    half_height: u32,
    group0_layout: wgpu::BindGroupLayout,
    group1_layout: wgpu::BindGroupLayout,
    group2_layout: wgpu::BindGroupLayout,
    settings_bind_group: wgpu::BindGroup,
}

impl BloomEffect {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        texture_bind_group_layout: Arc<wgpu::BindGroupLayout>,
        sampler: Arc<wgpu::Sampler>,
        width: u32,
        height: u32,
        _render_texture_view: &wgpu::TextureView,
        bloom_shader: &wgpu::ShaderModule,
    ) -> Self {
        let max_level = 8;
        let half_width = width / 2;
        let half_height = height / 2;

        let downsample_texture = create_mip_texture(
            &device,
            half_width,
            half_height,
            max_level,
            "Downsample Texture",
        );
        let horizontal_blur_texture = create_mip_texture(
            &device,
            half_width,
            half_height,
            max_level,
            "Horizontal Blur Texture",
        );
        let vertical_blur_texture = create_mip_texture(
            &device,
            half_width,
            half_height,
            max_level,
            "Vertical Blur Texture",
        );

        let downsample_views = create_mip_views(&downsample_texture, max_level);
        let horizontal_blur_views = create_mip_views(&horizontal_blur_texture, max_level);
        let vertical_blur_views = create_mip_views(&vertical_blur_texture, max_level);

        let settings = BloomSettings {
            min_brightness: 0.9,
            max_brightness: 1.0,
            blur_radius: 1.0,
            blur_type: 0,
        };
        let settings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bloom Settings Buffer"),
            contents: bytemuck::cast_slice(&[settings]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Group 0: Uniform buffer
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Settings Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Group 1: Texture and storage texture
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let group2_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bloom Textures Bind Group Layout"),
            entries: &{
                let mut entries = (0..8)
                    .map(|i| wgpu::BindGroupLayoutEntry {
                        binding: i,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>();
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                });
                entries
            },
        });
        let downsample_bind_groups = (1..max_level)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &group1_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &downsample_views[i as usize - 1],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &downsample_views[i as usize],
                            ),
                        },
                    ],
                    label: Some(&format!("Downsample Group 1 Bind Group Mip {}", i)),
                })
            })
            .collect::<Vec<_>>();

        let horizontal_blur_bind_groups = (0..max_level)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &group1_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &downsample_views[i as usize],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &horizontal_blur_views[i as usize],
                            ),
                        },
                    ],
                    label: Some(&format!("Horizontal Blur Group 1 Bind Group Mip {}", i)),
                })
            })
            .collect::<Vec<_>>();

        let vertical_blur_bind_groups = (0..max_level)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &group1_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &horizontal_blur_views[i as usize],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &vertical_blur_views[i as usize],
                            ),
                        },
                    ],
                    label: Some(&format!("Vertical Blur Group 1 Bind Group Mip {}", i)),
                })
            })
            .collect::<Vec<_>>();

        let texture_binding = wgpu::BindGroupLayoutEntry {
            binding: 0, // Will be overridden
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
            },
            count: None,
        };

        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        ..texture_binding
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        ..texture_binding
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        ..texture_binding
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        ..texture_binding
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        ..texture_binding
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        ..texture_binding
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        ..texture_binding
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        ..texture_binding
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        ..texture_binding
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let settings_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &group0_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: settings_buffer.as_entire_binding(),
            }],
            label: Some("Settings Bind Group"),
        });

        let prefilter_pipeline = create_compute_pipeline(
            &device,
            &[&group0_layout, &group1_layout],
            bloom_shader,
            "prefilter_main",
            "Prefilter Pipeline",
        );
        let downsample_pipeline = create_compute_pipeline(
            &device,
            &[&group0_layout, &group1_layout],
            bloom_shader,
            "downsample_main",
            "Downsample Pipeline",
        );

        let horizontal_blur_pipeline = create_compute_pipeline(
            &device,
            &[&group0_layout, &group1_layout],
            bloom_shader,
            "horizontal_blur_main",
            "Horizontal Blur Pipeline",
        );

        let vertical_blur_pipeline = create_compute_pipeline(
            &device,
            &[&group0_layout, &group1_layout],
            bloom_shader,
            "vertical_blur_main",
            "Vertical Blur Pipeline",
        );
        let composite_pipeline = create_compute_pipeline(
            &device,
            &[&group0_layout, &group1_layout, &group2_layout],
            bloom_shader,
            "composite_main",
            "Composite Pipeline",
        );

        Self {
            device,
            queue,
            texture_bind_group_layout,
            sampler,
            max_level,
            downsample_texture,
            downsample_views,
            horizontal_blur_texture,
            horizontal_blur_views,
            vertical_blur_texture,
            vertical_blur_views,
            settings_buffer,
            downsample_bind_groups,
            horizontal_blur_bind_groups,
            vertical_blur_bind_groups,
            prefilter_pipeline,
            downsample_pipeline,
            horizontal_blur_pipeline,
            vertical_blur_pipeline,
            composite_pipeline,
            composite_bind_group_layout,
            full_width: width,
            full_height: height,
            half_width,
            half_height,
            group0_layout,
            group1_layout,
            group2_layout,
            settings_bind_group,
        }
    }
    pub fn resize(&mut self, width: u32, height: u32, _render_texture_view: &wgpu::TextureView) {
        self.full_width = width;
        self.full_height = height;
        self.half_width = width / 2;
        self.half_height = height / 2;

        self.downsample_texture = create_mip_texture(
            &self.device,
            self.half_width,
            self.half_height,
            self.max_level,
            "Downsample Texture",
        );
        self.horizontal_blur_texture = create_mip_texture(
            &self.device,
            self.half_width,
            self.half_height,
            self.max_level,
            "Horizontal Blur Texture",
        );
        self.vertical_blur_texture = create_mip_texture(
            &self.device,
            self.half_width,
            self.half_height,
            self.max_level,
            "Vertical Blur Texture",
        );

        self.downsample_views = create_mip_views(&self.downsample_texture, self.max_level);
        self.horizontal_blur_views =
            create_mip_views(&self.horizontal_blur_texture, self.max_level);
        self.vertical_blur_views = create_mip_views(&self.vertical_blur_texture, self.max_level);

        self.downsample_bind_groups = (1..self.max_level)
            .map(|i| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.group1_layout, // Corrected from texture_bind_group_layout
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.downsample_views[i as usize - 1],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.downsample_views[i as usize],
                            ),
                        },
                    ],
                    label: Some(&format!("Downsample Texture Bind Group Mip {}", i)),
                })
            })
            .collect();

        self.horizontal_blur_bind_groups = (0..self.max_level)
            .map(|i| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.group1_layout, // Corrected from texture_bind_group_layout
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.downsample_views[i as usize],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.horizontal_blur_views[i as usize],
                            ),
                        },
                    ],
                    label: Some(&format!("Horizontal Blur Texture Bind Group Mip {}", i)),
                })
            })
            .collect();

        self.vertical_blur_bind_groups = (0..self.max_level)
            .map(|i| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.group1_layout, // Corrected from texture_bind_group_layout
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.horizontal_blur_views[i as usize],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.vertical_blur_views[i as usize],
                            ),
                        },
                    ],
                    label: Some(&format!("Vertical Blur Texture Bind Group Mip {}", i)),
                })
            })
            .collect();
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_texture_view: &wgpu::TextureView,
    ) {
        // Create the prefilter bind group
        let prefilter_group1_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.group1_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(scene_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.downsample_views[0]),
                    },
                ],
                label: Some("Prefilter Group 1 Bind Group"),
            });

        // Prefilter pass
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prefilter Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.prefilter_pipeline);
            cpass.set_bind_group(0, &self.settings_bind_group, &[]);
            cpass.set_bind_group(1, &prefilter_group1_bind_group, &[]);
            let dispatch_x = (self.half_width + 7) / 8;
            let dispatch_y = (self.half_height + 7) / 8;
            cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // Downsample pass (corrected)
        for i in 1..self.max_level {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Downsample Compute Pass Mip {}", i)),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.downsample_pipeline);
            cpass.set_bind_group(0, &self.settings_bind_group, &[]);
            cpass.set_bind_group(1, &self.downsample_bind_groups[i as usize - 1], &[]);
            let mip_width = (self.half_width >> i).max(1);
            let mip_height = (self.half_height >> i).max(1);
            let dispatch_x = (mip_width + 7) / 8;
            let dispatch_y = (mip_height + 7) / 8;
            cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // Blur passes
        for i in 0..self.max_level {
            let mip_width = (self.half_width >> i).max(1);
            let mip_height = (self.half_height >> i).max(1);
            let dispatch_x = (mip_width + 7) / 8;
            let dispatch_y = (mip_height + 7) / 8;

            // Horizontal blur
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Horizontal Blur Compute Pass Mip {}", i)),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.horizontal_blur_pipeline);
                cpass.set_bind_group(0, &self.settings_bind_group, &[]);
                cpass.set_bind_group(1, &self.horizontal_blur_bind_groups[i as usize], &[]);
                cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            // Vertical blur
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Vertical Blur Compute Pass Mip {}", i)),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.vertical_blur_pipeline);
                cpass.set_bind_group(0, &self.settings_bind_group, &[]);
                cpass.set_bind_group(1, &self.vertical_blur_bind_groups[i as usize], &[]);
                cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
        }
    }

    pub fn apply(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        scene_texture_view: &wgpu::TextureView,
    ) {
        let composite_group1_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.group1_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(scene_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(target_view),
                    },
                ],
                label: Some("Composite Group 1 Bind Group"),
            });

        let composite_group2_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.group2_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.vertical_blur_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.vertical_blur_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.vertical_blur_views[2]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.vertical_blur_views[3]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&self.vertical_blur_views[4]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&self.vertical_blur_views[5]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&self.vertical_blur_views[6]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(&self.vertical_blur_views[7]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
                label: Some("Composite Group 2 Bind Group"),
            });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Composite Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.composite_pipeline);
        cpass.set_bind_group(0, &self.settings_bind_group, &[]);
        cpass.set_bind_group(1, &composite_group1_bind_group, &[]);
        cpass.set_bind_group(2, &composite_group2_bind_group, &[]);
        let dispatch_x = (self.full_width + 7) / 8;
        let dispatch_y = (self.full_height + 7) / 8;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
}

fn create_mip_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    mip_count: u32,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: mip_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    })
}

fn create_mip_views(texture: &wgpu::Texture, mip_count: u32) -> Vec<wgpu::TextureView> {
    (0..mip_count)
        .map(|level| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Mip {}", level)),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        })
        .collect()
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
    shader: &wgpu::ShaderModule,
    entry_point: &str,
    label: &str,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts,
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}
