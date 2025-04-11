use std::sync::Arc;
use wgpu::{
    util::DeviceExt, BindGroup, BindGroupLayout, Device, Queue, RenderPipeline, Sampler,
    ShaderModule, TextureView,
};

// Trait for all post-processing effects
pub trait PostProcessingEffect {
    fn apply(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_texture_view: &wgpu::TextureView,
        output_texture_view: &wgpu::TextureView,
    );

    fn resize(&mut self, width: u32, height: u32);

    fn is_enabled(&self) -> bool;
    fn set_enabled(&mut self, enabled: bool);

    fn name(&self) -> &str;
}

// Main post-processing pipeline that manages all effects
pub struct PostProcessingPipeline {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    ping_texture: wgpu::Texture,
    ping_texture_view: wgpu::TextureView,
    pong_texture: wgpu::Texture,
    pong_texture_view: wgpu::TextureView,
    effects: Vec<Box<dyn PostProcessingEffect>>,
    texture_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    sampler: Arc<wgpu::Sampler>,
    width: u32,
    height: u32,
}

impl PostProcessingPipeline {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        width: u32,
        height: u32,
        texture_bind_group_layout: Arc<wgpu::BindGroupLayout>,
        sampler: Arc<wgpu::Sampler>,
    ) -> Self {
        // Create ping-pong textures (both HDR Rgba32Float format)
        let (ping_texture, ping_texture_view) = Self::create_texture(
            &device,
            width,
            height,
            "Ping Post-Processing Texture",
            wgpu::TextureFormat::Rgba32Float,
        );

        let (pong_texture, pong_texture_view) = Self::create_texture(
            &device,
            width,
            height,
            "Pong Post-Processing Texture",
            wgpu::TextureFormat::Rgba32Float,
        );

        Self {
            device,
            queue,
            ping_texture,
            ping_texture_view,
            pong_texture,
            pong_texture_view,
            effects: Vec::new(),
            texture_bind_group_layout,
            sampler,
            width,
            height,
        }
    }

    // Add an effect to the pipeline
    pub fn add_effect(&mut self, effect: Box<dyn PostProcessingEffect>) {
        self.effects.push(effect);
    }

    // Apply all enabled effects in sequence
    pub fn apply_effects(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_texture_view: &wgpu::TextureView,
        output_texture_view: &wgpu::TextureView,
    ) {
        // If no effects are enabled, just copy the input to output
        if self.effects.iter().all(|effect| !effect.is_enabled()) {
            Self::copy_texture(
                encoder,
                input_texture_view,
                output_texture_view,
                self.width,
                self.height,
            );
            return;
        }

        // First enabled effect reads from input, writes to ping
        let mut reading_from_input = true;
        let mut writing_to_ping = true;
        let mut effect_applied = false;

        for effect in self.effects.iter().filter(|e| e.is_enabled()) {
            if !effect_applied {
                // First effect
                effect.apply(
                    encoder,
                    input_texture_view,
                    if writing_to_ping {
                        &self.ping_texture_view
                    } else {
                        &self.pong_texture_view
                    },
                );
                reading_from_input = false;
                effect_applied = true;
            } else {
                // Subsequent effects
                effect.apply(
                    encoder,
                    if reading_from_input {
                        input_texture_view
                    } else if writing_to_ping {
                        &self.pong_texture_view
                    } else {
                        &self.ping_texture_view
                    },
                    if writing_to_ping {
                        &self.ping_texture_view
                    } else {
                        &self.pong_texture_view
                    },
                );
            }
            // Swap ping and pong roles
            writing_to_ping = !writing_to_ping;
        }

        // Copy the result of the last effect to the output
        if effect_applied {
            // The last write was to the opposite of writing_to_ping since we flipped it
            let last_texture_view = if writing_to_ping {
                &self.pong_texture_view
            } else {
                &self.ping_texture_view
            };

            Self::copy_texture(
                encoder,
                last_texture_view,
                output_texture_view,
                self.width,
                self.height,
            );
        } else {
            // No effects were applied, copy input to output
            Self::copy_texture(
                encoder,
                input_texture_view,
                output_texture_view,
                self.width,
                self.height,
            );
        }
    }

    // Resize all textures and effects
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;

        // Recreate ping-pong textures
        let (ping_texture, ping_texture_view) = Self::create_texture(
            &self.device,
            width,
            height,
            "Ping Post-Processing Texture",
            wgpu::TextureFormat::Rgba32Float,
        );

        let (pong_texture, pong_texture_view) = Self::create_texture(
            &self.device,
            width,
            height,
            "Pong Post-Processing Texture",
            wgpu::TextureFormat::Rgba32Float,
        );

        self.ping_texture = ping_texture;
        self.ping_texture_view = ping_texture_view;
        self.pong_texture = pong_texture;
        self.pong_texture_view = pong_texture_view;

        // Resize all effects
        for effect in &mut self.effects {
            effect.resize(width, height);
        }
    }

    // Helper to create a texture with the specified format
    fn create_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        label: &str,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        (texture, texture_view)
    }

    // Helper to copy textures using a blit operation
    fn copy_texture(
        encoder: &mut wgpu::CommandEncoder,
        source_view: &wgpu::TextureView,
        destination_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        // In wgpu, we can use a compute shader to copy between textures of the same format

        // First, get access to the textures
        let source_texture = source_view;
        let destination_texture = destination_view;

        // Use copyTextureToTexture for direct copying between textures
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &source_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &destination_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    // Alternative implementation using a simple compute shader for copying
    fn copy_texture_compute(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source_view: &wgpu::TextureView,
        destination_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        // Create a simple compute shader for copying
        let copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Texture Copy Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                r#"
            @group(0) @binding(0) var inputTexture: texture_2d<f32>;
            @group(0) @binding(1) var outputTexture: storage_texture_2d<rgba32float, write>;
            
            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let dims = textureDimensions(inputTexture);
                if (global_id.x >= dims.x || global_id.y >= dims.y) {
                    return;
                }
                
                let texel = textureLoad(inputTexture, vec2<i32>(global_id.xy), 0);
                textureStore(outputTexture, vec2<i32>(global_id.xy), texel);
            }
        "#,
            )),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Copy Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
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

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Texture Copy Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Texture Copy Pipeline"),
            layout: Some(&pipeline_layout),
            module: &copy_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Copy Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(destination_view),
                },
            ],
        });

        // Execute the compute pass
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Texture Copy Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_count_x = (width + 7) / 8;
        let workgroup_count_y = (height + 7) / 8;
        compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
    }
}

// Example implementation of BloomEffect using the new architecture
pub struct BloomEffect {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    enabled: bool,
    name: String,

    // Existing bloom effect resources
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

    // Additional resources for the effect
    composite_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    sampler: Arc<wgpu::Sampler>,
    group0_layout: wgpu::BindGroupLayout,
    group1_layout: wgpu::BindGroupLayout,
    group2_layout: wgpu::BindGroupLayout,
    settings_bind_group: wgpu::BindGroup,
    full_width: u32,
    full_height: u32,
    half_width: u32,
    half_height: u32,
}

impl PostProcessingEffect for BloomEffect {
    fn apply(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_texture_view: &wgpu::TextureView,
        output_texture_view: &wgpu::TextureView,
    ) {
        if !self.enabled {
            // If disabled, just copy the input to output
            PostProcessingPipeline::copy_texture(
                encoder,
                input_texture_view,
                output_texture_view,
                self.full_width,
                self.full_height,
            );
            return;
        }

        // Create prefilter bind group for this specific input
        let prefilter_group1_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.group1_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_texture_view),
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

        // Downsample pass
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

        // Composite pass - combine original image with bloom
        let composite_group1_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.group1_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(output_texture_view),
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

    fn resize(&mut self, width: u32, height: u32) {
        self.full_width = width;
        self.full_height = height;
        self.half_width = width / 2;
        self.half_height = height / 2;

        // Recreate bloom textures and bind groups
        // (Implementation similar to original BloomEffect resize method)
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// Example implementation of ColorCorrectionEffect using the new architecture
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorCorrectionUniform {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    _padding: f32,
}

pub struct ColorCorrectionEffect {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    enabled: bool,
    width: u32,
    height: u32,
    name: String,
}

impl ColorCorrectionEffect {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        shader: &wgpu::ShaderModule,
        width: u32,
        height: u32,
    ) -> Self {
        // Create uniform buffer
        let uniform = ColorCorrectionUniform {
            brightness: 1.0,
            contrast: 1.0,
            saturation: 1.0,
            _padding: 0.0,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Color Correction Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Color Correction Bind Group Layout"),
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
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Color Correction Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Color Correction Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create initial bind group with dummy textures (will be updated in apply)
        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let dummy_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Color Correction Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&dummy_view),
                },
            ],
        });

        Self {
            device,
            queue,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            pipeline,
            enabled: true,
            width,
            height,
            name: "Color Correction".to_string(),
        }
    }

    pub fn update_uniform(&mut self, uniform: ColorCorrectionUniform) {
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }
}

impl PostProcessingEffect for ColorCorrectionEffect {
    fn apply(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_texture_view: &wgpu::TextureView,
        output_texture_view: &wgpu::TextureView,
    ) {
        if !self.enabled {
            // If disabled, just copy the input to output
            PostProcessingPipeline::copy_texture(
                encoder,
                input_texture_view,
                output_texture_view,
                self.width,
                self.height,
            );
            return;
        }

        // Create a bind group for this specific input/output texture pair
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(input_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(output_texture_view),
                },
            ],
            label: Some("Color Correction Dynamic Bind Group"),
        });

        // Apply color correction
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Color Correction Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let dispatch_x = (self.width + 7) / 8;
        let dispatch_y = (self.height + 7) / 8;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn name(&self) -> &str {
        &self.name
    }
}
