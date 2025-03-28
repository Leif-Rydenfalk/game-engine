use wgpu::{Device, Queue, ComputePipeline, BindGroupLayout, BindGroup, ShaderModuleDescriptor, ShaderSource};
use std::sync::Arc;

/// A renderer for atmospheric effects using compute shaders.
pub struct AtmosphereRenderer {
    device: Arc<Device>,
    compute_pipeline: ComputePipeline,
    texture_bind_group: BindGroup,
    output_bind_group_layout: BindGroupLayout,
    // Store these so we can recreate the pipeline when the shader changes
    texture_bind_group_layout: BindGroupLayout,
    camera_bind_group_layout: BindGroupLayout,
}

impl AtmosphereRenderer {
    /// Creates a new AtmosphereRenderer with resources for atmospheric rendering.
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        camera_bind_group_layout: &BindGroupLayout,
    ) -> Self {
        // Create texture bind group layout and resources
        let (texture_bind_group_layout, texture_bind_group) = 
            Self::create_texture_resources(&device);
            
        // Create output bind group layout
        let output_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
            label: Some("atmosphere_output_bind_group_layout"),
        });

        // Load default embedded shader if none provided
        let shader_module = Self::create_default_shader_module(&device);
        
        // Create compute pipeline
        let compute_pipeline = Self::create_compute_pipeline(
            &device, 
            &texture_bind_group_layout, 
            camera_bind_group_layout,
            &output_bind_group_layout,
            &shader_module,
        );

        Self {
            device,
            compute_pipeline,
            texture_bind_group,
            output_bind_group_layout,
            texture_bind_group_layout,
            camera_bind_group_layout: camera_bind_group_layout.clone(),
        }
    }
    
    /// Creates the default shader module from embedded WGSL code
    fn create_default_shader_module(device: &Device) -> wgpu::ShaderModule {
        device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Atmosphere Compute Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/atmosphere.wgsl").into()),
        })
    }
    
    /// Updates the compute pipeline with a new shader module
    pub fn update_shader(&mut self, shader_module: &wgpu::ShaderModule) {
        // Create compute pipeline with the new shader
        let compute_pipeline = Self::create_compute_pipeline(
            &self.device, 
            &self.texture_bind_group_layout, 
            &self.camera_bind_group_layout,
            &self.output_bind_group_layout,
            shader_module,
        );
        
        // Update the pipeline
        self.compute_pipeline = compute_pipeline;
        
        println!("Reloaded atmosphere compute shader");
    }

    /// Creates texture resources for the atmosphere renderer
    fn create_texture_resources(
        device: &Device,
    ) -> (BindGroupLayout, BindGroup) {
        // Create a dummy 1x1 texture as a placeholder (required by the shader)
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Atmosphere Dummy Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create a sampler with clamping behavior
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Atmosphere Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Define the bind group layout for texture and sampler (group 0 in the shader)
        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("atmosphere_texture_bind_group_layout"),
        });

        // Create the bind group for texture and sampler
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("atmosphere_texture_bind_group"),
        });
        
        (texture_bind_group_layout, texture_bind_group)
    }
    
    /// Creates the compute pipeline for atmospheric rendering
    fn create_compute_pipeline(
        device: &Device,
        texture_bind_group_layout: &BindGroupLayout,
        camera_bind_group_layout: &BindGroupLayout,
        output_bind_group_layout: &BindGroupLayout,
        shader_module: &wgpu::ShaderModule,
    ) -> ComputePipeline {
        // Create the pipeline layout with all bind group layouts
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Atmosphere Compute Pipeline Layout"),
            bind_group_layouts: &[
                texture_bind_group_layout, // Group 0: Texture and sampler
                camera_bind_group_layout,  // Group 1: Camera uniform
                output_bind_group_layout,  // Group 2: Output texture
            ],
            push_constant_ranges: &[],
        });

        // Debug shader module information if available
        #[cfg(debug_assertions)]
        {
            println!("Creating compute pipeline with shader module: {:?}", shader_module);
            println!("Attempting to use entry point: 'main'");
        }

        // Create the compute pipeline
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Atmosphere Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Creates a bind group for the output texture
    pub fn create_output_bind_group(&self, device: &Device, texture_view: &wgpu::TextureView) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.output_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
            ],
            label: Some("atmosphere_output_bind_group"),
        })
    }

    /// Renders the atmosphere using compute shaders
    pub fn render(&self, compute_pass: &mut wgpu::ComputePass, output_bind_group: &BindGroup, width: u32, height: u32) {
        // Set the pipeline and bind groups
        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &self.texture_bind_group, &[]);
        // Camera bind group is set at index 1 externally before calling this method
        compute_pass.set_bind_group(2, output_bind_group, &[]);
        
        // Calculate workgroup counts (e.g., 16x16 threads per workgroup)
        let workgroup_size = 16;
        let work_x = (width + workgroup_size - 1) / workgroup_size;
        let work_y = (height + workgroup_size - 1) / workgroup_size;
        
        // Dispatch the compute shader
        compute_pass.dispatch_workgroups(work_x, work_y, 1);
    }
}