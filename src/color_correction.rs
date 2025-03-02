use std::sync::Arc;

use std::borrow::Cow;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{MemoryHints, SamplerDescriptor, ShaderSource};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorCorrectionUniform {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
}

pub struct ColorCorrectionEffect {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    sampler: Arc<wgpu::Sampler>,
    cc_pipeline: wgpu::RenderPipeline,
    cc_bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl ColorCorrectionEffect {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        input_texture_view: &wgpu::TextureView,
        sampler: Arc<wgpu::Sampler>,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        // Create bind group layout
        let cc_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Color Correction Bind Group Layout"),
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
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create pipeline layout
        let cc_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Color Correction Pipeline Layout"),
            bind_group_layouts: &[&cc_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Load shader
        let cc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Color Correction Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("color_correction.wgsl"))),
        });

        // Create render pipeline
        let cc_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Color Correction Pipeline"),
            layout: Some(&cc_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &cc_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &cc_shader,
                entry_point: Some("fs_main"),
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

        // Create uniform buffer with default values
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Color Correction Uniform Buffer"),
            contents: bytemuck::cast_slice(&[ColorCorrectionUniform {
                brightness: 1.0,
                contrast: 1.0,
                saturation: 1.0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &cc_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("Color Correction Bind Group"),
        });

        Self {
            device,
            queue,
            sampler,
            cc_pipeline,
            cc_bind_group_layout,
            uniform_buffer,
            bind_group,
        }
    }

    pub fn resize(&mut self, input_texture_view: &wgpu::TextureView) {
        self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.cc_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("Color Correction Bind Group"),
        });
    }

    pub fn apply(&self, encoder: &mut wgpu::CommandEncoder, target_view: &wgpu::TextureView) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Color Correction Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
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
        rpass.set_pipeline(&self.cc_pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..4, 0..1);
    }

    pub fn update_uniform(&self, uniform: ColorCorrectionUniform) {
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }
}
