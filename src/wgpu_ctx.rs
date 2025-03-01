use crate::vertex::{create_vertex_buffer_layout, VERTEX_INDEX_LIST, VERTEX_SQUARE};
use crate::{Model, ModelInstance, RgbaImg, Transform};
use cgmath::{Matrix4, SquareMatrix};
use hecs::World;
use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::MemoryHints::Performance;
use wgpu::{SamplerDescriptor, ShaderSource};
use winit::window::Window;

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    position: [f32; 3],
    _padding: f32,
}

pub struct WgpuCtx<'window> {
    surface: wgpu::Surface<'window>,
    surface_config: wgpu::SurfaceConfiguration,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_index_buffer: wgpu::Buffer,
    texture: wgpu::Texture,
    texture_image: RgbaImg,
    texture_size: wgpu::Extent3d,
    sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,
    models: Vec<Model>,
    texture_bind_group_layout: wgpu::BindGroupLayout,

    // Bloom textures
    bloom_mip_texture: wgpu::Texture,
    bloom_mip_texture_view: wgpu::TextureView,
    bloom_horizontal_texture: wgpu::Texture,
    bloom_horizontal_texture_view: wgpu::TextureView,
    bloom_vertical_texture: wgpu::Texture,
    bloom_vertical_texture_view: wgpu::TextureView,

    // Bloom compute pipelines and bind groups
    bloom_mip_compute_pipeline: wgpu::ComputePipeline,
    bloom_mip_bind_group: wgpu::BindGroup,
    bloom_horizontal_compute_pipeline: wgpu::ComputePipeline,
    bloom_horizontal_bind_group: wgpu::BindGroup,
    bloom_vertical_compute_pipeline: wgpu::ComputePipeline,
    bloom_vertical_bind_group: wgpu::BindGroup,

    // Updated post-process bind group layout to include both render_texture and bloom_vertical_texture
    post_process_bind_group_layout: wgpu::BindGroupLayout,

    // New fields for post-processing
    render_texture: wgpu::Texture,
    render_texture_view: wgpu::TextureView,
    output_texture: wgpu::Texture,
    output_texture_view: wgpu::TextureView,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    compute_bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    post_process_pipeline: wgpu::RenderPipeline,
    post_process_bind_group: wgpu::BindGroup,
}

impl<'window> WgpuCtx<'window> {
    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        (depth_texture, depth_texture_view)
    }

    pub async fn new_async(window: Arc<Window>) -> WgpuCtx<'window> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                // Request an adapter which can render to our surface
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find an appropriate adapter");
        println!(
            "Max storage buffers per shader stage: {}",
            adapter.limits().max_storage_buffers_per_shader_stage
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 2,
                        ..wgpu::Limits::default()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        // 获取窗口内部物理像素尺寸（没有标题栏）
        let size = window.inner_size();
        // 至少（w = 1, h = 1），否则Wgpu会panic
        let width = size.width.max(1);
        let height = size.height.max(1);
        // 获取一个默认配置
        let surface_config = surface.get_default_config(&adapter, width, height).unwrap();
        // 完成首次配置
        surface.configure(&device, &surface_config);

        let bytes: &[u8] = bytemuck::cast_slice(&VERTEX_SQUARE);
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage: wgpu::BufferUsages::VERTEX,
        });
        // 将顶点索引数据转为字节数据
        let vertex_index_bytes = bytemuck::cast_slice(&VERTEX_INDEX_LIST);
        // 创建顶点索引缓冲数据
        let vertex_index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: vertex_index_bytes,
            usage: wgpu::BufferUsages::INDEX, // 注意，usage字段使用INDEX枚举，表明是顶点索引
        });

        // 构造图片对象（这里为了代码简洁，我们假设图片加载没有问题，直接unwrap，请读者务必保证图片加载正确性）
        let img = RgbaImg::new("./assets/images/example-img.png").unwrap();
        // 纹理是以3D形式存储，如果想要表示2D纹理，只需要将下方的深度字段设置为1
        let texture_size = wgpu::Extent3d {
            width: img.width, // 图片的宽高
            height: img.height,
            depth_or_array_layers: 1, // <-- 设置为1表示2D纹理
        };
        // 构造Texture实例
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            // size字段用于表达纹理的基本尺寸结构（宽、高以及深度）
            size: texture_size,
            mip_level_count: 1, // 后面会详细介绍此字段
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // 大多数图像都是使用 sRGB 来存储的，我们需要在这里指定。
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // TEXTURE_BINDING 表示我们要在着色器中使用这个纹理。
            // COPY_DST 表示我们能将数据复制到这个纹理上。
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // 创建采样器
        let sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear, // Enable bilinear interpolation for magnification
            min_filter: wgpu::FilterMode::Linear, // Enable bilinear interpolation for minification
            mipmap_filter: wgpu::FilterMode::Linear, // Enable linear interpolation between mip levels (if used)
            ..Default::default()
        });

        // 创建绑定组布局
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    // 注意 Filtering 需要和上面 sample_type filterable: true 保持一致
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: None,
        });
        // 创建绑定组
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });

        // Create camera uniform buffer
        let camera_uniform = CameraUniform {
            view_proj: Matrix4::identity().into(),
            ..Default::default()
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create camera bind group layout
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        // Create camera bind group
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        // Update pipeline layout to include camera
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline =
            create_pipeline(&device, surface_config.format, &render_pipeline_layout);

        // Create depth texture
        let (depth_texture, depth_texture_view) =
            Self::create_depth_texture(&device, &surface_config);

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                ],
                label: Some("texture_bind_group_layout"),
            });

        // Create render texture (scene rendering target)
        let render_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Texture"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_config.format, // Typically Rgba8UnormSrgb
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let render_texture_view =
            render_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create output texture (compute shader output)
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output Texture"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float, // Linear space for storage
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let output_texture_view =
            output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create compute bind group layout
        // In WgpuCtx::new_async, replace the compute bind group layout
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        // Create compute bind group
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&render_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&output_texture_view),
                },
            ],
            label: Some("compute_bind_group"),
        });

        // Define and create compute shader
        const COMPUTE_SHADER_SOURCE: &str = include_str!("post_processing.txt");

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(COMPUTE_SHADER_SOURCE)),
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create bloom_mip_texture
        let bloom_mip_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Mip Texture"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bloom_mip_texture_view =
            bloom_mip_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create bloom_horizontal_texture
        let bloom_horizontal_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Horizontal Texture"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bloom_horizontal_texture_view =
            bloom_horizontal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create bloom_vertical_texture
        let bloom_vertical_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Vertical Texture"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bloom_vertical_texture_view =
            bloom_vertical_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bloom_mip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Mip Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("bloom_mip.txt"))),
        });

        let bloom_mip_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom Mip Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let bloom_mip_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bloom_mip_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&render_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bloom_mip_texture_view),
                },
            ],
            label: Some("bloom_mip_bind_group"),
        });

        let bloom_mip_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Mip Pipeline Layout"),
                bind_group_layouts: &[&bloom_mip_bind_group_layout],
                push_constant_ranges: &[],
            });

        let bloom_mip_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Bloom Mip Compute Pipeline"),
                layout: Some(&bloom_mip_pipeline_layout),
                module: &bloom_mip_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bloom_horizontal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Horizontal Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("bloom_horizontal.txt"))),
        });

        let bloom_horizontal_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom Horizontal Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let bloom_horizontal_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bloom_horizontal_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bloom_mip_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bloom_horizontal_texture_view),
                },
            ],
            label: Some("bloom_horizontal_bind_group"),
        });

        let bloom_horizontal_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Horizontal Pipeline Layout"),
                bind_group_layouts: &[&bloom_horizontal_bind_group_layout],
                push_constant_ranges: &[],
            });

        let bloom_horizontal_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Bloom Horizontal Compute Pipeline"),
                layout: Some(&bloom_horizontal_pipeline_layout),
                module: &bloom_horizontal_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bloom_vertical_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Vertical Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("bloom_vertical.txt"))),
        });

        let bloom_vertical_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom Vertical Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let bloom_vertical_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bloom_vertical_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bloom_horizontal_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bloom_vertical_texture_view),
                },
            ],
            label: Some("bloom_vertical_bind_group"),
        });

        let bloom_vertical_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Vertical Pipeline Layout"),
                bind_group_layouts: &[&bloom_vertical_bind_group_layout],
                push_constant_ranges: &[],
            });

        let bloom_vertical_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Bloom Vertical Compute Pipeline"),
                layout: Some(&bloom_vertical_pipeline_layout),
                module: &bloom_vertical_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let post_process_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Post Process Bind Group Layout"),
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
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let post_process_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &post_process_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&render_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bloom_vertical_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("post_process_bind_group"),
        });

        let post_process_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Post Process Pipeline Layout"),
                bind_group_layouts: &[&post_process_bind_group_layout],
                push_constant_ranges: &[],
            });

        let post_process_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Post Process Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("apply_bloom.txt"))),
        });

        let post_process_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Post Process Pipeline"),
                layout: Some(&post_process_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &post_process_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &post_process_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(surface_config.format.into())],
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

        WgpuCtx {
            surface,
            surface_config,
            adapter,
            device,
            queue,
            render_pipeline,
            vertex_buffer,
            vertex_index_buffer,
            texture,
            texture_image: img,
            texture_size,
            sampler,
            bind_group,
            camera_buffer,
            camera_bind_group,
            depth_texture,
            depth_texture_view,
            models: Vec::new(),
            bloom_mip_texture,
            bloom_mip_texture_view,
            bloom_horizontal_texture,
            bloom_horizontal_texture_view,
            bloom_vertical_texture,
            bloom_vertical_texture_view,
            bloom_mip_compute_pipeline,
            bloom_mip_bind_group,
            bloom_horizontal_compute_pipeline,
            bloom_horizontal_bind_group,
            bloom_vertical_compute_pipeline,
            bloom_vertical_bind_group,
            post_process_bind_group_layout,
            post_process_pipeline,
            post_process_bind_group,
            texture_bind_group_layout,
            render_texture,
            render_texture_view,
            output_texture,
            output_texture_view,
            compute_bind_group_layout,
            compute_bind_group,
            compute_pipeline,
        }
    }

    pub fn load_model<P: AsRef<Path>>(&mut self, path: P) -> Option<usize> {
        if let Some(mut model) = Model::load(&self.device, &self.queue, path) {
            // Create bind groups for all materials
            model.create_bind_groups(&self.device, &self.texture_bind_group_layout);

            // Upload textures to GPU
            model.upload_textures(&self.queue);

            // Add model to collection and return its index
            let index = self.models.len();
            self.models.push(model);
            Some(index)
        } else {
            None
        }
    }

    pub fn update_camera_uniform(
        &mut self,
        view_proj: Matrix4<f32>,
        view: Matrix4<f32>,
        position: [f32; 3],
    ) {
        let camera_uniform = CameraUniform {
            view_proj: view_proj.into(),
            view: view.into(),
            position,
            _padding: Default::default(),
        };
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );
    }

    pub fn new(window: Arc<Window>) -> WgpuCtx<'window> {
        pollster::block_on(WgpuCtx::new_async(window))
    }

    pub fn resize(&mut self, new_size: (u32, u32)) {
        let (width, height) = new_size;
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);

        // Recreate depth texture
        let (depth_texture, depth_texture_view) =
            Self::create_depth_texture(&self.device, &self.surface_config);
        self.depth_texture = depth_texture;
        self.depth_texture_view = depth_texture_view;

        // Recreate render_texture
        self.render_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Texture"),
            size: wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.render_texture_view = self
            .render_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Recreate bloom_mip_texture
        self.bloom_mip_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Mip Texture"),
            size: wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.bloom_mip_texture_view = self
            .bloom_mip_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Recreate bloom_horizontal_texture
        self.bloom_horizontal_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Horizontal Texture"),
            size: wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.bloom_horizontal_texture_view = self
            .bloom_horizontal_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Recreate bloom_vertical_texture
        self.bloom_vertical_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Vertical Texture"),
            size: wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.bloom_vertical_texture_view = self
            .bloom_vertical_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Recreate bloom bind groups
        self.bloom_mip_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bloom_mip_compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.render_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_mip_texture_view),
                },
            ],
            label: Some("bloom_mip_bind_group"),
        });

        self.bloom_horizontal_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self
                    .bloom_horizontal_compute_pipeline
                    .get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.bloom_mip_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.bloom_horizontal_texture_view,
                        ),
                    },
                ],
                label: Some("bloom_horizontal_bind_group"),
            });

        self.bloom_vertical_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self
                    .bloom_vertical_compute_pipeline
                    .get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.bloom_horizontal_texture_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.bloom_vertical_texture_view,
                        ),
                    },
                ],
                label: Some("bloom_vertical_bind_group"),
            });

        // Recreate post_process_bind_group
        self.post_process_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.post_process_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.render_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_vertical_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
            label: Some("post_process_bind_group"),
        });

        // Remove or update output_texture recreation if no longer needed
    }

    pub fn draw(&mut self, world: &World) {
        let surface_texture = self
            .surface
            .get_current_texture()
            .expect("Failed to acquire next swap chain texture");
        let surface_texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Step 1: Render the scene to render_texture (unchanged)
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Scene Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.render_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_bind_group(1, &self.camera_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_index_buffer(
                self.vertex_index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            rpass.draw_indexed(0..VERTEX_INDEX_LIST.len() as u32, 0, 0..1);

            for (_, (transform, model_instance)) in
                world.query::<(&Transform, &ModelInstance)>().iter()
            {
                if let Some(model) = self.models.get(model_instance.model) {
                    for mesh in &model.meshes {
                        if let Some(material_index) = mesh.material_index {
                            if let Some(material) = model.materials.get(material_index) {
                                if let Some(bind_group) = &material.bind_group {
                                    rpass.set_bind_group(0, bind_group, &[]);
                                    rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                    rpass.set_index_buffer(
                                        mesh.index_buffer.slice(..),
                                        wgpu::IndexFormat::Uint32,
                                    );
                                    rpass.draw_indexed(0..mesh.num_elements, 0, 0..1);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Step 2: Generate bloom_mip_texture
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bloom Mip Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bloom_mip_compute_pipeline);
            cpass.set_bind_group(0, &self.bloom_mip_bind_group, &[]);
            let workgroup_size = 8;
            let workgroup_count_x =
                (self.surface_config.width + workgroup_size - 1) / workgroup_size;
            let workgroup_count_y =
                (self.surface_config.height + workgroup_size - 1) / workgroup_size;
            cpass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
        }

        // Step 3: Apply horizontal blur
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bloom Horizontal Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bloom_horizontal_compute_pipeline);
            cpass.set_bind_group(0, &self.bloom_horizontal_bind_group, &[]);
            let workgroup_size = 8;
            let workgroup_count_x =
                (self.surface_config.width + workgroup_size - 1) / workgroup_size;
            let workgroup_count_y =
                (self.surface_config.height + workgroup_size - 1) / workgroup_size;
            cpass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
        }

        // Step 4: Apply vertical blur
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bloom Vertical Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bloom_vertical_compute_pipeline);
            cpass.set_bind_group(0, &self.bloom_vertical_bind_group, &[]);
            let workgroup_size = 8;
            let workgroup_count_x =
                (self.surface_config.width + workgroup_size - 1) / workgroup_size;
            let workgroup_count_y =
                (self.surface_config.height + workgroup_size - 1) / workgroup_size;
            cpass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
        }

        // Step 5: Render to surface with bloom
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Post Process Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.post_process_pipeline);
            rpass.set_bind_group(0, &self.post_process_bind_group, &[]);
            rpass.draw(0..4, 0..1); // Draw full-screen quad
        }

        // Write texture data (if still needed)
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.texture_image.bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * self.texture_image.width),
                rows_per_image: Some(self.texture_image.height),
            },
            self.texture_size,
        );

        self.queue.submit(Some(encoder.finish()));
        surface_texture.present();
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    swap_chain_format: wgpu::TextureFormat,
    pipeline_layout: &wgpu::PipelineLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("atmosphere.txt"))),
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[create_vertex_buffer_layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(swap_chain_format.into())],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}
