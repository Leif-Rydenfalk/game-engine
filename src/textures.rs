use crate::RgbaImg;
use std::{path::Path, sync::Arc};
use tracing::{debug, error, info, trace, warn};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupLayout,
}; // Keep DeviceExt for texture writing

// Helper macro for loading 2D textures (moved outside specific methods)
macro_rules! load_texture_2d {
    ($device:expr, $queue:expr, $path:expr, $label:expr) => {{
        let img = RgbaImg::new($path).unwrap();
        // |e| {
        //     error!("Failed to load texture at '{}': {}", $path, e);
        //     // Return a default 1x1 magenta texture on error? Or panic? Let's panic for now.
        //     panic!("Failed to load texture: {}", $path);
        // });
        let texture = $device.create_texture(&wgpu::TextureDescriptor {
            label: Some($label),
            size: wgpu::Extent3d {
                width: img.width,
                height: img.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1, // Consider generating mipmaps for better quality
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb, // Assuming sRGB for color textures
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        $queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &img.bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * img.width),
                rows_per_image: Some(img.height),
            },
            wgpu::Extent3d {
                width: img.width,
                height: img.height,
                depth_or_array_layers: 1,
            },
        );
        (
            texture.clone(),
            texture.create_view(&wgpu::TextureViewDescriptor::default()),
        )
    }};
}

#[derive(Debug)]
pub struct StaticTextures {
    pub device: Arc<wgpu::Device>, // Keep device ref if needed later, maybe for mipmaps
    pub queue: Arc<wgpu::Queue>,   // Keep queue ref if needed later
    pub rgb_noise_texture: wgpu::Texture,
    pub rgb_noise_texture_view: wgpu::TextureView,
    pub gray_noise_texture: wgpu::Texture,
    pub gray_noise_texture_view: wgpu::TextureView,
    pub gray_noise_cube_texture: wgpu::Texture,
    pub gray_noise_cube_texture_view: wgpu::TextureView,
    pub grain_texture: wgpu::Texture,
    pub grain_texture_view: wgpu::TextureView,
    pub dirt_texture: wgpu::Texture,
    pub dirt_texture_view: wgpu::TextureView,
    pub pebble_texture: wgpu::Texture,
    pub pebble_texture_view: wgpu::TextureView,
    pub texture_sampler: wgpu::Sampler, // Sampler is often shared too
    pub texture_bind_group_layout: BindGroupLayout,
    pub texture_bind_group: BindGroup,
}

impl StaticTextures {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        info!("Loading static textures...");

        // Texture sampler (can be shared by many textures/renderers)
        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shared Texture Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear, // Consider Nearest if using mipmaps later
            ..Default::default()
        });

        // Load textures using the macro
        let (rgb_noise_texture, rgb_noise_texture_view) = load_texture_2d!(
            device.as_ref(), // Pass device/queue refs to macro
            queue.as_ref(),
            "./assets/images/textures/rgbnoise.png",
            "Rgb noise Texture"
        );
        let (gray_noise_texture, gray_noise_texture_view) = load_texture_2d!(
            device.as_ref(),
            queue.as_ref(),
            "./assets/images/textures/graynoise.png",
            "Gray noise Texture"
        );
        let (grain_texture, grain_texture_view) = load_texture_2d!(
            device.as_ref(),
            queue.as_ref(),
            "./assets/images/textures/grain.png",
            "Grain Texture"
        );
        let (dirt_texture, dirt_texture_view) = load_texture_2d!(
            device.as_ref(),
            queue.as_ref(),
            "./assets/images/textures/mud.png", // Assuming mud.png is the dirt texture
            "Dirt Texture"
        );
        let (pebble_texture, pebble_texture_view) = load_texture_2d!(
            device.as_ref(),
            queue.as_ref(),
            "./assets/images/textures/pebble.png",
            "Pebble Texture"
        );

        // Load 3D gray noise texture
        let gray_noise_cube_data =
            std::fs::read("./assets/images/textures/graynoise_32x32x32_cube.bin")
                .expect("Failed to read gray noise cube")[20..20 + 32 * 32 * 32]
                .to_vec();
        let gray_noise_cube_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Gray Noise Cube Texture"),
            size: wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Unorm, // Use R8Unorm for single channel
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &gray_noise_cube_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &gray_noise_cube_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(32),  // 32 pixels * 1 byte/pixel
                rows_per_image: Some(32), // 32 rows
            },
            wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 32,
            },
        );
        let gray_noise_cube_texture_view =
            gray_noise_cube_texture.create_view(&wgpu::TextureViewDescriptor::default());

        info!("Static textures loaded.");

        // Texture bind group layout
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
                            view_dimension: wgpu::TextureViewDimension::D3,
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
                label: Some("texture_bind_group_layout"),
            });

        // Texture bind group (Use views/sampler from static_textures)
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    // Use view from static_textures
                    resource: wgpu::BindingResource::TextureView(&rgb_noise_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&gray_noise_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&gray_noise_cube_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&grain_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&dirt_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&pebble_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    // Use sampler from static_textures
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
            ],
            label: Some("texture_bind_group"),
        });

        Self {
            device, // Store Arc references
            queue,
            rgb_noise_texture,
            rgb_noise_texture_view,
            gray_noise_texture,
            gray_noise_texture_view,
            gray_noise_cube_texture,
            gray_noise_cube_texture_view,
            grain_texture,
            grain_texture_view,
            dirt_texture,
            dirt_texture_view,
            pebble_texture,
            pebble_texture_view,
            texture_sampler,
            texture_bind_group_layout,
            texture_bind_group,
        }
    }
}
