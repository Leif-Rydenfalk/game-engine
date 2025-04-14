use cgmath::{Matrix4, Point3, SquareMatrix};
use hecs::World;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{MemoryHints, PipelineCompilationOptions, SamplerDescriptor, ShaderSource};
use winit::window::Window;

use imgui::*;
use imgui_wgpu::{Renderer, RendererConfig};
use imgui_winit_support::WinitPlatform;

use tracing::{debug, error, info, trace, warn};

// --- ShaderHotReload implementation ---
pub struct ShaderHotReload {
    device: Arc<wgpu::Device>,
    shader_dir: PathBuf,
    modules: Mutex<HashMap<String, (wgpu::ShaderModule, SystemTime)>>,
    last_check: Mutex<SystemTime>,
}

impl ShaderHotReload {
    pub fn new(device: Arc<wgpu::Device>, shader_dir: impl AsRef<Path>) -> Self {
        Self {
            device,
            shader_dir: shader_dir.as_ref().to_path_buf(),
            modules: Mutex::new(HashMap::new()),
            last_check: Mutex::new(SystemTime::now()),
        }
    }

    pub fn get_shader(&self, name: &str) -> wgpu::ShaderModule {
        let mut modules = self.modules.lock().unwrap();
        let path = self.shader_dir.join(name);

        // Try to get modification time, if file doesn't exist or has other issues,
        // fallback to embedded shader
        let metadata = match fs::metadata(&path) {
            Ok(meta) => meta,
            Err(e) => {
                info!(
                    "Warning: Couldn't read shader file {}: {}",
                    path.display(),
                    e
                );
                // Return embedded fallback shader
                return self.create_fallback_shader(name);
            }
        };

        let modified = match metadata.modified() {
            Ok(time) => time,
            Err(e) => {
                info!(
                    "Warning: Couldn't get modification time for {}: {}",
                    path.display(),
                    e
                );
                // Return embedded fallback shader
                return self.create_fallback_shader(name);
            }
        };

        if let Some((module, last_modified)) = modules.get(name) {
            if *last_modified >= modified {
                return module.clone();
            }
        }

        // Load shader from file
        let source = match fs::read_to_string(&path) {
            Ok(content) => content,
            Err(e) => {
                info!(
                    "Warning: Failed to read shader file {}: {}",
                    path.display(),
                    e
                );
                // Return embedded fallback shader
                return self.create_fallback_shader(name);
            }
        };

        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(source)),
            });

        modules.insert(name.to_string(), (shader_module.clone(), modified));
        info!("Loaded shader: {}", name);

        shader_module
    }

    fn create_fallback_shader(&self, name: &str) -> wgpu::ShaderModule {
        // For simplicity, we're using empty shaders as fallbacks
        // In a real implementation, you would embed the shaders here
        let fallback_source = match name {
            "bloom.wgsl" => include_str!("shaders/bloom.wgsl"),
            "color_correction.wgsl" => include_str!("shaders/color_correction.wgsl"),
            "final_conversion.wgsl" => include_str!("shaders/final_conversion.wgsl"),
            "sky.wgsl" => include_str!("shaders/sky.wgsl"),
            "voxel.wgsl" => include_str!("shaders/voxel.wgsl"),
            "taa.wgsl" => include_str!("shaders/taa.wgsl"),
            _ => {
                info!("Unknown shader: {}, using empty fallback", name);
                "// Empty fallback shader\n@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }\n@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 1.0, 1.0); }"
            }
        };

        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{}_fallback", name)),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(fallback_source)),
            })
    }

    pub fn check_for_updates(&self) -> Vec<String> {
        let mut last_check = self.last_check.lock().unwrap();
        let now = SystemTime::now();
        let mut updated_shaders = Vec::new();

        // Only check every 1 second to avoid excessive file reads
        if now
            .duration_since(*last_check)
            .unwrap_or(Duration::from_secs(0))
            < Duration::from_secs(1)
        {
            return updated_shaders;
        }

        *last_check = now;
        let modules = self.modules.lock().unwrap();

        for (name, (_, last_modified)) in modules.iter() {
            let path = self.shader_dir.join(name);
            if let Ok(metadata) = fs::metadata(&path) {
                if let Ok(modified) = metadata.modified() {
                    if modified > *last_modified {
                        updated_shaders.push(name.clone());
                    }
                }
            }
        }

        updated_shaders
    }
}
