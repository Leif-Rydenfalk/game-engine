// #![feature(portable_simd)]

use crate::app::App;
use tracing::{info, Level};
use tracing_subscriber::{filter::LevelFilter, prelude::*};
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

mod app;
pub use app::*;

mod input;
pub use input::*;

mod imgui_state;
pub use imgui_state::*;

mod wgpu_ctx;
pub use wgpu_ctx::*;

// mod post_processing;
// pub use post_processing::*;

mod img_utils;
pub use img_utils::*;

mod vertex;
pub use vertex::*;

mod model;
pub use model::*;

mod world;
pub use world::*;

mod voxels;
pub use voxels::*;

mod sky;
pub use sky::*;

mod atmosphere;
pub use atmosphere::*;

mod bloom;
pub use bloom::*;

mod color_correction;
pub use color_correction::*;

// fn main() -> Result<(), EventLoopError> {
//     // let subscriber = FmtSubscriber::builder()
//     //     .with_max_level(Level::TRACE)
//     //     .finish();
//     // tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

//     // let subscriber = FmtSubscriber::builder()
//     //     .with_env_filter(EnvFilter::from_default_env())
//     //     .finish();
//     // tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

//     tracing_subscriber::registry()
//         .with(tracing_subscriber::fmt::layer())
//         .with(tracing_subscriber::filter::filter_fn(|metadata| {
//             let level = metadata.level();
//             *level == tracing::Level::INFO || *level == tracing::Level::WARN
//         }))
//         .init();

//     let event_loop = EventLoop::new().unwrap();
//     event_loop.set_control_flow(ControlFlow::Poll);
//     let mut app = App::default();
//     event_loop.run_app(&mut app)
// }

fn main() {
    // Create a sound manager
    let mut sound_manager = SoundManager::new();

    // Load all sounds from assets/sounds
    sound_manager.load_sounds().unwrap();

    // List available sound IDs
    let sound_ids = sound_manager.get_sound_ids();
    println!("Available sounds: {:?}", sound_ids);

    // Play a sound by its ID (filename without extension)
    sound_manager.play("missile").unwrap();

    // // Stop a specific sound
    // sound_manager.stop("explosion").unwrap();

    // // Stop all sounds
    // sound_manager.stop_all();

    thread::sleep(Duration::from_secs(10));
}

use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{fs, thread};

use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source};

/// Manages loading and playback of sound files from the assets/sounds directory
pub struct SoundManager {
    sounds: HashMap<String, Arc<Vec<u8>>>,
    output_stream: Option<(OutputStream, OutputStreamHandle)>,
    sinks: Mutex<HashMap<String, (Arc<Sink>, Instant)>>,
    cleanup_threshold: Duration,
}

impl Default for SoundManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SoundManager {
    /// Creates a new SoundManager
    pub fn new() -> Self {
        let output_stream = OutputStream::try_default().ok();

        SoundManager {
            sounds: HashMap::new(),
            output_stream,
            sinks: Mutex::new(HashMap::new()),
            cleanup_threshold: Duration::from_secs(60), // Clean up sinks after a minute
        }
    }

    /// Loads all AIF/AIFF sounds from the assets/sounds directory
    ///
    /// Note: rodio may not directly support AIF/AIFF format. If playback fails,
    /// consider converting your files to a format rodio supports (like WAV).
    pub fn load_sounds(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let assets_path = Path::new("assets/sounds");

        // Ensure the directory exists
        if !assets_path.exists() {
            return Err(format!("Directory not found: {:?}", assets_path).into());
        }

        for entry in fs::read_dir(assets_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(extension) = path.extension() {
                    let ext = extension.to_string_lossy().to_lowercase();

                    if ext == "aif" || ext == "aiff" {
                        let filename = path.file_stem().unwrap().to_string_lossy().to_string();
                        tracing::info!("Loading sound: {}", filename);

                        // Read the file into memory
                        let sound_data = fs::read(&path)?;
                        self.sounds.insert(filename, Arc::new(sound_data));
                    }
                }
            }
        }

        tracing::info!("Loaded {} sounds", self.sounds.len());

        if !self.sounds.is_empty() {
            tracing::warn!(
                "Note: rodio may not directly support AIFF format. \
                           If playback fails, consider converting your files to WAV format \
                           using a tool like ffmpeg or audacity."
            );
        }

        Ok(())
    }

    /// Plays a sound by its ID (filename without extension)
    pub fn play(&self, sound_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        // First, clean up any finished sinks
        self.cleanup_finished_sinks();

        if let Some(sound_data) = self.sounds.get(sound_id) {
            if let Some((_, stream_handle)) = &self.output_stream {
                let sink = Sink::try_new(stream_handle)?;
                let cursor = Cursor::new(sound_data.as_slice().to_vec());

                // Try to decode the sound
                let source = match Decoder::new(cursor) {
                    Ok(source) => source,
                    Err(err) => {
                        tracing::warn!("Failed to decode sound '{}': {}", sound_id, err);
                        return Err(format!(
                            "Failed to decode sound '{}': {}. \
                                          AIFF format may not be directly supported by rodio. \
                                          Consider converting your files to WAV format.",
                            sound_id, err
                        )
                        .into());
                    }
                };

                sink.append(source);

                let sink = Arc::new(sink);
                let unique_id = format!("{}_{}", sound_id, Instant::now().elapsed().as_micros());
                self.sinks
                    .lock()
                    .unwrap()
                    .insert(unique_id, (sink.clone(), Instant::now()));

                tracing::info!("Playing sound: {}", sound_id);
                return Ok(());
            } else {
                return Err("No audio output device available".into());
            }
        }

        Err(format!("Sound '{}' not found", sound_id).into())
    }

    /// Stops all instances of a sound playing with the given ID
    pub fn stop(&self, sound_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut sinks = self.sinks.lock().unwrap();
        let mut to_remove = Vec::new();

        for (id, (sink, _)) in sinks.iter() {
            if id.starts_with(sound_id) {
                sink.stop();
                to_remove.push(id.clone());
            }
        }

        if to_remove.is_empty() {
            return Err(format!("No active playback for sound '{}'", sound_id).into());
        }

        for id in to_remove {
            sinks.remove(&id);
        }

        tracing::info!("Stopped sound: {}", sound_id);
        Ok(())
    }

    /// Stops all playing sounds
    pub fn stop_all(&self) {
        let mut sinks = self.sinks.lock().unwrap();
        for (_, (sink, _)) in sinks.drain() {
            sink.stop();
        }
        tracing::info!("Stopped all sounds");
    }

    /// Cleans up any finished or old sinks
    fn cleanup_finished_sinks(&self) {
        let mut sinks = self.sinks.lock().unwrap();
        let now = Instant::now();
        let to_remove: Vec<String> = sinks
            .iter()
            .filter(|(_, (sink, created))| {
                sink.empty() || now.duration_since(*created) > self.cleanup_threshold
            })
            .map(|(id, _)| id.clone())
            .collect();

        for id in to_remove {
            if let Some((sink, _)) = sinks.remove(&id) {
                sink.stop();
            }
        }
    }

    /// Returns a list of all loaded sound IDs
    pub fn get_sound_ids(&self) -> Vec<String> {
        self.sounds.keys().cloned().collect()
    }

    /// Checks if a sound with the given ID is loaded
    pub fn has_sound(&self, sound_id: &str) -> bool {
        self.sounds.contains_key(sound_id)
    }
}

impl Drop for SoundManager {
    fn drop(&mut self) {
        self.stop_all();
    }
}
