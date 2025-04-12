// #![feature(portable_simd)]

use crate::app::App;
use tracing::{info, Level};
use tracing_subscriber::{filter::LevelFilter, prelude::*};
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

mod audio_converter;
pub use audio_converter::*;

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

use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{fs, thread};

use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source};

/// Represents a unique sound instance identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SoundInstanceId {
    sound_id: String,
    instance_number: u64,
}

impl SoundInstanceId {
    /// Creates a new SoundInstanceId
    fn new(sound_id: &str, instance_number: u64) -> Self {
        Self {
            sound_id: sound_id.to_string(),
            instance_number,
        }
    }

    /// Returns the original sound ID
    pub fn sound_id(&self) -> &str {
        &self.sound_id
    }

    /// Returns the instance number
    pub fn instance_number(&self) -> u64 {
        self.instance_number
    }
}

impl std::fmt::Display for SoundInstanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}_{}", self.sound_id, self.instance_number)
    }
}

fn main() {
    // Convert AIF to WAV before starting the app
    if AudioConverter::check_ffmpeg() {
        match AudioConverter::convert_aif_to_wav(
            Path::new("assets/sounds"),
            Path::new("assets/sounds_wav"),
        ) {
            Ok(files) => println!("Converted {} files to WAV format", files.len()),
            Err(e) => eprintln!("Failed to convert audio files: {}", e),
        }
    } else {
        println!("FFmpeg not found. Install FFmpeg to convert audio files automatically.");
    }

    // Create a sound manager (now using sounds_wav directory)
    let mut sound_manager = SoundManager::new();

    // Load all sounds from assets/sounds_wav
    sound_manager.load_sounds().unwrap();

    // List available sound IDs
    let sound_ids = sound_manager.get_sound_ids();
    println!("Available sounds: {:?}", sound_ids);

    // Play multiple instances of the same sound simultaneously
    println!("Playing multiple instances of the same sound simultaneously");

    // Example 1: Play 6 instances with short delay (overlapping)
    for _ in 0..6 {
        let instance_id = sound_manager.play("exp 2").unwrap();
        println!("Started sound instance: {}", instance_id);
        thread::sleep(Duration::from_millis(300));
    }

    // Wait for sounds to finish
    thread::sleep(Duration::from_secs(3));

    // Example 2: Play 3 instances at exactly the same time
    println!("\nPlaying 3 instances simultaneously (no delay):");
    let instance1 = sound_manager.play("exp 2").unwrap();
    let instance2 = sound_manager.play("exp 2").unwrap();
    let instance3 = sound_manager.play("exp 2").unwrap();

    println!(
        "Started instances: {}, {}, {}",
        instance1, instance2, instance3
    );

    // Wait for sounds to finish
    thread::sleep(Duration::from_secs(3));

    // Example 3: Stop a specific instance
    println!("\nPlaying 3 more instances but stopping one specifically:");
    let instance1 = sound_manager.play("exp 2").unwrap();
    let instance2 = sound_manager.play("exp 2").unwrap();
    let instance3 = sound_manager.play("exp 2").unwrap();

    // Stop just the second instance
    thread::sleep(Duration::from_millis(500));
    println!("Stopping instance: {}", instance2);
    sound_manager.stop_instance(&instance2).unwrap();

    // Wait for remaining sounds to finish
    thread::sleep(Duration::from_secs(5));
}

/// Manages loading and playback of sound files from the assets/sounds_wav directory
/// Supports playing multiple instances of the same sound simultaneously
pub struct SoundManager {
    sounds: HashMap<String, Arc<Vec<u8>>>,
    output_stream: Option<(OutputStream, OutputStreamHandle)>,
    sinks: Mutex<HashMap<SoundInstanceId, (Arc<Sink>, Instant)>>,
    cleanup_threshold: Duration,
    instance_counter: Mutex<u64>,
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
            instance_counter: Mutex::new(0),
        }
    }

    /// Loads all WAV sounds from the assets/sounds_wav directory
    pub fn load_sounds(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let assets_path = Path::new("assets/sounds_wav");

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

                    if ext == "wav" {
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

        Ok(())
    }

    /// Returns the next instance counter value
    fn next_instance_id(&self) -> u64 {
        let mut counter = self.instance_counter.lock().unwrap();
        *counter += 1;
        *counter
    }

    /// Plays a sound by its ID (filename without extension)
    /// Each call creates a new instance that plays independently,
    /// allowing multiple instances of the same sound to play simultaneously
    /// Returns the unique instance ID which can be used to stop this specific instance
    pub fn play(&self, sound_id: &str) -> Result<SoundInstanceId, Box<dyn std::error::Error>> {
        // First, clean up any finished sinks
        self.cleanup_finished_sinks();

        if let Some(sound_data) = self.sounds.get(sound_id) {
            if let Some((_, stream_handle)) = &self.output_stream {
                // Create a new sink for this instance
                let sink = Sink::try_new(stream_handle)?;
                let cursor = Cursor::new(sound_data.as_slice().to_vec());

                // Try to decode the sound
                let source = match Decoder::new(cursor) {
                    Ok(source) => source,
                    Err(err) => {
                        tracing::warn!("Failed to decode sound '{}': {}", sound_id, err);
                        return Err(
                            format!("Failed to decode sound '{}': {}", sound_id, err).into()
                        );
                    }
                };

                sink.append(source);

                let sink = Arc::new(sink);

                // Create a unique ID for this specific instance using our counter
                let instance_number = self.next_instance_id();
                let instance_id = SoundInstanceId::new(sound_id, instance_number);

                self.sinks
                    .lock()
                    .unwrap()
                    .insert(instance_id.clone(), (sink.clone(), Instant::now()));

                tracing::info!("Playing sound: {} (instance: {})", sound_id, instance_id);
                return Ok(instance_id);
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
            // Only match exact sound IDs
            if id.sound_id() == sound_id {
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

        tracing::info!("Stopped all instances of sound: {}", sound_id);
        Ok(())
    }

    /// Stops a specific instance of a sound by its unique instance ID
    pub fn stop_instance(
        &self,
        instance_id: &SoundInstanceId,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut sinks = self.sinks.lock().unwrap();

        if let Some((sink, _)) = sinks.remove(instance_id) {
            sink.stop();
            tracing::info!("Stopped sound instance: {}", instance_id);
            Ok(())
        } else {
            Err(format!("No active playback for instance '{}'", instance_id).into())
        }
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
        let to_remove: Vec<SoundInstanceId> = sinks
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
