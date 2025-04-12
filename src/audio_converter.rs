use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Utility to convert audio files from one format to another
pub struct AudioConverter;

impl AudioConverter {
    /// Convert all AIF/AIFF files in source_dir to WAV format in target_dir
    /// Requires FFmpeg to be installed and available in PATH
    pub fn convert_aif_to_wav(
        source_dir: &Path,
        target_dir: &Path,
    ) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        // Create target directory if it doesn't exist
        if !target_dir.exists() {
            fs::create_dir_all(target_dir)?;
        }

        // Find all AIF/AIFF files
        let mut converted_files = Vec::new();
        for entry in fs::read_dir(source_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(extension) = path.extension() {
                    let ext = extension.to_string_lossy().to_lowercase();

                    if ext == "aif" || ext == "aiff" {
                        let file_stem = path.file_stem().unwrap().to_string_lossy().to_string();
                        let output_path = target_dir.join(format!("{}.wav", file_stem));

                        // Check if WAV already exists and is newer than the source file
                        let needs_conversion = if output_path.exists() {
                            match (fs::metadata(&path), fs::metadata(&output_path)) {
                                (Ok(src_meta), Ok(dst_meta)) => {
                                    match (src_meta.modified(), dst_meta.modified()) {
                                        (Ok(src_time), Ok(dst_time)) => {
                                            // Only convert if source is newer than destination
                                            src_time > dst_time
                                        }
                                        _ => true, // Can't compare times, so convert to be safe
                                    }
                                }
                                _ => true, // Can't get metadata, so convert to be safe
                            }
                        } else {
                            true // Output doesn't exist, so conversion is needed
                        };

                        if needs_conversion {
                            // Run FFmpeg conversion
                            let status = Command::new("ffmpeg")
                                .arg("-i")
                                .arg(&path)
                                .arg("-y") // Overwrite output files
                                .arg(&output_path)
                                .status()?;

                            if status.success() {
                                tracing::info!(
                                    "Converted: {} -> {}",
                                    path.display(),
                                    output_path.display()
                                );
                                converted_files.push(output_path);
                            } else {
                                tracing::error!("Failed to convert: {}", path.display());
                            }
                        } else {
                            tracing::info!(
                                "Skipped conversion (already up to date): {} -> {}",
                                path.display(),
                                output_path.display()
                            );
                            // Still add to converted_files so it can be used
                            converted_files.push(output_path);
                        }
                    }
                }
            }
        }

        Ok(converted_files)
    }

    /// Check if FFmpeg is installed
    pub fn check_ffmpeg() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
}

// Usage example:
// fn main() {
//     // Always convert AIF files to WAV first
//     if AudioConverter::check_ffmpeg() {
//         let _ = AudioConverter::convert_aif_to_wav(
//             Path::new("assets/sounds"),
//             Path::new("assets/sounds_wav")
//         );
//     }
//
//     // Create sound manager and let it load from the WAV directory
//     let mut sound_manager = SoundManager::new();
//     sound_manager.load_sounds().unwrap();
// }
