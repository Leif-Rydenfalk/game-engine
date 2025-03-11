#![feature(portable_simd)]

use crate::app::App;
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

mod app;
pub use app::*;

mod input;
pub use input::*;

mod img_utils;
pub use img_utils::*;

mod vertex;
pub use vertex::*;

mod wgpu_ctx;
pub use wgpu_ctx::*;

mod components;
pub use components::*;

mod systems;
pub use systems::*;

mod world;
pub use world::*;

mod model;
pub use model::*;

mod bloom;
pub use bloom::*;

mod color_correction;
pub use color_correction::*;

fn main() -> Result<(), EventLoopError> {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)
}

/*
Get image frequencies with fft
Get filter frequencies with fft
Get convolved frequencies by multiplying image frequencies with filter frequencies
Get image back from convolved frequencies with ifft
*/

// use image::{ImageBuffer, Luma};
// use rustfft::{num_complex::Complex, FftPlanner};

// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     // Load the image
//     let img = image::open("./assets/images/tree.png")?;
//     let img_gray = img.to_luma8();
//     let (width, height) = img_gray.dimensions();

//     println!("Image dimensions: {}x{}", width, height);

//     // Convert image to complex values
//     let mut complex_data: Vec<Complex<f32>> = img_gray
//         .pixels()
//         .map(|p| Complex {
//             re: p[0] as f32,
//             im: 0.0,
//         })
//         .collect();

//     // Create FFT planner
//     let mut planner = FftPlanner::<f32>::new();

//     // Perform FFT on rows
//     let row_fft = planner.plan_fft_forward(width as usize);
//     for y in 0..height {
//         let start_idx = (y * width) as usize;
//         let end_idx = start_idx + width as usize;
//         row_fft.process(&mut complex_data[start_idx..end_idx]);
//     }

//     // Perform FFT on columns
//     // For this, we need to rearrange data or process it differently
//     let col_fft = planner.plan_fft_forward(height as usize);
//     let mut temp_column = vec![Complex { re: 0.0, im: 0.0 }; height as usize];

//     for x in 0..width {
//         // Extract column
//         for y in 0..height {
//             temp_column[y as usize] = complex_data[(y * width + x) as usize];
//         }

//         // Process column
//         col_fft.process(&mut temp_column);

//         // Put back the processed column
//         for y in 0..height {
//             complex_data[(y * width + x) as usize] = temp_column[y as usize];
//         }
//     }

//     // Shift FFT (move DC component to center)
//     let mut shifted_data = vec![Complex { re: 0.0, im: 0.0 }; (width * height) as usize];
//     for y in 0..height {
//         for x in 0..width {
//             let shift_y = (y + height / 2) % height;
//             let shift_x = (x + width / 2) % width;
//             let old_idx = (y * width + x) as usize;
//             let new_idx = (shift_y * width + shift_x) as usize;
//             shifted_data[new_idx] = complex_data[old_idx];
//         }
//     }
//     complex_data = shifted_data;

//     // Create magnitude image for visualization
//     let mut fft_mag = ImageBuffer::new(width, height);

//     // Use log scaling with a higher factor and add epsilon to avoid log(0)
//     for y in 0..height {
//         for x in 0..width {
//             let idx = (y * width + x) as usize;
//             let magnitude = (complex_data[idx].re.powi(2) + complex_data[idx].im.powi(2)).sqrt();

//             // Improved log scaling for better visualization
//             let log_mag = (1.0 + magnitude.max(1e-10)).ln() * 70.0;
//             let pixel_value = (log_mag.clamp(0.0, 255.0)) as u8;

//             fft_mag.put_pixel(x, y, Luma([pixel_value]));
//         }
//     }

//     img_gray.save("fft_input.png")?;
//     println!("Input image saved as fft_input.png");

//     // Save the FFT magnitude image
//     fft_mag.save("fft_result.png")?;
//     println!("FFT image saved as fft_result.png");

//     Ok(())
// }

// const LENGTH: usize = 1920 * 1080;
// const LENGTH: usize = 1280 * 720;
// const LENGTH: usize = 854 * 480;

// use std::simd::*;
// use std::time::Instant;

// fn main() {
//     const COUNT: usize = 1000;
//     let mut x = vec![f32x32::splat(0.0); COUNT];
//     let mut x_prev = x.clone();

//     let start = Instant::now();

//     let damp = f32x32::splat(0.99);

//     for i in 0..COUNT {
//         let diff = x[i] - x_prev[i];
//         x[i] += diff * damp;
//         x_prev[i] = x[i];
//     }

//     let duration = start.elapsed();
//     println!("{:?}, {:?}", duration, x[0]);
// }
