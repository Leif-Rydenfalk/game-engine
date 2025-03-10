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
