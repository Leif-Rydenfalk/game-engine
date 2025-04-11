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

fn main() -> Result<(), EventLoopError> {
    // let subscriber = FmtSubscriber::builder()
    //     .with_max_level(Level::TRACE)
    //     .finish();
    // tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    // let subscriber = FmtSubscriber::builder()
    //     .with_env_filter(EnvFilter::from_default_env())
    //     .finish();
    // tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::filter::filter_fn(|metadata| {
            let level = metadata.level();
            *level == tracing::Level::INFO || *level == tracing::Level::WARN
        }))
        .init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)
}
