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

fn main() -> Result<(), EventLoopError> {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)
}
