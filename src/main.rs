// #![feature(portable_simd)]

use crate::app::App;
use tracing::Level;
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

// fn main() {
//     use gilrs::{Button, Event, Gilrs};

//     let mut gilrs = Gilrs::new().unwrap();

//     // Iterate over all connected gamepads
//     for (_id, gamepad) in gilrs.gamepads() {
//         info!("{} is {:?}", gamepad.name(), gamepad.power_info());
//     }

//     let mut active_gamepad = None;

//     loop {
//         // Examine new events
//         while let Some(Event {
//             id, event, time, ..
//         }) = gilrs.next_event()
//         {
//             info!("{:?} New event from {}: {:?}", time, id, event);
//             active_gamepad = Some(id);
//         }

//         // You can also use cached gamepad state
//         if let Some(gamepad) = active_gamepad.map(|id| gilrs.gamepad(id)) {
//             if gamepad.is_pressed(Button::South) {
//                 info!("Button South is pressed (XBox - A, PS - X)");
//             }
//         }
//     }
// }

// extern crate sdl2;

// use sdl2::controller::{Axis, Button, GameController};
// use sdl2::event::Event;
// use sdl2::haptic::Haptic;
// use sdl2::keyboard::Keycode;
// use std::collections::HashMap;
// use std::time::Duration;

// // Store controller and optional haptic device together
// struct ControllerState {
//     controller: GameController,
//     haptic: Option<Haptic>,
// }

// fn main() -> Result<(), String> {
//     // --- Initialization ---
//     let sdl_context = sdl2::init()?;
//     let game_controller_subsystem = sdl_context.game_controller()?;
//     let haptic_subsystem = sdl_context.haptic()?;
//     let mut event_pump = sdl_context.event_pump()?;

//     // Keep track of opened controllers and their haptic devices
//     // Key: Joystick instance ID (i32)
//     let mut open_controllers: HashMap<i32, ControllerState> = HashMap::new();

//     info!("SDL2 Controller/Haptic Test Initialized.");
//     info!("Connect controllers now. Press ESC or close window to quit.");
//     info!("Press controller buttons to test input and trigger rumble.");

//     // --- Main Event Loop ---
//     'running: loop {
//         for event in event_pump.poll_iter() {
//             match event {
//                 // --- Quit Events ---
//                 Event::Quit { .. }
//                 | Event::KeyDown {
//                     keycode: Some(Keycode::Escape),
//                     ..
//                 } => break 'running,

//                 // --- Controller Device Events ---
//                 Event::ControllerDeviceAdded { which, .. } => {
//                     match game_controller_subsystem.open(which) {
//                         Ok(controller) => {
//                             let instance_id = controller.instance_id();
//                             info!(
//                                 "Controller Added: index={}, instance_id={}, name='{}'",
//                                 which,
//                                 instance_id,
//                                 controller.name()
//                             );
//                         }
//                         Err(e) => {
//                             einfo!("Error opening controller index {}: {}", which, e);
//                         }
//                     }
//                 }
//                 Event::ControllerDeviceRemoved { which, .. } => {
//                     // 'which' here is the instance_id
//                     if let Some(removed_state) = open_controllers.remove(&(which as i32)) {
//                         info!(
//                             "Controller Removed: instance_id={}, name='{}'",
//                             which,
//                             removed_state.controller.name()
//                         );
//                         // Haptic device will be closed automatically when `removed_state` is dropped
//                     } else {
//                         info!(
//                             "Controller Removed: instance_id={} (was not tracked)",
//                             which
//                         );
//                     }
//                 }

//                 // --- Controller Input Events ---
//                 Event::ControllerButtonDown { which, button, .. } => {
//                     info!(
//                         "Button Down: Controller instance_id={}, Button={:?}",
//                         which, button
//                     );

//                     if button == Button::A || button == Button::A {
//                         // Try to open haptic device associated with this controller
//                         let haptic_result = haptic_subsystem.open_from_joystick_id(which);

//                         match haptic_result {
//                             Ok(mut haptic_device) => {
//                                 info!(
//                                     "  Haptic device found and opened for instance_id={}",
//                                     which
//                                 );
//                                 // Initialize rumble effect (required before playing)
//                                 haptic_device.rumble_play(10.0, 100);
//                             }
//                             Err(e) => {
//                                 info!("  No haptic device found or error opening for instance_id={}: {}", which, e);
//                             }
//                         }
//                     }
//                 }
//                 Event::ControllerButtonUp { which, button, .. } => {
//                     info!(
//                         "Button Up:   Controller instance_id={}, Button={:?}",
//                         which, button
//                     );
//                 }
//                 Event::ControllerAxisMotion {
//                     which, axis, value, ..
//                 } => {
//                     // Axis motion can be noisy, only print significant changes
//                     if value.abs() > 1500 {
//                         // Axis range is typically -32768 to 32767
//                         info!(
//                             "Axis Motion: Controller instance_id={}, Axis={:?}, Value={}",
//                             which, axis, value
//                         );
//                     }
//                 }

//                 _ => {} // Ignore other events
//             }
//         }

//         // Small delay to prevent busy-waiting
//         ::std::thread::sleep(Duration::from_millis(10));
//     } // End of 'running loop

//     // --- Cleanup ---
//     // `open_controllers` HashMap goes out of scope here.
//     // When the `ControllerState` structs are dropped, their `GameController`
//     // and `Option<Haptic>` fields are also dropped, automatically closing them.
//     info!("Exiting...");
//     Ok(())
// }
