use gilrs::{Axis, Button, Event, EventType, GamepadId, Gilrs, GilrsBuilder};
use hecs::World;
use std::collections::HashMap;
use tracing::{debug, error, info, trace, warn};
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::KeyCode;

use crate::{sound, SoundManager, SoundManagerComponent};

pub struct Input {
    keys_current: HashMap<KeyCode, ElementState>,
    keys_previous: HashMap<KeyCode, ElementState>,
    mouse_buttons_current: HashMap<MouseButton, ElementState>,
    mouse_buttons_previous: HashMap<MouseButton, ElementState>,
    mouse_position: (f64, f64),
    mouse_delta: (f64, f64),
    scroll_delta: f64,

    // Track controller button states
    controller_buttons_current: HashMap<(usize, Button), bool>,
    controller_buttons_previous: HashMap<(usize, Button), bool>,

    // Track analog values for trigger buttons
    controller_trigger_values: HashMap<(usize, Button), f32>,

    // Tracks controller state
    pub gilrs: Option<Gilrs>,
}

impl Default for Input {
    fn default() -> Self {
        // Create instance with default values
        let mut input = Self {
            keys_current: Default::default(),
            keys_previous: Default::default(),
            mouse_buttons_current: Default::default(),
            mouse_buttons_previous: Default::default(),
            mouse_position: Default::default(),
            mouse_delta: Default::default(),
            scroll_delta: Default::default(),
            controller_buttons_current: Default::default(),
            controller_buttons_previous: Default::default(),
            controller_trigger_values: Default::default(),
            gilrs: Default::default(),
        };

        // Initialize Gilrs
        match GilrsBuilder::new().build() {
            Ok(gilrs) => {
                input.gilrs = Some(gilrs);
            }
            Err(err) => {
                warn!("Error initializing gilrs: {}", err);
            }
        }

        input
    }
}

impl Input {
    pub fn handle_key_input(&mut self, key: KeyCode, state: ElementState) {
        self.keys_current.insert(key, state);
    }

    pub fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        self.mouse_buttons_current.insert(button, state);
    }

    pub fn handle_cursor_moved(&mut self, position: &PhysicalPosition<f64>) {
        self.mouse_position = (position.x, position.y);
    }

    pub fn handle_mouse_motion(&mut self, delta: (f64, f64)) {
        self.mouse_delta.0 += delta.0;
        self.mouse_delta.1 += delta.1;
    }

    pub fn handle_mouse_scroll(&mut self, delta: f64) {
        self.scroll_delta += delta;
    }

    pub fn update(&mut self, world: &mut World) {
        // Save previous states
        self.keys_previous = self.keys_current.clone();
        self.mouse_buttons_previous = self.mouse_buttons_current.clone();
        self.controller_buttons_previous = self.controller_buttons_current.clone();

        // Reset deltas
        self.mouse_delta = (0.0, 0.0);
        self.scroll_delta = 0.0;

        // Update all controller button states first
        if let Some(gilrs) = &mut self.gilrs {
            // Process controller events
            while let Some(Event { id, event, .. }) = gilrs.next_event() {
                // info!("New event from {}: {:?}", id, event);

                let controller_index = Self::controller_index_from_id(gilrs, id);

                match event {
                    EventType::ButtonPressed(button, _) => {
                        // Already updated above, but just to be sure
                        self.controller_buttons_current
                            .insert((controller_index, button), true);
                    }
                    EventType::ButtonReleased(button, _) => {
                        // Already updated above, but just to be sure
                        self.controller_buttons_current
                            .insert((controller_index, button), false);
                    }
                    EventType::ButtonChanged(button, value, _) => {
                        // Special handling for triggers
                        if button == Button::LeftTrigger2 || button == Button::RightTrigger2 {
                            // Store the analog value
                            self.controller_trigger_values
                                .insert((controller_index, button), value);

                            // Also update the button state based on threshold
                            const TRIGGER_THRESHOLD: f32 = 0.1;
                            self.controller_buttons_current
                                .insert((controller_index, button), value > TRIGGER_THRESHOLD);
                        }

                        // for (_, sound_manager_component) in
                        //     world.query_mut::<&mut SoundManagerComponent>()
                        // {
                        //     let sound_manager = sound_manager_component.inner.lock().unwrap();
                        //     sound_manager.play("click 2").unwrap();
                        // }
                    }
                    // EventType::AxisChanged(axis, value, code) => {
                    //     for (_, sound_manager_component) in
                    //         world.query_mut::<&mut SoundManagerComponent>()
                    //     {
                    //         let sound_manager = sound_manager_component.inner.lock().unwrap();
                    //         sound_manager.play("click 2").unwrap();
                    //     }
                    // }
                    EventType::Connected => {
                        let gp = gilrs.gamepad(id);
                        let name = gp.name();
                        let is_ff_supported = gp.is_ff_supported();
                        let power_info = gp.power_info();

                        info!("Controller connected: {}", id);
                        info!("  Name: {}", name);
                        info!("  Force feedback supported: {}", is_ff_supported);
                        info!("  Power info: {:?}", power_info);
                        info!("  UUID: {:?}", gp.uuid());
                        info!("  Vendor ID: {:?}", gp.vendor_id());
                        info!("  Product ID: {:?}", gp.product_id());
                    }
                    EventType::Disconnected => {
                        info!("Controller disconnected: {}", id);

                        // Clean up button states for this controller
                        self.controller_buttons_current
                            .retain(|&(i, _), _| i != controller_index);
                        self.controller_buttons_previous
                            .retain(|&(i, _), _| i != controller_index);

                        // Clean up trigger values for this controller
                        self.controller_trigger_values
                            .retain(|&(i, _), _| i != controller_index);
                    }
                    _ => {} // Other events don't affect button state
                }
            }
        }
    }

    // Key state queries
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_current.get(&key) == Some(&ElementState::Pressed)
            && self.keys_previous.get(&key) != Some(&ElementState::Pressed)
    }

    pub fn is_key_released(&self, key: KeyCode) -> bool {
        self.keys_current.get(&key) == Some(&ElementState::Released)
            && self.keys_previous.get(&key) == Some(&ElementState::Pressed)
    }

    pub fn is_key_down(&self, key: KeyCode) -> bool {
        self.keys_current.get(&key) == Some(&ElementState::Pressed)
    }

    // Mouse state queries
    pub fn is_mouse_button_pressed(&self, button: MouseButton) -> bool {
        self.mouse_buttons_current.get(&button) == Some(&ElementState::Pressed)
            && self.mouse_buttons_previous.get(&button) != Some(&ElementState::Pressed)
    }

    pub fn is_mouse_button_released(&self, button: MouseButton) -> bool {
        self.mouse_buttons_current.get(&button) == Some(&ElementState::Released)
            && self.mouse_buttons_previous.get(&button) == Some(&ElementState::Pressed)
    }

    pub fn is_mouse_button_down(&self, button: MouseButton) -> bool {
        self.mouse_buttons_current.get(&button) == Some(&ElementState::Pressed)
    }

    pub fn mouse_position(&self) -> (f64, f64) {
        self.mouse_position
    }

    pub fn mouse_delta(&self) -> (f64, f64) {
        self.mouse_delta
    }

    pub fn scroll_delta(&self) -> f64 {
        self.scroll_delta
    }

    // Controller state queries
    pub fn connected_controllers_count(&self) -> usize {
        match &self.gilrs {
            Some(gilrs) => gilrs.gamepads().count(),
            None => 0,
        }
    }

    pub fn is_controller_connected(&self, controller_index: usize) -> bool {
        match &self.gilrs {
            Some(gilrs) => {
                if let Some(id) = self.controller_id_from_index(controller_index) {
                    gilrs.gamepad(id).is_connected()
                } else {
                    false
                }
            }
            None => false,
        }
    }

    // Helper to get GamepadId from index (maps our index to Gilrs' id)
    fn controller_id_from_index(&self, index: usize) -> Option<GamepadId> {
        match &self.gilrs {
            Some(gilrs) => {
                let mut current_index = 0;
                for (id, _) in gilrs.gamepads() {
                    if current_index == index {
                        return Some(id);
                    }
                    current_index += 1;
                }
                None
            }
            None => None,
        }
    }

    // Helper to get index from GamepadId (reverse of controller_id_from_index)
    fn controller_index_from_id(gilrs: &Gilrs, id: GamepadId) -> usize {
        let mut current_index = 0;
        for (pad_id, _) in gilrs.gamepads() {
            if pad_id == id {
                return current_index;
            }
            current_index += 1;
        }
        0 // Default to 0 if not found (shouldn't happen)
    }

    // Controller button state methods
    pub fn is_controller_button_pressed(&self, controller_index: usize, button: Button) -> bool {
        // Button is pressed when it's down now but wasn't down in the previous frame
        *self
            .controller_buttons_current
            .get(&(controller_index, button))
            .unwrap_or(&false)
            && !*self
                .controller_buttons_previous
                .get(&(controller_index, button))
                .unwrap_or(&false)
    }

    pub fn is_controller_button_released(&self, controller_index: usize, button: Button) -> bool {
        // Button is released when it's not down now but was down in the previous frame
        !*self
            .controller_buttons_current
            .get(&(controller_index, button))
            .unwrap_or(&false)
            && *self
                .controller_buttons_previous
                .get(&(controller_index, button))
                .unwrap_or(&false)
    }

    pub fn is_controller_button_down(&self, controller_index: usize, button: Button) -> bool {
        // Button is down when it's currently pressed
        *self
            .controller_buttons_current
            .get(&(controller_index, button))
            .unwrap_or(&false)
    }

    pub fn controller_axis_value(&self, controller_index: usize, axis: Axis) -> f32 {
        match &self.gilrs {
            Some(gilrs) => {
                if let Some(id) = self.controller_id_from_index(controller_index) {
                    gilrs.gamepad(id).value(axis)
                } else {
                    0.0
                }
            }
            None => 0.0,
        }
    }

    // Helper methods for common controller inputs
    pub fn controller_left_stick_x(&self, controller_index: usize) -> f32 {
        self.controller_axis_value(controller_index, Axis::LeftStickX)
    }

    pub fn controller_left_stick_y(&self, controller_index: usize) -> f32 {
        self.controller_axis_value(controller_index, Axis::LeftStickY)
    }

    pub fn controller_right_stick_x(&self, controller_index: usize) -> f32 {
        self.controller_axis_value(controller_index, Axis::RightStickX)
    }

    pub fn controller_right_stick_y(&self, controller_index: usize) -> f32 {
        self.controller_axis_value(controller_index, Axis::RightStickY)
    }

    pub fn controller_dpad_x(&self, controller_index: usize) -> f32 {
        self.controller_axis_value(controller_index, Axis::DPadX)
    }

    pub fn controller_dpad_y(&self, controller_index: usize) -> f32 {
        self.controller_axis_value(controller_index, Axis::DPadY)
    }

    // Get trigger values
    pub fn controller_left_trigger(&self, controller_index: usize) -> f32 {
        *self
            .controller_trigger_values
            .get(&(controller_index, Button::LeftTrigger2))
            .unwrap_or(&0.0)
    }

    pub fn controller_right_trigger(&self, controller_index: usize) -> f32 {
        *self
            .controller_trigger_values
            .get(&(controller_index, Button::RightTrigger2))
            .unwrap_or(&0.0)
    }

    // Get controller name
    pub fn controller_name(&self, controller_index: usize) -> Option<String> {
        match &self.gilrs {
            Some(gilrs) => self
                .controller_id_from_index(controller_index)
                .map(|id| gilrs.gamepad(id).name().to_string()),
            None => None,
        }
    }

    // Get all connected controller names
    pub fn controller_names(&self) -> Vec<String> {
        match &self.gilrs {
            Some(gilrs) => gilrs
                .gamepads()
                .map(|(_, gamepad)| gamepad.name().to_string())
                .collect(),
            None => Vec::new(),
        }
    }

    // Get raw stick vector (for joysticks)
    pub fn controller_stick_vector(
        &self,
        controller_index: usize,
        x_axis: Axis,
        y_axis: Axis,
    ) -> (f32, f32) {
        let x = self.controller_axis_value(controller_index, x_axis);
        let y = self.controller_axis_value(controller_index, y_axis);
        (x, y)
    }

    // Get left stick vector
    pub fn left_stick_vector(&self, controller_index: usize) -> (f32, f32) {
        self.controller_stick_vector(controller_index, Axis::LeftStickX, Axis::LeftStickY)
    }

    // Get right stick vector
    pub fn right_stick_vector(&self, controller_index: usize) -> (f32, f32) {
        self.controller_stick_vector(controller_index, Axis::RightStickX, Axis::RightStickY)
    }
}
