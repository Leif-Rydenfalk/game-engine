use gilrs::{Axis, Button, Event, EventType, GamepadId, Gilrs, GilrsBuilder};
use std::collections::HashMap;
use tracing::{debug, error, info, trace, warn};
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::KeyCode;

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

    // Only store the Gilrs instance, as it internally tracks all the controller state
    gilrs: Option<Gilrs>,
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

    pub fn update(&mut self) {
        // Save previous states
        self.keys_previous = self.keys_current.clone();
        self.mouse_buttons_previous = self.mouse_buttons_current.clone();
        self.controller_buttons_previous = self.controller_buttons_current.clone();

        // Reset deltas
        self.mouse_delta = (0.0, 0.0);
        self.scroll_delta = 0.0;

        // Update all controller button states first
        if let Some(gilrs) = &mut self.gilrs {
            // For each connected controller
            for (id, gamepad) in gilrs.gamepads() {
                let controller_index = Self::controller_index_from_id(gilrs, id);

                // We need to manually check common buttons to track state changes
                // This is a list of commonly used buttons
                let buttons_to_check = [
                    Button::South,
                    Button::East,
                    Button::North,
                    Button::West,
                    Button::LeftTrigger,
                    Button::RightTrigger,
                    Button::LeftTrigger2,
                    Button::RightTrigger2,
                    Button::Select,
                    Button::Start,
                    Button::Mode,
                    Button::LeftThumb,
                    Button::RightThumb,
                    Button::DPadUp,
                    Button::DPadDown,
                    Button::DPadLeft,
                    Button::DPadRight,
                ];

                for &button in &buttons_to_check {
                    let is_pressed = gamepad.is_pressed(button);
                    self.controller_buttons_current
                        .insert((controller_index, button), is_pressed);
                }
            }

            // Process controller events
            while let Some(Event { id, event, .. }) = gilrs.next_event() {
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
                    EventType::Connected => {
                        info!("Controller {} connected: {}", id, gilrs.gamepad(id).name());
                    }
                    EventType::Disconnected => {
                        info!("Controller disconnected: {}", id);

                        // Clean up button states for this controller
                        self.controller_buttons_current
                            .retain(|&(i, _), _| i != controller_index);
                        self.controller_buttons_previous
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

    // Get controller axis value with deadzone applied
    pub fn controller_axis_value_deadzone(
        &self,
        controller_index: usize,
        axis: Axis,
        deadzone: f32,
    ) -> f32 {
        let value = self.controller_axis_value(controller_index, axis);
        if value.abs() < deadzone {
            0.0
        } else {
            // Rescale the value to account for deadzone
            let sign = if value >= 0.0 { 1.0 } else { -1.0 };
            sign * (value.abs() - deadzone) / (1.0 - deadzone)
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

    pub fn controller_left_trigger(&self, controller_index: usize) -> f32 {
        self.controller_axis_value(controller_index, Axis::LeftZ)
    }

    pub fn controller_right_trigger(&self, controller_index: usize) -> f32 {
        self.controller_axis_value(controller_index, Axis::RightZ)
    }

    pub fn controller_dpad_x(&self, controller_index: usize) -> f32 {
        self.controller_axis_value(controller_index, Axis::DPadX)
    }

    pub fn controller_dpad_y(&self, controller_index: usize) -> f32 {
        self.controller_axis_value(controller_index, Axis::DPadY)
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

    // Get normalized 2D vector from a pair of controller axes (for joysticks)
    pub fn controller_stick_vector(
        &self,
        controller_index: usize,
        x_axis: Axis,
        y_axis: Axis,
        deadzone: f32,
    ) -> (f32, f32) {
        let x = self.controller_axis_value(controller_index, x_axis);
        let y = self.controller_axis_value(controller_index, y_axis);

        // Apply circular deadzone
        let length_squared = x * x + y * y;
        if length_squared < deadzone * deadzone {
            return (0.0, 0.0);
        }

        // Calculate normalized values
        let length = length_squared.sqrt();
        let normalized_x = x / length;
        let normalized_y = y / length;

        // Scale the magnitude to account for deadzone
        let normalized_length = (length - deadzone) / (1.0 - deadzone);
        let scaled_x = normalized_x * normalized_length;
        let scaled_y = normalized_y * normalized_length;

        (scaled_x, scaled_y)
    }

    // Get left stick vector with deadzone
    pub fn left_stick_vector(&self, controller_index: usize, deadzone: f32) -> (f32, f32) {
        self.controller_stick_vector(
            controller_index,
            Axis::LeftStickX,
            Axis::LeftStickY,
            deadzone,
        )
    }

    // Get right stick vector with deadzone
    pub fn right_stick_vector(&self, controller_index: usize, deadzone: f32) -> (f32, f32) {
        self.controller_stick_vector(
            controller_index,
            Axis::RightStickX,
            Axis::RightStickY,
            deadzone,
        )
    }

    // Enhanced function for analog trigger values (L2/R2)
    pub fn controller_trigger_value(
        &self,
        controller_index: usize,
        is_right_trigger: bool,
        deadzone: f32,
    ) -> f32 {
        let axis = if is_right_trigger {
            Axis::RightZ
        } else {
            Axis::LeftZ
        };
        let value = self.controller_axis_value(controller_index, axis);

        // Convert from [-1, 1] to [0, 1] range if necessary
        // Different controllers may report triggers differently
        let normalized = if value >= 0.0 {
            value // Already in [0, 1] range
        } else {
            (value + 1.0) / 2.0 // Convert from [-1, 1] to [0, 1]
        };

        // Apply deadzone
        if normalized < deadzone {
            0.0
        } else {
            // Rescale value after deadzone
            (normalized - deadzone) / (1.0 - deadzone)
        }
    }

    // Get analog value for any pressure-sensitive button (like on PS controllers)
    pub fn controller_button_pressure(
        &self,
        controller_index: usize,
        button: Button,
        deadzone: f32,
    ) -> f32 {
        // Special handling for trigger buttons which are often analog
        match button {
            Button::LeftTrigger2 => {
                self.controller_trigger_value(controller_index, false, deadzone)
            }
            Button::RightTrigger2 => {
                self.controller_trigger_value(controller_index, true, deadzone)
            }
            _ => {
                // For standard buttons, just return 0 or 1
                // Note: Only some controllers (like PS) support analog face buttons
                if self.is_controller_button_down(controller_index, button) {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}
