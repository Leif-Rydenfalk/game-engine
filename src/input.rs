// input_system.rs
use std::collections::HashMap;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::KeyCode;

#[derive(Default)]
pub struct Input {
    keys_current: HashMap<KeyCode, ElementState>,
    keys_previous: HashMap<KeyCode, ElementState>,
    mouse_buttons_current: HashMap<MouseButton, ElementState>,
    mouse_buttons_previous: HashMap<MouseButton, ElementState>,
    mouse_position: (f64, f64),
    mouse_delta: (f64, f64),
    scroll_delta: f64,
}

impl Input {
    pub fn new() -> Self {
        Self::default()
    }

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
        self.keys_previous = self.keys_current.clone();
        self.mouse_buttons_previous = self.mouse_buttons_current.clone();
        self.mouse_delta = (0.0, 0.0);
        self.scroll_delta = 0.0;
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
}
