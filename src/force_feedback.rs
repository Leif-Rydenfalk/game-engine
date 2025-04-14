use gilrs::GamepadId;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Manages multiple force feedback effects that can be active simultaneously
#[derive(Debug, Default)]
pub struct ForceFeedbackManager {
    /// All active feedback effects, mapped by their IDs
    effects: HashMap<String, ForceFeedbackEffect>,
    /// When the combined effect was last updated
    last_update: Option<Instant>,
}

/// A single force feedback effect with intensity values
#[derive(Debug, Clone)]
pub struct ForceFeedbackEffect {
    /// Strong motor intensity (0-65535)
    pub strong: u16,
    /// Weak motor intensity (0-65535)
    pub weak: u16,
    /// Optional expiration time for auto-removing effects
    pub expires_at: Option<Instant>,
}

impl ForceFeedbackManager {
    /// Create a new, empty force feedback manager
    pub fn new() -> Self {
        Self {
            effects: HashMap::new(),
            last_update: None,
        }
    }

    /// Set a named force feedback effect with the given intensities
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this effect
    /// * `strong` - Strong motor intensity (0-65535)
    /// * `weak` - Weak motor intensity (0-65535)
    /// * `duration` - Optional duration after which the effect will be automatically removed
    pub fn set_effect(&mut self, id: &str, strong: u16, weak: u16, duration: Option<Duration>) {
        // Create the expiration time if a duration was provided
        let expires_at = duration.map(|d| Instant::now() + d);

        self.effects.insert(
            id.to_string(),
            ForceFeedbackEffect {
                strong,
                weak,
                expires_at,
            },
        );

        // Update the last update time
        self.last_update = Some(Instant::now());
    }

    /// Remove a specific effect by ID
    pub fn remove_effect(&mut self, id: &str) -> bool {
        let removed = self.effects.remove(id).is_some();

        if removed {
            self.last_update = Some(Instant::now());
        }

        removed
    }

    /// Remove all active effects
    pub fn clear_effects(&mut self) {
        if !self.effects.is_empty() {
            self.effects.clear();
            self.last_update = Some(Instant::now());
        }
    }

    /// Get the combined values for all active effects
    ///
    /// This combines all effects using a simple maximum strategy - the strongest
    /// effect for each motor type (strong/weak) will be used.
    pub fn get_combined_values(&mut self) -> (u16, u16) {
        // Remove any expired effects first
        self.remove_expired_effects();

        // Calculate combined intensity using the maximum value approach
        let mut max_strong = 0u16;
        let mut max_weak = 0u16;

        for effect in self.effects.values() {
            max_strong = max_strong.max(effect.strong);
            max_weak = max_weak.max(effect.weak);
        }

        (max_strong, max_weak)
    }

    /// Remove all effects that have expired
    fn remove_expired_effects(&mut self) {
        let now = Instant::now();
        let expired_ids: Vec<String> = self
            .effects
            .iter()
            .filter_map(|(id, effect)| {
                if let Some(expires_at) = effect.expires_at {
                    if now >= expires_at {
                        Some(id.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        if !expired_ids.is_empty() {
            for id in expired_ids {
                self.effects.remove(&id);
            }
            self.last_update = Some(Instant::now());
        }
    }

    /// Check if there are any active effects
    pub fn has_active_effects(&self) -> bool {
        !self.effects.is_empty()
    }

    /// Get the time when the combined effect was last updated
    pub fn last_update(&self) -> Option<Instant> {
        self.last_update
    }
}

/// Extension trait for Input to provide force feedback functionality
pub trait ForceFeedbackExt {
    /// Set a named force feedback effect
    fn set_ff_effect(&mut self, id: &str, strong: u16, weak: u16, duration: Option<Duration>);

    /// Remove a specific force feedback effect
    fn remove_ff_effect(&mut self, id: &str) -> bool;

    /// Clear all force feedback effects
    fn clear_ff_effects(&mut self);

    /// Apply the current combined effect to the gamepad
    fn update_force_feedback(&mut self, controller_idx: usize) -> Result<(), String>;
}
