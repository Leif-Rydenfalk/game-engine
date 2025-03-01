use crate::*;
use cgmath::Point3;
use hecs::World;

// In world.rs, modify setup_camera_entity
pub fn setup_camera_entity(world: &mut World, window_size: Option<(u32, u32)>) -> hecs::Entity {
    // Calculate initial aspect ratio based on window size, or use default if not provided
    let aspect = if let Some((width, height)) = window_size {
        width as f32 / height as f32
    } else {
        16.0 / 9.0 // Default aspect ratio
    };

    world.spawn((
        Transform {
            position: Point3::new(0.0, 1.0, 3.0),
            ..Default::default()
        },
        Camera {
            aspect,
            ..Default::default()
        },
        CameraController::default(),
    ))
}

pub fn spawn_model_entity(
    world: &mut World,
    model_index: usize,
    position: Point3<f32>,
) -> hecs::Entity {
    world.spawn((
        Transform {
            position,
            ..Default::default()
        },
        ModelInstance { model: model_index },
    ))
}
