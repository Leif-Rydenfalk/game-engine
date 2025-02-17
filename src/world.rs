use crate::*;
use cgmath::Point3;
use hecs::World;

pub fn setup_camera_entity(world: &mut World) -> hecs::Entity {
    world.spawn((
        Transform {
            position: Point3::new(0.0, 1.0, 3.0),
            ..Default::default()
        },
        Camera::default(),
        CameraController::default(),
    ))
}

pub fn setup_player_entity(world: &mut World) -> hecs::Entity {
    world.spawn((Player::default(), Transform::default()))
}
