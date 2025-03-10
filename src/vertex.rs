#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)] // Added derives
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_uv: [f32; 2],
    pub normal: [f32; 3],
}

pub const VERTICES_SQUARE: &[Vertex] = &[
    // Front face
    Vertex {
        position: [-0.5, -0.5, 0.0],
        tex_uv: [0.0, 0.0],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.0],
        tex_uv: [1.0, 0.0],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.0],
        tex_uv: [1.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [-0.5, 0.5, 0.0],
        tex_uv: [0.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
];

pub const INDICES_SQUARE_REVERSED: &[u16] = &[
    0, 1, 2, 0, 2, 3, // Front
];

pub const INDICES_SQUARE: &[u16] = &[
    0, 1, 2, 1, 3, 2, // Front
];

pub const VERTICES_CUBE: &[Vertex] = &[
    // Front face
    Vertex {
        position: [-0.5, -0.5, 0.5],
        tex_uv: [0.0, 0.0],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.5],
        tex_uv: [1.0, 0.0],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        tex_uv: [1.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        tex_uv: [0.0, 1.0],
        normal: [0.0, 0.0, 1.0],
    },
    // Back face
    Vertex {
        position: [-0.5, -0.5, -0.5],
        tex_uv: [1.0, 0.0],
        normal: [0.0, 0.0, -1.0],
    },
    Vertex {
        position: [-0.5, 0.5, -0.5],
        tex_uv: [1.0, 1.0],
        normal: [0.0, 0.0, -1.0],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        tex_uv: [0.0, 1.0],
        normal: [0.0, 0.0, -1.0],
    },
    Vertex {
        position: [0.5, -0.5, -0.5],
        tex_uv: [0.0, 0.0],
        normal: [0.0, 0.0, -1.0],
    },
    // Left face
    Vertex {
        position: [-0.5, -0.5, -0.5],
        tex_uv: [0.0, 0.0],
        normal: [-1.0, 0.0, 0.0],
    },
    Vertex {
        position: [-0.5, -0.5, 0.5],
        tex_uv: [1.0, 0.0],
        normal: [-1.0, 0.0, 0.0],
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        tex_uv: [1.0, 1.0],
        normal: [-1.0, 0.0, 0.0],
    },
    Vertex {
        position: [-0.5, 0.5, -0.5],
        tex_uv: [0.0, 1.0],
        normal: [-1.0, 0.0, 0.0],
    },
    // Right face
    Vertex {
        position: [0.5, -0.5, 0.5],
        tex_uv: [0.0, 0.0],
        normal: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, -0.5],
        tex_uv: [1.0, 0.0],
        normal: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        tex_uv: [1.0, 1.0],
        normal: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        tex_uv: [0.0, 1.0],
        normal: [1.0, 0.0, 0.0],
    },
    // Top face
    Vertex {
        position: [-0.5, 0.5, 0.5],
        tex_uv: [0.0, 0.0],
        normal: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        tex_uv: [1.0, 0.0],
        normal: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        tex_uv: [1.0, 1.0],
        normal: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [-0.5, 0.5, -0.5],
        tex_uv: [0.0, 1.0],
        normal: [0.0, 1.0, 0.0],
    },
    // Bottom face
    Vertex {
        position: [-0.5, -0.5, -0.5],
        tex_uv: [0.0, 0.0],
        normal: [0.0, -1.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, -0.5],
        tex_uv: [1.0, 0.0],
        normal: [0.0, -1.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.5],
        tex_uv: [1.0, 1.0],
        normal: [0.0, -1.0, 0.0],
    },
    Vertex {
        position: [-0.5, -0.5, 0.5],
        tex_uv: [0.0, 1.0],
        normal: [0.0, -1.0, 0.0],
    },
];

pub const INDICIES_SQUARE: &[u16] = &[
    0, 1, 2, 2, 3, 0, // Front
    4, 5, 6, 6, 7, 4, // Back
    8, 9, 10, 10, 11, 8, // Left
    12, 13, 14, 14, 15, 12, // Right
    16, 17, 18, 18, 19, 16, // Top
    20, 21, 22, 22, 23, 20, // Bottom
];

pub fn create_vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    use std::mem::size_of;
    wgpu::VertexBufferLayout {
        array_stride: size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                // Position
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                // Tex UV
                offset: size_of::<[f32; 3]>() as wgpu::BufferAddress,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x2,
            },
            wgpu::VertexAttribute {
                // Normal
                offset: size_of::<[f32; 3]>() as u64 + size_of::<[f32; 2]>() as u64,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    }
}
