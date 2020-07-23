#![allow(dead_code)]
pub use cgmath::prelude::*;
pub use flatk::*;
pub use rand::prelude::*;
pub use std::ops::Mul;
pub use tensr::{Expr, IntoData, Matrix};

static SEED: [u8; 32] = [3; 32];

// Cgmath

pub fn matrix2_cgmath() -> cgmath::Matrix2<f64> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    cgmath::Matrix2::new(rng.gen(), rng.gen(), rng.gen(), rng.gen())
}

pub fn matrix3_cgmath() -> cgmath::Matrix3<f64> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    cgmath::Matrix3::new(
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
    )
}

pub fn matrix4_cgmath() -> cgmath::Matrix4<f64> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    cgmath::Matrix4::new(
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
    )
}

pub fn vector2_cgmath() -> cgmath::Vector2<f64> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    cgmath::Vector2::new(rng.gen(), rng.gen())
}

pub fn vector3_cgmath() -> cgmath::Vector3<f64> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    cgmath::Vector3::new(rng.gen(), rng.gen(), rng.gen())
}

pub fn vector4_cgmath() -> cgmath::Vector4<f64> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    cgmath::Vector4::new(rng.gen(), rng.gen(), rng.gen(), rng.gen())
}

// Local maths

pub fn matrix2() -> tensr::Matrix2<f64> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    tensr::Matrix2::new([[rng.gen(), rng.gen()], [rng.gen(), rng.gen()]])
}

pub fn matrix3() -> tensr::Matrix3<f64> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    tensr::Matrix3::new([
        [rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen()],
    ])
}

pub fn matrix4() -> tensr::Matrix4<f64> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    tensr::Matrix4::new([
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
    ])
}

pub fn vector2() -> tensr::Vector2<f64> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    tensr::Vector2::new([rng.gen(), rng.gen()])
}

pub fn vector3() -> tensr::Vector3<f64> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    tensr::Vector3::new([rng.gen(), rng.gen(), rng.gen()])
}

pub fn vector4() -> tensr::Vector4<f64> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    tensr::Vector4::new([rng.gen(), rng.gen(), rng.gen(), rng.gen()])
}
