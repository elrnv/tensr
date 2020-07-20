#![allow(dead_code)]
pub use cgmath::prelude::*;
pub use flatk::*;
pub use rand::{FromEntropy, IsaacRng, Rng};
pub use std::ops::Mul;
pub use tensr::{Expr, IntoData, Matrix};

// Cgmath

pub fn matrix2_cgmath() -> cgmath::Matrix2<f64> {
    let mut rng = IsaacRng::from_entropy();
    cgmath::Matrix2::new(rng.gen(), rng.gen(), rng.gen(), rng.gen())
}

pub fn matrix3_cgmath() -> cgmath::Matrix3<f64> {
    let mut rng = IsaacRng::from_entropy();
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
    let mut rng = IsaacRng::from_entropy();
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
    let mut rng = IsaacRng::from_entropy();
    cgmath::Vector2::new(rng.gen(), rng.gen())
}

pub fn vector3_cgmath() -> cgmath::Vector3<f64> {
    let mut rng = IsaacRng::from_entropy();
    cgmath::Vector3::new(rng.gen(), rng.gen(), rng.gen())
}

pub fn vector4_cgmath() -> cgmath::Vector4<f64> {
    let mut rng = IsaacRng::from_entropy();
    cgmath::Vector4::new(rng.gen(), rng.gen(), rng.gen(), rng.gen())
}

// Local maths

pub fn matrix2() -> tensr::Matrix2<f64> {
    let mut rng = IsaacRng::from_entropy();
    tensr::Matrix2::new([[rng.gen(), rng.gen()], [rng.gen(), rng.gen()]])
}

pub fn matrix3() -> tensr::Matrix3<f64> {
    let mut rng = IsaacRng::from_entropy();
    tensr::Matrix3::new([
        [rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen()],
    ])
}

pub fn matrix4() -> tensr::Matrix4<f64> {
    let mut rng = IsaacRng::from_entropy();
    tensr::Matrix4::new([
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
    ])
}

pub fn vector2() -> tensr::Vector2<f64> {
    let mut rng = IsaacRng::from_entropy();
    tensr::Vector2::new([rng.gen(), rng.gen()])
}

pub fn vector3() -> tensr::Vector3<f64> {
    let mut rng = IsaacRng::from_entropy();
    tensr::Vector3::new([rng.gen(), rng.gen(), rng.gen()])
}

pub fn vector4() -> tensr::Vector4<f64> {
    let mut rng = IsaacRng::from_entropy();
    tensr::Vector4::new([rng.gen(), rng.gen(), rng.gen(), rng.gen()])
}
