//! Scalar distance kernels. These are the portable reference that the SIMD
//! kernels are checked against, and the fallback on targets without a special
//! path.

/// Squared Euclidean distance, computed one element at a time.
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Dot product, computed one element at a time.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
