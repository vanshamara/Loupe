//! Distance functions for comparing vectors.
//!
//! These scalar versions are the reference. The SIMD kernels added later have to
//! match them, so the tests in this file double as the correctness check.

/// How two vectors are compared. Smaller always means closer, so search code can
/// treat every metric the same way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    L2,
    Dot,
    Cosine,
}

impl Metric {
    /// Returns the distance between two vectors under this metric, where a smaller
    /// result means the vectors are closer.
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => l2_squared(a, b),
            Metric::Dot => -dot(a, b),
            Metric::Cosine => cosine(a, b),
        }
    }
}

/// Squared Euclidean distance. It is squared because search only compares
/// distances to each other, and skipping the square root is faster.
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Dot product of two vectors.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cosine distance, which is one minus cosine similarity. A zero-length vector has
/// no direction, so it is treated as the farthest possible (1.0).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let norm_a = dot(a, a).sqrt();
    let norm_b = dot(b, b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    1.0 - dot(a, b) / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_squared_of_equal_vectors_is_zero() {
        let v = [1.0, 2.0, 3.0];
        assert_eq!(l2_squared(&v, &v), 0.0);
    }

    #[test]
    fn l2_squared_known_value() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        assert_eq!(l2_squared(&a, &b), 25.0);
    }

    #[test]
    fn dot_known_value() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(dot(&a, &b), 32.0);
    }

    #[test]
    fn cosine_of_same_direction_is_zero() {
        let a = [1.0, 0.0];
        let b = [2.0, 0.0];
        assert_eq!(cosine(&a, &b), 0.0);
    }

    #[test]
    fn cosine_of_orthogonal_is_one() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert_eq!(cosine(&a, &b), 1.0);
    }

    #[test]
    fn cosine_of_opposite_direction_is_two() {
        let a = [1.0, 0.0];
        let b = [-1.0, 0.0];
        assert_eq!(cosine(&a, &b), 2.0);
    }

    #[test]
    fn cosine_handles_zero_vector() {
        let a = [0.0, 0.0];
        let b = [1.0, 2.0];
        assert_eq!(cosine(&a, &b), 1.0);
    }

    #[test]
    fn metric_l2_matches_l2_squared() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        assert_eq!(Metric::L2.distance(&a, &b), 25.0);
    }

    #[test]
    fn metric_dot_is_negated_so_smaller_is_closer() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(Metric::Dot.distance(&a, &b), -32.0);
    }
}
