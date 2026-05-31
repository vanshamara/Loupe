//! Distance functions for comparing vectors.
//!
//! The scalar versions in [`scalar`] are the portable reference. On aarch64 the
//! public functions use NEON, and the SIMD-vs-scalar tests check that the two
//! agree within a small tolerance.

pub mod scalar;

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(target_arch = "aarch64")]
use neon as backend;
#[cfg(not(target_arch = "aarch64"))]
use scalar as backend;

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
    backend::l2_squared(a, b)
}

/// Dot product of two vectors.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    backend::dot(a, b)
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

    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }

        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    fn random_vector(rng: &mut Rng, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rng.next_f32()).collect()
    }

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

    #[test]
    fn dispatched_dot_matches_scalar() {
        let mut rng = Rng::new(99);
        for dim in [1usize, 2, 3, 4, 5, 7, 8, 16, 31, 64, 127, 128] {
            let a = random_vector(&mut rng, dim);
            let b = random_vector(&mut rng, dim);
            let reference = scalar::dot(&a, &b);
            let actual = dot(&a, &b);
            let tolerance = 1e-3 * reference.abs().max(1.0);
            assert!(
                (actual - reference).abs() <= tolerance,
                "dim {dim}: {actual} vs {reference}"
            );
        }
    }

    #[test]
    fn dispatched_l2_squared_matches_scalar() {
        let mut rng = Rng::new(7);
        for dim in [1usize, 2, 3, 4, 5, 7, 8, 16, 31, 64, 127, 128] {
            let a = random_vector(&mut rng, dim);
            let b = random_vector(&mut rng, dim);
            let reference = scalar::l2_squared(&a, &b);
            let actual = l2_squared(&a, &b);
            let tolerance = 1e-3 * reference.abs().max(1.0);
            assert!(
                (actual - reference).abs() <= tolerance,
                "dim {dim}: {actual} vs {reference}"
            );
        }
    }
}
