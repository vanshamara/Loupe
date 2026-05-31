//! NEON distance kernels for aarch64. NEON is part of the aarch64 baseline, so
//! these run on every aarch64 target without a feature check.

use std::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vsubq_f32};

/// Squared Euclidean distance, four lanes at a time, then a scalar tail.
pub(crate) fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut a_chunks = a.chunks_exact(4);
    let mut b_chunks = b.chunks_exact(4);
    let mut acc = unsafe { vdupq_n_f32(0.0) };
    for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        unsafe {
            let va = vld1q_f32(chunk_a.as_ptr());
            let vb = vld1q_f32(chunk_b.as_ptr());
            let diff = vsubq_f32(va, vb);
            acc = vfmaq_f32(acc, diff, diff);
        }
    }
    let mut sum = unsafe { vaddvq_f32(acc) };
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}

/// Dot product, four lanes at a time, then a scalar tail.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut a_chunks = a.chunks_exact(4);
    let mut b_chunks = b.chunks_exact(4);
    let mut acc = unsafe { vdupq_n_f32(0.0) };
    for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        unsafe {
            let va = vld1q_f32(chunk_a.as_ptr());
            let vb = vld1q_f32(chunk_b.as_ptr());
            acc = vfmaq_f32(acc, va, vb);
        }
    }
    let mut sum = unsafe { vaddvq_f32(acc) };
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        sum += x * y;
    }
    sum
}
