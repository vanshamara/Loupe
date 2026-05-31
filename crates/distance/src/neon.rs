//! NEON distance kernels for aarch64. NEON is part of the aarch64 baseline, so
//! these run on every aarch64 target without a feature check.
//!
//! Each step works on 16 lanes with four independent accumulators. The four
//! accumulators run in parallel instead of waiting on each other, which keeps the
//! multiply-add units busy.

use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vsubq_f32};

/// Squared Euclidean distance, 16 lanes per step, then a scalar tail.
pub(crate) fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut a_chunks = a.chunks_exact(16);
    let mut b_chunks = b.chunks_exact(16);
    let mut sum = unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
            let pa = chunk_a.as_ptr();
            let pb = chunk_b.as_ptr();
            let d0 = vsubq_f32(vld1q_f32(pa), vld1q_f32(pb));
            let d1 = vsubq_f32(vld1q_f32(pa.add(4)), vld1q_f32(pb.add(4)));
            let d2 = vsubq_f32(vld1q_f32(pa.add(8)), vld1q_f32(pb.add(8)));
            let d3 = vsubq_f32(vld1q_f32(pa.add(12)), vld1q_f32(pb.add(12)));
            acc0 = vfmaq_f32(acc0, d0, d0);
            acc1 = vfmaq_f32(acc1, d1, d1);
            acc2 = vfmaq_f32(acc2, d2, d2);
            acc3 = vfmaq_f32(acc3, d3, d3);
        }
        let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        vaddvq_f32(acc)
    };
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}

/// Dot product, 16 lanes per step, then a scalar tail.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut a_chunks = a.chunks_exact(16);
    let mut b_chunks = b.chunks_exact(16);
    let mut sum = unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
            let pa = chunk_a.as_ptr();
            let pb = chunk_b.as_ptr();
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa), vld1q_f32(pb));
            acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(4)), vld1q_f32(pb.add(4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(8)), vld1q_f32(pb.add(8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(12)), vld1q_f32(pb.add(12)));
        }
        let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        vaddvq_f32(acc)
    };
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        sum += x * y;
    }
    sum
}
