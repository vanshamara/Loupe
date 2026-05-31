//! AVX-512 and AVX2 distance kernels for x86_64, chosen at runtime by CPU
//! feature. Four accumulators per step keep the multiply-add units busy, the same
//! idea as the NEON kernels.

use std::arch::x86_64::*;

/// Squared Euclidean distance, using the widest SIMD the running CPU supports.
pub(crate) fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx512f") {
        unsafe { l2_squared_avx512(a, b) }
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { l2_squared_avx2(a, b) }
    } else {
        crate::scalar::l2_squared(a, b)
    }
}

/// Dot product, using the widest SIMD the running CPU supports.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx512f") {
        unsafe { dot_avx512(a, b) }
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { dot_avx2(a, b) }
    } else {
        crate::scalar::dot(a, b)
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut a_chunks = a.chunks_exact(32);
    let mut b_chunks = b.chunks_exact(32);
    let mut sum = unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
            let pa = chunk_a.as_ptr();
            let pb = chunk_b.as_ptr();
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa), _mm256_loadu_ps(pb), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(8)), _mm256_loadu_ps(pb.add(8)), acc1);
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(16)),
                _mm256_loadu_ps(pb.add(16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(24)),
                _mm256_loadu_ps(pb.add(24)),
                acc3,
            );
        }
        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        hsum256(acc)
    };
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        sum += x * y;
    }
    sum
}

#[target_feature(enable = "avx2,fma")]
unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut a_chunks = a.chunks_exact(32);
    let mut b_chunks = b.chunks_exact(32);
    let mut sum = unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
            let pa = chunk_a.as_ptr();
            let pb = chunk_b.as_ptr();
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa), _mm256_loadu_ps(pb));
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(8)), _mm256_loadu_ps(pb.add(8)));
            let d2 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(16)), _mm256_loadu_ps(pb.add(16)));
            let d3 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(24)), _mm256_loadu_ps(pb.add(24)));
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
            acc2 = _mm256_fmadd_ps(d2, d2, acc2);
            acc3 = _mm256_fmadd_ps(d3, d3, acc3);
        }
        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        hsum256(acc)
    };
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}

#[target_feature(enable = "avx")]
unsafe fn hsum256(v: __m256) -> f32 {
    let low = _mm256_castps256_ps128(v);
    let high = _mm256_extractf128_ps::<1>(v);
    let sum = _mm_add_ps(low, high);
    let shuffled = _mm_movehdup_ps(sum);
    let partial = _mm_add_ps(sum, shuffled);
    let upper = _mm_movehl_ps(shuffled, partial);
    _mm_cvtss_f32(_mm_add_ss(partial, upper))
}

#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    let mut a_chunks = a.chunks_exact(64);
    let mut b_chunks = b.chunks_exact(64);
    let mut sum = unsafe {
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
            let pa = chunk_a.as_ptr();
            let pb = chunk_b.as_ptr();
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(pa), _mm512_loadu_ps(pb), acc0);
            acc1 = _mm512_fmadd_ps(
                _mm512_loadu_ps(pa.add(16)),
                _mm512_loadu_ps(pb.add(16)),
                acc1,
            );
            acc2 = _mm512_fmadd_ps(
                _mm512_loadu_ps(pa.add(32)),
                _mm512_loadu_ps(pb.add(32)),
                acc2,
            );
            acc3 = _mm512_fmadd_ps(
                _mm512_loadu_ps(pa.add(48)),
                _mm512_loadu_ps(pb.add(48)),
                acc3,
            );
        }
        let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        _mm512_reduce_add_ps(acc)
    };
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        sum += x * y;
    }
    sum
}

#[target_feature(enable = "avx512f")]
unsafe fn l2_squared_avx512(a: &[f32], b: &[f32]) -> f32 {
    let mut a_chunks = a.chunks_exact(64);
    let mut b_chunks = b.chunks_exact(64);
    let mut sum = unsafe {
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
            let pa = chunk_a.as_ptr();
            let pb = chunk_b.as_ptr();
            let d0 = _mm512_sub_ps(_mm512_loadu_ps(pa), _mm512_loadu_ps(pb));
            let d1 = _mm512_sub_ps(_mm512_loadu_ps(pa.add(16)), _mm512_loadu_ps(pb.add(16)));
            let d2 = _mm512_sub_ps(_mm512_loadu_ps(pa.add(32)), _mm512_loadu_ps(pb.add(32)));
            let d3 = _mm512_sub_ps(_mm512_loadu_ps(pa.add(48)), _mm512_loadu_ps(pb.add(48)));
            acc0 = _mm512_fmadd_ps(d0, d0, acc0);
            acc1 = _mm512_fmadd_ps(d1, d1, acc1);
            acc2 = _mm512_fmadd_ps(d2, d2, acc2);
            acc3 = _mm512_fmadd_ps(d3, d3, acc3);
        }
        let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        _mm512_reduce_add_ps(acc)
    };
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}
