use criterion::{Criterion, black_box, criterion_group, criterion_main};
use loupe_distance::{dot, l2_squared, scalar};

fn distance_benchmarks(c: &mut Criterion) {
    let dim = 128;
    let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.013).collect();
    let b: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 * 0.017).collect();

    c.bench_function("dot/scalar", |bencher| {
        bencher.iter(|| scalar::dot(black_box(&a), black_box(&b)))
    });
    c.bench_function("dot/simd", |bencher| {
        bencher.iter(|| dot(black_box(&a), black_box(&b)))
    });
    c.bench_function("l2/scalar", |bencher| {
        bencher.iter(|| scalar::l2_squared(black_box(&a), black_box(&b)))
    });
    c.bench_function("l2/simd", |bencher| {
        bencher.iter(|| l2_squared(black_box(&a), black_box(&b)))
    });
}

criterion_group!(benches, distance_benchmarks);
criterion_main!(benches);
