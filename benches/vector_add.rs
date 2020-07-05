use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use flatk::*;
use tensr::*;
//use rayon::prelude::*;
//use approx::assert_relative_eq;

/// Generate a random vector of float values between -1 and 1.
pub fn random_vec(n: usize) -> Vec<f64> {
    use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};
    let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
    let range = Uniform::new(-1.0, 1.0);
    (0..n).map(move |_| rng.sample(range)).collect()
}

pub fn lazy_expr(a: &[f64], b: &[f64]) -> Vec<f64> {
    (a.expr() + b.expr()).eval()
}

pub fn lazy_expr_assign(mut a: Vec<f64>, b: &[f64]) -> Vec<f64> {
    *&mut a.expr_mut() += b.expr();
    a
}

pub fn manual(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len());
    out.extend(a.iter().zip(b.iter()).map(|(a, b)| a + b));
    out
}

pub fn manual_init(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len()];
    for ((a, b), out) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
        *out = a + b;
    }
    out
}

pub fn manual_assign(mut a: Vec<f64>, b: &[f64]) -> Vec<f64> {
    for (a, b) in a.iter_mut().zip(b.iter()) {
        *a += b;
    }
    a
}

fn vector_vector_add_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vector Vector Add");

    for &n in &[10000, 50000, 100000, 150000] {
        let a = random_vec(n);
        let b = random_vec(n);

        group.bench_with_input(
            BenchmarkId::new("Lazy Expr", n),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter_batched(
                    || (a.view(), b.view()),
                    |(a, b)| lazy_expr(a, b),
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(BenchmarkId::new("Manual", n), &(&a, &b), |bench, (a, b)| {
            bench.iter_batched(
                || (a.view(), b.view()),
                |(a, b)| manual(a, b),
                BatchSize::SmallInput,
            )
        });
        group.bench_with_input(
            BenchmarkId::new("Manual Init", n),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter_batched(
                    || (a.view(), b.view()),
                    |(a, b)| manual_init(a, b),
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Lazy Expr Assign", n),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter_batched(
                    || ((*a).clone(), b.view()),
                    |(a, b)| lazy_expr_assign(a, b),
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("Manual Assign", n),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter_batched(
                    || ((*a).clone(), b.view()),
                    |(a, b)| manual_assign(a, b),
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

criterion_group!(benches, vector_vector_add_benchmark);
criterion_main!(benches);
