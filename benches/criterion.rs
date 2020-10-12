#[macro_use]
extern crate criterion;
extern crate num;
extern crate rand;

#[cfg(feature = "parallel")]
extern crate rayon;

extern crate accurate;

use std::ops::AddAssign;

use criterion::{Bencher, BenchmarkId, Criterion, Throughput};

use num::Float;

use rand::distributions::Standard;
use rand::prelude::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use accurate::dot::{Dot2, Dot3, Dot4, Dot5, Dot6, Dot7, Dot8, Dot9, NaiveDot, OnlineExactDot};
use accurate::sum::{NaiveSum, OnlineExactSum, Sum2, Sum3, Sum4, Sum5, Sum6, Sum7, Sum8, Sum9};
use accurate::traits::*;

fn mk_vec<T>(n: usize) -> Vec<T>
where
    Standard: Distribution<T>,
{
    let rng = rand::thread_rng();
    rng.sample_iter::<T, _>(&Standard).take(n).collect()
}

fn regular_add<F>(b: &mut Bencher, n: &usize)
where
    F: Float,
    Standard: Distribution<F>,
{
    let d = mk_vec::<F>(*n);
    b.iter(|| {
        let mut s = F::zero();
        for &x in &d {
            s = s + x;
        }
        criterion::black_box(s);
    });
}

fn regular_add_assign<F>(b: &mut Bencher, n: &usize)
where
    F: Float + AddAssign,
    Standard: Distribution<F>,
{
    let d = mk_vec::<F>(*n);
    b.iter(|| {
        let mut s = F::zero();
        for &x in &d {
            s += x;
        }
        criterion::black_box(s);
    });
}

fn fold<F>(b: &mut Bencher, n: &usize)
where
    F: Float,
    Standard: Distribution<F>,
{
    let d = mk_vec::<F>(*n);
    b.iter(|| {
        let s = d.iter().fold(F::zero(), |acc, &x| acc + x);
        criterion::black_box(s);
    });
}

fn sum_with<Acc, F>(b: &mut Bencher, n: &usize)
where
    F: Float,
    Acc: SumAccumulator<F>,
    Standard: Distribution<F>,
{
    let d = mk_vec::<F>(*n);
    b.iter(|| {
        let s = d.iter().cloned().sum_with_accumulator::<Acc>();
        criterion::black_box(s);
    });
}

fn regular_dot<F>(b: &mut Bencher, n: &usize)
where
    F: Float,
    Standard: Distribution<F>,
{
    let xs = mk_vec::<F>(*n);
    let ys = mk_vec::<F>(*n);

    b.iter(|| {
        let mut d = F::zero();
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            d = d + x * y
        }
        criterion::black_box(d);
    });
}

fn regular_dot_assign<F>(b: &mut Bencher, n: &usize)
where
    F: Float + AddAssign,
    Standard: Distribution<F>,
{
    let xs = mk_vec::<F>(*n);
    let ys = mk_vec::<F>(*n);

    b.iter(|| {
        let mut d = F::zero();
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            d += x * y
        }
        criterion::black_box(d);
    });
}

fn dot_fold<F>(b: &mut Bencher, n: &usize)
where
    F: Float,
    Standard: Distribution<F>,
{
    let xs = mk_vec::<F>(*n);
    let ys = mk_vec::<F>(*n);

    b.iter(|| {
        let d = xs
            .iter()
            .zip(ys.iter())
            .fold(F::zero(), |acc, (&x, &y)| acc + x * y);
        criterion::black_box(d);
    });
}

fn dot_with<Acc, F>(b: &mut Bencher, n: &usize)
where
    F: Float,
    Acc: DotAccumulator<F>,
    Standard: Distribution<F>,
{
    let xs = mk_vec::<F>(*n);
    let ys = mk_vec::<F>(*n);

    b.iter(|| {
        let d = xs
            .iter()
            .cloned()
            .zip(ys.iter().cloned())
            .dot_with_accumulator::<Acc>();
        criterion::black_box(d);
    });
}

#[cfg(feature = "parallel")]
fn parallel_sum_with<Acc, F>(b: &mut Bencher, n: &usize)
where
    F: Float + Copy + Send + Sync,
    Acc: ParallelSumAccumulator<F>,
    Standard: Distribution<F>,
{
    let d = mk_vec::<F>(*n);
    b.iter(|| {
        let s = d
            .par_iter()
            .map(|&x| x)
            .parallel_sum_with_accumulator::<Acc>();
        criterion::black_box(s);
    });
}

#[cfg(feature = "parallel")]
fn parallel_dot_with<Acc, F>(b: &mut Bencher, n: &usize)
where
    F: Float + Copy + Send + Sync,
    Acc: ParallelDotAccumulator<F>,
    Standard: Distribution<F>,
{
    let xs = mk_vec::<F>(*n);
    let ys = mk_vec::<F>(*n);

    b.iter(|| {
        let d = xs
            .par_iter()
            .zip(ys.par_iter())
            .map(|(&x, &y)| (x, y))
            .parallel_dot_with_accumulator::<Acc>();
        criterion::black_box(d);
    });
}

macro_rules! bench1 {
    ($c:expr, $name:expr, $f:ident, { $($tf:ty),* }) => {
        $(_bench($c, concat!($name, " on ", stringify!($tf)), $f::<$tf>);)*
    }
}

macro_rules! bench2 {
    ($c:expr, $name:expr, $f:ident, { $($tacc:ty),* }, $tfs:tt) => {
        $(bench2_aux! { $c, $name, $f, $tacc, $tfs })*
    }
}

macro_rules! bench2_aux {
    ($c:expr, $name:expr, $f:ident, $tacc:ty, { $($tf:ty),* }) => {
        $(_bench($c, concat!($name, " with ", stringify!($tacc), " on ", stringify!($tf)), $f::<$tacc, $tf>);)*
    }
}

fn _bench(c: &mut Criterion, id: &str, f: fn(&mut Bencher, &usize)) {
    let mut group = c.benchmark_group("all");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new(id, *size), size, f);
    }

    group.finish();
}

#[cfg(feature = "parallel")]
fn bench_parallel(c: &mut Criterion) {
    bench2! {
        c,
        "parallel sum",
        parallel_sum_with,
        {
            NaiveSum<_>,
            Sum2<_>, Sum3<_>, Sum4<_>, Sum5<_>, Sum6<_>, Sum7<_>, Sum8<_>, Sum9<_>,
            OnlineExactSum<_>
        },
        { f32, f64 }
    }

    bench2! {
        c,
        "parallel dot",
        parallel_dot_with,
        {
            NaiveDot<_>,
            Dot2<_>, Dot3<_>, Dot4<_>, Dot5<_>, Dot6<_>, Dot7<_>, Dot8<_>, Dot9<_>,
            OnlineExactDot<_>
        },
        { f32, f64 }
    }
}

#[cfg(not(feature = "parallel"))]
fn bench_parallel(_: &mut Criterion) {}

fn bench_serial(c: &mut Criterion) {
    bench1! { c, "add", regular_add, { f32, f64 } }
    bench1! { c, "add assign", regular_add_assign, { f32, f64 } }
    bench1! { c, "fold", fold, { f32, f64 } }

    bench2! {
        c,
        "sum",
        sum_with,
        {
            NaiveSum<_>,
            Sum2<_>, Sum3<_>, Sum4<_>, Sum5<_>, Sum6<_>, Sum7<_>, Sum8<_>, Sum9<_>,
            OnlineExactSum<_>
        },
        { f32, f64 }
    }

    bench1! { c, "dot", regular_dot, { f32, f64 } }
    bench1! { c, "dot assign", regular_dot_assign, { f32, f64 } }
    bench1! { c, "dot with fold", dot_fold, { f32, f64 } }

    bench2! {
        c,
        "dot",
        dot_with,
        {
            NaiveDot<_>,
            Dot2<_>, Dot3<_>, Dot4<_>, Dot5<_>, Dot6<_>, Dot7<_>, Dot8<_>, Dot9<_>,
            OnlineExactDot<_>
        },
        { f32, f64 }
    }
}

criterion_group!(serial, bench_serial);
criterion_group!(parallel, bench_parallel);
criterion_main!(serial, parallel);
