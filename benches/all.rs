#![feature(test)]

extern crate test;

extern crate ieee754;
extern crate num;
extern crate rand;

#[cfg(feature = "parallel")]
extern crate rayon;

extern crate accurate;

use std::ops::AddAssign;

use test::Bencher;

use num::Float;

use rand::{Rand, Rng};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use accurate::*;

const N: usize = 100_000;

fn mk_vec<T>(n: usize) -> Vec<T>
    where T: Rand
{
    let mut rng = rand::thread_rng();
    rng.gen_iter::<T>().take(n).collect()
}

fn regular_add<F>(b: &mut Bencher)
    where F: Float + Rand
{
    let d = mk_vec::<F>(N);
    b.iter(|| {
        let mut s = F::zero();
        for &x in &d {
            s = s + x;
        }
        test::black_box(s);
    });
}
#[bench] fn regular_add_f32(b: &mut Bencher) { regular_add::<f32>(b); }
#[bench] fn regular_add_f64(b: &mut Bencher) { regular_add::<f64>(b); }

fn regular_add_assign<F>(b: &mut Bencher)
    where F: Float + Rand + AddAssign
{
    let d = mk_vec::<F>(N);
    b.iter(|| {
        let mut s = F::zero();
        for &x in &d {
            s += x;
        }
        test::black_box(s);
    });
}
#[bench] fn regular_add_assign_f32(b: &mut Bencher) { regular_add_assign::<f32>(b); }
#[bench] fn regular_add_assign_f64(b: &mut Bencher) { regular_add_assign::<f64>(b); }

fn fold<F>(b: &mut Bencher)
    where F: Float + Rand
{
    let d = mk_vec::<F>(N);
    b.iter(|| {
        let s = d.iter().fold(F::zero(), |acc, &x| acc + x);
        test::black_box(s);
    });
}
#[bench] fn fold_f32(b: &mut Bencher) { fold::<f32>(b); }
#[bench] fn fold_f64(b: &mut Bencher) { fold::<f64>(b); }

fn sum_with<Acc, F>(b: &mut Bencher)
    where F: Float + Rand,
          Acc: SumAccumulator<F>
{
    let d = mk_vec::<F>(N);
    b.iter(|| {
        let s = d.iter().cloned().sum_with_accumulator::<Acc>();
        test::black_box(s);
    });
}
#[bench] fn sum_with_naive_f32(b: &mut Bencher) { sum_with::<Naive<_>, f32>(b); }
#[bench] fn sum_with_naive_f64(b: &mut Bencher) { sum_with::<Naive<_>, f64>(b); }

#[bench] fn sum_with_sum2_f32(b: &mut Bencher) { sum_with::<Sum2<_>, f32>(b); }
#[bench] fn sum_with_sum2_f64(b: &mut Bencher) { sum_with::<Sum2<_>, f64>(b); }

#[bench] fn sum_with_sum3_f32(b: &mut Bencher) { sum_with::<Sum3<_>, f32>(b); }
#[bench] fn sum_with_sum3_f64(b: &mut Bencher) { sum_with::<Sum3<_>, f64>(b); }

#[bench] fn sum_with_sum4_f32(b: &mut Bencher) { sum_with::<Sum4<_>, f32>(b); }
#[bench] fn sum_with_sum4_f64(b: &mut Bencher) { sum_with::<Sum4<_>, f64>(b); }

#[bench] fn sum_with_sum5_f32(b: &mut Bencher) { sum_with::<Sum5<_>, f32>(b); }
#[bench] fn sum_with_sum5_f64(b: &mut Bencher) { sum_with::<Sum5<_>, f64>(b); }

#[bench] fn sum_with_sum6_f32(b: &mut Bencher) { sum_with::<Sum6<_>, f32>(b); }
#[bench] fn sum_with_sum6_f64(b: &mut Bencher) { sum_with::<Sum6<_>, f64>(b); }

#[bench] fn sum_with_sum7_f32(b: &mut Bencher) { sum_with::<Sum7<_>, f32>(b); }
#[bench] fn sum_with_sum7_f64(b: &mut Bencher) { sum_with::<Sum7<_>, f64>(b); }

#[bench] fn sum_with_sum8_f32(b: &mut Bencher) { sum_with::<Sum8<_>, f32>(b); }
#[bench] fn sum_with_sum8_f64(b: &mut Bencher) { sum_with::<Sum8<_>, f64>(b); }

#[bench] fn sum_with_sum9_f32(b: &mut Bencher) { sum_with::<Sum9<_>, f32>(b); }
#[bench] fn sum_with_sum9_f64(b: &mut Bencher) { sum_with::<Sum9<_>, f64>(b); }

#[bench] fn sum_with_oes_f32(b: &mut Bencher) { sum_with::<OnlineExactSum<_>, f32>(b); }
#[bench] fn sum_with_oes_f64(b: &mut Bencher) { sum_with::<OnlineExactSum<_>, f64>(b); }

fn regular_dot<F>(b: &mut Bencher)
    where F: Float + Rand
{
    let xs = mk_vec::<F>(N);
    let ys = mk_vec::<F>(N);

    b.iter(|| {
        let mut d = F::zero();
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            d = d + x * y
        }
        test::black_box(d);
    });
}
#[bench] fn regular_dot_f32(b: &mut Bencher) { regular_dot::<f32>(b); }
#[bench] fn regular_dot_f64(b: &mut Bencher) { regular_dot::<f64>(b); }

fn regular_dot_assign<F>(b: &mut Bencher)
    where F: Float + Rand + AddAssign
{
    let xs = mk_vec::<F>(N);
    let ys = mk_vec::<F>(N);

    b.iter(|| {
        let mut d = F::zero();
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            d += x * y
        }
        test::black_box(d);
    });
}
#[bench] fn regular_dot_assign_f32(b: &mut Bencher) { regular_dot_assign::<f32>(b); }
#[bench] fn regular_dot_assign_f64(b: &mut Bencher) { regular_dot_assign::<f64>(b); }

fn dot_fold<F>(b: &mut Bencher)
    where F: Float + Rand
{
    let xs = mk_vec::<F>(N);
    let ys = mk_vec::<F>(N);

    b.iter(|| {
        let d = xs.iter().zip(ys.iter()).fold(F::zero(), |acc, (&x, &y)| acc + x * y);
        test::black_box(d);
    });
}
#[bench] fn dot_fold_f32(b: &mut Bencher) { dot_fold::<f32>(b); }
#[bench] fn dot_fold_f64(b: &mut Bencher) { dot_fold::<f64>(b); }

fn dot_with<Acc, F>(b: &mut Bencher)
    where F: Float + Rand,
          Acc: DotAccumulator<F>
{
    let xs = mk_vec::<F>(N);
    let ys = mk_vec::<F>(N);
    b.iter(|| {
        let d = xs.iter().cloned().zip(ys.iter().cloned()).dot_with_accumulator::<Acc>();
        test::black_box(d);
    });
}
#[bench] fn dot_with_dot2_f32(b: &mut Bencher) { dot_with::<Dot2<_>, f32>(b); }
#[bench] fn dot_with_dot2_f64(b: &mut Bencher) { dot_with::<Dot2<_>, f64>(b); }

#[bench] fn dot_with_dot3_f32(b: &mut Bencher) { dot_with::<Dot3<_>, f32>(b); }
#[bench] fn dot_with_dot3_f64(b: &mut Bencher) { dot_with::<Dot3<_>, f64>(b); }

#[bench] fn dot_with_dot4_f32(b: &mut Bencher) { dot_with::<Dot4<_>, f32>(b); }
#[bench] fn dot_with_dot4_f64(b: &mut Bencher) { dot_with::<Dot4<_>, f64>(b); }

#[bench] fn dot_with_dot5_f32(b: &mut Bencher) { dot_with::<Dot5<_>, f32>(b); }
#[bench] fn dot_with_dot5_f64(b: &mut Bencher) { dot_with::<Dot5<_>, f64>(b); }

#[bench] fn dot_with_dot6_f32(b: &mut Bencher) { dot_with::<Dot6<_>, f32>(b); }
#[bench] fn dot_with_dot6_f64(b: &mut Bencher) { dot_with::<Dot6<_>, f64>(b); }

#[bench] fn dot_with_dot7_f32(b: &mut Bencher) { dot_with::<Dot7<_>, f32>(b); }
#[bench] fn dot_with_dot7_f64(b: &mut Bencher) { dot_with::<Dot7<_>, f64>(b); }

#[bench] fn dot_with_dot8_f32(b: &mut Bencher) { dot_with::<Dot8<_>, f32>(b); }
#[bench] fn dot_with_dot8_f64(b: &mut Bencher) { dot_with::<Dot8<_>, f64>(b); }

#[bench] fn dot_with_dot9_f32(b: &mut Bencher) { dot_with::<Dot9<_>, f32>(b); }
#[bench] fn dot_with_dot9_f64(b: &mut Bencher) { dot_with::<Dot9<_>, f64>(b); }

#[bench] fn dot_with_oed_f32(b: &mut Bencher) { dot_with::<OnlineExactDot<_>, f32>(b); }
#[bench] fn dot_with_oed_f64(b: &mut Bencher) { dot_with::<OnlineExactDot<_>, f64>(b); }

#[cfg(feature = "parallel")]
fn parallel_sum_with<Acc, F>(b: &mut Bencher)
    where F: Float + Rand + Copy + Send + Sync,
          Acc: ParallelSumAccumulator<F>
{
    let d = mk_vec::<F>(N);
    b.iter(|| {
        let s = d.par_iter().map(|&x| x).parallel_sum_with_accumulator::<Acc>();
        test::black_box(s);
    });
}
#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_naive_f32(b: &mut Bencher) { parallel_sum_with::<Naive<_>, f32>(b); }
#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_naive_f64(b: &mut Bencher) { parallel_sum_with::<Naive<_>, f64>(b); }

#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum2_f32(b: &mut Bencher) { parallel_sum_with::<Sum2<_>, f32>(b); }
#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum2_f64(b: &mut Bencher) { parallel_sum_with::<Sum2<_>, f64>(b); }

#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum3_f32(b: &mut Bencher) { parallel_sum_with::<Sum3<_>, f32>(b); }
#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum3_f64(b: &mut Bencher) { parallel_sum_with::<Sum3<_>, f64>(b); }

#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum4_f32(b: &mut Bencher) { parallel_sum_with::<Sum4<_>, f32>(b); }
#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum4_f64(b: &mut Bencher) { parallel_sum_with::<Sum4<_>, f64>(b); }

#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum5_f32(b: &mut Bencher) { parallel_sum_with::<Sum5<_>, f32>(b); }
#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum5_f64(b: &mut Bencher) { parallel_sum_with::<Sum5<_>, f64>(b); }

#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum6_f32(b: &mut Bencher) { parallel_sum_with::<Sum6<_>, f32>(b); }
#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum6_f64(b: &mut Bencher) { parallel_sum_with::<Sum6<_>, f64>(b); }

#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum7_f32(b: &mut Bencher) { parallel_sum_with::<Sum7<_>, f32>(b); }
#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum7_f64(b: &mut Bencher) { parallel_sum_with::<Sum7<_>, f64>(b); }

#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum8_f32(b: &mut Bencher) { parallel_sum_with::<Sum8<_>, f32>(b); }
#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum8_f64(b: &mut Bencher) { parallel_sum_with::<Sum8<_>, f64>(b); }

#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum9_f32(b: &mut Bencher) { parallel_sum_with::<Sum9<_>, f32>(b); }
#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_sum9_f64(b: &mut Bencher) { parallel_sum_with::<Sum9<_>, f64>(b); }

#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_oes_f32(b: &mut Bencher) { parallel_sum_with::<OnlineExactSum<_>, f32>(b); }
#[cfg(feature = "parallel")]
#[bench] fn parallel_sum_with_oes_f64(b: &mut Bencher) { parallel_sum_with::<OnlineExactSum<_>, f64>(b); }
