extern crate accurate;
extern crate rand;

#[cfg(feature = "parallel")]
extern crate rayon;

use rand::distributions::Standard;
use rand::prelude::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use accurate::sum::OnlineExactSum;
use accurate::traits::*;

fn mk_vec<T>(n: usize) -> Vec<T>
where
    Standard: Distribution<T>,
{
    let rng = rand::thread_rng();
    rng.sample_iter::<T, _>(&Standard).take(n).collect()
}

#[test]
fn oes_add() {
    let xs = mk_vec::<f64>(100_000);
    let ys = mk_vec::<f64>(100_000);

    let s = OnlineExactSum::zero()
        .absorb(xs.iter().cloned())
        .absorb(ys.iter().cloned());

    let s1 = OnlineExactSum::zero().absorb(xs.iter().cloned());

    let s2 = OnlineExactSum::zero().absorb(ys.iter().cloned());

    assert_eq!(s.sum(), (s1 + s2).sum());
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_sum_oes() {
    let xs = mk_vec::<f64>(100_000);

    let s1 = xs
        .par_iter()
        .map(|&x| x)
        .parallel_sum_with_accumulator::<OnlineExactSum<_>>();
    let s2 = xs
        .iter()
        .cloned()
        .sum_with_accumulator::<OnlineExactSum<_>>();

    assert_eq!(s1, s2);
}
