extern crate float;
extern crate num;
extern crate rand;
extern crate rayon;

extern crate accurate;

use std::io;
use std::io::prelude::*;
use std::fs::OpenOptions;

use float::Float as BigFloat;

use num::{Float, Integer, ToPrimitive, Zero};

use rand::Rng;

use rayon::prelude::*;

use accurate::traits::*;
use accurate::dot::{NaiveDot, Dot2, Dot3, Dot4, Dot5, Dot6, Dot7, Dot8, Dot9, OnlineExactDot};
use accurate::sum::{NaiveSum, Sum2, Sum3, Sum4, Sum5, Sum6, Sum7, Sum8, Sum9, OnlineExactSum};
use accurate::util::two_product;

type F = f64;

fn dot_exact<Iter>(iter: Iter) -> F
    where Iter: Iterator<Item = (F, F)>
{
    let mut acc = BigFloat::zero(2048);

    for (x, y) in iter {
        let a = BigFloat::from(x).with_precision(2048);
        let b = BigFloat::from(y).with_precision(2048);
        let c = a * b;
        acc = acc + c;
    }

    F::from(acc)
}

fn sum_exact<Iter>(iter: Iter) -> F
    where Iter: Iterator<Item = F>
{
    let mut acc = BigFloat::zero(2048);

    for x in iter {
        acc = acc + BigFloat::from(x).with_precision(2048);
    }

    F::from(acc)
}

fn gendot2(n: usize, cnd: F) -> (Vec<F>, Vec<F>, F, F) {
    let m = (n / 2).to_i32().unwrap();
    let eps = (-24.0).exp2();
    let l = (cnd.log2() / -(eps.log2())).floor().to_i32().unwrap();

    let mut rng = rand::thread_rng();
    let mut x;
    let mut y;

    if n.mod_floor(&2) == 0 {
        let c = (1..m - 1).map(|i| {
            let r = if l > 0 { i.mod_floor(&l) } else { 1 };
            rng.gen_range(0.0, 1.0) * eps.powi(r)
        }).collect::<Vec<_>>();

        x = vec![1.0];
        x.extend_from_slice(&c);
        x.push(0.5 / cnd);
        x.push(-1.0);
        x.extend(c.into_iter().map(|x| -x));
        x.push(0.5 / cnd);

        let b = (1..m - 1).map(|_| {
            rng.gen_range(0.0, 1.0)
        }).collect::<Vec<_>>();

        y = vec![1.0];
        y.extend_from_slice(&b);
        y.push(1.0);
        y.push(1.0);
        y.extend_from_slice(&b);
        y.push(1.0);
    } else {
        let c = (1..m).map(|i| {
            let r = i.mod_floor(&l);
            rng.gen_range(0.0, 1.0) * eps.powi(r)
        }).collect::<Vec<_>>();

        x = vec![1.0];
        x.extend_from_slice(&c);
        x.push(1.0 / cnd);
        x.push(-1.0);
        x.extend(c.into_iter().map(|x| -x));

        let b = (1..m).map(|_| {
            rng.gen_range(0.0, 1.0)
        }).collect::<Vec<_>>();

        y = vec![1.0];
        y.extend_from_slice(&b);
        y.push(1.0);
        y.push(1.0);
        y.extend_from_slice(&b);
    }

    assert_eq!(x.len(), n);
    assert_eq!(x.len(), y.len());

    let d = dot_exact(x.iter().cloned().zip(y.iter().cloned()));
    let absd = dot_exact(
        x.iter().cloned().map(|x| x.abs()).zip(
            y.iter().cloned().map(|x| x.abs()))
        );
    let c = 2.0 * absd / d.abs();

    (x, y, d, c)
}

fn gensum(n: usize, cnd: F) -> (Vec<F>, F, F) {
    let (x, y, _, _) = gendot2(n / 2, cnd);
    let mut z = vec![];
    for (x, y) in x.into_iter().zip(y.into_iter()) {
        let (a, b) = two_product(x, y);
        z.push(a);
        z.push(b);
    }
    let s = sum_exact(z.iter().cloned());
    let c = sum_exact(z.iter().cloned().map(|x| x.abs())) / s.abs();
    (z, s, c)
}

fn gen_dots() -> (Vec<Vec<F>>, Vec<Vec<F>>, Vec<F>, Vec<F>) {
    println!("Generating dot products.");
    let mut xs = vec![];
    let mut ys = vec![];
    let mut ds = vec![];
    let mut cs = vec![];

    let emax = 300;
    for e in 0 .. emax + 1 {
        print!("Working on exponent {} of {}", e, emax);
        for _ in 0 .. 10 {
            print!(".");
            io::stdout().flush().unwrap();
            let (x, y, d, c) = gendot2(1000, 10.0.powi(e));
            xs.push(x);
            ys.push(y);
            ds.push(d);
            cs.push(c);
        }
        println!(" done.");
    }

    (xs, ys, ds, cs)
}

fn gen_sums() -> (Vec<Vec<F>>, Vec<F>, Vec<F>) {
    println!("Generating sums.");
    let mut zs = vec![];
    let mut ss = vec![];
    let mut cs = vec![];

    let emax = 280;
    for e in 0 .. emax + 1 {
        print!("Working on exponent {} of {}", e, emax);
        for _ in 0 .. 10 {
            print!(".");
            io::stdout().flush().unwrap();
            let (z, s, c) = gensum(2000, 10.0.powi(e));
            zs.push(z);
            ss.push(s);
            cs.push(c);
        }
        println!(" done.");
    }

    (zs, ss, cs)
}

fn beautify(name: &str) -> &str {
    name.trim_matches(|c: char| !c.is_alphanumeric())
}

macro_rules! dot {
    ($acct:path, $xs:expr, $ys:expr, $ds:expr, $cs:expr) => {
        _dot::<$acct>(beautify(stringify!($acct)), $xs, $ys, $ds, $cs);
    }
}

fn _dot<Acc>(name: &str, xs: &[Vec<F>], ys: &[Vec<F>], ds: &[F], cs: &[F])
    where Acc: DotAccumulator<F>
{
    print!("Testing dot product with `{}`...", name);
    let mut f = OpenOptions::new().write(true).truncate(true).create(true)
        .open(format!("{}.csv", name)).unwrap();
    for i in 0..xs.len() {
        let d = xs[i].iter().cloned().zip(ys[i].iter().cloned()).dot_with_accumulator::<Acc>();
        let e = ((d - ds[i]).abs() / ds[i].abs()).min(1.0).max(1.0e-16);
        writeln!(&mut f, "{}, {}", cs[i], e).unwrap();
    }
    println!(" done.");
}

macro_rules! sum {
    ($acct:path, $zs:expr, $ds:expr, $cs:expr) => {
        _sum::<$acct>(beautify(stringify!($acct)), $zs, $ds, $cs);
    }
}

fn _sum<Acc>(name: &str, zs: &[Vec<F>], ds: &[F], cs: &[F])
    where Acc: SumAccumulator<F>
{
    print!("Testing sum with `{}`...", name);
    let mut f = OpenOptions::new().write(true).truncate(true).create(true)
        .open(format!("{}.csv", name)).unwrap();
    for i in 0..zs.len() {
        let d = zs[i].iter().cloned().sum_with_accumulator::<Acc>();
        let e = ((d - ds[i]).abs() / ds[i].abs()).min(1.0).max(1.0e-16);
        writeln!(&mut f, "{}, {}", cs[i], e).unwrap();
    }
    println!(" done.");
}

macro_rules! parallel_sum {
    ($acct:path, $zs:expr, $ds:expr, $cs:expr) => {
        _parallel_sum::<$acct>(beautify(stringify!($acct)), $zs, $ds, $cs);
    }
}

fn _parallel_sum<Acc>(name: &str, zs: &[Vec<F>], ds: &[F], cs: &[F])
    where Acc: ParallelSumAccumulator<F>
{
    print!("Testing parallel sum with `{}`...", name);
    let mut f = OpenOptions::new().write(true).truncate(true).create(true)
        .open(format!("Parallel{}.csv", name)).unwrap();
    for i in 0..zs.len() {
        let d = zs[i].par_iter().map(|&x| x).parallel_sum_with_accumulator::<Acc>();
        let e = ((d - ds[i]).abs() / ds[i].abs()).min(1.0).max(1.0e-16);
        writeln!(&mut f, "{}, {}", cs[i], e).unwrap();
    }
    println!(" done.");
}

fn main() {
    let (xs, ys, ds, cs) = gen_dots();

    dot!(NaiveDot<_>, &xs, &ys, &ds, &cs);

    dot!(Dot2<_>, &xs, &ys, &ds, &cs);
    dot!(Dot3<_>, &xs, &ys, &ds, &cs);
    dot!(Dot4<_>, &xs, &ys, &ds, &cs);
    dot!(Dot5<_>, &xs, &ys, &ds, &cs);
    dot!(Dot6<_>, &xs, &ys, &ds, &cs);
    dot!(Dot7<_>, &xs, &ys, &ds, &cs);
    dot!(Dot8<_>, &xs, &ys, &ds, &cs);
    dot!(Dot9<_>, &xs, &ys, &ds, &cs);

    dot!(OnlineExactDot<_>, &xs, &ys, &ds, &cs);

    let (zs, ds, cs) = gen_sums();

    sum!(NaiveSum<_>, &zs, &ds, &cs);

    sum!(Sum2<_>, &zs, &ds, &cs);
    sum!(Sum3<_>, &zs, &ds, &cs);
    sum!(Sum4<_>, &zs, &ds, &cs);
    sum!(Sum5<_>, &zs, &ds, &cs);
    sum!(Sum6<_>, &zs, &ds, &cs);
    sum!(Sum7<_>, &zs, &ds, &cs);
    sum!(Sum8<_>, &zs, &ds, &cs);
    sum!(Sum9<_>, &zs, &ds, &cs);

    sum!(OnlineExactSum<_>, &zs, &ds, &cs);

    parallel_sum!(NaiveSum<_>, &zs, &ds, &cs);

    parallel_sum!(Sum2<_>, &zs, &ds, &cs);
    parallel_sum!(Sum3<_>, &zs, &ds, &cs);
    parallel_sum!(Sum4<_>, &zs, &ds, &cs);
    parallel_sum!(Sum5<_>, &zs, &ds, &cs);
    parallel_sum!(Sum6<_>, &zs, &ds, &cs);
    parallel_sum!(Sum7<_>, &zs, &ds, &cs);
    parallel_sum!(Sum8<_>, &zs, &ds, &cs);
    parallel_sum!(Sum9<_>, &zs, &ds, &cs);

    parallel_sum!(OnlineExactSum<_>, &zs, &ds, &cs);
}
