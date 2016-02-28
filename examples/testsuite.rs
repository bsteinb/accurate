extern crate accurate;

use std::env;
use std::io::prelude::*;
use std::io::BufReader;
use std::fs::{File, OpenOptions};
use std::str::FromStr;

use accurate::*;

type F = f64;

fn read_dots(f: File) -> (Vec<Vec<F>>, Vec<Vec<F>>, Vec<F>, Vec<F>) {
    let reader = BufReader::new(f);

    let mut xs = vec![];
    let mut ys = vec![];
    let mut ds = vec![];
    let mut cs = vec![];
    for (i, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        match i % 4 {
            0 => xs.push(read_vec(&line)),
            1 => ys.push(read_vec(&line)),
            2 => ds.push(read_num(&line)),
            3 => cs.push(read_num(&line)),
            _ => unreachable!()
        }
    }

    (xs, ys, ds, cs)
}

fn read_sums(f: File) -> (Vec<Vec<F>>, Vec<F>, Vec<F>) {
    let reader = BufReader::new(f);

    let mut zs = vec![];
    let mut ds = vec![];
    let mut cs = vec![];
    for (i, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        match i % 3 {
            0 => zs.push(read_vec(&line)),
            1 => ds.push(read_num(&line)),
            2 => cs.push(read_num(&line)),
            _ => unreachable!()
        }
    }

    (zs, ds, cs)
}

fn read_vec(line: &str) -> Vec<F> {
    let mut xs = vec![];
    for s in line.split(',') {
        xs.push(F::from_str(s).unwrap());
    }
    xs
}

fn read_num(line: &str) -> F {
    F::from_str(line).unwrap()
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
    println!("Testing dot product with `{}`.", name);
    let mut f = OpenOptions::new().write(true).truncate(true).create(true)
        .open(format!("{}.csv", name)).unwrap();
    for i in 0..xs.len() {
        let d = xs[i].iter().cloned().zip(ys[i].iter().cloned()).dot_with_accumulator::<Acc>();
        let e = ((d - ds[i]).abs() / ds[i].abs()).min(1.0).max(1.0e-16);

        println!("exact:  {}", ds[i]);
        println!("result: {}", d);
        println!("condition: {:.2e}, relative error: {:.2e}", cs[i], e);
        writeln!(&mut f, "{}, {}", cs[i], e).unwrap();
    }
    println!("");
}

macro_rules! sum {
    ($acct:path, $zs:expr, $ds:expr, $cs:expr) => {
        _sum::<$acct>(beautify(stringify!($acct)), $zs, $ds, $cs);
    }
}

fn _sum<Acc>(name: &str, zs: &[Vec<F>], ds: &[F], cs: &[F])
    where Acc: SumAccumulator<F>
{
    println!("Testing sum with `{}`.", name);
    let mut f = OpenOptions::new().write(true).truncate(true).create(true)
        .open(format!("{}.csv", name)).unwrap();
    for i in 0..zs.len() {
        let d = zs[i].iter().cloned().sum_with_accumulator::<Acc>();
        let e = ((d - ds[i]).abs() / ds[i].abs()).min(1.0).max(1.0e-16);

        println!("exact:  {}", ds[i]);
        println!("result: {}", d);
        println!("condition: {:.2e}, relative error: {:.2e}", cs[i], e);
        writeln!(&mut f, "{}, {}", cs[i], e).unwrap();
    }
    println!("");
}

fn main() {
    let fname_dots = env::args().nth(1).expect("<progname> illdot.csv illsum.csv");
    let fname_sums = env::args().nth(2).expect("<progname> illdot.csv illsum.csv");

    let f = File::open(fname_dots).unwrap();
    let (xs, ys, ds, cs) = read_dots(f);

    dot!(Dot2<_>, &xs, &ys, &ds, &cs);
    dot!(Dot3<_>, &xs, &ys, &ds, &cs);
    dot!(Dot4<_>, &xs, &ys, &ds, &cs);
    dot!(Dot5<_>, &xs, &ys, &ds, &cs);
    dot!(Dot6<_>, &xs, &ys, &ds, &cs);
    dot!(Dot7<_>, &xs, &ys, &ds, &cs);
    dot!(Dot8<_>, &xs, &ys, &ds, &cs);
    dot!(Dot9<_>, &xs, &ys, &ds, &cs);

    dot!(OnlineExactDot<_>, &xs, &ys, &ds, &cs);

    let f = File::open(fname_sums).unwrap();
    let (zs, ds, cs) = read_sums(f);

    sum!(Naive<_>, &zs, &ds, &cs);

    sum!(Sum2<_>, &zs, &ds, &cs);
    sum!(Sum3<_>, &zs, &ds, &cs);
    sum!(Sum4<_>, &zs, &ds, &cs);
    sum!(Sum5<_>, &zs, &ds, &cs);
    sum!(Sum6<_>, &zs, &ds, &cs);
    sum!(Sum7<_>, &zs, &ds, &cs);
    sum!(Sum8<_>, &zs, &ds, &cs);
    sum!(Sum9<_>, &zs, &ds, &cs);

    sum!(OnlineExactSum<_>, &zs, &ds, &cs);
}
