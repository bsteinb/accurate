extern crate criterion_plot;
extern crate num;
extern crate rand;
extern crate rayon;
extern crate rug;

extern crate accurate;

use std::io;
use std::io::prelude::*;
use std::path::Path;

use criterion_plot::prelude::*;

use rug::Float as BigFloat;

use num::{Float, Integer, ToPrimitive};

use rand::Rng;

use rayon::prelude::*;

use accurate::dot::{Dot2, Dot3, Dot4, Dot5, Dot6, Dot7, Dot8, Dot9, NaiveDot, OnlineExactDot};
use accurate::sum::{NaiveSum, OnlineExactSum, Sum2, Sum3, Sum4, Sum5, Sum6, Sum7, Sum8, Sum9};
use accurate::traits::*;
use accurate::util::two_product;

type F = f64;

fn dot_exact<Iter>(iter: Iter) -> F
where
    Iter: Iterator<Item = (F, F)>,
{
    let mut acc = BigFloat::new(2048);

    for (x, y) in iter {
        let a = BigFloat::with_val(2048, x);
        let b = BigFloat::with_val(2048, y);
        let c = a * b;
        acc = acc + c;
    }

    F::from(acc.to_f64())
}

fn sum_exact<Iter>(iter: Iter) -> F
where
    Iter: Iterator<Item = F>,
{
    let mut acc = BigFloat::new(2048);

    for x in iter {
        acc = acc + BigFloat::with_val(2048, x);
    }

    F::from(acc.to_f64())
}

fn gendot2(n: usize, cnd: F) -> (Vec<F>, Vec<F>, F, F) {
    let m = (n / 2).to_i32().unwrap();
    let eps = (-24.0).exp2();
    let l = (cnd.log2() / -(eps.log2())).floor().to_i32().unwrap();

    let mut rng = rand::thread_rng();
    let mut x;
    let mut y;

    if n.mod_floor(&2) == 0 {
        let c = (1..m - 1)
            .map(|i| {
                let r = if l > 0 { i.mod_floor(&l) } else { 1 };
                rng.gen_range(0.0, 1.0) * eps.powi(r)
            })
            .collect::<Vec<_>>();

        x = vec![1.0];
        x.extend_from_slice(&c);
        x.push(0.5 / cnd);
        x.push(-1.0);
        x.extend(c.into_iter().map(|x| -x));
        x.push(0.5 / cnd);

        let b = (1..m - 1)
            .map(|_| rng.gen_range(0.0, 1.0))
            .collect::<Vec<_>>();

        y = vec![1.0];
        y.extend_from_slice(&b);
        y.push(1.0);
        y.push(1.0);
        y.extend_from_slice(&b);
        y.push(1.0);
    } else {
        let c = (1..m)
            .map(|i| {
                let r = i.mod_floor(&l);
                rng.gen_range(0.0, 1.0) * eps.powi(r)
            })
            .collect::<Vec<_>>();

        x = vec![1.0];
        x.extend_from_slice(&c);
        x.push(1.0 / cnd);
        x.push(-1.0);
        x.extend(c.into_iter().map(|x| -x));

        let b = (1..m).map(|_| rng.gen_range(0.0, 1.0)).collect::<Vec<_>>();

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
        x.iter()
            .cloned()
            .map(|x| x.abs())
            .zip(y.iter().cloned().map(|x| x.abs())),
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
    for e in 0..emax + 1 {
        print!("Working on exponent {} of {}", e, emax);
        for _ in 0..10 {
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
    for e in 0..emax + 1 {
        print!("Working on exponent {} of {}", e, emax);
        for _ in 0..10 {
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

fn make_figure(filename: &'static str, title: &'static str) -> Figure {
    let mut f = Figure::new();
    f.set(Title(title))
        .set(Output(Path::new(filename)))
        .configure(Axis::BottomX, |a| {
            a.set(Label("condition number")).set(Scale::Logarithmic)
        })
        .configure(Axis::LeftY, |a| {
            a.set(Label("relative error"))
                .set(Scale::Logarithmic)
                .set(Range::Limits(1.0e-17, 10.0))
        })
        .configure(Key, |k| {
            k.set(Position::Inside(Vertical::Center, Horizontal::Right))
                .set(Title(""))
        });
    f
}

fn draw_figure(mut figure: Figure) {
    assert!(figure
        .draw()
        .expect("could not execute gnuplot")
        .wait_with_output()
        .expect("could not wait on gnuplot")
        .status
        .success());
}

fn make_color() -> Color {
    let mut rng = rand::thread_rng();
    Color::Rgb(rng.gen(), rng.gen(), rng.gen())
}

fn plot(figure: &mut Figure, label: &'static str, xs: &[F], ys: &[F]) {
    figure.plot(
        Points {
            x: &xs[..],
            y: &ys[..],
        },
        |l| {
            l.set(Label(label))
                .set(PointType::FilledCircle)
                .set(PointSize(0.2))
                .set(make_color())
        },
    );
}

macro_rules! dot {
    ($filename:expr, $title:expr, ($xs:expr, $ys:expr, $ds:expr, $cs:expr), $($acct:path),*) => {
        let mut figure = make_figure($filename, $title);
        $(dot_::<$acct>(&mut figure, beautify(stringify!($acct)), $xs, $ys, $ds, $cs);)*
        draw_figure(figure);
    }
}

fn dot_<Acc>(
    figure: &mut Figure,
    name: &'static str,
    xs: &[Vec<F>],
    ys: &[Vec<F>],
    ds: &[F],
    cs: &[F],
) where
    Acc: DotAccumulator<F>,
{
    print!("Testing dot product with `{}`...", name);
    let mut es = vec![];
    for i in 0..xs.len() {
        let d = xs[i]
            .iter()
            .cloned()
            .zip(ys[i].iter().cloned())
            .dot_with_accumulator::<Acc>();
        es.push(((d - ds[i]).abs() / ds[i].abs()).min(1.0).max(1.0e-16));
    }
    plot(figure, name, &cs[..], &es[..]);
    println!(" done.");
}

macro_rules! parallel_dot {
    ($filename:expr, $title:expr, ($xs:expr, $ys:expr, $ds:expr, $cs:expr), $($acct:path),*) => {
        let mut figure = make_figure($filename, $title);
        $(parallel_dot_::<$acct>(&mut figure, beautify(stringify!($acct)), $xs, $ys, $ds, $cs);)*
        draw_figure(figure);
    }
}

fn parallel_dot_<Acc>(
    figure: &mut Figure,
    name: &'static str,
    xs: &[Vec<F>],
    ys: &[Vec<F>],
    ds: &[F],
    cs: &[F],
) where
    Acc: ParallelDotAccumulator<F>,
{
    print!("Testing parallel dot with `{}`...", name);
    let mut es = vec![];
    for i in 0..xs.len() {
        let d = xs[i]
            .par_iter()
            .zip(ys[i].par_iter())
            .map(|(&x, &y)| (x, y))
            .parallel_dot_with_accumulator::<Acc>();
        es.push(((d - ds[i]).abs() / ds[i].abs()).min(1.0).max(1.0e-16));
    }
    plot(figure, name, &cs[..], &es[..]);
    println!(" done.");
}

macro_rules! sum {
    ($filename:expr, $title:expr, ($zs:expr, $ds:expr, $cs:expr), $($acct:path),*) => {
        let mut figure = make_figure($filename, $title);
        $(sum_::<$acct>(&mut figure, beautify(stringify!($acct)), $zs, $ds, $cs);)*
        draw_figure(figure);
    }
}

fn sum_<Acc>(figure: &mut Figure, name: &'static str, zs: &[Vec<F>], ds: &[F], cs: &[F])
where
    Acc: SumAccumulator<F>,
{
    print!("Testing sum with `{}`...", name);
    let mut es = vec![];
    for i in 0..zs.len() {
        let d = zs[i].iter().cloned().sum_with_accumulator::<Acc>();
        es.push(((d - ds[i]).abs() / ds[i].abs()).min(1.0).max(1.0e-16));
    }
    plot(figure, name, &cs[..], &es[..]);
    println!(" done.");
}

macro_rules! parallel_sum {
    ($filename:expr, $title:expr, ($zs:expr, $ds:expr, $cs:expr), $($acct:path),*) => {
        let mut figure = make_figure($filename, $title);
        $(parallel_sum_::<$acct>(&mut figure, beautify(stringify!($acct)), $zs, $ds, $cs);)*
        draw_figure(figure);
    }
}

fn parallel_sum_<Acc>(figure: &mut Figure, name: &'static str, zs: &[Vec<F>], ds: &[F], cs: &[F])
where
    Acc: ParallelSumAccumulator<F>,
{
    print!("Testing parallel sum with `{}`...", name);
    let mut es = vec![];
    for i in 0..zs.len() {
        let d = zs[i]
            .par_iter()
            .map(|&x| x)
            .parallel_sum_with_accumulator::<Acc>();
        es.push(((d - ds[i]).abs() / ds[i].abs()).min(1.0).max(1.0e-16));
    }
    plot(figure, name, &cs[..], &es[..]);
    println!(" done.");
}

fn main() {
    let (xs, ys, ds, cs) = gen_dots();

    dot! {
        "NaiveDot.svg",
        "NaiveDot, double precision",
        (&xs, &ys, &ds, &cs),
        NaiveDot<_>
    }
    dot! {
        "DotK.svg",
        "DotK for K = 2...9, double precision",
        (&xs, &ys, &ds, &cs),
        Dot2<_>, Dot3<_>, Dot4<_>, Dot5<_>, Dot6<_>, Dot7<_>, Dot8<_>, Dot9<_>
    }
    dot! {
        "OnlineExactDot.svg",
        "OnlineExactDot, double precision",
        (&xs, &ys, &ds, &cs),
        OnlineExactDot<_>
    };

    parallel_dot! {
        "ParallelNaiveDot.svg",
        "Parallel NaiveDot, double precision",
        (&xs, &ys, &ds, &cs),
        NaiveDot<_>
    };
    parallel_dot! {
        "ParallelDotK.svg",
        "Parallel DotK for K = 2...9, double precision",
        (&xs, &ys, &ds, &cs),
        Dot2<_>, Dot3<_>, Dot4<_>, Dot5<_>, Dot6<_>, Dot7<_>, Dot8<_>, Dot9<_>
    };
    parallel_dot! {
        "ParallelOnlineExactDot.svg",
        "Parallel OnlineExactDot, double precision",
        (&xs, &ys, &ds, &cs),
        OnlineExactDot<_>
    };

    let (zs, ds, cs) = gen_sums();

    sum! {
        "NaiveSum.svg",
        "NaiveSum, double precision",
        (&zs, &ds, &cs),
        NaiveSum<_>
    };
    sum! {
        "SumK.svg",
        "SumK for K = 2...9, double precision",
        (&zs, &ds, &cs),
        Sum2<_>, Sum3<_>, Sum4<_>, Sum5<_>, Sum6<_>, Sum7<_>, Sum8<_>, Sum9<_>
    };
    sum! {
        "OnlineExactSum.svg",
        "OnlineExactSum, double precision",
        (&zs, &ds, &cs),
        OnlineExactSum<_>
    };

    parallel_sum! {
        "ParallelNaiveSum.svg",
        "Parallel NaiveSum, double precision",
        (&zs, &ds, &cs),
        NaiveSum<_>
    };
    parallel_sum! {
        "ParallelSumK.svg",
        "Parallel SumK for K = 2...9, double precision",
        (&zs, &ds, &cs),
        Sum2<_>, Sum3<_>, Sum4<_>, Sum5<_>, Sum6<_>, Sum7<_>, Sum8<_>, Sum9<_>
    };
    parallel_sum! {
        "ParallelOnlineExactSum.svg",
        "Parallel OnlineExactSum, double precision",
        (&zs, &ds, &cs),
        OnlineExactSum<_>
    };
}
