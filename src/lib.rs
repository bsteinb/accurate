#![deny(missing_docs)]

//! A collection of (more or less) accurate floating point algorithms

#![cfg_attr(feature = "unstable", feature(op_assign_traits))]
#![cfg_attr(feature = "unstable", feature(augmented_assignments))]

extern crate ieee754;
extern crate num;

use std::ops::Add;

use ieee754::Ieee754;

use num::traits::{Float, PrimInt, Zero, One, ToPrimitive};

/// Accumulates terms of a sum
pub trait SumAccumulator<F>: Add<F, Output = Self> + From<F> + Default {
    /// The sum of all terms accumulated so far
    fn sum(&self) -> F;

    /// Absorb the items of an iterator into the accumulator
    ///
    /// # Examples
    ///
    /// ```
    /// use accurate::*;
    ///
    /// let s = Sum2::default().absorb(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(6.0f64, s.sum())
    /// ```
    fn absorb<I>(self, it: I) -> Self
        where I: IntoIterator<Item = F>
    {
        it.into_iter().fold(self, |acc, x| acc + x)
    }
}

/// Accumulates terms of a dot product
pub trait DotAccumulator<F>: Add<(F, F), Output = Self> + From<F> + Default {
    /// The dot product of all terms accumulated so far
    fn dot(&self) -> F;

    /// Absorb the items of an iterator into the accumulator
    ///
    /// # Examples
    ///
    /// ```
    /// use accurate::*;
    ///
    /// let x = vec![1.0, 2.0, 3.0];
    /// let y = x.clone();
    ///
    /// let d = Dot2::default().absorb(x.into_iter().zip(y.into_iter()));
    /// assert_eq!(14.0f64, d.dot())
    /// ```
    fn absorb<I>(self, it: I) -> Self
        where I: IntoIterator<Item = (F, F)>
    {
        it.into_iter().fold(self, |acc, x| acc + x)
    }
}

/// Sum transformation
///
/// Transforms a sum `a + b` into the pair `(x, y)` where
///
/// ```not_rust
/// x = fl(a + b)
/// ```
///
/// is the sum of `a` and `b` with floating point rounding applied and
///
/// ```not_rust
/// y = a + b - x
/// ```
///
/// is the remainder of the addition.
///
/// # References
///
/// From Knuth's AoCP, Volume 2: Seminumerical Algorithms
pub fn two_sum<F>(a: F, b: F) -> (F, F)
    where F: Float
{
    let x = a + b;
    let z = x - a;
    let y = (a - (x - z)) + (b - z);
    (x , y)
}

/// Naive floating point summation
///
/// ![](https://rockshrub.de/accurate/Naive.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let s = Naive::default() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
#[derive(Copy, Clone)]
pub struct Naive<F>(F);

impl<F> Add<F> for Naive<F>
    where F: Float
{
    type Output = Self;

    fn add(self, rhs: F) -> Self::Output {
        Naive(self.0 + rhs)
    }
}

impl<F> SumAccumulator<F> for Naive<F>
    where F: Float
{
    fn sum(&self) -> F {
        self.0
    }
}

impl<F> From<F> for Naive<F>
    where F: Float
{
    fn from(x: F) -> Self {
        Naive(x)
    }
}

impl<F> Default for Naive<F>
    where F: Float
{
    fn default() -> Self {
        Self::from(F::zero())
    }
}

/// Calculates a sum using cascaded accumulators for the remainder terms
///
/// See also `Sum2`... `Sum9`.
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
#[derive(Copy, Clone)]
pub struct SumK<F, C> {
    s: F,
    c: C
}

impl<F, C> SumAccumulator<F> for SumK<F, C>
    where F: Float,
          C: SumAccumulator<F> + Clone
{
    fn sum(&self) -> F {
        self.c.clone().add(self.s).sum()
    }
}

impl<F, C> Add<F> for SumK<F, C>
    where F: Float,
          C: SumAccumulator<F>
{
    type Output = Self;

    fn add(self, rhs: F) -> Self::Output {
        let (x, y) = two_sum(self.s, rhs);
        SumK { s: x, c: self.c + y }
    }
}

impl<F, C> From<F> for SumK<F, C>
    where F: Float,
          C: SumAccumulator<F>
{
    fn from(x: F) -> Self {
        SumK { s: x, c: C::from(F::zero()) }
    }
}

impl<F, C> Default for SumK<F, C>
    where F: Float,
          C: SumAccumulator<F>
{
    fn default() -> Self {
        Self::from(F::zero())
    }
}

/// SumK with two cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let s = Sum2::default() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Sum2<F> = SumK<F, Naive<F>>;

/// SumK with three cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let s = Sum3::default() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Sum3<F> = SumK<F, SumK<F, Naive<F>>>;

/// SumK with four cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let s = Sum4::default() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Sum4<F> = SumK<F, SumK<F, SumK<F, Naive<F>>>>;

/// SumK with five cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let s = Sum5::default() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Sum5<F> = SumK<F, SumK<F, SumK<F, SumK<F, Naive<F>>>>>;

/// SumK with six cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let s = Sum6::default() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Sum6<F> = SumK<F, SumK<F, SumK<F, SumK<F, SumK<F, Naive<F>>>>>>;

/// SumK with seven cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let s = Sum7::default() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Sum7<F> = SumK<F, SumK<F, SumK<F, SumK<F, SumK<F, SumK<F, Naive<F>>>>>>>;

/// SumK with eight cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let s = Sum8::default() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Sum8<F> = SumK<F, SumK<F, SumK<F, SumK<F, SumK<F, SumK<F, SumK<F, Naive<F>>>>>>>>;

/// SumK with nine cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let s = Sum9::default() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Sum9<F> = SumK<F, SumK<F, SumK<F, SumK<F, SumK<F, SumK<F, SumK<F, SumK<F, Naive<F>>>>>>>>>;

/// Product transformation
///
/// Transforms a product `a * b` into the pair `(x, y)` so that
///
/// ```not_rust
/// x = fl(a * b)
/// ```
///
/// is the product of `a` and `b` with floating point rounding applied and
///
/// ```not_rust
/// y = a * b - x
/// ```
///
/// is the remainder of the product.
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub fn two_product_fma<F>(a: F, b: F) -> (F, F)
    where F: Float
{
    let x = a * b;
    let y = a.mul_add(b, -x);
    (x, y)
}

/// Calculates a dot product using both product transformation and cascaded accumulators
///
/// See also `Dot2`... `Dot9`.
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
#[derive(Copy, Clone)]
pub struct DotK<F, R> {
    p: F,
    r: R
}

impl<F, R> DotAccumulator<F> for DotK<F, R>
    where F: Float,
          R: SumAccumulator<F> + Clone
{
    fn dot(&self) -> F {
        self.r.clone().add(self.p).sum()
    }
}

impl<F, R> Add<(F, F)> for DotK<F, R>
    where F: Float,
          R: SumAccumulator<F>
{
    type Output = Self;

    fn add(self, (a, b): (F, F)) -> Self {
        //let (a, b) = ab;
        let (h, r1) = two_product_fma(a, b);
        let (p, r2) = two_sum(self.p, h);
        DotK { p: p, r: (self.r + r1) + r2 }
    }
}

impl<F, R> From<F> for DotK<F, R>
    where F: Float,
          R: SumAccumulator<F>
{
    fn from(x: F) -> Self {
        DotK { p: x, r: R::from(F::zero()) }
    }
}

impl<F, R> Default for DotK<F, R>
    where F: Float,
          R: SumAccumulator<F>
{
    fn default() -> Self {
        DotK::from(F::zero())
    }
}

/// DotK with two cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let d = Dot2::default() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Dot2<F> = DotK<F, Naive<F>>;

/// DotK with three cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let d = Dot2::default() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Dot3<F> = DotK<F, Sum2<F>>;

/// DotK with four cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let d = Dot2::default() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Dot4<F> = DotK<F, Sum3<F>>;

/// DotK with five cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let d = Dot2::default() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Dot5<F> = DotK<F, Sum4<F>>;

/// DotK with six cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let d = Dot2::default() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Dot6<F> = DotK<F, Sum5<F>>;

/// DotK with seven cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let d = Dot2::default() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Dot7<F> = DotK<F, Sum6<F>>;

/// DotK with eight cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let d = Dot2::default() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Dot8<F> = DotK<F, Sum7<F>>;

/// DotK with nine cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let d = Dot2::default() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Dot9<F> = DotK<F, Sum8<F>>;

trait HalfUlp {
    fn is_half_ulp(self) -> bool;
    fn half_ulp(self) -> Self;
}

impl<F> HalfUlp for F
    where F: Float + Ieee754,
          F::Significand: PrimInt
{
    fn is_half_ulp(self) -> bool {
        let (_, _, m) = self.decompose_raw();
        m.count_ones() == 1
    }

    fn half_ulp(self) -> Self {
        self.ulp().unwrap_or(Self::zero()) / F::one().exp2()
    }
}

trait Magnify {
    fn magnify(self) -> Self;
}

impl<F> Magnify for F
    where F: Ieee754,
          F::Significand: PrimInt
{
    fn magnify(self) -> Self {
        let (s, e, m) = self.decompose_raw();
        Self::recompose_raw(s, e, m | F::Significand::one())
    }
}

fn round3<F>(s0: F, s1: F, tau: F) -> F
    where F: Float + HalfUlp + Magnify
{
    if s1.is_half_ulp() && s1.signum() == tau {
        s1.magnify() + s0
    } else {
        s0
    }
}

/// Calculates the correctly rounded sum of the numbers in the slice `xs`
///
/// This algorithm works in place by mutating the contents of the slice. It is used by
/// `OnlineExactSum`.
///
/// # References
///
/// Based on [Zhu and Hayes 09](http://dx.doi.org/10.1137/070710020)
pub fn i_fast_sum_in_place<F>(xs: &mut [F]) -> F
    where F: Float + Ieee754,
          F::Significand: PrimInt
{
    let mut n = xs.len();
    i_fast_sum_in_place_aux(xs, &mut n, true)
}

fn i_fast_sum_in_place_aux<F>(xs: &mut [F], n: &mut usize, recurse: bool) -> F
    where F: Float + Ieee754,
          F::Significand: PrimInt
{
    // Step 1
    let mut s = F::zero();

    // Step 2
    for i in 0 .. *n {
        let (a, b) = two_sum(s, xs[i]);
        s = a;
        xs[i] = b;
    }

    // Step 3
    loop {
        // Step 3(1)
        let mut count: usize = 0; // slices are indexed from 0
        let mut st = F::zero();
        let mut sm = F::zero();

        // Step 3(2)
        for i in 0 .. *n {
            // Step 3(2)(a)
            let (a, b) = two_sum(st, xs[i]);
            st = a;
            // Step 3(2)(b)
            if b.abs() != F::zero() {
                xs[count] = b;
                // Step 3(2)(b)(i)
                count = count + 1;
                // Step 3(2)(b)(ii)
                sm = sm.max(st.abs());
            }
        }

        // Step 3(3)
        let em = F::from(count).unwrap() * sm.half_ulp();

        // Step 3(4)
        let (a, b) = two_sum(s, st);
        s = a;
        st = b;
        xs[count] = st;
        *n = count + 1;

        // Step 3(5)
        if (em == F::zero()) || (em < s.half_ulp()) {
            // Step 3(5)(a)
            if !recurse { return s; }

            // Step 3(5)(b)
            let (w1, e1) = two_sum(st, em);
            // Step 3(5)(c)
            let (w2, e2) = two_sum(st, -em);

            // Step 3(5)(d)
            if (w1 + s != s) || (w2 + s != s) || (round3(s, w1, e1) != s) || (round3(s, w2, e2) != s) {
                // Step 3(5)(d)(i)
                let mut s1 = i_fast_sum_in_place_aux(xs, n, false);

                // Step 3(5)(d)(ii)
                let (a, b) = two_sum(s, s1);
                s = a;
                s1 = b;

                // Step 3(5)(d)(iii)
                let s2 = i_fast_sum_in_place_aux(xs, n, false);

                // Step 3(5)(d)(iv)
                s = round3(s, s1, s2);
            }

            // Step 3(5)(e)
            return s;
        }
    }
}

/// Describes the layout of an IEEE754 number
pub trait Ieee754Ext: Ieee754 {
    /// The length of the number format's mantissa field in bits
    fn mantissa_length() -> u32;
    /// The length of the number format's exponent field in bits
    fn exponent_length() -> u32;
}

impl Ieee754Ext for f32 {
    fn mantissa_length() -> u32 { 24 }
    fn exponent_length() -> u32 { 8 }
}

impl Ieee754Ext for f64 {
    fn mantissa_length() -> u32 { 53 }
    fn exponent_length() -> u32 { 11 }
}

/// Calculates a sum using separate accumulators for each possible exponent
///
/// ![](https://rockshrub.de/accurate/OnlineExactSum.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let s = OnlineExactSum::default() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Zhu and Hayes 10](http://dx.doi.org/10.1145/1824801.1824815)
#[derive(Clone)]
pub struct OnlineExactSum<F> {
    i: usize,
    a1: Vec<F>,
    a2: Vec<F>
}

impl<F> SumAccumulator<F> for OnlineExactSum<F>
    where F: Float + Ieee754Ext,
          F::Significand: PrimInt,
          F::RawExponent: PrimInt
{
    fn sum(&self) -> F {
        // Step 5
        let mut a = Vec::with_capacity(2 * 2.pow(F::exponent_length()));
        a.extend(self.a1.iter().cloned().filter(|&x| x.abs() != F::zero()));
        a.extend(self.a2.iter().cloned().filter(|&x| x.abs() != F::zero()));

        // Step 6
        i_fast_sum_in_place(&mut a[..])
    }
}

impl<F> Add<F> for OnlineExactSum<F>
    where F:Float + Ieee754Ext,
          F::RawExponent: PrimInt
{
    type Output = Self;

    fn add(mut self, rhs: F) -> Self::Output {
        // Step 4(2)
        let j = rhs.decompose_raw().1.to_usize().unwrap();

        // Step 4(3)
        let (a, e) = two_sum(self.a1[j], rhs);
        self.a1[j] = a;

        // Step 4(4)
        self.a2[j] = self.a2[j] + e;

        // Step 4(5)
        self.i += 1;

        // Step 4(6)
        if self.i >= 2.pow(F::mantissa_length() / 2) {
            // Step 4(6)(a)
            let mut b1 = vec![F::zero(); 2.pow(F::exponent_length())];
            let mut b2 = vec![F::zero(); 2.pow(F::exponent_length())];

            // Step 4(6)(b)
            for &y in self.a1.iter().chain(self.a2.iter()) {
                // Step 4(6)(b)(i)
                let j = y.decompose_raw().1.to_usize().unwrap();

                // Step 4(6)(b)(ii)
                let (b, e) = two_sum(b1[j], y);
                b1[j] = b;

                // Step 4(6)(b)(iii)
                b2[j] = b2[j] + e;
            }

            // Step 4(6)(c)
            self.a1 = b1;
            self.a2 = b2;

            // Step 4(6)(d)
            self.i = 2 * 2.pow(F::exponent_length());
        }

        self
    }
}

impl<F> From<F> for OnlineExactSum<F>
    where F: Float + Ieee754Ext,
          F::Significand: PrimInt,
          F::RawExponent: PrimInt
{
    fn from(x: F) -> Self {
        OnlineExactSum::default() + x
    }
}

impl<F> Default for OnlineExactSum<F>
    where F: Float + Ieee754Ext
{
    fn default() -> Self {
        // Steps 1, 2, 3
        OnlineExactSum {
            i: 0,
            a1: vec![F::zero(); 2.pow(F::exponent_length())],
            a2: vec![F::zero(); 2.pow(F::exponent_length())]
        }
    }
}

/// Calculates the dot product using product transformation and `OnlineExactSum`
///
/// ![](https://rockshrub.de/accurate/OnlineExactDot.svg)
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let d = OnlineExactDot::default() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
pub struct OnlineExactDot<F> {
    s: OnlineExactSum<F>
}

impl<F> DotAccumulator<F> for OnlineExactDot<F>
    where F: Float + Ieee754Ext,
          F::Significand: PrimInt,
          F::RawExponent: PrimInt
{
    fn dot(&self) -> F {
        self.s.sum()
    }
}

impl<F> Add<(F, F)> for OnlineExactDot<F>
    where F:Float + Ieee754Ext,
          F::RawExponent: PrimInt
{
    type Output = Self;

    fn add(mut self, (a, b): (F, F)) -> Self::Output {
        let (h, r1) = two_product_fma(a, b);
        self.s = (self.s + h) + r1;
        self
    }
}

impl<F> From<F> for OnlineExactDot<F>
    where F: Float + Ieee754Ext,
          F::Significand: PrimInt,
          F::RawExponent: PrimInt
{
    fn from(x: F) -> Self {
        OnlineExactDot { s: OnlineExactSum::from(x) }
    }
}

impl<F> Default for OnlineExactDot<F>
    where F: Float + Ieee754Ext,
          F::Significand: PrimInt,
          F::RawExponent: PrimInt
{
    fn default() -> Self {
        OnlineExactDot::from(F::zero())
    }
}

/// Sums the items of an iterator
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let s = vec![1.0, 2.0, 3.0].sum_with_accumulator::<Sum2<_>>();
/// assert_eq!(6.0f64, s);
/// ```
pub trait SumWithAccumulator<F> {
    /// Sums the items of an iterator
    fn sum_with_accumulator<Acc>(self) -> F
        where Acc: SumAccumulator<F>;
}

impl<I, F> SumWithAccumulator<F> for I
    where I: IntoIterator<Item = F>
{
    fn sum_with_accumulator<Acc>(self) -> F
        where Acc: SumAccumulator<F>
    {
        Acc::default().absorb(self).sum()
    }
}

/// Calculates the dot product of the items of an iterator
///
/// # Examples
///
/// ```
/// use accurate::*;
///
/// let xy = vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
/// let d = xy.dot_with_accumulator::<Dot2<_>>();
/// assert_eq!(14.0f64, d);
/// ```
pub trait DotWithAccumulator<F> {
    /// Calculates the dot product of the items of an iterator
    fn dot_with_accumulator<Acc>(self) -> F
        where Acc: DotAccumulator<F>;
}

impl<I, F> DotWithAccumulator<F> for I
    where I: IntoIterator<Item = (F, F)>
{
    fn dot_with_accumulator<Acc>(self) -> F
        where Acc: DotAccumulator<F>
    {
        Acc::default().absorb(self).dot()
    }
}

#[cfg(feature = "unstable")]
use std::ops::AddAssign;

#[cfg(feature = "unstable")]
impl<F> AddAssign<F> for Naive<F>
    where F: Float
{
    fn add_assign(&mut self, rhs: F) {
        self.0 = self.0 + rhs;
    }
}

#[cfg(feature = "unstable")]
impl<F, C> AddAssign<F> for SumK<F, C>
    where F: Float,
          C: SumAccumulator<F> + AddAssign<F>
{
    fn add_assign(&mut self, rhs: F) {
        let (x, y) = two_sum(self.s, rhs);
        self.s = x;
        self.c += y;
    }
}

#[cfg(feature = "unstable")]
impl<F> AddAssign<F> for OnlineExactSum<F>
    where F: Float + Ieee754Ext,
          F::RawExponent: PrimInt
{
    fn add_assign(&mut self, rhs: F) {
        // Step 4(2)
        let j = rhs.decompose_raw().1.to_usize().unwrap();

        // Step 4(3)
        let (a, e) = two_sum(self.a1[j], rhs);
        self.a1[j] = a;

        // Step 4(4)
        self.a2[j] = self.a2[j] + e;

        // Step 4(5)
        self.i += 1;

        // Step 4(6)
        if self.i >= 2.pow(F::mantissa_length() / 2) {
            // Step 4(6)(a)
            let mut b1 = vec![F::zero(); 2.pow(F::exponent_length())];
            let mut b2 = vec![F::zero(); 2.pow(F::exponent_length())];

            // Step 4(6)(b)
            for &y in self.a1.iter().chain(self.a2.iter()) {
                // Step 4(6)(b)(i)
                let j = y.decompose_raw().1.to_usize().unwrap();

                // Step 4(6)(b)(ii)
                let (b, e) = two_sum(b1[j], y);
                b1[j] = b;

                // Step 4(6)(b)(iii)
                b2[j] = b2[j] + e;
            }

            // Step 4(6)(c)
            self.a1 = b1;
            self.a2 = b2;

            // Step 4(6)(d)
            self.i = 2 * 2.pow(F::exponent_length());
        }
    }
}
