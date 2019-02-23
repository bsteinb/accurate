//! The `SumK` algorithm

use std::ops::{Add, AddAssign};

use num_traits::Float;

use super::traits::SumAccumulator;
use util::traits::TwoSum;
use util::two_sum;

/// `SumK` with two cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::sum::Sum2;
///
/// let s = Sum2::zero() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
#[derive(Copy, Clone, Debug)]
pub struct Sum2<F> {
    s: F,
    c: F,
}

impl<F> SumAccumulator<F> for Sum2<F>
where
    F: Float + TwoSum + AddAssign,
{
    #[inline]
    fn sum(self) -> F {
        self.c + self.s
    }
}

impl<F> Add<F> for Sum2<F>
where
    Sum2<F>: AddAssign<F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: F) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F> From<F> for Sum2<F>
where
    F: Float,
{
    fn from(x: F) -> Self {
        Sum2 { s: x, c: F::zero() }
    }
}

impl<F> Add for Sum2<F>
where
    F: Float + TwoSum,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let (s, c) = two_sum(self.s, rhs.s);
        Sum2 {
            s,
            c: (self.c + c) + rhs.c,
        }
    }
}

unsafe impl<F> Send for Sum2<F> where F: Send {}

impl<F> AddAssign<F> for Sum2<F>
where
    F: Float + TwoSum + AddAssign,
{
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        let (x, y) = two_sum(self.s, rhs);
        self.s = x;
        self.c += y;
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
#[derive(Copy, Clone, Debug)]
pub struct SumK<F, C> {
    s: F,
    c: C,
}

impl<F, C> SumAccumulator<F> for SumK<F, C>
where
    F: Float + TwoSum,
    C: SumAccumulator<F>,
{
    #[inline]
    fn sum(self) -> F {
        (self.c + self.s).sum()
    }
}

impl<F, C> Add<F> for SumK<F, C>
where
    SumK<F, C>: AddAssign<F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: F) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F, C> From<F> for SumK<F, C>
where
    F: Float,
    C: SumAccumulator<F>,
{
    fn from(x: F) -> Self {
        SumK { s: x, c: C::zero() }
    }
}

impl<F, C> Add for SumK<F, C>
where
    F: Float + TwoSum,
    C: SumAccumulator<F>,
    C::Output: Add<C, Output = C>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let (s, c) = two_sum(self.s, rhs.s);
        SumK {
            s,
            c: (self.c + c) + rhs.c,
        }
    }
}

unsafe impl<F, C> Send for SumK<F, C>
where
    F: Send,
    C: Send,
{
}

impl<F, C> AddAssign<F> for SumK<F, C>
where
    F: Float + TwoSum,
    C: SumAccumulator<F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        let (x, y) = two_sum(self.s, rhs);
        self.s = x;
        self.c += y;
    }
}

/// `SumK` with three cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::sum::Sum3;
///
/// let s = Sum3::zero() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Sum3<F> = SumK<F, Sum2<F>>;

/// `SumK` with four cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::sum::Sum4;
///
/// let s = Sum4::zero() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Sum4<F> = SumK<F, Sum3<F>>;

/// `SumK` with five cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::sum::Sum5;
///
/// let s = Sum5::zero() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Sum5<F> = SumK<F, Sum4<F>>;

/// `SumK` with six cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::sum::Sum6;
///
/// let s = Sum6::zero() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Sum6<F> = SumK<F, Sum5<F>>;

/// `SumK` with seven cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::sum::Sum7;
///
/// let s = Sum7::zero() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Sum7<F> = SumK<F, Sum6<F>>;

/// `SumK` with eight cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::sum::Sum8;
///
/// let s = Sum8::zero() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Sum8<F> = SumK<F, Sum7<F>>;

/// `SumK` with nine cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::sum::Sum9;
///
/// let s = Sum9::zero() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Sum9<F> = SumK<F, Sum8<F>>;
