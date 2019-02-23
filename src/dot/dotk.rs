//! The `DotK` algorithm

use std::ops::Add;

use num_traits::Float;

use super::traits::DotAccumulator;
use sum::traits::SumAccumulator;
use sum::{Sum2, Sum3, Sum4, Sum5, Sum6, Sum7, Sum8};
use util::traits::{TwoProduct, TwoSum};
use util::{two_product, two_sum};

/// `DotK` with two cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::dot::Dot2;
///
/// let d = Dot2::zero() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
#[derive(Copy, Clone, Debug)]
pub struct Dot2<F> {
    p: F,
    r: F,
}

impl<F> DotAccumulator<F> for Dot2<F>
where
    F: Float + TwoProduct + TwoSum,
{
    #[inline]
    fn dot(self) -> F {
        self.r + self.p
    }
}

impl<F> Add<(F, F)> for Dot2<F>
where
    F: Float + TwoProduct + TwoSum,
{
    type Output = Self;

    #[inline]
    fn add(self, (a, b): (F, F)) -> Self {
        let (h, r1) = two_product(a, b);
        let (p, r2) = two_sum(self.p, h);
        Dot2 {
            p,
            r: (self.r + r1) + r2,
        }
    }
}

impl<F> From<F> for Dot2<F>
where
    F: Float,
{
    fn from(x: F) -> Self {
        Dot2 { p: x, r: F::zero() }
    }
}

impl<F> Add for Dot2<F>
where
    F: Float + TwoSum,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let (p, r) = two_sum(self.p, rhs.p);
        Dot2 {
            p,
            r: (self.r + r) + rhs.r,
        }
    }
}

unsafe impl<F> Send for Dot2<F> where F: Send {}

/// Calculates a dot product using both product transformation and cascaded accumulators
///
/// See also `Dot2`... `Dot9`.
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
#[derive(Copy, Clone, Debug)]
pub struct DotK<F, R> {
    p: F,
    r: R,
}

impl<F, R> DotAccumulator<F> for DotK<F, R>
where
    F: Float + TwoProduct + TwoSum,
    R: SumAccumulator<F>,
{
    #[inline]
    fn dot(self) -> F {
        (self.r + self.p).sum()
    }
}

impl<F, R> Add<(F, F)> for DotK<F, R>
where
    F: TwoProduct + TwoSum,
    R: SumAccumulator<F>,
{
    type Output = Self;

    #[inline]
    fn add(self, (a, b): (F, F)) -> Self {
        let (h, r1) = two_product(a, b);
        let (p, r2) = two_sum(self.p, h);
        DotK {
            p,
            r: (self.r + r1) + r2,
        }
    }
}

impl<F, R> From<F> for DotK<F, R>
where
    F: Float,
    R: SumAccumulator<F>,
{
    fn from(x: F) -> Self {
        DotK { p: x, r: R::zero() }
    }
}

impl<F, R> Add for DotK<F, R>
where
    F: Float + TwoSum,
    R: SumAccumulator<F>,
    R::Output: Add<R, Output = R>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let (p, r) = two_sum(self.p, rhs.p);
        DotK {
            p,
            r: (self.r + r) + rhs.r,
        }
    }
}

unsafe impl<F, R> Send for DotK<F, R>
where
    F: Send,
    R: Send,
{
}

/// `DotK` with three cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::dot::Dot3;
///
/// let d = Dot3::zero() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Dot3<F> = DotK<F, Sum2<F>>;

/// `DotK` with four cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::dot::Dot4;
///
/// let d = Dot4::zero() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Dot4<F> = DotK<F, Sum3<F>>;

/// `DotK` with five cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::dot::Dot5;
///
/// let d = Dot5::zero() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Dot5<F> = DotK<F, Sum4<F>>;

/// `DotK` with six cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::dot::Dot6;
///
/// let d = Dot6::zero() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Dot6<F> = DotK<F, Sum5<F>>;

/// `DotK` with seven cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::dot::Dot7;
///
/// let d = Dot7::zero() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Dot7<F> = DotK<F, Sum6<F>>;

/// `DotK` with eight cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::dot::Dot8;
///
/// let d = Dot8::zero() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Dot8<F> = DotK<F, Sum7<F>>;

/// `DotK` with nine cascaded accumulators
///
/// ![](https://rockshrub.de/accurate/DotK.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::dot::Dot9;
///
/// let d = Dot9::zero() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type Dot9<F> = DotK<F, Sum8<F>>;
