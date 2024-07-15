//! The `SumK` algorithm

use sum::cascaded::Cascaded;
use sum::NaiveSum;
use util::Knuth;

/// Calculates a sum using cascaded accumulators for the remainder terms
///
/// See also `Sum2`... `Sum9`.
///
/// ![](https://rockshrub.de/accurate/SumK.svg)
///
/// # References
///
/// Based on [Ogita, Rump and Oishi 05](http://dx.doi.org/10.1137/030601818)
pub type SumK<F, C> = Cascaded<F, C, Knuth>;

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
pub type Sum2<F> = SumK<F, NaiveSum<F>>;

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
