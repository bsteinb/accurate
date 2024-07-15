//! Common infrastructure

use cfg_if::cfg_if;

use num_traits::Float;

#[cfg(feature = "parallel")]
use std::ops::Add;

#[cfg(feature = "parallel")]
use rayon::iter::plumbing::Reducer;

pub mod traits;

use self::traits::{Round3, TwoProduct, TwoSum};

/// Dekker's two term sum transformation
#[derive(Copy, Clone, Debug)]
pub struct Dekker;

impl<F> TwoSum<F> for Dekker
where
    F: Float,
{
    #[inline]
    fn two_sum(a: F, b: F) -> (F, F) {
        let x = a + b;
        let y = (a - x) + b;
        (x, y)
    }
}

/// Neumaier's error-free two term sum transformation
#[derive(Copy, Clone, Debug)]
pub struct Neumaier;

impl<F> TwoSum<F> for Neumaier
where
    F: Float,
{
    #[inline]
    fn two_sum(a: F, b: F) -> (F, F) {
        if a.abs() >= b.abs() {
            Dekker::two_sum(a, b)
        } else {
            Dekker::two_sum(b, a)
        }
    }
}

/// Knuth's branch-less Error-free transformations of two term sums
///
/// # References
///
/// From Knuth's AoCP, Volume 2: Seminumerical Algorithms
#[derive(Copy, Clone, Debug)]
pub struct Knuth;

impl<F> TwoSum<F> for Knuth
where
    F: Float,
{
    #[inline]
    fn two_sum(a: F, b: F) -> (F, F) {
        let x = a + b;
        let z = x - a;
        let y = (a - (x - z)) + (b - z);
        (x, y)
    }
}

/// Knuth's branch-less Error-free transformations of two term sums
pub fn two_sum<F>(a: F, b: F) -> (F, F)
where
    F: Float,
{
    Knuth::two_sum(a, b)
}

cfg_if! {
    if #[cfg(feature = "fma")] {
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
        #[inline]
        pub fn two_product<F>(a: F, b: F) -> (F, F)
            where F: TwoProduct
        {
            let x = a * b;
            let y = a.mul_add(b, -x);
            (x, y)
        }
    } else {
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
        /// Based on [Dekker 71](http://dx.doi.org/10.1007/BF01397083)
        #[inline]
        pub fn two_product<F>(x: F, y: F) -> (F, F)
            where F: TwoProduct
        {
            let a = x * y;
            let (x1, x2) = x.split();
            let (y1, y2) = y.split();
            let b = x2 * y2 - (((a - x1 * y1) - x2 * y1) - x1 * y2);
            (a, b)
        }
    }
}

/// Correctly rounded sum of three non-overlapping numbers
///
/// Calculates the correctly rounded sum of three numbers `s0`, `s1` and `s2` which are
/// non-overlapping, i.e.:
///
/// ```not_rust
/// s0.abs() > s1.abs() > s2.abs()
/// fl(s0 + s1) = s0
/// fl(s1 + s2) = s1
/// ```
///
/// # References
///
/// Based on [Zhu and Hayes 09](http://dx.doi.org/10.1137/070710020)
#[inline]
pub fn round3<F>(s0: F, s1: F, s2: F) -> F
where
    F: Round3,
{
    debug_assert!(s0 == s0 + s1);
    debug_assert!(s1 == s1 + s2);
    if s1.has_half_ulp_form() && s1.sign() == s2.sign() {
        s0 + if s1.is_sign_positive() {
            s1.next()
        } else {
            s1.prev()
        }
    } else {
        s0
    }
}

/// Reduce two parallel results using `Add`
#[cfg(feature = "parallel")]
#[derive(Copy, Clone, Debug)]
pub struct AddReducer;

#[cfg(feature = "parallel")]
impl<Acc> Reducer<Acc> for AddReducer
where
    Acc: Add<Acc, Output = Acc>,
{
    #[inline]
    fn reduce(self, left: Acc, right: Acc) -> Acc {
        left + right
    }
}
