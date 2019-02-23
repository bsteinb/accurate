//! Naive floating point summation

use std::ops::{Add, AddAssign};

use num_traits::Float;

use super::traits::SumAccumulator;

/// Naive floating point summation
///
/// ![](https://rockshrub.de/accurate/NaiveSum.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::sum::NaiveSum;
///
/// let s = NaiveSum::zero() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
#[derive(Copy, Clone, Debug)]
pub struct NaiveSum<F>(F);

impl<F> SumAccumulator<F> for NaiveSum<F>
where
    F: Float,
{
    #[inline]
    fn sum(self) -> F {
        self.0
    }
}

impl<F> Add<F> for NaiveSum<F>
where
    NaiveSum<F>: AddAssign<F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: F) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F> From<F> for NaiveSum<F>
where
    F: Float,
{
    fn from(x: F) -> Self {
        NaiveSum(x)
    }
}

impl<F> Add for NaiveSum<F>
where
    F: Float,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        NaiveSum(self.0 + rhs.0)
    }
}

unsafe impl<F> Send for NaiveSum<F> where F: Send {}

impl<F> AddAssign<F> for NaiveSum<F>
where
    F: Float,
{
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        self.0 = self.0 + rhs;
    }
}
