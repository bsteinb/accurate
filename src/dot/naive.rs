//! Naive floating point dot product

use std::ops::Add;

use num_traits::Float;

use super::traits::DotAccumulator;

/// Naive floating point dot product
///
/// ![](https://rockshrub.de/accurate/NaiveDot.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::dot::NaiveDot;
///
/// let d = NaiveDot::zero() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
#[derive(Copy, Clone, Debug)]
pub struct NaiveDot<F>(F);

impl<F> DotAccumulator<F> for NaiveDot<F>
where
    F: Float,
{
    #[inline]
    fn dot(self) -> F {
        self.0
    }
}

impl<F> Add<(F, F)> for NaiveDot<F>
where
    F: Float,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: (F, F)) -> Self::Output {
        NaiveDot(self.0 + rhs.0 * rhs.1)
    }
}

impl<F> From<F> for NaiveDot<F>
where
    F: Float,
{
    fn from(x: F) -> Self {
        NaiveDot(x)
    }
}

impl<F> Add for NaiveDot<F>
where
    F: Float,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        NaiveDot(self.0 + rhs.0)
    }
}

unsafe impl<F> Send for NaiveDot<F> where F: Send {}
