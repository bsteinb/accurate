//! Kahan summation

use std::ops::{Add, AddAssign};

use num_traits::Float;

use crate::sum::traits::SumAccumulator;

/// Kahan summation
///
/// ![](https://rockshrub.de/accurate/Kahan.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::sum::Kahan;
///
/// let s = Kahan::zero() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Kahan 65](https://doi.org/10.1145%2F363707.363723)
#[derive(Copy, Clone, Debug)]
pub struct Kahan<F> {
    sum: F,
    c: F,
}

impl<F> SumAccumulator<F> for Kahan<F>
where
    F: Float,
{
    #[inline]
    fn sum(self) -> F {
        self.sum
    }
}

impl<F> Add<F> for Kahan<F>
where
    Kahan<F>: AddAssign<F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: F) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F> From<F> for Kahan<F>
where
    F: Float,
{
    fn from(x: F) -> Self {
        Kahan {
            sum: x,
            c: F::zero(),
        }
    }
}

impl<F> Add for Kahan<F>
where
    F: Float,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs.sum;
        self += rhs.c;
        self
    }
}

unsafe impl<F> Send for Kahan<F> where F: Send {}

impl<F> AddAssign<F> for Kahan<F>
where
    F: Float,
{
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        let t = self.sum;
        let y = rhs + self.c;
        self.sum = t + y;
        self.c = (t - self.sum) + y;
    }
}
