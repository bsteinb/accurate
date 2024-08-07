//! Cascaded accumulators

use std::marker::PhantomData;
use std::ops::{Add, AddAssign};

#[cfg(doc)]
use embed_doc_image::embed_doc_image;

use num_traits::Float;

use crate::sum::traits::SumAccumulator;
use crate::sum::NaiveSum;
use crate::util::traits::TwoSum;
use crate::util::Neumaier as NeumaierTwoSum;

#[derive(Copy, Clone, Debug)]
pub struct Cascaded<F, C, T> {
    s: F,
    c: C,
    t: PhantomData<T>,
}

impl<F, C, T> SumAccumulator<F> for Cascaded<F, C, T>
where
    F: Float,
    C: SumAccumulator<F>,
    T: TwoSum<F>,
{
    #[inline]
    fn sum(self) -> F {
        (self.c + self.s).sum()
    }
}

impl<F, C, T> Add<F> for Cascaded<F, C, T>
where
    Cascaded<F, C, T>: AddAssign<F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: F) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F, C, T> From<F> for Cascaded<F, C, T>
where
    F: Float,
    C: SumAccumulator<F>,
{
    fn from(x: F) -> Self {
        Cascaded {
            s: x,
            c: C::zero(),
            t: PhantomData,
        }
    }
}

impl<F, C, T> Add for Cascaded<F, C, T>
where
    F: Float,
    C: SumAccumulator<F>,
    C::Output: Add<C, Output = C>,
    T: TwoSum<F>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let (s, c) = T::two_sum(self.s, rhs.s);
        Cascaded {
            s,
            c: (self.c + c) + rhs.c,
            t: PhantomData,
        }
    }
}

unsafe impl<F, C, T> Send for Cascaded<F, C, T>
where
    F: Send,
    C: Send,
{
}

impl<F, C, T> AddAssign<F> for Cascaded<F, C, T>
where
    F: Float,
    C: SumAccumulator<F>,
    T: TwoSum<F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        let (x, y) = T::two_sum(self.s, rhs);
        self.s = x;
        self.c += y;
    }
}

/// Neumaier summation
///
/// ![][Kahan]
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::sum::Neumaier;
///
/// let s = Neumaier::zero() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Neumaier 74](https://doi.org/10.1002%2Fzamm.19740540106)
#[cfg_attr(doc, embed_doc_image("Kahan", "images/Kahan.svg"))]
pub type Neumaier<F> = Cascaded<F, NaiveSum<F>, NeumaierTwoSum>;

/// Klein summation
///
/// ![][Kahan]
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::sum::Klein;
///
/// let s = Klein::zero() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Klein 06](https://doi.org/10.1007%2Fs00607-005-0139-x)
#[cfg_attr(doc, embed_doc_image("Kahan", "images/Kahan.svg"))]
pub type Klein<F> = Cascaded<F, Cascaded<F, NaiveSum<F>, NeumaierTwoSum>, NeumaierTwoSum>;
