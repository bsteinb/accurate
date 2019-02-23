//! Summation traits

use std::ops::{Add, AddAssign};

use num_traits::Zero;

#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

pub use sum::ifastsum::IFastSum;

#[cfg(feature = "parallel")]
use sum::SumConsumer;

/// Accumulates terms of a sum
pub trait SumAccumulator<F>: Add<F, Output = Self> + AddAssign<F> + From<F> + Clone {
    /// Initial value for an accumulator
    fn zero() -> Self
    where
        F: Zero,
    {
        Self::from(F::zero())
    }

    /// The sum of all terms accumulated so far
    fn sum(self) -> F;

    /// Absorb the items of an iterator into the accumulator
    ///
    /// # Examples
    ///
    /// ```
    /// use accurate::traits::*;
    /// use accurate::sum::Sum2;
    ///
    /// let s = Sum2::zero().absorb(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(6.0f64, s.sum())
    /// ```
    fn absorb<I>(self, it: I) -> Self
    where
        I: IntoIterator<Item = F>,
    {
        it.into_iter().fold(self, |acc, x| acc + x)
    }
}

/// Sums the items of an iterator
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::sum::Sum2;
///
/// let s = vec![1.0, 2.0, 3.0].sum_with_accumulator::<Sum2<_>>();
/// assert_eq!(6.0f64, s);
/// ```
pub trait SumWithAccumulator<F> {
    /// Sums the items of an iterator
    fn sum_with_accumulator<Acc>(self) -> F
    where
        Acc: SumAccumulator<F>,
        F: Zero;
}

impl<I, F> SumWithAccumulator<F> for I
where
    I: IntoIterator<Item = F>,
{
    fn sum_with_accumulator<Acc>(self) -> F
    where
        Acc: SumAccumulator<F>,
        F: Zero,
    {
        Acc::zero().absorb(self).sum()
    }
}

/// A `SumAccumulator` that can be used in parallel computations
#[cfg(feature = "parallel")]
pub trait ParallelSumAccumulator<F>:
    SumAccumulator<F> + Add<Self, Output = Self> + Send + Sized
{
    /// Turns an accumulator into a consumer
    #[inline]
    fn into_consumer(self) -> SumConsumer<Self> {
        SumConsumer(self)
    }
}

#[cfg(feature = "parallel")]
impl<Acc, F> ParallelSumAccumulator<F> for Acc where
    Acc: SumAccumulator<F> + Add<Acc, Output = Acc> + Send + Sized
{
}

/// Sums the items of an iterator, possibly in parallel
///
/// # Examples
///
/// ```
/// # extern crate accurate;
/// # extern crate rayon;
///
/// use rayon::prelude::*;
///
/// use accurate::traits::*;
/// use accurate::sum::OnlineExactSum;
///
/// # fn main() {
/// let s = vec![1.0, 2.0, 3.0].par_iter().map(|&x| x)
///     .parallel_sum_with_accumulator::<OnlineExactSum<_>>();
/// assert_eq!(6.0f64, s);
/// # }
/// ```
#[cfg(feature = "parallel")]
pub trait ParallelSumWithAccumulator<F>: ParallelIterator<Item = F>
where
    F: Send,
{
    /// Sums the items of an iterator, possibly in parallel
    fn parallel_sum_with_accumulator<Acc>(self) -> F
    where
        Acc: ParallelSumAccumulator<F>,
        F: Zero,
    {
        self.drive_unindexed(Acc::zero().into_consumer()).sum()
    }
}

#[cfg(feature = "parallel")]
impl<T, F> ParallelSumWithAccumulator<F> for T
where
    T: ParallelIterator<Item = F>,
    F: Zero + Send,
{
}
