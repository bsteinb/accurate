//! Dot product traits

use std::ops::Add;

use num_traits::Zero;

#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

#[cfg(feature = "parallel")]
use super::DotConsumer;

/// Accumulates terms of a dot product
pub trait DotAccumulator<F>: Add<(F, F), Output = Self> + From<F> + Clone {
    /// Initial value for an accumulator
    fn zero() -> Self
    where
        F: Zero,
    {
        Self::from(F::zero())
    }

    /// The dot product of all terms accumulated so far
    fn dot(self) -> F;

    /// Absorb the items of an iterator into the accumulator
    ///
    /// # Examples
    ///
    /// ```
    /// use accurate::traits::*;
    /// use accurate::dot::Dot2;
    ///
    /// let x = vec![1.0, 2.0, 3.0];
    /// let y = x.clone();
    ///
    /// let d = Dot2::zero().absorb(x.into_iter().zip(y.into_iter()));
    /// assert_eq!(14.0f64, d.dot())
    /// ```
    fn absorb<I>(self, it: I) -> Self
    where
        I: IntoIterator<Item = (F, F)>,
    {
        it.into_iter().fold(self, |acc, x| acc + x)
    }
}

/// Calculates the dot product of the items of an iterator
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::dot::Dot2;
///
/// let xy = vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
/// let d = xy.dot_with_accumulator::<Dot2<_>>();
/// assert_eq!(14.0f64, d);
/// ```
pub trait DotWithAccumulator<F> {
    /// Calculates the dot product of the items of an iterator
    fn dot_with_accumulator<Acc>(self) -> F
    where
        Acc: DotAccumulator<F>,
        F: Zero;
}

impl<I, F> DotWithAccumulator<F> for I
where
    I: IntoIterator<Item = (F, F)>,
{
    fn dot_with_accumulator<Acc>(self) -> F
    where
        Acc: DotAccumulator<F>,
        F: Zero,
    {
        Acc::zero().absorb(self).dot()
    }
}

/// A `DotAccumulator` that can be used in parallel computations
#[cfg(feature = "parallel")]
pub trait ParallelDotAccumulator<F>:
    DotAccumulator<F> + Add<Self, Output = Self> + Send + Sized
{
    /// Turns an accumulator into a consumer
    #[inline]
    fn into_consumer(self) -> DotConsumer<Self> {
        DotConsumer(self)
    }
}

#[cfg(feature = "parallel")]
impl<Acc, F> ParallelDotAccumulator<F> for Acc where
    Acc: DotAccumulator<F> + Add<Acc, Output = Acc> + Send + Sized
{
}

/// Calculates the dot product of an iterator, possibly in parallel
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
/// use accurate::dot::OnlineExactDot;
///
/// # fn main() {
/// let d = vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
///     .par_iter().map(|&x| x)
///     .parallel_dot_with_accumulator::<OnlineExactDot<_>>();
/// assert_eq!(14.0f64, d);
/// # }
/// ```
#[cfg(feature = "parallel")]
pub trait ParallelDotWithAccumulator<F>: ParallelIterator<Item = (F, F)>
where
    F: Send,
{
    /// Calculate the dot product of an iterator, possibly in parallel
    fn parallel_dot_with_accumulator<Acc>(self) -> F
    where
        Acc: ParallelDotAccumulator<F>,
        F: Zero,
    {
        self.drive_unindexed(Acc::zero().into_consumer()).dot()
    }
}

#[cfg(feature = "parallel")]
impl<T, F> ParallelDotWithAccumulator<F> for T
where
    T: ParallelIterator<Item = (F, F)>,
    F: Zero + Send,
{
}
