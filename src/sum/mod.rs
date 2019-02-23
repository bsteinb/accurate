//! Algorithms for summation

pub mod traits;

mod ifastsum;
mod naive;
mod onlineexactsum;
mod sumk;

pub use self::ifastsum::i_fast_sum_in_place;
pub use self::naive::NaiveSum;
pub use self::onlineexactsum::OnlineExactSum;
pub use self::sumk::{Sum2, Sum3, Sum4, Sum5, Sum6, Sum7, Sum8, Sum9, SumK};

#[cfg(feature = "parallel")]
use num_traits::Zero;

#[cfg(feature = "parallel")]
use rayon::iter::plumbing::{Consumer, Folder, UnindexedConsumer};

#[cfg(feature = "parallel")]
use self::traits::ParallelSumAccumulator;
#[cfg(feature = "parallel")]
use self::traits::SumAccumulator;
#[cfg(feature = "parallel")]
use util::AddReducer;

/// Adapts a `SumAccumulator` into a `Folder`
#[cfg(feature = "parallel")]
#[derive(Copy, Clone, Debug)]
pub struct SumFolder<Acc>(Acc);

#[cfg(feature = "parallel")]
impl<Acc, F> Folder<F> for SumFolder<Acc>
where
    Acc: SumAccumulator<F>,
{
    type Result = Acc;

    #[inline]
    fn consume(self, item: F) -> Self {
        SumFolder(self.0 + item)
    }

    #[inline]
    fn complete(self) -> Self::Result {
        self.0
    }

    #[inline]
    fn full(&self) -> bool {
        false
    }
}

/// Adapts a `ParallelSumAccumulator` into a `Consumer`
#[cfg(feature = "parallel")]
#[derive(Copy, Clone, Debug)]
pub struct SumConsumer<Acc>(Acc);

#[cfg(feature = "parallel")]
impl<Acc, F> Consumer<F> for SumConsumer<Acc>
where
    Acc: ParallelSumAccumulator<F>,
    F: Zero + Send,
{
    type Folder = SumFolder<Acc>;
    type Reducer = AddReducer;
    type Result = Acc;

    #[inline]
    fn split_at(self, _index: usize) -> (Self, Self, Self::Reducer) {
        (self, Acc::zero().into_consumer(), AddReducer)
    }

    #[inline]
    fn into_folder(self) -> Self::Folder {
        SumFolder(self.0)
    }

    #[inline]
    fn full(&self) -> bool {
        false
    }
}

#[cfg(feature = "parallel")]
impl<Acc, F> UnindexedConsumer<F> for SumConsumer<Acc>
where
    Acc: ParallelSumAccumulator<F>,
    F: Zero + Send,
{
    #[inline]
    fn split_off_left(&self) -> Self {
        Acc::zero().into_consumer()
    }

    #[inline]
    fn to_reducer(&self) -> Self::Reducer {
        AddReducer
    }
}
