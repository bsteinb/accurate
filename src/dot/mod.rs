//! Algorithms for dot product

pub mod traits;

mod dotk;
mod naive;
mod onlineexactdot;

pub use self::dotk::{Dot2, Dot3, Dot4, Dot5, Dot6, Dot7, Dot8, Dot9, DotK};
pub use self::naive::NaiveDot;
pub use self::onlineexactdot::OnlineExactDot;

#[cfg(feature = "parallel")]
use num_traits::Zero;

#[cfg(feature = "parallel")]
use rayon::iter::plumbing::{Consumer, Folder, UnindexedConsumer};

#[cfg(feature = "parallel")]
use self::traits::DotAccumulator;
#[cfg(feature = "parallel")]
use self::traits::ParallelDotAccumulator;
#[cfg(feature = "parallel")]
use util::AddReducer;

/// Adapts a `DotAccumulator` into a `Folder`
#[cfg(feature = "parallel")]
#[derive(Copy, Clone, Debug)]
pub struct DotFolder<Acc>(Acc);

#[cfg(feature = "parallel")]
impl<Acc, F> Folder<(F, F)> for DotFolder<Acc>
where
    Acc: DotAccumulator<F>,
{
    type Result = Acc;

    #[inline]
    fn consume(self, item: (F, F)) -> Self {
        DotFolder(self.0 + item)
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

/// Adapts a `ParallelDotAccumulator` into a `Consumer`
#[cfg(feature = "parallel")]
#[derive(Copy, Clone, Debug)]
pub struct DotConsumer<Acc>(Acc);

#[cfg(feature = "parallel")]
impl<Acc, F> Consumer<(F, F)> for DotConsumer<Acc>
where
    Acc: ParallelDotAccumulator<F>,
    F: Zero + Send,
{
    type Folder = DotFolder<Acc>;
    type Reducer = AddReducer;
    type Result = Acc;

    #[inline]
    fn split_at(self, _index: usize) -> (Self, Self, Self::Reducer) {
        (self, Acc::zero().into_consumer(), AddReducer)
    }

    #[inline]
    fn into_folder(self) -> Self::Folder {
        DotFolder(self.0)
    }

    #[inline]
    fn full(&self) -> bool {
        false
    }
}

#[cfg(feature = "parallel")]
impl<Acc, F> UnindexedConsumer<(F, F)> for DotConsumer<Acc>
where
    Acc: ParallelDotAccumulator<F>,
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
