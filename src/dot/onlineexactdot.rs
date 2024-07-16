//! `OnlineExactSum` for dot product

use std::ops::Add;

#[cfg(doc)]
use embed_doc_image::embed_doc_image;

use super::traits::DotAccumulator;
use crate::sum::traits::SumAccumulator;
use crate::sum::OnlineExactSum;
use crate::util::traits::TwoProduct;
use crate::util::two_product;

/// Calculates the dot product using product transformation and `OnlineExactSum`
///
/// ![][OnlineExactDot]
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::dot::OnlineExactDot;
///
/// let d = OnlineExactDot::zero() + (1.0, 1.0) + (2.0, 2.0) + (3.0, 3.0);
/// assert_eq!(14.0f64, d.dot());
/// ```
#[cfg_attr(doc, embed_doc_image("OnlineExactDot", "images/OnlineExactDot.svg"))]
#[derive(Clone, Debug)]
pub struct OnlineExactDot<F> {
    s: OnlineExactSum<F>,
}

impl<F> DotAccumulator<F> for OnlineExactDot<F>
where
    F: TwoProduct,
    OnlineExactSum<F>: SumAccumulator<F>,
{
    fn zero() -> Self {
        OnlineExactDot::from(F::zero())
    }

    #[inline]
    fn dot(self) -> F {
        self.s.sum()
    }
}

impl<F> Add<(F, F)> for OnlineExactDot<F>
where
    F: TwoProduct,
    OnlineExactSum<F>: SumAccumulator<F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, (a, b): (F, F)) -> Self::Output {
        let (h, r1) = two_product(a, b);
        self.s = (self.s + h) + r1;
        self
    }
}

impl<F> From<F> for OnlineExactDot<F>
where
    OnlineExactSum<F>: SumAccumulator<F>,
{
    fn from(x: F) -> Self {
        OnlineExactDot {
            s: OnlineExactSum::from(x),
        }
    }
}

impl<F> Add for OnlineExactDot<F>
where
    OnlineExactSum<F>: Add<Output = OnlineExactSum<F>>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        OnlineExactDot { s: self.s + rhs.s }
    }
}

unsafe impl<F> Send for OnlineExactDot<F> where F: Send {}
