//! The `OnlineExactSum` algorithm

use std::ops::{Add, AddAssign};

use num_traits::Float;

use super::i_fast_sum_in_place;
use super::traits::{IFastSum, SumAccumulator};
use util::traits::{FloatFormat, RawExponent, TwoSum};
use util::two_sum;

/// Calculates a sum using separate accumulators for each possible exponent
///
/// ![](https://rockshrub.de/accurate/OnlineExactSum.svg)
///
/// # Examples
///
/// ```
/// use accurate::traits::*;
/// use accurate::sum::OnlineExactSum;
///
/// let s = OnlineExactSum::zero() + 1.0 + 2.0 + 3.0;
/// assert_eq!(6.0f64, s.sum());
/// ```
///
/// # References
///
/// Based on [Zhu and Hayes 10](http://dx.doi.org/10.1145/1824801.1824815)
#[derive(Clone, Debug)]
pub struct OnlineExactSum<F> {
    i: usize,
    a1: Box<[F]>,
    a2: Box<[F]>,
}

impl<F> OnlineExactSum<F>
where
    F: TwoSum + FloatFormat + RawExponent,
{
    fn new() -> Self {
        // Steps 1, 2, 3
        OnlineExactSum {
            i: 0,
            a1: vec![F::zero(); F::base_pow_exponent_digits()].into_boxed_slice(),
            a2: vec![F::zero(); F::base_pow_exponent_digits()].into_boxed_slice(),
        }
    }

    #[inline(never)]
    fn compact(&mut self) {
        // Step 4(6)(a)
        let mut b1v = vec![F::zero(); F::base_pow_exponent_digits()].into_boxed_slice();
        let mut b2v = vec![F::zero(); F::base_pow_exponent_digits()].into_boxed_slice();

        // Step 4(6)(b)
        for &y in self.a1.iter().chain(self.a2.iter()) {
            // Step 4(6)(b)(i)
            let j = y.raw_exponent();
            // These accesses are guaranteed to be within bounds, because:
            debug_assert_eq!(b1v.len(), F::base_pow_exponent_digits());
            debug_assert_eq!(b2v.len(), F::base_pow_exponent_digits());
            debug_assert!(j < F::base_pow_exponent_digits());
            let b1 = unsafe { b1v.get_unchecked_mut(j) };
            let b2 = unsafe { b2v.get_unchecked_mut(j) };

            // Step 4(6)(b)(ii)
            let (b, e) = two_sum(*b1, y);
            *b1 = b;

            // Step 4(6)(b)(iii)
            *b2 = *b2 + e;
        }

        // Step 4(6)(c)
        self.a1 = b1v;
        self.a2 = b2v;

        // Step 4(6)(d)
        self.i = 2 * F::base_pow_exponent_digits();
    }
}

impl<F> SumAccumulator<F> for OnlineExactSum<F>
where
    F: Float + TwoSum + IFastSum + FloatFormat + RawExponent,
{
    fn zero() -> Self {
        Self::new()
    }

    #[inline]
    fn sum(self) -> F {
        // Step 5
        let mut a = self.a1.into_vec();
        let mut b = self.a2.into_vec();
        a.append(&mut b);
        a.retain(|&x| x != F::zero());

        // Step 6
        i_fast_sum_in_place(&mut a[..])
    }
}

impl<F> Add<F> for OnlineExactSum<F>
where
    OnlineExactSum<F>: AddAssign<F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: F) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F> From<F> for OnlineExactSum<F>
where
    F: TwoSum + FloatFormat + RawExponent,
{
    fn from(x: F) -> Self {
        Self::new() + x
    }
}

impl<F> Add for OnlineExactSum<F>
where
    F: Float + TwoSum + IFastSum + FloatFormat + RawExponent,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.absorb(rhs.a1.iter().cloned().chain(rhs.a2.iter().cloned()))
    }
}

unsafe impl<F> Send for OnlineExactSum<F> where F: Send {}

impl<F> AddAssign<F> for OnlineExactSum<F>
where
    F: TwoSum + FloatFormat + RawExponent,
{
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        // Step 4(2)
        {
            let j = rhs.raw_exponent();
            // These accesses are guaranteed to be within bounds, because:
            debug_assert_eq!(self.a1.len(), F::base_pow_exponent_digits());
            debug_assert_eq!(self.a2.len(), F::base_pow_exponent_digits());
            debug_assert!(j < F::base_pow_exponent_digits());
            let a1 = unsafe { self.a1.get_unchecked_mut(j) };
            let a2 = unsafe { self.a2.get_unchecked_mut(j) };

            // Step 4(3)
            let (a, e) = two_sum(*a1, rhs);
            *a1 = a;

            // Step 4(4)
            *a2 = *a2 + e;
        }

        // Step 4(5)
        // This addition is guaranteed not to overflow because the next step ascertains that (at
        // this point):
        debug_assert!(self.i < F::base_pow_significand_digits_half());
        // and (for `f32` and `f64`) we have:
        debug_assert!(F::base_pow_significand_digits_half() < usize::max_value());
        // thus we can assume:
        debug_assert!(self.i.checked_add(1).is_some());
        self.i += 1;

        // Step 4(6)
        if self.i >= F::base_pow_significand_digits_half() {
            self.compact();
        }
    }
}
