//! The `iFastSum` algorithm

use num_traits::Float;

use util::traits::{HalfUlp, Round3, TwoSum};
use util::{round3, two_sum};

/// Calculates the correctly rounded sum of numbers in a slice
pub trait IFastSum: TwoSum + HalfUlp + Round3 {}

impl<F> IFastSum for F where F: TwoSum + HalfUlp + Round3 {}

/// Calculates the correctly rounded sum of numbers in a slice
///
/// This algorithm works in place by mutating the contents of the slice. It is used by
/// `OnlineExactSum`.
///
/// # References
///
/// Based on [Zhu and Hayes 09](http://dx.doi.org/10.1137/070710020)
pub fn i_fast_sum_in_place<F>(xs: &mut [F]) -> F
where
    F: IFastSum,
{
    let mut n = xs.len();
    i_fast_sum_in_place_aux(xs, &mut n, true)
}

fn i_fast_sum_in_place_aux<F>(xs: &mut [F], n: &mut usize, recurse: bool) -> F
where
    F: IFastSum,
{
    // Step 1
    let mut s = F::zero();

    // Step 2
    // The following accesses are guaranteed to be inside bounds, because:
    debug_assert!(*n <= xs.len());
    for i in 0..*n {
        let x = unsafe { xs.get_unchecked_mut(i) };
        let (a, b) = two_sum(s, *x);
        s = a;
        *x = b;
    }

    // Step 3
    loop {
        // Step 3(1)
        let mut count: usize = 0; // slices are indexed from 0
        let mut st = F::zero();
        let mut sm = F::zero();

        // Step 3(2)
        // The following accesses are guaranteed to be inside bounds, because:
        debug_assert!(*n <= xs.len());
        for i in 0..*n {
            // Step 3(2)(a)
            let (a, b) = two_sum(st, unsafe { *xs.get_unchecked(i) });
            st = a;
            // Step 3(2)(b)
            if b != F::zero() {
                // The following access is guaranteed to be inside bounds, because:
                debug_assert!(count < xs.len());
                unsafe {
                    *xs.get_unchecked_mut(count) = b;
                }

                // Step 3(2)(b)(i)
                // The following addition is guaranteed not to overflow, because:
                debug_assert!(count < usize::max_value());
                // and thus:
                debug_assert!(count.checked_add(1).is_some());
                count += 1;

                // Step 3(2)(b)(ii)
                sm = sm.max(Float::abs(st));
            }
        }

        // Step 3(3)
        let em = F::from(count).expect("count not representable as floating point number")
            * sm.half_ulp();

        // Step 3(4)
        let (a, b) = two_sum(s, st);
        s = a;
        st = b;
        // The following access is guaranteed to be inside bounds, because:
        debug_assert!(count < xs.len());
        unsafe {
            *xs.get_unchecked_mut(count) = st;
        }
        // The following addition is guaranteed not to overflow, because:
        debug_assert!(count < usize::max_value());
        // and thus:
        debug_assert!(count.checked_add(1).is_some());
        *n = count + 1;

        // Step 3(5)
        if (em == F::zero()) || (em < s.half_ulp()) {
            // Step 3(5)(a)
            if !recurse {
                return s;
            }

            // Step 3(5)(b)
            let (w1, e1) = two_sum(st, em);
            // Step 3(5)(c)
            let (w2, e2) = two_sum(st, -em);

            // Step 3(5)(d)
            if (w1 + s != s)
                || (w2 + s != s)
                || (round3(s, w1, e1) != s)
                || (round3(s, w2, e2) != s)
            {
                // Step 3(5)(d)(i)
                let mut s1 = i_fast_sum_in_place_aux(xs, n, false);

                // Step 3(5)(d)(ii)
                let (a, b) = two_sum(s, s1);
                s = a;
                s1 = b;

                // Step 3(5)(d)(iii)
                let s2 = i_fast_sum_in_place_aux(xs, n, false);

                // Step 3(5)(d)(iv)
                s = round3(s, s1, s2);
            }

            // Step 3(5)(e)
            return s;
        }
    }
}
