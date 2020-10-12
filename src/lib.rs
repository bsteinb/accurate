//! A collection of (more or less) accurate floating point algorithms
//!
//! This crate implements several algorithms for floating point summation and dot product. The
//! algorithms are realized as types that implement the `SumAccumulator` and `DotAccumulator`
//! trait.
//!
//! # Basic usage
//!
//! Calculating a sum (or a dot product) begins by initializing an accumulator to zero:
//!
//! ```
//! use accurate::traits::*; // Most functionality is derived from traits in this module
//! use accurate::sum::NaiveSum; // Chose a specific algorithm to perform summation / dot product
//!
//! let s = NaiveSum::<f32>::zero();
//! ```
//!
//! The accumulator traits are generic over the type of the underlying floating point numbers and
//! the `zero()` constructor is supported if the number type implements the Zero trait.
//! Alternatively the accumulator traits imply that an accumulator can be constructed `from()` an
//! arbitrary value of the number type.
//!
//! ```
//! # use accurate::traits::*;
//! # use accurate::sum::NaiveSum;
//! let s = NaiveSum::from(42.0f64);
//! ```
//!
//! The actual calculation is performed via the `Add<F, Output = Self>` trait that is also implied
//! by the `SumAccumulator` trait, where `F` is the type of the floating point numbers.
//!
//! ```
//! # use accurate::traits::*;
//! use accurate::sum::Sum2;
//!
//! let s = Sum2::zero() + 1.0f64 + 2.0 + 3.0;
//! ```
//!
//! For dot products, the `DotAccumulator` trait implies `Add<(F, F), Output = Self>` to allow
//! accumulation of the products of pairs into the final result.
//!
//! ```
//! # use accurate::traits::*;
//! use accurate::dot::NaiveDot;
//!
//! let d = NaiveDot::zero() + (1.0f64, 1.0f64) + (2.0, 2.0) + (3.0, 3.0);
//! ```
//!
//! Once all of the terms have been accumulated, the result can be evaluated using the `sum()` and
//! `dot()` methods respectively.
//!
//! ```
//! # use accurate::traits::*;
//! # use accurate::sum::Sum2;
//! # use accurate::dot::NaiveDot;
//! let s = Sum2::zero() + 1.0f64 + 2.0 + 3.0;
//! assert_eq!(6.0, s.sum());
//!
//! let d = NaiveDot::zero() + (1.0f64, 1.0f64) + (2.0, 2.0) + (3.0, 3.0);
//! assert_eq!(14.0, d.dot());
//! ```
//!
//! Both `sum()` and `dot()` take their argument by value, because the evaluation of the final
//! result is in some cases a destructive operation on the internal state of the accumulator.
//! However, the evaluation of partial results is supported by `clone()`ing the accumulator.
//!
//! ```
//! # use accurate::traits::*;
//! # use accurate::sum::Sum2;
//! let s = Sum2::zero() + 1.0f32 + 2.0;
//! assert_eq!(3.0, s.clone().sum());
//! let s = s + 3.0;
//! assert_eq!(6.0, s.sum());
//! ```
//!
//! # Iterator consumption
//!
//! Accumulators can be used in `fold()` operations on iterators as one would expect.
//!
//! ```
//! # use accurate::traits::*;
//! # use accurate::sum::Sum2;
//! use accurate::dot::Dot2;
//!
//! let s = vec![1.0f32, 2.0, 3.0].into_iter().fold(Sum2::zero(), |acc, x| acc + x);
//! assert_eq!(6.0, s.sum());
//!
//! let d = vec![1.0f32, 2.0, 3.0].into_iter()
//!     .zip(vec![1.0, 2.0, 3.0].into_iter())
//!     .fold(Dot2::zero(), |acc, xy| acc + xy);
//! assert_eq!(14.0, d.dot());
//! ```
//!
//! For convenience, the accumulator traits also define `absorb()` methods to absorb values from
//! anything that implements `IntoIterator`.
//!
//! ```
//! # use accurate::traits::*;
//! # use accurate::sum::Sum2;
//! # use accurate::dot::Dot2;
//!
//! let s = Sum2::zero().absorb(vec![1.0f32, 2.0, 3.0]);
//! assert_eq!(6.0, s.sum());
//!
//! let d = Dot2::zero().absorb(vec![(1.0f32, 1.0), (2.0, 2.0), (3.0, 3.0)]);
//! assert_eq!(14.0, d.dot());
//! ```
//!
//! And for even more convenience, suitable iterators are extended by a `sum_with_accumulator()`
//! (and `dot_with_accumulator()`) method that directly evaluates to the result in the floating
//! point number type.
//!
//! ```
//! # use accurate::traits::*;
//! # use accurate::sum::Sum2;
//! # use accurate::dot::Dot2;
//!
//! let s = Sum2::zero().absorb(vec![1.0f32, 2.0, 3.0]);
//! assert_eq!(6.0f64, vec![1.0, 2.0, 3.0].into_iter().sum_with_accumulator::<Sum2<_>>());
//!
//! assert_eq!(14.0f64, vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)].into_iter()
//!     .dot_with_accumulator::<Dot2<_>>());
//! ```
//!
#![cfg_attr(
    feature = "parallel",
    doc = "
# Parallel computation

If compiled with the `parallel` feature enabled (which is the default) the `rayon` parallel
iterator facilities are used to perform large calculations in parallel. Parallel calculations are
performed through the `parallel_sum_with_accumulator()` and `parallel_dot_with_accumulator()`
extension methods on parallel iterators.

```
# extern crate accurate;
extern crate rayon;

use rayon::prelude::*;

# use accurate::traits::*;
# use accurate::sum::Sum2;
# fn main() {
let xs = vec![1.0f64; 100_000];
let s = xs.par_iter().map(|&x| x).parallel_sum_with_accumulator::<Sum2<_>>();
assert_eq!(100_000.0, s);
# }
```
"
)]
#![deny(missing_docs)]
#![warn(missing_copy_implementations)]
#![warn(missing_debug_implementations)]
#![warn(trivial_casts)]
#![warn(trivial_numeric_casts)]
// This has false positives on #[macro_use],
// see https://github.com/rust-lang/rust/issues/30849
// #![warn(unused_extern_crates)]
#![warn(unused_import_braces)]
#![warn(unused_qualifications)]
#![warn(unused_results)]
#![deny(warnings)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::cast_precision_loss)]
#![warn(clippy::cast_sign_loss)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::many_single_char_names)]
#![warn(clippy::mut_mut)]
#![warn(clippy::mutex_integer)]
#![warn(clippy::non_ascii_literal)]
#![warn(clippy::print_stdout)]
#![warn(clippy::single_match_else)]
#![warn(clippy::string_add)]
#![warn(clippy::string_add_assign)]
#![warn(clippy::unicode_not_nfc)]
#![warn(clippy::unwrap_used)]
#![warn(clippy::wrong_pub_self_convention)]
#![allow(clippy::suspicious_op_assign_impl)]

#[macro_use]
extern crate cfg_if;
#[cfg(doctest)]
#[macro_use]
extern crate doc_comment;
extern crate ieee754;
extern crate num_traits;

#[cfg(feature = "parallel")]
extern crate rayon;

#[cfg(doctest)]
doctest!("../README.md");

pub mod dot;
pub mod sum;
pub mod util;

/// Includes all traits of this crate
pub mod traits {
    #[doc(inline)]
    pub use dot::traits::*;
    #[doc(inline)]
    pub use sum::traits::*;
    #[doc(inline)]
    pub use util::traits::*;
}
