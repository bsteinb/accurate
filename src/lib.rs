//! A collection of (more or less) accurate floating point algorithms

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

#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]
#![cfg_attr(feature="clippy", warn(cast_possible_truncation))]
#![cfg_attr(feature="clippy", warn(cast_possible_wrap))]
#![cfg_attr(feature="clippy", warn(cast_precision_loss))]
#![cfg_attr(feature="clippy", warn(cast_sign_loss))]
#![cfg_attr(feature="clippy", warn(mut_mut))]
#![cfg_attr(feature="clippy", warn(mutex_integer))]
#![cfg_attr(feature="clippy", warn(non_ascii_literal))]
#![cfg_attr(feature="clippy", warn(option_unwrap_used))]
#![cfg_attr(feature="clippy", warn(print_stdout))]
#![cfg_attr(feature="clippy", warn(result_unwrap_used))]
#![cfg_attr(feature="clippy", warn(single_match_else))]
#![cfg_attr(feature="clippy", warn(string_add))]
#![cfg_attr(feature="clippy", warn(string_add_assign))]
#![cfg_attr(feature="clippy", warn(unicode_not_nfc))]
#![cfg_attr(feature="clippy", warn(wrong_pub_self_convention))]

#[macro_use]
extern crate cfg_if;
extern crate ieee754;
extern crate num;

#[cfg(feature = "parallel")]
extern crate rayon;

pub mod dot;
pub mod sum;
pub mod util;

/// Includes all traits of this crate
pub mod traits {
    pub use dot::traits::*;
    pub use sum::traits::*;
    pub use util::traits::*;
}
