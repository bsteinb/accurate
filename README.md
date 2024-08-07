# (More or less) accurate floating point algorithms

[![GitHub Actions status][gh-actions-shield]][gh-actions] [![Documentation: hosted][doc-shield]][doc] [![License: Apache License 2.0 or MIT][license-shield]][license] [![latest GitHub release][release-shield]][release] [![crate on crates.io][crate-shield]][crate]

This crate provides several algorithms that implement highly accurate or even guaranteed correct
sum and dot product for floating-point numbers without resorting to arbitrary precision arithmetic.
These algorithms are available:

- Kahan summation, based on [Kahan 65](https://doi.org/10.1145%2F363707.363723)
- Neumaier summation, based on [Neumaier 74](https://doi.org/10.1002%2Fzamm.19740540106)
- Klein summation, based on [Klein 06](https://doi.org/10.1007%2Fs00607-005-0139-x)
- Accurate sum and dot product, based on [Ogita, Rump, and Oishi 05](http://dx.doi.org/10.1137/030601818)
- Online exact summation, based on [Zhu and Hayes 10](http://dx.doi.org/10.1145/1824801.1824815)

[gh-actions-shield]: https://img.shields.io/github/actions/workflow/status/bsteinb/accurate/test.yml?branch=master&style=flat-square
[gh-actions]: https://github.com/bsteinb/accurate/actions
[doc-shield]: https://img.shields.io/badge/documentation-docs.rs-blue.svg?style=flat-square
[doc]: https://docs.rs/accurate/
[license-shield]: https://img.shields.io/badge/license-Apache_License_2.0_or_MIT-blue.svg?style=flat-square
[license]: https://github.com/bsteinb/accurate#license
[release-shield]: https://img.shields.io/github/release/bsteinb/accurate.svg?style=flat-square
[release]: https://github.com/bsteinb/accurate/releases/latest
[crate-shield]: https://img.shields.io/crates/v/accurate.svg?style=flat-square
[crate]: https://crates.io/crates/accurate

## Usage

Add the `accurate` crate as a dependency in your `Cargo.toml`:

```toml
[dependencies]
accurate = "0.4"
```

Then use it in your program like this:

```rust
extern crate accurate;

use accurate::traits::*;
use accurate::sum::Sum2;

fn main() {
  let x = vec![1.0, 2.0, 3.0];
  let s = x.sum_with_accumulator::<Sum2<_>>();
  assert_eq!(6.0f64, s);
}
```

## Documentation

Documentation for the latest version of the crate is [on docs.rs][doc].

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
