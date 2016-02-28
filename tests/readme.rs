extern crate accurate;

use accurate::*;

#[test]
fn main() {
    let x = vec![1.0, 2.0, 3.0];
    let s = x.sum_with_accumulator::<Sum2<_>>();
    assert_eq!(6.0f64, s);
}
