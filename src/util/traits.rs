//! Common traits

use ieee754::Ieee754;

use num_traits::{Float, PrimInt, ToPrimitive, Zero};

/// Sum transformation
pub trait TwoSum: Float {}

impl<F> TwoSum for F where F: Float {}

/// Split a floating-point number
pub trait Split: Float {
    /// Split factor used in the algorithm
    fn split_factor() -> Self;

    /// Split a floating-point number
    ///
    /// Splits a number `x` into two parts:
    ///
    /// ```not_rust
    /// x = h + t
    /// ```
    ///
    /// with `h` and `t` nonoverlapping and `t.abs() <= h.abs()`
    ///
    /// # References
    ///
    /// Due to Veltkamp, published in [Dekker 71](http://dx.doi.org/10.1007/BF01397083)
    #[inline]
    fn split(self) -> (Self, Self) {
        let x = self;
        let c = Self::split_factor() * x;
        let h = c - (c - x);
        let t = x - h;
        (h, t)
    }
}

impl Split for f32 {
    #[inline]
    fn split_factor() -> Self {
        4097.0
    }
}

impl Split for f64 {
    #[inline]
    fn split_factor() -> Self {
        134_217_729.0
    }
}

cfg_if! {
    if #[cfg(feature = "fma")] {
        /// Product transformation
        pub trait TwoProduct: Float { }

        impl<F> TwoProduct for F where F: Float { }
    } else {
        /// Product transformation
        pub trait TwoProduct: Float + Split { }

        impl<F> TwoProduct for F where F: Float + Split { }
    }
}

/// Half a unit in the last place (ULP)
pub trait HalfUlp {
    /// Check whether something has the form of half a ULP
    fn has_half_ulp_form(self) -> bool;

    /// Calculate half a ULP of a number
    fn half_ulp(self) -> Self;
}

impl<F> HalfUlp for F
where
    F: Float + Ieee754,
    F::Significand: Zero + Eq,
{
    #[inline]
    fn has_half_ulp_form(self) -> bool {
        // self is not zero and significand has all zero visible bits
        self != F::zero() && self.decompose_raw().2 == F::Significand::zero()
    }

    #[inline]
    fn half_ulp(self) -> Self {
        self.ulp().unwrap_or_else(Self::zero) / F::one().exp2()
    }
}

/// Correctly rounded sum of three non-overlapping numbers
pub trait Round3: Float + Ieee754 + HalfUlp {}

impl<F> Round3 for F where F: Float + Ieee754 + HalfUlp {}

/// Describes the layout of a floating-point number
pub trait FloatFormat {
    /// The number format's base
    fn base() -> u32;

    /// The length of the number format's significand field
    fn significand_digits() -> u32;

    /// The length of the number format's exponent field
    fn exponent_digits() -> u32;

    /// The base raised to the power of the exponent`s length
    #[inline]
    fn base_pow_exponent_digits() -> usize {
        Self::base()
            .to_usize()
            .expect("floating-point base cannot be converted to usize")
            .pow(Self::exponent_digits())
    }

    /// The base raised to the power of half the mantissa`s length
    #[inline]
    fn base_pow_significand_digits_half() -> usize {
        Self::base()
            .to_usize()
            .expect("floating-point base cannot be converted to usize")
            .pow(Self::significand_digits() / 2)
    }
}

impl FloatFormat for f32 {
    #[inline]
    fn base() -> u32 {
        2
    }

    #[inline]
    fn significand_digits() -> u32 {
        24
    }

    #[inline]
    fn exponent_digits() -> u32 {
        8
    }

    #[inline]
    fn base_pow_exponent_digits() -> usize {
        256
    }

    #[inline]
    fn base_pow_significand_digits_half() -> usize {
        4096
    }
}

impl FloatFormat for f64 {
    #[inline]
    fn base() -> u32 {
        2
    }

    #[inline]
    fn significand_digits() -> u32 {
        53
    }

    #[inline]
    fn exponent_digits() -> u32 {
        11
    }

    #[inline]
    fn base_pow_exponent_digits() -> usize {
        2048
    }

    #[inline]
    fn base_pow_significand_digits_half() -> usize {
        67_108_864
    }
}

/// Extract the raw exponent of a floating-point number
pub trait RawExponent {
    /// The raw bits of the exponent
    fn raw_exponent(self) -> usize;
}

impl<F> RawExponent for F
where
    F: Ieee754,
    F::RawExponent: PrimInt,
{
    #[inline]
    fn raw_exponent(self) -> usize {
        self.decompose_raw()
            .1
            .to_usize()
            .expect("exponent does not fit in a usize.")
    }
}
