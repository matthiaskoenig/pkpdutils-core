//! The normal distribution function and its quantile.
//!
//! Written by hand, since the crate takes no dependency: the distribution function is the
//! complementary error function by a power series near the origin and by a continued fraction
//! (Abramowitz & Stegun 7.1.14, evaluated with the modified Lentz algorithm) in the tails, both
//! to the last places and with a relative error which stays small down to the underflow of the
//! density. The quantile starts from the rational approximation of Abramowitz & Stegun 26.2.23
//! and is refined by Halley steps against the distribution function, which is what carries it
//! to full precision. The survival function and the inverse survival function are the
//! distribution function and the quantile at the mirrored argument, exact by symmetry.

use std::f64::consts::{FRAC_2_SQRT_PI, PI, SQRT_2};

/// The complementary error function `erfc(z)` for `z >= 0`.
fn erfc_nonnegative(z: f64) -> f64 {
    if z < 1.5 {
        // erf(z) = 2/sqrt(pi) * exp(-z^2) * sum z^(2n+1) 2^n / (2n+1)!!, all terms positive
        let z2 = z * z;
        let mut term = z;
        let mut sum = z;
        let mut n = 0.0_f64;
        while term > sum * 1e-18 {
            n += 1.0;
            term *= 2.0 * z2 / (2.0 * n + 1.0);
            sum += term;
        }
        1.0 - FRAC_2_SQRT_PI * (-z2).exp() * sum
    } else {
        // erfc(z) = exp(-z^2) / sqrt(pi) * 1 / (z + (1/2) / (z + 1 / (z + (3/2) / (z + ...))))
        let tiny = 1e-300;
        let mut f = z;
        let mut c = z;
        let mut d = 0.0_f64;
        for k in 1..2000 {
            let a = 0.5 * f64::from(k);
            d = z + a * d;
            if d == 0.0 {
                d = tiny;
            }
            c = z + a / c;
            if c == 0.0 {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = c * d;
            f *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        (-z * z).exp() / (PI.sqrt() * f)
    }
}

/// The standard normal distribution function `P(X <= x)`.
pub(crate) fn normal_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let z = x.abs() / SQRT_2;
    let tail = 0.5 * erfc_nonnegative(z);
    if x <= 0.0 { tail } else { 1.0 - tail }
}

/// The standard normal survival function `P(X > x)`, exactly `normal_cdf(-x)`.
pub(crate) fn normal_sf(x: f64) -> f64 {
    normal_cdf(-x)
}

/// The standard normal quantile, the inverse of [`normal_cdf`]: `-inf` at 0, `inf` at 1 and
/// `NaN` outside `[0, 1]`.
pub(crate) fn normal_ppf(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    if p > 0.5 {
        // 1 - p is exact for p >= 0.5
        return -normal_ppf(1.0 - p);
    }
    let t = (-2.0 * p.ln()).sqrt();
    let mut x = -(t
        - (2.515517 + t * (0.802853 + t * 0.010328))
            / (1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308))));
    for _ in 0..8 {
        let density = (-0.5 * x * x).exp() / (2.0 * PI).sqrt();
        let u = (normal_cdf(x) - p) / density;
        let step = u / (1.0 + 0.5 * x * u);
        if !step.is_finite() {
            break;
        }
        x -= step;
        if step.abs() <= 1e-16 * x.abs() {
            break;
        }
    }
    x
}

/// The standard normal inverse survival function, exactly `-normal_ppf(p)`.
pub(crate) fn normal_isf(p: f64) -> f64 {
    -normal_ppf(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    // the table is generated once and committed, by:
    //   python -c "from scipy.stats import norm; [print(f'({p:e}, {norm.ppf(p):.17e}),') for p in (1e-300,1e-100,1e-10,1e-6,0.001,0.025,0.1,0.25,0.5,0.75,0.9,0.975,0.999,1-1e-6,1-1e-10)]"
    // (the two probabilities near one are written as the float expression, which `{p:e}` rounds)
    const NORMAL_PPF: [(f64, f64); 15] = [
        (1.000000e-300, -3.70470962993612005e+01),
        (1.000000e-100, -2.12734535609653221e+01),
        (1.000000e-10, -6.36134090240405570e+00),
        (1.000000e-06, -4.75342430882289868e+00),
        (1.000000e-03, -3.09023230616781319e+00),
        (2.500000e-02, -1.95996398454005449e+00),
        (1.000000e-01, -1.28155156554460037e+00),
        (2.500000e-01, -6.74489750196081705e-01),
        (5.000000e-01, 0.00000000000000000e+00),
        (7.500000e-01, 6.74489750196081705e-01),
        (9.000000e-01, 1.28155156554460037e+00),
        (9.750000e-01, 1.95996398454005405e+00),
        (9.990000e-01, 3.09023230616781319e+00),
        (1.0 - 1e-6, 4.75342430881708733e+00),
        (1.0 - 1e-10, 6.36134088969742173e+00),
    ];

    #[test]
    fn the_normal_quantile_matches_scipy_over_the_whole_domain() {
        let mut worst = 0.0_f64;
        for (p, expected) in NORMAL_PPF {
            let got = normal_ppf(p);
            if expected == 0.0 {
                assert_eq!(got, 0.0);
                continue;
            }
            let relative = ((got - expected) / expected).abs();
            let bound = if (1e-6..=1.0 - 1e-6).contains(&p) {
                1e-12
            } else {
                1e-9
            };
            assert!(
                relative <= bound,
                "ppf({p:e}) = {got:e}, scipy {expected:e}, relative {relative:e}"
            );
            worst = worst.max(relative);
        }
        assert!(worst < 1e-9);
    }

    #[test]
    fn the_distribution_function_matches_known_values() {
        // independent of the quantile: Phi(0), Phi(1.96) = 0.9750021048517795, Phi(-37.047...) = 1e-300
        assert_eq!(normal_cdf(0.0), 0.5);
        assert!(((normal_cdf(1.959_963_984_540_054) - 0.975) / 0.975).abs() < 1e-14);
        assert!(((normal_cdf(-6.361_340_902_404_056) - 1e-10) / 1e-10).abs() < 1e-12);
        assert!(((normal_cdf(-37.047_096_299_361_2) - 1e-300) / 1e-300).abs() < 1e-9);
        assert!(normal_cdf(f64::NAN).is_nan());
    }

    #[test]
    fn the_normal_quantile_is_zero_at_one_half_and_symmetric() {
        // dyadic probabilities, whose complement is exact in floating point
        assert_eq!(normal_ppf(0.5), 0.0);
        for p in [
            2f64.powi(-40),
            2f64.powi(-20),
            2f64.powi(-10),
            0.03125,
            0.125,
            0.25,
            0.375,
        ] {
            let low = normal_ppf(p);
            let high = normal_ppf(1.0 - p);
            assert!(low < 0.0 && high > 0.0);
            assert!(
                (low + high).abs() <= 1e-15 * low.abs().max(1.0),
                "p = {p:e}"
            );
        }
        assert_eq!(normal_isf(0.025), -normal_ppf(0.025));
        assert_eq!(normal_sf(1.3), normal_cdf(-1.3));
    }

    #[test]
    fn the_normal_quantile_is_infinite_at_the_bounds_and_nan_outside() {
        assert_eq!(normal_ppf(0.0), f64::NEG_INFINITY);
        assert_eq!(normal_ppf(1.0), f64::INFINITY);
        assert!(normal_ppf(-0.1).is_nan());
        assert!(normal_ppf(1.1).is_nan());
        assert!(normal_ppf(f64::NAN).is_nan());
    }

    #[test]
    fn the_normal_quantile_is_monotone_down_to_the_smallest_probabilities() {
        let mut previous = f64::NEG_INFINITY;
        let mut p = 1e-320_f64;
        while p < 0.5 {
            let x = normal_ppf(p);
            assert!(x > previous, "not monotone at {p:e}");
            previous = x;
            p *= 1.7;
        }
    }

    #[test]
    fn the_survival_function_is_the_inverse_of_the_inverse_survival_function() {
        for p in [1e-10, 1e-6, 0.001, 0.025, 0.3, 0.5, 0.7, 0.975, 0.999] {
            let back = normal_sf(normal_isf(p));
            assert!(((back - p) / p).abs() <= 1e-12, "p = {p:e}, back {back:e}");
        }
    }
}
