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

/// `exp(-x^2 / 2) = exp(-z^2)` with `z = x / sqrt(2)`, from `x` itself since the rounding of
/// `x / sqrt(2)` costs `z^2 * 2e-16` of relative accuracy, 1.4e-13 at the edge of the tail, and
/// with the rounding error of `x * x` (recovered by a fused multiply-add) put back.
fn exp_neg_half_square(x: f64) -> f64 {
    let square = x * x;
    (-0.5 * square).exp() * (-0.5 * x.mul_add(x, -square)).exp()
}

/// The complementary error function `erfc(x / sqrt(2))` for `x >= 0`.
fn erfc_nonnegative(x: f64) -> f64 {
    let z = x / SQRT_2;
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
        1.0 - FRAC_2_SQRT_PI * exp_neg_half_square(x) * sum
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
        exp_neg_half_square(x) / (PI.sqrt() * f)
    }
}

/// The standard normal distribution function `P(X <= x)`.
pub(crate) fn normal_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let tail = 0.5 * erfc_nonnegative(x.abs());
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

// ---------------------------------------------------------------------------------------------
// The Student t distribution
//
// `P(|T| > t)` is the regularized incomplete beta function `I_x(df/2, 1/2)` with
// `x = df / (df + t^2)` (Abramowitz & Stegun 26.7.1), evaluated by its continued fraction with
// the modified Lentz algorithm in the region where it converges quickly and through the
// complement `I_y(1/2, df/2)`, `y = 1 - x = t^2 / (df + t^2)`, elsewhere. The two regions give
// the tail and the central mass `P(|T| < t)` each without cancellation. One and two degrees of
// freedom have closed forms and are taken directly.
// ---------------------------------------------------------------------------------------------

/// Beyond these degrees of freedom the statistic is normal to within `1/(4 df)`, below `1e-12`.
const STUDENT_T_NORMAL_DF: f64 = 1e12;

/// Coefficients `B_2k / (2k (2k - 1))` of the Stirling series of `ln Gamma`.
const STIRLING: [f64; 6] = [
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360_360.0,
];

/// The Stirling series `sum c_k / z^(2k-1)` of `ln Gamma(z)` beyond the leading terms.
fn stirling_series(z: f64) -> f64 {
    let z2 = z * z;
    let mut power = z;
    let mut sum = 0.0;
    for c in STIRLING {
        sum += c / power;
        power *= z2;
    }
    sum
}

/// The ratio `Gamma(a + 1/2) / Gamma(a)` for `a > 0`.
///
/// The argument is shifted above 30 by the recurrence, where the difference of the two Stirling
/// series is summed term by term and the leading terms are combined into
/// `a ln(1 + 1/(2a)) - 1/2` beside the exact `sqrt(a)`, so that no two large logarithms cancel.
fn gamma_half_ratio(a: f64) -> f64 {
    let mut shifted = a;
    let mut correction = 1.0;
    while shifted < 30.0 {
        correction *= shifted / (shifted + 0.5);
        shifted += 1.0;
    }
    // sqrt(a) is taken out so that only a small logarithm, of order 1/a, is exponentiated
    let log_small = shifted * (0.5 / shifted).ln_1p() - 0.5 + stirling_series(shifted + 0.5)
        - stirling_series(shifted);
    shifted.sqrt() * log_small.exp() * correction
}

/// The continued fraction of the incomplete beta function (Numerical Recipes `betacf`, modified
/// Lentz), `I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * betacf(x, a, b)`.
fn beta_continued_fraction(x: f64, y: f64, a: f64, b: f64) -> f64 {
    let tiny = 1e-300;
    let mut c = 1.0;
    // 1 - (a + b) x / (a + 1), written with y = 1 - x where the difference cancels, for x near 1
    let mut d = if x > 0.5 {
        ((1.0 - b) + (a + b) * y) / (a + 1.0)
    } else {
        1.0 - (a + b) * x / (a + 1.0)
    };
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..400_000 {
        let m = f64::from(m);
        let m2 = 2.0 * m;
        let even = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2));
        d = 1.0 + even * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + even / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let even_delta = d * c;
        h *= even_delta;
        let odd = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0));
        d = 1.0 + odd * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + odd / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        // both half steps must have stopped moving, one can be 1 by chance while the sum drifts
        if (delta - 1.0).abs() < 1e-16 && (even_delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// `(P(|T| > s), P(|T| < s))` of a Student t statistic with `df` degrees of freedom at `s >= 0`,
/// the one of the two which converges quickly computed directly and the other as its complement.
fn student_t_two_sided(s: f64, df: f64) -> (f64, f64) {
    if s == 0.0 {
        return (1.0, 0.0);
    }
    if s == f64::INFINITY {
        return (0.0, 1.0);
    }
    if df == 1.0 {
        return if s > 1.0 {
            let tail = 2.0 / PI * (1.0 / s).atan();
            (tail, 1.0 - tail)
        } else {
            let central = 2.0 / PI * s.atan();
            (1.0 - central, central)
        };
    }
    if df == 2.0 {
        let root = (2.0 + s * s).sqrt();
        return if s > 1.0 {
            let tail = 2.0 / (root * (root + s));
            (tail, 1.0 - tail)
        } else {
            let central = s / root;
            (1.0 - central, central)
        };
    }
    let a = 0.5 * df;
    // u = t^2 / df, kept finite for statistics whose square overflows
    let scaled = s / df.sqrt();
    let (x, y, log_one_plus_u) = if scaled > 1e100 {
        (0.0, 1.0, 2.0 * scaled.ln())
    } else {
        let u = scaled * scaled;
        (1.0 / (1.0 + u), u / (1.0 + u), u.ln_1p())
    };
    // x^a y^(1/2) Gamma(a + 1/2) / (Gamma(a) Gamma(1/2)), common to both incomplete betas
    let front = (-a * log_one_plus_u).exp() * y.sqrt() * gamma_half_ratio(a) / PI.sqrt();
    // The fraction converges fast below x = (a + 1) / (a + b + 2), the complement above it.
    // Near the switch the direct fraction loses about 1 / y digits to cancellation and the
    // complement about 1 / tail, which bounds the relative error by 5e-13 at 1e4 degrees of
    // freedom (measured against scipy on a dense grid); moving the switch makes it worse.
    if x < (a + 1.0) / (a + 2.5) {
        let tail = front * beta_continued_fraction(x, y, a, 0.5) / a;
        (tail, 1.0 - tail)
    } else {
        let central = front * beta_continued_fraction(y, x, 0.5, a) / 0.5;
        (1.0 - central, central)
    }
}

/// The Student t distribution function `P(T <= t)` for `df >= 1` degrees of freedom.
pub(crate) fn student_t_cdf(t: f64, df: f64) -> f64 {
    if t.is_nan() || df.is_nan() || df < 1.0 {
        return f64::NAN;
    }
    if df > STUDENT_T_NORMAL_DF {
        return normal_cdf(t);
    }
    let (tail, central) = student_t_two_sided(t.abs(), df);
    if t <= 0.0 {
        0.5 * tail
    } else {
        0.5 + 0.5 * central
    }
}

/// The Student t survival function `P(T > t)`, exactly `student_t_cdf(-t, df)`.
pub(crate) fn student_t_sf(t: f64, df: f64) -> f64 {
    student_t_cdf(-t, df)
}

/// The Student t density at `t >= 0`.
fn student_t_pdf(s: f64, df: f64) -> f64 {
    let scaled = s / df.sqrt();
    let log_one_plus_u = if scaled > 1e100 {
        2.0 * scaled.ln()
    } else {
        (scaled * scaled).ln_1p()
    };
    gamma_half_ratio(0.5 * df) / (df * PI).sqrt() * (-0.5 * (df + 1.0) * log_one_plus_u).exp()
}

/// The `t >= 0` with `P(T > t) = p` for `0 < p < 1/2`.
fn student_t_upper_quantile(p: f64, df: f64) -> f64 {
    if df == 1.0 {
        return 1.0 / (PI * p).tan();
    }
    if df == 2.0 {
        return (1.0 - 2.0 * p) / (2.0 * p * (1.0 - p)).sqrt();
    }
    // Cornish-Fisher expansion about the normal quantile (Abramowitz & Stegun 26.7.5)
    let z = normal_isf(p);
    let z2 = z * z;
    let mut t =
        z + z * (z2 + 1.0) / (4.0 * df) + z * (5.0 * z2 * z2 + 16.0 * z2 + 3.0) / (96.0 * df * df);
    if !(t.is_finite() && t > 0.0) {
        t = z.max(1.0);
    }
    let sf = |t: f64| 0.5 * student_t_two_sided(t, df).0;
    // bracket the root: sf(lo) > p >= sf(hi)
    let mut lo = 0.0_f64;
    let mut hi;
    if sf(t) > p {
        lo = t;
        hi = 2.0 * t;
        while sf(hi) > p {
            lo = hi;
            hi *= 2.0;
            if !hi.is_finite() {
                return f64::INFINITY;
            }
        }
    } else {
        hi = t;
    }
    let log_p = p.ln();
    let mut t = if sf(t) > p { lo } else { hi };
    for _ in 0..500 {
        let s = sf(t);
        if s > p {
            lo = t;
        } else {
            hi = t;
        }
        // Newton on ln sf, which is concave: t - (ln sf - ln p) / (d ln sf / dt)
        let mut next = if s > 0.0 {
            t + (s.ln() - log_p) * s / student_t_pdf(t, df)
        } else {
            f64::NAN
        };
        if !(next.is_finite() && next > lo && next < hi) {
            next = if lo > 0.0 && hi / lo > 8.0 {
                (lo * hi).sqrt()
            } else {
                0.5 * (lo + hi)
            };
        }
        let step = (next - t).abs();
        t = next;
        if step <= 2e-16 * t || hi - lo <= 2e-16 * hi {
            break;
        }
    }
    t
}

/// The Student t inverse survival function: the `t` with `P(T > t) = p`. Infinite at 0 and 1,
/// `NaN` outside `[0, 1]` and below one degree of freedom.
pub(crate) fn student_t_isf(p: f64, df: f64) -> f64 {
    if p.is_nan() || df.is_nan() || !(0.0..=1.0).contains(&p) || df < 1.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p == 1.0 {
        return f64::NEG_INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    if df > STUDENT_T_NORMAL_DF {
        return normal_isf(p);
    }
    if p > 0.5 {
        -student_t_upper_quantile(1.0 - p, df)
    } else {
        student_t_upper_quantile(p, df)
    }
}

/// The Student t quantile, the inverse of [`student_t_cdf`], exactly `-student_t_isf(p, df)`.
pub(crate) fn student_t_ppf(p: f64, df: f64) -> f64 {
    -student_t_isf(p, df)
}

#[cfg(test)]
mod tests {
    // the tables keep the 17 digits scipy prints, verbatim
    #![allow(clippy::excessive_precision)]

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
            // 1e-12 is the requirement inside [1e-6, 1 - 1e-6] and 1e-9 outside; the measured
            // error is below 1e-15 everywhere, so the bound is pinned tighter than the spec
            let bound = 1e-14;
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

    // the upper tail well past the quantile table, which exercises the continued fraction of the
    // tail. These are not scipy values: scipy.stats.norm.sf is only good to about 1e-13 there
    // (7.61985302416046961e-24 and 5.72557122252392658e-300), so the reference is the Mills
    // ratio continued fraction evaluated in 60 digit decimal arithmetic:
    //   python -I -c "from decimal import *; getcontext().prec=60; pi=Decimal('3.14159265358979323846264338327950288419716939937510582097494');
    //   exec('def Q(x):\n x=Decimal(x); f=x\n for k in range(3000,0,-1): f=x+Decimal(k)/f\n return (-x*x/2).exp()/(2*pi).sqrt()/f'); print(Q(10), Q(37))"
    #[test]
    fn the_normal_survival_function_matches_a_high_precision_reference_in_the_upper_tail() {
        for (x, expected) in [
            (10.0, 7.619_853_024_160_526e-24),
            (37.0, 5.725_571_222_524_577e-300),
        ] {
            let got = normal_sf(x);
            assert!(
                ((got - expected) / expected).abs() <= 2e-15,
                "sf({x}) = {got:e}, reference {expected:e}"
            );
        }
    }
    // generated and committed, by (p, df, scipy.stats.t.isf(p, df)):
    //   python -I -c "from scipy.stats import t; ps=[1e-300,1e-100,1e-10,1e-6,0.001,0.025,0.1,0.25,0.5,0.75,0.9,0.975,0.999,1-1e-6,1-1e-10]; dfs=[1.0,2.0,3.0,5.0,10.0,30.0,100.0,1000.0,10000.0,2.5,7.3,41.7]; [print(f'    ({p!r}, {df!r}, {t.isf(p,df):.17e}),') for df in dfs for p in ps if abs(t.isf(p,df))<1e300]"
    const STUDENT_T_ISF: [(f64, f64, f64); 175] = [
        (1e-300, 1.0, 3.18309886183790701e+299),
        (1e-100, 1.0, 3.18309886183790691e+99),
        (1e-10, 1.0, 3.18309886183790636e+09),
        (1e-06, 1.0, 3.18309886182743474e+05),
        (0.001, 1.0, 3.18308838985550437e+02),
        (0.025, 1.0, 1.27062047361747048e+01),
        (0.1, 1.0, 3.07768353717525356e+00),
        (0.25, 1.0, 1.00000000000000022e+00),
        (0.5, 1.0, 0.00000000000000000e+00),
        (0.75, 1.0, -1.00000000000000022e+00),
        (0.9, 1.0, -3.07768353717525445e+00),
        (0.975, 1.0, -1.27062047361746941e+01),
        (0.999, 1.0, -3.18308838985550153e+02),
        (0.999999, 1.0, -3.18309886173590261e+05),
        (0.9999999999, 1.0, -3.18309859846714783e+09),
        (1e-300, 2.0, 7.07106781186547481e+149),
        (1e-100, 2.0, 7.07106781186547512e+49),
        (1e-10, 2.0, 7.07106781080481596e+04),
        (1e-06, 2.0, 7.07105720525933862e+02),
        (0.001, 2.0, 2.23271247701198732e+01),
        (0.025, 2.0, 4.30265272974946367e+00),
        (0.1, 2.0, 1.88561808316412671e+00),
        (0.25, 2.0, 8.16496580927726145e-01),
        (0.5, 2.0, 0.00000000000000000e+00),
        (0.75, 2.0, -8.16496580927726145e-01),
        (0.9, 2.0, -1.88561808316412716e+00),
        (0.975, 2.0, -4.30265272974946189e+00),
        (0.999, 2.0, -2.23271247701198661e+01),
        (0.999999, 2.0, -7.07105720515767189e+02),
        (0.9999999999, 2.0, -7.07106751827344706e+04),
        (1e-100, 3.0, 2.22576982382244193e+33),
        (1e-10, 3.0, 2.22576928468309279e+03),
        (1e-06, 3.0, 1.03299467780419349e+02),
        (0.001, 3.0, 1.02145318524073847e+01),
        (0.025, 3.0, 3.18244630528370864e+00),
        (0.1, 3.0, 1.63774435369620885e+00),
        (0.25, 3.0, 7.64892328404344402e-01),
        (0.5, 3.0, 0.00000000000000000e+00),
        (0.75, 3.0, -7.64892328404344402e-01),
        (0.9, 3.0, -1.63774435369620908e+00),
        (0.975, 3.0, -3.18244630528370775e+00),
        (0.999, 3.0, -1.02145318524073829e+01),
        (0.999999, 3.0, -1.03299467779428980e+02),
        (0.9999999999, 3.0, -2.22576922329607441e+03),
        (1e-100, 5.0, 1.56839255909933777e+20),
        (1e-10, 5.0, 1.56825592708894334e+02),
        (1e-06, 5.0, 2.47710297205159478e+01),
        (0.001, 5.0, 5.89342953135600922e+00),
        (0.025, 5.0, 2.57058183563631548e+00),
        (0.1, 5.0, 1.47588404882448132e+00),
        (0.25, 5.0, 7.26686843800422588e-01),
        (0.5, 5.0, 0.00000000000000000e+00),
        (0.75, 5.0, -7.26686843800422588e-01),
        (0.9, 5.0, -1.47588404882448154e+00),
        (0.975, 5.0, -2.57058183563631459e+00),
        (0.999, 5.0, -5.89342953135600922e+00),
        (0.999999, 5.0, -2.47710297203724927e+01),
        (0.9999999999, 5.0, -1.56825590113280697e+02),
        (1e-100, 10.0, 2.56452571894819794e+10),
        (1e-10, 10.0, 2.54660080216977249e+01),
        (1e-06, 10.0, 9.75199549094058149e+00),
        (0.001, 10.0, 4.14370049404658936e+00),
        (0.025, 10.0, 2.22813885198627526e+00),
        (0.1, 10.0, 1.37218364111033586e+00),
        (0.25, 10.0, 6.99812061312431677e-01),
        (0.5, 10.0, 0.00000000000000000e+00),
        (0.75, 10.0, -6.99812061312431677e-01),
        (0.9, 10.0, -1.37218364111033608e+00),
        (0.975, 10.0, -2.22813885198627393e+00),
        (0.999, 10.0, -4.14370049404658936e+00),
        (0.999999, 10.0, -9.75199549090985585e+00),
        (0.9999999999, 10.0, -2.54660078080160126e+01),
        (1e-300, 30.0, 5.01785753605050812e+10),
        (1e-100, 30.0, 1.08106450011439756e+04),
        (1e-10, 30.0, 9.37748978040714398e+00),
        (1e-06, 30.0, 5.87111712041888190e+00),
        (0.001, 30.0, 3.38518486682930497e+00),
        (0.025, 30.0, 2.04227245630123821e+00),
        (0.1, 30.0, 1.31041502539139554e+00),
        (0.25, 30.0, 6.82755693321292711e-01),
        (0.5, 30.0, 0.00000000000000000e+00),
        (0.75, 30.0, -6.82755693321292711e-01),
        (0.9, 30.0, -1.31041502539139554e+00),
        (0.975, 30.0, -2.04227245630123777e+00),
        (0.999, 30.0, -3.38518486682930453e+00),
        (0.999999, 30.0, -5.87111712040862432e+00),
        (0.9999999999, 30.0, -9.37748974607977459e+00),
        (1e-300, 100.0, 9.68273428706956474e+03),
        (1e-100, 100.0, 9.63147430778309257e+01),
        (1e-10, 100.0, 7.08337548140072215e+00),
        (1e-06, 100.0, 5.04883087722834567e+00),
        (0.001, 100.0, 3.17373949373878306e+00),
        (0.025, 100.0, 1.98397151852355202e+00),
        (0.1, 100.0, 1.29007476134651600e+00),
        (0.25, 100.0, 6.76951043011471687e-01),
        (0.5, 100.0, 0.00000000000000000e+00),
        (0.75, 100.0, -6.76951043011471687e-01),
        (0.9, 100.0, -1.29007476134651600e+00),
        (0.975, 100.0, -1.98397151852355180e+00),
        (0.999, 100.0, -3.17373949373878306e+00),
        (0.999999, 100.0, -5.04883087722144630e+00),
        (0.9999999999, 100.0, -7.08337546418368902e+00),
        (1e-300, 1000.0, 5.42913885530517462e+01),
        (1e-100, 1000.0, 2.39306170878264375e+01),
        (1e-10, 1000.0, 6.42787628313421333e+00),
        (1e-06, 1000.0, 4.78160862045835078e+00),
        (0.001, 1000.0, 3.09840216391292333e+00),
        (0.025, 1000.0, 1.96233908082640851e+00),
        (0.1, 1000.0, 1.28239872146092448e+00),
        (0.25, 1000.0, 6.74735164607009330e-01),
        (0.5, 1000.0, 0.00000000000000000e+00),
        (0.75, 1000.0, -6.74735164607009330e-01),
        (0.9, 1000.0, -1.28239872146092471e+00),
        (0.975, 1000.0, -1.96233908082640829e+00),
        (0.999, 1000.0, -3.09840216391292333e+00),
        (0.999999, 1000.0, -4.78160862045243906e+00),
        (0.9999999999, 1000.0, -6.42787627003302831e+00),
        (1e-300, 10000.0, 3.83563843210042350e+01),
        (1e-100, 10000.0, 2.15169741939149830e+01),
        (1e-10, 10000.0, 6.36794135148280294e+00),
        (1e-06, 10000.0, 4.75622968505677957e+00),
        (0.001, 10000.0, 3.09104751603061212e+00),
        (0.025, 10000.0, 1.96020123989062633e+00),
        (0.1, 10000.0, 1.28163622973047753e+00),
        (0.25, 10000.0, 6.74514284483592741e-01),
        (0.5, 10000.0, 0.00000000000000000e+00),
        (0.75, 10000.0, -6.74514284483592741e-01),
        (0.9, 10000.0, -1.28163622973047753e+00),
        (0.975, 10000.0, -1.96020123989062589e+00),
        (0.999, 10000.0, -3.09104751603061212e+00),
        (0.999999, 10000.0, -4.75622968505095756e+00),
        (0.9999999999, 10000.0, -6.36794133873722945e+00),
        (1e-100, 2.5, 8.76543788227999143e+39),
        (1e-10, 2.5, 8.76543777136457174e+03),
        (1e-06, 2.5, 2.20173429178236859e+02),
        (0.001, 2.5, 1.38221931108659639e+01),
        (0.025, 2.5, 3.57465484200368344e+00),
        (0.1, 2.5, 1.73025092880717613e+00),
        (0.25, 2.5, 7.85013682992299233e-01),
        (0.5, 2.5, 0.00000000000000000e+00),
        (0.75, 2.5, -7.85013682992299233e-01),
        (0.9, 2.5, -1.73025092880717657e+00),
        (0.975, 2.5, -3.57465484200368167e+00),
        (0.999, 2.5, -1.38221931108659604e+01),
        (0.999999, 2.5, -2.20173429175704257e+02),
        (0.9999999999, 2.5, -8.76543748126235369e+03),
        (1e-100, 7.3, 1.03385310291830312e+14),
        (1e-10, 7.3, 4.84272030346379765e+01),
        (1e-06, 7.3, 1.34935230607446410e+01),
        (0.001, 7.3, 4.68886599353837319e+00),
        (0.025, 7.3, 2.34506673654770337e+00),
        (0.1, 7.3, 1.40892231774804211e+00),
        (0.25, 7.3, 7.09573020456990999e-01),
        (0.5, 7.3, 0.00000000000000000e+00),
        (0.75, 7.3, -7.09573020456990999e-01),
        (0.9, 7.3, -1.40892231774804233e+00),
        (0.975, 7.3, -2.34506673654770292e+00),
        (0.999, 7.3, -4.68886599353837230e+00),
        (0.999999, 7.3, -1.34935230606895953e+01),
        (0.9999999999, 7.3, -4.84272024842252620e+01),
        (1e-300, 41.7, 9.44612320000880361e+07),
        (1e-100, 41.7, 1.51038150001611257e+03),
        (1e-10, 41.7, 8.33889778297558237e+00),
        (1e-06, 41.7, 5.51706750203032570e+00),
        (0.001, 41.7, 3.29751885705094816e+00),
        (0.025, 41.7, 2.01851192299642479e+00),
        (0.1, 41.7, 1.30218524972607885e+00),
        (0.25, 41.7, 6.80418717785332916e-01),
        (0.5, 41.7, 0.00000000000000000e+00),
        (0.75, 41.7, -6.80418717785332916e-01),
        (0.9, 41.7, -1.30218524972607885e+00),
        (0.975, 41.7, -2.01851192299642479e+00),
        (0.999, 41.7, -3.29751885705094772e+00),
        (0.999999, 41.7, -5.51706750202156915e+00),
        (0.9999999999, 41.7, -8.33889775685721979e+00),
    ];

    // (t, df, scipy.stats.t.sf(t, df)), which tests the distribution function independently of
    // the quantile:
    //   python -I -c "from scipy.stats import t; dfs=[1.0,2.0,3.0,5.0,10.0,30.0,100.0,1000.0,10000.0,2.5,7.3,41.7]; [print(f'    ({x!r}, {df!r}, {t.sf(x,df):.17e}),') for df in dfs for x in (0.1,1.0,2.5,10.0,40.0) if t.sf(x,df)>0]"
    const STUDENT_T_SF: [(f64, f64, f64); 59] = [
        (0.1, 1.0, 4.68274482569446482e-01),
        (1.0, 1.0, 2.50000000000000056e-01),
        (2.5, 1.0, 1.21118941590843410e-01),
        (10.0, 1.0, 3.17255174305535667e-02),
        (40.0, 1.0, 7.95608991202581203e-03),
        (0.1, 2.0, 4.64732719207070100e-01),
        (1.0, 2.0, 2.11324865405187107e-01),
        (2.5, 2.0, 6.48058601107554116e-02),
        (10.0, 2.0, 4.92622851166284528e-03),
        (40.0, 2.0, 3.12207336092370298e-04),
        (0.1, 3.0, 4.63326174400402968e-01),
        (1.0, 3.0, 1.95501109477885238e-01),
        (2.5, 3.0, 4.38533235040327804e-02),
        (10.0, 3.0, 1.06419952920707513e-03),
        (40.0, 3.0, 1.71903403945792530e-05),
        (0.1, 5.0, 4.62115070577330200e-01),
        (1.0, 5.0, 1.81608733824561441e-01),
        (2.5, 5.0, 2.72450496711881297e-02),
        (10.0, 5.0, 8.54737878714818139e-05),
        (40.0, 5.0, 9.20598108588647708e-08),
        (0.1, 10.0, 4.61160359282204113e-01),
        (1.0, 10.0, 1.70446566151029988e-01),
        (2.5, 10.0, 1.57234221183044087e-02),
        (10.0, 10.0, 7.94776587798206244e-07),
        (40.0, 10.0, 1.14042887154287676e-12),
        (0.1, 30.0, 4.60504805895135594e-01),
        (1.0, 30.0, 1.62654307713015089e-01),
        (2.5, 30.0, 9.05782453403334437e-03),
        (10.0, 30.0, 2.28762570411480903e-11),
        (40.0, 30.0, 6.86302259720320855e-28),
        (0.1, 100.0, 4.60272265547925608e-01),
        (1.0, 100.0, 1.59862077892061666e-01),
        (2.5, 100.0, 7.02289456203858582e-03),
        (10.0, 100.0, 4.95084449229706527e-17),
        (40.0, 100.0, 1.23105380107003544e-63),
        (0.1, 1000.0, 4.60182184511802039e-01),
        (1.0, 1000.0, 1.58776209042336275e-01),
        (2.5, 1000.0, 6.28928390054540355e-03),
        (10.0, 1000.0, 8.33535147930006874e-23),
        (40.0, 1000.0, 5.23942607758663608e-210),
        (0.1, 10000.0, 4.60173165015537511e-01),
        (1.0, 10000.0, 1.58667352165214576e-01),
        (2.5, 10000.0, 6.21760977529184876e-03),
        (10.0, 10000.0, 9.81640371433190942e-24),
        (0.1, 2.5, 4.63903272031244873e-01),
        (1.0, 2.5, 2.02030513639136822e-01),
        (2.5, 2.5, 5.22849815390777004e-02),
        (10.0, 2.5, 2.22074788365371175e-03),
        (40.0, 2.5, 7.09781714524668992e-05),
        (0.1, 7.3, 4.61517748569914732e-01),
        (1.0, 7.3, 1.74649228226557235e-01),
        (2.5, 7.3, 1.98251173328002142e-02),
        (10.0, 7.3, 8.05012760074050624e-06),
        (40.0, 7.3, 4.01856758395686239e-10),
        (0.1, 41.7, 4.60411776975155396e-01),
        (1.0, 41.7, 1.61539118507723983e-01),
        (2.5, 41.7, 8.22028082387284836e-03),
        (10.0, 41.7, 6.07637552367111743e-13),
        (40.0, 41.7, 3.42395398137863883e-35),
    ];

    fn relative(got: f64, expected: f64) -> f64 {
        ((got - expected) / expected).abs()
    }

    #[test]
    fn the_student_t_quantile_matches_scipy_over_the_whole_domain() {
        let (mut worst_integer, mut worst_fraction) = (0.0_f64, 0.0_f64);
        for (p, df, expected) in STUDENT_T_ISF {
            let got = student_t_isf(p, df);
            if expected == 0.0 {
                assert_eq!(got, 0.0, "isf({p}, {df})");
                continue;
            }
            let error = relative(got, expected);
            let bound = if (1e-6..=1.0 - 1e-6).contains(&p) {
                1e-12
            } else {
                1e-9
            };
            assert!(
                error <= bound,
                "isf({p:e}, {df}) = {got:e}, scipy {expected:e}, relative {error:e}"
            );
            if df.fract() == 0.0 {
                worst_integer = worst_integer.max(error);
            } else {
                worst_fraction = worst_fraction.max(error);
            }
        }
        eprintln!(
            "worst isf error: integer df {worst_integer:e}, fractional df {worst_fraction:e}"
        );
    }

    #[test]
    fn the_student_t_survival_function_matches_scipy_over_the_whole_domain() {
        let (mut worst_integer, mut worst_fraction) = (0.0_f64, 0.0_f64);
        for (t, df, expected) in STUDENT_T_SF {
            let got = student_t_sf(t, df);
            let error = relative(got, expected);
            let bound = if expected >= 1e-6 { 1e-12 } else { 1e-9 };
            assert!(
                error <= bound,
                "sf({t}, {df}) = {got:e}, scipy {expected:e}, relative {error:e}"
            );
            if df.fract() == 0.0 {
                worst_integer = worst_integer.max(error);
            } else {
                worst_fraction = worst_fraction.max(error);
            }
            // the distribution function is the complement, also of a negative statistic
            assert!((student_t_cdf(t, df) - (1.0 - expected)).abs() <= 1e-12);
            assert!(relative(student_t_cdf(-t, df), expected) <= bound);
        }
        eprintln!("worst sf error: integer df {worst_integer:e}, fractional df {worst_fraction:e}");
    }

    #[test]
    fn the_student_t_quantile_at_one_degree_of_freedom_is_the_cauchy_closed_form() {
        for p in [0.001, 0.025, 0.1, 0.25] {
            let expected = 1.0 / (PI * p).tan();
            let got = student_t_isf(p, 1.0);
            assert!(relative(got, expected) <= 1e-12, "p = {p}");
            // the general branch, one ulp of df away, agrees with the closed form
            let general = student_t_isf(p, 1.0 + 1e-9);
            assert!(relative(general, expected) <= 1e-7, "p = {p}");
        }
        // the closed form of the distribution function: F(1) = 3/4
        assert!(relative(student_t_cdf(1.0, 1.0), 0.75) <= 1e-15);
    }

    #[test]
    fn the_student_t_quantile_at_two_degrees_of_freedom_is_the_closed_form() {
        // t = (1 - 2p) / sqrt(2 p (1 - p)), against scipy.stats.t.isf(p, 2)
        assert!(relative(student_t_isf(0.025, 2.0), 4.302_652_729_749_463_7) <= 1e-13);
        assert!(relative(student_t_isf(0.05, 2.0), 2.919_985_580_353_725_6) <= 1e-13);
    }

    #[test]
    fn the_student_t_quantile_approaches_the_normal_at_ten_thousand_degrees_of_freedom() {
        for p in [1e-6, 0.001, 0.025, 0.1, 0.25, 0.4] {
            let z = normal_isf(p);
            let large = student_t_isf(p, 1e4);
            assert!(relative(large, z) <= 1e-3, "p = {p}");
            assert!(large > z, "heavier tail than the normal, p = {p}");
            assert!(
                relative(student_t_isf(p, 1e8), z) <= 1e-7,
                "p = {p}, 1e8 degrees of freedom"
            );
            // beyond the switch to the normal the two are the same function
            assert_eq!(student_t_isf(p, 1e13), z);
            assert_eq!(student_t_cdf(z, 1e13), normal_cdf(z));
        }
    }

    #[test]
    fn the_survival_function_of_an_infinite_statistic_is_exactly_zero() {
        for df in [1.0, 2.0, 5.0, 2.5, 1e4] {
            assert_eq!(student_t_sf(f64::INFINITY, df), 0.0);
            assert_eq!(student_t_sf(f64::NEG_INFINITY, df), 1.0);
            assert_eq!(student_t_cdf(f64::INFINITY, df), 1.0);
            assert_eq!(student_t_cdf(f64::NEG_INFINITY, df), 0.0);
            assert_eq!(student_t_cdf(0.0, df), 0.5);
        }
    }

    #[test]
    fn the_student_t_quantile_is_the_inverse_of_the_survival_function() {
        for (p, df, _) in STUDENT_T_ISF {
            if p == 1.0 - 1e-10 || df == 1.0 {
                continue;
            }
            let back = student_t_sf(student_t_isf(p, df), df);
            assert!(
                relative(back, p) <= 1e-10,
                "p = {p:e}, df = {df}, back {back:e}"
            );
        }
    }

    #[test]
    fn the_student_t_functions_are_nan_or_infinite_outside_their_domain() {
        assert_eq!(student_t_isf(0.0, 5.0), f64::INFINITY);
        assert_eq!(student_t_isf(1.0, 5.0), f64::NEG_INFINITY);
        assert_eq!(student_t_ppf(0.0, 5.0), f64::NEG_INFINITY);
        assert_eq!(student_t_ppf(0.5, 5.0), 0.0);
        assert!(student_t_isf(1.5, 5.0).is_nan());
        assert!(student_t_isf(0.3, 0.5).is_nan());
        assert!(student_t_cdf(f64::NAN, 5.0).is_nan());
        assert!(student_t_cdf(1.0, 0.5).is_nan());
        // scipy.stats.t.ppf(0.975, 10) and ppf(0.025, 10)
        assert!(relative(student_t_ppf(0.975, 10.0), 2.228_138_851_986_274) <= 1e-13);
        assert!(relative(student_t_ppf(0.025, 10.0), -2.228_138_851_986_274) <= 1e-13);
    }

    #[test]
    fn the_student_t_quantile_is_monotone_down_to_the_smallest_probabilities() {
        for df in [1.0, 1.5, 3.0, 41.7, 1e4] {
            let mut previous = 0.0_f64;
            let mut p = 0.45_f64;
            while p > 1e-300 {
                let t = student_t_isf(p, df);
                assert!(t > previous, "not monotone at p = {p:e}, df = {df}");
                previous = t;
                p *= 0.3;
            }
        }
    }
}
