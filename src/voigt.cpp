/* voigt.cpp — voigt profile approximations for the lyman-alpha
 * cross section. three implementations available, selected at
 * compile time via VOIGT_FUNCTION in rt_definitions.h.
 *
 * Niranjan Reji, Raman Research Institute, March 2026
 * assisted by Claude (Anthropic) */

#include "common.h"
#include <rt_definitions.h>

/**
 * @brief Smith et al. (2015) / COLT continued-fraction approximation.
 * @param x dimensionless frequency offset from line center
 * @param a voigt damping parameter
 * @return voigt profile H(a, x)
 */
double voigt_smith(double x, double a) {
    const double z = x * x;

    /* far-wing asymptotic expansion */
    if (z >= 25.0) {
        double H = z - 5.5;
        H = z - 3.5 - 5.0 / H;
        H = z - 1.5 - 1.5 / H;
        return (a / sqrt_pi) / H;
    }

    const double ez = exp(-z);

    /* intermediate regime */
    if (z > 3.0) {
        double H = z - B8;
        H = z - B6 + B7 / H;
        H = z + B4 + B5 / H;
        H = z - B2 + B3 / H;
        H = B0 + B1 / H;
        return ez + a * H;
    }

    /* core region */
    {
        double H = z - A6;
        H = z - A4 + A5 / H;
        H = z - A2 + A3 / H;
        H = A0 + A1 / H;
        return ez * (1.0 - a * H);
    }
}

/**
 * @brief Tasitsiomi (2006) analytic fitting formula.
 * @param x dimensionless frequency offset from line center
 * @param a voigt damping parameter
 * @return voigt profile H(a, x)
 */
double voigt_tasitsiomi(double x, double a) {
    const double x2 = x * x;
    const double z = (x2 - 0.855) / (x2 + 3.42);

    double q = 0.0;
    if (z > 0.0) {
        q = z * (1.0 + 21.0 / x2) * a / pi / (x2 + 1.0);
        q *= ((5.674 * z - 9.207) * z + 4.421) * z + 0.1117;
    }
    return sqrt_pi * q + exp(-x2);
}

/**
 * @brief Humlicek (1982) region-4 rational approximation.
 * @param x dimensionless frequency offset from line center
 * @param a voigt damping parameter
 * @return voigt profile H(a, x)
 */
double voigt_humlicek(double x, double a) {
    using C = std::complex<double>;

    const double absx = fabs(x);
    const double s = absx + a;

    const C t(a, -x);
    const C u = t * t;

    C w4;

    if (s >= 15.0) {
        w4 = t * inv_sqrt_pi / (0.5 + u);
    } else if (s >= 5.5) {
        w4 = t * (1.410474 + u * inv_sqrt_pi) / (0.75 + u * (3.0 + u));
    } else if (a >= (0.195 * absx - 0.176)) {
        w4 = (16.4955 + t*(20.20933 + t*(11.96482 + t*(3.778987 + t*inv_sqrt_pi))))
           / (16.4955 + t*(38.82363 + t*(39.27121 + t*(21.69274 + t*(6.699398 + t)))));
    } else {
        w4 = exp(u) - t*(36183.30536 - u*(3321.990492 - u*(1540.786893 - u*(219.0312964
           - u*(35.76682780 - u*(1.320521697 - u*inv_sqrt_pi))))))
           / (32066.59372 - u*(24322.84021 - u*(9022.227659 - u*(2186.181081
           - u*(364.2190727 - u*(61.57036588 - u*(1.841438936 - u)))))));
    }

    return w4.real();
}
