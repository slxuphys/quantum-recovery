"""Standalone odd sign-polynomial and QSP/QSVT phase generation.

This module combines a fast Remez construction of the unique coherent
max-min polynomial with two phase-factorization branches:

* ``symmetry="symmetric"`` uses palindromic QSP homotopy and supports every
  odd degree ``k+1``;
* ``symmetry="antisymmetric"`` uses structure-preserving Bezout spectral
  factorization and arbitrary-precision layer stripping.  Strict
  antisymmetry is available when ``k+1 == 1 (mod 4)``.

The polynomial can be selected by either the gap endpoint ``eps`` or its
optimal lower ripple/floor ``m``.  Exactly one must be supplied.

For a general positive interval ``0 < a < b <= 1``, use
``generate_interval_polynomial(k, a, b)`` and
``generate_interval_angles(k, a, b)``.  These solve the bounded odd
interval-maximin problem by adaptive linear-program exchange and symmetric
QSP factorization.

For the modulus-squared objective ``max min_[a,1] |P(x)|^2``, use
``generate_probability_polynomial(k, a)`` and
``generate_probability_angles(k, a)``. The latter is the analytic FPAA
sequence with ``gamma=a`` and is verified through ``qsvt_pq``.

For positive samples ``lambdas`` no larger than one, use
``generate_weighted_polynomial(k, lambdas)`` and
``generate_weighted_angles(k, lambdas)`` to maximize
``abs(sum(lambda_i**2 * P(lambda_i)))`` under the same global bound.

The antisymmetric implementation computes a strict phase sequence for the
upper-first odd equal-ripple family.  The v2 implementation adds a
Chebyshev-endpoint predictor, secant continuation, and a structure-seeded
simultaneous Aberth spectral-factor solver.

The family is

    1, m, 1, m, ..., 1

on the positive half interval.  It supports

    N = 4*r + 1,  N >= 5,  -1 <= m <= 1.

Compared with a direct monomial-coefficient implementation, it uses three
structure-preserving ideas:

1. Critical-point residuals and their analytic Jacobian are evaluated by a cached, vectorized Gauss-Legendre rule. Since P'(x) has degree N-1, the quadrature is exact in exact arithmetic.
2. The halfband/Bézout representation

       (1 + P(x))/2 = A(x) R(x),
       A(x)R(x) + A(-x)R(-x) = 1,

   is reconstructed in arbitrary precision.  Therefore

       1 - P(x)^2 = 4 A(x)A(-x)R(x)R(-x),

   and the fragile subtraction/expansion of 1-P^2 is avoided.
3. The complementary polynomial is built from a complex factor L of R,
   followed by arbitrary-precision QSP layer stripping.  No global nonlinear
   phase optimization is required.

QSP convention
--------------

    W(x) = [[x, i sqrt(1-x^2)],
            [i sqrt(1-x^2), x]],

    U_phi(x) = exp(i phi[0] Z)
               prod_{j=1}^N [W(x) exp(i phi[j] Z)].

The returned phase vector obeys

    phi[j] = -phi[N-j],
    phi[0] = phi[N] = 0,

and implements the real upper-left polynomial P_{N,m}.

Dependencies: NumPy, SciPy, and mpmath.  Matplotlib is optional for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import argparse
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial import chebyshev as npcheb
from scipy.optimize import brentq, least_squares, linprog

ArrayLike = Iterable[float] | np.ndarray


@dataclass(frozen=True)
class StructuredAntisymmetricQSPResult:
    N: int
    m: float
    phases: np.ndarray
    independent_phases: np.ndarray
    critical_points: np.ndarray
    upper_extrema: np.ndarray
    spectral_signs: tuple[int, ...]
    selected_roots_of_R: np.ndarray
    mp_dps: int
    ripple_error: float
    qsp_function_error: float
    qsp_imaginary_error: float
    antisymmetry_error: float
    layer_stripping_tail: float
    bezout_error: float
    unitarity_error: float
    target_power_coefficients: np.ndarray
    target_chebyshev_coefficients: np.ndarray

    def plot(
        self,
        *,
        independent: bool = False,
        degrees: bool = False,
        ax=None,
        show: bool = True,
        save: str | Path | None = None,
    ):
        return plot_angles(
            self,
            independent=independent,
            degrees=degrees,
            ax=ax,
            show=show,
            save=save,
        )


# ---------------------------------------------------------------------------
# Basic validation and ordered critical points
# ---------------------------------------------------------------------------


def _validate(N: int, m: float) -> tuple[int, int, float]:
    if not isinstance(N, (int, np.integer)):
        raise TypeError("N must be an integer.")
    if N < 5 or N % 4 != 1:
        raise ValueError("N must satisfy N = 4*r+1 >= 5.")
    m = float(m)
    if not -1.0 <= m <= 1.0:
        raise ValueError("m must lie in [-1,1].")
    n = (N - 1) // 2
    r = (N - 1) // 4
    return n, r, m


def _chebyshev_critical_points(N: int) -> np.ndarray:
    n = (N - 1) // 2
    return np.array(
        [np.cos((n - j) * np.pi / N) for j in range(n)], dtype=float
    )


@lru_cache(maxsize=None)
def _chebyshev_critical_first_variation(N: int) -> np.ndarray:
    """Return dc_j/dm at the Chebyshev endpoint m=-1.

    If P_{N,m}=T_N+(m+1)Q_N+O((m+1)^2), then a critical root c_j
    satisfies dc_j/dm=-Q_N'(c_j)/T_N''(c_j).  The closed Chebyshev
    expansion of Q_N makes this predictor inexpensive and removes the
    difficult first continuation solve at large N.
    """
    c = _chebyshev_critical_points(N)
    q = np.zeros(N + 1, dtype=float)
    for k in range(1, N, 2):
        q[k] = (-1) ** ((k - 1) // 2) / (
            N * np.sin(k * np.pi / (2 * N))
        )
    # N=4r+1 in this module, so sigma_N=+1.
    q[N] = -(N - 1) / (2 * N)

    t = np.zeros(N + 1, dtype=float)
    t[N] = 1.0
    q_prime = npcheb.chebval(c, npcheb.chebder(q))
    t_second = npcheb.chebval(c, npcheb.chebder(t, 2))
    return -q_prime / t_second


def _points_to_logits(c: np.ndarray) -> np.ndarray:
    gaps = np.diff(np.r_[0.0, c, 1.0])
    if np.any(gaps <= 0.0):
        raise ValueError("Critical points must be strictly ordered in (0,1).")
    return np.log(gaps[:-1] / gaps[-1])


def _logits_to_points(z: np.ndarray) -> np.ndarray:
    z_aug = np.r_[np.asarray(z, dtype=float), 0.0]
    z_aug -= np.max(z_aug)
    gaps = np.exp(z_aug)
    gaps /= np.sum(gaps)
    return np.cumsum(gaps[:-1])


# ---------------------------------------------------------------------------
# Exact-degree quadrature representation of the primitive
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _quadrature_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a cached long-double Gauss-Legendre rule."""
    nodes, weights = leggauss(order)
    return (
        np.asarray(nodes, dtype=np.longdouble),
        np.asarray(weights, dtype=np.longdouble),
    )


def _primitive_values(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorized evaluation of integral_0^x prod_j(t^2-c_j^2) dt.

    An (n+1)-point Gauss-Legendre rule is exact for the degree-2n integrand.
    The quadrature rule is cached because it depends only on n, not on the
    current critical-point iterate.
    """
    c_ld = np.asarray(c, dtype=np.longdouble)
    x_ld = np.atleast_1d(np.asarray(x, dtype=np.longdouble))
    nodes, weights = _quadrature_rule(len(c_ld) + 1)

    t = (x_ld[:, None] / 2) * (nodes[None, :] + 1)
    factors = t[:, :, None] * t[:, :, None] - (c_ld * c_ld)[None, None, :]
    integrand = np.prod(factors, axis=2)
    return (x_ld / 2) * np.sum(integrand * weights[None, :], axis=1)


def _primitive_values_and_derivatives(
    c: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return primitive values and derivatives with respect to every c_j.

    For f(t,c)=prod_k(t^2-c_k^2),

        d f / d c_j = -2 c_j prod_{k != j}(t^2-c_k^2).

    The Leibniz endpoint term vanishes when x=c_i because f(c_i,c)=0.
    Prefix/suffix products avoid division by a possibly small factor.
    """
    c_ld = np.asarray(c, dtype=np.longdouble)
    x_ld = np.atleast_1d(np.asarray(x, dtype=np.longdouble))
    nodes, weights = _quadrature_rule(len(c_ld) + 1)

    t = (x_ld[:, None] / 2) * (nodes[None, :] + 1)
    factors = t[:, :, None] * t[:, :, None] - (c_ld * c_ld)[None, None, :]
    integrand = np.prod(factors, axis=2)
    values = (x_ld / 2) * np.sum(integrand * weights[None, :], axis=1)

    number_of_points, quadrature_order, n = factors.shape
    left = np.ones_like(factors)
    right = np.ones_like(factors)
    if n > 1:
        left[:, :, 1:] = np.cumprod(factors[:, :, :-1], axis=2)
        right[:, :, :-1] = np.cumprod(
            factors[:, :, 1:][:, :, ::-1], axis=2
        )[:, :, ::-1]
    product_except = left * right

    derivative_integrals = (
        (x_ld[:, None] / 2)
        * np.sum(weights[None, :, None] * product_except, axis=1)
        * (-2 * c_ld[None, :])
    )
    return values, derivative_integrals


def _logits_to_points_and_jacobian(
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered points and dc/dz for the softmax-gap map."""
    z_aug = np.r_[np.asarray(z, dtype=np.longdouble), np.longdouble(0)]
    z_aug -= np.max(z_aug)
    gaps = np.exp(z_aug)
    gaps /= np.sum(gaps)
    c = np.cumsum(gaps[:-1])

    n = len(c)
    indicator = np.tri(n, n, dtype=np.longdouble)
    jacobian = (indicator - c[:, None]) * gaps[:-1][None, :]
    return c, jacobian


def _critical_residual_and_jacobian(
    z: np.ndarray,
    N: int,
    m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the critical-point equations and their analytic Jacobian."""
    del N  # N is implied by len(z); retained for a stable internal API.
    c, dc_dz = _logits_to_points_and_jacobian(z)
    primitive, dprimitive_dc = _primitive_values_and_derivatives(
        c, np.r_[c, np.longdouble(1)]
    )
    scale = primitive[0]
    n = len(c)

    if scale == 0 or not np.isfinite(scale):
        return (
            np.full(n, 1e100, dtype=float),
            np.zeros((n, n), dtype=float),
        )

    endpoint_target = 1.0 if n % 2 == 0 else m
    target = np.array(
        [1.0 if j % 2 == 0 else m for j in range(n)] + [endpoint_target],
        dtype=np.longdouble,
    )
    selected = np.r_[np.arange(1, n), n]
    ratios = primitive / scale
    residual = ratios[selected] - target[selected]

    selected_primitive = primitive[selected, None]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        jacobian_c = (
            dprimitive_dc[selected] * scale
            - selected_primitive * dprimitive_dc[0][None, :]
        ) / (scale * scale)
        jacobian_z = jacobian_c @ dc_dz

    residual = np.nan_to_num(
        residual,
        nan=np.longdouble(1e100),
        posinf=np.longdouble(1e100),
        neginf=np.longdouble(-1e100),
    )
    jacobian_z = np.nan_to_num(
        jacobian_z, nan=0.0, posinf=0.0, neginf=0.0
    )
    residual = np.clip(residual, -np.longdouble(1e100), np.longdouble(1e100))
    jacobian_z = np.clip(
        jacobian_z, -np.longdouble(1e100), np.longdouble(1e100)
    )
    return np.asarray(residual, dtype=float), np.asarray(jacobian_z, dtype=float)


class _CriticalEvaluator:
    """Cache residual and Jacobian when SciPy requests both at the same z."""

    def __init__(self, N: int, m: float):
        self.N = N
        self.m = m
        self._z: np.ndarray | None = None
        self._residual: np.ndarray | None = None
        self._jacobian: np.ndarray | None = None

    def _evaluate(self, z: np.ndarray) -> None:
        z_array = np.asarray(z, dtype=float)
        if self._z is None or not np.array_equal(z_array, self._z):
            self._z = z_array.copy()
            self._residual, self._jacobian = _critical_residual_and_jacobian(
                z_array, self.N, self.m
            )

    def residual(self, z: np.ndarray) -> np.ndarray:
        self._evaluate(z)
        assert self._residual is not None
        return self._residual

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        self._evaluate(z)
        assert self._jacobian is not None
        return self._jacobian


def _critical_points_adaptive(
    N: int,
    m: float,
    *,
    tolerance: float,
    initial_step: float,
    min_step: float,
    max_nfev: int,
    continuation_growth: float = 2.2,
) -> np.ndarray:
    """Continuation with an endpoint perturbation and secant predictor."""
    if m == -1.0:
        return _chebyshev_critical_points(N)
    if m == 1.0:
        return np.ones((N - 1) // 2, dtype=float)

    c = _chebyshev_critical_points(N)
    first_variation = _chebyshev_critical_first_variation(N)
    current = -1.0

    # The first nondegenerate solve should shrink like 1/N.  This avoids the
    # severely ill-conditioned jump -1 -> -0.95 at N~100.
    effective_initial_step = min(float(initial_step), 0.5 / N)
    step = min(
        effective_initial_step,
        max((m + 1.0) / 10.0, min_step * 8),
    )

    z_current: np.ndarray | None = None
    z_previous: np.ndarray | None = None
    m_previous: float | None = None
    first = True

    while current < m - 8 * np.finfo(float).eps:
        trial = min(m, current + step)

        if first:
            # First-order perturbation is dramatically more accurate than a
            # uniform displacement of all critical points.
            guess_points = c + (trial + 1.0) * first_variation
            predictor = _points_to_logits(guess_points)
            first = False
        elif (
            z_previous is not None
            and m_previous is not None
            and current > m_previous
        ):
            alpha = (trial - current) / (current - m_previous)
            predictor = z_current + alpha * (z_current - z_previous)
        else:
            assert z_current is not None
            predictor = z_current

        guesses = [predictor]
        if z_current is not None and not np.array_equal(predictor, z_current):
            guesses.append(z_current)

        accepted = None
        error = float("inf")
        for guess in guesses:
            evaluator = _CriticalEvaluator(N, trial)
            solution = least_squares(
                evaluator.residual,
                guess,
                jac=evaluator.jacobian,
                method="trf",
                xtol=tolerance,
                ftol=tolerance,
                gtol=tolerance,
                x_scale="jac",
                max_nfev=max_nfev,
            )
            error = float(np.linalg.norm(solution.fun, ord=np.inf))
            if solution.success and error <= 2e-8:
                accepted = solution
                break

        if accepted is not None:
            z_previous = None if z_current is None else z_current.copy()
            m_previous = None if z_current is None else current
            z_current = accepted.x
            c = _logits_to_points(z_current)
            current = trial
            step = min(0.60, step * continuation_growth)
        else:
            step *= 0.5
            if step < min_step:
                raise RuntimeError(
                    f"Critical-point continuation failed near m={trial:.16g}; "
                    f"residual={error:.3e}. Increase precision or reduce min_step."
                )
    return c


# ---------------------------------------------------------------------------
# Arbitrary-precision polynomial helpers, ascending power order
# ---------------------------------------------------------------------------


def _padd(a, b):
    n = max(len(a), len(b))
    out = [mp.mpc(0) for _ in range(n)]
    for i, value in enumerate(a):
        out[i] += value
    for i, value in enumerate(b):
        out[i] += value
    return out


def _pscale(a, scalar):
    return [scalar * value for value in a]


def _pmul(a, b):
    out = [mp.mpc(0) for _ in range(len(a) + len(b) - 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _peval(a, x):
    value = mp.mpc(0)
    for coefficient in reversed(a):
        value = value * x + coefficient
    return value


def _pderiv(a):
    return [(i + 1) * a[i + 1] for i in range(len(a) - 1)]


def _ppad(a, length):
    return list(a) + [mp.mpc(0)] * (length - len(a))


def _pmul_x(a):
    return [mp.mpc(0)] + list(a)


def _pmul_one_minus_x2(a):
    out = [mp.mpc(0)] * (len(a) + 2)
    for i, value in enumerate(a):
        out[i] += value
        out[i + 2] -= value
    return out


def _chebyshev_rows(x, degree: int):
    values = [mp.mpf(0)] * (degree + 1)
    derivatives = [mp.mpf(0)] * (degree + 1)
    values[0] = 1
    if degree >= 1:
        values[1] = x
        derivatives[1] = 1
    for k in range(1, degree):
        values[k + 1] = 2 * x * values[k] - values[k - 1]
        derivatives[k + 1] = (
            2 * values[k] + 2 * x * derivatives[k] - derivatives[k - 1]
        )
    return values, derivatives


def _chebyshev_to_power(coefficients):
    degree = len(coefficients) - 1
    t0 = [mp.mpf(1)]
    result = _pscale(t0, coefficients[0])
    if degree == 0:
        return result

    t1 = [mp.mpf(0), mp.mpf(1)]
    result = _padd(result, _pscale(t1, coefficients[1]))
    previous, current = t0, t1

    for k in range(1, degree):
        twice_x_current = [mp.mpf(0)] + [2 * value for value in current]
        previous_padded = previous + [mp.mpf(0)] * (
            len(twice_x_current) - len(previous)
        )
        following = [
            twice_x_current[i] - previous_padded[i]
            for i in range(len(twice_x_current))
        ]
        result = _padd(result, _pscale(following, coefficients[k + 1]))
        previous, current = current, following
    return result


def _power_to_chebyshev_via_samples(P) -> np.ndarray:
    """Convert an mp power polynomial to stable float Chebyshev coefficients."""
    degree = len(P) - 1
    count = degree + 1
    theta = [mp.pi * (j + mp.mpf("0.5")) / count for j in range(count)]
    values = [_peval(P, mp.cos(value)) for value in theta]
    coefficients = []
    for order in range(count):
        scale = mp.mpf(1) / count if order == 0 else mp.mpf(2) / count
        coefficient = scale * mp.fsum(
            values[j] * mp.cos(order * theta[j]) for j in range(count)
        )
        coefficients.append(float(mp.re(coefficient)))
    result = np.asarray(coefficients, dtype=float)
    result[0::2] = 0.0
    return result


# ---------------------------------------------------------------------------
# Structured halfband/Bézout reconstruction
# ---------------------------------------------------------------------------


def _m1_structured_polynomials(N: int):
    n = (N - 1) // 2
    multiplicity = n + 1

    A = [mp.mpf(1)]
    for _ in range(multiplicity):
        A = _pmul(A, [mp.mpf(1), mp.mpf(1)])

    # R(x) = 2^{-(n+1)} sum_{k=0}^n binom(n+k,k) ((1-x)/2)^k.
    u = [mp.mpf("0.5"), mp.mpf("-0.5")]
    power = [mp.mpf(1)]
    D = [mp.mpf(0)]
    for k in range(n + 1):
        D = _padd(D, _pscale(power, mp.binomial(n + k, k)))
        power = _pmul(power, u)
    R = _pscale(D, mp.power(2, -multiplicity))

    P = _pscale(_pmul(A, R), 2)
    P[0] -= 1
    P = [mp.re(value) for value in P]
    for i in range(0, len(P), 2):
        P[i] = mp.mpf(0)
    R = [mp.re(value) for value in R]
    return A, R, P


def _structured_polynomials(upper_extrema: np.ndarray, N: int, m: float):
    """Construct A, R, and P using confluent Newton interpolation.

    R is the Hermite interpolant of 1/A at x=1 and at every doubled upper
    extremum.  A confluent divided-difference table costs O(r^2), avoids the
    generic high-precision LU solve, and is substantially faster at N~100.
    """
    if m == 1.0:
        return _m1_structured_polynomials(N)

    upper = [mp.mpf(repr(float(value))) for value in upper_extrema]

    # For N=4r+1, P(1)=1 and A includes the simple endpoint factor 1+x.
    # For N=4r+3, P(1)=m and every zero represented in A is a doubled
    # negative upper extremum.
    A = [mp.mpf(1), mp.mpf(1)] if N % 4 == 1 else [mp.mpf(1)]
    for a in upper:
        A = _pmul(A, [a * a, 2 * a, mp.mpf(1)])
    derivative_A = _pderiv(A)

    if N % 4 == 1:
        nodes = [mp.mpf(1)]
        values = [1 / _peval(A, mp.mpf(1))]
        repeated_derivatives: list[mp.mpf | None] = [None]
    else:
        nodes = []
        values = []
        repeated_derivatives = []

    for a in upper:
        Aval = _peval(A, a)
        Aprime = _peval(derivative_A, a)
        value = 1 / Aval
        derivative_value = -Aprime / (Aval * Aval)
        nodes.extend((a, a))
        values.extend((value, value))
        # The derivative belongs to the second copy in the confluent table.
        repeated_derivatives.extend((None, derivative_value))

    size = len(nodes)
    divided = [
        [mp.mpf(0) for _ in range(size)]
        for _ in range(size)
    ]
    for i, value in enumerate(values):
        divided[i][0] = value

    for order in range(1, size):
        for i in range(order, size):
            if nodes[i] == nodes[i - order]:
                if order != 1 or repeated_derivatives[i] is None:
                    raise RuntimeError("Unexpected higher confluent multiplicity.")
                divided[i][order] = repeated_derivatives[i]
            else:
                divided[i][order] = (
                    divided[i][order - 1] - divided[i - 1][order - 1]
                ) / (nodes[i] - nodes[i - order])

    newton_coefficients = [divided[j][j] for j in range(size)]
    R = [newton_coefficients[0]]
    basis = [mp.mpf(1)]
    for j in range(1, size):
        basis = _pmul(basis, [-nodes[j - 1], mp.mpf(1)])
        R = _padd(R, _pscale(basis, newton_coefficients[j]))
    R = [mp.re(value) for value in R]

    P = _pscale(_pmul(A, R), 2)
    P[0] -= 1
    P = [mp.re(value) for value in P]
    for i in range(0, len(P), 2):
        P[i] = mp.mpf(0)
    return A, R, P


def _bezout_error(A, R) -> mp.mpf:
    Aminus = [value * ((-1) ** i) for i, value in enumerate(A)]
    Rminus = [value * ((-1) ** i) for i, value in enumerate(R)]
    lhs = _padd(_pmul(A, R), _pmul(Aminus, Rminus))
    lhs[0] -= 1
    return max(abs(value) for value in lhs)


# ---------------------------------------------------------------------------
# Factor R, form Q directly, and strip phases in arbitrary precision
# ---------------------------------------------------------------------------


def _legacy_refined_roots(R, *, iterations: int = 120):
    """Find and validate all roots of R using two complementary seed bases.

    Power-basis companion roots become unreliable when roots approach the real
    axis.  Chebyshev-colleague roots fail in different locations, so combining
    both seed sets and refining in arbitrary precision is substantially more
    robust.  Unconverged Newton iterates are rejected rather than silently used.
    """
    double_coefficients = np.array([float(mp.re(value)) for value in R])
    power_seeds = np.roots(double_coefficients[::-1])
    chebyshev_coefficients = npcheb.poly2cheb(double_coefficients)
    chebyshev_seeds = npcheb.chebroots(chebyshev_coefficients)
    seeds = list(power_seeds) + list(chebyshev_seeds)

    derivative = _pderiv(R)
    degree = len(R) - 1
    stopping = mp.power(10, -max(30, mp.mp.dps - 20))
    residual_tolerance = mp.power(10, -max(30, min(100, mp.mp.dps // 2)))
    merge_tolerance = mp.power(10, -max(20, min(60, mp.mp.dps // 3)))

    roots = []

    def refine(seed):
        z = mp.mpc(float(np.real(seed)), float(np.imag(seed)))
        for _ in range(iterations):
            f = _peval(R, z)
            fp = _peval(derivative, z)
            if abs(fp) == 0:
                break
            correction = f / fp
            z -= correction
            if abs(correction) < stopping:
                break
        return z, abs(_peval(R, z))

    def add_if_new(z, residual):
        if residual > residual_tolerance:
            return
        if any(abs(z - existing) < merge_tolerance for existing in roots):
            return
        roots.append(z)

    for seed in seeds:
        z, residual = refine(seed)
        add_if_new(z, residual)

    # Near-real conjugate pairs can be missed by both double-precision seed
    # generators.  Perturb the available seeds vertically before giving up.
    if len(roots) < degree:
        augmented = []
        for seed in seeds:
            scale = max(1.0, abs(complex(seed)))
            for relative_imaginary_shift in (1e-4, 1e-3, 1e-2, 5e-2):
                shift = 1j * relative_imaginary_shift * scale
                augmented.extend((seed + shift, seed - shift))
        for seed in augmented:
            z, residual = refine(seed)
            add_if_new(z, residual)
            if len(roots) == degree:
                break

    if len(roots) != degree:
        raise RuntimeError(
            f"Root refinement found {len(roots)} validated roots of a "
            f"degree-{degree} Bézout polynomial. Increase mp_dps or choose "
            "a different continuation/branch."
        )

    roots.sort(key=lambda z: (float(mp.re(z)), float(mp.im(z))))
    return roots


def _factor_relative_error(R, upper_roots) -> mp.mpf:
    """Relative coefficient error of R-L*conjugate(L)."""
    leading = mp.re(R[-1])
    L = [mp.sqrt(leading)]
    for root in upper_roots:
        L = _pmul(L, [-root, mp.mpf(1)])
    reconstructed = _pmul(L, [mp.conj(value) for value in L])
    target = _ppad(R, len(reconstructed))
    scale = max(mp.mpf(1), max(abs(value) for value in R))
    return max(
        abs(reconstructed[i] - target[i]) for i in range(len(reconstructed))
    ) / scale


def _structured_upper_root_seeds(
    critical_points: np.ndarray,
    m: float,
) -> list[complex]:
    """One upper-half-plane seed per lower ripple interval.

    At m=-1 every conjugate pair collapses onto a lower critical point.  For
    finite m, the normalized imaginary splitting is well approximated by

        sqrt((1+m)/2) * sqrt((b-a_left)(a_right-b)).

    These seeds preserve the ordering of the spectral pairs and are much more
    reliable than global companion roots near the real axis.
    """
    c = np.asarray(critical_points, dtype=float)
    upper = c[::2]
    lower = c[1::2]
    height = math.sqrt(max(0.0, (1.0 + m) / 2.0))
    seeds: list[complex] = []
    for j, b in enumerate(lower):
        left = upper[j]
        right = upper[j + 1] if j + 1 < len(upper) else 1.0
        width = math.sqrt(max(0.0, (b - left) * (right - b)))
        imaginary = max(1e-16, height * width)
        seeds.append(complex(float(b), imaginary))
    return seeds


def _seeds_from_double_roots(R, count: int) -> list[complex]:
    """Upper-half-plane seeds for the exact m=1 polynomial.

    The m=1 roots are not localized near distinct critical intervals, so use
    the ordinary companion roots.  If a very close conjugate pair collapses
    to two real roots in double precision, reconstruct a complex seed from
    their midpoint and half-separation.
    """
    coefficients = np.array([float(mp.re(value)) for value in R], dtype=float)
    coefficients /= np.max(np.abs(coefficients))

    def extract(roots: np.ndarray) -> list[complex]:
        scale = max(1.0, float(np.max(np.abs(roots))))
        tolerance = 1e-10 * scale
        positive = [complex(z) for z in roots if np.imag(z) > tolerance]
        real = sorted(float(np.real(z)) for z in roots if abs(np.imag(z)) <= tolerance)
        for j in range(0, len(real) - 1, 2):
            center = 0.5 * (real[j] + real[j + 1])
            imaginary = max(0.5 * abs(real[j + 1] - real[j]), 1e-8 * (1 + abs(center)))
            positive.append(complex(center, imaginary))
        positive.sort(key=lambda z: z.real)
        return positive

    candidates = extract(np.roots(coefficients[::-1]))
    if len(candidates) != count:
        cheb = npcheb.poly2cheb(coefficients)
        alternative = extract(npcheb.chebroots(cheb))
        if len(alternative) == count:
            candidates = alternative
    if len(candidates) != count:
        raise RuntimeError(
            f"Could not form {count} upper-half-plane seeds from the "
            f"degree-{len(R)-1} m=1 Bézout polynomial."
        )
    return candidates


def _aberth_upper_roots(
    R,
    seeds: Sequence[complex],
    *,
    factor_digits: int,
    max_iterations: int,
):
    """Simultaneously refine one root from every conjugate pair.

    The Aberth correction includes interactions with both the represented
    upper roots and their implicit conjugates.  This prevents two Newton
    starts from converging to the same root, which was the dominant failure
    and runtime source in the older independent-refinement implementation.
    """
    derivative = _pderiv(R)
    roots = [mp.mpc(float(z.real), abs(float(z.imag))) for z in seeds]
    target = mp.power(10, -factor_digits)

    factor_error = mp.inf
    for iteration in range(max_iterations):
        # Coefficient reconstruction is more expensive than one Aberth update.
        # Checking every second iteration preserves robustness while avoiding
        # dozens of redundant O(r^2) polynomial products at m=1.
        if iteration >= 4 and iteration % 2 == 0:
            factor_error = _factor_relative_error(R, roots)
            if factor_error < target:
                roots.sort(key=lambda z: (float(mp.re(z)), float(mp.im(z))))
                return roots, factor_error

        updated: list[mp.mpc] = []
        for i, z in enumerate(roots):
            value = _peval(R, z)
            derivative_value = _peval(derivative, z)
            if derivative_value == 0:
                raise RuntimeError("Aberth refinement encountered a zero derivative.")
            newton = value / derivative_value

            interaction = mp.mpc(0)
            for j, other in enumerate(roots):
                if j != i:
                    interaction += 1 / (z - other)
                # Include every lower-half-plane conjugate, including z's own.
                interaction += 1 / (z - mp.conj(other))

            denominator = 1 - newton * interaction
            correction = newton if denominator == 0 else newton / denominator
            candidate = z - correction

            # Preserve the selected upper branch.  Damping is preferable to
            # allowing a root to jump through the real axis.
            damping = mp.mpf(1)
            while mp.im(candidate) <= 0 and damping > mp.mpf("1e-8"):
                damping /= 2
                candidate = z - damping * correction
            if mp.im(candidate) <= 0:
                candidate = mp.conj(candidate)
            updated.append(candidate)
        roots = updated

    factor_error = _factor_relative_error(R, roots)
    if factor_error >= target:
        raise RuntimeError(
            "Structure-seeded Aberth refinement did not reach the requested "
            f"factor accuracy: {mp.nstr(factor_error, 6)} >= 1e-{factor_digits}."
        )
    roots.sort(key=lambda z: (float(mp.re(z)), float(mp.im(z))))
    return roots, factor_error


def _pair_conjugate_roots(roots):
    unused = list(range(len(roots)))
    pairs = []
    while unused:
        i = unused.pop(0)
        z = roots[i]
        if not unused:
            raise RuntimeError("Unpaired root in the Bézout polynomial R.")
        j = min(unused, key=lambda index: abs(roots[index] - mp.conj(z)))
        unused.remove(j)
        pairs.append((z, roots[j]))
    pairs.sort(key=lambda pair: float(mp.re((pair[0] + pair[1]) / 2)))
    return pairs


def _companion_from_R(
    critical_points: np.ndarray,
    upper_extrema: np.ndarray,
    R,
    P,
    spectral_signs: Sequence[int],
    m: float,
    *,
    root_factor_digits: int,
    root_max_iterations: int,
):
    pair_count = len(spectral_signs)
    if m == 1.0:
        seeds = _seeds_from_double_roots(R, pair_count)
    else:
        seeds = _structured_upper_root_seeds(critical_points, m)

    positive_roots, _ = _aberth_upper_roots(
        R,
        seeds,
        factor_digits=root_factor_digits,
        max_iterations=root_max_iterations,
    )
    if len(positive_roots) != pair_count:
        raise RuntimeError(
            f"Expected {pair_count} spectral root pairs, found {len(positive_roots)}."
        )

    selected = []
    for root, sign in zip(positive_roots, spectral_signs):
        if sign not in (-1, 1):
            raise ValueError("Every spectral sign must be +1 or -1.")
        selected.append(root if sign > 0 else mp.conj(root))

    leading = mp.re(R[-1])
    if leading <= 0:
        raise RuntimeError("R must have positive leading coefficient on this branch.")

    # R(x)=L(x) conjugate(L(x)).
    L = [mp.sqrt(leading)]
    for root in selected:
        L = _pmul(L, [-root, mp.mpf(1)])

    # H(x^2)=2 L(x)L(-x), so H*conjugate(H)=4R(x)R(-x).
    Lminus = [value * ((-1) ** i) for i, value in enumerate(L)]
    even_product = _pmul(L, Lminus)
    H_y = [2 * even_product[2 * i] for i in range((len(even_product) + 1) // 2)]

    upper_factor_y = [mp.mpf(1)]
    for value in upper_extrema:
        a = mp.mpf(repr(float(value)))
        upper_factor_y = _pmul(upper_factor_y, [-a * a, mp.mpf(1)])

    q_y = _pmul(upper_factor_y, H_y)
    Q = [mp.mpc(0) for _ in range(2 * (len(q_y) - 1) + 1)]
    for i, value in enumerate(q_y):
        Q[2 * i] = value

    # Twist Q so the final stripped endpoint phase is zero.
    phase = mp.arg(P[-1]) - mp.arg(Q[-1])
    twist = mp.exp(1j * phase)
    Q = [twist * value for value in Q]
    return Q, selected


def _unitarity_error(P, Q) -> mp.mpf:
    q_conjugate = [mp.conj(value) for value in Q]
    q_abs_squared = _pmul(Q, q_conjugate)
    weighted = _pmul([mp.mpf(1), mp.mpf(0), mp.mpf(-1)], q_abs_squared)
    lhs = _padd(_pmul(P, P), weighted)
    lhs[0] -= 1
    return max(abs(value) for value in lhs)


def _strip_phases(P, Q):
    P = [mp.mpc(value) for value in P]
    Q = [mp.mpc(value) for value in Q]
    N = len(P) - 1
    phases = [mp.mpf(0)] * (N + 1)
    largest_tail = mp.mpf(0)

    for degree in range(N, 0, -1):
        if abs(Q[degree - 1]) == 0:
            raise RuntimeError(
                f"Layer stripping found a zero leading Q coefficient at degree {degree}."
            )
        phi = mp.arg(P[degree] / Q[degree - 1]) / 2
        phases[degree] = phi
        phase = mp.exp(1j * phi)
        phase_conjugate = mp.exp(-1j * phi)

        term1 = _pscale(_pmul_one_minus_x2(Q), phase)
        term2 = _pscale(_pmul_x(P), phase_conjugate)
        length = max(len(term1), len(term2))
        raw_P = _padd(_ppad(term1, length), _ppad(term2, length))

        term1 = _pscale(_pmul_x(Q), phase)
        term2 = _pscale(P, phase_conjugate)
        length = max(len(term1), len(term2))
        raw_Q = _padd(_ppad(term1, length), _pscale(_ppad(term2, length), -1))

        discarded = raw_P[degree:] + raw_Q[max(degree - 1, 0):]
        if discarded:
            largest_tail = max(largest_tail, max(abs(value) for value in discarded))

        # Parity and leading-term cancellation reduce the expected lengths by two.
        P = raw_P[:degree]
        Q = raw_Q[: degree - 1] if degree > 1 else [mp.mpc(0)]

    phases[0] = mp.arg(P[0])
    return phases, largest_tail


def _enforce_antisymmetry(phases):
    values = np.array([float(value) for value in phases], dtype=float)
    N = len(values) - 1
    n = (N - 1) // 2
    left = values[1 : n + 1]
    mirrored = values[-2 : -n - 2 : -1]
    theta = 0.5 * (left - mirrored)
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return np.r_[0.0, theta, -theta[::-1], 0.0]


# ---------------------------------------------------------------------------
# Coherent max-min polynomial (Remez) and symmetric phase factorization
# ---------------------------------------------------------------------------


def _validate_k(k: int) -> int:
    if (
        not isinstance(k, (int, np.integer))
        or isinstance(k, bool)
        or k < 0
        or k % 2
    ):
        raise ValueError("k must be a nonnegative even integer.")
    return int(k)


def _real_roots_in_interval(
    polynomial: npcheb.Chebyshev,
    left: float,
    right: float,
    *,
    imag_tol: float = 1e-7,
    duplicate_tol: float = 1e-8,
) -> np.ndarray:
    roots = polynomial.roots()
    roots = np.real(roots[np.abs(np.imag(roots)) <= imag_tol])
    roots = np.sort(roots[(roots > left + 1e-12) & (roots < right - 1e-12)])
    if roots.size <= 1:
        return roots
    distinct = [roots[0]]
    for root in roots[1:]:
        if abs(root - distinct[-1]) > duplicate_tol:
            distinct.append(root)
    return np.asarray(distinct, dtype=float)


def _bracketed_derivative_roots(
    polynomial: npcheb.Chebyshev,
    left: float,
    right: float,
) -> np.ndarray:
    """Robust fallback for high-degree Chebyshev companion-root failures."""
    derivative = polynomial.deriv()
    count = max(2001, 64 * polynomial.degree())
    theta = np.linspace(np.pi, 0.0, count)
    grid = (left + right) / 2.0 + (right - left) / 2.0 * np.cos(theta)
    values = derivative(grid)
    roots: list[float] = []
    for index in range(count - 1):
        a, b = grid[index], grid[index + 1]
        fa, fb = values[index], values[index + 1]
        if fa == 0.0:
            root = float(a)
        elif fa * fb < 0.0:
            root = float(brentq(derivative, a, b, xtol=2e-14, rtol=2e-14))
        else:
            continue
        if left + 1e-12 < root < right - 1e-12:
            if not roots or abs(root - roots[-1]) > 1e-9:
                roots.append(root)
    return np.asarray(roots, dtype=float)


@lru_cache(maxsize=256)
def _coherent_data(
    k: int,
    eps: float,
) -> tuple[np.ndarray, float, float, np.ndarray, int]:
    """Chebyshev coefficients, floor, sign error, alternants, iterations."""
    k = _validate_k(k)
    eps = float(eps)
    if not np.isfinite(eps) or not 0.0 <= eps <= 1.0:
        raise ValueError("eps must lie in [0, 1].")
    degree = k + 1

    if k == 0 or eps == 0.0 or eps == 1.0:
        coefficients = np.zeros(degree + 1, dtype=float)
        coefficients[1] = 1.0
        coefficients.setflags(write=False)
        points = np.array([eps, 1.0], dtype=float)
        points.setflags(write=False)
        floor = eps
        error = (1.0 - eps) / (1.0 + eps)
        return coefficients, float(floor), float(error), points, 0

    n = k // 2
    odd_degrees = 2 * np.arange(n + 1) + 1
    index = np.arange(n + 2)
    points = np.sort(
        (1.0 + eps) / 2.0
        + (1.0 - eps) / 2.0 * np.cos(np.pi * index / (n + 1))
    )

    tolerance = 1e-12
    for iteration in range(1, 101):
        theta = np.arccos(points)
        matrix = np.empty((n + 2, n + 2), dtype=float)
        matrix[:, :-1] = np.cos(np.outer(theta, odd_degrees))
        matrix[:, -1] = (-1.0) ** np.arange(n + 2)
        solution = np.linalg.solve(matrix, np.ones(n + 2))

        sign_coefficients = np.zeros(degree + 1, dtype=float)
        sign_coefficients[odd_degrees] = solution[:-1]
        sign_polynomial = npcheb.Chebyshev(sign_coefficients)
        extrema = _real_roots_in_interval(sign_polynomial.deriv(), eps, 1.0)
        if extrema.size != n:
            extrema = _bracketed_derivative_roots(sign_polynomial, eps, 1.0)
            if extrema.size != n:
                raise RuntimeError(
                    "Remez critical-root extraction is ill-conditioned: "
                    f"found {extrema.size}, expected {n}. Reduce k/eps or use "
                    "the m-parameterized antisymmetric path."
                )
        new_points = np.r_[eps, extrema, 1.0]
        errors = 1.0 - sign_polynomial(new_points)
        spread = float(np.ptp(np.abs(errors)))
        ripple = float(np.max(np.abs(errors)))
        points = new_points
        if spread <= tolerance * max(1.0, ripple):
            break
    else:
        raise RuntimeError("Remez exchange did not converge.")

    theta = np.arccos(points)
    matrix[:, :-1] = np.cos(np.outer(theta, odd_degrees))
    matrix[:, -1] = (-1.0) ** np.arange(n + 2)
    solution = np.linalg.solve(matrix, np.ones(n + 2))
    sign_error = abs(float(solution[-1]))
    sign_coefficients = np.zeros(degree + 1, dtype=float)
    sign_coefficients[odd_degrees] = solution[:-1]
    coefficients = sign_coefficients / (1.0 + sign_error)
    coefficients[0::2] = 0.0
    floor = (1.0 - sign_error) / (1.0 + sign_error)

    polynomial = npcheb.Chebyshev(coefficients)
    critical = _real_roots_in_interval(polynomial.deriv(), -1.0, 1.0)
    if np.max(np.abs(polynomial(np.r_[-1.0, critical, 1.0]))) > 1.0 + 2e-8:
        raise RuntimeError("Computed polynomial violates |P(x)| <= 1.")

    coefficients.setflags(write=False)
    points = points.copy()
    points.setflags(write=False)
    return coefficients, float(floor), sign_error, points, iteration


@lru_cache(maxsize=256)
def _eps_from_floor(k: int, m: float) -> float:
    k = _validate_k(k)
    m = float(m)
    if not np.isfinite(m) or not 0.0 <= m <= 1.0:
        raise ValueError("m must lie in [0, 1] for a coherent sign polynomial.")
    if k == 0 or m in (0.0, 1.0):
        return m

    def residual(eps: float) -> float:
        return _coherent_data(k, float(eps))[1] - m

    # Do not bracket with eps=1: for large k that asks float64 Remez to
    # resolve a ripple far below machine precision even when the requested m
    # is moderate.  Grow a degree-scaled bracket only until it crosses m.
    lower = 0.0
    upper = min(1.0, 1.0 / max(1, 16 * k))
    upper_residual = residual(upper)
    while upper_residual < 0.0:
        lower = upper
        upper = min(1.0, 1.6 * upper)
        try:
            upper_residual = residual(upper)
        except RuntimeError:
            # A failure above a valid lower point normally means that the
            # ripple has entered float64 saturation. Treat the failed point
            # as an upper bound and approach it without evaluating farther
            # into the saturated region.
            for _ in range(60):
                midpoint = 0.5 * (lower + upper)
                try:
                    midpoint_residual = residual(midpoint)
                except RuntimeError:
                    upper = midpoint
                    continue
                if midpoint_residual < 0.0:
                    lower = midpoint
                else:
                    upper = midpoint
                if upper - lower <= 2e-13 * max(1.0, upper):
                    break
            return float(0.5 * (lower + upper))
        if upper == 1.0 and upper_residual < 0.0:
            raise RuntimeError("Could not bracket eps for the requested floor.")
    return float(brentq(residual, lower, upper, xtol=2e-13, rtol=2e-13))


def _resolve_parameter(
    k: int,
    *,
    eps: float | None,
    m: float | None,
) -> tuple[float, float]:
    k = _validate_k(k)
    if (eps is None) == (m is None):
        raise ValueError("Supply exactly one of eps or m.")
    if eps is not None:
        eps = float(eps)
        _, floor, *_ = _coherent_data(k, eps)
        return eps, floor
    assert m is not None
    m = float(m)
    resolved_eps = _eps_from_floor(k, m)
    return resolved_eps, m


@lru_cache(maxsize=64)
def _structured_chebyshev_data(
    degree: int,
    m: float,
    mp_dps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build an odd equal-ripple target without spectral factorization.

    This is the robust fallback when float64 Remez cannot extract all
    high-degree critical roots.  It also recovers eps from P(eps)=m.
    """
    if degree < 3 or degree % 2 != 1:
        raise ValueError("degree must be an odd integer at least 3.")
    if not 0.0 <= m <= 1.0:
        raise ValueError("m must lie in [0, 1] for a coherent sign polynomial.")
    if mp_dps is None:
        mp_dps = max(80, degree + 40)
    critical = _critical_points_adaptive(
        degree,
        m,
        tolerance=2e-12,
        initial_step=0.05,
        min_step=1e-8,
        max_nfev=12000,
        continuation_growth=2.2,
    )
    with mp.workdps(mp_dps):
        _, _, P = _structured_polynomials(critical[::2], degree, m)
        coefficients = _power_to_chebyshev_via_samples(P)
    polynomial = npcheb.Chebyshev(coefficients)
    eps = float(brentq(lambda value: polynomial(value) - m, 0.0, critical[0]))
    coefficients.setflags(write=False)
    critical = critical.copy()
    critical.setflags(write=False)
    return coefficients, critical, eps


@lru_cache(maxsize=64)
def _structured_floor_from_eps(degree: int, eps: float) -> float:
    """Invert the structured family when high-degree Remez is unavailable."""
    eps = float(eps)
    if not 0.0 <= eps <= 1.0:
        raise ValueError("eps must lie in [0, 1].")
    if eps in (0.0, 1.0):
        return eps

    def residual(floor: float) -> float:
        if floor == 0.0:
            family_eps = 0.0
        elif floor == 1.0:
            family_eps = 1.0
        else:
            family_eps = _structured_chebyshev_data(degree, floor)[2]
        return family_eps - eps

    return float(brentq(residual, 0.0, 1.0, xtol=2e-12, rtol=2e-12))


def generate_polynomial(
    k: int,
    *,
    eps: float | None = None,
    m: float | None = None,
) -> npcheb.Chebyshev:
    """Return the optimal odd degree-``k+1`` coherent sign polynomial.

    ``k`` is the even number of internal QSVT angles.  Select the polynomial
    with either its positive interval endpoint ``eps`` or its optimal lower
    ripple/floor ``m``.
    """
    k = _validate_k(k)
    if m is not None and k >= 64 and m not in (0.0, 1.0):
        coefficients, _, _ = _structured_chebyshev_data(k + 1, float(m))
        return npcheb.Chebyshev(coefficients.copy())
    if eps is not None and k >= 64 and eps not in (0.0, 1.0):
        floor = _structured_floor_from_eps(k + 1, float(eps))
        coefficients, _, _ = _structured_chebyshev_data(k + 1, floor)
        return npcheb.Chebyshev(coefficients.copy())
    try:
        resolved_eps, _ = _resolve_parameter(k, eps=eps, m=m)
        coefficients, *_ = _coherent_data(k, resolved_eps)
    except RuntimeError:
        if m is None or k + 1 < 3:
            raise
        coefficients, _, _ = _structured_chebyshev_data(k + 1, float(m))
    return npcheb.Chebyshev(coefficients.copy())


def _validate_interval(k: int, a: float, b: float) -> tuple[int, float, float]:
    k = _validate_k(k)
    a = float(a)
    b = float(b)
    if not np.isfinite(a) or not np.isfinite(b) or not 0.0 < a < b <= 1.0:
        raise ValueError("The interval must satisfy 0 < a < b <= 1.")
    return k, a, b


def _critical_candidates(
    polynomial: npcheb.Chebyshev,
    left: float,
    right: float,
) -> np.ndarray:
    roots = _real_roots_in_interval(polynomial.deriv(), left, right)
    # A sign-change scan catches real derivative roots occasionally perturbed
    # by the colleague eigenproblem. Taking the union is inexpensive because
    # the polynomial degree is at most k+1.
    bracketed = _bracketed_derivative_roots(polynomial, left, right)
    return np.unique(np.r_[left, roots, bracketed, right])


def _linprog_with_retry(objective, constraints, upper_bounds, variable_bounds):
    """Solve a dense LP, retrying HiGHS IPM after an automatic-method failure."""
    options = {
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
    }
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=upper_bounds,
        bounds=variable_bounds,
        method="highs",
        options=options,
    )
    if not result.success:
        result = linprog(
            objective,
            A_ub=constraints,
            b_ub=upper_bounds,
            bounds=variable_bounds,
            method="highs-ipm",
            options=options,
        )
    return result


@lru_cache(maxsize=128)
def _interval_maxmin_data(
    k: int,
    a: float,
    b: float,
    grid_size: int,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, float, float, float, int, np.ndarray, np.ndarray]:
    """Solve the bounded odd interval-maximin problem by LP exchange."""
    k, a, b = _validate_interval(k, a, b)
    if grid_size < k + 3:
        raise ValueError("grid_size must be at least k+3.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    coefficient_count = k // 2 + 1
    odd_degrees = 2 * np.arange(coefficient_count) + 1

    # Mix uniform and Chebyshev-spaced points. The exchange loop subsequently
    # inserts the exact critical points of every LP solution.
    uniform = np.linspace(0.0, 1.0, grid_size)
    chebyshev = np.cos(np.linspace(np.pi / 2.0, 0.0, grid_size))
    bound_points = np.unique(np.r_[uniform, chebyshev])
    interval_points = np.unique(
        np.r_[
            np.linspace(a, b, grid_size),
            (a + b) / 2.0
            + (b - a) / 2.0 * np.cos(np.linspace(np.pi, 0.0, grid_size)),
        ]
    )

    result = None
    polynomial = None
    interval_minimum = -np.inf
    global_maximum = np.inf

    for iteration in range(1, max_iterations + 1):
        bound_rows = npcheb.chebvander(bound_points, k + 1)[:, odd_degrees]
        interval_rows = npcheb.chebvander(interval_points, k + 1)[:, odd_degrees]

        # Variables are the odd Chebyshev coefficients followed by mu.
        # p <= 1, -p <= 1 on [0,1], and -p+mu <= 0 on [a,b].
        zeros_bound = np.zeros((bound_points.size, 1))
        constraints = np.vstack(
            (
                np.hstack((bound_rows, zeros_bound)),
                np.hstack((-bound_rows, zeros_bound)),
                np.hstack(
                    (-interval_rows, np.ones((interval_points.size, 1)))
                ),
            )
        )
        upper_bounds = np.r_[
            np.ones(2 * bound_points.size), np.zeros(interval_points.size)
        ]
        objective = np.r_[np.zeros(coefficient_count), -1.0]
        variable_bounds = [(None, None)] * coefficient_count + [(-1.0, 1.0)]
        result = _linprog_with_retry(
            objective,
            constraints,
            upper_bounds,
            variable_bounds,
        )
        if not result.success:
            raise RuntimeError(f"Interval maximin LP failed: {result.message}")

        coefficients = np.zeros(k + 2, dtype=float)
        coefficients[odd_degrees] = result.x[:-1]
        polynomial = npcheb.Chebyshev(coefficients)

        bound_candidates = _critical_candidates(polynomial, 0.0, 1.0)
        interval_candidates = _critical_candidates(polynomial, a, b)
        bound_values = polynomial(bound_candidates)
        interval_values = polynomial(interval_candidates)
        global_maximum = float(np.max(np.abs(bound_values)))
        interval_minimum = float(np.min(interval_values))
        lp_floor = float(result.x[-1])

        bound_points = np.unique(np.r_[bound_points, bound_candidates])
        interval_points = np.unique(np.r_[interval_points, interval_candidates])
        if (
            global_maximum <= 1.0 + tolerance
            and lp_floor - interval_minimum <= tolerance
        ):
            break
    else:
        raise RuntimeError(
            "Interval maximin exchange did not converge; increase "
            "max_iterations or grid_size."
        )

    assert polynomial is not None and result is not None
    # Normalize the active global bound. This removes a tiny feasibility
    # excess or slack permitted by the LP solver and improves the floor while
    # keeping the returned polynomial bounded to working precision.
    if global_maximum <= 0.0:
        raise RuntimeError("Interval maximin solver returned a zero polynomial.")
    scale = global_maximum
    coefficients = polynomial.coef / scale
    polynomial = npcheb.Chebyshev(coefficients)
    bound_candidates = _critical_candidates(polynomial, 0.0, 1.0)
    interval_candidates = _critical_candidates(polynomial, a, b)
    global_maximum = float(np.max(np.abs(polynomial(bound_candidates))))
    interval_minimum = float(np.min(polynomial(interval_candidates)))

    coefficients = np.asarray(coefficients, dtype=float)
    coefficients.setflags(write=False)
    bound_candidates.setflags(write=False)
    interval_candidates.setflags(write=False)
    return (
        coefficients,
        interval_minimum,
        global_maximum,
        float(result.x[-1]),
        iteration,
        bound_candidates,
        interval_candidates,
    )


def generate_interval_polynomial(
    k: int,
    a: float,
    b: float,
    *,
    grid_size: int | None = None,
    max_iterations: int = 30,
    tolerance: float = 2e-8,
    return_info: bool = False,
):
    """Maximize ``min p(x)`` on ``[a,b]`` for a globally bounded odd p.

    The polynomial has degree at most ``k+1``, odd parity, and satisfies
    ``|p(x)| <= 1`` on ``[-1,1]``. The semi-infinite linear program is solved
    by adaptive exchange at the polynomial's critical points.
    """
    k, a, b = _validate_interval(k, a, b)
    if grid_size is None:
        grid_size = max(65, 8 * (k + 2) + 1)
    data = _interval_maxmin_data(
        k, a, b, int(grid_size), int(max_iterations), float(tolerance)
    )
    coefficients, floor, max_abs, lp_floor, iterations, bound, interval = data
    polynomial = npcheb.Chebyshev(coefficients.copy())
    if not return_info:
        return polynomial
    info = {
        "degree": k + 1,
        "k": k,
        "a": a,
        "b": b,
        "optimal_floor": float(floor),
        "lp_floor": float(lp_floor),
        "global_max_abs": float(max_abs),
        "bound_violation": float(max(0.0, max_abs - 1.0)),
        "exchange_iterations": int(iterations),
        "bound_critical_points": bound.copy(),
        "interval_critical_points": interval.copy(),
    }
    return polynomial, info


def _validate_weighted_lambdas(lambdas: Any) -> np.ndarray:
    values = np.asarray(lambdas, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("lambdas must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError("Every lambda must satisfy 0 < lambda <= 1.")
    return np.sort(values)


@lru_cache(maxsize=128)
def _weighted_polynomial_data(
    k: int,
    lambdas: tuple[float, ...],
    grid_size: int,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, float, float, float, int, np.ndarray]:
    """Optimize sum lambda_i^2 p(lambda_i) under the global QSP bound."""
    k = _validate_k(k)
    values = _validate_weighted_lambdas(lambdas)
    if grid_size < k + 3:
        raise ValueError("grid_size must be at least k+3.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    weights = values * values
    weight_sum = float(np.sum(weights))
    normalized_weights = weights / weight_sum
    coefficient_count = k // 2 + 1
    odd_degrees = 2 * np.arange(coefficient_count) + 1
    objective_row = (
        npcheb.chebvander(values, k + 1)[:, odd_degrees].T
        @ normalized_weights
    )
    bound_points = np.unique(
        np.r_[
            np.linspace(0.0, 1.0, grid_size),
            np.cos(np.linspace(np.pi / 2.0, 0.0, grid_size)),
        ]
    )

    result = None
    polynomial = None
    global_maximum = np.inf
    for iteration in range(1, max_iterations + 1):
        rows = npcheb.chebvander(bound_points, k + 1)[:, odd_degrees]
        constraints = np.vstack((rows, -rows))
        upper_bounds = np.ones(2 * bound_points.size)
        result = _linprog_with_retry(
            -objective_row,
            constraints,
            upper_bounds,
            [(None, None)] * coefficient_count,
        )
        if not result.success:
            raise RuntimeError(f"Weighted polynomial LP failed: {result.message}")

        coefficients = np.zeros(k + 2, dtype=float)
        coefficients[odd_degrees] = result.x
        polynomial = npcheb.Chebyshev(coefficients)
        critical_points = _critical_candidates(polynomial, 0.0, 1.0)
        critical_values = polynomial(critical_points)
        global_maximum = float(np.max(np.abs(critical_values)))
        violations = critical_points[
            np.abs(critical_values) > 1.0 + tolerance
        ]
        if violations.size == 0:
            break
        bound_points = np.unique(np.r_[bound_points, violations])
    else:
        raise RuntimeError(
            "Weighted polynomial exchange did not converge; increase "
            "max_iterations or grid_size."
        )

    assert polynomial is not None and result is not None
    if global_maximum <= 0.0:
        raise RuntimeError("Weighted solver returned a zero polynomial.")
    coefficients = np.asarray(polynomial.coef / global_maximum, dtype=float)
    polynomial = npcheb.Chebyshev(coefficients)
    critical_points = _critical_candidates(polynomial, 0.0, 1.0)
    global_maximum = float(np.max(np.abs(polynomial(critical_points))))
    objective_value = float(np.dot(weights, polynomial(values)))

    coefficients.setflags(write=False)
    critical_points.setflags(write=False)
    return (
        coefficients,
        objective_value,
        global_maximum,
        float(-result.fun * weight_sum),
        iteration,
        critical_points,
    )


def generate_weighted_polynomial(
    k: int,
    lambdas: Any,
    *,
    grid_size: int | None = None,
    max_iterations: int = 30,
    tolerance: float = 2e-8,
    return_info: bool = False,
):
    """Maximize ``abs(sum(lambda_i**2 * p(lambda_i)))`` for bounded odd p.

    ``lambdas`` must be positive and no larger than one. They need not be
    normalized. Symmetry of the feasible set makes the absolute objective
    equivalent to selecting the positive maximizing branch.
    """
    k = _validate_k(k)
    values = _validate_weighted_lambdas(lambdas)
    if grid_size is None:
        grid_size = max(65, 8 * (k + 2) + 1)
    data = _weighted_polynomial_data(
        k,
        tuple(float(value) for value in values),
        int(grid_size),
        int(max_iterations),
        float(tolerance),
    )
    coefficients, objective, max_abs, lp_objective, iterations, critical = data
    polynomial = npcheb.Chebyshev(coefficients.copy())
    if not return_info:
        return polynomial
    info = {
        "degree": k + 1,
        "k": k,
        "lambdas": values.copy(),
        "weights": values * values,
        "weight_sum": float(np.sum(values * values)),
        "normalized_weights": values * values / np.sum(values * values),
        "weighted_objective": float(objective),
        "absolute_weighted_objective": float(abs(objective)),
        "lp_objective": float(lp_objective),
        "global_max_abs": float(max_abs),
        "bound_violation": float(max(0.0, max_abs - 1.0)),
        "exchange_iterations": int(iterations),
        "bound_critical_points": critical.copy(),
    }
    return polynomial, info


def coherent_maxmin_floor(k: int, eps: float) -> float:
    """Return the optimal coherent floor for ``[eps, 1]``."""
    return _coherent_data(_validate_k(k), float(eps))[1]


def eps_from_m(k: int, m: float) -> float:
    """Return the interval endpoint ``eps`` corresponding to ripple ``m``.

    The conversion is uniquely defined for a fixed odd degree ``k+1`` and
    satisfies ``coherent_maxmin_floor(k, eps) == m``.  Float64 Remez inversion
    is used at ordinary degrees; high degrees and ill-conditioned cases use
    the structured arbitrary-precision equal-ripple family.
    """
    k = _validate_k(k)
    m = float(m)
    if not np.isfinite(m) or not 0.0 <= m <= 1.0:
        raise ValueError("m must lie in [0, 1].")
    if k == 0 or m in (0.0, 1.0):
        return m
    if k >= 64:
        return float(_structured_chebyshev_data(k + 1, m)[2])
    try:
        return _eps_from_floor(k, m)
    except RuntimeError:
        return float(_structured_chebyshev_data(k + 1, m)[2])


def m_from_eps(k: int, eps: float) -> float:
    """Return the optimal lower ripple ``m`` for interval ``[eps, 1]``.

    This is the public inverse of :func:`eps_from_m`.  Float64 Remez is used
    at ordinary degrees; high degrees and ill-conditioned cases invert the
    structured arbitrary-precision equal-ripple family.
    """
    k = _validate_k(k)
    eps = float(eps)
    if not np.isfinite(eps) or not 0.0 <= eps <= 1.0:
        raise ValueError("eps must lie in [0, 1].")
    if k == 0 or eps in (0.0, 1.0):
        return eps
    if k >= 64:
        return _structured_floor_from_eps(k + 1, eps)
    try:
        return coherent_maxmin_floor(k, eps)
    except RuntimeError:
        return _structured_floor_from_eps(k + 1, eps)


def _symmetric_values_and_jacobian(
    reduced: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduced = np.asarray(reduced, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1)
    phases = np.r_[reduced, reduced[::-1]]
    count = phases.size
    root = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    W = np.empty((x.size, 2, 2), dtype=complex)
    W[:, 0, 0] = W[:, 1, 1] = x
    W[:, 0, 1] = W[:, 1, 0] = 1j * root

    rotations = []
    derivatives = []
    for phi in phases:
        z = np.exp(1j * phi)
        rotations.append(np.array([[z, 0], [0, np.conj(z)]], dtype=complex))
        derivatives.append(np.array([[1j * z, 0], [0, -1j * np.conj(z)]], dtype=complex))

    identity = np.broadcast_to(np.eye(2, dtype=complex), (x.size, 2, 2)).copy()
    left = [identity]
    for j in range(1, count):
        left.append((left[-1] @ rotations[j - 1]) @ W)
    right = [None] * count
    right[-1] = identity
    for j in range(count - 2, -1, -1):
        right[j] = (W @ rotations[j + 1]) @ right[j + 1]

    unitary = left[-1] @ rotations[-1]
    jacobian = np.empty((x.size, count), dtype=float)
    for j in range(count):
        jacobian[:, j] = np.real(
            ((left[j] @ derivatives[j]) @ right[j])[:, 0, 0]
        )
    reduced_jacobian = jacobian[:, : len(reduced)] + jacobian[:, len(reduced) :][:, ::-1]
    return np.real(unitary[:, 0, 0]), reduced_jacobian, phases


def _symmetric_phases(
    coefficients: np.ndarray,
    k: int,
    *,
    homotopy_steps: int | None,
    max_nfev: int,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if k == 0:
        return np.zeros(2), {"factorization_error": 0.0, "total_nfev": 0}
    count = k // 2 + 1
    nodes = np.cos((2 * np.arange(count) + 1) * np.pi / (4 * count))
    target = npcheb.chebval(nodes, coefficients)
    reduced = np.zeros(count)
    reduced[0] = np.pi / 4
    steps = max(30, k) if homotopy_steps is None else int(homotopy_steps)
    if steps < 1:
        raise ValueError("homotopy_steps must be positive.")
    solver_tolerance = max(1e-14, min(1e-10, tolerance * 0.1))
    total_nfev = 0
    result = None
    for scale in np.linspace(0.0, 1.0, steps + 1)[1:]:
        evaluator = lambda z: _symmetric_values_and_jacobian(z, nodes)
        result = least_squares(
            lambda z: evaluator(z)[0] - scale * target,
            reduced,
            jac=lambda z: evaluator(z)[1],
            method="trf",
            xtol=solver_tolerance,
            ftol=solver_tolerance,
            gtol=solver_tolerance,
            max_nfev=max_nfev,
        )
        reduced = result.x
        total_nfev += result.nfev
    assert result is not None
    error = float(np.max(np.abs(result.fun)))
    if error > tolerance:
        polished = least_squares(
            lambda z: _symmetric_values_and_jacobian(z, nodes)[0] - target,
            reduced,
            jac=lambda z: _symmetric_values_and_jacobian(z, nodes)[1],
            method="lm",
            xtol=solver_tolerance,
            ftol=solver_tolerance,
            gtol=solver_tolerance,
            max_nfev=max(20000, max_nfev),
        )
        total_nfev += polished.nfev
        polished_error = float(np.max(np.abs(polished.fun)))
        if polished_error < error:
            reduced, result, error = polished.x, polished, polished_error
    if error > 1e-7:
        raise RuntimeError(f"Symmetric phase factorization residual is {error:.3e}.")
    return np.r_[reduced, reduced[::-1]], {
        "factorization_error": error,
        "total_nfev": total_nfev,
        "homotopy_steps": steps,
        "solver_status": int(result.status),
    }


# ---------------------------------------------------------------------------
# Direct functional diagnostics (avoid ill-conditioned power coefficients)
# ---------------------------------------------------------------------------


def _qsp_u00(x: float, phases: np.ndarray) -> complex:
    s = math.sqrt(max(0.0, 1.0 - x * x))
    W = np.array([[x, 1j * s], [1j * s, x]], dtype=complex)
    U = np.diag(np.exp(1j * np.array([phases[0], -phases[0]])))
    for phase in phases[1:]:
        U = U @ W @ np.diag(np.exp(1j * np.array([phase, -phase])))
    return complex(U[0, 0])


def _qsp_u00_vectorized(points: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Evaluate U_00 on an array of points using the scalar QSP recurrence."""
    x = np.asarray(points, dtype=float)
    P_value = np.full(x.shape, np.exp(1j * phases[0]), dtype=complex)
    Q_value = np.zeros(x.shape, dtype=complex)
    one_minus_x2 = 1.0 - x * x

    for phi in phases[1:]:
        old_P = P_value
        old_Q = Q_value
        phase = np.exp(1j * phi)
        P_value = phase * (x * old_P - one_minus_x2 * old_Q)
        Q_value = np.conj(phase) * (old_P + x * old_Q)
    return P_value


def _functional_errors(P, phases: np.ndarray, grid_size: int):
    theta = np.linspace(0.0, np.pi, grid_size)
    points = np.cos(theta)
    values = _qsp_u00_vectorized(points, phases)

    # Keep the target evaluation in arbitrary precision.  Direct evaluation of
    # its large monomial coefficients in float/long-double can lose many digits.
    targets = np.array(
        [float(mp.re(_peval(P, mp.mpf(repr(float(x)))))) for x in points],
        dtype=float,
    )
    function_error = float(np.max(np.abs(values.real - targets)))
    imaginary_error = float(np.max(np.abs(values.imag)))
    return function_error, imaginary_error


def _ripple_error(P, critical_points: np.ndarray, m: float) -> float:
    error = 0.0
    for j, point in enumerate(critical_points):
        target = 1.0 if j % 2 == 0 else m
        value = float(mp.re(_peval(P, mp.mpf(repr(float(point))))))
        error = max(error, abs(value - target))
    endpoint = float(mp.re(_peval(P, mp.mpf(1))))
    return max(error, abs(endpoint - 1.0))


def qsvt_pq(
    x: Any,
    angles: Any,
    *,
    convention: str = "coherent",
):
    """Evaluate the QSP/QSVT polynomial pair ``(P, Q)`` over ``x``.

    The returned values satisfy

    ``P(x) = U(x)[0, 0]`` and
    ``Q(x) = U(x)[1, 0] / (1j*sqrt(1-x**2))``.

    ``Q`` is evaluated directly by polynomial recurrence rather than literal
    division, so its endpoint values at ``x=+/-1`` remain finite.

    Parameters
    ----------
    x:
        Scalar or NumPy-compatible array.
    angles:
        For ``convention="coherent"``, the ``k`` internal angles in the
        ``D(theta)=diag(exp(i*theta),-1)`` convention used by
        ``coherent_maxmin_qsvt.py``.  For ``convention="standard"``, the
        complete ``N+1`` phase vector for
        ``exp(i*phi[0]Z) prod_j(W exp(i*phi[j]Z))``.
    convention:
        ``"coherent"`` (aliases ``"qsvt"`` and ``"d"``) or ``"standard"``
        (alias ``"qsp"``).
    """
    phases = np.asarray(angles, dtype=float).reshape(-1)
    x_array = np.asarray(x)
    scalar = x_array.ndim == 0
    shape = x_array.shape
    flat = x_array.reshape(-1).astype(np.complex128)
    key = convention.lower().replace("-", "_")

    one_minus_x2 = 1.0 - flat * flat
    if key in {"coherent", "qsvt", "d", "d_theta"}:
        p_value = np.ones(flat.shape, dtype=np.complex128)
        q_value = np.zeros(flat.shape, dtype=np.complex128)
        for theta in phases:
            old_p = p_value
            old_q = q_value
            p_value = np.exp(1j * theta) * (
                flat * old_p - one_minus_x2 * old_q
            )
            q_value = -(old_p + flat * old_q)

        old_p = p_value
        old_q = q_value
        p_value = flat * old_p - one_minus_x2 * old_q
        q_value = old_p + flat * old_q
    elif key in {"standard", "qsp", "z_phases"}:
        if phases.size < 2:
            raise ValueError("A standard-QSP sequence must contain at least two phases.")
        # Standard QSP is defined on real x.  In the canonical matrix form,
        # the existing complementary recurrence represents U_01/(i*s), while
        # U_10/(i*s) is its complex conjugate.
        real_x = flat.real
        one_minus_x2 = 1.0 - real_x * real_x
        p_value = np.full(
            real_x.shape, np.exp(1j * phases[0]), dtype=np.complex128
        )
        upper_q = np.zeros(real_x.shape, dtype=np.complex128)
        for phi in phases[1:]:
            old_p = p_value
            old_q = upper_q
            phase = np.exp(1j * phi)
            p_value = phase * (real_x * old_p - one_minus_x2 * old_q)
            upper_q = np.conj(phase) * (old_p + real_x * old_q)
        q_value = np.conj(upper_q)
    else:
        raise ValueError(f"Unknown phase convention {convention!r}.")

    p_result = p_value.reshape(shape)
    q_result = q_value.reshape(shape)
    if scalar:
        return p_result.item(), q_result.item()
    return p_result, q_result


def qsvt_p(
    x: Any,
    angles: Any,
    *,
    convention: str = "coherent",
):
    """Backward-compatible wrapper returning only ``P`` from :func:`qsvt_pq`."""
    return qsvt_pq(x, angles, convention=convention)[0]


def FPAA_angle(k, gamma):
    """Return the fixed-point amplitude-amplification internal angles."""
    k = _validate_k(k)
    half = k // 2
    length = 2 * half + 1
    indices = np.arange(1, half + 1)
    angles = np.zeros(2 * half)
    angles[1::2] = np.pi + 2 * np.arctan(
        np.tan(2 * np.pi * indices / length) * gamma
    )
    angles[0::2] = angles[1::2][::-1]
    return angles


def _validate_probability_problem(k: int, a: float) -> tuple[int, float]:
    k = _validate_k(k)
    a = float(a)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("a must satisfy 0 < a < 1.")
    return k, a


def _log_cosh(value: np.ndarray | float):
    absolute = np.abs(value)
    return absolute + np.log1p(np.exp(-2.0 * absolute)) - np.log(2.0)


def _scaled_fpaa_failure_amplitude(
    x: np.ndarray,
    degree: int,
    a: float,
    log_denominator: float,
) -> np.ndarray:
    """Evaluate delta*T_degree(sqrt(1-x^2)/sqrt(1-a^2)) stably."""
    x = np.asarray(x, dtype=float)
    argument = np.sqrt(np.maximum(0.0, 1.0 - x * x)) / np.sqrt(1.0 - a * a)
    result = np.empty_like(argument)
    inside = argument <= 1.0
    result[inside] = (
        np.cos(degree * np.arccos(np.clip(argument[inside], 0.0, 1.0)))
        * np.exp(-log_denominator)
    )
    if np.any(~inside):
        numerator = degree * np.arccosh(argument[~inside])
        result[~inside] = np.exp(_log_cosh(numerator) - log_denominator)
    return result


@lru_cache(maxsize=128)
def _probability_polynomial_data(
    k: int,
    a: float,
) -> tuple[np.ndarray, float, float]:
    """Return Chebyshev coefficients, optimal probability, and delta."""
    k, a = _validate_probability_problem(k, a)
    amplitude_degree = k + 1
    probability_degree = 2 * amplitude_degree
    denominator_argument = 1.0 / np.sqrt(1.0 - a * a)
    log_denominator = float(
        _log_cosh(amplitude_degree * np.arccosh(denominator_argument))
    )
    delta = float(np.exp(-log_denominator))
    optimal_probability = float(-np.expm1(-2.0 * log_denominator))

    # Interpolate the exact even probability polynomial at first-kind
    # Chebyshev nodes using the corresponding discrete cosine transform.
    count = probability_degree + 1
    index = np.arange(count)
    theta = np.pi * (index + 0.5) / count
    nodes = np.cos(theta)
    failure = _scaled_fpaa_failure_amplitude(
        nodes, amplitude_degree, a, log_denominator
    )
    values = 1.0 - failure * failure
    orders = np.arange(count)
    coefficients = (2.0 / count) * (
        np.cos(np.outer(orders, theta)) @ values
    )
    coefficients[0] *= 0.5
    coefficients[1::2] = 0.0
    coefficients[np.abs(coefficients) < 5e-15] = 0.0
    coefficients.setflags(write=False)
    return coefficients, optimal_probability, delta


def generate_probability_polynomial(
    k: int,
    a: float,
    *,
    return_info: bool = False,
):
    """Return the optimal success-probability polynomial ``|P(x)|^2``.

    The underlying odd QSP amplitude has degree at most ``k+1`` and is
    globally bounded by one. This solves

    ``max_P min_{x in [a,1]} |P(x)|^2``.

    The returned real, even Chebyshev polynomial has degree at most
    ``2*(k+1)``.
    """
    k, a = _validate_probability_problem(k, a)
    coefficients, floor, delta = _probability_polynomial_data(k, a)
    polynomial = npcheb.Chebyshev(coefficients.copy())
    if not return_info:
        return polynomial
    info = {
        "amplitude_degree": k + 1,
        "probability_degree": 2 * (k + 1),
        "k": k,
        "a": a,
        "gamma": a,
        "failure_amplitude_delta": float(delta),
        "optimal_probability": float(floor),
    }
    return polynomial, info


def _fpaa_standard_phases(angles: np.ndarray) -> np.ndarray:
    """Convert coherent internal FPAA angles to a full standard phase list."""
    internal = 0.5 * (np.asarray(angles, dtype=float)[::-1] - np.pi)
    return np.r_[0.0, internal, 0.0]


def generate_probability_angles(
    k: int,
    a: float,
    *,
    convention: str = "coherent",
    return_info: bool = False,
    verification_grid_size: int | None = None,
):
    """Return FPAA phases attaining the optimal modulus-squared floor.

    The FPAA parameter is exactly ``gamma=a``. The returned sequence is
    checked against :func:`generate_probability_polynomial` using
    :func:`qsvt_pq`.
    """
    k, a = _validate_probability_problem(k, a)
    coherent_angles = FPAA_angle(k, a)
    key = convention.lower().replace("-", "_")
    if key in {"coherent", "qsvt", "d", "d_theta"}:
        angles = coherent_angles
        convention_name = "coherent"
    elif key in {"standard", "qsp", "z_phases"}:
        angles = _fpaa_standard_phases(coherent_angles)
        convention_name = "standard"
    else:
        raise ValueError(f"Unknown phase convention {convention!r}.")

    probability, probability_info = generate_probability_polynomial(
        k, a, return_info=True
    )
    if verification_grid_size is None:
        verification_grid_size = max(2001, 40 * (k + 2))
    if verification_grid_size < 2 * (k + 1) + 1:
        raise ValueError(
            "verification_grid_size must be at least 2*(k+1)+1."
        )
    grid = np.cos(np.linspace(0.0, np.pi, int(verification_grid_size)))
    p_value, q_value = qsvt_pq(grid, angles, convention=convention_name)
    generated_probability = np.abs(p_value) ** 2
    verification_error = float(
        np.max(np.abs(generated_probability - probability(grid)))
    )
    unitarity_error = float(
        np.max(
            np.abs(
                generated_probability
                + (1.0 - grid * grid) * np.abs(q_value) ** 2
                - 1.0
            )
        )
    )

    # Evaluate at the exact equal-ripple points in [a,1].
    amplitude_degree = k + 1
    ripple_index = np.arange(amplitude_degree // 2 + 1)
    failure_argument = np.cos(np.pi * ripple_index / amplitude_degree)
    ripple_points = np.sqrt(
        np.maximum(0.0, 1.0 - (1.0 - a * a) * failure_argument**2)
    )
    ripple_p, _ = qsvt_pq(
        ripple_points, angles, convention=convention_name
    )
    verified_floor = float(np.min(np.abs(ripple_p) ** 2))
    floor_error = abs(verified_floor - probability_info["optimal_probability"])
    if verification_error > 3e-10 or floor_error > 3e-10:
        raise RuntimeError(
            "FPAA angles failed optimal probability verification: "
            f"polynomial error={verification_error:.3e}, "
            f"floor error={floor_error:.3e}."
        )
    if unitarity_error > 3e-10:
        raise RuntimeError(
            "FPAA angles failed P/Q unitarity verification: "
            f"{unitarity_error:.3e}."
        )

    info = {
        **probability_info,
        "convention": convention_name,
        "verified_probability_floor": verified_floor,
        "probability_verification_error": verification_error,
        "floor_verification_error": float(floor_error),
        "qsvt_pq_unitarity_error": unitarity_error,
        "ripple_points": ripple_points,
        "fpaa_coherent_angles": coherent_angles.copy(),
    }
    return (angles, info) if return_info else angles


def generate_angles(
    k: int,
    *,
    eps: float | None = None,
    m: float | None = None,
    symmetry: str = "auto",
    convention: str = "coherent",
    return_info: bool = False,
    homotopy_steps: int | None = None,
    max_nfev: int = 6000,
    tolerance: float = 1e-11,
    **antisymmetric_options,
):
    """Generate phases for the selected degree-``k+1`` sign polynomial.

    ``symmetry="auto"`` (the default) selects the faster antisymmetric branch
    when its degree constraint is met, otherwise the symmetric branch.
    The symmetric branch supports every nonnegative even ``k``.  The strict
    antisymmetric branch supports ``k=0`` and ``k`` divisible by four; this is
    exactly the degree class in which antisymmetry can realize the upper-first
    coherent polynomial with ``P(1)=1``.

    ``convention="coherent"`` returns ``k`` internal D angles.
    ``convention="standard"`` returns all ``k+2`` standard Z-rotation phases.
    """
    k = _validate_k(k)
    branch = symmetry.lower().replace("-", "_")
    if branch == "auto":
        branch = "antisymmetric" if (k + 1) % 4 == 1 else "symmetric"
    anti_branch = branch in {"antisymmetric", "antisymmetry", "anti"}
    if (eps is None) == (m is None):
        raise ValueError("Supply exactly one of eps or m.")

    # With m supplied, the antisymmetric construction already produces the
    # target and should not pay for an inverse Remez solve first.
    direct_antisymmetric_m = anti_branch and m is not None
    if direct_antisymmetric_m:
        floor = float(m)
        if not np.isfinite(floor) or not 0.0 <= floor <= 1.0:
            raise ValueError("m must lie in [0, 1] for a coherent sign polynomial.")
        resolved_eps = float("nan")
        coefficients = None
        sign_error = (1.0 - floor) / (1.0 + floor)
        alternants = np.empty(0)
        remez_iterations = 0
    else:
        if m is not None and k >= 64 and m not in (0.0, 1.0):
            floor = float(m)
            coefficients, critical, resolved_eps = _structured_chebyshev_data(
                k + 1, floor
            )
            sign_error = (1.0 - floor) / (1.0 + floor)
            alternants = np.r_[resolved_eps, critical, 1.0]
            remez_iterations = 0
        elif eps is not None and k >= 64 and eps not in (0.0, 1.0):
            resolved_eps = float(eps)
            floor = _structured_floor_from_eps(k + 1, resolved_eps)
            coefficients, critical, _ = _structured_chebyshev_data(k + 1, floor)
            sign_error = (1.0 - floor) / (1.0 + floor)
            alternants = np.r_[resolved_eps, critical, 1.0]
            remez_iterations = 0
        else:
            try:
                resolved_eps, floor = _resolve_parameter(k, eps=eps, m=m)
                coefficients, _, sign_error, alternants, remez_iterations = _coherent_data(
                    k, resolved_eps
                )
            except RuntimeError:
                if m is None or k + 1 < 3:
                    raise
                floor = float(m)
                coefficients, critical, resolved_eps = _structured_chebyshev_data(
                    k + 1, floor
                )
                sign_error = (1.0 - floor) / (1.0 + floor)
                alternants = np.r_[resolved_eps, critical, 1.0]
                remez_iterations = 0
    convention_key = convention.lower().replace("-", "_")
    coherent_convention = convention_key in {"coherent", "qsvt", "d", "d_theta"}
    standard_convention = convention_key in {"standard", "qsp", "z_phases"}
    if not (coherent_convention or standard_convention):
        raise ValueError(f"Unknown phase convention {convention!r}.")

    diagnostics: dict[str, Any]
    if branch in {"symmetric", "symmetry", "palindromic"}:
        # In the D convention these edge cases have the exact canonical
        # realization consisting entirely of zero internal angles.
        if coherent_convention and (k == 0 or resolved_eps in (0.0, 1.0)):
            angles = np.zeros(k)
            standard_phases = None
            alignment = 0.0
            diagnostics = {"factorization_error": 0.0, "total_nfev": 0}
        else:
            standard_phases, diagnostics = _symmetric_phases(
                coefficients,
                k,
                homotopy_steps=homotopy_steps,
                max_nfev=max_nfev,
                tolerance=tolerance,
            )
            if standard_convention:
                angles = standard_phases.copy()
                alignment = 0.0
            else:
                internal = standard_phases[1:-1][::-1]
                angles = np.angle(np.exp(1j * (np.pi + 2 * internal)))
                alignment = float(
                    np.angle(
                        np.exp(1j * (np.sum(internal) - 2 * standard_phases[0]))
                    )
                )
        symmetry_name = "symmetric"
    elif branch in {"antisymmetric", "antisymmetry", "anti"}:
        degree = k + 1
        if degree != 1 and degree % 4 != 1:
            raise ValueError(
                "Strict antisymmetric construction requires degree k+1 = 4*r+1. "
                "Use symmetry='symmetric' for degrees congruent to 3 mod 4."
            )
        if degree == 1:
            standard_phases = np.zeros(2)
            if coefficients is None:
                coefficients = np.array([0.0, 1.0])
                resolved_eps = floor
                alternants = np.array([resolved_eps, 1.0])
            diagnostics = {
                "ripple_error": 0.0,
                "qsp_function_error": 0.0,
                "antisymmetry_error": 0.0,
            }
        else:
            anti_result = generate_structured_antisymmetric_qsp(
                degree, floor, **antisymmetric_options
            )
            standard_phases = anti_result.phases.copy()
            if coefficients is None:
                coefficients = anti_result.target_chebyshev_coefficients.copy()
                polynomial = npcheb.Chebyshev(coefficients)
                resolved_eps = float(
                    brentq(
                        lambda value: polynomial(value) - floor,
                        0.0,
                        anti_result.critical_points[0],
                    )
                )
                alternants = np.r_[
                    resolved_eps, anti_result.critical_points, 1.0
                ]
            diagnostics = {
                "ripple_error": anti_result.ripple_error,
                "qsp_function_error": anti_result.qsp_function_error,
                "qsp_imaginary_error": anti_result.qsp_imaginary_error,
                "antisymmetry_error": anti_result.antisymmetry_error,
                "layer_stripping_tail": anti_result.layer_stripping_tail,
                "bezout_error": anti_result.bezout_error,
                "unitarity_error": anti_result.unitarity_error,
                "mp_dps": anti_result.mp_dps,
            }
        alignment = 0.0
        if standard_convention:
            angles = standard_phases
        else:
            internal = standard_phases[1:-1][::-1]
            angles = np.angle(np.exp(1j * (np.pi + 2 * internal)))
        symmetry_name = "antisymmetric"
    else:
        raise ValueError("symmetry must be 'symmetric' or 'antisymmetric'.")

    assert coefficients is not None
    grid = np.cos(np.linspace(0.0, np.pi, max(401, 20 * (k + 2))))
    target = npcheb.chebval(grid, coefficients)
    values = qsvt_p(grid, angles, convention=convention)
    projection = np.real(np.exp(-1j * alignment) * values)
    verification_error = float(np.max(np.abs(projection - target)))
    if verification_error > 3e-7:
        raise RuntimeError(
            f"Generated phases failed polynomial verification: {verification_error:.3e}."
        )

    info = {
        "degree": k + 1,
        "k": k,
        "eps": float(resolved_eps),
        "m": float(floor),
        "optimal_floor": float(floor),
        "sign_error": float(sign_error),
        "alternation_points": alternants.copy(),
        "remez_iterations": int(remez_iterations),
        "symmetry": symmetry_name,
        "convention": "coherent" if coherent_convention else "standard",
        "alignment_phase": float(alignment),
        "verification_error": verification_error,
        **diagnostics,
    }
    return (angles, info) if return_info else angles


def generate_interval_angles(
    k: int,
    a: float,
    b: float,
    *,
    convention: str = "coherent",
    return_info: bool = False,
    grid_size: int | None = None,
    max_iterations: int = 30,
    lp_tolerance: float = 2e-8,
    homotopy_steps: int | None = None,
    max_nfev: int = 6000,
    factorization_tolerance: float = 1e-11,
):
    """Generate symmetric phases for the bounded interval-maximin polynomial.

    This solves ``max_p min_{x in [a,b]} p(x)`` under odd degree/parity and
    global boundedness constraints, factors the result through symmetric QSP,
    and verifies both the target projection and QSP unitarity using
    :func:`qsvt_pq`.
    """
    polynomial, polynomial_info = generate_interval_polynomial(
        k,
        a,
        b,
        grid_size=grid_size,
        max_iterations=max_iterations,
        tolerance=lp_tolerance,
        return_info=True,
    )
    coefficients = np.asarray(polynomial.coef, dtype=float)
    standard_phases, factorization_info = _symmetric_phases(
        coefficients,
        k,
        homotopy_steps=homotopy_steps,
        max_nfev=max_nfev,
        tolerance=factorization_tolerance,
    )

    key = convention.lower().replace("-", "_")
    if key in {"standard", "qsp", "z_phases"}:
        angles = standard_phases.copy()
        alignment = 0.0
        convention_name = "standard"
    elif key in {"coherent", "qsvt", "d", "d_theta"}:
        internal = standard_phases[1:-1][::-1]
        angles = np.angle(np.exp(1j * (np.pi + 2 * internal)))
        alignment = float(
            np.angle(
                np.exp(1j * (np.sum(internal) - 2 * standard_phases[0]))
            )
        )
        convention_name = "coherent"
    else:
        raise ValueError(f"Unknown phase convention {convention!r}.")

    verification_grid = np.cos(
        np.linspace(0.0, np.pi, max(2001, 40 * (k + 2)))
    )
    p_value, q_value = qsvt_pq(
        verification_grid, angles, convention=convention_name
    )
    target = polynomial(verification_grid)
    projection = np.real(np.exp(-1j * alignment) * p_value)
    verification_error = float(np.max(np.abs(projection - target)))
    unitarity_error = float(
        np.max(
            np.abs(
                np.abs(p_value) ** 2
                + (1.0 - verification_grid**2) * np.abs(q_value) ** 2
                - 1.0
            )
        )
    )
    if verification_error > 3e-7:
        raise RuntimeError(
            "Interval phase sequence failed target verification: "
            f"{verification_error:.3e}."
        )
    if unitarity_error > 3e-7:
        raise RuntimeError(
            "Interval phase sequence failed P/Q unitarity verification: "
            f"{unitarity_error:.3e}."
        )

    info = {
        **polynomial_info,
        **factorization_info,
        "symmetry": "symmetric",
        "convention": convention_name,
        "alignment_phase": alignment,
        "verification_error": verification_error,
        "qsvt_pq_unitarity_error": unitarity_error,
        "standard_phases": standard_phases.copy(),
    }
    return (angles, info) if return_info else angles


def generate_weighted_angles(
    k: int,
    lambdas: Any,
    *,
    convention: str = "coherent",
    return_info: bool = False,
    grid_size: int | None = None,
    max_iterations: int = 30,
    lp_tolerance: float = 2e-8,
    homotopy_steps: int | None = None,
    max_nfev: int = 6000,
    factorization_tolerance: float = 1e-11,
):
    """Generate symmetric phases for the optimal weighted polynomial."""
    polynomial, polynomial_info = generate_weighted_polynomial(
        k,
        lambdas,
        grid_size=grid_size,
        max_iterations=max_iterations,
        tolerance=lp_tolerance,
        return_info=True,
    )
    standard_phases, factorization_info = _symmetric_phases(
        np.asarray(polynomial.coef, dtype=float),
        k,
        homotopy_steps=homotopy_steps,
        max_nfev=max_nfev,
        tolerance=factorization_tolerance,
    )

    key = convention.lower().replace("-", "_")
    if key in {"standard", "qsp", "z_phases"}:
        angles = standard_phases.copy()
        alignment = 0.0
        convention_name = "standard"
    elif key in {"coherent", "qsvt", "d", "d_theta"}:
        internal = standard_phases[1:-1][::-1]
        angles = np.angle(np.exp(1j * (np.pi + 2 * internal)))
        alignment = float(
            np.angle(
                np.exp(1j * (np.sum(internal) - 2 * standard_phases[0]))
            )
        )
        convention_name = "coherent"
    else:
        raise ValueError(f"Unknown phase convention {convention!r}.")

    verification_grid = np.cos(
        np.linspace(0.0, np.pi, max(2001, 40 * (k + 2)))
    )
    p_value, q_value = qsvt_pq(
        verification_grid, angles, convention=convention_name
    )
    target = polynomial(verification_grid)
    projection = np.real(np.exp(-1j * alignment) * p_value)
    verification_error = float(np.max(np.abs(projection - target)))
    unitarity_error = float(
        np.max(
            np.abs(
                np.abs(p_value) ** 2
                + (1.0 - verification_grid**2) * np.abs(q_value) ** 2
                - 1.0
            )
        )
    )

    values = polynomial_info["lambdas"]
    weights = polynomial_info["weights"]
    generated_p, _ = qsvt_pq(values, angles, convention=convention_name)
    generated_sum = complex(np.dot(weights, generated_p))
    generated_objective = float(abs(generated_sum))
    objective_error = abs(
        generated_objective - polynomial_info["absolute_weighted_objective"]
    )
    if verification_error > 3e-7 or objective_error > 3e-7:
        raise RuntimeError(
            "Weighted phase sequence failed verification: "
            f"polynomial error={verification_error:.3e}, "
            f"objective error={objective_error:.3e}."
        )
    if unitarity_error > 3e-7:
        raise RuntimeError(
            "Weighted phase sequence failed P/Q unitarity verification: "
            f"{unitarity_error:.3e}."
        )

    info = {
        **polynomial_info,
        **factorization_info,
        "symmetry": "symmetric",
        "convention": convention_name,
        "alignment_phase": alignment,
        "verification_error": verification_error,
        "qsvt_pq_unitarity_error": unitarity_error,
        "generated_weighted_sum": generated_sum,
        "generated_absolute_weighted_objective": generated_objective,
        "weighted_objective_verification_error": float(objective_error),
        "standard_phases": standard_phases.copy(),
    }
    return (angles, info) if return_info else angles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_structured_antisymmetric_qsp(
    N: int,
    m: float,
    *,
    spectral_signs: Sequence[int] | None = None,
    conjugate: bool = False,
    mp_dps: int | None = None,
    critical_tolerance: float = 2e-12,
    initial_step: float = 0.05,
    min_step: float = 1e-8,
    max_nfev: int = 12000,
    continuation_growth: float = 2.2,
    root_factor_digits: int | None = None,
    root_max_iterations: int = 60,
    diagnostic_grid_size: int = 201,
) -> StructuredAntisymmetricQSPResult:
    """Generate one structure-preserving antisymmetric QSP phase sequence.

    Parameters
    ----------
    N:
        Degree N=4*r+1, N>=5.
    m:
        Lower ripple level in [-1,1].
    spectral_signs:
        Length-r tuple of +/-1 values selecting one root from each conjugate
        pair of the Bézout polynomial R.  The default is all +1.
    conjugate:
        Negate the final phase vector and conjugate the complementary branch.
    mp_dps:
        Arbitrary-precision decimal digits used for Bézout reconstruction,
        factorization refinement, and layer stripping.  The default is
        max(80,N+40).
    critical_tolerance:
        Tolerance for the adaptive critical-point continuation.
    continuation_growth:
        Multiplicative growth of accepted continuation steps.
    root_factor_digits:
        Requested relative coefficient accuracy for R=L*conjugate(L).  The
        default retains about 50 guard digits relative to mp_dps.
    root_max_iterations:
        Maximum simultaneous Aberth iterations.
    diagnostic_grid_size:
        Number of Chebyshev-spaced points used for direct functional checks.
        Set to 0 to skip the optional grid diagnostic.
    """
    n, r, m = _validate(N, m)
    if mp_dps is None:
        mp_dps = max(80, N + 40)
    if root_factor_digits is None:
        # High-degree layer stripping amplifies tiny spectral-factor errors.
        # Retain roughly 50 guard digits while avoiding needless full-dps roots.
        root_factor_digits = min(110, max(60, mp_dps - 50))
    if root_factor_digits >= mp_dps - 10:
        raise ValueError("root_factor_digits must leave at least 10 guard digits.")
    if mp_dps < 40:
        raise ValueError("mp_dps should be at least 40.")

    if spectral_signs is None:
        signs = tuple(1 for _ in range(r))
    else:
        signs = tuple(int(value) for value in spectral_signs)
        if len(signs) != r:
            raise ValueError(f"spectral_signs must have length r={r}.")
        if any(value not in (-1, 1) for value in signs):
            raise ValueError("Every spectral sign must be +1 or -1.")

    critical_points = _critical_points_adaptive(
        N,
        m,
        tolerance=critical_tolerance,
        initial_step=initial_step,
        min_step=min_step,
        max_nfev=max_nfev,
        continuation_growth=continuation_growth,
    )
    upper_extrema = critical_points[::2]

    with mp.workdps(mp_dps):
        if m == -1.0:
            # The upper-first family equals T_N for N=4r+1.
            phases = np.zeros(N + 1, dtype=float)
            selected = []

            # Build T_N in arbitrary precision only for diagnostics/output.
            cheb = [mp.mpf(0)] * (N + 1)
            cheb[N] = 1
            P = _chebyshev_to_power(cheb)
            A = [mp.mpf(1)]
            R = [mp.mpf(1)]
            bezout = mp.mpf(0)
            unity = mp.mpf(0)
            strip_tail = mp.mpf(0)
        else:
            A, R, P = _structured_polynomials(upper_extrema, N, m)
            bezout = _bezout_error(A, R)
            Q, selected = _companion_from_R(
                critical_points,
                upper_extrema,
                R,
                P,
                signs,
                m,
                root_factor_digits=root_factor_digits,
                root_max_iterations=root_max_iterations,
            )
            unity = _unitarity_error(P, Q)
            raw_phases, strip_tail = _strip_phases(P, Q)
            phases = _enforce_antisymmetry(raw_phases)

        if conjugate:
            phases = -phases
            selected = [mp.conj(value) for value in selected]

        ripple = _ripple_error(P, critical_points, m)
        if diagnostic_grid_size > 0:
            function_error, imaginary_error = _functional_errors(
                P, phases, diagnostic_grid_size
            )
        else:
            function_error = float("nan")
            imaginary_error = float("nan")
        power_coefficients = np.array([float(mp.re(value)) for value in P])
        chebyshev_coefficients = _power_to_chebyshev_via_samples(P)

    selected_numpy = np.array(
        [complex(float(mp.re(value)), float(mp.im(value))) for value in selected],
        dtype=complex,
    )
    independent = phases[1 : n + 1].copy()

    return StructuredAntisymmetricQSPResult(
        N=N,
        m=m,
        phases=phases,
        independent_phases=independent,
        critical_points=critical_points,
        upper_extrema=upper_extrema,
        spectral_signs=signs,
        selected_roots_of_R=selected_numpy,
        mp_dps=mp_dps,
        ripple_error=float(ripple),
        qsp_function_error=float(function_error),
        qsp_imaginary_error=float(imaginary_error),
        antisymmetry_error=float(np.max(np.abs(phases + phases[::-1]))),
        layer_stripping_tail=float(strip_tail),
        bezout_error=float(bezout),
        unitarity_error=float(unity),
        target_power_coefficients=power_coefficients,
        target_chebyshev_coefficients=chebyshev_coefficients,
    )


def structured_antisymmetric_angles(N: int, m: float, **kwargs) -> np.ndarray:
    # Angle-only calls should not pay for an optional function-space diagnostic.
    kwargs.setdefault("diagnostic_grid_size", 0)
    return generate_structured_antisymmetric_qsp(N, m, **kwargs).phases


def plot_angles(
    phases_or_result: ArrayLike | StructuredAntisymmetricQSPResult,
    *,
    independent: bool = False,
    degrees: bool = False,
    ax=None,
    show: bool = True,
    save: str | Path | None = None,
):
    import matplotlib.pyplot as plt

    if isinstance(phases_or_result, StructuredAntisymmetricQSPResult):
        result = phases_or_result
        values = result.independent_phases if independent else result.phases
        title = f"Structured antisymmetric QSP phases: N={result.N}, m={result.m:g}"
    else:
        full = np.asarray(phases_or_result, dtype=float)
        if independent:
            n = (len(full) - 2) // 2
            values = full[1 : n + 1]
        else:
            values = full
        title = "Structured antisymmetric QSP phase sequence"

    if degrees:
        values = np.degrees(values)
        ylabel = "phase (degrees)"
    else:
        ylabel = "phase (radians)"

    if ax is None:
        _, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(np.arange(len(values)), values, marker="o")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("independent phase index" if independent else "phase index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    if save is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(Path(save), dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def _format_array(array: np.ndarray, precision: int = 12) -> str:
    return np.array2string(
        np.asarray(array),
        precision=precision,
        separator=", ",
        max_line_width=120,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Structure-preserving antisymmetric equal-ripple QSP phases."
    )
    parser.add_argument("N", type=int, help="Degree N=4*r+1")
    parser.add_argument("m", type=float, help="Lower ripple level in [-1,1]")
    parser.add_argument("--dps", type=int, default=None)
    parser.add_argument(
        "--signs",
        type=str,
        default=None,
        help="Comma-separated spectral signs, e.g. +1,-1,+1,-1",
    )
    parser.add_argument("--conjugate", action="store_true")
    parser.add_argument("--root-digits", type=int, default=None)
    parser.add_argument(
        "--grid",
        type=int,
        default=201,
        help="Diagnostic grid size; use 0 to skip functional diagnostics.",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    signs = None
    if args.signs:
        signs = tuple(int(item.strip()) for item in args.signs.split(","))

    result = generate_structured_antisymmetric_qsp(
        args.N,
        args.m,
        spectral_signs=signs,
        conjugate=args.conjugate,
        mp_dps=args.dps,
        root_factor_digits=args.root_digits,
        diagnostic_grid_size=args.grid,
    )
    print("phases =", _format_array(result.phases))
    print("critical_points =", _format_array(result.critical_points))
    print(f"ripple_error        = {result.ripple_error:.3e}")
    print(f"qsp_function_error  = {result.qsp_function_error:.3e}")
    print(f"qsp_imaginary_error = {result.qsp_imaginary_error:.3e}")
    print(f"bezout_error        = {result.bezout_error:.3e}")
    print(f"unitarity_error     = {result.unitarity_error:.3e}")

    if args.plot or args.save:
        result.plot(show=args.plot, save=args.save)


__all__ = [
    "FPAA_angle",
    "StructuredAntisymmetricQSPResult",
    "coherent_maxmin_floor",
    "eps_from_m",
    "generate_angles",
    "generate_interval_angles",
    "generate_interval_polynomial",
    "generate_probability_angles",
    "generate_probability_polynomial",
    "generate_polynomial",
    "generate_structured_antisymmetric_qsp",
    "generate_weighted_angles",
    "generate_weighted_polynomial",
    "plot_angles",
    "qsvt_p",
    "qsvt_pq",
    "m_from_eps",
    "structured_antisymmetric_angles",
]


if __name__ == "__main__":
    _main()
