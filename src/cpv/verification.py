"""Verification tools: error metrics, convergence fitting, algebraic checks.

A simulation that runs is not a result. What makes a numerical answer usable is
evidence of three different kinds, and this module provides the machinery for
all three:

1. **Algebraic structure.** An operator that should be Hermitian must be
   Hermitian to round-off, a scattering matrix that should be unitary must be
   unitary, and operators that should commute must commute. These are exact
   statements, so they are checked at the `1e-12` level rather than fitted.

2. **Limiting cases.** The numerical result must reproduce a closed form where
   one exists: the high-temperature limit of a partition function, the
   gyrofrequency of a charged particle, the singlet energy of two spins.

3. **Convergence order.** Where no closed form exists, the error must shrink at
   the rate the discretisation promises. A second-order integrator whose error
   falls like the first power of the step size is wrong even when its answer
   looks plausible, and :func:`convergence_order` measures that exponent.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "relative_error", "max_abs_error", "ConvergenceStudy", "convergence_order",
    "run_convergence_study", "is_hermitian", "is_unitary", "commutator",
    "commutes", "check_hermitian", "check_unitary", "check_commutes",
    "spectral_error",
]

# Tolerance for statements that hold exactly in exact arithmetic. Chosen as a
# few hundred times the double-precision epsilon (2.2e-16), which leaves room
# for the O(n) accumulation in a matrix product of the sizes used here.
EXACT_TOL = 1e-12


# ---------------------------------------------------------------------------
# error metrics
# ---------------------------------------------------------------------------
def relative_error(got, want, *, floor: float = 0.0) -> float:
    """Max relative error, scaled by ``want`` where that is meaningful.

    ``floor`` guards the denominator: pass the natural scale of the problem so
    a reference value that happens to be near zero does not inflate the error
    without bound.
    """
    got_arr = np.asarray(got, dtype=complex if np.iscomplexobj(got) else float)
    want_arr = np.asarray(want, dtype=complex if np.iscomplexobj(want) else float)
    if got_arr.shape != want_arr.shape:
        raise ValueError(f"shape mismatch: {got_arr.shape} vs {want_arr.shape}")
    scale = np.maximum(np.abs(want_arr), floor)
    scale = np.where(scale == 0.0, 1.0, scale)
    return float(np.max(np.abs(got_arr - want_arr) / scale))


def max_abs_error(got, want) -> float:
    """Max absolute error."""
    got_arr = np.asarray(got)
    want_arr = np.asarray(want)
    if got_arr.shape != want_arr.shape:
        raise ValueError(f"shape mismatch: {got_arr.shape} vs {want_arr.shape}")
    return float(np.max(np.abs(got_arr - want_arr)))


def spectral_error(got_eigenvalues, want_eigenvalues) -> float:
    """Max absolute error between two spectra, after sorting both.

    Eigenvalue routines make no promise about ordering, so comparing spectra
    without sorting measures the solver's output order rather than its accuracy.
    """
    got = np.sort(np.real_if_close(np.asarray(got_eigenvalues)))
    want = np.sort(np.real_if_close(np.asarray(want_eigenvalues)))
    if got.shape != want.shape:
        raise ValueError(
            f"spectra have different sizes: {got.shape} vs {want.shape}"
        )
    return float(np.max(np.abs(got - want)))


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ConvergenceStudy:
    """Errors measured at a sequence of resolutions, and the fitted order."""

    steps: tuple[float, ...]
    errors: tuple[float, ...]
    order: float
    r_squared: float
    expected_order: float | None = None

    @property
    def converged(self) -> bool:
        """True when the error is already at round-off, so the order is moot."""
        return min(self.errors) < 1e-13

    def matches_expected(self, tol: float = 0.25) -> bool:
        if self.expected_order is None:
            return True
        return self.converged or abs(self.order - self.expected_order) <= tol

    def table(self) -> str:
        """A markdown table of step, error and the ratio between consecutive rows."""
        lines = ["| step | error | ratio | implied order |",
                 "|---|---|---|---|"]
        prev = None
        for step, err in zip(self.steps, self.errors):
            if prev is None:
                lines.append(f"| {step:.4g} | {err:.4g} | — | — |")
            else:
                ratio = prev[1] / err if err > 0 else float("inf")
                implied = np.log(ratio) / np.log(prev[0] / step) if err > 0 else float("nan")
                lines.append(f"| {step:.4g} | {err:.4g} | {ratio:.2f} | {implied:.2f} |")
            prev = (step, err)
        lines.append("")
        lines.append(f"fitted order: **{self.order:.3f}** (R² = {self.r_squared:.4f})")
        if self.expected_order is not None:
            lines.append(f"expected order: {self.expected_order:g}")
        return "\n".join(lines)


def convergence_order(steps: Sequence[float], errors: Sequence[float],
                      *, expected: float | None = None) -> ConvergenceStudy:
    """Fit ``error ~ C * step**p`` by least squares in log-log space.

    Points already at round-off are dropped before fitting: once the
    discretisation error falls below the floating-point noise floor the slope
    measures the noise, not the method. If fewer than three usable points
    remain the study is reported as converged rather than fitted.
    """
    steps_arr = np.asarray(steps, dtype=float)
    errors_arr = np.asarray(errors, dtype=float)
    if steps_arr.shape != errors_arr.shape:
        raise ValueError("steps and errors must have the same length")
    if steps_arr.size < 3:
        raise ValueError(f"need at least 3 resolutions to fit an order, got {steps_arr.size}")
    if np.any(steps_arr <= 0.0):
        raise ValueError("step sizes must be positive")

    usable = errors_arr > 1e-13
    if usable.sum() < 3:
        return ConvergenceStudy(
            steps=tuple(steps_arr.tolist()), errors=tuple(errors_arr.tolist()),
            order=float("nan"), r_squared=float("nan"), expected_order=expected,
        )

    log_h = np.log(steps_arr[usable])
    log_e = np.log(errors_arr[usable])
    slope, intercept = np.polyfit(log_h, log_e, 1)

    predicted = slope * log_h + intercept
    ss_res = float(np.sum((log_e - predicted) ** 2))
    ss_tot = float(np.sum((log_e - log_e.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return ConvergenceStudy(
        steps=tuple(steps_arr.tolist()), errors=tuple(errors_arr.tolist()),
        order=float(slope), r_squared=float(r_squared), expected_order=expected,
    )


def run_convergence_study(solve: Callable[[int], float], reference: float,
                          n_steps: Sequence[int], *, span: float = 1.0,
                          expected: float | None = None) -> ConvergenceStudy:
    """Call ``solve(n)`` at each resolution and fit the order of its error.

    ``solve`` returns a scalar result for ``n`` steps; ``span / n`` is used as
    the step size, so the fitted exponent is in terms of ``h``.
    """
    steps, errors = [], []
    for n in n_steps:
        value = solve(int(n))
        steps.append(span / n)
        denom = abs(reference) if reference != 0.0 else 1.0
        errors.append(abs(value - reference) / denom)
    return convergence_order(steps, errors, expected=expected)


# ---------------------------------------------------------------------------
# algebraic property checks
# ---------------------------------------------------------------------------
def is_hermitian(matrix, tol: float = EXACT_TOL) -> bool:
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    return bool(np.max(np.abs(m - m.conj().T)) <= tol)


def is_unitary(matrix, tol: float = EXACT_TOL) -> bool:
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    return bool(np.max(np.abs(m.conj().T @ m - np.eye(m.shape[0]))) <= tol)


def commutator(a, b) -> np.ndarray:
    """``[A, B] = AB - BA``."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    return a_arr @ b_arr - b_arr @ a_arr


def commutes(a, b, tol: float = EXACT_TOL) -> bool:
    return bool(np.max(np.abs(commutator(a, b))) <= tol)


def check_hermitian(matrix, *, name: str = "matrix", tol: float = EXACT_TOL) -> float:
    """Raise unless ``matrix`` is Hermitian; return the measured deviation."""
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise AssertionError(f"{name}: not square, shape {m.shape}")
    deviation = float(np.max(np.abs(m - m.conj().T)))
    if deviation > tol:
        raise AssertionError(
            f"{name}: not Hermitian, max |H - H†| = {deviation:.3e} > {tol:.0e}"
        )
    return deviation


def check_unitary(matrix, *, name: str = "matrix", tol: float = EXACT_TOL) -> float:
    """Raise unless ``matrix`` is unitary; return the measured deviation."""
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise AssertionError(f"{name}: not square, shape {m.shape}")
    deviation = float(np.max(np.abs(m.conj().T @ m - np.eye(m.shape[0]))))
    if deviation > tol:
        raise AssertionError(
            f"{name}: not unitary, max |U†U - I| = {deviation:.3e} > {tol:.0e}"
        )
    return deviation


def check_commutes(a, b, *, name: str = "[A, B]", tol: float = EXACT_TOL) -> float:
    """Raise unless ``a`` and ``b`` commute; return the measured norm."""
    deviation = float(np.max(np.abs(commutator(a, b))))
    if deviation > tol:
        raise AssertionError(
            f"{name}: does not vanish, max |[A, B]| = {deviation:.3e} > {tol:.0e}"
        )
    return deviation
