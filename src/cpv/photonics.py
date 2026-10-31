r"""Linear optics as matrix calculations: beam splitters, phases, interference.

Conventions
-----------
A two-mode state is a complex amplitude vector :math:`(a_1, a_2)`, and every
component is a 2x2 unitary acting on it. Intensity is :math:`|a_i|^2`, so a
lossless component conserves :math:`\sum_i |a_i|^2` exactly -- that is the same
statement as :math:`U^\dagger U = \mathbb{1}`, which is why unitarity is the
first thing checked for every matrix this module produces.

The symmetric lossless beam splitter is

.. math::
    B(\theta, \varphi) = \begin{pmatrix}
        \cos\theta & i e^{-i\varphi}\sin\theta \\
        i e^{i\varphi}\sin\theta & \cos\theta
    \end{pmatrix},

with transmissivity :math:`T = \cos^2\theta` and reflectivity
:math:`R = \sin^2\theta`. The factor of :math:`i` on reflection is the phase a
lossless symmetric splitter is obliged to have: unitarity forces the
reflection and transmission phases to differ by :math:`\pi/2`. At
:math:`\theta = \pi/4` this is the familiar
:math:`\frac{1}{\sqrt2}\begin{pmatrix}1 & i\\ i & 1\end{pmatrix}`.

Cascades
--------
:func:`cascade` multiplies in **optical order**: the first argument is the
component the light meets first, so ``cascade(A, B)`` is the matrix ``B @ A``.
Getting this backwards is a silent error for symmetric components and a visible
one for a phase shifter, which is why the convention is stated here and tested.

Two-photon interference
-----------------------
With creation operators transforming as
:math:`a_j^\dagger \to \sum_i U_{ij} c_i^\dagger`, the input
:math:`a_1^\dagger a_2^\dagger |0\rangle` has output amplitudes

.. math::
    A(2,0) = \sqrt{2}\,U_{11}U_{12}, \qquad
    A(1,1) = U_{11}U_{22} + U_{21}U_{12}, \qquad
    A(0,2) = \sqrt{2}\,U_{21}U_{22},

where :math:`A(1,1)` is the permanent of ``U``. For the balanced splitter the
permanent is :math:`(1 \cdot 1 + i \cdot i)/2 = 0`: the Hong-Ou-Mandel dip, and
an **exact** zero rather than a small number, which makes it a sharp test of
the phase convention.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = [
    "beam_splitter", "phase_shifter", "mach_zehnder", "cascade", "propagate",
    "intensities", "visibility", "mach_zehnder_scan", "TwoPhotonOutput",
    "two_photon_output", "splitting_ratio", "two_beam_visibility",
]


def beam_splitter(theta: float = math.pi / 4, phi: float = 0.0) -> np.ndarray:
    r"""Symmetric lossless beam splitter :math:`B(\theta, \varphi)`.

    ``theta = pi/4`` is 50:50. Reflectivity is :math:`\sin^2\theta`.
    """
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [c, 1.0j * np.exp(-1.0j * phi) * s],
        [1.0j * np.exp(1.0j * phi) * s, c],
    ], dtype=complex)


def splitting_ratio(theta: float) -> tuple[float, float]:
    r"""``(transmissivity, reflectivity)`` = :math:`(\cos^2\theta, \sin^2\theta)`."""
    return math.cos(theta) ** 2, math.sin(theta) ** 2


def phase_shifter(phi: float, mode: int = 0) -> np.ndarray:
    r"""Phase :math:`e^{i\varphi}` on one mode, identity on the other."""
    if mode not in (0, 1):
        raise ValueError(f"mode must be 0 or 1, got {mode}")
    diagonal = [1.0 + 0.0j, 1.0 + 0.0j]
    diagonal[mode] = np.exp(1.0j * phi)
    return np.diag(diagonal)


def cascade(*components) -> np.ndarray:
    """Matrix of a chain of components, given in the order the light meets them.

    ``cascade(A, B, C)`` is ``C @ B @ A``.
    """
    if not components:
        raise ValueError("cascade needs at least one component")
    out = np.eye(2, dtype=complex)
    for component in components:
        matrix = np.asarray(component, dtype=complex)
        if matrix.shape != (2, 2):
            raise ValueError(f"component has shape {matrix.shape}, expected (2, 2)")
        out = matrix @ out
    return out


def mach_zehnder(phi: float, *, theta_in: float = math.pi / 4,
                 theta_out: float = math.pi / 4, arm: int = 0) -> np.ndarray:
    r"""Mach-Zehnder interferometer: splitter, phase in one arm, splitter.

    With both splitters balanced and the input in mode 0, the output
    intensities are :math:`(\sin^2(\varphi/2), \cos^2(\varphi/2))` -- derived
    in ``docs/03-photonics.md`` and checked against this matrix in the tests.
    """
    return cascade(beam_splitter(theta_in), phase_shifter(phi, mode=arm),
                   beam_splitter(theta_out))


def propagate(matrix, state) -> np.ndarray:
    """Apply a component matrix to an amplitude vector."""
    u = np.asarray(matrix, dtype=complex)
    a = np.asarray(state, dtype=complex).ravel()
    if u.shape != (2, 2) or a.size != 2:
        raise ValueError(f"expected a (2, 2) matrix and a 2-vector, got {u.shape} and {a.shape}")
    return u @ a


def intensities(state) -> np.ndarray:
    r""":math:`(|a_1|^2, |a_2|^2)`."""
    a = np.asarray(state, dtype=complex).ravel()
    if a.size != 2:
        raise ValueError(f"expected a 2-vector, got size {a.size}")
    return np.abs(a) ** 2


def visibility(values) -> float:
    r"""Fringe visibility :math:`(I_{\max} - I_{\min}) / (I_{\max} + I_{\min})`."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError("visibility needs at least two samples of the fringe")
    if np.any(arr < -1e-12):
        raise ValueError("intensities must be non-negative")
    total = float(arr.max() + arr.min())
    if total <= 0.0:
        raise ValueError("fringe is identically zero; visibility is undefined")
    return float((arr.max() - arr.min()) / total)


def two_beam_visibility(i1: float, i2: float) -> float:
    r"""Visibility of two-beam interference, :math:`2\sqrt{I_1 I_2}/(I_1 + I_2)`.

    The closed form the matrix calculation is checked against: equal arms give
    :math:`V = 1`, and any imbalance reduces it.
    """
    if i1 < 0.0 or i2 < 0.0:
        raise ValueError("intensities must be non-negative")
    if i1 + i2 == 0.0:
        raise ValueError("both arms are dark; visibility is undefined")
    return 2.0 * math.sqrt(i1 * i2) / (i1 + i2)


def mach_zehnder_scan(phases, *, theta_in: float = math.pi / 4,
                      theta_out: float = math.pi / 4,
                      state=(1.0, 0.0), arm: int = 0) -> np.ndarray:
    """Output intensities over a phase scan. Returns shape ``(n_phases, 2)``."""
    phases_arr = np.atleast_1d(np.asarray(phases, dtype=float))
    out = np.empty((phases_arr.size, 2))
    for k, phi in enumerate(phases_arr):
        matrix = mach_zehnder(float(phi), theta_in=theta_in, theta_out=theta_out, arm=arm)
        out[k] = intensities(propagate(matrix, state))
    return out


@dataclasses.dataclass(frozen=True)
class TwoPhotonOutput:
    """Output distribution for one photon into each input mode."""

    p_20: float
    p_11: float
    p_02: float

    @property
    def total(self) -> float:
        return self.p_20 + self.p_11 + self.p_02

    @property
    def coincidence(self) -> float:
        """Probability of one photon in each output -- the HOM observable."""
        return self.p_11

    def as_dict(self) -> dict:
        return {"p_20": self.p_20, "p_11": self.p_11, "p_02": self.p_02,
                "total": self.total}


def two_photon_output(matrix) -> TwoPhotonOutput:
    r"""Two-photon output distribution for the input :math:`|1,1\rangle`.

    Uses the permanent/determinant amplitudes given in the module docstring.
    For a unitary ``matrix`` the three probabilities sum to 1, which the tests
    check independently of the individual values.
    """
    u = np.asarray(matrix, dtype=complex)
    if u.shape != (2, 2):
        raise ValueError(f"expected a (2, 2) matrix, got {u.shape}")

    a_20 = math.sqrt(2.0) * u[0, 0] * u[0, 1]
    a_11 = u[0, 0] * u[1, 1] + u[1, 0] * u[0, 1]      # permanent of u
    a_02 = math.sqrt(2.0) * u[1, 0] * u[1, 1]

    return TwoPhotonOutput(
        p_20=float(abs(a_20) ** 2),
        p_11=float(abs(a_11) ** 2),
        p_02=float(abs(a_02) ** 2),
    )
