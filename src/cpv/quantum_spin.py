r"""Quantum spin systems: operators, Hamiltonians, spectra and time evolution.

Conventions
-----------
``hbar = 1`` throughout, so spin operators carry units of angular momentum
divided by hbar and the spin-1/2 operators are half the Pauli matrices.

The basis for an ``n``-site chain is the product basis ordered so that site 0
is the **most significant** factor in the Kronecker product:

.. math::
    |m_0 m_1 \ldots m_{n-1}\rangle
        = |m_0\rangle \otimes |m_1\rangle \otimes \cdots

For spin-1/2 that makes basis index ``k`` the bit pattern of ``k`` with bit
``n-1-i`` holding site ``i``, and ``0`` meaning spin up. ``basis_labels``
returns the strings, so nothing depends on remembering this.

Operator algebra
----------------
The spin-``s`` matrices are built from the ladder operators,

.. math::
    S^\pm |s, m\rangle = \hbar\sqrt{s(s+1) - m(m\pm 1)}\,|s, m\pm 1\rangle,

with :math:`S^x = (S^+ + S^-)/2` and :math:`S^y = (S^+ - S^-)/2i`. Building
:math:`S^x` and :math:`S^y` this way rather than writing them out keeps the
commutation relations :math:`[S^x, S^y] = i S^z` exact by construction, which
:func:`cpv.verification.check_commutes` then confirms to round-off.

Hamiltonian
-----------
:func:`heisenberg_chain` builds

.. math::
    H = J \sum_{\langle ij\rangle}
            \left(\Delta_x S^x_i S^x_j + \Delta_y S^y_i S^y_j
                  + \Delta_z S^z_i S^z_j\right)
        - h \sum_i S^z_i

with ``anisotropy = (1, 1, 1)`` giving the isotropic Heisenberg model and
``(0, 0, 1)`` the Ising model. The field enters with a minus sign so that a
positive ``h`` lowers the energy of the spin-up state.
"""

from __future__ import annotations

import functools
from typing import Iterable, Sequence

import numpy as np

from .verification import EXACT_TOL

__all__ = [
    "PAULI", "spin_operators", "spin_dimension", "embed", "basis_labels",
    "heisenberg_chain", "total_spin_operators", "total_spin_squared",
    "eigensystem", "ground_state", "expectation", "state_probabilities",
    "evolve", "thermal_density_matrix", "magnetization_operator",
]

PAULI = {
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    "i": np.eye(2, dtype=complex),
}


def spin_dimension(s: float) -> int:
    """Hilbert-space dimension ``2s + 1`` for a single spin-``s``."""
    dim = 2.0 * s + 1.0
    if abs(dim - round(dim)) > 1e-12 or round(dim) < 2:
        raise ValueError(f"spin must be a positive half-integer, got s={s}")
    return int(round(dim))


@functools.lru_cache(maxsize=16)
def spin_operators(s: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return ``(Sx, Sy, Sz)`` for a single spin-``s``, in units of hbar.

    Built from the ladder operators so that :math:`[S^x, S^y] = i S^z` and
    :math:`S^2 = s(s+1)\mathbb{1}` hold to round-off rather than by assertion.
    """
    dim = spin_dimension(s)
    m = np.array([s - k for k in range(dim)])  # descending: m = s, s-1, ..., -s

    sz = np.diag(m).astype(complex)

    # S+ |s, m> = sqrt(s(s+1) - m(m+1)) |s, m+1>; |s, m+1> sits one row up.
    s_plus = np.zeros((dim, dim), dtype=complex)
    for k in range(1, dim):
        mk = m[k]
        s_plus[k - 1, k] = np.sqrt(s * (s + 1.0) - mk * (mk + 1.0))
    s_minus = s_plus.conj().T

    sx = 0.5 * (s_plus + s_minus)
    sy = (s_plus - s_minus) / (2.0j)
    return sx, sy, sz


def _kron_all(matrices: Iterable[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for m in matrices:
        out = np.kron(out, m)
    return out


def embed(operator: np.ndarray, site: int, n_sites: int, s: float = 0.5) -> np.ndarray:
    """Place a single-site ``operator`` at ``site`` of an ``n_sites`` chain.

    Site 0 is the most significant Kronecker factor.
    """
    dim = spin_dimension(s)
    op = np.asarray(operator, dtype=complex)
    if op.shape != (dim, dim):
        raise ValueError(f"operator has shape {op.shape}, expected ({dim}, {dim})")
    if not 0 <= site < n_sites:
        raise IndexError(f"site {site} out of range for a chain of {n_sites}")
    identity = np.eye(dim, dtype=complex)
    return _kron_all(op if i == site else identity for i in range(n_sites))


def basis_labels(n_sites: int, s: float = 0.5) -> list[str]:
    """Product-basis labels in the order the matrices use.

    For spin-1/2 these are strings of ``u``/``d``, site 0 leftmost.
    """
    dim = spin_dimension(s)
    if dim == 2:
        symbols = ["u", "d"]
    else:
        symbols = [f"{s - k:+g}" for k in range(dim)]
    labels = [""]
    for _ in range(n_sites):
        labels = [prefix + sym for prefix in labels for sym in symbols]
    return labels


def heisenberg_chain(n_sites: int, *, coupling: float = 1.0, field: float = 0.0,
                     anisotropy: Sequence[float] = (1.0, 1.0, 1.0),
                     periodic: bool = False, s: float = 0.5) -> np.ndarray:
    r"""Spin chain Hamiltonian.

    .. math::
        H = J \sum_{\langle ij\rangle} \sum_{a} \Delta_a S^a_i S^a_j
            - h \sum_i S^z_i
    """
    if n_sites < 1:
        raise ValueError(f"need at least one site, got {n_sites}")
    if len(anisotropy) != 3:
        raise ValueError(f"anisotropy must have three components, got {len(anisotropy)}")

    dim = spin_dimension(s) ** n_sites
    sx, sy, sz = spin_operators(s)
    single = (sx, sy, sz)

    h_matrix = np.zeros((dim, dim), dtype=complex)

    if periodic and n_sites == 2:
        # On two sites the wrap bond joins the same pair as the open bond, so a
        # ring either has one bond (and equals the open chain) or two (and
        # doubles J). Silently picking one is how a factor of 2 reaches a
        # published number, so the caller is asked to say which they mean.
        raise ValueError(
            "periodic=True is ambiguous for n_sites=2: the wrap bond is the same "
            "pair as the open bond. Use periodic=False, or coupling=2*J to mean "
            "the two-bond ring."
        )

    bonds = [(i, i + 1) for i in range(n_sites - 1)]
    if periodic:
        bonds.append((n_sites - 1, 0))

    for i, j in bonds:
        for delta, op in zip(anisotropy, single):
            if delta == 0.0:
                continue
            h_matrix += coupling * delta * (embed(op, i, n_sites, s)
                                            @ embed(op, j, n_sites, s))

    if field != 0.0:
        for i in range(n_sites):
            h_matrix -= field * embed(sz, i, n_sites, s)

    # The sum of Hermitian products of commuting embedded operators is Hermitian;
    # symmetrising removes the O(1e-17) asymmetry from the matrix products so
    # downstream eigh calls see an exactly Hermitian matrix.
    return 0.5 * (h_matrix + h_matrix.conj().T)


def total_spin_operators(n_sites: int, s: float = 0.5):
    r"""``(Sx_tot, Sy_tot, Sz_tot)``, each the sum over sites."""
    single = spin_operators(s)
    return tuple(
        sum(embed(op, i, n_sites, s) for i in range(n_sites)) for op in single
    )


def total_spin_squared(n_sites: int, s: float = 0.5) -> np.ndarray:
    r""":math:`S^2 = S_x^2 + S_y^2 + S_z^2` for the whole chain."""
    sx, sy, sz = total_spin_operators(n_sites, s)
    out = sx @ sx + sy @ sy + sz @ sz
    return 0.5 * (out + out.conj().T)


def magnetization_operator(n_sites: int, s: float = 0.5) -> np.ndarray:
    r"""Total :math:`S^z` per site, i.e. :math:`\frac{1}{N}\sum_i S^z_i`."""
    _, _, sz_tot = total_spin_operators(n_sites, s)
    return sz_tot / n_sites


def eigensystem(hamiltonian) -> tuple[np.ndarray, np.ndarray]:
    """Ascending eigenvalues and the corresponding eigenvectors as columns.

    Uses ``eigh``, which is only valid for a Hermitian matrix, so hermiticity
    is checked rather than assumed: a non-Hermitian input would otherwise be
    silently symmetrised by the LAPACK driver.
    """
    h_matrix = np.asarray(hamiltonian, dtype=complex)
    deviation = float(np.max(np.abs(h_matrix - h_matrix.conj().T)))
    if deviation > EXACT_TOL:
        raise ValueError(
            f"Hamiltonian is not Hermitian: max |H - H†| = {deviation:.3e}"
        )
    values, vectors = np.linalg.eigh(h_matrix)
    return np.real(values), vectors


def ground_state(hamiltonian) -> tuple[float, np.ndarray]:
    """Lowest eigenvalue and its eigenvector."""
    values, vectors = eigensystem(hamiltonian)
    return float(values[0]), vectors[:, 0]


def expectation(operator, state) -> float | complex:
    r""":math:`\langle \psi | A | \psi \rangle` for a normalised ``state``.

    A real value is returned when the imaginary part is at round-off, which is
    the case for any Hermitian operator, so a complex result is a signal that
    the operator was not Hermitian.
    """
    psi = np.asarray(state, dtype=complex).ravel()
    norm = float(np.linalg.norm(psi))
    if abs(norm - 1.0) > 1e-10:
        raise ValueError(f"state is not normalised: ||psi|| = {norm:.12f}")
    value = complex(psi.conj() @ (np.asarray(operator, dtype=complex) @ psi))
    return value.real if abs(value.imag) < 1e-12 else value


def state_probabilities(state, *, labels: Sequence[str] | None = None,
                        cutoff: float = 0.0) -> dict[str, float] | np.ndarray:
    r"""Basis-state probabilities :math:`|\langle b | \psi \rangle|^2`.

    Returns a plain array when ``labels`` is omitted, and a label-keyed dict
    otherwise, dropping entries below ``cutoff``.
    """
    psi = np.asarray(state, dtype=complex).ravel()
    norm = float(np.linalg.norm(psi))
    if abs(norm - 1.0) > 1e-10:
        raise ValueError(f"state is not normalised: ||psi|| = {norm:.12f}")
    probs = np.abs(psi) ** 2
    if labels is None:
        return probs
    if len(labels) != probs.size:
        raise ValueError(f"{len(labels)} labels for {probs.size} amplitudes")
    return {label: float(p) for label, p in zip(labels, probs) if p >= cutoff}


def evolve(hamiltonian, state, times) -> np.ndarray:
    r"""Exact unitary evolution :math:`|\psi(t)\rangle = e^{-iHt}|\psi(0)\rangle`.

    The propagator is applied through the eigenbasis rather than by a series
    expansion or a time-stepping scheme, so the result is exact to round-off at
    every ``t`` and carries no step-size error at all. Returns shape
    ``(n_times, dim)``.
    """
    values, vectors = eigensystem(hamiltonian)
    psi = np.asarray(state, dtype=complex).ravel()
    if psi.size != values.size:
        raise ValueError(f"state has {psi.size} amplitudes, H is {values.size}-dimensional")

    coefficients = vectors.conj().T @ psi            # amplitudes in the eigenbasis
    t_arr = np.atleast_1d(np.asarray(times, dtype=float))
    phases = np.exp(-1.0j * np.outer(t_arr, values))  # (n_times, dim)
    return (phases * coefficients[None, :]) @ vectors.conj().T


def thermal_density_matrix(hamiltonian, beta: float) -> np.ndarray:
    r""":math:`\rho = e^{-\beta H} / Z`, built in the eigenbasis.

    The energies are shifted by the ground state before exponentiating, which
    is what keeps ``beta * E`` from overflowing at low temperature. The shift
    cancels exactly in the ratio.
    """
    values, vectors = eigensystem(hamiltonian)
    weights = np.exp(-beta * (values - values[0]))
    weights /= weights.sum()
    return (vectors * weights[None, :]) @ vectors.conj().T
