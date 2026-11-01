r"""Statistical physics of small spin systems: partition functions and observables.

Conventions
-----------
``k_B = 1``, so ``beta = 1/T``. The Ising energy is

.. math::
    E(\{s\}) = -J \sum_{\langle ij \rangle} s_i s_j - h \sum_i s_i ,
    \qquad s_i = \pm 1,

so positive ``J`` is ferromagnetic and positive ``h`` favours ``s_i = +1``.

Two independent routes to the same numbers
------------------------------------------
Everything here is computed twice, by methods that share no code:

* **Brute-force enumeration.** All :math:`2^N` configurations, their energies
  and magnetisations. Exact for any ``N`` small enough to enumerate, any
  boundary condition, and any field.
* **Transfer matrix.** For the periodic 1-D chain,
  :math:`Z = \operatorname{tr} T^N = \lambda_+^N + \lambda_-^N` with

  .. math::
      T_{ss'} = \exp\!\left[\beta J s s' + \tfrac{1}{2}\beta h (s + s')\right],
      \qquad
      \lambda_\pm = e^{\beta J}\cosh\beta h
          \pm \sqrt{e^{2\beta J}\sinh^2\!\beta h + e^{-2\beta J}} .

  Cost is independent of ``N``, so it also reaches sizes enumeration cannot.

Agreement between the two is the primary verification; the limiting cases
below are the secondary one.

Numerical stability
-------------------
:math:`\ln Z` is accumulated with the log-sum-exp shift

.. math::
    \ln \sum_i e^{-\beta E_i}
        = -\beta E_{\min} + \ln \sum_i e^{-\beta (E_i - E_{\min})},

so no term exceeds 1 before summation. Without the shift, ``beta = 50`` on a
system with ``E_min = -20`` overflows a double at ``exp(1000)``. The same shift
is applied inside every observable.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np

__all__ = [
    "Observables", "log_partition_function", "partition_function",
    "boltzmann_weights", "free_energy", "internal_energy", "heat_capacity",
    "entropy", "observables", "ising_configurations", "ising_spectrum",
    "ising_log_z_transfer", "ising_transfer_eigenvalues", "magnetization",
    "heat_capacity_from_derivative",
]


# ---------------------------------------------------------------------------
# generic ensemble machinery
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Observables:
    """Thermodynamic observables at one temperature."""

    beta: float
    log_z: float
    free_energy: float          # F = -ln(Z) / beta
    internal_energy: float      # U = <E>
    heat_capacity: float        # C = beta**2 (<E^2> - <E>^2)
    entropy: float              # S = beta (U - F)
    magnetization: float | None = None   # per spin, if provided

    @property
    def temperature(self) -> float:
        return 1.0 / self.beta

    def as_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["temperature"] = self.temperature
        return data


def _as_energies(energies) -> np.ndarray:
    arr = np.asarray(energies, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("energy spectrum is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("energy spectrum contains non-finite entries")
    return arr


def _check_beta(beta: float) -> float:
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"beta must be finite and positive, got {beta}")
    return beta


def log_partition_function(energies, beta: float, *,
                           degeneracies=None) -> float:
    r""":math:`\ln Z` by log-sum-exp, shifted by the ground-state energy."""
    e = _as_energies(energies)
    beta = _check_beta(beta)
    g = np.ones_like(e) if degeneracies is None else np.asarray(degeneracies, float).ravel()
    if g.shape != e.shape:
        raise ValueError(f"{g.size} degeneracies for {e.size} energies")
    if np.any(g <= 0.0):
        raise ValueError("degeneracies must be positive")

    e_min = float(e.min())
    shifted = np.exp(-beta * (e - e_min))
    return float(-beta * e_min + math.log(float(np.dot(g, shifted))))


def partition_function(energies, beta: float, *, degeneracies=None) -> float:
    r""":math:`Z`. Overflows for large ``beta``; prefer :func:`log_partition_function`."""
    return math.exp(log_partition_function(energies, beta, degeneracies=degeneracies))


def boltzmann_weights(energies, beta: float, *, degeneracies=None) -> np.ndarray:
    r"""Normalised probabilities :math:`p_i = g_i e^{-\beta E_i} / Z`."""
    e = _as_energies(energies)
    beta = _check_beta(beta)
    g = np.ones_like(e) if degeneracies is None else np.asarray(degeneracies, float).ravel()
    weights = g * np.exp(-beta * (e - e.min()))
    return weights / weights.sum()


def free_energy(energies, beta: float, *, degeneracies=None) -> float:
    r""":math:`F = -\ln Z / \beta`."""
    return -log_partition_function(energies, beta, degeneracies=degeneracies) / _check_beta(beta)


def internal_energy(energies, beta: float, *, degeneracies=None) -> float:
    r""":math:`U = \langle E \rangle`."""
    e = _as_energies(energies)
    p = boltzmann_weights(e, beta, degeneracies=degeneracies)
    return float(np.dot(p, e))


def heat_capacity(energies, beta: float, *, degeneracies=None) -> float:
    r""":math:`C = \beta^2 \left(\langle E^2 \rangle - \langle E \rangle^2\right)`.

    The variance is taken about the mean rather than as
    ``<E^2> - <E>^2``: at low temperature the two moments agree to many digits
    and their difference loses precision, exactly as in the one-pass variance
    formula. Computing ``<(E - U)^2>`` keeps every term on the scale of the
    fluctuation.
    """
    e = _as_energies(energies)
    beta = _check_beta(beta)
    p = boltzmann_weights(e, beta, degeneracies=degeneracies)
    u = float(np.dot(p, e))
    variance = float(np.dot(p, (e - u) ** 2))
    return beta * beta * variance


def entropy(energies, beta: float, *, degeneracies=None) -> float:
    r""":math:`S = \beta (U - F)`, equivalently :math:`-\sum_i p_i \ln p_i`."""
    beta = _check_beta(beta)
    u = internal_energy(energies, beta, degeneracies=degeneracies)
    f = free_energy(energies, beta, degeneracies=degeneracies)
    return beta * (u - f)


def magnetization(energies, magnetizations, beta: float) -> float:
    r"""Thermal average of ``magnetizations`` (already per spin)."""
    e = _as_energies(energies)
    m = np.asarray(magnetizations, dtype=float).ravel()
    if m.shape != e.shape:
        raise ValueError(f"{m.size} magnetisations for {e.size} energies")
    return float(np.dot(boltzmann_weights(e, beta), m))


def observables(energies, beta: float, *, magnetizations=None,
                degeneracies=None) -> Observables:
    """Every observable at one temperature, from one spectrum."""
    beta = _check_beta(beta)
    log_z = log_partition_function(energies, beta, degeneracies=degeneracies)
    u = internal_energy(energies, beta, degeneracies=degeneracies)
    f = -log_z / beta
    return Observables(
        beta=beta, log_z=log_z, free_energy=f, internal_energy=u,
        heat_capacity=heat_capacity(energies, beta, degeneracies=degeneracies),
        entropy=beta * (u - f),
        magnetization=None if magnetizations is None
        else magnetization(energies, magnetizations, beta),
    )


def heat_capacity_from_derivative(energies, beta: float, *, h: float = 1e-4,
                                  degeneracies=None) -> float:
    r""":math:`C = \mathrm{d}U/\mathrm{d}T` by a centred difference in ``T``.

    An independent route to :func:`heat_capacity`, which uses the fluctuation
    formula instead. The two agreeing is a non-trivial check: the fluctuation
    identity :math:`C = \beta^2 \operatorname{Var}(E)` follows from
    differentiating :math:`\ln Z` twice, so a mismatch means one of the two
    derivations is implemented wrongly.
    """
    beta = _check_beta(beta)
    temperature = 1.0 / beta
    step = h * max(temperature, 1.0)
    u_plus = internal_energy(energies, 1.0 / (temperature + step), degeneracies=degeneracies)
    u_minus = internal_energy(energies, 1.0 / (temperature - step), degeneracies=degeneracies)
    return (u_plus - u_minus) / (2.0 * step)


# ---------------------------------------------------------------------------
# Ising chain: brute force
# ---------------------------------------------------------------------------
def ising_configurations(n_sites: int) -> np.ndarray:
    """All :math:`2^N` spin configurations as a ``(2**N, N)`` array of +-1."""
    if n_sites < 1:
        raise ValueError(f"need at least one site, got {n_sites}")
    if n_sites > 22:
        raise ValueError(
            f"enumerating {n_sites} sites means 2**{n_sites} configurations; "
            "use ising_log_z_transfer instead"
        )
    return np.array(list(itertools.product((1, -1), repeat=n_sites)), dtype=float)


def ising_spectrum(n_sites: int, *, coupling: float = 1.0, field: float = 0.0,
                   periodic: bool = True) -> tuple[np.ndarray, np.ndarray]:
    r"""Energies and per-spin magnetisations of every configuration.

    .. math::
        E = -J \sum_{\langle ij \rangle} s_i s_j - h \sum_i s_i

    With ``periodic=True`` the chain is a ring with exactly ``N`` bonds. At
    ``N = 2`` the two sites are therefore bonded twice; that is the convention
    :math:`Z = \operatorname{tr} T^N` implies, and the two methods agree only
    if both use it.
    """
    spins = ising_configurations(n_sites)

    bond_sum = np.zeros(spins.shape[0])
    if n_sites > 1:
        bond_sum += np.sum(spins[:, :-1] * spins[:, 1:], axis=1)
        if periodic:
            # The ring has exactly N bonds, i -> i+1 mod N. At N = 2 that means
            # sites 0 and 1 are bonded *twice*, which looks like double counting
            # but is what `tr T**N` computes: the transfer matrix has no way to
            # express a 2-site ring with one bond. Matching the convention here
            # is what makes the two methods agree at every N; see
            # docs/02-statistical-physics.md.
            bond_sum += spins[:, -1] * spins[:, 0]

    energies = -coupling * bond_sum - field * np.sum(spins, axis=1)
    return energies, np.mean(spins, axis=1)


# ---------------------------------------------------------------------------
# Ising chain: transfer matrix
# ---------------------------------------------------------------------------
def ising_transfer_eigenvalues(beta: float, *, coupling: float = 1.0,
                               field: float = 0.0) -> tuple[float, float]:
    r"""The two eigenvalues :math:`\lambda_\pm` of the symmetric transfer matrix."""
    beta = _check_beta(beta)
    bj = beta * coupling
    bh = beta * field
    centre = math.exp(bj) * math.cosh(bh)
    radical = math.sqrt(math.exp(2.0 * bj) * math.sinh(bh) ** 2 + math.exp(-2.0 * bj))
    return centre + radical, centre - radical


def ising_log_z_transfer(n_sites: int, beta: float, *, coupling: float = 1.0,
                         field: float = 0.0) -> float:
    r""":math:`\ln Z` of the **periodic** chain by transfer matrix.

    .. math:: Z = \lambda_+^N + \lambda_-^N

    Evaluated by binary exponentiation of the positive transfer matrix in the
    log domain. This avoids both overflow and cancellation between opposite-sign
    eigenvalues of antiferromagnetic odd rings at low temperature.
    """
    if n_sites < 2:
        raise ValueError(f"the transfer matrix needs at least 2 sites, got {n_sites}")
    beta = _check_beta(beta)
    transfer = beta * np.array([[coupling + field, -coupling],
                                [-coupling, coupling - field]], dtype=float)
    product = np.array([[0.0, -np.inf], [-np.inf, 0.0]])

    def multiply(left, right):
        return np.logaddexp.reduce(left[:, :, None] + right[None, :, :], axis=1)

    power = n_sites
    while power:
        if power % 2:
            product = multiply(product, transfer)
        power //= 2
        if power:
            transfer = multiply(transfer, transfer)
    return float(np.logaddexp(product[0, 0], product[1, 1]))
