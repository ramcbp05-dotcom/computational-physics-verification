"""Computational physics modules, each one verified rather than merely plotted.

Four physics modules:

``quantum_spin``   spin operators, Heisenberg Hamiltonians, eigenvalues,
                   state probabilities and unitary time evolution
``statistical``    partition functions and thermodynamic observables of small
                   spin systems, by brute-force enumeration and by transfer
                   matrix
``photonics``      beam splitters, phase shifters, Mach-Zehnder interferometers
                   and two-photon interference, as matrix calculations
``trajectories``   charged-particle motion in electromagnetic fields, and the
                   gravitational two-body problem

One support module:

``verification``   convergence-order fitting, error metrics and the algebraic
                   property checks (hermiticity, unitarity, commutators) that
                   every physics module is tested against

Units
-----
All modules work in natural units: hbar = 1, k_B = 1, and the gravitational
constant times the total mass GM = 1 unless a function takes it as an
argument. Each module's docstring states its own conventions.
"""

from . import photonics, quantum_spin, statistical, trajectories, verification

__version__ = "0.1.0"
__all__ = ["quantum_spin", "statistical", "photonics", "trajectories", "verification"]
