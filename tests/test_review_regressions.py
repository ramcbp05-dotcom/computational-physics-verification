"""Scientific counterexamples that must remain correct."""
import numpy as np
import pytest
from scipy.linalg import expm
from cpv.quantum_spin import PAULI, evolve
from cpv.statistical import ising_spectrum, log_partition_function, ising_log_z_transfer


@pytest.mark.parametrize("dimension", [2, 3, 5])
def test_complex_hermitian_evolution_matches_matrix_exponential(dimension):
    rng = np.random.default_rng(dimension)
    a = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    h = a + a.conj().T
    psi = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    psi /= np.linalg.norm(psi)
    times = np.array([0.0, -0.3, 0.7, 2.0])
    expected = np.array([expm(-1j * t * h) @ psi for t in times])
    np.testing.assert_allclose(evolve(h, psi, times), expected, atol=2e-14, rtol=2e-14)


def test_pauli_y_preserves_initial_state_at_zero_time():
    psi = np.ones(2) / np.sqrt(2)
    np.testing.assert_allclose(evolve(PAULI["y"], psi, [0])[0], psi, atol=1e-14)


@pytest.mark.parametrize("n", [2, 3, 5, 8])
@pytest.mark.parametrize("beta", [20.0, 100.0, 400.0])
@pytest.mark.parametrize("coupling,field", [(-1.0, 0.0), (-1.0, 0.3), (1.0, 0.0), (1.0, -0.3)])
def test_transfer_log_partition_at_low_temperature(n, beta, coupling, field):
    energies, _ = ising_spectrum(n, coupling=coupling, field=field)
    expected = log_partition_function(energies, beta)
    actual = ising_log_z_transfer(n, beta, coupling=coupling, field=field)
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-14, abs=2e-12)
