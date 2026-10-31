r"""Verification of the statistical-physics module.

The primary check is **two independent methods agreeing**: brute-force
enumeration of all :math:`2^N` configurations against the transfer-matrix
closed form. They share no code, so agreement to 1e-13 across 384 parameter
combinations is strong evidence both are right.

The secondary checks are limits with known answers (high and low temperature,
zero coupling) and a thermodynamic identity (the fluctuation formula for
:math:`C` against :math:`\mathrm{d}U/\mathrm{d}T`).
"""

import itertools
import math

import numpy as np
import pytest

from cpv import statistical as st

EXACT = 1e-12


class TestEnsembleMachinery:
    def test_two_level_system_closed_form(self):
        r""":math:`Z = e^{\beta\epsilon/2} + e^{-\beta\epsilon/2} = 2\cosh(\beta\epsilon/2)`."""
        eps, beta = 1.3, 0.8
        energies = [-eps / 2, eps / 2]
        want = 2.0 * math.cosh(beta * eps / 2.0)
        assert st.partition_function(energies, beta) == pytest.approx(want, rel=1e-13)
        assert st.log_partition_function(energies, beta) == pytest.approx(
            math.log(want), rel=1e-13
        )

    def test_two_level_internal_energy_closed_form(self):
        r""":math:`U = -\tfrac{\epsilon}{2}\tanh(\beta\epsilon/2)`."""
        eps, beta = 1.3, 0.8
        got = st.internal_energy([-eps / 2, eps / 2], beta)
        want = -0.5 * eps * math.tanh(beta * eps / 2.0)
        assert got == pytest.approx(want, rel=1e-13)

    def test_two_level_heat_capacity_closed_form(self):
        r""":math:`C = (\beta\epsilon/2)^2 \operatorname{sech}^2(\beta\epsilon/2)`."""
        eps, beta = 1.3, 0.8
        x = beta * eps / 2.0
        want = x ** 2 / math.cosh(x) ** 2
        assert st.heat_capacity([-eps / 2, eps / 2], beta) == pytest.approx(want, rel=1e-11)

    def test_degeneracies_are_equivalent_to_repeated_levels(self):
        energies = [0.0, 1.0]
        beta = 0.9
        with_g = st.log_partition_function(energies, beta, degeneracies=[1, 3])
        expanded = st.log_partition_function([0.0, 1.0, 1.0, 1.0], beta)
        assert with_g == pytest.approx(expanded, rel=1e-14)

    def test_weights_normalise_and_order_correctly(self):
        p = st.boltzmann_weights([0.0, 1.0, 2.0], 1.0)
        assert p.sum() == pytest.approx(1.0, abs=1e-14)
        assert p[0] > p[1] > p[2], "lower energy must be more probable"

    def test_entropy_matches_the_shannon_form(self):
        r""":math:`\beta(U - F) = -\sum_i p_i \ln p_i`."""
        energies = np.array([0.0, 0.7, 1.9, 3.3])
        for beta in (0.2, 1.0, 4.0):
            p = st.boltzmann_weights(energies, beta)
            shannon = float(-np.sum(p * np.log(p)))
            assert st.entropy(energies, beta) == pytest.approx(shannon, rel=1e-11), (
                f"beta={beta}: thermodynamic {st.entropy(energies, beta)} vs "
                f"Shannon {shannon}"
            )

    def test_no_overflow_at_very_low_temperature(self):
        r"""Without the ground-state shift, ``exp(50 * 20)`` overflows."""
        energies = np.array([-20.0, -19.5, 0.0, 12.0])
        value = st.log_partition_function(energies, 50.0)
        assert math.isfinite(value), f"lnZ = {value}"
        assert value == pytest.approx(50.0 * 20.0, rel=1e-6), (
            "at beta=50 the ground state dominates, so lnZ -> -beta*E0"
        )

    def test_free_energy_bounds_the_internal_energy(self):
        r""":math:`F = U - TS \le U` because :math:`S \ge 0`."""
        energies = np.array([0.0, 1.0, 2.5, 4.0])
        for beta in (0.1, 1.0, 10.0):
            assert st.free_energy(energies, beta) <= st.internal_energy(energies, beta) + EXACT

    def test_heat_capacity_is_non_negative(self):
        energies = np.array([0.0, 1.0, 2.5, 4.0])
        for beta in (0.05, 0.5, 2.0, 20.0):
            assert st.heat_capacity(energies, beta) >= -EXACT

    def test_invalid_beta_raises(self):
        for bad in (0.0, -1.0, math.inf, math.nan):
            with pytest.raises(ValueError, match="finite and positive"):
                st.log_partition_function([0.0, 1.0], bad)

    def test_empty_spectrum_raises(self):
        with pytest.raises(ValueError, match="empty"):
            st.log_partition_function([], 1.0)


class TestIsingSpectrum:
    @pytest.mark.parametrize("n", [1, 2, 3, 6, 10])
    def test_configuration_count(self, n):
        spins = st.ising_configurations(n)
        assert spins.shape == (2 ** n, n)
        assert set(np.unique(spins)) == {-1.0, 1.0}

    def test_enumeration_refuses_an_impossible_size(self):
        with pytest.raises(ValueError, match="use ising_log_z_transfer"):
            st.ising_configurations(30)

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_energies_match_a_direct_loop(self, n):
        """Independent re-implementation, written as a plain Python loop."""
        coupling, field = 1.3, -0.4
        energies, mags = st.ising_spectrum(n, coupling=coupling, field=field, periodic=True)
        for k, spins in enumerate(itertools.product((1, -1), repeat=n)):
            bonds = sum(spins[i] * spins[(i + 1) % n] for i in range(n))
            want = -coupling * bonds - field * sum(spins)
            assert energies[k] == pytest.approx(want, abs=1e-12), (
                f"config {spins}: got {energies[k]}, expected {want}"
            )
            assert mags[k] == pytest.approx(sum(spins) / n, abs=1e-12)

    def test_ground_state_energy_of_the_ferromagnetic_ring(self):
        r"""All aligned: :math:`E_0 = -JN - h N` with ``N`` bonds."""
        for n in (3, 4, 8):
            energies, _ = st.ising_spectrum(n, coupling=1.0, field=0.5, periodic=True)
            assert energies.min() == pytest.approx(-1.0 * n - 0.5 * n, abs=1e-12)

    def test_open_chain_has_one_bond_fewer(self):
        energies_open, _ = st.ising_spectrum(5, coupling=1.0, periodic=False)
        energies_ring, _ = st.ising_spectrum(5, coupling=1.0, periodic=True)
        assert energies_open.min() == pytest.approx(-4.0, abs=1e-12)
        assert energies_ring.min() == pytest.approx(-5.0, abs=1e-12)

    def test_two_site_ring_is_a_two_bond_ring(self):
        r"""The ring has ``N`` bonds at every ``N``, so at ``N = 2`` the two
        sites are bonded twice. That is what ``tr T**2`` computes, and matching
        it is why the two methods agree at ``N = 2``."""
        energies, _ = st.ising_spectrum(2, coupling=1.0, field=0.0, periodic=True)
        assert sorted(set(energies)) == [-2.0, 2.0], (
            f"got {sorted(set(energies))}; one bond would give [-1, 1]"
        )


class TestTwoIndependentMethods:
    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 8, 10, 12])
    @pytest.mark.parametrize("field", [0.0, 0.25, 1.0, -0.6])
    @pytest.mark.parametrize("coupling", [1.0, -1.0, 0.4])
    @pytest.mark.parametrize("beta", [0.05, 0.4, 1.5, 6.0])
    def test_brute_force_matches_the_transfer_matrix(self, n, field, coupling, beta):
        r""":math:`\ln Z` from :math:`2^N` configurations vs
        :math:`\lambda_+^N + \lambda_-^N`."""
        energies, _ = st.ising_spectrum(n, coupling=coupling, field=field, periodic=True)
        brute = st.log_partition_function(energies, beta)
        transfer = st.ising_log_z_transfer(n, beta, coupling=coupling, field=field)
        assert transfer == pytest.approx(brute, rel=1e-11, abs=1e-11), (
            f"N={n} J={coupling} h={field} beta={beta}: "
            f"brute={brute:.12f} transfer={transfer:.12f}"
        )

    def test_transfer_eigenvalues_satisfy_their_characteristic_equation(self):
        r""":math:`\lambda_\pm` are the roots of
        :math:`\lambda^2 - (\operatorname{tr}T)\lambda + \det T = 0`."""
        for beta, coupling, field in ((0.3, 1.0, 0.4), (2.0, -0.7, 0.0), (1.1, 0.5, -1.2)):
            lam_p, lam_m = st.ising_transfer_eigenvalues(
                beta, coupling=coupling, field=field)
            bj, bh = beta * coupling, beta * field
            trace = math.exp(bj + bh) + math.exp(bj - bh)
            det = math.exp(2 * bj) - math.exp(-2 * bj)
            assert lam_p + lam_m == pytest.approx(trace, rel=1e-12)
            assert lam_p * lam_m == pytest.approx(det, rel=1e-12)

    @pytest.mark.parametrize("n,beta,coupling,field", [
        (200, 5.0, 1.0, 0.0), (200, 0.3, 1.0, 0.4),
        (1000, 2.0, 1.0, 0.0), (64, 1.5, -1.0, 0.7),
    ])
    def test_transfer_matrix_reaches_sizes_enumeration_cannot(self, n, beta,
                                                              coupling, field):
        r"""Checked against an explicit matrix power, not against an asymptote.

        ``N = 200`` would need :math:`2^{200}` configurations to enumerate, and
        :math:`\lambda_+^{200}` overflows a double. The reference here is built
        by scaling the transfer matrix by :math:`\lambda_+` before raising it to
        the ``N``-th power, so :math:`\operatorname{tr}` stays of order 1:

        .. math::
            \ln Z = N \ln\lambda_+ + \ln \operatorname{tr}
                     \left[(T/\lambda_+)^N\right].

        That route uses matrix powers rather than the closed-form eigenvalues,
        so it is independent of the formula under test.
        """
        value = st.ising_log_z_transfer(n, beta, coupling=coupling, field=field)
        assert math.isfinite(value), f"lnZ = {value}"

        bj, bh = beta * coupling, beta * field
        transfer = np.array([
            [math.exp(bj + bh), math.exp(-bj)],
            [math.exp(-bj), math.exp(bj - bh)],
        ])
        lam_p, _ = st.ising_transfer_eigenvalues(beta, coupling=coupling, field=field)
        scaled = np.linalg.matrix_power(transfer / lam_p, n)
        reference = n * math.log(lam_p) + math.log(float(np.trace(scaled)))

        assert value == pytest.approx(reference, rel=1e-11), (
            f"N={n} beta={beta} J={coupling} h={field}: "
            f"closed form {value:.9f} vs matrix power {reference:.9f}"
        )

    def test_low_temperature_asymptote_of_the_long_chain(self):
        r"""At large ``beta`` the ferromagnetic ring approaches
        :math:`\ln Z \to \beta N + \ln 2`, the two-fold degenerate ground
        state. It is an *asymptote*, not an identity: at ``beta = 5``,
        ``N = 200`` the exact value exceeds it by 4.10e-5, which is the
        contribution of the excited band.
        """
        exact = st.ising_log_z_transfer(200, 5.0, coupling=1.0)
        asymptote = 5.0 * 200 + math.log(2.0)
        gap = exact - asymptote
        assert 0.0 < gap < 1e-4, (
            f"exact {exact:.10f} - asymptote {asymptote:.10f} = {gap:.3e}; "
            "the correction must be positive (extra states only add to Z) and small"
        )

        # Raising beta must shrink the gap towards zero.
        tighter = st.ising_log_z_transfer(200, 8.0, coupling=1.0) - (8.0 * 200 + math.log(2.0))
        assert 0.0 < tighter < gap, (
            f"gap at beta=8 is {tighter:.3e}, not smaller than {gap:.3e} at beta=5"
        )

    def test_transfer_matrix_needs_two_sites(self):
        with pytest.raises(ValueError, match="at least 2 sites"):
            st.ising_log_z_transfer(1, 1.0)


class TestLimitingCases:
    @pytest.mark.parametrize("n", [4, 8, 12])
    def test_high_temperature_entropy_is_n_log_two(self, n):
        r"""Every configuration equally likely: :math:`S \to N\ln 2`."""
        energies, _ = st.ising_spectrum(n, coupling=1.0, periodic=True)
        obs = st.observables(energies, 1e-8)
        assert obs.entropy == pytest.approx(n * math.log(2.0), rel=1e-9), (
            f"N={n}: S = {obs.entropy:.10f}, expected {n * math.log(2.0):.10f}"
        )
        assert obs.log_z == pytest.approx(n * math.log(2.0), rel=1e-9)
        assert obs.internal_energy == pytest.approx(0.0, abs=1e-6)
        assert obs.heat_capacity == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("n", [4, 8, 12])
    def test_low_temperature_limits(self, n):
        r""":math:`U \to E_0` and :math:`S \to \ln g_0`; at zero field the
        ferromagnetic ground state is two-fold degenerate, so
        :math:`S \to \ln 2`."""
        energies, _ = st.ising_spectrum(n, coupling=1.0, periodic=True)
        obs = st.observables(energies, 80.0)
        assert obs.internal_energy == pytest.approx(energies.min(), rel=1e-12)
        assert obs.entropy == pytest.approx(math.log(2.0), abs=1e-9), (
            f"N={n}: S = {obs.entropy:.10f}, expected ln 2 = {math.log(2.0):.10f}"
        )
        assert obs.heat_capacity == pytest.approx(0.0, abs=1e-8), (
            "C must vanish as T -> 0 (the third law)"
        )

    def test_a_field_removes_the_ground_state_degeneracy(self):
        energies, _ = st.ising_spectrum(8, coupling=1.0, field=0.4, periodic=True)
        obs = st.observables(energies, 80.0)
        assert obs.entropy == pytest.approx(0.0, abs=1e-9), (
            f"S = {obs.entropy:.3e}; a field makes the ground state unique, so S -> 0"
        )

    @pytest.mark.parametrize("beta", [0.2, 1.0, 4.0])
    @pytest.mark.parametrize("field", [0.3, 0.7, 2.0])
    def test_zero_coupling_gives_the_curie_result(self, beta, field):
        r"""Independent spins: :math:`m = \tanh(\beta h)` exactly."""
        energies, mags = st.ising_spectrum(6, coupling=0.0, field=field)
        got = st.magnetization(energies, mags, beta)
        assert got == pytest.approx(math.tanh(beta * field), rel=1e-12), (
            f"beta={beta} h={field}: m = {got:.12f}, tanh = {math.tanh(beta * field):.12f}"
        )

    def test_zero_coupling_partition_function_factorises(self):
        r""":math:`Z = [2\cosh(\beta h)]^N` for independent spins."""
        n, beta, field = 7, 0.9, 0.55
        energies, _ = st.ising_spectrum(n, coupling=0.0, field=field)
        want = n * math.log(2.0 * math.cosh(beta * field))
        assert st.log_partition_function(energies, beta) == pytest.approx(want, rel=1e-12)

    def test_zero_field_magnetization_vanishes_by_symmetry(self):
        r"""Flipping every spin is an exact symmetry at :math:`h = 0`."""
        energies, mags = st.ising_spectrum(8, coupling=1.0, field=0.0, periodic=True)
        for beta in (0.1, 1.0, 10.0):
            assert st.magnetization(energies, mags, beta) == pytest.approx(0.0, abs=1e-14)


class TestThermodynamicConsistency:
    @pytest.mark.parametrize("beta", [0.3, 1.0, 3.0])
    def test_fluctuation_formula_matches_du_dt(self, beta):
        r""":math:`C = \beta^2\operatorname{Var}(E)` and
        :math:`C = \mathrm{d}U/\mathrm{d}T` are the same quantity derived two
        ways. Measured agreement here is 1e-8 at ``beta = 0.3`` and 1e-6 at
        ``beta = 3``, which is the centred-difference truncation error, not a
        disagreement between the two identities.
        """
        energies, _ = st.ising_spectrum(8, coupling=1.0, field=0.3, periodic=True)
        fluctuation = st.heat_capacity(energies, beta)
        derivative = st.heat_capacity_from_derivative(energies, beta)
        assert derivative == pytest.approx(fluctuation, rel=1e-4, abs=1e-9), (
            f"beta={beta}: fluctuation {fluctuation:.12f} vs dU/dT {derivative:.12f}"
        )

    def test_energy_is_minus_d_lnz_d_beta(self):
        r""":math:`U = -\partial \ln Z / \partial \beta`, by centred difference."""
        energies, _ = st.ising_spectrum(6, coupling=1.0, field=0.2, periodic=True)
        beta, h = 1.0, 1e-5
        numerical = -(st.log_partition_function(energies, beta + h)
                      - st.log_partition_function(energies, beta - h)) / (2 * h)
        assert numerical == pytest.approx(st.internal_energy(energies, beta), rel=1e-8)

    def test_observables_bundle_is_self_consistent(self):
        energies, mags = st.ising_spectrum(6, coupling=1.0, field=0.3, periodic=True)
        obs = st.observables(energies, 1.2, magnetizations=mags)
        assert obs.free_energy == pytest.approx(-obs.log_z / obs.beta, rel=1e-14)
        assert obs.entropy == pytest.approx(
            obs.beta * (obs.internal_energy - obs.free_energy), rel=1e-12
        )
        assert obs.temperature == pytest.approx(1.0 / obs.beta, rel=1e-15)
        assert obs.magnetization is not None

    def test_internal_energy_decreases_as_temperature_falls(self):
        energies, _ = st.ising_spectrum(8, coupling=1.0, field=0.2, periodic=True)
        betas = [0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0]
        values = [st.internal_energy(energies, b) for b in betas]
        assert all(b <= a + 1e-12 for a, b in zip(values, values[1:])), (
            f"U should be monotonically decreasing in beta, got {values}"
        )
