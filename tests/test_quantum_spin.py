r"""Verification of the quantum-spin module.

Three kinds of evidence, in order of strength:

1. **Operator algebra**, exact to round-off: :math:`[S^x, S^y] = iS^z`,
   :math:`S^2 = s(s+1)`, hermiticity, :math:`[H, S^2] = 0` at zero field.
2. **Closed-form spectra**: one spin in a field, two spins (singlet/triplet),
   the Ising limit.
3. **Structural identities**: the trace of a zero-field Heisenberg chain
   vanishes, probabilities normalise, unitary evolution preserves the norm.
"""

import math

import numpy as np
import pytest

from cpv import quantum_spin as qs
from cpv import verification as V

EXACT = 1e-12
SPINS = [0.5, 1.0, 1.5, 2.0, 2.5]


class TestSingleSpinAlgebra:
    @pytest.mark.parametrize("s", SPINS)
    def test_dimension(self, s):
        sx, sy, sz = qs.spin_operators(s)
        dim = int(round(2 * s + 1))
        assert sx.shape == (dim, dim), f"s={s}: Sx has shape {sx.shape}"

    @pytest.mark.parametrize("s", SPINS)
    def test_su2_commutation_relations(self, s):
        r"""All three of :math:`[S^a, S^b] = i\epsilon_{abc}S^c`."""
        sx, sy, sz = qs.spin_operators(s)
        np.testing.assert_allclose(V.commutator(sx, sy), 1.0j * sz, atol=EXACT)
        np.testing.assert_allclose(V.commutator(sy, sz), 1.0j * sx, atol=EXACT)
        np.testing.assert_allclose(V.commutator(sz, sx), 1.0j * sy, atol=EXACT)

    @pytest.mark.parametrize("s", SPINS)
    def test_casimir_eigenvalue(self, s):
        r""":math:`S^2 = s(s+1)\mathbb{1}`."""
        sx, sy, sz = qs.spin_operators(s)
        dim = int(round(2 * s + 1))
        s_sq = sx @ sx + sy @ sy + sz @ sz
        np.testing.assert_allclose(s_sq, s * (s + 1) * np.eye(dim), atol=EXACT)

    @pytest.mark.parametrize("s", SPINS)
    def test_operators_are_hermitian(self, s):
        for name, op in zip("xyz", qs.spin_operators(s)):
            assert V.check_hermitian(op, name=f"S{name} (s={s})") < EXACT

    @pytest.mark.parametrize("s", SPINS)
    def test_sz_spectrum_is_the_magnetic_quantum_numbers(self, s):
        _, _, sz = qs.spin_operators(s)
        want = np.arange(-s, s + 0.5, 1.0)
        assert V.spectral_error(np.linalg.eigvalsh(sz), want) < EXACT

    @pytest.mark.parametrize("s", SPINS)
    def test_traces_vanish(self, s):
        r"""Each :math:`S^a` is traceless: the m values are symmetric about 0."""
        for name, op in zip("xyz", qs.spin_operators(s)):
            assert abs(np.trace(op)) < EXACT, f"tr S{name} = {np.trace(op)}"

    def test_spin_half_is_half_the_pauli_matrices(self):
        sx, sy, sz = qs.spin_operators(0.5)
        for op, pauli in zip((sx, sy, sz), ("x", "y", "z")):
            np.testing.assert_allclose(op, 0.5 * qs.PAULI[pauli], atol=EXACT)

    def test_invalid_spin_raises(self):
        for bad in (0.0, 0.3, -1.0):
            with pytest.raises(ValueError, match="positive half-integer"):
                qs.spin_operators(bad)


class TestManyBodyConstruction:
    def test_embed_places_the_operator_at_the_right_site(self):
        _, _, sz = qs.spin_operators(0.5)
        op = qs.embed(sz, 0, 2)
        # site 0 is the most significant factor: diag = (+1/2, +1/2, -1/2, -1/2)
        np.testing.assert_allclose(np.diag(op).real, [0.5, 0.5, -0.5, -0.5], atol=EXACT)
        op1 = qs.embed(sz, 1, 2)
        np.testing.assert_allclose(np.diag(op1).real, [0.5, -0.5, 0.5, -0.5], atol=EXACT)

    def test_basis_labels_match_the_matrix_order(self):
        assert qs.basis_labels(2) == ["uu", "ud", "du", "dd"]
        assert len(qs.basis_labels(4)) == 16

    def test_embed_rejects_a_bad_site(self):
        _, _, sz = qs.spin_operators(0.5)
        with pytest.raises(IndexError):
            qs.embed(sz, 3, 3)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_hamiltonian_dimension_and_hermiticity(self, n):
        h = qs.heisenberg_chain(n, field=0.3)
        assert h.shape == (2 ** n, 2 ** n)
        assert V.check_hermitian(h, name=f"H (n={n})") < EXACT

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_zero_field_hamiltonian_is_traceless(self, n):
        r""":math:`\operatorname{tr}(S^a_i S^a_j) = 0` for :math:`i \neq j`."""
        periodic = n > 2
        h = qs.heisenberg_chain(n, periodic=periodic)
        assert abs(np.trace(h)) < EXACT, f"n={n}: tr H = {np.trace(h)}"

    def test_two_site_periodic_is_rejected_as_ambiguous(self):
        with pytest.raises(ValueError, match="ambiguous for n_sites=2"):
            qs.heisenberg_chain(2, periodic=True)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_total_spin_squared_commutes_with_the_isotropic_hamiltonian(self, n):
        r""":math:`[H, S^2] = 0` for isotropic coupling, at zero field."""
        periodic = n > 2
        h = qs.heisenberg_chain(n, periodic=periodic)
        assert V.check_commutes(h, qs.total_spin_squared(n), name=f"[H, S^2] (n={n})") < EXACT

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_a_uniform_field_does_not_spoil_s_squared(self, n):
        r"""A uniform Zeeman term leaves :math:`S^2` conserved.

        This is easy to get backwards. A field along :math:`z` certainly
        singles out an axis, and it does lift the degeneracy inside a
        multiplet, but :math:`S^2` commutes with *every* component of the
        total spin, so :math:`[S^2, \sum_i S^z_i] = 0` and the field commutes
        with :math:`S^2` too. :math:`S^2` therefore remains a good quantum
        number; what the field removes is the degeneracy, not the symmetry
        label. The measured commutator here is 4e-16.
        """
        periodic = n > 2
        h = qs.heisenberg_chain(n, field=0.7, periodic=periodic)
        assert V.check_commutes(h, qs.total_spin_squared(n),
                                name=f"[H(h), S^2] (n={n})") < EXACT

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_total_sz_commutes_with_the_hamiltonian_even_in_a_field(self, n):
        r"""U(1) symmetry about the field axis."""
        periodic = n > 2
        h = qs.heisenberg_chain(n, field=0.7, periodic=periodic)
        _, _, sz_tot = qs.total_spin_operators(n)
        assert V.check_commutes(h, sz_tot, name=f"[H, Sz] (n={n})") < EXACT

    @pytest.mark.parametrize("anisotropy,label", [
        ((1.0, 1.0, 0.5), "XXZ with Delta_z = 0.5"),
        ((1.0, 1.0, 2.0), "XXZ with Delta_z = 2.0"),
        ((0.0, 0.0, 1.0), "Ising"),
        ((1.0, 1.0, 0.0), "XY"),
    ])
    def test_exchange_anisotropy_is_what_breaks_s_squared(self, anisotropy, label):
        r"""Anisotropic *coupling* breaks :math:`[H, S^2] = 0`; a uniform field
        does not. All four of these leave :math:`S^z_{tot}` conserved."""
        h = qs.heisenberg_chain(4, coupling=1.0, anisotropy=anisotropy, periodic=True)
        assert not V.commutes(h, qs.total_spin_squared(4)), (
            f"{label}: [H, S^2] should not vanish for anisotropic exchange"
        )
        _, _, sz_tot = qs.total_spin_operators(4)
        assert V.commutes(h, sz_tot), f"{label}: [H, Sz] should still vanish"

    def test_a_staggered_field_breaks_s_squared(self):
        r"""A site-dependent field is the other way to break it: the Zeeman term
        is then no longer proportional to :math:`S^z_{tot}`."""
        n = 4
        _, _, sz_site = qs.spin_operators(0.5)
        h = qs.heisenberg_chain(n, coupling=1.0, periodic=True)
        staggered = h + sum((-1.0) ** i * 0.7 * qs.embed(sz_site, i, n) for i in range(n))
        assert not V.commutes(staggered, qs.total_spin_squared(n)), (
            "a staggered field must break S^2 conservation"
        )
        _, _, sz_tot = qs.total_spin_operators(n)
        assert V.commutes(staggered, sz_tot), (
            "a staggered field along z still conserves total Sz"
        )


class TestClosedFormSpectra:
    @pytest.mark.parametrize("field", [0.5, 1.0, 2.0, 7.3])
    def test_single_spin_in_a_field(self, field):
        r""":math:`H = -h S^z` has eigenvalues :math:`\mp h/2`."""
        energies, _ = qs.eigensystem(qs.heisenberg_chain(1, field=field))
        assert V.spectral_error(energies, [-field / 2, field / 2]) < EXACT

    @pytest.mark.parametrize("coupling", [1.0, 2.5, -1.0])
    def test_two_spin_singlet_triplet(self, coupling):
        r""":math:`H = J\,\mathbf S_1\cdot\mathbf S_2` gives
        :math:`-3J/4` (singlet) and :math:`+J/4` (triplet, threefold)."""
        energies, _ = qs.eigensystem(qs.heisenberg_chain(2, coupling=coupling))
        want = [-0.75 * coupling] + [0.25 * coupling] * 3
        assert V.spectral_error(energies, want) < EXACT, (
            f"J={coupling}: got {np.sort(energies)}, expected {sorted(want)}"
        )

    def test_singlet_is_the_antisymmetric_combination(self):
        r"""The ground state of the antiferromagnet is
        :math:`(|ud\rangle - |du\rangle)/\sqrt2`."""
        _, psi = qs.ground_state(qs.heisenberg_chain(2, coupling=1.0))
        want = np.array([0.0, 1.0, -1.0, 0.0]) / math.sqrt(2.0)
        overlap = abs(complex(want.conj() @ psi))
        assert overlap == pytest.approx(1.0, abs=1e-10), (
            f"|<singlet|psi>| = {overlap:.12f}; amplitudes {np.round(psi, 6)}"
        )

    def test_singlet_has_zero_total_spin(self):
        _, psi = qs.ground_state(qs.heisenberg_chain(2, coupling=1.0))
        assert qs.expectation(qs.total_spin_squared(2), psi) == pytest.approx(0.0, abs=1e-10)

    def test_triplet_has_total_spin_two(self):
        r"""The excited level has :math:`S^2 = S(S+1) = 2` for :math:`S = 1`."""
        energies, vectors = qs.eigensystem(qs.heisenberg_chain(2, coupling=1.0))
        s_sq = qs.total_spin_squared(2)
        for k in (1, 2, 3):
            value = qs.expectation(s_sq, vectors[:, k])
            assert value == pytest.approx(2.0, abs=1e-10), f"level {k}: S^2 = {value}"

    @pytest.mark.parametrize("field", [0.0, 0.6, 2.4])
    def test_two_spin_zeeman_splitting(self, field):
        r"""With :math:`H = J\,\mathbf S_1 \cdot \mathbf S_2 - h S^z_{tot}` the
        levels are :math:`-3J/4` for the singlet and :math:`J/4 - h m` for the
        triplet, :math:`m \in \{-1, 0, 1\}`.

        This is the companion to
        :meth:`test_a_uniform_field_does_not_spoil_s_squared`: the field splits
        the triplet linearly in ``m`` while every triplet state keeps
        :math:`S^2 = 2`.
        """
        coupling = 1.0
        h = qs.heisenberg_chain(2, coupling=coupling, field=field)
        energies, vectors = qs.eigensystem(h)

        want = sorted([-0.75 * coupling] + [0.25 * coupling - field * m
                                            for m in (-1.0, 0.0, 1.0)])
        assert V.spectral_error(energies, want) < EXACT, (
            f"h={field}: got {np.sort(energies)}, expected {want}"
        )

        s_sq = qs.total_spin_squared(2)
        _, _, sz_tot = qs.total_spin_operators(2)
        for k in range(4):
            total_spin = qs.expectation(s_sq, vectors[:, k])
            assert total_spin == pytest.approx(0.0, abs=1e-10) or \
                   total_spin == pytest.approx(2.0, abs=1e-10), (
                f"h={field}, level {k}: S^2 = {total_spin}, expected 0 or 2"
            )
            m_value = qs.expectation(sz_tot, vectors[:, k])
            assert abs(m_value - round(m_value)) < 1e-10, (
                f"h={field}, level {k}: Sz = {m_value} is not an integer"
            )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_ising_limit_is_diagonal_with_classical_energies(self, n):
        r"""With ``anisotropy=(0, 0, 1)`` the Hamiltonian is classical."""
        periodic = n > 2
        h = qs.heisenberg_chain(n, coupling=1.0, anisotropy=(0.0, 0.0, 1.0),
                                periodic=periodic)
        off_diagonal = h - np.diag(np.diag(h))
        assert np.max(np.abs(off_diagonal)) < EXACT, "the Ising limit must be diagonal"

        bonds = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if periodic else [])
        want = []
        for k in range(2 ** n):
            spins = [0.5 if not (k >> (n - 1 - i)) & 1 else -0.5 for i in range(n)]
            want.append(sum(spins[i] * spins[j] for i, j in bonds))
        assert V.spectral_error(np.diag(h).real, want) < EXACT

    def test_ferromagnet_ground_state_is_fully_polarised(self):
        r"""For :math:`J < 0` the ground state is the maximal-:math:`S^z` state
        once a small field selects it."""
        h = qs.heisenberg_chain(4, coupling=-1.0, field=0.01)
        energy, psi = qs.ground_state(h)
        probs = qs.state_probabilities(psi)
        assert probs[0] == pytest.approx(1.0, abs=1e-6), (
            f"P(all up) = {probs[0]:.9f}; the polarised state should dominate"
        )


class TestStatesAndEvolution:
    def test_probabilities_normalise_and_are_labelled(self):
        _, psi = qs.ground_state(qs.heisenberg_chain(2, coupling=1.0))
        probs = qs.state_probabilities(psi, labels=qs.basis_labels(2))
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-12)
        assert probs["ud"] == pytest.approx(0.5, abs=1e-10)
        assert probs["du"] == pytest.approx(0.5, abs=1e-10)
        assert probs["uu"] == pytest.approx(0.0, abs=1e-10)

    def test_unnormalised_state_is_rejected(self):
        with pytest.raises(ValueError, match="not normalised"):
            qs.state_probabilities(np.array([1.0, 1.0]))
        with pytest.raises(ValueError, match="not normalised"):
            qs.expectation(np.eye(2), np.array([2.0, 0.0]))

    def test_evolution_preserves_the_norm(self):
        h = qs.heisenberg_chain(3, coupling=1.0, field=0.4, periodic=True)
        psi0 = np.zeros(8, dtype=complex)
        psi0[3] = 1.0
        out = qs.evolve(h, psi0, np.linspace(0.0, 5.0, 25))
        norms = np.linalg.norm(out, axis=1)
        assert np.max(np.abs(norms - 1.0)) < EXACT, (
            f"max norm deviation {np.max(np.abs(norms - 1.0)):.3e}"
        )

    def test_evolution_at_t_zero_is_the_identity(self):
        h = qs.heisenberg_chain(2, coupling=1.3)
        psi0 = np.array([0.6, 0.0, 0.8, 0.0], dtype=complex)
        np.testing.assert_allclose(qs.evolve(h, psi0, [0.0])[0], psi0, atol=EXACT)

    def test_rabi_oscillation_matches_the_closed_form(self):
        r"""A spin along :math:`+x` precesses about a field along :math:`+z`:
        :math:`\langle S^x \rangle = \tfrac12\cos(ht)`,
        :math:`\langle S^y\rangle = -\tfrac12\sin(ht)`."""
        field = 1.7
        h = qs.heisenberg_chain(1, field=field)
        psi0 = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
        times = np.linspace(0.0, 4.0, 41)
        states = qs.evolve(h, psi0, times)

        sx, sy, _ = qs.spin_operators(0.5)
        got_x = np.array([qs.expectation(sx, s).real for s in states])
        got_y = np.array([qs.expectation(sy, s).real for s in states])

        np.testing.assert_allclose(got_x, 0.5 * np.cos(field * times), atol=1e-12)
        np.testing.assert_allclose(got_y, -0.5 * np.sin(field * times), atol=1e-12)

    def test_an_eigenstate_only_picks_up_a_phase(self):
        h = qs.heisenberg_chain(2, coupling=1.0)
        energies, vectors = qs.eigensystem(h)
        psi = vectors[:, 0]
        t = 2.3
        got = qs.evolve(h, psi, [t])[0]
        want = np.exp(-1.0j * energies[0] * t) * psi
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_evolution_is_periodic_for_the_two_spin_chain(self):
        r"""The spectrum is :math:`\{-3J/4, J/4\}`, so the gap is :math:`J` and
        the state returns (up to a phase) after :math:`2\pi/J`."""
        h = qs.heisenberg_chain(2, coupling=1.0)
        psi0 = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
        revived = qs.evolve(h, psi0, [2.0 * math.pi])[0]
        overlap = abs(complex(psi0.conj() @ revived))
        assert overlap == pytest.approx(1.0, abs=1e-10), f"|<psi0|psi(T)>| = {overlap}"

    def test_non_hermitian_input_is_refused(self):
        bad = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
        with pytest.raises(ValueError, match="not Hermitian"):
            qs.eigensystem(bad)


class TestThermalState:
    def test_density_matrix_has_unit_trace_and_is_hermitian(self):
        h = qs.heisenberg_chain(3, coupling=1.0, field=0.2, periodic=True)
        rho = qs.thermal_density_matrix(h, beta=1.4)
        assert np.trace(rho).real == pytest.approx(1.0, abs=1e-12)
        assert V.check_hermitian(rho, name="rho") < EXACT
        assert np.min(np.linalg.eigvalsh(rho)) > -EXACT, "rho must be positive"

    def test_low_temperature_limit_is_the_ground_state_projector(self):
        h = qs.heisenberg_chain(2, coupling=1.0)
        rho = qs.thermal_density_matrix(h, beta=200.0)
        _, psi = qs.ground_state(h)
        np.testing.assert_allclose(rho, np.outer(psi, psi.conj()), atol=1e-10)

    def test_high_temperature_limit_is_maximally_mixed(self):
        h = qs.heisenberg_chain(3, coupling=1.0, periodic=True)
        rho = qs.thermal_density_matrix(h, beta=1e-9)
        np.testing.assert_allclose(rho, np.eye(8) / 8.0, atol=1e-8)

    def test_no_overflow_at_very_low_temperature(self):
        r"""The ground-state shift is what keeps :math:`e^{-\beta E}` finite."""
        h = qs.heisenberg_chain(4, coupling=1.0, field=3.0)
        rho = qs.thermal_density_matrix(h, beta=500.0)
        assert np.all(np.isfinite(rho)), "beta = 500 overflowed"
        assert np.trace(rho).real == pytest.approx(1.0, abs=1e-12)
