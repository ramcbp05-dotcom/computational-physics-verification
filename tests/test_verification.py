"""Tests for the verification tooling itself.

These matter more than they look: every physics claim in the repository is
asserted through this module, so an order-fitting routine that silently returns
the wrong exponent would hide a broken integrator.
"""

import numpy as np
import pytest

from cpv import verification as V


class TestErrorMetrics:
    def test_relative_error_is_zero_for_identical_input(self):
        a = np.array([1.0, 2.0, 3.0])
        assert V.relative_error(a, a) == 0.0

    def test_relative_error_scales_by_the_reference(self):
        assert V.relative_error([1.01], [1.0]) == pytest.approx(0.01, rel=1e-12)
        assert V.relative_error([101.0], [100.0]) == pytest.approx(0.01, rel=1e-12)

    def test_relative_error_floor_guards_a_near_zero_reference(self):
        """Without a floor, a reference of 1e-300 makes any error look infinite."""
        unguarded = V.relative_error([1e-12], [1e-300])
        guarded = V.relative_error([1e-12], [1e-300], floor=1.0)
        assert unguarded > 1e200, "expected the unguarded version to blow up"
        assert guarded == pytest.approx(1e-12, rel=1e-9)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            V.relative_error([1.0, 2.0], [1.0])

    def test_spectral_error_sorts_both_spectra(self):
        """Eigenvalue routines make no ordering promise."""
        assert V.spectral_error([3.0, 1.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_spectral_error_detects_a_real_difference(self):
        assert V.spectral_error([1.0, 2.0, 3.1], [1.0, 2.0, 3.0]) == pytest.approx(0.1)

    def test_spectral_error_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="different sizes"):
            V.spectral_error([1.0], [1.0, 2.0])


class TestConvergenceOrder:
    @pytest.mark.parametrize("order", [1.0, 2.0, 3.0, 4.0])
    def test_recovers_a_synthetic_power_law(self, order):
        steps = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
        errors = 3.7 * steps ** order
        study = V.convergence_order(steps, errors, expected=order)
        assert study.order == pytest.approx(order, abs=1e-9), (
            f"fitted {study.order:.4f} for a synthetic order-{order} sequence"
        )
        assert study.r_squared == pytest.approx(1.0, abs=1e-12)
        assert study.matches_expected()

    def test_r_squared_detects_a_non_power_law(self):
        steps = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
        errors = np.array([1.0, 1.0, 1.0, 0.5, 1e-4])   # not a straight line in log-log
        study = V.convergence_order(steps, errors)
        assert study.r_squared < 0.9, (
            f"R^2 = {study.r_squared:.4f}; a ragged sequence should not fit well"
        )

    def test_round_off_points_are_excluded_from_the_fit(self):
        """Below ~1e-13 the slope measures floating-point noise, not the method."""
        steps = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125])
        errors = np.array([1e-4, 2.5e-5, 6.25e-6, 1.5625e-6, 1e-15, 2e-15])
        study = V.convergence_order(steps, errors, expected=2.0)
        assert study.order == pytest.approx(2.0, abs=0.05), (
            f"fitted {study.order:.4f}; the two round-off points should be dropped"
        )

    def test_all_points_at_round_off_reports_converged(self):
        study = V.convergence_order([0.1, 0.05, 0.025], [1e-16, 2e-16, 1e-16],
                                    expected=4.0)
        assert study.converged
        assert study.matches_expected(), "a converged study must not fail its order check"

    def test_too_few_resolutions_raises(self):
        with pytest.raises(ValueError, match="at least 3 resolutions"):
            V.convergence_order([0.1, 0.05], [1e-2, 2.5e-3])

    def test_non_positive_step_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            V.convergence_order([0.1, 0.0, 0.025], [1.0, 1.0, 1.0])

    def test_matches_expected_rejects_a_wrong_order(self):
        steps = np.array([0.1, 0.05, 0.025, 0.0125])
        study = V.convergence_order(steps, 2.0 * steps ** 1.0, expected=2.0)
        assert not study.matches_expected(), (
            "an order-1 sequence must not pass a check for order 2 -- this is the "
            "exact failure a first-order integrator masquerading as second order "
            "would produce"
        )

    def test_table_renders(self):
        steps = np.array([0.1, 0.05, 0.025, 0.0125])
        table = V.convergence_order(steps, steps ** 2, expected=2.0).table()
        assert "implied order" in table
        assert "fitted order" in table

    def test_run_convergence_study_wraps_a_solver(self):
        """A first-order quadrature of exp over [0, 1] by the left rectangle rule."""
        import math

        def solve(n):
            h = 1.0 / n
            return float(sum(math.exp(k * h) for k in range(n)) * h)

        study = V.run_convergence_study(solve, math.e - 1.0,
                                        [16, 32, 64, 128, 256], expected=1.0)
        assert study.order == pytest.approx(1.0, abs=0.05), (
            f"left rectangle rule fitted order {study.order:.4f}, expected 1"
        )


class TestAlgebraicChecks:
    def test_hermitian_accepts_a_pauli_matrix(self):
        sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        assert V.is_hermitian(sz)
        assert V.check_hermitian(sz) == 0.0

    def test_hermitian_rejects_a_non_hermitian_matrix(self):
        bad = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
        assert not V.is_hermitian(bad)
        with pytest.raises(AssertionError, match="not Hermitian"):
            V.check_hermitian(bad, name="bad")

    def test_unitary_accepts_a_rotation(self):
        theta = 0.7
        u = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]], dtype=complex)
        assert V.is_unitary(u)
        assert V.check_unitary(u) < 1e-15

    def test_unitary_rejects_a_scaled_matrix(self):
        with pytest.raises(AssertionError, match="not unitary"):
            V.check_unitary(1.001 * np.eye(2), name="scaled identity")

    def test_non_square_is_neither(self):
        rect = np.zeros((2, 3))
        assert not V.is_hermitian(rect)
        assert not V.is_unitary(rect)

    def test_commutator_of_pauli_matrices(self):
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        np.testing.assert_allclose(V.commutator(sx, sy), 2.0j * sz, atol=1e-15)

    def test_commutes_detects_both_outcomes(self):
        sz = np.diag([1.0, -1.0]).astype(complex)
        assert V.commutes(sz, sz)
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        assert not V.commutes(sx, sz)
        with pytest.raises(AssertionError, match="does not vanish"):
            V.check_commutes(sx, sz, name="[Sx, Sz]")
