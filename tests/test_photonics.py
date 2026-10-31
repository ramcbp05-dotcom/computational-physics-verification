r"""Verification of the photonics module.

Linear optics is a place where the algebra does nearly all the verification.
Every component is a 2x2 unitary, so:

* **unitarity** is checked for every matrix the module produces, and it is the
  same statement as energy conservation;
* the **Mach-Zehnder** output has a closed form,
  :math:`(\sin^2(\varphi/2), \cos^2(\varphi/2))`, derived in
  ``docs/03-photonics.md`` and compared term by term here;
* the **two-photon** coincidence probability has the closed form
  :math:`(T - R)^2`, which is an exact zero at 50:50 -- the Hong-Ou-Mandel dip.
  An exact zero is a much sharper test of the phase convention than a small
  number would be.
"""

import math

import numpy as np
import pytest

from cpv import photonics as ph
from cpv import verification as V

EXACT = 1e-12
THETAS = [0.0, 0.1, 0.4, math.pi / 4, 1.0, 1.2, math.pi / 2]
PHASES = [0.0, 0.3, 1.0, math.pi / 2, 2.0, math.pi, 5.0]


class TestComponentsAreUnitary:
    @pytest.mark.parametrize("theta", THETAS)
    @pytest.mark.parametrize("phi", [0.0, 0.7, math.pi])
    def test_beam_splitter(self, theta, phi):
        assert V.check_unitary(ph.beam_splitter(theta, phi),
                               name=f"B({theta:.3f}, {phi:.3f})") < EXACT

    @pytest.mark.parametrize("phi", PHASES)
    @pytest.mark.parametrize("mode", [0, 1])
    def test_phase_shifter(self, phi, mode):
        assert V.check_unitary(ph.phase_shifter(phi, mode),
                               name=f"P({phi:.3f}, mode={mode})") < EXACT

    @pytest.mark.parametrize("phi", PHASES)
    def test_mach_zehnder(self, phi):
        assert V.check_unitary(ph.mach_zehnder(phi), name=f"MZI({phi:.3f})") < EXACT

    def test_unbalanced_mach_zehnder(self):
        for theta_in, theta_out in ((0.3, 1.1), (math.pi / 4, 0.2), (0.9, 0.9)):
            matrix = ph.mach_zehnder(0.8, theta_in=theta_in, theta_out=theta_out)
            assert V.check_unitary(matrix, name="unbalanced MZI") < EXACT

    def test_a_cascade_of_unitaries_is_unitary(self):
        rng = np.random.default_rng(7)
        components = [ph.beam_splitter(float(t), float(p))
                      for t, p in rng.uniform(0.0, math.pi, size=(6, 2))]
        components += [ph.phase_shifter(float(p)) for p in rng.uniform(0.0, 7.0, size=4)]
        assert V.check_unitary(ph.cascade(*components), name="random cascade") < 1e-11

    def test_splitting_ratio_sums_to_one(self):
        for theta in THETAS:
            t, r = ph.splitting_ratio(theta)
            assert t + r == pytest.approx(1.0, abs=1e-15), f"theta={theta}: T+R={t + r}"

    def test_balanced_splitter_is_the_canonical_matrix(self):
        want = np.array([[1.0, 1.0j], [1.0j, 1.0]]) / math.sqrt(2.0)
        np.testing.assert_allclose(ph.beam_splitter(), want, atol=EXACT)

    def test_reflection_carries_a_quarter_wave_relative_phase(self):
        r"""Unitarity forces the reflected and transmitted phases to differ by
        :math:`\pi/2` for a symmetric lossless splitter."""
        b = ph.beam_splitter(0.6)
        relative = np.angle(b[1, 0]) - np.angle(b[0, 0])
        assert abs(relative) == pytest.approx(math.pi / 2, abs=1e-12), (
            f"relative phase {relative:.9f}, expected +-pi/2"
        )


class TestEnergyConservation:
    @pytest.mark.parametrize("theta", THETAS)
    def test_single_component(self, theta):
        state = np.array([0.6, 0.8j])
        out = ph.propagate(ph.beam_splitter(theta), state)
        assert ph.intensities(out).sum() == pytest.approx(1.0, abs=1e-14)

    @pytest.mark.parametrize("phi", PHASES)
    def test_mach_zehnder_conserves_total_intensity(self, phi):
        out = ph.propagate(ph.mach_zehnder(phi), (1.0, 0.0))
        assert ph.intensities(out).sum() == pytest.approx(1.0, abs=1e-14)

    def test_a_phase_shifter_changes_no_intensity(self):
        r"""A phase is unobservable without a second path to interfere with."""
        state = np.array([0.6, 0.8])
        before = ph.intensities(state)
        for phi in PHASES:
            after = ph.intensities(ph.propagate(ph.phase_shifter(phi), state))
            np.testing.assert_allclose(after, before, atol=1e-15)


class TestCascadeOrder:
    def test_cascade_is_optical_order(self):
        r"""``cascade(A, B)`` is ``B @ A``: ``A`` is met first."""
        a = ph.beam_splitter(0.3)
        b = ph.phase_shifter(1.1)
        np.testing.assert_allclose(ph.cascade(a, b), b @ a, atol=EXACT)

    def test_order_matters_for_non_commuting_components(self):
        r"""A phase before and after a splitter are physically different."""
        a = ph.beam_splitter(math.pi / 4)
        b = ph.phase_shifter(1.1)
        assert np.max(np.abs(ph.cascade(a, b) - ph.cascade(b, a))) > 0.1, (
            "a splitter and a phase shifter do not commute; if they appear to, "
            "the cascade order is being lost"
        )

    def test_single_component_cascade_is_the_identity_on_it(self):
        a = ph.beam_splitter(0.9, 0.4)
        np.testing.assert_allclose(ph.cascade(a), a, atol=EXACT)

    def test_empty_cascade_raises(self):
        with pytest.raises(ValueError, match="at least one component"):
            ph.cascade()

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="expected"):
            ph.cascade(np.eye(3))


class TestMachZehnderClosedForm:
    def test_output_matches_sin_cos_half_phase(self):
        r"""The derivation gives
        :math:`I = (\sin^2(\varphi/2), \cos^2(\varphi/2))` for input in mode 0."""
        phases = np.linspace(0.0, 4.0 * math.pi, 201)
        got = ph.mach_zehnder_scan(phases)
        want = np.column_stack([np.sin(phases / 2) ** 2, np.cos(phases / 2) ** 2])
        error = float(np.max(np.abs(got - want)))
        assert error < 1e-14, f"max deviation from the closed form: {error:.3e}"

    def test_dark_and_bright_fringes(self):
        r""":math:`\varphi = 0` sends everything to mode 1;
        :math:`\varphi = \pi` sends everything to mode 0."""
        dark = ph.intensities(ph.propagate(ph.mach_zehnder(0.0), (1.0, 0.0)))
        np.testing.assert_allclose(dark, [0.0, 1.0], atol=1e-15)
        bright = ph.intensities(ph.propagate(ph.mach_zehnder(math.pi), (1.0, 0.0)))
        np.testing.assert_allclose(bright, [1.0, 0.0], atol=1e-15)

    def test_balanced_interferometer_has_unit_visibility(self):
        scan = ph.mach_zehnder_scan(np.linspace(0.0, 2.0 * math.pi, 401))
        assert ph.visibility(scan[:, 0]) == pytest.approx(1.0, abs=1e-12), (
            f"V = {ph.visibility(scan[:, 0]):.12f}"
        )

    def test_fringes_are_two_pi_periodic(self):
        for phi in (0.3, 1.7, 4.4):
            a = ph.intensities(ph.propagate(ph.mach_zehnder(phi), (1.0, 0.0)))
            b = ph.intensities(ph.propagate(ph.mach_zehnder(phi + 2 * math.pi), (1.0, 0.0)))
            np.testing.assert_allclose(a, b, atol=1e-14)

    def test_phase_in_the_other_arm_flips_the_fringe(self):
        r"""Only the *difference* between the arms is observable."""
        phi = 0.9
        arm0 = ph.intensities(ph.propagate(ph.mach_zehnder(phi, arm=0), (1.0, 0.0)))
        arm1 = ph.intensities(ph.propagate(ph.mach_zehnder(-phi, arm=1), (1.0, 0.0)))
        np.testing.assert_allclose(arm0, arm1, atol=1e-14)

    @pytest.mark.parametrize("theta_in", [0.15, 0.4, 0.6, 1.2])
    def test_unbalanced_visibility_matches_the_two_beam_formula(self, theta_in):
        r"""With the first splitter unbalanced the arms carry
        :math:`I_1 = \cos^2\theta` and :math:`I_2 = \sin^2\theta`, and the
        visibility must be :math:`2\sqrt{I_1 I_2}/(I_1 + I_2)`."""
        scan = ph.mach_zehnder_scan(np.linspace(0.0, 2.0 * math.pi, 801),
                                    theta_in=theta_in)
        got = ph.visibility(scan[:, 0])
        t, r = ph.splitting_ratio(theta_in)
        want = ph.two_beam_visibility(t, r)
        assert got == pytest.approx(want, rel=1e-9), (
            f"theta_in={theta_in}: matrix gives V={got:.9f}, "
            f"two-beam formula gives {want:.9f}"
        )

    def test_a_fully_reflecting_first_splitter_gives_no_fringes(self):
        scan = ph.mach_zehnder_scan(np.linspace(0.0, 2.0 * math.pi, 201),
                                    theta_in=math.pi / 2)
        assert ph.visibility(scan[:, 0]) == pytest.approx(0.0, abs=1e-12), (
            "with all the light in one arm there is nothing to interfere with"
        )

    def test_visibility_input_validation(self):
        with pytest.raises(ValueError, match="at least two samples"):
            ph.visibility([0.5])
        with pytest.raises(ValueError, match="identically zero"):
            ph.visibility([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="non-negative"):
            ph.visibility([-1.0, 1.0])

    def test_two_beam_visibility_edge_cases(self):
        assert ph.two_beam_visibility(1.0, 1.0) == pytest.approx(1.0)
        assert ph.two_beam_visibility(1.0, 0.0) == pytest.approx(0.0)
        with pytest.raises(ValueError, match="both arms are dark"):
            ph.two_beam_visibility(0.0, 0.0)


class TestTwoPhotonInterference:
    def test_hong_ou_mandel_dip_is_an_exact_zero(self):
        r"""At 50:50 the permanent is :math:`(1 \cdot 1 + i \cdot i)/2 = 0`."""
        out = ph.two_photon_output(ph.beam_splitter())
        assert out.p_11 < 1e-30, (
            f"P(1,1) = {out.p_11:.3e}; the HOM dip is an exact algebraic zero, "
            "not a small number -- a non-zero value means the pi/2 reflection "
            "phase is wrong"
        )
        assert out.p_20 == pytest.approx(0.5, abs=1e-14)
        assert out.p_02 == pytest.approx(0.5, abs=1e-14)

    @pytest.mark.parametrize("theta", THETAS)
    def test_probabilities_sum_to_one(self, theta):
        out = ph.two_photon_output(ph.beam_splitter(theta))
        assert out.total == pytest.approx(1.0, abs=1e-14), (
            f"theta={theta}: probabilities sum to {out.total:.15f}"
        )

    @pytest.mark.parametrize("theta", THETAS)
    def test_coincidence_probability_is_t_minus_r_squared(self, theta):
        r"""The permanent of :math:`B(\theta, \varphi)` is
        :math:`\cos^2\theta - \sin^2\theta = T - R`, independently of
        :math:`\varphi`, so :math:`P(1,1) = (T - R)^2`."""
        t, r = ph.splitting_ratio(theta)
        for phi in (0.0, 0.8, 2.5):
            out = ph.two_photon_output(ph.beam_splitter(theta, phi))
            assert out.p_11 == pytest.approx((t - r) ** 2, abs=1e-14), (
                f"theta={theta} phi={phi}: P(1,1) = {out.p_11:.12f}, "
                f"(T-R)^2 = {(t - r) ** 2:.12f}"
            )

    def test_bunching_terms_are_symmetric_for_a_symmetric_splitter(self):
        for theta in (0.3, math.pi / 4, 1.1):
            out = ph.two_photon_output(ph.beam_splitter(theta))
            assert out.p_20 == pytest.approx(out.p_02, abs=1e-14), (
                f"theta={theta}: P(2,0)={out.p_20} != P(0,2)={out.p_02}"
            )

    def test_identity_leaves_the_photons_split(self):
        r"""No splitter means no interference: the photons stay one per mode."""
        out = ph.two_photon_output(np.eye(2, dtype=complex))
        assert out.p_11 == pytest.approx(1.0, abs=1e-14)
        assert out.p_20 == pytest.approx(0.0, abs=1e-14)

    def test_a_full_swap_also_leaves_one_per_mode(self):
        swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        out = ph.two_photon_output(swap)
        assert out.p_11 == pytest.approx(1.0, abs=1e-14)

    def test_dip_is_a_minimum_in_theta(self):
        r"""Scanning the splitting ratio traces the HOM dip, minimum at 50:50."""
        thetas = np.linspace(0.0, math.pi / 2, 201)
        coincidence = np.array([ph.two_photon_output(ph.beam_splitter(float(t))).p_11
                                for t in thetas])
        argmin = int(np.argmin(coincidence))
        assert thetas[argmin] == pytest.approx(math.pi / 4, abs=0.01), (
            f"dip at theta={thetas[argmin]:.4f}, expected pi/4={math.pi / 4:.4f}"
        )
        assert coincidence[argmin] < 1e-20
        assert coincidence[0] == pytest.approx(1.0, abs=1e-14), (
            "a fully transmitting splitter must give P(1,1) = 1"
        )

    def test_dip_survives_an_extra_phase_before_the_splitter(self):
        r"""A phase on one input mode cannot fill the dip: it changes the
        permanent only by an overall factor."""
        matrix = ph.cascade(ph.phase_shifter(1.37), ph.beam_splitter())
        assert ph.two_photon_output(matrix).p_11 < 1e-30

    def test_bad_matrix_shape_raises(self):
        with pytest.raises(ValueError, match="expected a \\(2, 2\\) matrix"):
            ph.two_photon_output(np.eye(3))


class TestInputValidation:
    def test_propagate_shape_checks(self):
        with pytest.raises(ValueError, match="expected"):
            ph.propagate(np.eye(3), (1.0, 0.0))
        with pytest.raises(ValueError, match="expected"):
            ph.propagate(np.eye(2), (1.0, 0.0, 0.0))

    def test_intensities_shape_check(self):
        with pytest.raises(ValueError, match="expected a 2-vector"):
            ph.intensities([1.0, 2.0, 3.0])

    def test_phase_shifter_mode_check(self):
        with pytest.raises(ValueError, match="mode must be 0 or 1"):
            ph.phase_shifter(1.0, mode=2)
