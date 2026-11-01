r"""Verification of the trajectory module.

An integrator that produces a plausible-looking orbit is not evidence of
anything. What is checked here:

* **closed forms** (gyrofrequency, gyroradius, E x B drift speed, Kepler's
  third law, the vis-viva relation;
* **conserved quantities**) :math:`|\mathbf v|` under a pure magnetic field,
  specific energy and angular momentum under gravity;
* **convergence order** (leapfrog must measure 2 and RK4 must measure 4,
  because a first-order scheme mislabelled as second order is the failure mode
  that a plot cannot show;
* **the structural difference between the schemes**) the symplectic
  integrator's energy error is *bounded*, and RK4's grows with run length.
  Measured below: leapfrog's drift is flat at 2.68e-05 from 5 orbits to 320,
  while RK4's grows from 1.5e-10 to 3.1e-09.
"""

import math

import numpy as np
import pytest

from cpv import trajectories as tr
from cpv import verification as V

MU = 1.0


class TestChargedParticleClosedForms:
    @pytest.mark.parametrize("charge,b_z,mass,want", [
        (1.0, 1.0, 1.0, 1.0), (1.0, 2.0, 1.0, 2.0), (2.0, 3.0, 1.5, 4.0),
        (-1.0, 2.5, 2.0, 1.25), (3.0, 0.4, 1.5, 0.8),
    ])
    def test_gyrofrequency(self, charge, b_z, mass, want):
        got = tr.gyrofrequency(charge, [0.0, 0.0, b_z], mass)
        assert got == pytest.approx(want, rel=1e-14), (
            f"q={charge} B={b_z} m={mass}: omega_c = {got}, expected |q|B/m = {want}"
        )

    def test_gyrofrequency_is_independent_of_the_sign_of_the_charge(self):
        assert tr.gyrofrequency(1.0, [0, 0, 1], 1.0) == tr.gyrofrequency(-1.0, [0, 0, 1], 1.0)

    @pytest.mark.parametrize("charge,b_z,mass,v_perp", [
        (1.0, 1.0, 1.0, 1.0), (-1.0, 2.5, 2.0, 0.7), (3.0, 0.4, 1.5, 2.0),
    ])
    def test_gyroradius(self, charge, b_z, mass, v_perp):
        got = tr.gyroradius(charge, [0.0, 0.0, b_z], mass, v_perp)
        want = mass * v_perp / (abs(charge) * b_z)
        assert got == pytest.approx(want, rel=1e-14)

    def test_exb_drift_closed_form(self):
        drift = tr.exb_drift([1.0, 0.0, 0.0], [0.0, 0.0, 2.0])
        np.testing.assert_allclose(drift, [0.0, -0.5, 0.0], atol=1e-15)

    def test_exb_drift_is_independent_of_charge_and_mass(self):
        r"""The drift speed is :math:`E/B`, with no :math:`q/m` in it at all."""
        drift = tr.exb_drift([0.6, 0.0, 0.0], [0.0, 0.0, 1.5])
        assert float(np.linalg.norm(drift)) == pytest.approx(0.6 / 1.5, rel=1e-14)

    def test_exb_drift_is_perpendicular_to_both_fields(self):
        e = np.array([0.3, 0.7, 0.0])
        b = np.array([0.0, 0.0, 1.2])
        drift = tr.exb_drift(e, b)
        assert abs(float(drift @ e)) < 1e-15
        assert abs(float(drift @ b)) < 1e-15

    def test_zero_field_cases_raise(self):
        with pytest.raises(ValueError, match="zero magnetic field"):
            tr.exb_drift([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="zero magnetic field"):
            tr.gyroradius(1.0, [0.0, 0.0, 0.0], 1.0, 1.0)

    def test_guiding_centre_sits_one_gyroradius_away(self):
        r"""For :math:`q > 0` starting at :math:`[r_c, 0, 0]` with
        :math:`\mathbf v` along :math:`+y`, the centre is at
        :math:`[2 r_c, 0, 0]`; flipping the charge puts it at the origin."""
        b = [0.0, 0.0, 1.0]
        centre_pos = tr.guiding_centre([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], b, 1.0, 1.0)
        np.testing.assert_allclose(centre_pos, [2.0, 0.0, 0.0], atol=1e-14)
        centre_neg = tr.guiding_centre([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], b, -1.0, 1.0)
        np.testing.assert_allclose(centre_neg, [0.0, 0.0, 0.0], atol=1e-14)

    def test_lorentz_force_is_perpendicular_to_velocity_for_pure_b(self):
        r"""Which is why a magnetic field does no work."""
        v = np.array([0.3, -0.7, 1.1])
        a = tr.lorentz_acceleration(v, [0, 0, 0], [0.2, 0.5, -0.9], 1.0, 1.0)
        assert abs(float(a @ v)) < 1e-15, f"a . v = {a @ v}"


class TestChargedParticleIntegration:
    @pytest.mark.parametrize("steps_per_orbit", [4, 8, 20, 100, 1000])
    def test_boris_conserves_speed_at_any_step_size(self, steps_per_orbit):
        r"""The Boris rotation is an exact finite rotation, so :math:`|\mathbf v|`
        is preserved whatever the step. Measured: 7e-15 at four steps per orbit.
        """
        period = 2.0 * math.pi
        traj = tr.integrate_charged_particle(
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.5], b_field=[0.0, 0.0, 1.0],
            dt=period / steps_per_orbit, n_steps=steps_per_orbit * 50, method="boris")
        assert traj.speed_drift < 1e-13, (
            f"{steps_per_orbit} steps/orbit: |v| drifted by {traj.speed_drift:.3e}"
        )

    def test_rk4_does_not_conserve_speed_at_a_coarse_step(self):
        r"""The contrast that makes the previous test meaningful: at four steps
        per orbit RK4 loses 55 % of the speed, where Boris loses 7e-15."""
        period = 2.0 * math.pi
        traj = tr.integrate_charged_particle(
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.5], b_field=[0.0, 0.0, 1.0],
            dt=period / 4, n_steps=200, method="rk4")
        assert traj.speed_drift > 0.1, (
            f"RK4 speed drift was only {traj.speed_drift:.3e}; the comparison "
            "with Boris is the point of this pair of tests"
        )

    @pytest.mark.parametrize("charge,b_z,mass,v_perp", [
        (1.0, 1.0, 1.0, 1.0), (-1.0, 2.5, 2.0, 0.7),
        (3.0, 0.4, 1.5, 2.0), (-2.0, 1.3, 0.7, 1.1),
    ])
    def test_orbit_radius_equals_the_gyroradius(self, charge, b_z, mass, v_perp):
        """Measured about the guiding centre, so the result holds for either sign."""
        b = [0.0, 0.0, b_z]
        omega = tr.gyrofrequency(charge, b, mass)
        r_c = tr.gyroradius(charge, b, mass, v_perp)
        x0 = np.array([r_c, 0.0, 0.0])
        v0 = np.array([0.0, v_perp, 0.0])
        centre = tr.guiding_centre(x0, v0, b, charge, mass)

        traj = tr.integrate_charged_particle(
            x0, v0, b_field=b, charge=charge, mass=mass,
            dt=(2.0 * math.pi / omega) / 2000, n_steps=2000 * 3, method="boris")
        radii = np.linalg.norm(traj.positions - centre, axis=1)
        assert radii.mean() == pytest.approx(r_c, rel=1e-11), (
            f"mean radius {radii.mean():.12f} vs r_c {r_c:.12f}"
        )
        # The drift-kick-drift arrangement makes this an exact circle; the
        # asymmetric one leaves a 3.1e-03 spread.
        assert float(np.ptp(radii)) / r_c < 1e-11, (
            f"radius varies by {float(np.ptp(radii)) / r_c:.3e} of r_c; with a "
            "symmetric Boris push the gyro-orbit is a circle to round-off"
        )

    def test_parallel_velocity_is_untouched_by_the_magnetic_field(self):
        v_par = 0.37
        period = 2.0 * math.pi
        traj = tr.integrate_charged_particle(
            [1.0, 0.0, 0.0], [0.0, 1.0, v_par], b_field=[0.0, 0.0, 1.0],
            dt=period / 500, n_steps=500 * 4, method="boris")
        np.testing.assert_allclose(traj.velocities[:, 2], v_par, atol=1e-13)
        assert traj.positions[-1, 2] == pytest.approx(v_par * traj.times[-1], rel=1e-12), (
            "motion along B must be free"
        )

    @pytest.mark.parametrize("steps_per_orbit,tol", [(100, 1e-3), (500, 1e-4), (2000, 1e-5)])
    def test_exb_drift_of_the_guiding_centre(self, steps_per_orbit, tol):
        r"""Starting from rest, the guiding centre drifts at
        :math:`\mathbf E \times \mathbf B / B^2`. The residual is the
        integrator's orbit-phase error and falls with resolution."""
        e_field = [1.0, 0.0, 0.0]
        b_field = [0.0, 0.0, 2.0]
        period = 2.0 * math.pi / 2.0
        traj = tr.integrate_charged_particle(
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], e_field=e_field, b_field=b_field,
            dt=period / steps_per_orbit, n_steps=steps_per_orbit * 200, method="boris")

        mean_velocity = (traj.positions[-1] - traj.positions[0]) / traj.times[-1]
        want = tr.exb_drift(e_field, b_field)
        relative = abs(mean_velocity[1] - want[1]) / abs(want[1])
        assert relative < tol, (
            f"{steps_per_orbit} steps/gyroperiod: drift {mean_velocity[1]:.9f} "
            f"vs analytic {want[1]:.9f}, relative error {relative:.3e}"
        )

    def test_energy_is_conserved_in_static_fields(self):
        r"""Kinetic plus :math:`-q\,\mathbf E\cdot\mathbf r` is constant.

        Measured 5.1e-14 absolute on an energy scale of 5e-3, i.e. ~1e-11
        relative. The asymmetric drift arrangement gives 1.5e-04 here, so this
        test is what pins the position update down.
        """
        traj = tr.integrate_charged_particle(
            [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], e_field=[0.5, 0.0, 0.0],
            b_field=[0.0, 0.0, 1.0], dt=1e-3, n_steps=20000, method="boris")
        assert traj.energy_drift < 1e-9, f"energy drifted by {traj.energy_drift:.3e}"

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError, match="method must be one of"):
            tr.integrate_charged_particle([0, 0, 0], [1, 0, 0], method="verlet")
        with pytest.raises(ValueError, match="dt must be positive"):
            tr.integrate_charged_particle([0, 0, 0], [1, 0, 0], dt=0.0)
        with pytest.raises(ValueError, match="3-vectors"):
            tr.integrate_charged_particle([0, 0], [1, 0, 0])


class TestOrbitElements:
    @pytest.mark.parametrize("a,e", [(1.0, 0.0), (1.0, 0.3), (2.5, 0.6), (0.4, 0.85)])
    def test_elements_round_trip_from_a_periapsis_state(self, a, e):
        x0, v0 = tr.periapsis_state(a, e, MU)
        el = tr.orbit_elements(x0, v0, MU)
        assert el.bound
        assert el.semi_major_axis == pytest.approx(a, rel=1e-12), (
            f"a = {el.semi_major_axis}, expected {a}"
        )
        assert el.eccentricity == pytest.approx(e, abs=1e-12), (
            f"e = {el.eccentricity}, expected {e}"
        )
        assert el.periapsis == pytest.approx(a * (1 - e), rel=1e-12)
        assert el.apoapsis == pytest.approx(a * (1 + e), rel=1e-12)

    @pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 7.3])
    def test_keplers_third_law(self, a):
        assert tr.orbital_period(a, MU) == pytest.approx(
            2.0 * math.pi * math.sqrt(a ** 3 / MU), rel=1e-14
        )

    def test_period_scales_as_a_to_the_three_halves(self):
        r""":math:`T(4a)/T(a) = 8`."""
        assert tr.orbital_period(4.0) / tr.orbital_period(1.0) == pytest.approx(8.0, rel=1e-14)

    def test_circular_orbit_has_zero_eccentricity_and_vis_viva_speed(self):
        r = 2.3
        v = tr.circular_speed(r, MU)
        el = tr.orbit_elements([r, 0.0, 0.0], [0.0, v, 0.0], MU)
        assert el.eccentricity == pytest.approx(0.0, abs=1e-12)
        assert el.semi_major_axis == pytest.approx(r, rel=1e-13)
        assert el.specific_energy == pytest.approx(-MU / (2 * r), rel=1e-13)

    def test_specific_energy_is_minus_mu_over_two_a(self):
        for a, e in ((1.0, 0.2), (3.0, 0.5)):
            x0, v0 = tr.periapsis_state(a, e, MU)
            el = tr.orbit_elements(x0, v0, MU)
            assert el.specific_energy == pytest.approx(-MU / (2 * a), rel=1e-12)

    def test_angular_momentum_matches_the_closed_form(self):
        r""":math:`L = \sqrt{\mu a (1 - e^2)}`."""
        for a, e in ((1.0, 0.0), (1.0, 0.4), (2.0, 0.7)):
            x0, v0 = tr.periapsis_state(a, e, MU)
            el = tr.orbit_elements(x0, v0, MU)
            assert el.angular_momentum == pytest.approx(
                math.sqrt(MU * a * (1 - e ** 2)), rel=1e-12
            )

    def test_escape_speed_is_the_parabolic_boundary(self):
        r = 1.5
        v_escape = math.sqrt(2.0 * MU / r)
        el = tr.orbit_elements([r, 0.0, 0.0], [0.0, v_escape, 0.0], MU)
        assert not el.bound, f"specific energy {el.specific_energy:.3e} should be >= 0"
        assert el.apoapsis == math.inf
        assert el.period == math.inf
        assert el.eccentricity == pytest.approx(1.0, abs=1e-12)

    def test_hyperbolic_orbit_is_unbound(self):
        el = tr.orbit_elements([1.0, 0.0, 0.0], [0.0, 2.0, 0.0], MU)
        assert not el.bound
        assert el.eccentricity > 1.0

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError, match="at the origin"):
            tr.orbit_elements([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], MU)
        with pytest.raises(ValueError, match="eccentricity must be in"):
            tr.periapsis_state(1.0, 1.0)
        with pytest.raises(ValueError, match="semi-major axis must be positive"):
            tr.orbital_period(-1.0)


class TestTwoBodyIntegration:
    @pytest.mark.parametrize("method", ["leapfrog", "rk4"])
    @pytest.mark.parametrize("e", [0.0, 0.3, 0.6])
    def test_angular_momentum_is_conserved(self, method, e):
        r"""For a central force every leapfrog sub-step is parallel to
        :math:`\mathbf r` or :math:`\mathbf v`, so :math:`\mathbf L` is exact to
        round-off. RK4 conserves it well but not exactly."""
        x0, v0 = tr.periapsis_state(1.0, e, MU)
        period = tr.orbital_period(1.0, MU)
        traj = tr.integrate_two_body(x0, v0, mu=MU, dt=period / 4000,
                                     n_steps=4000 * 10, method=method)
        tol = 1e-12 if method == "leapfrog" else 1e-10
        assert traj.angular_momentum_drift < tol, (
            f"{method}, e={e}: L drifted by {traj.angular_momentum_drift:.3e}"
        )

    @pytest.mark.parametrize("e", [0.0, 0.3, 0.6])
    def test_radius_stays_between_periapsis_and_apoapsis(self, e):
        x0, v0 = tr.periapsis_state(1.0, e, MU)
        el = tr.orbit_elements(x0, v0, MU)
        traj = tr.integrate_two_body(x0, v0, mu=MU, dt=el.period / 4000,
                                     n_steps=4000 * 10, method="leapfrog")
        assert traj.radii.min() == pytest.approx(el.periapsis, abs=1e-5), (
            f"e={e}: periapsis {traj.radii.min():.9f} vs {el.periapsis:.9f}"
        )
        assert traj.radii.max() == pytest.approx(el.apoapsis, abs=1e-4), (
            f"e={e}: apoapsis {traj.radii.max():.9f} vs {el.apoapsis:.9f}"
        )

    def test_circular_orbit_returns_to_its_start_after_one_period(self):
        x0, v0 = tr.periapsis_state(1.0, 0.0, MU)
        period = tr.orbital_period(1.0, MU)
        traj = tr.integrate_two_body(x0, v0, mu=MU, dt=period / 20000,
                                     n_steps=20000, method="rk4")
        closure = float(np.linalg.norm(traj.positions[-1] - x0))
        assert closure < 1e-6, (
            f"the orbit closed to {closure:.3e}; Kepler's third law gives the "
            "period, so a circular orbit must return to its start"
        )

    def test_motion_stays_in_the_initial_plane(self):
        x0, v0 = tr.periapsis_state(1.0, 0.4, MU)
        traj = tr.integrate_two_body(x0, v0, mu=MU, dt=1e-3, n_steps=20000)
        assert float(np.max(np.abs(traj.positions[:, 2]))) < 1e-15, (
            "a central force cannot move the orbit out of its plane"
        )

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError, match="method must be one of"):
            tr.integrate_two_body([1, 0, 0], [0, 1, 0], method="boris")
        with pytest.raises(ValueError, match="at the singularity"):
            tr.integrate_two_body([0, 0, 0], [0, 1, 0])
        with pytest.raises(ValueError, match="mu must be positive"):
            tr.integrate_two_body([1, 0, 0], [0, 1, 0], mu=0.0)


class TestSymplecticVsNonSymplectic:
    def test_leapfrog_energy_error_is_bounded_in_time(self):
        r"""Measured: 2.682e-05 at 5, 20, 80 and 320 orbits: flat to three
        significant figures. A symplectic scheme's energy error oscillates
        rather than accumulating."""
        x0, v0 = tr.periapsis_state(1.0, 0.5, MU)
        period = tr.orbital_period(1.0, MU)
        drifts = []
        for n_orbits in (5, 20, 80):
            traj = tr.integrate_two_body(x0, v0, mu=MU, dt=period / 2000,
                                         n_steps=2000 * n_orbits, method="leapfrog")
            drifts.append(traj.energy_drift)
        growth = max(drifts) / min(drifts)
        assert growth < 1.5, (
            f"leapfrog energy drift grew {growth:.2f}x from 5 to 80 orbits "
            f"({drifts}); a symplectic scheme must stay bounded"
        )

    def test_rk4_energy_error_grows_with_run_length(self):
        r"""The contrast: RK4 is far more accurate per step and its error is
        *secular*. Measured 1.5e-10 at 5 orbits, 3.1e-09 at 320."""
        x0, v0 = tr.periapsis_state(1.0, 0.5, MU)
        period = tr.orbital_period(1.0, MU)
        drifts = []
        for n_orbits in (5, 80):
            traj = tr.integrate_two_body(x0, v0, mu=MU, dt=period / 2000,
                                         n_steps=2000 * n_orbits, method="rk4")
            drifts.append(traj.energy_drift)
        assert drifts[1] > 3.0 * drifts[0], (
            f"RK4 energy drift went {drifts[0]:.3e} -> {drifts[1]:.3e} over a 16x "
            "longer run; it should grow roughly linearly in time"
        )

    def test_rk4_is_more_accurate_per_step_than_leapfrog(self):
        """Symplectic does not mean accurate; the two properties are separate."""
        x0, v0 = tr.periapsis_state(1.0, 0.5, MU)
        period = tr.orbital_period(1.0, MU)
        kwargs = dict(mu=MU, dt=period / 2000, n_steps=2000 * 5)
        lf = tr.integrate_two_body(x0, v0, method="leapfrog", **kwargs).energy_drift
        rk = tr.integrate_two_body(x0, v0, method="rk4", **kwargs).energy_drift
        assert rk < lf, f"rk4 drift {rk:.3e} should be below leapfrog's {lf:.3e}"


class TestConvergenceOrders:
    @pytest.mark.parametrize("method,expected", [("leapfrog", 2.0), ("rk4", 4.0)])
    def test_order_of_the_two_body_integrators(self, method, expected):
        r"""Fitted from the final-position error over one orbit at ``e = 0.3``.
        Measured: leapfrog 2.000 (R² = 1.00000), RK4 4.118 (R² = 0.9998)."""
        x0, v0 = tr.periapsis_state(1.0, 0.3, MU)
        period = tr.orbital_period(1.0, MU)
        reference = tr.integrate_two_body(
            x0, v0, mu=MU, dt=period / 400_000, n_steps=400_000, method="rk4"
        ).positions[-1]

        n_steps = [500, 1000, 2000, 4000, 8000]
        errors = []
        for n in n_steps:
            final = tr.integrate_two_body(x0, v0, mu=MU, dt=period / n,
                                          n_steps=n, method=method).positions[-1]
            errors.append(float(np.linalg.norm(final - reference)))

        study = V.convergence_order([period / n for n in n_steps], errors,
                                     expected=expected)
        assert study.matches_expected(tol=0.3), (
            f"{method}: fitted order {study.order:.3f}, expected {expected}\n"
            f"{study.table()}"
        )
        assert study.r_squared > 0.995, (
            f"{method}: R^2 = {study.r_squared:.5f}; the errors do not follow a "
            "clean power law"
        )

    def test_boris_is_second_order_in_position(self):
        r"""The Boris push conserves :math:`|\mathbf v|` exactly but is still
        only second-order accurate in the trajectory itself."""
        period = 2.0 * math.pi
        b = [0.0, 0.0, 1.0]
        x0 = np.array([1.0, 0.0, 0.0])
        v0 = np.array([0.0, 1.0, 0.0])
        # Exact solution after one period: back to the start.
        n_steps = [200, 400, 800, 1600, 3200]
        errors = []
        for n in n_steps:
            traj = tr.integrate_charged_particle(x0, v0, b_field=b, dt=period / n,
                                                 n_steps=n, method="boris")
            errors.append(float(np.linalg.norm(traj.positions[-1] - x0)))
        study = V.convergence_order([period / n for n in n_steps], errors, expected=2.0)
        assert study.matches_expected(tol=0.3), (
            f"Boris fitted order {study.order:.3f}, expected 2\n{study.table()}"
        )


class TestTrajectoryDiagnostics:
    def test_n_steps_and_shapes(self):
        traj = tr.integrate_two_body([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], n_steps=50)
        assert traj.n_steps == 50
        assert traj.positions.shape == (51, 3)
        assert traj.velocities.shape == (51, 3)
        assert traj.energy.shape == (51,)
        assert traj.angular_momentum.shape == (51, 3)

    def test_times_are_uniform(self):
        traj = tr.integrate_two_body([1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                     dt=0.01, n_steps=100)
        np.testing.assert_allclose(np.diff(traj.times), 0.01, rtol=1e-12)

    def test_drifts_are_zero_for_a_single_step_at_the_start(self):
        traj = tr.integrate_two_body([1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                     dt=1e-9, n_steps=1)
        assert traj.energy_drift < 1e-12
        assert traj.angular_momentum_drift < 1e-12
