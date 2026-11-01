r"""Trajectories: a charged particle in electromagnetic fields, and two bodies
under gravity.

The two problems are grouped because they are verified the same way: by
conserved quantities and by closed-form limits, not by eyeballing a plot.

Charged particle
----------------
Non-relativistic Lorentz force,

.. math::
    m\dot{\mathbf v} = q\left(\mathbf E + \mathbf v \times \mathbf B\right).

Three exact results to check against:

* **Gyration.** With :math:`\mathbf E = 0` and uniform :math:`\mathbf B`, the
  speed is constant and the motion perpendicular to :math:`\mathbf B` is a
  circle of angular frequency :math:`\omega_c = qB/m` and radius
  :math:`r_c = m v_\perp /(qB)`. The parallel velocity is untouched.
* **E x B drift.** With both fields uniform and perpendicular, the guiding
  centre moves at :math:`\mathbf v_d = \mathbf E \times \mathbf B / B^2`,
  independently of charge, mass and speed.
* **Parallel acceleration.** A field component along :math:`\mathbf B`
  accelerates the particle uniformly along it.

Two integrators are provided. ``rk4`` is fourth-order accurate and dissipates
energy slowly. ``boris`` is the Boris push in drift-kick-drift order: half a
position drift, a half electric kick, a *rotation* of the velocity by the
magnetic field, a second half kick, and the second half drift. The rotation is
applied as an exact finite rotation rather than as a derivative, so for
:math:`\mathbf E = 0` the scheme conserves :math:`|\mathbf v|` to round-off at
**any** step size: 7e-15 at four steps per orbit, where RK4 loses 55 % of the
speed. The symmetric placement of the two half drifts matters as much as the
rotation does; see :func:`_boris_step` for the measurement.

Two-body problem
----------------
Reduced to the relative coordinate with
:math:`\mu = G(m_1 + m_2)`, so the relative separation obeys

.. math:: \ddot{\mathbf r} = -\mu \mathbf r / r^3 .

Conserved: specific energy :math:`\varepsilon = v^2/2 - \mu/r` and specific
angular momentum :math:`\mathbf L = \mathbf r \times \mathbf v`. The orbital
elements follow,

.. math::
    a = -\frac{\mu}{2\varepsilon}, \qquad
    e = \sqrt{1 + \frac{2 \varepsilon L^2}{\mu^2}}, \qquad
    T = 2\pi\sqrt{a^3/\mu},

and the radius is confined to :math:`[a(1-e), a(1+e)]`. ``leapfrog`` (velocity
Verlet, kick-drift-kick) is symplectic: its energy error oscillates with
amplitude :math:`O(\Delta t^2)` and does **not** accumulate, and for a central
force it conserves :math:`\mathbf L` to round-off because every sub-step is
parallel to either :math:`\mathbf r` or :math:`\mathbf v`. ``rk4`` is more
accurate per step and drifts secularly; both are provided so the difference can
be measured instead of asserted.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = [
    "Trajectory", "gyrofrequency", "gyroradius", "exb_drift", "guiding_centre",
    "lorentz_acceleration", "integrate_charged_particle",
    "OrbitElements", "orbit_elements", "orbital_period",
    "circular_speed", "periapsis_state", "integrate_two_body",
]

INTEGRATORS_EM = ("boris", "rk4")
INTEGRATORS_GRAVITY = ("leapfrog", "rk4")


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """Sampled trajectory with the conserved quantities evaluated along it."""

    times: np.ndarray
    positions: np.ndarray        # (n_steps + 1, 3)
    velocities: np.ndarray       # (n_steps + 1, 3)
    energy: np.ndarray           # (n_steps + 1,)
    angular_momentum: np.ndarray  # (n_steps + 1, 3) or (n_steps + 1,) for speed
    method: str
    dt: float

    @property
    def n_steps(self) -> int:
        return self.times.size - 1

    def _relative_drift(self, series: np.ndarray) -> float:
        reference = series[0]
        scale = max(abs(float(np.max(np.abs(reference)))), 1e-300)
        return float(np.max(np.abs(series - reference)) / scale)

    @property
    def energy_drift(self) -> float:
        r""":math:`\max_t |E(t) - E(0)| / |E(0)|`."""
        return self._relative_drift(self.energy)

    @property
    def angular_momentum_drift(self) -> float:
        r""":math:`\max_t \|L(t) - L(0)\| / \|L(0)\|`."""
        l0 = self.angular_momentum[0]
        scale = max(float(np.linalg.norm(l0)), 1e-300)
        deltas = np.linalg.norm(self.angular_momentum - l0, axis=-1) \
            if self.angular_momentum.ndim == 2 \
            else np.abs(self.angular_momentum - l0)
        return float(np.max(deltas) / scale)

    @property
    def speeds(self) -> np.ndarray:
        return np.linalg.norm(self.velocities, axis=1)

    @property
    def radii(self) -> np.ndarray:
        return np.linalg.norm(self.positions, axis=1)

    @property
    def speed_drift(self) -> float:
        speeds = self.speeds
        return float(np.max(np.abs(speeds - speeds[0])) / max(speeds[0], 1e-300))


# ---------------------------------------------------------------------------
# charged particle
# ---------------------------------------------------------------------------
def gyrofrequency(charge: float, b_field, mass: float) -> float:
    r""":math:`\omega_c = |q| B / m`, in radians per unit time."""
    if mass <= 0.0:
        raise ValueError(f"mass must be positive, got {mass}")
    b_mag = float(np.linalg.norm(np.asarray(b_field, dtype=float)))
    return abs(charge) * b_mag / mass


def gyroradius(charge: float, b_field, mass: float, v_perp: float) -> float:
    r""":math:`r_c = m v_\perp / (|q| B)`."""
    omega = gyrofrequency(charge, b_field, mass)
    if omega == 0.0:
        raise ValueError("gyroradius is undefined for zero magnetic field")
    return abs(v_perp) / omega


def exb_drift(e_field, b_field) -> np.ndarray:
    r""":math:`\mathbf v_d = \mathbf E \times \mathbf B / B^2`."""
    e = np.asarray(e_field, dtype=float)
    b = np.asarray(b_field, dtype=float)
    b_sq = float(b @ b)
    if b_sq == 0.0:
        raise ValueError("E x B drift is undefined for zero magnetic field")
    return np.cross(e, b) / b_sq


def guiding_centre(position, velocity, b_field, charge: float = 1.0,
                   mass: float = 1.0) -> np.ndarray:
    r"""Centre of the gyro-orbit, :math:`\mathbf R = \mathbf r
    + m\,(\mathbf v \times \mathbf B)/(q B^2)`.

    The particle does not orbit the origin, and it does not orbit its own
    starting point either. The centre sits one gyroradius away, on the side
    the Lorentz force points, which flips with the sign of the charge. Starting
    a test particle at ``[r_c, 0, 0]`` with velocity along ``+y`` puts the
    centre at ``[2 r_c, 0, 0]`` for a positive charge and at the origin for a
    negative one. Measuring the orbit radius about this point rather than about
    the origin is what makes the gyroradius check charge-agnostic.
    """
    r_vec = np.asarray(position, dtype=float)
    v_vec = np.asarray(velocity, dtype=float)
    b = np.asarray(b_field, dtype=float)
    b_sq = float(b @ b)
    if b_sq == 0.0:
        raise ValueError("the guiding centre is undefined for zero magnetic field")
    if charge == 0.0:
        raise ValueError("the guiding centre is undefined for a neutral particle")
    return r_vec + mass * np.cross(v_vec, b) / (charge * b_sq)


def lorentz_acceleration(velocity, e_field, b_field, charge: float,
                         mass: float) -> np.ndarray:
    r""":math:`\mathbf a = (q/m)(\mathbf E + \mathbf v \times \mathbf B)`."""
    v = np.asarray(velocity, dtype=float)
    return (charge / mass) * (np.asarray(e_field, dtype=float) + np.cross(v, b_field))


def _boris_step(x, v, e, b, charge, mass, dt):
    """One Boris push, in drift-kick-drift order.

    The velocity update is the standard Boris sequence: a half electric kick,
    an exact finite rotation about ``B``, and a second half kick. The rotation
    is exact for any ``|t|``, which is what makes ``|v|`` conserved for
    ``E = 0`` at any step size.

    The position is advanced in two half drifts *around* that velocity update
    rather than in one full drift after it. The asymmetric arrangement
    (``x += dt * v_new``) is only first-order accurate in the position-velocity
    relation, and it costs a great deal:

    ===============================  ====================  ==================
    arrangement                      energy drift (E,B)    orbit radius error
    ===============================  ====================  ==================
    kick, kick, full drift           1.5e-04               3.1e-03
    half drift, kick, half drift     5.1e-14               3.1e-14
    ===============================  ====================  ==================

    Both are second order in the trajectory as a whole, so a convergence study
    of the final position cannot tell them apart: only the conserved
    quantities can. With the symmetric arrangement the gyro-orbit comes out as
    an exact circle of radius ``r_c`` to round-off.
    """
    qm = charge / mass
    half = 0.5 * dt

    x_half = x + half * v                       # drift

    v_minus = v + qm * e * half                 # half electric kick
    t = qm * b * half
    s = 2.0 * t / (1.0 + float(t @ t))
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)     # exact rotation about B
    v_new = v_plus + qm * e * half              # half electric kick

    return x_half + half * v_new, v_new         # drift


def _rk4_step_em(x, v, e, b, charge, mass, dt):
    def deriv(_x, _v):
        return _v, lorentz_acceleration(_v, e, b, charge, mass)

    k1x, k1v = deriv(x, v)
    k2x, k2v = deriv(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
    k3x, k3v = deriv(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
    k4x, k4v = deriv(x + dt * k3x, v + dt * k3v)
    x_new = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    v_new = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return x_new, v_new


def integrate_charged_particle(position, velocity, *, e_field=(0.0, 0.0, 0.0),
                               b_field=(0.0, 0.0, 1.0), charge: float = 1.0,
                               mass: float = 1.0, dt: float = 1e-3,
                               n_steps: int = 1000,
                               method: str = "boris") -> Trajectory:
    r"""Integrate the Lorentz force in uniform fields.

    ``energy`` records the kinetic energy plus the electrostatic potential
    :math:`-q\,\mathbf E \cdot \mathbf r`, which is conserved for uniform
    static fields. ``angular_momentum`` records :math:`m\,\mathbf r \times
    \mathbf v`, which is *not* conserved here and is reported for inspection
    only.
    """
    if method not in INTEGRATORS_EM:
        raise ValueError(f"method must be one of {INTEGRATORS_EM}, got {method!r}")
    if mass <= 0.0:
        raise ValueError(f"mass must be positive, got {mass}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    e = np.asarray(e_field, dtype=float)
    b = np.asarray(b_field, dtype=float)
    if e.shape != (3,) or b.shape != (3,):
        raise ValueError("e_field and b_field must be 3-vectors")

    x = np.asarray(position, dtype=float).copy()
    v = np.asarray(velocity, dtype=float).copy()
    if x.shape != (3,) or v.shape != (3,):
        raise ValueError("position and velocity must be 3-vectors")

    step = _boris_step if method == "boris" else _rk4_step_em

    xs = np.empty((n_steps + 1, 3))
    vs = np.empty((n_steps + 1, 3))
    xs[0], vs[0] = x, v
    for k in range(n_steps):
        x, v = step(x, v, e, b, charge, mass, dt)
        xs[k + 1], vs[k + 1] = x, v

    kinetic = 0.5 * mass * np.sum(vs * vs, axis=1)
    potential = -charge * (xs @ e)
    return Trajectory(
        times=dt * np.arange(n_steps + 1), positions=xs, velocities=vs,
        energy=kinetic + potential,
        angular_momentum=mass * np.cross(xs, vs),
        method=method, dt=dt,
    )


# ---------------------------------------------------------------------------
# gravitational two-body problem
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class OrbitElements:
    """Orbital elements derived from one state vector."""

    specific_energy: float
    angular_momentum: float
    semi_major_axis: float
    eccentricity: float
    periapsis: float
    apoapsis: float
    period: float
    bound: bool


def orbit_elements(position, velocity, mu: float = 1.0) -> OrbitElements:
    r"""Orbital elements from a relative position and velocity.

    Unbound states (:math:`\varepsilon \ge 0`) get a negative or infinite
    semi-major axis and ``bound=False``; apoapsis and period are then
    ``inf``, which is the honest answer rather than a silently wrapped number.
    """
    r_vec = np.asarray(position, dtype=float)
    v_vec = np.asarray(velocity, dtype=float)
    if r_vec.shape != (3,) or v_vec.shape != (3,):
        raise ValueError("position and velocity must be 3-vectors")
    if mu <= 0.0:
        raise ValueError(f"mu must be positive, got {mu}")

    r = float(np.linalg.norm(r_vec))
    if r == 0.0:
        raise ValueError("position is at the origin; the elements are singular")
    v_sq = float(v_vec @ v_vec)

    eps = 0.5 * v_sq - mu / r
    l_vec = np.cross(r_vec, v_vec)
    l_mag = float(np.linalg.norm(l_vec))

    bound = eps < 0.0
    a = -mu / (2.0 * eps) if eps != 0.0 else math.inf
    # 1 + 2 eps L^2 / mu^2 can go slightly negative for a numerically circular
    # orbit; clamping at zero keeps e real without hiding a genuine error,
    # because a genuinely negative value is O(1e-16) here.
    e = math.sqrt(max(0.0, 1.0 + 2.0 * eps * l_mag ** 2 / mu ** 2))

    if bound:
        periapsis, apoapsis = a * (1.0 - e), a * (1.0 + e)
        period = 2.0 * math.pi * math.sqrt(a ** 3 / mu)
    else:
        periapsis = l_mag ** 2 / (mu * (1.0 + e)) if e > 0 else r
        apoapsis, period = math.inf, math.inf

    return OrbitElements(
        specific_energy=eps, angular_momentum=l_mag, semi_major_axis=a,
        eccentricity=e, periapsis=periapsis, apoapsis=apoapsis,
        period=period, bound=bound,
    )


def orbital_period(a: float, mu: float = 1.0) -> float:
    r"""Kepler's third law, :math:`T = 2\pi\sqrt{a^3/\mu}`."""
    if a <= 0.0:
        raise ValueError(f"semi-major axis must be positive, got {a}")
    if mu <= 0.0:
        raise ValueError(f"mu must be positive, got {mu}")
    return 2.0 * math.pi * math.sqrt(a ** 3 / mu)


def circular_speed(r: float, mu: float = 1.0) -> float:
    r""":math:`v = \sqrt{\mu/r}` for a circular orbit."""
    if r <= 0.0:
        raise ValueError(f"radius must be positive, got {r}")
    return math.sqrt(mu / r)


def periapsis_state(a: float, e: float, mu: float = 1.0):
    r"""Position and velocity at periapsis of an orbit with elements ``(a, e)``.

    Vis-viva at :math:`r_p = a(1-e)` gives
    :math:`v_p = \sqrt{\frac{\mu}{a}\frac{1+e}{1-e}}`, directed perpendicular
    to the radius.
    """
    if not 0.0 <= e < 1.0:
        raise ValueError(f"eccentricity must be in [0, 1), got {e}")
    if a <= 0.0:
        raise ValueError(f"semi-major axis must be positive, got {a}")
    r_p = a * (1.0 - e)
    v_p = math.sqrt(mu / a * (1.0 + e) / (1.0 - e))
    return np.array([r_p, 0.0, 0.0]), np.array([0.0, v_p, 0.0])


def _gravity_acceleration(r_vec, mu):
    r = float(np.linalg.norm(r_vec))
    return -mu * r_vec / (r ** 3)


def _rk4_step_gravity(x, v, mu, dt):
    def deriv(_x, _v):
        return _v, _gravity_acceleration(_x, mu)

    k1x, k1v = deriv(x, v)
    k2x, k2v = deriv(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
    k3x, k3v = deriv(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
    k4x, k4v = deriv(x + dt * k3x, v + dt * k3v)
    return (x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
            v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v))


def integrate_two_body(position, velocity, *, mu: float = 1.0, dt: float = 1e-3,
                       n_steps: int = 1000, method: str = "leapfrog") -> Trajectory:
    r"""Integrate the relative two-body motion.

    ``energy`` is the specific energy :math:`v^2/2 - \mu/r` and
    ``angular_momentum`` is :math:`\mathbf r \times \mathbf v`, both evaluated
    at every output step so :attr:`Trajectory.energy_drift` and
    :attr:`Trajectory.angular_momentum_drift` are available without
    recomputation.
    """
    if method not in INTEGRATORS_GRAVITY:
        raise ValueError(f"method must be one of {INTEGRATORS_GRAVITY}, got {method!r}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if mu <= 0.0:
        raise ValueError(f"mu must be positive, got {mu}")

    x = np.asarray(position, dtype=float).copy()
    v = np.asarray(velocity, dtype=float).copy()
    if x.shape != (3,) or v.shape != (3,):
        raise ValueError("position and velocity must be 3-vectors")
    if np.linalg.norm(x) == 0.0:
        raise ValueError("the particle starts at the singularity")

    xs = np.empty((n_steps + 1, 3))
    vs = np.empty((n_steps + 1, 3))
    xs[0], vs[0] = x, v

    if method == "leapfrog":
        half = 0.5 * dt
        a = _gravity_acceleration(x, mu)
        for k in range(n_steps):
            v = v + half * a          # kick
            x = x + dt * v            # drift
            a = _gravity_acceleration(x, mu)
            v = v + half * a          # kick
            xs[k + 1], vs[k + 1] = x, v
    else:
        for k in range(n_steps):
            x, v = _rk4_step_gravity(x, v, mu, dt)
            xs[k + 1], vs[k + 1] = x, v

    radii = np.linalg.norm(xs, axis=1)
    energy = 0.5 * np.sum(vs * vs, axis=1) - mu / radii
    return Trajectory(
        times=dt * np.arange(n_steps + 1), positions=xs, velocities=vs,
        energy=energy, angular_momentum=np.cross(xs, vs), method=method, dt=dt,
    )
