#!/usr/bin/env python3
"""Generate the figures in ``figures/``.

Each figure is a *verification* figure: it shows a numerical result next to the
closed form or conserved quantity it is being checked against, so the plot
carries the comparison rather than just the simulation. A figure that only
shows a trajectory tells the reader nothing they can check.

    python scripts/make_figures.py
    python scripts/make_figures.py --only trajectories
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cpv import photonics as ph        # noqa: E402
from cpv import quantum_spin as qs     # noqa: E402
from cpv import statistical as st      # noqa: E402
from cpv import trajectories as tr     # noqa: E402
from cpv import verification as V      # noqa: E402

FIGURES = Path(__file__).resolve().parent.parent / "figures"
DPI = 140

plt.rcParams.update({
    "figure.constrained_layout.use": True,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.frameon": False,
})


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def figure_quantum_spin() -> list[Path]:
    paths = []

    # (a) Heisenberg chain spectrum vs field, with the exact two-site levels.
    fields = np.linspace(0.0, 3.0, 121)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    spectra = np.array([qs.eigensystem(qs.heisenberg_chain(2, coupling=1.0, field=h))[0]
                        for h in fields])
    for k in range(4):
        axes[0].plot(fields, spectra[:, k], lw=1.4,
                     label="numerical" if k == 0 else None)
    exact = np.column_stack([np.full_like(fields, -0.75)]
                            + [0.25 - fields * m for m in (1.0, 0.0, -1.0)])
    for k in range(4):
        axes[0].plot(fields, exact[:, k], "k--", lw=0.8, dashes=(4, 3),
                     label="exact $-3J/4$, $J/4 - hm$" if k == 0 else None)
    axes[0].set_xlabel("field $h$")
    axes[0].set_ylabel("energy")
    axes[0].set_title(f"Two spins: Zeeman splitting\n"
                      f"max deviation {np.max(np.abs(spectra - exact)):.1e}")
    axes[0].legend(loc="lower left")

    # (b) Rabi precession against the closed form.
    field = 1.7
    times = np.linspace(0.0, 6.0, 400)
    states = qs.evolve(qs.heisenberg_chain(1, field=field),
                       np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0), times)
    sx, sy, _ = qs.spin_operators(0.5)
    got_x = np.array([qs.expectation(sx, s).real for s in states])
    got_y = np.array([qs.expectation(sy, s).real for s in states])
    axes[1].plot(times, got_x, lw=1.6, label=r"$\langle S^x \rangle$ numerical")
    axes[1].plot(times, 0.5 * np.cos(field * times), "k--", lw=0.8, dashes=(4, 3),
                 label=r"$\frac{1}{2}\cos(ht)$")
    axes[1].plot(times, got_y, lw=1.6, label=r"$\langle S^y \rangle$ numerical")
    axes[1].plot(times, -0.5 * np.sin(field * times), "k:", lw=0.9,
                 label=r"$-\frac{1}{2}\sin(ht)$")
    error = max(np.max(np.abs(got_x - 0.5 * np.cos(field * times))),
                np.max(np.abs(got_y + 0.5 * np.sin(field * times))))
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("expectation")
    axes[1].set_title(f"Single spin precession, $h = {field}$\n"
                      f"max deviation {error:.1e}")
    axes[1].legend(loc="lower left", ncol=2, fontsize=7)
    paths.append(_save(fig, "01-quantum-spin"))

    # (c) Ground-state probabilities of a 4-site chain.
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    _, psi = qs.ground_state(qs.heisenberg_chain(4, coupling=1.0, periodic=True))
    probs = qs.state_probabilities(psi)
    labels = qs.basis_labels(4)
    order = np.argsort(probs)[::-1]
    ax.bar(range(len(probs)), probs[order], color="#3b6ea5")
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels([labels[i] for i in order], rotation=90, fontsize=6)
    ax.set_ylabel("probability")
    ax.set_title("Heisenberg 4-site ring, ground state in the product basis "
                 f"(sum = {probs.sum():.12f})")
    paths.append(_save(fig, "01-quantum-spin-ground-state"))
    return paths


# ---------------------------------------------------------------------------
def figure_statistical() -> list[Path]:
    temperatures = np.linspace(0.08, 6.0, 300)
    betas = 1.0 / temperatures
    n_sites = 8
    energies, mags = st.ising_spectrum(n_sites, coupling=1.0, field=0.0, periodic=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))

    # (a) lnZ: brute force vs transfer matrix.
    brute = np.array([st.log_partition_function(energies, b) for b in betas])
    transfer = np.array([st.ising_log_z_transfer(n_sites, b, coupling=1.0) for b in betas])
    axes[0].plot(temperatures, brute, lw=2.2, label=r"brute force ($2^8$ states)")
    axes[0].plot(temperatures, transfer, "k--", lw=0.9, dashes=(4, 3),
                 label="transfer matrix")
    axes[0].set_xlabel("temperature $T$")
    axes[0].set_ylabel(r"$\ln Z$")
    axes[0].set_title("Two independent methods\n"
                      f"max abs difference {np.max(np.abs(brute - transfer)):.1e}")
    axes[0].legend(loc="upper right")

    # (b) U, S with their limits marked.
    u = np.array([st.internal_energy(energies, b) for b in betas])
    s = np.array([st.entropy(energies, b) for b in betas])
    axes[1].plot(temperatures, u, lw=1.8, label="$U$")
    axes[1].plot(temperatures, s, lw=1.8, label="$S$")
    axes[1].axhline(float(energies.min()), color="0.4", ls=":", lw=1.0)
    axes[1].axhline(n_sites * math.log(2.0), color="0.4", ls=":", lw=1.0)
    axes[1].axhline(math.log(2.0), color="0.4", ls="--", lw=0.8)
    axes[1].annotate(r"$E_0 = -N$", (4.6, float(energies.min()) + 0.3), fontsize=7,
                     color="0.3")
    axes[1].annotate(r"$N\ln 2$", (4.6, n_sites * math.log(2.0) - 0.6), fontsize=7,
                     color="0.3")
    axes[1].annotate(r"$\ln 2$", (0.2, math.log(2.0) + 0.3), fontsize=7, color="0.3")
    axes[1].set_xlabel("temperature $T$")
    axes[1].set_title("Limiting cases, $N = 8$ ring, $h = 0$")
    axes[1].legend(loc="center right")

    # (c) heat capacity, two derivations.
    c_fluct = np.array([st.heat_capacity(energies, b) for b in betas])
    c_deriv = np.array([st.heat_capacity_from_derivative(energies, b) for b in betas])
    axes[2].plot(temperatures, c_fluct, lw=2.2,
                 label=r"$\beta^2 \mathrm{Var}(E)$")
    axes[2].plot(temperatures, c_deriv, "k--", lw=0.9, dashes=(4, 3),
                 label=r"$\mathrm{d}U/\mathrm{d}T$")
    rel = np.max(np.abs(c_fluct - c_deriv) / np.maximum(np.abs(c_fluct), 1e-12))
    axes[2].set_xlabel("temperature $T$")
    axes[2].set_ylabel("$C$")
    axes[2].set_title("Fluctuation formula vs derivative\n"
                      f"max relative difference {rel:.1e}")
    axes[2].legend(loc="upper right")
    path_a = _save(fig, "02-statistical-physics")

    # (d) Curie law at zero coupling.
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    energies0, mags0 = st.ising_spectrum(6, coupling=0.0, field=0.7)
    b_grid = np.linspace(0.05, 5.0, 200)
    got = np.array([st.magnetization(energies0, mags0, b) for b in b_grid])
    ax.plot(b_grid, got, lw=2.2, label="numerical, $N = 6$, $J = 0$")
    ax.plot(b_grid, np.tanh(b_grid * 0.7), "k--", lw=0.9, dashes=(4, 3),
            label=r"$\tanh(\beta h)$")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("magnetisation per spin")
    ax.set_title("Zero-coupling limit, $h = 0.7$\n"
                 f"max abs deviation {np.max(np.abs(got - np.tanh(b_grid * 0.7))):.1e}")
    ax.legend(loc="lower right")
    return [path_a, _save(fig, "02-statistical-curie")]


# ---------------------------------------------------------------------------
def figure_photonics() -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))

    # (a) MZI fringes vs the closed form.
    phases = np.linspace(0.0, 4.0 * math.pi, 400)
    scan = ph.mach_zehnder_scan(phases)
    axes[0].plot(phases, scan[:, 0], lw=2.0, label="port 0, matrix product")
    axes[0].plot(phases, np.sin(phases / 2) ** 2, "k--", lw=0.9, dashes=(4, 3),
                 label=r"$\sin^2(\varphi/2)$")
    axes[0].plot(phases, scan[:, 1], lw=2.0, label="port 1, matrix product")
    axes[0].plot(phases, np.cos(phases / 2) ** 2, "k:", lw=0.9,
                 label=r"$\cos^2(\varphi/2)$")
    want = np.column_stack([np.sin(phases / 2) ** 2, np.cos(phases / 2) ** 2])
    axes[0].set_xlabel(r"internal phase $\varphi$")
    axes[0].set_ylabel("intensity")
    axes[0].set_title("Balanced Mach-Zehnder\n"
                      f"max deviation {np.max(np.abs(scan - want)):.1e}")
    axes[0].legend(loc="center right", fontsize=7)

    # (b) visibility vs splitting ratio.
    thetas = np.linspace(0.02, math.pi / 2 - 0.02, 120)
    measured, closed = [], []
    for theta in thetas:
        fringe = ph.mach_zehnder_scan(np.linspace(0.0, 2 * math.pi, 801),
                                       theta_in=float(theta))[:, 0]
        measured.append(ph.visibility(fringe))
        t, r = ph.splitting_ratio(float(theta))
        closed.append(ph.two_beam_visibility(t, r))
    measured = np.array(measured)
    closed = np.array(closed)
    axes[1].plot(np.sin(thetas) ** 2, measured, lw=2.2, label="from the matrix scan")
    axes[1].plot(np.sin(thetas) ** 2, closed, "k--", lw=0.9, dashes=(4, 3),
                 label=r"$2\sqrt{I_1 I_2}/(I_1 + I_2)$")
    axes[1].set_xlabel("first splitter reflectivity $R$")
    axes[1].set_ylabel("fringe visibility")
    axes[1].set_title("Visibility vs imbalance\n"
                      f"max deviation {np.max(np.abs(measured - closed)):.1e}")
    axes[1].legend(loc="lower center")

    # (c) the HOM dip.
    thetas = np.linspace(0.0, math.pi / 2, 300)
    coincidence = np.array([ph.two_photon_output(ph.beam_splitter(float(t))).p_11
                            for t in thetas])
    t_vals = np.cos(thetas) ** 2
    r_vals = np.sin(thetas) ** 2
    axes[2].plot(t_vals, coincidence, lw=2.2, label="permanent of $B(\\theta)$")
    axes[2].plot(t_vals, (t_vals - r_vals) ** 2, "k--", lw=0.9, dashes=(4, 3),
                 label=r"$(T - R)^2$")
    axes[2].axvline(0.5, color="0.5", ls=":", lw=1.0)
    axes[2].annotate("HOM dip:\nexact zero", (0.5, 0.35), fontsize=7,
                     ha="center", color="0.3")
    axes[2].set_xlabel("transmissivity $T$")
    axes[2].set_ylabel("$P(1,1)$")
    axes[2].set_title("Two-photon coincidence\n"
                      f"$P(1,1)$ at 50:50 = "
                      f"{ph.two_photon_output(ph.beam_splitter()).p_11:.1e}")
    axes[2].legend(loc="upper center")
    return [_save(fig, "03-photonics")]


# ---------------------------------------------------------------------------
def figure_trajectories() -> list[Path]:
    paths = []
    period = 2.0 * math.pi

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))

    # (a) gyro-orbit against the analytic circle.
    b_z, charge, mass, v_perp = 1.0, 1.0, 1.0, 1.0
    r_c = tr.gyroradius(charge, [0, 0, b_z], mass, v_perp)
    x0 = np.array([r_c, 0.0, 0.0])
    v0 = np.array([0.0, v_perp, 0.3])
    centre = tr.guiding_centre(x0, v0, [0, 0, b_z], charge, mass)
    traj = tr.integrate_charged_particle(x0, v0, b_field=[0, 0, b_z],
                                         charge=charge, mass=mass,
                                         dt=period / 400, n_steps=800, method="boris")
    axes[0].plot(traj.positions[:, 0], traj.positions[:, 1], lw=1.6, label="Boris push")
    angle = np.linspace(0.0, 2 * math.pi, 200)
    axes[0].plot(centre[0] + r_c * np.cos(angle), centre[1] + r_c * np.sin(angle),
                 "k--", lw=0.9, dashes=(4, 3), label=r"circle of radius $r_c$")
    axes[0].plot(*centre[:2], "k+", ms=8, label="guiding centre")
    radii = np.linalg.norm(traj.positions - centre, axis=1)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[0].set_title(f"Gyration, $r_c = {r_c:g}$\n"
                      f"radius spread {np.ptp(radii) / r_c:.1e} of $r_c$")
    axes[0].legend(loc="upper right", fontsize=7)

    # (b) E x B drift.
    e_field = [1.0, 0.0, 0.0]
    b_field = [0.0, 0.0, 2.0]
    gyroperiod = 2.0 * math.pi / 2.0
    drift_traj = tr.integrate_charged_particle(
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], e_field=e_field, b_field=b_field,
        dt=gyroperiod / 500, n_steps=500 * 6, method="boris")
    axes[1].plot(drift_traj.positions[:, 0], drift_traj.positions[:, 1], lw=1.4,
                 label="cycloid")
    v_d = tr.exb_drift(e_field, b_field)
    axes[1].plot(v_d[0] * drift_traj.times, v_d[1] * drift_traj.times, "k--", lw=1.0,
                 dashes=(4, 3), label=r"$\mathbf{E}\times\mathbf{B}/B^2$")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$y$")
    axes[1].set_title("Guiding-centre drift from rest")
    axes[1].legend(loc="lower left", fontsize=7)

    # (c) Kepler orbit with r_min / r_max marked.
    a, e = 1.0, 0.6
    xk, vk = tr.periapsis_state(a, e)
    el = tr.orbit_elements(xk, vk)
    orbit = tr.integrate_two_body(xk, vk, dt=el.period / 4000, n_steps=4000 * 2,
                                  method="leapfrog")
    axes[2].plot(orbit.positions[:, 0], orbit.positions[:, 1], lw=1.4, label="leapfrog")
    axes[2].plot(0, 0, "ko", ms=5, label="focus")
    for radius, name in ((el.periapsis, "$a(1-e)$"), (el.apoapsis, "$a(1+e)$")):
        axes[2].plot(radius * np.cos(angle), radius * np.sin(angle), ":", lw=0.8,
                     color="0.5")
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("$x$")
    axes[2].set_ylabel("$y$")
    axes[2].set_title(f"Kepler orbit, $e = {e}$\n"
                      f"$r$ in [{orbit.radii.min():.6f}, {orbit.radii.max():.6f}], "
                      f"exact [{el.periapsis:.4f}, {el.apoapsis:.4f}]")
    axes[2].legend(loc="upper right", fontsize=7)
    paths.append(_save(fig, "04-trajectories"))

    # (d) the verification figure: conservation and convergence.
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    xk, vk = tr.periapsis_state(1.0, 0.5)
    for method, style in (("leapfrog", "-"), ("rk4", "--")):
        traj = tr.integrate_two_body(xk, vk, dt=period / 2000, n_steps=2000 * 60,
                                     method=method)
        rel = np.abs((traj.energy - traj.energy[0]) / traj.energy[0])
        axes[0].semilogy(traj.times / period, np.maximum(rel, 1e-18), style, lw=1.2,
                         label=method)
    axes[0].set_xlabel("orbits")
    axes[0].set_ylabel(r"$|E(t) - E(0)| / |E(0)|$")
    axes[0].set_title("Symplectic vs non-symplectic\n"
                      "leapfrog bounded, RK4 secular")
    axes[0].legend(loc="lower right")

    xk, vk = tr.periapsis_state(1.0, 0.3)
    reference = tr.integrate_two_body(xk, vk, dt=period / 400_000,
                                      n_steps=400_000, method="rk4").positions[-1]
    n_steps = [500, 1000, 2000, 4000, 8000]
    for method, expected, marker in (("leapfrog", 2.0, "o"), ("rk4", 4.0, "s")):
        errors = [
            float(np.linalg.norm(
                tr.integrate_two_body(xk, vk, dt=period / n, n_steps=n,
                                      method=method).positions[-1] - reference))
            for n in n_steps
        ]
        steps = [period / n for n in n_steps]
        study = V.convergence_order(steps, errors, expected=expected)
        axes[1].loglog(steps, errors, marker=marker, lw=1.3,
                       label=f"{method}: fitted {study.order:.2f} (expect {expected:g})")
        guide = np.array(errors[0]) * (np.array(steps) / steps[0]) ** expected
        axes[1].loglog(steps, guide, "k:", lw=0.7)
    axes[1].set_xlabel(r"step $\Delta t$")
    axes[1].set_ylabel("final-position error")
    axes[1].set_title("Convergence order\n(dotted: the promised slope)")
    axes[1].legend(loc="lower right", fontsize=7)
    paths.append(_save(fig, "04-trajectories-verification"))
    return paths


BUILDERS = {
    "quantum_spin": figure_quantum_spin,
    "statistical": figure_statistical,
    "photonics": figure_photonics,
    "trajectories": figure_trajectories,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(BUILDERS), action="append")
    args = parser.parse_args()

    selected = args.only or sorted(BUILDERS)
    written = []
    for name in selected:
        written.extend(BUILDERS[name]())
    for path in written:
        print(f"wrote {path.relative_to(path.parent.parent)}")
    print(f"{len(written)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
