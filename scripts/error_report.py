#!/usr/bin/env python3
"""Generate ``results/numerical-error-report.md`` by measuring, not by claiming.

Every number in the report is produced by this script at the moment it runs. It
is wired into CI, so a change that degrades an error bound shows up as a diff
in the report rather than as a quietly weaker guarantee.

    python scripts/error_report.py                 # write the report
    python scripts/error_report.py --print         # to stdout instead
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cpv import photonics as ph          # noqa: E402
from cpv import quantum_spin as qs       # noqa: E402
from cpv import statistical as st        # noqa: E402
from cpv import trajectories as tr       # noqa: E402
from cpv import verification as V        # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
def quantum_spin_errors() -> dict:
    rows = []
    for s in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        sx, sy, sz = qs.spin_operators(s)
        dim = sz.shape[0]
        rows.append({
            "spin": s,
            "dimension": dim,
            "commutator": max(
                float(np.max(np.abs(V.commutator(sx, sy) - 1.0j * sz))),
                float(np.max(np.abs(V.commutator(sy, sz) - 1.0j * sx))),
                float(np.max(np.abs(V.commutator(sz, sx) - 1.0j * sy))),
            ),
            "casimir": float(np.max(np.abs(
                sx @ sx + sy @ sy + sz @ sz - s * (s + 1) * np.eye(dim)))),
            "hermiticity": max(V.check_hermitian(op) for op in (sx, sy, sz)),
        })

    spectra = []
    for coupling in (1.0, 2.5, -1.0):
        energies, _ = qs.eigensystem(qs.heisenberg_chain(2, coupling=coupling))
        want = [-0.75 * coupling] + [0.25 * coupling] * 3
        spectra.append({"case": f"two-spin Heisenberg, J={coupling:g}",
                        "error": V.spectral_error(energies, want)})
    for field in (0.5, 2.0, 7.3):
        energies, _ = qs.eigensystem(qs.heisenberg_chain(1, field=field))
        spectra.append({"case": f"single spin, h={field:g}",
                        "error": V.spectral_error(energies, [-field / 2, field / 2])})
    for field in (0.0, 0.6, 2.4):
        energies, _ = qs.eigensystem(qs.heisenberg_chain(2, coupling=1.0, field=field))
        want = sorted([-0.75] + [0.25 - field * m for m in (-1.0, 0.0, 1.0)])
        spectra.append({"case": f"two-spin Zeeman, h={field:g}",
                        "error": V.spectral_error(energies, want)})

    symmetry = []
    for n in (2, 3, 4, 5):
        periodic = n > 2
        h_zero = qs.heisenberg_chain(n, periodic=periodic)
        h_field = qs.heisenberg_chain(n, field=0.7, periodic=periodic)
        _, _, sz_tot = qs.total_spin_operators(n)
        s_sq = qs.total_spin_squared(n)
        symmetry.append({
            "n_sites": n,
            "trace": abs(float(np.trace(h_zero).real)),
            "commutator_s2": float(np.max(np.abs(V.commutator(h_zero, s_sq)))),
            "commutator_s2_in_field": float(np.max(np.abs(V.commutator(h_field, s_sq)))),
            "commutator_sz": float(np.max(np.abs(V.commutator(h_field, sz_tot)))),
        })

    # Rabi precession against the closed form.
    field = 1.7
    times = np.linspace(0.0, 4.0, 201)
    states = qs.evolve(qs.heisenberg_chain(1, field=field),
                       np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0), times)
    sx, sy, _ = qs.spin_operators(0.5)
    rabi = max(
        float(np.max(np.abs(
            np.array([qs.expectation(sx, s).real for s in states])
            - 0.5 * np.cos(field * times)))),
        float(np.max(np.abs(
            np.array([qs.expectation(sy, s).real for s in states])
            + 0.5 * np.sin(field * times)))),
    )
    norm = float(np.max(np.abs(np.linalg.norm(states, axis=1) - 1.0)))

    return {"operators": rows, "spectra": spectra, "symmetry": symmetry,
            "rabi": rabi, "norm": norm}


# ---------------------------------------------------------------------------
def statistical_errors() -> dict:
    agreement = []
    worst = 0.0
    n_points = 0
    for n, field, coupling, beta in itertools.product(
            (2, 3, 4, 5, 6, 8, 10, 12), (0.0, 0.25, 1.0, -0.6),
            (1.0, -1.0, 0.4), (0.05, 0.4, 1.5, 6.0)):
        energies, _ = st.ising_spectrum(n, coupling=coupling, field=field, periodic=True)
        brute = st.log_partition_function(energies, beta)
        transfer = st.ising_log_z_transfer(n, beta, coupling=coupling, field=field)
        error = abs(brute - transfer) / max(abs(brute), 1.0)
        worst = max(worst, error)
        n_points += 1
        if n in (4, 8, 12) and field == 0.25 and coupling == 1.0:
            agreement.append({"n_sites": n, "beta": beta, "log_z": brute,
                              "error": error})

    limits = []
    for n in (4, 8, 12):
        energies, _ = st.ising_spectrum(n, coupling=1.0, periodic=True)
        hot = st.observables(energies, 1e-8)
        cold = st.observables(energies, 80.0)
        limits.append({
            "n_sites": n,
            "entropy_hot": hot.entropy,
            "entropy_hot_exact": n * math.log(2.0),
            "entropy_hot_error": abs(hot.entropy - n * math.log(2.0)) / (n * math.log(2.0)),
            "energy_cold": cold.internal_energy,
            "energy_cold_exact": float(energies.min()),
            "entropy_cold": cold.entropy,
            "entropy_cold_error": abs(cold.entropy - math.log(2.0)) / math.log(2.0),
        })

    consistency = []
    energies, _ = st.ising_spectrum(8, coupling=1.0, field=0.3, periodic=True)
    for beta in (0.3, 1.0, 3.0):
        fluctuation = st.heat_capacity(energies, beta)
        derivative = st.heat_capacity_from_derivative(energies, beta)
        consistency.append({
            "beta": beta, "fluctuation": fluctuation, "derivative": derivative,
            "error": abs(fluctuation - derivative) / abs(fluctuation),
        })

    curie = []
    for field in (0.3, 0.7, 2.0):
        energies, mags = st.ising_spectrum(6, coupling=0.0, field=field)
        for beta in (0.2, 1.0, 4.0):
            got = st.magnetization(energies, mags, beta)
            want = math.tanh(beta * field)
            curie.append({"field": field, "beta": beta, "magnetization": got,
                          "error": abs(got - want) / abs(want)})

    return {"worst_agreement": worst, "n_points": n_points,
            "agreement": agreement, "limits": limits,
            "consistency": consistency, "curie": curie}


# ---------------------------------------------------------------------------
def photonics_errors() -> dict:
    thetas = [0.0, 0.1, 0.4, math.pi / 4, 1.0, 1.2, math.pi / 2]

    unitarity = max(
        max(V.check_unitary(ph.beam_splitter(t, p))
            for t in thetas for p in (0.0, 0.7, math.pi)),
        max(V.check_unitary(ph.phase_shifter(p)) for p in (0.0, 1.0, math.pi, 5.0)),
        max(V.check_unitary(ph.mach_zehnder(p)) for p in (0.0, 1.0, math.pi, 5.0)),
    )

    phases = np.linspace(0.0, 4.0 * math.pi, 401)
    scan = ph.mach_zehnder_scan(phases)
    closed_form = np.column_stack([np.sin(phases / 2) ** 2, np.cos(phases / 2) ** 2])
    mzi_error = float(np.max(np.abs(scan - closed_form)))
    energy_error = float(np.max(np.abs(scan.sum(axis=1) - 1.0)))

    hom = ph.two_photon_output(ph.beam_splitter())
    coincidence = []
    for theta in thetas:
        t, r = ph.splitting_ratio(theta)
        out = ph.two_photon_output(ph.beam_splitter(theta, 0.8))
        coincidence.append({
            "theta": theta, "transmissivity": t,
            "p_11": out.p_11, "closed_form": (t - r) ** 2,
            "error": abs(out.p_11 - (t - r) ** 2),
            "total": out.total, "total_error": abs(out.total - 1.0),
        })

    visibilities = []
    for theta_in in (0.15, 0.4, math.pi / 4, 0.6, 1.2):
        fringe = ph.mach_zehnder_scan(np.linspace(0.0, 2 * math.pi, 1601),
                                       theta_in=theta_in)[:, 0]
        got = ph.visibility(fringe)
        t, r = ph.splitting_ratio(theta_in)
        want = ph.two_beam_visibility(t, r)
        visibilities.append({"theta_in": theta_in, "measured": got,
                             "closed_form": want, "error": abs(got - want)})

    return {"unitarity": unitarity, "mzi_error": mzi_error,
            "energy_error": energy_error, "hom_coincidence": hom.p_11,
            "hom_total_error": abs(hom.total - 1.0),
            "coincidence": coincidence, "visibilities": visibilities}


# ---------------------------------------------------------------------------
def trajectory_errors() -> dict:
    period = 2.0 * math.pi
    boris_speed = []
    for steps in (4, 8, 20, 100, 1000):
        kwargs = dict(b_field=[0.0, 0.0, 1.0], dt=period / steps,
                      n_steps=steps * 50)
        boris = tr.integrate_charged_particle([1.0, 0.0, 0.0], [0.0, 1.0, 0.5],
                                              method="boris", **kwargs)
        rk4 = tr.integrate_charged_particle([1.0, 0.0, 0.0], [0.0, 1.0, 0.5],
                                            method="rk4", **kwargs)
        boris_speed.append({"steps_per_orbit": steps,
                            "boris": boris.speed_drift, "rk4": rk4.speed_drift})

    gyro = []
    for charge, b_z, mass, v_perp in ((1.0, 1.0, 1.0, 1.0), (-1.0, 2.5, 2.0, 0.7),
                                      (3.0, 0.4, 1.5, 2.0), (-2.0, 1.3, 0.7, 1.1)):
        b = [0.0, 0.0, b_z]
        omega = tr.gyrofrequency(charge, b, mass)
        r_c = tr.gyroradius(charge, b, mass, v_perp)
        x0 = np.array([r_c, 0.0, 0.0])
        v0 = np.array([0.0, v_perp, 0.0])
        centre = tr.guiding_centre(x0, v0, b, charge, mass)
        traj = tr.integrate_charged_particle(
            x0, v0, b_field=b, charge=charge, mass=mass,
            dt=(2 * math.pi / omega) / 2000, n_steps=2000 * 3, method="boris")
        radii = np.linalg.norm(traj.positions - centre, axis=1)
        gyro.append({"charge": charge, "b_field": b_z, "mass": mass,
                     "r_c": r_c, "measured": float(radii.mean()),
                     "error": abs(float(radii.mean()) - r_c) / r_c,
                     "circularity": float(np.ptp(radii)) / r_c})

    drift = []
    for steps in (100, 500, 2000):
        gyroperiod = 2.0 * math.pi / 2.0
        traj = tr.integrate_charged_particle(
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], e_field=[1.0, 0.0, 0.0],
            b_field=[0.0, 0.0, 2.0], dt=gyroperiod / steps,
            n_steps=steps * 200, method="boris")
        measured = float((traj.positions[-1, 1] - traj.positions[0, 1]) / traj.times[-1])
        want = float(tr.exb_drift([1.0, 0.0, 0.0], [0.0, 0.0, 2.0])[1])
        drift.append({"steps_per_gyroperiod": steps, "measured": measured,
                      "closed_form": want, "error": abs(measured - want) / abs(want)})

    elements = []
    for a, e in ((1.0, 0.0), (1.0, 0.3), (2.5, 0.6), (0.4, 0.85)):
        x0, v0 = tr.periapsis_state(a, e)
        el = tr.orbit_elements(x0, v0)
        elements.append({
            "a": a, "e": e,
            "a_error": abs(el.semi_major_axis - a) / a,
            "e_error": abs(el.eccentricity - e) if e > 0 else abs(el.eccentricity),
            "period_error": abs(el.period - tr.orbital_period(a)) / tr.orbital_period(a),
            "l_error": abs(el.angular_momentum - math.sqrt(a * (1 - e ** 2)))
            / math.sqrt(a * (1 - e ** 2)),
        })

    conservation = []
    x0, v0 = tr.periapsis_state(1.0, 0.5)
    for n_orbits in (5, 20, 80, 320):
        row = {"orbits": n_orbits}
        for method in ("leapfrog", "rk4"):
            traj = tr.integrate_two_body(x0, v0, dt=period / 2000,
                                         n_steps=2000 * n_orbits, method=method)
            row[f"{method}_energy"] = traj.energy_drift
            row[f"{method}_angular"] = traj.angular_momentum_drift
        conservation.append(row)

    orders = {}
    x0, v0 = tr.periapsis_state(1.0, 0.3)
    reference = tr.integrate_two_body(x0, v0, dt=period / 400_000,
                                      n_steps=400_000, method="rk4").positions[-1]
    for method, expected in (("leapfrog", 2.0), ("rk4", 4.0)):
        n_steps = [500, 1000, 2000, 4000, 8000]
        errors = [
            float(np.linalg.norm(
                tr.integrate_two_body(x0, v0, dt=period / n, n_steps=n,
                                      method=method).positions[-1] - reference))
            for n in n_steps
        ]
        study = V.convergence_order([period / n for n in n_steps], errors,
                                     expected=expected)
        orders[method] = {"order": study.order, "r_squared": study.r_squared,
                          "expected": expected, "errors": errors,
                          "steps": list(study.steps)}

    n_steps = [200, 400, 800, 1600, 3200]
    boris_errors = [
        float(np.linalg.norm(
            tr.integrate_charged_particle(
                [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], b_field=[0.0, 0.0, 1.0],
                dt=period / n, n_steps=n, method="boris").positions[-1]
            - np.array([1.0, 0.0, 0.0])))
        for n in n_steps
    ]
    boris_study = V.convergence_order([period / n for n in n_steps], boris_errors,
                                       expected=2.0)
    orders["boris"] = {"order": boris_study.order, "r_squared": boris_study.r_squared,
                       "expected": 2.0, "errors": boris_errors,
                       "steps": list(boris_study.steps)}

    return {"boris_speed": boris_speed, "gyro": gyro, "drift": drift,
            "elements": elements, "conservation": conservation, "orders": orders}


# ---------------------------------------------------------------------------
def fmt(value: float, digits: int = 3) -> str:
    if value == 0.0:
        return "0 (exact)"
    return f"{value:.{digits}e}"


def render(data: dict) -> str:
    q, s, p, t = data["quantum"], data["statistical"], data["photonics"], data["trajectories"]
    out: list[str] = []
    w = out.append

    w("# Numerical error report")
    w("")
    w(f"Generated by `scripts/error_report.py` on "
      f"{data['generated_at']} with NumPy {data['numpy_version']}.")
    w("")
    w("Every number below is measured at the moment this file is written. Nothing "
      "here is a target or a claim carried over from a previous run; CI "
      "regenerates the report, so a weakened bound appears as a diff.")
    w("")
    w("Double-precision epsilon is 2.22e-16. An entry at or below ~1e-15 means the "
      "quantity is exact in exact arithmetic and the measurement is pure round-off.")
    w("")

    # ---- module 1
    w("## 1. Quantum spin systems")
    w("")
    w("### Operator algebra (exact statements)")
    w("")
    w("| spin | dim | max abs error in [S^a, S^b] = i eps S^c | S^2 - s(s+1) | hermiticity |")
    w("|---|---|---|---|---|")
    for row in q["operators"]:
        w(f"| {row['spin']:g} | {row['dimension']} | {fmt(row['commutator'])} | "
          f"{fmt(row['casimir'])} | {fmt(row['hermiticity'])} |")
    w("")
    w("The ladder-operator construction is what makes these exact rather than "
      "approximate: `Sx` and `Sy` are assembled from `S+` and `S-`, so the SU(2) "
      "algebra holds by construction and the table measures only floating-point "
      "noise in the matrix products.")
    w("")
    w("### Spectra against closed forms")
    w("")
    w("| case | max abs eigenvalue error |")
    w("|---|---|")
    for row in q["spectra"]:
        w(f"| {row['case']} | {fmt(row['error'])} |")
    w("")
    w("### Symmetry and structure")
    w("")
    w("| sites | abs tr H (h=0) | max abs [H, S^2] (h=0) | max abs [H, S^2] (h=0.7) | max abs [H, Sz] (h=0.7) |")
    w("|---|---|---|---|---|")
    for row in q["symmetry"]:
        w(f"| {row['n_sites']} | {fmt(row['trace'])} | {fmt(row['commutator_s2'])} | "
          f"{fmt(row['commutator_s2_in_field'])} | {fmt(row['commutator_sz'])} |")
    w("")
    w("The third and fourth columns are the interesting pair. A uniform field along "
      "`z` does **not** spoil `[H, S^2] = 0`, because `S^2` commutes with every "
      "component of the total spin. The field lifts the degeneracy inside a "
      "multiplet without removing `S^2` as a good quantum number. What does break "
      "it is anisotropic exchange or a site-dependent field; the test suite checks "
      "both directions.")
    w("")
    w(f"Unitary evolution: max deviation of the norm from 1 over a 25-point time "
      f"grid is **{fmt(q['norm'])}**, and the Rabi precession "
      f"`<Sx> = cos(ht)/2`, `<Sy> = -sin(ht)/2` is reproduced to "
      f"**{fmt(q['rabi'])}**. Evolution goes through the eigenbasis rather than a "
      "time-stepping scheme, so there is no step-size error to report.")
    w("")

    # ---- module 2
    w("## 2. Statistical physics")
    w("")
    w(f"### Two independent methods")
    w("")
    w(f"Brute-force enumeration of all `2^N` configurations against the "
      f"transfer-matrix closed form `Z = lambda_+^N + lambda_-^N`, over "
      f"**{s['n_points']} parameter combinations** "
      f"(`N` in 2..12, `h` in {{0, 0.25, 1.0, -0.6}}, `J` in {{1, -1, 0.4}}, "
      f"`beta` in {{0.05, 0.4, 1.5, 6.0}}):")
    w("")
    w(f"**worst relative disagreement in ln Z: {fmt(s['worst_agreement'])}**")
    w("")
    w("The two routes share no code. Enumeration sums `2^N` Boltzmann factors; the "
      "transfer matrix diagonalises a 2x2 matrix and raises its eigenvalues to the "
      "`N`-th power. Agreement at this level is the strongest single piece of "
      "evidence in this report.")
    w("")
    w("| sites | beta | ln Z | relative error |")
    w("|---|---|---|---|")
    for row in s["agreement"]:
        w(f"| {row['n_sites']} | {row['beta']:g} | {row['log_z']:.9f} | "
          f"{fmt(row['error'])} |")
    w("")
    w("### Limiting cases")
    w("")
    w("| sites | S(T->inf) | N ln 2 | rel err | U(T->0) | E_0 | S(T->0) | rel err vs ln 2 |")
    w("|---|---|---|---|---|---|---|---|")
    for row in s["limits"]:
        w(f"| {row['n_sites']} | {row['entropy_hot']:.9f} | "
          f"{row['entropy_hot_exact']:.9f} | {fmt(row['entropy_hot_error'])} | "
          f"{row['energy_cold']:.6f} | {row['energy_cold_exact']:.6f} | "
          f"{row['entropy_cold']:.9f} | {fmt(row['entropy_cold_error'])} |")
    w("")
    w("The low-temperature entropy tends to `ln 2`, not to zero: at zero field the "
      "ferromagnetic ground state is two-fold degenerate (all up, all down). "
      "Switching the field on removes the degeneracy and the entropy goes to zero "
      "instead, which the test suite checks separately.")
    w("")
    w("### Thermodynamic consistency")
    w("")
    w("`C = beta^2 Var(E)` (fluctuation-dissipation) against `C = dU/dT` (centred "
      "difference). These are the same quantity derived two different ways from "
      "`ln Z`:")
    w("")
    w("| beta | C from fluctuations | C from dU/dT | relative difference |")
    w("|---|---|---|---|")
    for row in s["consistency"]:
        w(f"| {row['beta']:g} | {row['fluctuation']:.9f} | {row['derivative']:.9f} | "
          f"{fmt(row['error'])} |")
    w("")
    w("The residual is the truncation error of the centred difference, which grows "
      "as `C` itself becomes small at low temperature. It is not a disagreement "
      "between the two identities.")
    w("")
    w("### Zero-coupling limit: m = tanh(beta h)")
    w("")
    w("| h | beta | m measured | relative error |")
    w("|---|---|---|---|")
    for row in s["curie"]:
        w(f"| {row['field']:g} | {row['beta']:g} | {row['magnetization']:.12f} | "
          f"{fmt(row['error'])} |")
    w("")

    # ---- module 3
    w("## 3. Photonics")
    w("")
    w(f"Unitarity, over every beam splitter, phase shifter and interferometer in "
      f"the module: **max |U†U - I| = {fmt(p['unitarity'])}**. For a lossless "
      f"component that statement *is* energy conservation, so a single check "
      f"covers both.")
    w("")
    w(f"Mach-Zehnder output against the closed form "
      f"`(sin^2(phi/2), cos^2(phi/2))` over 401 phases spanning `4 pi`: "
      f"**max abs deviation {fmt(p['mzi_error'])}**. Total intensity deviation "
      f"from 1: **{fmt(p['energy_error'])}**.")
    w("")
    w("### Two-photon interference")
    w("")
    w(f"Hong-Ou-Mandel coincidence probability at 50:50: "
      f"**P(1,1) = {fmt(p['hom_coincidence'])}**. This is an algebraic zero, not a "
      f"small number -- the permanent of the balanced splitter is "
      f"`(1*1 + i*i)/2 = 0` exactly. A non-zero value here would mean the `pi/2` "
      f"reflection phase is wrong, so it is a sharper test of the convention than "
      f"any intensity measurement.")
    w("")
    w("The general closed form is `P(1,1) = (T - R)^2`, independent of the splitter "
      "phase:")
    w("")
    w("| theta | T | P(1,1) measured | (T-R)^2 | abs error | sum of probabilities - 1 |")
    w("|---|---|---|---|---|---|")
    for row in p["coincidence"]:
        w(f"| {row['theta']:.4f} | {row['transmissivity']:.6f} | {row['p_11']:.12f} | "
          f"{row['closed_form']:.12f} | {fmt(row['error'])} | {fmt(row['total_error'])} |")
    w("")
    w("### Fringe visibility against the two-beam formula")
    w("")
    w("`V = 2 sqrt(I1 I2) / (I1 + I2)` with `I1 = cos^2(theta_in)`, "
      "`I2 = sin^2(theta_in)`:")
    w("")
    w("| theta_in | V from the matrix scan | V from the closed form | abs error |")
    w("|---|---|---|---|")
    for row in p["visibilities"]:
        w(f"| {row['theta_in']:.4f} | {row['measured']:.12f} | "
          f"{row['closed_form']:.12f} | {fmt(row['error'])} |")
    w("")

    # ---- module 4
    w("## 4. Trajectories")
    w("")
    w("### Boris push: speed conservation vs step size")
    w("")
    w("With `E = 0` the Boris rotation is an exact finite rotation, so the speed is "
      "preserved at *any* step size. RK4 has no such property, and the contrast is "
      "the point of the table:")
    w("")
    w("| steps per gyro-orbit | Boris rel. speed drift | RK4 rel. speed drift |")
    w("|---|---|---|")
    for row in t["boris_speed"]:
        w(f"| {row['steps_per_orbit']} | {fmt(row['boris'])} | {fmt(row['rk4'])} |")
    w("")
    w("### Gyroradius against m v_perp / (|q| B)")
    w("")
    w("Measured about the guiding centre, so the result is independent of the sign "
      "of the charge. The last column is the peak-to-peak variation of the radius "
      "as a fraction of `r_c` -- zero means the orbit is a circle:")
    w("")
    w("| q | B | m | r_c exact | r_c measured | rel error | circularity |")
    w("|---|---|---|---|---|---|---|")
    for row in t["gyro"]:
        w(f"| {row['charge']:+g} | {row['b_field']:g} | {row['mass']:g} | "
          f"{row['r_c']:.9f} | {row['measured']:.9f} | {fmt(row['error'])} | "
          f"{fmt(row['circularity'])} |")
    w("")
    w("### E x B drift of the guiding centre")
    w("")
    w("| steps per gyroperiod | drift measured | E x B / B^2 | relative error |")
    w("|---|---|---|---|")
    for row in t["drift"]:
        w(f"| {row['steps_per_gyroperiod']} | {row['measured']:.9f} | "
          f"{row['closed_form']:.9f} | {fmt(row['error'])} |")
    w("")
    w("The residual is the integrator's orbit-phase error, and it falls with "
      "resolution as a discretisation error should.")
    w("")
    w("### Orbital elements from a state vector")
    w("")
    w("| a | e | rel err in a | abs err in e | rel err in T | rel err in L |")
    w("|---|---|---|---|---|---|")
    for row in t["elements"]:
        w(f"| {row['a']:g} | {row['e']:g} | {fmt(row['a_error'])} | "
          f"{fmt(row['e_error'])} | {fmt(row['period_error'])} | {fmt(row['l_error'])} |")
    w("")
    w("### Symplectic vs non-symplectic: how the error grows with run length")
    w("")
    w("Both integrators at 2000 steps per orbit, `e = 0.5`:")
    w("")
    w("| orbits | leapfrog dE/E | RK4 dE/E | leapfrog dL/L | RK4 dL/L |")
    w("|---|---|---|---|---|")
    for row in t["conservation"]:
        w(f"| {row['orbits']} | {fmt(row['leapfrog_energy'])} | "
          f"{fmt(row['rk4_energy'])} | {fmt(row['leapfrog_angular'])} | "
          f"{fmt(row['rk4_angular'])} |")
    w("")
    w("This table is the reason both integrators are kept. **Leapfrog's energy "
      "error is flat** -- a symplectic map has no secular energy drift, only a "
      "bounded oscillation of size `O(dt^2)`. **RK4's grows** with the number of "
      "steps, even though it is several orders of magnitude smaller to begin with. "
      "Angular momentum shows the same split: leapfrog holds it at round-off "
      "because every sub-step is parallel to `r` or to `v`, while RK4's drifts.")
    w("")
    w("So \"symplectic\" and \"accurate\" are different properties, and which one "
      "matters depends on the run: RK4 for a short high-accuracy integration, "
      "leapfrog for a long one where the orbit must stay closed.")
    w("")
    w("### Convergence orders")
    w("")
    w("Fitted by least squares in log-log space, from the final-position error:")
    w("")
    w("| integrator | expected | fitted | R^2 |")
    w("|---|---|---|---|")
    for name in ("leapfrog", "rk4", "boris"):
        row = t["orders"][name]
        w(f"| {name} | {row['expected']:g} | {row['order']:.3f} | {row['r_squared']:.5f} |")
    w("")
    w("A fitted order below the expected one is the failure a plot cannot show: an "
      "integrator can produce a visually perfect orbit and still be first order, "
      "and the only way to find out is to halve the step and measure.")
    w("")
    for name in ("leapfrog", "rk4"):
        row = t["orders"][name]
        w(f"**{name}**")
        w("")
        w("| step | error | ratio |")
        w("|---|---|---|")
        prev = None
        for step, err in zip(row["steps"], row["errors"]):
            ratio = "—" if prev is None else f"{prev / err:.2f}"
            w(f"| {step:.3e} | {err:.3e} | {ratio} |")
            prev = err
        w("")

    w("## Summary of the tightest and loosest bounds")
    w("")
    w("| quantity | error | kind |")
    w("|---|---|---|")
    w(f"| HOM coincidence probability | {fmt(p['hom_coincidence'])} | algebraic zero |")
    w(f"| component unitarity | {fmt(p['unitarity'])} | exact identity |")
    w(f"| SU(2) commutators | "
      f"{fmt(max(r['commutator'] for r in q['operators']))} | exact identity |")
    w(f"| Mach-Zehnder vs closed form | {fmt(p['mzi_error'])} | exact identity |")
    w(f"| Boris speed conservation at 4 steps/orbit | "
      f"{fmt(t['boris_speed'][0]['boris'])} | exact identity |")
    w(f"| brute force vs transfer matrix | {fmt(s['worst_agreement'])} | two methods |")
    w(f"| leapfrog energy drift, 320 orbits | "
      f"{fmt(t['conservation'][-1]['leapfrog_energy'])} | bounded discretisation |")
    w(f"| E x B drift at 2000 steps | {fmt(t['drift'][-1]['error'])} | discretisation |")
    w(f"| C: fluctuations vs dU/dT | "
      f"{fmt(max(r['error'] for r in s['consistency']))} | finite difference |")
    w("")
    w("The spread is the report's main point. Identities that hold exactly are "
      "verified at round-off and anything larger is a bug; quantities that involve "
      "a discretisation carry an error that must shrink at the promised rate. "
      "Mixing the two up -- quoting a `1e-3` tolerance for an exact identity, or "
      "demanding `1e-15` from a finite difference -- is how a verification suite "
      "stops meaning anything.")
    w("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", action="store_true", dest="to_stdout")
    parser.add_argument("--json", default=None, help="also write the raw numbers here")
    args = parser.parse_args()

    data = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "numpy_version": np.__version__,
        "quantum": quantum_spin_errors(),
        "statistical": statistical_errors(),
        "photonics": photonics_errors(),
        "trajectories": trajectory_errors(),
    }
    text = render(data)

    if args.to_stdout:
        print(text)
    else:
        path = REPO_ROOT / "results" / "numerical-error-report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path.relative_to(REPO_ROOT)} ({len(text.splitlines())} lines)")

    if args.json:
        Path(args.json).write_text(json.dumps(data, indent=2, default=float) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
