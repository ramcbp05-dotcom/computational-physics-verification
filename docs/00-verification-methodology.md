# Verification methodology

A simulation that runs is not a result. This repository is organised around a
single question asked of every number it produces: *what independent fact says
this is right?*

There are only four kinds of answer, and they carry very different weight.

## 1. Exact algebraic identities: verified at round-off

Some statements hold exactly in exact arithmetic, so the only error is
floating-point noise. These are the strongest checks available, and they are
cheap:

| identity | module | measured |
|---|---|---|
| `[S^a, S^b] = i eps_abc S^c` | quantum_spin | ≤ 1.6e-15 |
| `S^2 = s(s+1)` | quantum_spin | ≤ 1.8e-15 |
| `U†U = 1` for every optical component | photonics | ≤ 2.2e-16 |
| `tr H = 0` for a zero-field Heisenberg chain | quantum_spin | exactly 0 |
| `[H, S^2] = 0` for isotropic exchange | quantum_spin | exactly 0 |
| `|v|` conserved by the Boris push at `E = 0` | trajectories | 7.0e-15 |
| HOM coincidence at 50:50 | photonics | 4.9e-32 |

The right tolerance for these is `~1e-12`, a few hundred times machine epsilon.
Anything larger is a bug, not a tolerance question. Accepting `1e-6` for an
exact identity throws away almost all of the check's power.

Two of these deserve a note.

**Why the commutators are exact rather than approximate.** The spin matrices
are built from the ladder operators, with `Sx = (S+ + S-)/2` and `Sy = (S+ -
S-)/2i`. Writing `Sx` and `Sy` out explicitly instead would make the SU(2)
algebra a coincidence of the entries; building them this way makes it
structural, and the measurement then confirms only that the matrix products
were computed correctly.

**Why the HOM zero is worth more than a small number.** The two-photon
coincidence amplitude is the permanent of the splitter matrix, `U00 U11 + U10
U01`. At 50:50 with the correct `pi/2` reflection phase this is `(1·1 +
i·i)/2`, which is zero *algebraically*. A coincidence probability of `1e-4`
would not be "close enough": it would mean the phase convention is wrong. Exact
zeros are the sharpest tests in a numerical suite.

## 2. Closed forms: verified at the accuracy of the method

Where an analytic solution exists it is compared term by term, not
qualitatively:

| closed form | measured deviation |
|---|---|
| two spins: `-3J/4` (singlet), `J/4 - hm` (triplet) | ≤ 1.1e-16 |
| Rabi precession `<Sx> = cos(ht)/2` | 3.9e-16 |
| Mach-Zehnder output `(sin²(φ/2), cos²(φ/2))` | 4.4e-16 |
| two-photon coincidence `(T − R)^2` | ≤ 1.1e-16 |
| fringe visibility `2√(I₁I₂)/(I₁+I₂)` | ≤ 4.0e-16 |
| zero-coupling magnetisation `tanh(βh)` | ≤ 1.4e-16 |
| gyroradius `m v⊥ /(|q|B)` | ≤ 2.4e-16 |
| Kepler's third law `T = 2π√(a³/μ)` | ≤ 7.1e-16 |

Several of these come out at round-off because the "numerical" route is itself
exact: unitary time evolution goes through the eigenbasis, not a time-stepping
scheme, so there is no step-size error to report.

## 3. Two independent methods: verified against each other

The strongest form of evidence when no closed form covers the general case:
compute the same quantity twice by routes that share no code.

The partition function of the Ising ring is computed by enumerating all `2^N`
configurations and, separately, from `Z = λ₊^N + λ₋^N`. Over **384 parameter
combinations** (`N` from 2 to 12, four fields including a negative one, three
couplings including an antiferromagnetic one, four temperatures spanning two
decades), the **worst relative disagreement in `ln Z` is 2.4e-13**.

This is also where a convention error surfaces. The first version of this check
failed at `N = 2` by up to 90 %: the brute-force enumeration treated the 2-site
ring as having one bond while `tr T²` implies two. Neither convention is wrong,
but they are different models, and only an independent second method makes the
difference visible. The library now documents the `N`-bond ring convention and
`quantum_spin` **raises** rather than choosing silently.

## 4. Convergence order: verified by measuring the exponent

For anything discretised, the error must shrink at the rate the method
promises. This is the check a plot cannot substitute for: an integrator can
produce a visually perfect closed orbit and still be first order.

`verification.convergence_order` fits `error ∝ h^p` by least squares in log-log
space, dropping points already at round-off.

| integrator | expected | fitted | R² |
|---|---|---|---|
| leapfrog (velocity Verlet) | 2 | 2.000 | 1.00000 |
| RK4 | 4 | 4.118 | 0.99979 |
| Boris push | 2 | 2.000 | 1.00000 |

## What this caught

Measuring rather than asserting found four real errors during development, all
of them in work that looked correct:

1. **A first-order Boris push.** The velocity update was textbook, and the
   position was advanced as `x += dt * v_new`: an asymmetric arrangement.
   Convergence order was still 2, so an order study alone would have passed it.
   The conserved quantities gave it away: energy drift `1.5e-04` and a
   gyro-orbit radius varying by `3.1e-03` of `r_c`. Moving to drift-kick-drift
   gave `5.1e-14` and `3.1e-14`: ten orders of magnitude, from reordering three
   lines.

2. **A wrong symmetry claim.** A test asserted that a uniform field along `z`
   breaks `[H, S^2] = 0`. It does not: `S^2` commutes with every component of
   the total spin, so the Zeeman term commutes with it too. The field lifts the
   degeneracy *within* a multiplet without removing `S^2` as a good quantum
   number. What does break it is anisotropic exchange or a site-dependent
   field, and the suite now checks both directions.

3. **The `N = 2` ring convention**, above.

4. **An unreachable tolerance.** An early version of the variance check in a
   sister repository demanded `1e-10` relative accuracy at a condition number
   of `1e12`, which no float64 algorithm can deliver. Tolerances here are set
   from measurement: the error report records what each quantity actually
   achieves, and the test bounds sit a safe margin above it.

## Reproducing

```bash
python -m pytest tests/ -q          # 723 tests
python scripts/error_report.py      # regenerate results/numerical-error-report.md
python scripts/make_figures.py      # regenerate figures/
```

CI runs all three. The error report is committed, so a change that weakens a
bound appears as a diff in a pull request rather than as a quietly looser
guarantee.
