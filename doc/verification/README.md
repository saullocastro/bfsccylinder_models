# Verification scripts

Every number quoted in `doc/nlprebuck_implementation.tex` that is not a
literature value comes from one of the scripts in this directory or from the
test suite in `tests/`. They are listed, with the section each one feeds, in
the appendix of that document. The exceptions are the figures measured on
code that no longer exists, before `bfsccylinder` 0.6.0 or before a change to
the models, which the document marks as such where it quotes them.

These are *studies*, not unit tests: several of them take tens of minutes and
sweep meshes. The fast, assertive checks belong in `tests/`.

## Running

Each script puts the repository root first on `sys.path`, so they can be run
from anywhere:

```
python doc/verification/kinematics_vs_element.py
```

That ordering matters. An installed `bfsccylinder_models` shadows the working
tree whenever a script is run by absolute path, because Python puts the
*script's* directory on `sys.path`, not the current one; the bootstrap at the
top of each script defeats that. If you intend to verify an installed build
instead of the working tree, remove the bootstrap.

Requires `bfsccylinder >= 0.6.0` — earlier versions do not have a tangent
stiffness matrix consistent with the internal force vector, which several of
these checks measure.

## The scripts

| script | what it establishes | runtime |
| --- | --- | --- |
| `kinematics_vs_element.py` | the strain and its first derivative assumed by the Koiter expansion match the element to machine precision, for both kinematics; a control with `-Sv/R` removed fails, so the check is sharp; `w` between two axial stations is the Hermite interpolation used to find the crest of a mode | seconds |
| `tangent_consistency.py` | `KC0 + KCNL + KG` is the derivative of `fint`, and equals the Koiter second variation; identifies the stress resultants the element's `KG` actually uses; the directional Taylor test and the fit of the coefficients of `KCNL` and `KG` | seconds |
| `cts_vs_constant_stiffness.py` | the CTS models reproduce `koiter_cylinder.py` and `koiter_cylinder_sanders.py` in the constant-stiffness limit, `NLprebuck` on and off | ~10 min |
| `cts_mesh_convergence.py` | mesh convergence of `Pcr` and `b_1111` for a steered design; pass `nl` for the non-linear pre-buckling variant | ~20 min / ~35 min |
| `cts_mode_spectrum.py` | circumferential harmonic content of the lowest modes of the steered design against mesh, and of that design and two constant-angle ones on the `ny=60` mesh; pass `nl` for non-linear pre-buckling | ~20 min |
| `reference_mode_cluster.py` | size of the near-critical cluster of the Sun and Arbocz shells, which is what sizes a multi-mode expansion | ~5 min |
| `second_order_rhs.py` | the right-hand side of the second-order fields that makes a multi-mode expansion consistent: the residual along a post-buckling ray of a two-mode polynomial energy is O(s^3) with the `T^-1` form and O(s^2) with the old `1/m` factor; pass `orthogonal` for T-orthogonal modes | seconds |
| `reference_b_convergence.py` | single-mode `b_1111` of AW-CYL-1-1 against `ny`, `nx`, the expansion point, the member of the degenerate pair and, with `sanders`, the kinematics; the crest-normalized `b`, and the lowest multiplier of every circumferential harmonic; with `linear`, about the linear pre-buckling state, `60 17 sanders linear` being the Waters shell | 1–25 min per configuration |
| `reference_load_stepping.py` | how the load stepping reaches the expansion point on the reference meshes: the state at the reference load, `s` and the permissible `eta` per load step, the Newton-Raphson residuals, what the eigen solver returns, `a_111`, the conditioning of the reduced tangent with and without scaling, and the residual of `phi2` along the mode with consistent and inconsistent operators | ~6 min |
| `reference_b_reproducibility.py` | `b_1111` with one and with the default number of BLAS threads, and with and without the degenerate partner in the column border of the bordered system | ~7 min |
| `quadrature_convergence.py` | zero energy modes of one element, and `lambda_c` and `b_1111` of the reference shells, against the number of Gauss points | ~10 min |
| `element_loop_timing.py` | time per element of the Koiter element loop, version 0.3.2 against `koiter_tensors.py`, for m = 1, 2, 5, 8 | ~1 min |
| `benchmark_koiter_vectorization.py` | the same DOE09 and Waters cases run with two checkouts (default: this one and `../bfsccylinder_models_baseline` at `3251981`) in subprocesses; results, Koiter time (direct and `t(m) - t(0)`, split into element loop and bordered solves) and peak memory. `run`, `report`, `fields` (second-order fields `uij` of both versions). `benchmark_koiter_vectorization_hpc.sh` is its production-size job for a PBS cluster | minutes to hours |
| `bordered_roundoff_floor.py` | how far `uij` and `b_ijkl` move when the right-hand side of the bordered solves is perturbed at round-off level, the floor for any comparison of two versions | ~2 min |
| `multimode_b_symmetry.py` | where the index non-symmetry of the multi-mode `b_ijkl` comes from, see below; changes nothing | ~2 min |
| `estimate_doe09_core_hours.py` | the core-hour estimate of the DOE09 `generate_qsubs.py`, reproduced without executing it (it submits jobs) | seconds |

## Literature verification cases

These live in `tests/` because they assert rather than report:

| test | shell | reference |
| --- | --- | --- |
| `test_koiter_cylinder_newton_raphson.py` | Sun et al. §3.1 and NASA AW-CYL-1-1 | Sun et al. 2020; Arbocz, Starnes & Nemeth 2001 (ANILISA, STAGS-A) |
| `test_koiter_cylinder_Waters.py`, `_sanders.py` | Waters shell | Arbocz & Starnes 2002 |
| `test_koiter_cylinder_CTS.py`, `_sanders.py` | CTS cylinder, constant-stiffness limit | cross-check against `koiter_cylinder.py`, `_sanders.py` |
| `test_buckling_mode_cluster.py` | Sun et al. §3.1 | degeneracy and cluster structure |
| `test_second_order_conditions.py` | Sun et al. §3.1, coarse | the orthogonality conditions of the bordered system hold along every direction of the null space, against a central difference of the compiled tangent; the rebuilt degenerate partner is in the column border |
| `test_linBuck_VAFW.py`, `test_Zhihua_error.py` | VAFW cylinders | linear buckling |
| `test_cts_shares_nlprebuck_algorithm.py` | — | source parity between the CTS and the constant-stiffness models |
| `test_koiter_tensors.py` | random elements | the vectorized element loop against the loop of version 0.3.2, both kinematics, with and without `NLprebuck` |

## Vectorized Koiter tensors: results

The element loop of the Koiter tensors, repeated in the four models up to
version 0.3.2 with Python loops over every pair and quadruple of modes, is
now `bfsccylinder_models/koiter_tensors.py`, shared by the four models, and
the second-order fields are solved for `i <= j` only (`uij = uji`). Section
"Cost of the element loop" of `doc/nlprebuck_implementation.tex` describes
it. Every figure below is on one core, one BLAS thread.

### Element loop, per element

`element_loop_timing.py`, workstation (Ryzen 7 PRO 250), idle. Four Sanders
elements with random data, `NLprebuck` on, m Koiter modes and m + 2
directions of the null space, as in the DOE runs.

| m | old loop (ms/element) | new loop (ms/element) | speedup |
|---|---|---|---|
| 1 | 14.4 | 0.43 | 34 |
| 2 | 52 | 0.49 | 106 |
| 5 | 1109 | 0.66 | 1686 |
| 8 | 5836 | 1.06 | 5489 |

The old loop grew as about m^4; the new one is nearly independent of m. The
old loop ran at 0.7 s per element for m = 5 in an earlier measurement on the
same workstation and at 2.0 s on a core of the cluster.

### Whole model, both versions

`benchmark_koiter_vectorization.py fields`, workstation, with 8 other
single-threaded runs sharing it, so the absolute times are inflated for both
versions alike. DOE09 case 6 with `run_case.py`'s solvers (PARDISO + GMRES)
and distinct modes, and the Waters shell with SuperLU; 960 elements each.
Koiter time is from the printed buckling load to the return. Results are in
`local_results/`.

| case | pre-buckling | ny | m | total base / new (s) | Koiter base / new (s) | Koiter speedup | peak mem base / new (GB) |
|---|---|---|---|---|---|---|---|
| DOE09 6 | LIN | 40 | 2 | 156 / 19 | 146.7 / 8.8 | 17 | 0.31 / 0.31 |
| DOE09 6 | LIN | 40 | 5 | 1932 / 141 | 1919.1 / 127.7 | 15 | 0.31 / 0.33 |
| DOE09 6 | NL | 40 | 2 | 320 / 181 | 161.6 / 10.5 | 15 | 0.35 / 0.35 |
| DOE09 6 | NL | 40 | 5 | 1965 / 234 | 1786.4 / 47.0 | 38 | 0.35 / 0.37 |
| Waters | LIN | 60 | 2 | 169 / 26 | 155.5 / 11.5 | 14 | 0.37 / 0.37 |
| Waters | NL | 60 | 2 | 295 / 152 | 162.2 / 11.7 | 14 | 0.42 / 0.42 |

Of the new Koiter time, the element loop is about 1 s at this size; the rest
is the bordered solves, m(m+1)/2 of them, whose cost does not depend on the
refactor.

Maximum difference, new against baseline, relative to the largest |value| of
each quantity:

| case | pre-buckling | ny | m | Pcr, load_mult | a_ijk | b_ijkl | uij against baseline uij | baseline uij against uji | null-space vectors |
|---|---|---|---|---|---|---|---|---|---|
| DOE09 6 | LIN | 40 | 2 | 0 | 8e-16 abs. (|a| < 1.3e-13) | 1.0e-13 | 2.1e-13 | 1.5e-13 | 4 = 4 |
| DOE09 6 | LIN | 40 | 5 | 0 | 1.1e-14 (max |a| 1.5) | 5.9e-14 | 7.0e-13 | 5.3e-13 | 7 = 7 |
| DOE09 6 | NL | 40 | 2 | 0 | 2e-16 abs. (|a| < 1.2e-16) | 7.5e-14 | 6.2e-14 | 3.7e-14 | 3 = 3 |
| DOE09 6 | NL | 40 | 5 | 0 | 6e-16 abs. (|a| < 1.6e-12) | 6.4e-14 | 9.0e-14 | 4.9e-14 | 6 = 6 |
| Waters | LIN | 60 | 2 | 0 | 4e-16 abs. (|a| < 6.7e-12) | 8.8e-11 | 1.5e-7 | 2.0e-8 | 2 = 2 |
| Waters | NL | 60 | 2 | 0 | 5e-16 abs. (|a| < 1.3e-11) | 1.3e-11 | 1.5e-7 | 1.7e-8 | 4 = 4 |

- Pcr and the multipliers are identical to the last bit, the eigenvalue
  analysis being unchanged.
- a_ijk vanish on these cylinders except for the axisymmetric mode of DOE09
  case 6 LIN m = 5. Where they are round-off, the difference is given in
  absolute terms.
- On the Waters shell the bordered system amplifies round-off by about 1e8.
  `bordered_roundoff_floor.py` perturbs only its right-hand side by 4e-16
  relative; this moves `uij` by 2.6e-7 (LIN) and 2.2e-7 (NL), and b_ijkl by
  9.9e-11 and 1.9e-11. Those are the differences in the table, so they are
  the floor of that solve, not an effect of the refactor.

### Production size, cluster

From `cluster_results/REPORT.md`: DOE09 at ny=160, `koiter_num_modes=5`,
`num_eigvals=12`, one core of the `hpc12` cluster.

| case | pre-buckling | elements | Koiter (ms/element) | element loop (ms/element) | bordered solves (ms/element) | peak mem (GB) |
|---|---|---|---|---|---|---|
| 0 | NL | 25600 | 16.76 | 1.32 | 15.26 | 8.31 |
| 1 | NL | 20160 | 15.30 | 1.45 | 13.74 | 6.46 |
| 6 | NL | 12800 | 13.40 | 1.42 | 11.88 | 4.06 |
| 0 | LIN | 25600 | 15.83 | 1.48 | 14.22 | 7.24 |

- **Baseline estimate:** the baseline was not run to completion at this size.
  At 2.0 s per element for its element loop, plus 25 bordered solves of
  26 s, the Koiter section of case 0 NL would take about 51,900 s, against
  429 s now (about 120×). The whole run would take about 53,000 s, against
  1,724 s (about 30×).
- **Per-element growth:** the Koiter time per element grows 25% from 12,800
  to 25,600 elements. That growth comes from the bordered solves.

### Cost of the 20,000 DOE09 runs

`estimate_doe09_core_hours.py`: ny=160, 19,903 elements per run on average.

| koiter_time_per_element | pre-buckling + eigen (core-h) | Koiter (core-h) | total (core-h) |
|---|---|---|---|
| 0.7 s (version 0.3.2) | 13933 | 77402 | 91335 |
| 0.0168 s (now, case 0 NL, ny=160; set in `generate_qsubs.py`) | 13933 | 1858 | 15791 |

### Not measured

The full matrix of `benchmark_koiter_vectorization.py run`, DOE09 cases 0, 1
and 6 at ny=40 and 80 with m = 0 to 8 and three repeats, was stopped. There
were two reasons:
- **Baseline cost.** The baseline element loop takes 5 to 8 h per m=8 run at
  ny=80.
- **ARPACK failures.** `eigsh` stops with ARPACK error -8 for
  `num_eigvals=20` on most of those meshes, in both versions. For m=8 a
  working value has to be found per mesh; `EIGVALS_M8` in the script records
  the ones found.

The element-loop timing above covers the scaling with m exactly, since the
loop is linear in the number of elements.

### The multi-mode b_ijkl

`multimode_b_symmetry.py`, Waters shell, 3 modes. This analyses the known
non-symmetry; the coefficients are not changed.

- **j, k, l.** b_ijkl is symmetric under j <-> l to 1e-15, but not under
  j <-> k or k <-> l (23 to 53% of its largest entry). The formula sums two of
  the three pairings of mode i with j, k and l, each weighted 3,
  `3 phi3_ij.u_kl + 3 phi3_il.u_jk`, instead of all three weighted 2. After
  symmetrizing over j, k and l, b_ijkl equals the three-pairing form to 1e-15.
  The two forms coincide for i = j = k = l, and wherever b_ijkl is contracted
  with xi_j xi_k xi_l as in the amplitude equations. Individual entries
  differ: b_1122 is -0.125, against -0.102 symmetrized (NL). The DOE09
  post-processing reads only b_1111 and the b_iiii, which are unaffected.
- **i against the others.** b_ijkl differs from b_jikl by the per-equation
  factor -1/(6 lambda_i phi20_i.u_i). Multiplied back by it and symmetrized
  over j, k and l, the coefficients are symmetric in all four indices to
  1e-11, the size of the a_ijk terms.
- **The a-terms.** The terms in a_ijk have irregular index patterns:
  `a_iij a_ikl` appears twice, and there is a `phi30_il.u_i` term. They
  vanish on these cylinders, but have not been checked for an asymmetric
  bifurcation.
