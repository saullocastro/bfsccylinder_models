# DOE09: normalization and completeness of the multi-mode Koiter analysis

Method of the multi-mode post-buckling coefficients of
`koiter_cylinder_CTS_sanders.fkoiter_cylinder_CTS_circum` for the DOE09
designs A, B and C (cases 0, 1 and 6), and the setup chosen for the 20,000
runs of the DOE. No DOE run has been submitted; the mesh is open, see
[Decision](#9-decision).

## Mesh convergence: restarted

All previous convergence studies (`_k5_nodal`, `_k5c`, `_k5g`, the
reassessment studies a to m, and the SS4 comparison) were removed from the
branch, with their results, tables, generators and post scripts, and the
sections of this report that presented them: the Reassessment, Current
results, All convergence studies, the Summary, and Sections 1, 2 and 7.
They were run on the SS3 edges anchored at one node, most of them with v
and w fixed at the edge nodes only; git history keeps them (up to commit
`1fc65e2`).

The study is restarted from a low ny on the inertia relief edges of
`bfsccylinder_models/edges.py` (commit `1fc65e2`), with the axial load on
both edges and no node anchored:

- **SS3-IR**: v = w = 0 along both edges, the axial translation removed by
  inertia relief (mass-weighted mean axial displacement zero). It gives the
  SS3 results to round off, 1e-12 in Pcr and 5e-9 in b_1111 on the Waters
  shell, the axial translation being a null vector of every operator.
- **free-IR**: no edge condition, all six rigid body modes removed by
  inertia relief. The lowest buckling modes are the n = 2 to 6 ovalizations
  of the free edges, at a small fraction of the SS3-IR load (case 6 at
  ny = 40: 1039 N against 6989 N).

[`scripts/generate_qsubs_convergence.py`](scripts/generate_qsubs_convergence.py):
cases 0, 1 and 6, NL, eps1 = 0.0005, both edge conditions, ny = 24, 32, 40,
48, 64, 80, 96, 120 and 160, and axial factors F = 0.5, 1 and 2 (largest
axial element length dy/F: elements twice as long axially, square, half as
long), 162 runs. Tables:
[`checks/convergence_post.py`](checks/convergence_post.py), results in
`results/DOE09_conv_ir.jsonl.gz` once the runs are done.

The sections below describe the method (normalization, mode set, crest);
the ny quoted in them locate their evidence, not a mesh recommendation.

## Update: element crest and cut degenerate clusters

Two corrections since the first version of this report. Sections 5 and 6
below are kept as first written, with notes where these corrections apply.

**1. The crest is now computed with the kinematics of the element.**
`ElementField` in `run_case.py` evaluates w anywhere in an element as
Sw(xi, eta) @ q, with `update_Sw` of `BFSCCylinderSanders`, the element the
model is assembled with. The crest is the largest |w| on a 9 x 9 grid of every
element, refined by L-BFGS-B in (xi, eta) inside the 20 elements of largest
grid values; the RMS integrates w**2 with 4 x 4 Gauss points per element,
exact for the bicubic w. Checks, [`checks/crest_methods.py`](checks/crest_methods.py)
and [`checks/element_crest_reference.py`](checks/element_crest_reference.py):

- the bicubic Hermite reconstruction used before equals the element Sw to
  3e-16 at random interior points, so it was the right field, sampled at the
  wrong points;
- the crest falls inside the elements, e.g. at (xi, eta) = (0.59, -0.24);
- the refined crest agrees with a 41 x 41 grid refined in 200 elements to
  1e-15 for every mode (cases 0 LIN and 6 NL, ny = 80); the refinement needed
  the objective scaled to order one, the modes being in metres;
- case 0, LIN, ny = 80, crest over the largest nodal translation:

  | mode | edge envelope (old crest_w) | 11 x 11 grid (old crest_e) | element, refined |
  |---|---|---|---|
  | 0 | 1.04858 | 1.05277 | 1.05343 |
  | 1 | 1.03412 | 1.04206 | 1.04240 |
  | 8 | 1.02918 | 1.03800 | 1.03808 |

  so the old crest_w were 0.5-0.8 % low (1-1.6 % on the crest-normalized
  b_iiii) and the old crest_e up to 0.4 % low (up to 0.8 % on b_min_t);
- rms_w from the element is 1.8 % below the old edge-based value, which
  sampled w on the node columns only.

**2. The crest depends on the rotation of the field against the mesh.** The
minimum direction of b is defined only up to a rotation of the cylinder, and
a shift by a fraction of an element is not a symmetry of the mesh, so the
finite element crest of the combined mode varies along the family:
10.533 to 10.585 (0.5 %) for case 0, LIN, ny = 80. `pair_rotation` extends
the exact shift by one element, a rotation by n dtheta in the plane of each
distinct mode and its partner, to any angle; it reproduces the exact shift
of w to 8e-9. crest_w of every mode and crest_e are now the largest over 8
rotations within one element (`crest_e_min` the smallest, as a measure of the
discretization); e.g. crest_w of mode 0 becomes 1.0572.

**3. The 5th distinct mode cuts a degenerate cluster in 10 of the 24 runs,
so the invariance of Section 6 does not hold in general.** Repeating the
check of Section 6 after these changes, the reference slice gave
b_min_energy = -0.4678 instead of -0.5975, the other slice -0.5975 again;
the minimization is not at fault (40, 400 and 2000 starts agree). The n = 21
eigenspace of case 1, NL, ny = 80, is four-dimensional too (multipliers
7.4e-3 above the critical one, equal to round off), and 5 distinct modes take
n = 22 (2), n = 23 (2) and **one** of the two n = 21 modes, with its partner:
half of that eigenspace, the other half left out, chosen by round off. The
agreement to 7e-6 of Section 6 was the two runs happening to return the same
half. Over the 24 runs of the removed Section 7 (git history), the 5th distinct mode and the next one
share their multiplier to better than 1e-5 in:

| run | gap to the next distinct mode |
|---|---|
| 0 NL ny=120 | 5.3e-6 |
| 0 NL ny=200 | 2.9e-6 |
| 1 NL ny=80, 120, 160, 200 | 3.7e-13, 6.0e-15, 1.1e-15, 8.9e-15 |
| 6 LIN ny=80 | 3.1e-10 |
| 6 NL ny=120, 160, 200 | 2.5e-10, 1.9e-12, 1.3e-10 |

and to 3-4e-5 in 0 NL ny=160, 6 LIN ny=160 and 200. **The b_min and b_min_t
of the removed Section 7 for these runs depend on round off.** The fix is a Koiter set
made of complete clusters: at least 5 distinct modes, extended to the end of
the cluster of the 5th, with their partners, 12 modes when the clusters are
four-fold as in the NL cases. That makes the number of Koiter modes vary from
run to run, and `fkoiter_cylinder_CTS_circum` takes koiter_num_modes before
its eigenvalue analysis; see [Decision](#9-decision), option 4.


## 3. What the modes are

[`checks/mode_pairs.py`](checks/mode_pairs.py), case 0, LIN, ny = 80. The
modes with the same wave number n, which the eigen solver returns in pairs of
nearly equal multiplier (1e-4 apart), are **not** rotations of each other:

- their axial profiles are orthogonal, overlaps 0.0048, 0.0027, 0.0081,
  0.0012, 0.016 and 0.0006 for the six pairs;
- one is symmetric and the other antisymmetric about mid-length, the number of
  sign changes of the profile differing by one;
- their sum is localized at one end of the cylinder, their difference at the
  other: end-localized buckles, coupled weakly through the length.

The true rotated partner of each mode is not among the returned eigenvectors
(residual 1.0 in their span), as expected from a Krylov solver with one
starting vector; `degenerate_partner` rebuilds it. So num_distinct = 12 is
right and `run_case.use_distinct_modes` works as designed.

## 4. Normalization: the literature

- **Single-mode.** Rahman, Jansen and Wijker (2007), and Rahman (2009),
  Ch. 2: "the buckling modes are scaled such that the maximum radial
  displacement is equal to the shell thickness", the convention of ANILISA,
  whose mode amplitude is in thicknesses. Whether the maximum is taken at the
  nodes or over the field is not stated. Castro and Jansen (2021): "customarily
  re-scaled dividing by the maximum normal displacement amplitude and
  multiplying by the plate or shell thickness". This is what the models do,
  at the nodes.
- **Multi-mode.** Rahman (2009), Ch. 3, the basis of Rahman, Jansen and
  Gürdal (2009, AIAA 2009-2557):
  - the modes are scaled by an energy norm, Eq. (3.21),
    lambda_I q_I^T [dK_D + dK_G] q_I = 1, that is lambda_I Delta_I = 1
    (Eq. 3.15), the convention of Byskov and Hutchinson;
  - A_ijkl is averaged over the permutations of its indices, which with that
    normalization makes b_Ijkl fully symmetric (Eq. 3.14, 3.16);
  - the cluster is summarized by the minimum direction of Salerno,
    b_ijkI e_i e_j e_k = b e_I with e.e = 1 (Eq. 3.31), whose lowest
    eigenvalue is the post-buckling coefficient of the cluster;
  - the thickness enters only when results are reported: modes rescaled to a
    maximum out-of-plane displacement equal to the thickness for comparison
    with ANILISA, and imperfection amplitudes set by "the maximum out-of-plane
    displacement of the imperfection shape", the combined one.

Section sec:normalisation of `doc/nlprebuck_implementation.tex` already
recommends the crest normalization over the nodal one, for the single-mode b.

## 5. Normalizations implemented

All in `run_case.py` and [`scripts/koiter_post.py`](scripts/koiter_post.py),
without changing the library. Every term of b_ijkl in `b_coefficients`
carries the four mode amplitudes and its denominator d_i = phi20_i . u_i the
square of that of mode i, so for modes rescaled by s_i, exactly,

    b'_ijkl = b_ijkl s_j s_k s_l / s_i,    a'_ijk = a_ijk s_j s_k / s_i

Checked against a copy of the library multiplying the 5 modes by 1.3, 0.7,
1.9, 0.55 and 1.15 (case 6, NL, ny = 40): 1.6e-7 relative difference on
b_ijkl (a_ijk, 5e-9, is zero by symmetry). The normalization can therefore
be chosen after the runs; each run stores the scales:

| name | scale s_i | stored |
|---|---|---|
| nodal | 1, the models: largest nodal translation = h | `b_ijkl`, `a_ijk` |
| crest | 1/crest_w: crest of w, between nodes too, = h | `crest_w` |
| rms | 1/rms_w: RMS of w over the surface = h | `rms_w` |
| energy | 1/sqrt(lambda_i abs(d_i)), Rahman Eq. (3.21) | `lambda_d` |

- `crest_w` follows the envelope of the mode and its rotated partner along the
  axis with the cubic Hermite interpolation in w and w_x, as
  `reference_b_convergence.py` does. Case 0, LIN, ny = 80: crest_w = 1.049,
  1.049, 1.046, 1.046, 1.029, moving b_iiii from -0.0789 to -0.0717.
- `lambda_d` is captured by wrapping `b_coefficients` of the model.
- `b_min_energy`: minimum of the symmetrized energy-normalized b_ijkl over
  e.e = 1 (Salerno), `e_min` its direction.
- `b_min_t = b_min_energy / crest_e**2`, crest_e being the crest of w of the
  combined mode sum_i e_i s_i u_i in thicknesses, found with the bicubic
  Hermite interpolation of the element (w, w_x, w_y, w_xy), since a
  combination of wave numbers is not a single harmonic. On single modes the
  bicubic crest agrees with the envelope one exactly when the crest is on a
  node, and to 0.4 % at ny = 80 in case 0, 3.3 elements per wavelength.

## 6. The mode set: four-fold eigenspaces

### Evidence

[`checks/fourfold_degeneracy.py`](checks/fourfold_degeneracy.py), case 1,
NL, ny = 80:

- multipliers over the critical one minus one: 0, 1.7e-14, 3.1e-4, 3.1e-4,
  7.4e-3, ...;
- the two n = 22 modes and their rotated partners span four dimensions,
  singular values 1.018, 1.018, 0.982, 0.982; the n = 23 modes likewise,
  1.177, 1.177, 0.784, 0.784;
- the returned n = 23 modes are neither symmetric nor antisymmetric: mode 2
  has its amplitude at the left end (0.041 against 0.012), mode 3 at the right
  one (0.016 against 0.039).

So the symmetric/antisymmetric splitting is below round off, each wave number
has a four-dimensional eigenspace (2 axial shapes x 2 rotations), and the
eigen solver returns an arbitrary two-dimensional slice of it.
`canonical_modes` then treats the slice as a rotation pair.

### Consequences, 5-mode expansion

Case 1, NL, ny = 80, with the 5-mode `run_case.py` of that stage:

- [`checks/pair_mixing.py`](checks/pair_mixing.py), modes 0 and 1 rotated by
  30 degrees in their plane after the ordering of `use_distinct_modes`: the
  5-mode b_min (energy normalization, computed afterwards from the stored
  b_ijkl) went from -0.475 to -0.449, and b_2222 from -0.295 to -0.255,
  although modes 2 and 3 were not touched;
- [`checks/same_process_resolve.py`](checks/same_process_resolve.py), the
  same model solved twice in one process, the second time with the same
  mixing of modes 0 and 1: the pre-buckling state agreed to 3e-14, but mode 2
  differed (relative difference 1.27), b_2222 = -0.286 against -0.255. The
  mixing of modes 0 and 1 cannot change mode 2, so the slice of the n = 23
  eigenspace returned by the eigen solver changed between the two solves.

### Fix

`use_distinct_modes` with `koiter_rotation_closed = True`:

1. the 5 distinct modes, each followed by its rotated partner, 10 Koiter
   modes; an axisymmetric mode, which has no partner, takes one column;
2. every column made K_C-orthonormal to the previous ones (K_C captured from
   the eigen solve), the partner being built from the mode so made. The
   Koiter section of the models assumes the modes orthogonal in the metric of
   the load term, using d_i only and never d_ij; eigenvectors are, but a
   partner from `degenerate_partner` is orthogonal to its own mode only.

[`checks/eigenspace_slice.py`](checks/eigenspace_slice.py) feeds the model a
different slice of the same eigenspaces, raw eigenvector 1 replaced by
cos 30 m1 + sin 30 P(m0), and eigenvector 3 likewise. Case 1, NL, ny = 80:

| | original slice | other slice |
|---|---|---|
| closed, not orthonormalized: b_min_energy | -0.5733 | -0.7540 |
| closed and K_C-orthonormal: b_min_energy | **-0.597485** | **-0.597481** |
| closed and K_C-orthonormal: b_min_t | -0.2218 | -0.2192 |
| closed and K_C-orthonormal: nodal b_0000 | -0.2608 | -0.2710 |

b_min is invariant to 7e-6. b_min_t differs by 1.2 %: the minimum direction
is defined only up to a rotation of the cylinder, and the crest of the
rotated field falls differently between the nodes.

**Note (Update, point 3):** this agreement was fortuitous. The 5th distinct
mode cuts the four-fold n = 21 eigenspace, and a later repetition of the same
check gave -0.4678 against -0.5975.

## 8. Cost

The 10-mode Koiter section takes 30 ms per element more than the 5-mode one,
the median of the 24 pairs of runs (23 to 38 ms, one-core runs on shared
nodes), 47 ms per element in all, now in `koiter_time_per_element` of
`generate_qsubs.py`. Peak memory 13.5 GB for the largest run, case 0, NL,
ny = 200, 57 min.

| ny | core-hours | jobs |
|---|---|---|
| 160 | 19,100 | 402 |
| 200 | 28,900 | 1,058 |

(15,800 core-hours at ny = 160 with the 5-mode setup.)

## 9. Decision

Open, until the convergence study above is done: the mesh of the DOE (ny
and the axial factor F, possibly elongated elements), the quantity of the
DOE (b_min_t of the critical cluster or of a window of wave numbers about
it), and the edge condition, SS3-IR or free-IR. The options of the previous
version of this section, which rested on the removed studies, are in git
history.

## 10. Issues in the library

Reported, not changed; both are worked around in `run_case.py`:

1. `canonical_modes` (`cyclic_symmetry.py`) treats every group of two equal
   multipliers as a rotation pair and rotates it to put a crest on the y = 0
   generator. In a four-fold eigenspace the two returned vectors are not a
   rotation pair, and the result depends on round off.
2. The Koiter section (`koiter_cylinder_CTS_sanders.py`, the null space and
   `b_coefficients`) assumes the Koiter modes orthogonal in the metric of the
   load term, d_ij = 0 for i != j. Eigenvectors of distinct multipliers are;
   modes combined with partners built by `degenerate_partner` inside a
   four-fold eigenspace are not, unless made so as in Section 6.
3. The known non-symmetry of the multi-mode b_ijkl (`doc/nlprebuck_implementation.tex`,
   "Known limitations") is untouched; b_min uses the part of b_ijkl
   symmetric in all four indices, as Rahman (2009) does, which is also the
   only part the quartic form sees.

## 11. Changes to the DOE09 driver

- `run_case.py`
  - `koiter_num_distinct = 5`, `koiter_rotation_closed = True`,
    `koiter_num_modes = 10`, `use_distinct_modes` as in Section 6;
  - `use_koiter_denominators` (lambda_d), and b_min_energy, e_min, crest_e,
    b_min_t, koiter_distinct in the RESULT line;
  - since the Update: `ElementField` (crest and RMS of w from the element
    kinematics, replacing `mode_amplitudes` and `field_crest`),
    `pair_rotation` and `orbit_crest` (largest crest over the rotations within
    one element), crest_e_min, rotation_error, and crest_method =
    'element_orbit', which `generate_qsubs.py` and `post.py` require of an
    output; since `1fc65e2`, `--edges SS3|SS4|SS3-IR|free-IR`.
- `koiter_post.py`: `rescaled`, `energy_scales`, `symmetrized`,
  `min_direction`.
- `post.py`: `DOE09_koiter.npz` with the nodal b_ijkl and a_ijk of the 10
  modes, the scales of the other normalizations, b_min_energy, b_min_t,
  crest_e and e_min; b_min_t and b_min_energy in `DOE09_output.txt`.
- `generate_qsubs.py`: requires the closed 10-mode setup, reruns outputs of
  earlier setups, `koiter_time_per_element = 0.047`, `python -u` so that a
  crashed run keeps its log.
- `generate_qsubs_convergence.py`: the restarted convergence study, see
  the top of this report.

The scripts in [`checks/`](checks) were run from the DOE09 directory, next to
`DOE09.txt`, with the `run_case.py` of their stage: `crest_methods.py` and
`element_crest_reference.py` with the one of the Update, `mode_pairs.py`,
`pair_mixing.py` and `same_process_resolve.py` with the 5-mode one, `fourfold_degeneracy.py` and
`eigenspace_slice.py` with the 10-mode one of [`scripts/`](scripts).
`cluster_subsets.py` and `convergence_post.py` run from this directory on
the stored results in `results/`.

## References

- Castro, S.G.P. and Jansen, E.L. (2021). Displacement-based formulation of
  Koiter's method: application to multi-modal post-buckling finite element
  analysis of plates. Thin-Walled Structures 159, 107217.
- Rahman, T. (2009). A perturbation approach for geometrically nonlinear
  structural analysis using a general purpose finite element code. PhD
  thesis, Delft University of Technology.
  https://resolver.tudelft.nl/uuid:80e11dbd-90be-44f1-bb36-049503a265bd
- Rahman, T., Jansen, E.L. and Gürdal, Z. (2009). Finite element based
  multi-mode initial post-buckling analysis of composite cylindrical shells.
  AIAA 2009-2557.
- Rahman, T., Jansen, E.L. and Wijker, J.J. (2007). Finite element based
  initial post-buckling analysis of conical shell structures. 1st CEAS
  European Air and Space Conference, CEAS-2007-164, 1809-1816.
