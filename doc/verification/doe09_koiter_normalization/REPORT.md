# DOE09: normalization and completeness of the multi-mode Koiter analysis

Mesh convergence of the multi-mode post-buckling coefficients of
`koiter_cylinder_CTS_sanders.fkoiter_cylinder_CTS_circum` (version 0.4.0) for
the DOE09 designs A, B and C (cases 0, 1 and 6), and the setup chosen for the
20,000 runs of the DOE. No DOE run has been submitted; the decision on the
mesh is open, see [Decision](#9-decision).

The library is unchanged. Every change is in the DOE09 driver, copied in
[`scripts/`](scripts): `run_case.py`, `koiter_post.py` (new), `post.py`,
`post_convergence.py`, `generate_qsubs.py` and
`generate_qsubs_convergence.py`. The tables of the three convergence studies
are in [`tables/`](tables), the diagnostic scripts in [`checks/`](checks).

## Current results: complete clusters (study `_k5g`)

Latest setup, option 4 (b) of the [Decision](#9-decision):

- **library** (this branch): `koiter_num_modes` of the four models may be a
  callable of the multipliers and eigenvectors, called once after the last
  eigenvalue analysis and returning the number of Koiter modes, recorded in
  `out['koiter_num_modes']`. New test
  [`tests/test_koiter_num_modes_callable.py`](../../../tests/test_koiter_num_modes_callable.py):
  a callable gives the same b_ijkl as the number it returns, CTS and constant
  stiffness models, LIN and NL, and 0 or too many modes are handled. The whole
  test suite passes on the branch, 37 tests, PARDISO blocked;
- **driver**: at least 5 distinct modes, then every further distinct mode
  whose multiplier equals that of the last taken to 1e-5 (`koiter_cluster_rtol`,
  the tolerance of `canonical_modes`), each followed by its rotated partner,
  K_C-orthonormal, energy normalization, minimum direction of Salerno, and
  `b_min_t` with the element crest over the rotations within one element
  (see the Update below). num_eigvals = 16. `koiter_gap` in the RESULT line
  is the relative gap to the first distinct mode left out. The jobs run the
  library of this branch (`PYTHONPATH`), and `run_case.py` stops if the
  library does not take a callable.

**Invariance** ([`checks/eigenspace_slice.py`](checks/eigenspace_slice.py),
case 1, NL, ny = 80, the 12 Koiter modes n = 22, 23 and 21, four of each):

| | original slice | other slice |
|---|---|---|
| b_min_energy | -0.600351 | -0.600336 |
| b_min_t | -0.21678 | -0.21683 |
| crest_e (smallest over the rotations) | 1.66415 (1.61627) | 1.66396 (1.61481) |

against -0.4678 and -0.5975 for the same check with the n = 21 group cut.

**Convergence**, [`tables/DOE09_convergence_k5g.txt`](tables/DOE09_convergence_k5g.txt),
24 runs, no failure, no fallback to SuperLU, no incomplete pre-buckling
iteration. m is the number of Koiter modes, gap the relative gap in
multiplier to the first distinct mode left out.

| case | ny | b_min_t | b_min_energy | crest_e | m | gap |
|---|---|---|---|---|---|---|
| 0 LIN | 80 | -0.07182 | -8.054 | 10.590 | 10 | 3.0e-4 |
| | 120 | 0.12649 | 0.267 | 1.452 | 10 | 7.7e-4 |
| | 160 | 0.12337 | 0.289 | 1.530 | 10 | 8.3e-4 |
| | 200 | 0.12936 | 0.316 | 1.563 | 10 | 1.3e-3 |
| 0 NL | 80 | -0.12523 | -19.297 | 12.413 | 10 | 1.9e-4 |
| | 120 | -0.14419 | -23.145 | 12.670 | 12 | 5.1e-4 |
| | 160 | -0.15668 | -19.859 | 11.258 | **10** | **2.7e-5** |
| | 200 | -0.17366 | -26.232 | 12.290 | 12 | 1.4e-3 |
| 1 LIN | 80 | -3.60439 | -4.698 | 1.142 | 9 | 2.0e-3 |
| | 120 | 0.03722 | 0.008 | 0.473 | 10 | 3.8e-3 |
| | 160 | 0.03722 | 0.008 | 0.473 | 10 | 3.8e-3 |
| | 200 | 0.03722 | 0.008 | 0.473 | 10 | 3.8e-3 |
| 1 NL | 80 | -0.21629 | -0.600 | 1.666 | 12 | 1.5e-4 |
| | 120 | -0.17662 | -0.596 | 1.836 | 12 | 4.4e-3 |
| | 160 | -0.18859 | -0.631 | 1.830 | 12 | 5.7e-3 |
| | 200 | -0.19150 | -0.641 | 1.829 | 12 | 5.2e-3 |
| 6 LIN | 80 | -2.62314 | -120.150 | 6.768 | 12 | 7.1e-3 |
| | 120 | -0.03712 | -0.592 | 3.994 | 10 | 6.0e-5 |
| | 160 | -0.04851 | -0.803 | 4.067 | 10 | 3.0e-5 |
| | 200 | -0.05317 | -0.860 | 4.023 | 10 | 4.1e-5 |
| 6 NL | 80 | -0.42118 | -29.054 | 8.306 | 10 | 2.1e-4 |
| | 120 | -0.18102 | -7.570 | 6.467 | 12 | 1.7e-3 |
| | 160 | -0.22709 | -8.990 | 6.292 | 12 | 5.0e-3 |
| | 200 | -0.23225 | -9.203 | 6.295 | 12 | 3.9e-3 |

b_min_t from ny = 120, 160 and 200:

| case | 160 to 200 | Richardson | error at 160 | error at 200 |
|---|---|---|---|---|
| 0 LIN | 4.9 % | not monotone: 0.1265, 0.1234, 0.1294 | | |
| 0 NL | 10.8 % | not monotone (step ratio 1.36) | | |
| 1 LIN | 0 | constant, 0.03722 | 0 | 0 |
| 1 NL | 1.5 % | order 4.4, -0.1932 | 2.4 % | 0.9 % |
| 6 LIN | 9.6 % | order 2.5, -0.0595 | 18.5 % | 10.6 % |
| 6 NL | 2.3 % | order 7.3, -0.2335 | 2.7 % | 0.5 % |

Observations:

- ny = 80 is unusable, as before.
- **Cases 1 NL and 6 NL are converged at ny = 160 to about 2.5 %**, and to
  1 % at ny = 200; with the clusters cut (Section 7) they were not
  reproducible. Case 1 LIN, governed by an axisymmetric mode, is mesh
  independent.
- **Case 0 NL is not, and the reason is the Koiter set, not the mesh**: at
  ny = 160 the 5th distinct mode and the next one are 2.7e-5 apart, above the
  1e-5 of `koiter_cluster_rtol`, so the set stops at 10 modes, while at
  ny = 120 and 200 their split falls below 1e-5 and the set has 12. The
  symmetric/antisymmetric splitting of the end-localized modes is of that
  order and varies with the mesh, so a set defined by an equality of
  multipliers changes from mesh to mesh near the tolerance. A near-degenerate
  pair split by 2.7e-5 is also only resolved by the eigen solver to about
  tol/gap = 1e-6/2.7e-5.
- Case 6 LIN converges slowly (order 2.5), with gaps of 3-6e-5 at every mesh,
  a dense cluster; b_min_t is small, -0.05.
- Case 0 LIN oscillates within 5 % about 0.126, a positive b.
- The rotation of the pairs reproduces the shift by one element to 2e-4 at
  worst (0 LIN, ny = 80) and to 1e-5 or better from ny = 120.

**Cost.** 12 Koiter modes in 10 of the 24 runs; for those, 20 ms more per
element than with 10 (median, 9.6 to 38.7 ms), so 67 ms per element, about
55 ms on average, 60 ms in `generate_qsubs.py`:

| ny | core-hours (60 ms) | core-hours (67 ms) | jobs |
|---|---|---|---|
| 160 | 20,600 | 21,300 | 432 |
| 200 | 31,000 | 32,200 | 1,133 |

Peak memory 14.1 GB (case 0, NL, ny = 200, 70 min).

## All convergence studies and their data

Five studies of cases 0, 1 and 6, LIN and NL, ny = 80, 120, 160 and 200, 24
runs each, in the order they were run. The RESULT line of every run, with all
b_ijkl and a_ijk, is in [`results/`](results) as gzipped JSON lines, one
`{"file": ..., "result": ...}` per run (`result` null for a run that wrote
none), e.g.

    import gzip, json
    runs = [json.loads(l) for l in gzip.open('results/DOE09_conv_k5g.jsonl.gz', 'rt')]

| study | Koiter set | crest | normalization reported | runs with RESULT | table | data |
|---|---|---|---|---|---|---|
| single-mode | 1 mode | none | nodal b_1111 | 24 | [`tables/DOE09_convergence_single_mode.txt`](tables/DOE09_convergence_single_mode.txt) | outputs not on the cluster, table only |
| `_k5` nodal | first 5 distinct modes | none | nodal b_iiii | 23, one killed by the SuperLU fallback at 8 GB | [`tables/DOE09_convergence_k5_nodal.txt`](tables/DOE09_convergence_k5_nodal.txt) | [`results/DOE09_conv_k5_nodal.jsonl.gz`](results/DOE09_conv_k5_nodal.jsonl.gz) |
| `_k5` crest | first 5 distinct modes | edge envelope | nodal, crest and RMS b_iiii | 10, cancelled for the energy normalization | below | [`results/DOE09_conv_k5_crest.jsonl.gz`](results/DOE09_conv_k5_crest.jsonl.gz) |
| `_k5c` | 5 distinct modes and partners, 10 | 11 x 11 bicubic grid | energy b_min, b_min_t | 24 | [`tables/DOE09_convergence_k5c.txt`](tables/DOE09_convergence_k5c.txt), Section 7 | [`results/DOE09_conv_k5c.jsonl.gz`](results/DOE09_conv_k5c.jsonl.gz) |
| `_k5g` | at least 5 distinct modes, groups completed, and partners, 9 to 12 | element, refined, largest over rotations | energy b_min, b_min_t | 24 | [`tables/DOE09_convergence_k5g.txt`](tables/DOE09_convergence_k5g.txt), Current results | [`results/DOE09_conv_k5g.jsonl.gz`](results/DOE09_conv_k5g.jsonl.gz) |

Pcr agrees between all five, the Koiter section not changing the buckling
analysis. The `_k5` crest study, b_iiii of the 5 modes with the nodal
normalization, crest_w from the edge envelope (low by 0.5-0.8 %, see the
Update), and b_iiii with the crest normalization, b_iiii/crest_w**2:

| case, ny | b_iiii nodal | crest_w | b_iiii crest |
|---|---|---|---|
| 0 ny080 LIN | -0.0789/-0.0790/-0.1516/-0.1518/-0.0503 | 1.0486/1.0488/1.0463/1.0464/1.0292 | -0.0717/-0.0718/-0.1385/-0.1386/-0.0475 |
| 0 ny080 NL | -0.2430/-0.2432/-0.1326/-0.1327/-0.1549 | 1.0116/1.0116/1.0109/1.0109/1.0123 | -0.2375/-0.2376/-0.1298/-0.1298/-0.1512 |
| 0 ny120 LIN | 0.2733/0.2502/0.2945/0.2248/0.3160 | 1.0043/1.0038/1.0025/1.0003/1.0022 | 0.2709/0.2483/0.2930/0.2247/0.3147 |
| 1 ny080 LIN | -303.2/-222.4/-121.4/-102.7/-101.9 | 4.2764/4.2333/4.2327/4.1341/4.1181 | -16.58/-12.41/-6.775/-6.009/-6.011 |
| 1 ny080 NL | -0.2747/-0.2668/-0.2624/-0.2657/-0.3193 | 1.0242/1.0264/1.0257/1.0256/1.0227 | -0.2619/-0.2532/-0.2494/-0.2526/-0.3052 |
| 1 ny120 LIN | 0.0406/0.0275/0.0275/0.0415/0.0269 | 1.0440/1.0440/1.0441/1.0435/1.0447 | 0.0372/0.0252/0.0252/0.0381/0.0246 |
| 1 ny160 LIN | 0.0406/0.0272/0.0276/0.0415/0.0259 | 1.0440/1.0441/1.0439/1.0435/1.0448 | 0.0372/0.0250/0.0254/0.0381/0.0238 |
| 6 ny080 LIN | -1.9523/-1.9740/-0.8076/-0.8316/-5.1110 | 1.0300/1.0357/1.0272/1.0424/1.0246 | -1.8402/-1.8402/-0.7654/-0.7654/-4.8684 |
| 6 ny080 NL | -0.5362/-0.4485/-0.7212/-0.6129/-0.4558 | 1.0000/1.0000/1.0003/1.0082/1.0000 | -0.5362/-0.4485/-0.7208/-0.6029/-0.4558 |
| 6 ny120 LIN | 0.0165/-0.0074/0.0223/0.0166/0.0014 | 1.0000/1.0074/1.0000/1.0000/1.0001 | 0.0165/-0.0072/0.0223/0.0166/0.0014 |

In case 6, LIN, ny = 80, the crest normalization makes the two members of
each pair agree (-1.9523 and -1.9740 become -1.8402 twice), as
sec:normalisation of the implementation notes found for the rotation of a
pair.

### ny = 160 for the DOE

With the present setup (`_k5g`), what ny = 160 gives, from the table and
extrapolation of Current results:

| quantity | at ny = 160 |
|---|---|
| Pcr | within 0.5 % of ny = 200 (0.19 %, 0.08 %, 0.51 %) |
| b_min_t, 1 NL and 6 NL | within about 2.5 % of the extrapolated value |
| b_min_t, 1 LIN | mesh independent |
| b_min_t, 0 LIN | within 5 %, oscillating |
| b_min_t, 0 NL | 11 % from ny = 200, the Koiter set changing between meshes (10 modes at 160, 12 at 200) |
| b_min_t, 6 LIN | about 18 % from the extrapolated value, slow convergence of a small b (-0.05) |
| cost | about 20,600 core-hours, 432 jobs |

## Update: element crest and cut degenerate clusters

Two corrections since the first version of this report. Sections 5 to 7
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
half. Over the 24 runs of Section 7, the 5th distinct mode and the next one
share their multiplier to better than 1e-5 in:

| run | gap to the next distinct mode |
|---|---|
| 0 NL ny=120 | 5.3e-6 |
| 0 NL ny=200 | 2.9e-6 |
| 1 NL ny=80, 120, 160, 200 | 3.7e-13, 6.0e-15, 1.1e-15, 8.9e-15 |
| 6 LIN ny=80 | 3.1e-10 |
| 6 NL ny=120, 160, 200 | 2.5e-10, 1.9e-12, 1.3e-10 |

and to 3-4e-5 in 0 NL ny=160, 6 LIN ny=160 and 200. **The b_min and b_min_t
of Section 7 for these runs depend on round off.** The fix is a Koiter set
made of complete clusters: at least 5 distinct modes, extended to the end of
the cluster of the 5th, with their partners, 12 modes when the clusters are
four-fold as in the NL cases. That makes the number of Koiter modes vary from
run to run, and `fkoiter_cylinder_CTS_circum` takes koiter_num_modes before
its eigenvalue analysis; see [Decision](#9-decision), option 4.

The convergence study was rerun with these corrections and the groups
completed, see [Current results](#current-results-complete-clusters-study-_k5g);
the outputs of Section 7 are kept as `k5c_grid11` in the DOE09 directory.

## Summary

**See [Current results](#current-results-complete-clusters-study-_k5g)
for the latest setup and convergence study; the summary below is of the
first version.**

1. Pcr is converged at ny = 160: within 0.5 % of ny = 200.
2. The single-mode b and the multi-mode b_iiii of the models do not converge,
   and are not reproducible, for two independent reasons:
   - **normalization**: the modes are scaled by their largest *nodal*
     translation, which misses the crest of a mode between nodes (3-5 % at
     ny = 80, b moving by 6-10 %);
   - **mode set**: the critical modes of these cylinders come as symmetric and
     antisymmetric pairs of end-localized buckles. When the two share their
     multiplier to round off (case 1 with NLprebuck: 1.7e-14), each wave number
     has a four-dimensional eigenspace, of which the eigen solver returns an
     arbitrary two-dimensional slice. The 5-mode b_ijkl then depend on round
     off (13 % on b_iiii, 5-30 % on the minimum of b).
3. Following the multi-mode analysis of Rahman (2009), the expansion is now
   made on the 5 distinct modes **and their rotated partners**
   (koiter_num_modes = 10), made K_C-orthonormal, with the **energy
   normalization** lambda_i |d_i| = 1, and summarized by the most imperfection
   sensitive direction of Salerno, b_min. b_min is invariant under the choice
   of the eigenspace slice to 7e-6. `b_min_t` is b_min for the combined mode
   scaled to a crest of w equal to the thickness.
4. With this setup, b_min_t varies smoothly from ny = 120 on; ny = 160 is
   within 2-7 % of ny = 200 for NL and within 10 % for LIN, but it is not yet
   converged for all cases.
5. Cost: the 10-mode Koiter section takes 47 ms per element; the DOE is
   19,100 core-hours at ny = 160 and 28,900 at ny = 200.

## 1. Pcr and the single-mode b

Single-mode study, [`tables/DOE09_convergence_single_mode.txt`](tables/DOE09_convergence_single_mode.txt),
NLprebuck = True:

| case | Pcr ny=160 | Pcr ny=200 | diff | b ny=80 | b ny=120 | b ny=160 | b ny=200 |
|---|---|---|---|---|---|---|---|
| 0 | 3872.2 | 3864.8 | 0.19 % | -0.243 | -0.334 | -0.235 | -0.220 |
| 1 | 8961.5 | 8954.2 | 0.08 % | -0.264 | -0.230 | -0.263 | -0.281 |
| 6 | 4558.0 | 4534.7 | 0.51 % | -0.536 | -0.244 | -0.275 | -0.330 |

b oscillates with the mesh. mu1_ratio, the multiplier of the second mode over
the critical one, is 1.00001 to 1.002, below the 0.005 gap for which the
single-mode b is determinate (sec:convergence of the manuscript).

## 2. The 5-mode study with the nodal normalization

[`tables/DOE09_convergence_k5_nodal.txt`](tables/DOE09_convergence_k5_nodal.txt)
(koiter_num_modes = 5, 23 of 24 runs; case 6, NL, ny = 120 segfaulted when the
SuperLU fallback ran out of its 8 GB, and its rerun was superseded).

The b_iiii of individual modes are not comparable across meshes, the modes
changing order (case 0 NL: critical n = 30 at ny = 160, 31 at ny = 200). The
minimum over unit xi of the quartic form b(xi) = b_ijkl xi_i xi_j xi_k xi_l
of the nodal-scaled modes, over the critical pair and over all 5 modes:

| case | ny=80 | ny=120 | ny=160 | ny=200 |
|---|---|---|---|---|
| 0 NL, critical pair | -0.487 | -0.670 | -0.473 | -0.248 |
| 0 NL, 5 modes | -1.083 | -0.916 | -0.604 | -0.612 |
| 1 NL, critical pair | -0.348 | -0.303 | -0.284 | -0.321 |
| 1 NL, 5 modes | -0.656 | -0.451 | -0.435 | -0.640 |
| 6 NL, critical pair | -0.855 | | -0.301 | -0.331 |
| 6 NL, 5 modes | -1.474 | | -0.592 | -0.584 |
| 0 LIN, 5 modes | -0.344 | 0.219 | 0.224 | 0.238 |
| 1 LIN, 5 modes | -1528 | 0.011 | 0.021 | 0.021 |
| 6 LIN, 5 modes | -5.594 | -0.103 | -0.119 | -0.147 |

Changes of 10-50 % from ny = 160 to 200, not monotone.

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

## 7. Convergence with the closed basis and the energy normalization

**Note (Update):** these results use the old crests (points 1 and 2 of the
Update), and in 10 of the 24 runs the Koiter set cuts a degenerate cluster
(point 3), so b_min and b_min_t of those runs depend on round off.

[`tables/DOE09_convergence_k5c.txt`](tables/DOE09_convergence_k5c.txt), all
24 runs finished, no PARDISO fallback to SuperLU, no incomplete nonlinear
pre-buckling iteration. Case 6, NL, ny = 120, which segfaulted in the study
of Section 2, finished at 8 GB with a 2.8 GB peak.

b_min_t:

| case | ny=80 | ny=120 | ny=160 | ny=200 | 160 to 200 |
|---|---|---|---|---|---|
| 0 LIN | -0.0722 | 0.1267 | 0.1240 | 0.1294 | 4.3 % |
| 0 NL | -0.1281 | -0.1464 | -0.1576 | -0.1658 | 5.2 % |
| 1 LIN | -0.8761 | 0.03756 | 0.03756 | 0.03756 | 0 % |
| 1 NL | -0.2301 | -0.1733 | -0.1792 | -0.1910 | 6.6 % |
| 6 LIN | -1.7409 | -0.0373 | -0.0486 | -0.0534 | 9.8 % |
| 6 NL | -0.4224 | -0.1919 | -0.2189 | -0.2239 | 2.3 % |

b_min_energy and crest_e:

| case | b_min_energy ny=120 / 160 / 200 | crest_e ny=120 / 160 / 200 |
|---|---|---|
| 0 LIN | 0.267 / 0.289 / 0.316 | 1.451 / 1.526 / 1.562 |
| 0 NL | -22.53 / -19.86 / -21.21 | 12.41 / 11.23 / 11.31 |
| 1 LIN | 0.0083 / 0.0083 / 0.0083 | 0.470 / 0.470 / 0.470 |
| 1 NL | -0.545 / -0.550 / -0.634 | 1.773 / 1.753 / 1.821 |
| 6 LIN | -0.592 / -0.803 / -0.860 | 3.985 / 4.064 / 4.016 |
| 6 NL | -7.539 / -7.760 / -8.215 | 6.268 / 5.954 / 6.057 |

Richardson extrapolation of b_min_t from ny = 120, 160 and 200:

| case | order | extrapolated | error at 160 | error at 200 |
|---|---|---|---|---|
| 0 LIN | not monotone | | | |
| 0 NL | 0.20 | -0.347 | 55 % | 52 % |
| 1 LIN | constant | 0.0376 | 0 | 0 |
| 1 NL | not monotone (step ratio 2.0) | | | |
| 6 LIN | 2.36 | -0.0602 | 19 % | 11 % |
| 6 NL | 5.44 | -0.2260 | 3.1 % | 0.9 % |

Observations:

- ny = 80 is unusable: the signs flip against the finer meshes.
- From ny = 120 on the values vary smoothly, against the 10-50 % oscillation of
  Section 2. Case 6 NL is in the asymptotic range; cases 0 NL and 1 NL still
  drift by 5-7 % per step, and the extrapolation for 0 NL (order 0.2) is not
  reliable, so their error at ny = 160 may exceed the 160 to 200 difference.
- Case 1 LIN is governed by an axisymmetric mode, e_min = mode 0 alone, and
  is mesh independent from ny = 120.
- The minimum directions mix all 10 modes; e_min of every run is in the
  RESULT line of its output.

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

Open, one of:

1. **ny = 240 and 280 for cases 0, 1 and 6, LIN and NL, first** (12 runs, a
   few hours each), to see whether b_min_t settles before committing the DOE.
   Recommended in the first version; superseded by option 4, which has to
   come first.
2. **DOE at ny = 160**, 19,100 core-hours, with a mesh uncertainty of about
   5-10 % on b_min_t from this study; Pcr converged.
3. **DOE at ny = 200**, 28,900 core-hours, not shown to be converged either.
4. **Complete clusters first** (see the Update at the top), then rerun the
   convergence study. **Done, (b)**, see Current results at the top:
   - (a) in the DOE09 driver: run the model with 10 Koiter modes, and when the
     final eigenvalue analysis shows that the 5th distinct mode cuts a cluster,
     run it again with the number of modes that completes it. No library
     change, but the runs concerned cost twice (10 of the 24 of the study,
     mostly NLprebuck);
   - (b) in the library, on this branch: let koiter_num_modes be chosen after
     the eigenvalue analysis, e.g. a minimum number of distinct modes
     completed to whole clusters. No rerun, but the DOE then runs on the
     branch until it is released;
   - either way, 12 modes instead of 10 for most NL cases: 78 bordered solves
     instead of 55, and about 1.5 times the Koiter cost of 10 modes.

Open after the `_k5g` study:

5. **Group near-degenerate modes as well**: raise `koiter_cluster_rtol` from
   1e-5 to about 1e-4, above the symmetric/antisymmetric splittings seen
   (2.7e-5 to 6e-5 in 0 NL and 6 LIN), so that the set does not change from
   mesh to mesh; check first, on the 24 runs, how many modes that takes, the
   dense cluster of 6 LIN possibly chaining beyond num_eigvals. Recommended
   before the DOE, as it is what keeps case 0 NL from converging.
6. **DOE at ny = 160 with the present set**, 20,600 core-hours: b_min_t
   within about 2.5 % for 1 NL and 6 NL, 5 % for 0 LIN, 10-20 % for 0 NL and
   6 LIN, and Pcr within 0.5 %.
7. **DOE at ny = 200**, 31,000 core-hours: within 1 % for 1 NL and 6 NL,
   but 0 NL and 6 LIN still not shown to be converged.

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
    'element_orbit', which `generate_qsubs.py`, `generate_qsubs_convergence.py`
    and `post.py` require of an output.
- `koiter_post.py`: `rescaled`, `energy_scales`, `symmetrized`,
  `min_direction`.
- `post.py`: `DOE09_koiter.npz` with the nodal b_ijkl and a_ijk of the 10
  modes, the scales of the other normalizations, b_min_energy, b_min_t,
  crest_e and e_min; b_min_t and b_min_energy in `DOE09_output.txt`.
- `post_convergence.py`: columns b_min_t, b_min_energy, crest_e, b_iiii_crest,
  b_iiii_rms, crest_w; default study `_k5c`.
- `generate_qsubs.py`: requires the closed 10-mode setup, reruns outputs of
  earlier setups, `koiter_time_per_element = 0.047`, `python -u` so that a
  crashed run keeps its log.
- `generate_qsubs_convergence.py`: suffix `_k5c`, walltime 12 h, `python -u`,
  reruns outputs without b_min_energy.

The scripts in [`checks/`](checks) were run from the DOE09 directory, next to
`DOE09.txt`, with the `run_case.py` of their stage: `crest_methods.py` and
`element_crest_reference.py` with the one of the Update, `mode_pairs.py`,
`pair_mixing.py` and `same_process_resolve.py` with the 5-mode one, `fourfold_degeneracy.py` and
`eigenspace_slice.py` with the 10-mode one of [`scripts/`](scripts).

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
