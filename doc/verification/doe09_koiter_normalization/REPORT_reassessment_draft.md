## Reassessment runs: the quantity, the expansion point and the mesh

<!-- DRAFT, merged into REPORT.md after the Reassessment once studies (f) to
(k) are complete; sections marked PENDING wait for them -->

Studies (a) to (k) of
[`scripts/generate_qsubs_reassess.py`](scripts/generate_qsubs_reassess.py),
run from the DOE09 working directory; tables of
[`checks/reassessment_post.py`](checks/reassessment_post.py), RESULT lines in
[`results/DOE09_reassess_<study>.jsonl.gz`](results). New options of
`run_case.py`, each recorded in the RESULT line and each leaving the RESULT
of a run without it unchanged: `--distinct`, `--num-eigvals`,
`--axial-factor`, `--eps1`, `--nint`, `--kinematics`, `--thickness-factor`,
`--nxxunit`.

| study | runs | question |
|---|---|---|
| (a) truncation | cases 0, 1, 6 NL, ny = 120 and 160, K = 5, 7, 9 distinct modes; 0 and 6 LIN, ny = 160, K = 9 | does b_min settle as clusters are added |
| (b) axial | cases 0, 1, 6 LIN and NL, ny = 160, dx max = dy/F, F = 1.5, 2, 3 | axial part of the mesh error |
| (c) expansion point | cases 0, 1, 6 NL, ny = 120, eps1 = 0.005 to 0.0005 | sensitivity to lambda_b/lambda_c |
| (d) ny = 240 | cases 0, 1, 6 LIN and NL, the `_k5g` setup | fourth circumferential mesh |
| (e) | the NL sequences of (b) and (d) with eps1 = 0.0005 | the mesh error with the expansion point fixed |
| (f) | case 1 NL at F = 2.5, 4, 5; ny = 240 at F = 1.5, cases 1 and 6; case 0 ny = 280 | case 1 axially, the candidate mesh |
| (g) | case 1 NL, ny = 120 and 200 at F = 2 to 5 | a grid in ny and nx |
| (h) quadrature | case 0 NL, ny = 160, 5 and 6 Gauss points per direction | under-integration of phi4 |
| (i) kinematics | cases 0 and 6 NL, Donnell | the v/R terms of Sanders |
| (j) R/h | case 6 NL, tow thickness x 0.7 and x 2 | whether the error grows with R/h |
| (k) edges | cases 0, 1, 6 NL, v = w = 0 along the whole edge | the simply supported edge condition |

From (e) on, every run is NL with eps1 = 0.0005.

### 1. Truncation: b_min_t of a window about n_c settles, the full set does not

(a), b_min_t over the prefix and window subsets of the K = 9 run (clusters in
the order of the set, and wave numbers n_c +- j):

| NL run | one cluster | window n_c +- 1 | window n_c +- 2 | change +-1 to +-2 | full set K = 5 / 7 / 9 |
|---|---|---|---|---|---|
| 0, ny=120 | -0.1994 | -0.14421 | -0.14184 | +1.6 % | -0.1442 / -0.1384 / -0.1421 |
| 0, ny=160 | -0.2273 | -0.16456 | -0.15908* | +3.3 % | -0.1567 / -0.1628 / -0.1588 |
| 1, ny=120 | -0.2447 | -0.17685 | -0.17334 | +2.0 % | -0.1766 / -0.1815 / -0.1732 |
| 1, ny=160 | -0.2612 | -0.18846 | -0.18492 | +1.9 % | -0.1886 / -0.1788 / -0.1849 |
| 6, ny=120 | -0.2492 | -0.18081 | -0.17940 | +0.8 % | -0.1810 / -0.1785 / -0.1792 |
| 6, ny=160 | -0.3133 | -0.22734 | -0.22349 | +1.7 % | -0.2271 / -0.2264 / -0.2235 |

\* the n = 28 cluster is cut at ny = 160 in case 0.

- b_min_energy does not settle: it grows about linearly with the number of
  clusters (case 0 NL ny = 120: -10.8, -23.1, -37.6 for 1, 3 and 5
  clusters). b_min_t does from three clusters on, crest_e**2 growing with it
  (7.36, 12.67, 16.27).
- The windows settle best, 0.8 to 3.3 % from n_c +- 1 to n_c +- 2. The full
  sets, taken in multiplier order, swing by up to 5 % (case 1), the fourth
  cluster falling on one side of n_c.
- A subset of a larger run is the smaller run: b_min_energy agrees to 5
  digits and b_min_t to 0.3 % or better, so the second-order orthogonality
  to the other modes has no measurable effect, and every window can be read
  out of one run. The exception is a smaller run with a cut cluster (0 NL,
  ny = 160, K = 5: -5 %).
- e_min spreads evenly over the clusters present, 0.13 to 0.26 each with
  five.
- A single cluster, b_min_t -0.20 to -0.31, is 28 % more negative than the
  window n_c +- 1, and the window of three is within 0.8 to 3.3 % of the
  window of five.

### 2. The cluster of a wave number

A cluster is now the lowest symmetric and antisymmetric mode of n and their
rotated partners, 4 modes (`clusters` of
[`checks/cluster_subsets.py`](checks/cluster_subsets.py) and
`cluster_subsets` of `run_case.py`). A further distinct mode of the same n,
another axial shape, enters the set on some meshes only, and made the
cluster of n a different quantity from mesh to mesh:

- 6 LIN: b_min of n = 26, -0.201, -0.697, -0.703 at ny = 160, 200, 240
  with every mode of n, is -0.201, -0.203, -0.207 with the lowest pair; the
  remark of the Reassessment that the LIN cases behave differently came
  partly from this;
- case 6 NL at thickness x 2: n = 22 went -2.03, -1.29, -1.32 at ny = 112,
  140, 168, and goes -1.217, -1.293, -1.320.

The NL tables of designs A, B and C are unchanged by it, each n having only
its lowest pair in their sets.

### 3. The expansion point

(c) and the ny = 120 runs of (e), linear fit in lambda_b/lambda_c and its
value at 1:

| NL, ny = 120 | lambda_b/lambda_c at eps1 = 0.005 | at 0.0005 | b of the critical cluster, eps1 = 0.005 against the fit at 1 | b_min_t window n_c +- 1, same | Pcr, same |
|---|---|---|---|---|---|
| 0 | 0.99748 | 0.99977 | -1.1 % | -0.8 % | +0.11 % |
| 1 | 0.99739 | 0.99974 | -4.6 % | -4.9 % | +0.08 % |
| 6 | 0.99676 | 0.99970 | -2.8 % | -2.8 % | +0.18 % |

- b of the critical cluster moves by 1.8 % (case 1), 0.8 % (6) and 0.35 %
  (0) per 0.001 of lambda_b/lambda_c, linearly (rms residual of the fit
  below 1e-4 of the value).
  The default eps1 = 0.005 leaves lambda_b/lambda_c anywhere from 0.9951 to
  0.9983 depending on the load stepping, so from mesh to mesh the expansion
  point moved b by as much as the mesh error being measured.
- **eps1 = 0.0005** puts lambda_b/lambda_c at 0.9996 to 0.9998 in every run
  of (e) to (k), and b within 0.1 % (case 0), 0.25 % (6) and 0.5 % (1) of
  its value at 1. It costs 2 to 5 more load steps.
- **The load unit.** The load stepping starts at lambda = 1, the load of
  `Nxxunit` = 1000 N/m, 2513 N. Case 6 at thickness x 0.7 has Pcr of about
  2470 N: the first step overshot and the expansion point stayed at
  lambda_b/lambda_c 1.002 to 1.023, past the bifurcation, with only a
  WARNING in the output. (j) was rerun with `--nxxunit 500`. Any DOE design
  with Pcr below 2513 N does the same; see the Decision.

### 4. What does not cause the ny error

- **Quadrature (h).** 5 and 6 Gauss points per direction against the
  default 4, case 0 NL, ny = 160: b of the critical cluster changes by
  0.002 %, b_min_t by 0.003 %. phi4, with products of four derivatives of
  the bicubic w, is under-integrated by 4 points, but the error is
  negligible.
- **Kinematics (i).** PENDING (Donnell equals Sanders to 0.1-0.2 % at
  ny = 120-200 so far).
- **R/h (j).** PENDING.
- **Edge condition (k).** PENDING.

### 5. The mesh, per design

PENDING: circumferential and axial parts per case from (e), (f), (g), (k),
the 2D fit of case 1, and the elements per wave.

### 6. Recommendation

PENDING.
