# SS4 edges and the ny convergence from a low ny

Status report of 2026-09-24. It complements
[`REPORT_reassessment_draft.md`](REPORT_reassessment_draft.md) (studies a to
k) and will be merged into `REPORT.md` with it.

## 1. What changed in the library

Commit `6eefe0c`, *ENH: SS4 edges, u uniform along each edge, in the Koiter
models*.

- New module [`bfsccylinder_models/edges.py`](../../../bfsccylinder_models/edges.py):
  `edge_space(x, L, DOF, edges, y)` returns an `EdgeSpace`, the independent
  unknowns `a` of the displacement vector `u = T a`.
  - `space.matrix(K)` = `T^T K T`, `space.force(f)` = `T^T f`,
    `space.expand(a)` = `T a`, `space.restrict(u)` = `a`.
  - `space.free` is the mask of the DOFs not fixed to zero (the tied ones
    included), what the cyclic-symmetry helpers take as `bu`.
- New keyword `edges='SS3'` (default) or `edges='SS4'` in the four Koiter
  models: `koiter_cylinder.py`, `koiter_cylinder_sanders.py`,
  `koiter_cylinder_CTS.py`, `koiter_cylinder_CTS_sanders.py`. Every
  `K[bu][:, bu]`, `f[bu]` and `u[bu] = ...` goes through the `EdgeSpace`
  instead. `linbuck_VAFW.py` is unchanged.
- `canonical_modes` (cyclic_symmetry.py) accepts the `EdgeSpace` in place of
  the mask.

### The two edge conditions

| | x = 0 | x = L | axial rigid body |
|---|---|---|---|
| **SS3** (default, as in `bd740a8`) | v = v,y = w = w,y = 0, u free | same | u = 0 at the single node x = L/2, y = 0 |
| **SS4** | SS3 conditions + u = u,y = 0 | SS3 conditions + u,y = 0, and the u of every edge node tied to **one shared unknown** | removed by u = 0 at x = 0 |

- SS4 is still **force controlled**: the same Nxx is applied at x = L, and
  its resultant is the reaction of the shared unknown.
- The tie together with u,y = 0 makes u uniform along the whole edge, for the
  same Hermite argument as v with v,y in `bd740a8`.
- The tie is an exact change of basis, not a penalty.
- For SS3, `T` is the plain selection of the free DOFs (the code even keeps
  the old indexing path), so **SS3 results are unchanged to the last digit**.
- The axisymmetric pre-buckling subspace and the rotation of a mode by one
  element both keep the tied displacements equal, so the rest of the
  algorithm carries over unchanged.
- The axisymmetric projection now removes the axial translation at station
  x = 0 for SS4 (x = L/2 for SS3).

### Tests

- The 37 existing tests pass unchanged.
- New [`tests/test_edges.py`](../../../tests/test_edges.py), 3 tests:
  - the SS3 space is the selection;
  - the SS4 space: u uniform at x = L, zero at x = 0, and `T^T f` of the
    tied unknown is the resultant of the nodal forces;
  - Waters shell, ny = 40, SS4: the pre-buckling state, the mode and the
    second-order field keep u uniform at x = L and zero at x = 0; the
    regression values; and SS4 Pcr > SS3 Pcr.

`doc/nlprebuck_implementation.tex`, section *Edge conditions*, has a new SS-4
paragraph.

### First measurement: the Waters shell, Sanders, ny = 40

The shell of `tests/test_koiter_cylinder_Waters_sanders.py`, Nxxunit =
1000 N/m, one Koiter mode:

| pre-buckling | Pcr SS3 (N) | Pcr SS4 (N) | change | b_1111 SS3 | b_1111 SS4 |
|---|---|---|---|---|---|
| linear | 184158 | 194510 | +5.6 % | -0.0441 | +0.1369 |
| nonlinear, eps1 = 0.0005 | 177474 | 186757 | +5.2 % | -0.0522 | -0.2973 |

So on this shell, b is very sensitive to the axial edge condition.

## 2. Driver and job scripts (this commit)

- [`scripts/run_case.py`](scripts/run_case.py):
  - new option `--edges SS3|SS4` (default SS3);
  - records `edges` in the RESULT line;
  - the library check now requires the `edges` parameter. The old check
    looked for the source line `bk[8::DOF] = checkSS`, which no longer
    exists, so without this fix every new job would have refused to start.
  - `distinct_first` works on the `EdgeSpace`.
- [`scripts/generate_qsubs_reassess.py`](scripts/generate_qsubs_reassess.py):
  new study **(m)**.
- [`checks/reassessment_post.py`](checks/reassessment_post.py):
  - `table_l` and `table_m`;
  - studies l and m added to the archive list. l was missing, so its runs
    would never have been archived.

## 3. Study (m): SS3 and SS4 from a low ny

All runs: DOE09 cases 0, 1 and 6 (designs A, B, C), NL, eps1 = 0.0005, axial
factor F = 1 (the setup of study k), 5 distinct Koiter modes with complete
clusters.

- SS4: ny = 40 to 240 (case 1 up to 200).
- SS3: ny = 40, 60 and 80, plus ny = 120 for case 6. The SS3 rows from
  ny = 120 (160 for case 6) are those of study (k).

30 runs, all finished, archived in
[`results/DOE09_reassess_m.jsonl.gz`](results/DOE09_reassess_m.jsonl.gz).

Column key:

- **ny/n_c**: elements per wave of the critical wave number n_c.
- **m**: number of Koiter modes.
- **b_min_t**: b_min_t of the full set.
- **crit b_t**: b_min_t of the critical cluster alone.
- **win1 b_t**: b_min_t of the window n_c ± 1 (`*` = a cluster of the
  window is incomplete).
- **l/lc**: lambda_b/lambda_c, 0.9995 to 0.9998 in every run.

### Case 0 (design A)

| ny | nx | ny/n_c | n_c | Pcr SS3 | Pcr SS4 | m SS3/SS4 | b_min_t SS3 | b_min_t SS4 | crit b_t SS3 | crit b_t SS4 | win1 SS3 | win1 SS4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | 61 | 2.0 / 2.0 | 20 / 20 | 5166.04 | 5496.83 | 8 / 8 | -0.0252 | -0.0143 | -0.0371 | -0.0338 | - | - |
| 60 | 81 | 2.3 / 2.0 | 26 / 30 | 4362.39 | 4528.26 | 12 / 12 | -0.0570 | -0.0729 | -0.0779 | -0.0262 | -0.0568 | - |
| 80 | 91 | 3.0 / 2.9 | 27 / 28 | 4130.13 | 4257.35 | 10 / 10 | -0.1040 | -0.0745 | -0.1607 | -0.1180 | -0.1040* | -0.0747* |
| 120 | 121 | 4.0 / 3.9 | 30 / 31 | 3919.01 | 4006.62 | 12 / 10 | -0.1452 | -0.1199 | -0.2006 | -0.1864 | -0.1451 | -0.1199* |
| 160 | 161 | 5.3 / 5.0 | 30 / 32 | 3876.86 | 3957.53 | 10 / 12 | -0.1569 | -0.1435 | -0.2276 | -0.2157 | -0.1570* | -0.1434 |
| 200 | 191 | 6.5 / 6.2 | 31 / 32 | 3862.97 | 3942.97 | 12 / 12 | -0.1728 | -0.1507 | -0.2388 | -0.2262 | -0.1729 | -0.1507 |
| 240 | 221 | 7.7 / 7.5 | 31 / 32 | 3857.38 | 3937.01 | 12 / 12 | -0.1759 | -0.1537 | -0.2431 | -0.2304 | -0.1761 | -0.1537 |

### Case 1 (design B)

| ny | nx | ny/n_c | n_c | Pcr SS3 | Pcr SS4 | b_min_t SS3 | b_min_t SS4 | crit b_t SS3 | crit b_t SS4 | win1 SS3 | win1 SS4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | 47 | 3.1 / 3.1 | 13 / 13 | 14150.37 | 14150.20 | -0.1915 | -0.1858 | -0.2366 | -0.2367 | -0.1785* | -0.1743* |
| 60 | 55 | 3.0 / 2.9 | 20 / 21 | 13436.63 | 13687.27 | -0.1185 | -0.0869 | -0.1633 | -0.1474 | -0.1185 | -0.0867 |
| 80 | 67 | 3.6 / 3.3 | 22 / 24 | 12954.40 | 13164.81 | -0.2057 | -0.1585 | -0.2850 | -0.2562 | -0.2057 | -0.1581 |
| 120 | 127 | 5.0 / 4.8 | 24 / 25 | 8984.49 | 9164.60 | -0.1691 | -0.1188 | -0.2341 | -0.2085 | -0.1689 | -0.1188 |
| 160 | 127 | 6.7 / 6.4 | 24 / 25 | 8956.55 | 9134.75 | -0.1819 | -0.1318 | -0.2521 | -0.2278 | -0.1821 | -0.1317 |
| 200 | 127 | 8.3 / 8.0 | 24 / 25 | 8948.73 | 9127.09 | -0.1852 | -0.1352 | -0.2567 | -0.2329 | -0.1855 | -0.1354 |

m = 12 from ny = 60 (10 at ny = 40). Pcr drops from 12954 N to 8984 N
between ny = 80 and 120: below ny = 120 the meshes miss the critical mode of
case 1 entirely.

### Case 6 (design C)

| ny | nx | ny/n_c | n_c | Pcr SS3 | Pcr SS4 | m SS3/SS4 | b_min_t SS3 | b_min_t SS4 | crit b_t SS3 | crit b_t SS4 | win1 SS3 | win1 SS4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | 25 | 2.0 / 2.0 | 20 / 20 | 8176.13 | 8176.11 | 7 / 7 | -0.0830 | -0.0803 | -0.1333 | -0.1328 | - | - |
| 60 | 33 | 2.4 / 2.4 | 25 / 25 | 5925.25 | 5925.25 | 10 / 10 | -0.2072 | -0.2057 | -0.2525 | -0.2522 | - | - |
| 80 | 41 | 3.0 / 3.0 | 27 / 27 | 5166.46 | 5166.46 | 10 / 10 | -0.4176 | -0.4160 | -0.5726 | -0.5725 | - | - |
| 120 | 65 | 4.1 / 4.0 | 29 / 30 | 4574.21 | 4636.42 | 12 / 12 | -0.1733 | -0.1437 | -0.2390 | -0.2247 | -0.1732 | -0.1435 |
| 160 | 81 | 5.3 / 5.2 | 30 / 31 | 4552.70 | 4607.09 | 12 / 12 | -0.2197 | -0.1839 | -0.3030 | -0.2813 | -0.2197 | -0.1839 |
| 200 | 97 | 6.7 / 6.5 | 30 / 31 | 4526.46 | 4576.34 | 12 / 12 | -0.2257 | -0.1901 | -0.3117 | -0.2906 | -0.2257 | -0.1901 |
| 240 | 121 | 8.0 / 7.7 | 30 / 31 | 4507.61 | 4555.46 | 12 / 14 | -0.2307 | -0.1978 | -0.3185 | -0.2969 | -0.2308 | -0.1942 |

### Richardson extrapolation in ny, ny >= 120

Each fit uses three consecutive meshes. "err" is the error of the finest of
the three against the extrapolated value.

| case | quantity | SS3: meshes, order, extrapolated, err | SS4: meshes, order, extrapolated, err |
|---|---|---|---|
| 0 | Pcr | 160-200-240: p 3.5, 3851.0 N, +0.2 % | 160-200-240: p 3.4, 3930.0 N, +0.2 % |
| 0 | b_min_t | 160-200-240: p 6.9, -0.1771, +0.7 % (120-160-200 not monotone) | 160-200-240: p 3.3, -0.1573, +2.3 % |
| 0 | crit b_t | 160-200-240: p 3.6, -0.2479, +1.9 % | 160-200-240: p 3.4, -0.2353, +2.1 % |
| 1 | Pcr | 120-160-200: p 3.9, 8943.1 N, +0.1 % | 120-160-200: p 4.2, 9122.2 N, +0.1 % |
| 1 | b_min_t | 120-160-200: p 4.3, -0.1873, +1.1 % | 120-160-200: p 4.2, -0.1374, +1.6 % |
| 1 | crit b_t | 120-160-200: p 4.2, -0.2596, +1.1 % | 120-160-200: p 4.1, -0.2364, +1.5 % |
| 6 | Pcr | 160-200-240: p 0.6, 4353.6 N, +3.5 % (120-160-200 not monotone) | 160-200-240: p 0.9, 4440.2 N, +2.6 % (120-160-200 not monotone) |
| 6 | b_min_t | 120-160-200: p 6.7, -0.2275, +0.8 %; 160-200-240 not monotone | 120-160-200: p 6.2, -0.1922, +1.1 %; 160-200-240 not monotone |
| 6 | crit b_t | 120-160-200: p 6.6, -0.3143, +0.8 %; 160-200-240: p 0.2, unusable | 120-160-200: p 5.9, -0.2940, +1.1 %; 160-200-240: p 0.9, unusable |

### Findings

1. **Below ny = 120 nothing is converged, for either edge condition.**
   - At ny = 40 to 80 there are only 2 to 3.6 elements per wave, and the
     critical wave number is still moving: 20 → 27 (case 0), 13 → 22
     (case 1), 20 → 27 (case 6).
   - Pcr is still 7 to 45 % above its fine-mesh value (case 1 at ny = 80:
     12954 N against 8949 N).
   - b_min_t changes by up to a factor of 5 (case 6: -0.083 to -0.418),
     with no monotone trend.
   - No three-mesh fit over these meshes is usable. The low-ny end only
     confirms the lower bound of about 5 elements per wave of n_c found
     before.
2. **SS4 changes the value, not the convergence.** From ny = 120, SS3 and
   SS4 converge at the same order in ny, about 3.5 (case 0) and 4 (case 1),
   with similar errors at the finest mesh (0.7-2.3 %).
3. **SS4 compared with SS3, converged values:**

   | case | Pcr | b_min_t | crit b_t |
   |---|---|---|---|
   | 0 | +2.1 % | -0.157 vs -0.177, 11 % less negative | -0.235 vs -0.248, 5 % |
   | 1 | +2.0 % | -0.137 vs -0.187, 27 % less negative | -0.236 vs -0.260, 9 % |
   | 6 | +1.1 % at ny = 160-240 | -0.198 vs -0.231 at ny = 240, 14 % less negative | -0.297 vs -0.319 at ny = 240, 7 % |

   The edge condition moves b by more than the remaining mesh error, so the
   DOE has to state which edge condition it uses.
4. **Case 6 is not converged at F = 1 for either edge condition.**
   - Pcr falls 0.5 % per ny step at 160-240, with order below 1.
   - b_min_t is not monotone over 160-200-240.
   - SS4 ny = 240 takes 14 Koiter modes instead of 12, a cluster entering
     the set.
   - Studies (b) and (e) found that case 6 needs the axial refinement
     F = 1.5. These F = 1 sequences are therefore limited by the axial mesh,
     and do not measure the ny error of case 6.
5. **Case 6 at ny = 40-80:** SS3 and SS4 agree to 4-5 digits in Pcr and
   to 0.3-3 % in b. The critical modes of those coarse meshes carry almost no
   axial displacement at the edges. This is a property of the coarse modes,
   not a result about the edge condition.

## 4. Still running and still to do

- **Study (l)** (four more DOE designs, R/h 1538/1461/1148/419, at the
  candidate setup ny = 160/200/240, F = 1.5, eps1 = 0.0005, Nxxunit =
  500 N/m): 3 jobs still running at the time of writing.
- When the queue empties, `~/DOE09/reassess_logs/finalize.sh` archives every
  study and writes `tables_final.txt` with all tables, a to m.
- Then: the REPORT.md section after the Reassessment (truncation, per-cluster
  mesh error, eps1, the ny mechanism, the mesh rule), the Decision, and the
  constants of `generate_qsubs.py`.

**Open question for you:** should the mesh recommendation and the DOE runs
use SS3 (the edge condition of all studies so far) or SS4? Or should the
report give both? From finding 2, the mesh rule is the same for both; what
changes is b itself, by 11 to 27 %.
