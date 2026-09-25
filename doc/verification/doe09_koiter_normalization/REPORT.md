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

The study is restarted from a low ny on the edges that every model of the
library now has, **SS3-IR** (`bfsccylinder_models/edges.py`): v = w = 0
along both edges, the axial load on both edges, no node anchored, the axial
translation removed by inertia relief (mass-weighted mean axial
displacement zero). It gives the results of the SS3 edges anchored at one
node to round off, 1e-12 in Pcr and 5e-9 in b_1111 on the Waters shell, the
translation being a null vector of every operator.

The first 162 runs were made with free edges as well, **free-IR**: no edge
condition, all six rigid body modes removed by inertia relief. That option
was removed from the models afterwards (it is in git history up to commit
`1fc65e2`), for the reason of finding 5 below.

[`scripts/generate_qsubs_convergence.py`](scripts/generate_qsubs_convergence.py):
cases 0, 1 and 6, NL, eps1 = 0.0005, ny = 24, 32, 40, 48, 64, 80, 96, 120
and 160, and axial factors F = 0.5, 1 and 2 (largest axial element length
dy/F: elements twice as long axially, square, half as long), then ny = 200
and 240 at F = 2 and F = 3 at ny = 120 and 160; 93 SS3-IR runs. Tables:
[`checks/convergence_post.py`](checks/convergence_post.py), results of the
SS3-IR runs in `results/DOE09_conv_ir.jsonl.gz`.

The sections below describe the method (normalization, mode set, crest);
the ny quoted in them locate their evidence, not a mesh recommendation.

### Results of the first runs (2026-09-25)

All 162 runs, 81 SS3-IR and 81 free-IR, finished, none with an error.
Tables of the SS3-IR runs below; b_min_t is that of
the full Koiter set (at least 5 distinct modes, complete clusters), crit b_t
that of the critical cluster alone, m the number of Koiter modes.

**Findings**

1. **F = 0.5 does not coarsen cases 0 and 1.** The CTS mesher keeps a
   minimum number of nodes per transition and plateau region, a floor finer
   than 2 dy, so F = 0.5 gives the meshes of F = 1. For case 6 it does
   coarsen, and it is clearly worse: at ny = 160, Pcr 4890 N against
   4518-4553 N, and b_min_t -0.63 against -0.22. **Elements longer axially
   than around are too coarse.** Testing them on cases 0 and 1 would need an
   nx below the mesher's floor.
2. **The axial mesh decides case 1.** With nx = 67 (dx = 23.4 mm, F <= 1 up
   to ny = 96, and F = 0.5 up to 160), Pcr settles near 12,780 N at n = 23:
   the mesh misses the critical mode, and refining ny does not recover it.
   From nx = 127 (dx <= 11.7 mm), Pcr is 8950-9400 N and n = 23-24, as on the
   fine meshes of the removed studies.
3. **Below about 3.5 elements per wave of n_c nothing is usable**, for
   either edge condition. n_c keeps jumping (8, 16, 20, 26... for case 0),
   Pcr is 10-70 % high, and b changes sign and size. No Richardson estimate
   over those meshes is meaningful.
4. **SS3-IR, from ny = 96:**
   - case 0: F = 1 and F = 2 agree (nx >= ny already); Pcr converges at
     order 3.4, 0.6 % high at ny = 160; b_min_t at order 4-5, 2.4-3.2 % from
     its estimate -0.161 at ny = 160; the critical cluster alone is slower,
     order 2-2.5 and 10-12 %;
   - case 1 (F = 2): Pcr 0.5 % from its estimate 8913 N at ny = 160; b_min_t
     -0.158, -0.168, -0.186 at ny = 96, 120, 160 (nx 127, 169, 169), not
     monotone in its increments, so no estimate;
   - case 6: F = 1 and F = 2 agree in Pcr within 0.8 % at ny = 160, 4518 N
     at F = 2, 0.5 % from its estimate; b_min_t -0.193 and -0.218 at
     ny = 120 and 160 (F = 2), order about 2, 12 % from its estimate -0.248.
   **At ny = 160 (5.3 elements per wave of n_c = 30), Pcr is converged
   within 0.7 %, b is not**: 2.4-3.2 % for case 0, about 12 % for case 6,
   unknown for case 1. SS3-IR is being extended to ny = 200 and 240 at
   F = 2, and to F = 3 at ny = 120 and 160 (12 runs, submitted 2026-09-25).
5. **free-IR does not converge to a buckling load of the shell**, and was
   removed from the models for that reason. Every run
   buckles at n = 2, the ovalization of the free edges, and Pcr falls with
   every refinement: case 0 from 905 N at ny = 24 to 95 N at ny = 160 (F =
   2), Richardson order about 2 towards roughly 80 N; case 1 to 564 N; case
   6 to 162 N, still 44 % above its estimate. At ny = 160 that is 2.5 %
   (case 0), 6.3 % (case 1) and 3.6 % (case 6) of the SS3-IR load. Its b_min_t is essentially zero, apart from values of -2 to -950
   that come from near-singular solves. **Supporting the edges radially is
   what makes these cylinders a well-posed buckling problem**; free-IR only
   measures ovalization of the free edges.
6. The combined fit f = f_inf + a/ny^p + c/nx^q of `convergence_post.py` is
   not reliable yet: over the 6 meshes of case 0 with ny/n_c >= 4 it has as
   many data as parameters; for case 1 the nx = 67 meshes, which miss the
   critical mode, spoil it; for case 6 the F = 0.5 meshes do. It is repeated
   in the Extension over F >= 2, the default of `convergence_post.py` since.

**SS3-IR, case 0**

| F | ny | nx | dx max (mm) | n_c | ny/n_c | Pcr (N) | m | b_min_t | crit b_t |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 24 | 51 | 99.2 | 8 | 3.0 | 7543.9 | 10 | -0.0305 | -0.0168 |
| 0.5 | 32 | 61 | 49.6 | 16 | 2.0 | 5701.3 | 8 | -0.0009 | -0.0008 |
| 0.5 | 40 | 61 | 49.6 | 20 | 2.0 | 5166.0 | 8 | -0.0252 | -0.0371 |
| 0.5 | 48 | 71 | 33.1 | 16 | 3.0 | 4491.8 | 10 | -0.1587 | -0.1437 |
| 0.5 | 64 | 81 | 24.8 | 26 | 2.5 | 4306.2 | 10 | -0.0587 | -0.0846 |
| 0.5 | 80 | 91 | 19.8 | 27 | 3.0 | 4130.1 | 10 | -0.1040 | -0.1607 |
| 0.5 | 96 | 111 | 14.2 | 29 | 3.3 | 3995.1 | 12 | -0.1194 | -0.1650 |
| 0.5 | 120 | 121 | 12.4 | 30 | 4.0 | 3919.0 | 12 | -0.1452 | -0.2006 |
| 0.5 | 160 | 161 | 8.3 | 30 | 5.3 | 3876.9 | 10 | -0.1570 | -0.2276 |
| 1 | 24 | 51 | 99.2 | 8 | 3.0 | 7543.9 | 10 | -0.0305 | -0.0168 |
| 1 | 32 | 61 | 49.6 | 16 | 2.0 | 5701.3 | 8 | -0.0009 | -0.0008 |
| 1 | 40 | 61 | 49.6 | 20 | 2.0 | 5166.0 | 8 | -0.0252 | -0.0371 |
| 1 | 48 | 71 | 33.1 | 16 | 3.0 | 4491.8 | 10 | -0.1587 | -0.1437 |
| 1 | 64 | 81 | 24.8 | 26 | 2.5 | 4306.2 | 10 | -0.0587 | -0.0846 |
| 1 | 80 | 91 | 19.8 | 27 | 3.0 | 4130.1 | 10 | -0.1040 | -0.1607 |
| 1 | 96 | 111 | 14.2 | 29 | 3.3 | 3995.1 | 12 | -0.1194 | -0.1650 |
| 1 | 120 | 121 | 12.4 | 30 | 4.0 | 3919.0 | 12 | -0.1452 | -0.2006 |
| 1 | 160 | 161 | 8.3 | 30 | 5.3 | 3876.9 | 10 | -0.1570 | -0.2276 |
| 2 | 24 | 61 | 49.6 | 10 | 2.4 | 6462.7 | 8 | 0.0027 | 0.0089 |
| 2 | 32 | 71 | 33.1 | 14 | 2.3 | 5193.6 | 8 | 0.0075 | 0.0075 |
| 2 | 40 | 81 | 24.8 | 20 | 2.0 | 4803.7 | 8 | -0.0326 | -0.0477 |
| 2 | 48 | 91 | 19.8 | 24 | 2.0 | 4581.4 | 8 | -0.0384 | -0.0560 |
| 2 | 64 | 111 | 14.2 | 24 | 2.7 | 4295.3 | 10 | -0.0628 | -0.0904 |
| 2 | 80 | 121 | 12.4 | 27 | 3.0 | 4092.3 | 10 | -0.0893 | -0.1279 |
| 2 | 96 | 141 | 9.9 | 29 | 3.3 | 3987.1 | 12 | -0.1188 | -0.1640 |
| 2 | 120 | 171 | 7.6 | 30 | 4.0 | 3915.0 | 12 | -0.1462 | -0.2021 |
| 2 | 160 | 221 | 5.5 | 30 | 5.3 | 3875.6 | 10 | -0.1572 | -0.2279 |

Richardson in ny over the last three meshes of each F:

| F | meshes | Pcr: order, extrapolated, error at the finest | b_min_t: same | crit b_t: same |
|---|---|---|---|---|
| 0.5 | 96-120-160 | p 3.4, 3851.1, +0.7 % | p 4.1, -0.16208, +3.2 % | p 2.1, -0.25997, +12.5 % |
| 1 | 96-120-160 | p 3.4, 3851.1, +0.7 % | p 4.1, -0.16208, +3.2 % | p 2.1, -0.25997, +12.5 % |
| 2 | 96-120-160 | p 3.4, 3852.2, +0.6 % | p 4.7, -0.16101, +2.4 % | p 2.5, -0.25197, +9.5 % |

**SS3-IR, case 1**

| F | ny | nx | dx max (mm) | n_c | ny/n_c | Pcr (N) | m | b_min_t | crit b_t |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 24 | 39 | 76.6 | 5 | 4.8 | 15580.6 | 10 | 0.0296 | 0.0575 |
| 0.5 | 32 | 43 | 51.1 | 11 | 2.9 | 14755.8 | 10 | -0.6148 | -1.0226 |
| 0.5 | 40 | 47 | 38.3 | 13 | 3.1 | 14150.4 | 10 | -0.1915 | -0.2366 |
| 0.5 | 48 | 51 | 30.7 | 15 | 3.2 | 13934.6 | 10 | -0.1212 | -0.2453 |
| 0.5 | 64 | 59 | 23.4 | 21 | 3.0 | 13284.8 | 12 | -0.1404 | -0.1903 |
| 0.5 | 80 | 67 | 23.4 | 22 | 3.6 | 12954.4 | 12 | -0.2060 | -0.2850 |
| 0.5 | 96 | 67 | 23.4 | 23 | 4.2 | 12839.9 | 12 | -0.2353 | -0.3255 |
| 0.5 | 120 | 67 | 23.4 | 23 | 5.2 | 12794.2 | 12 | -0.2597 | -0.3599 |
| 0.5 | 160 | 67 | 23.4 | 23 | 7.0 | 12779.3 | 12 | -0.2703 | -0.3744 |
| 1 | 24 | 39 | 76.6 | 5 | 4.8 | 15580.6 | 10 | 0.0296 | 0.0575 |
| 1 | 32 | 43 | 51.1 | 11 | 2.9 | 14755.8 | 10 | -0.6148 | -1.0226 |
| 1 | 40 | 47 | 38.3 | 13 | 3.1 | 14150.4 | 10 | -0.1915 | -0.2366 |
| 1 | 48 | 51 | 30.7 | 15 | 3.2 | 13934.6 | 10 | -0.1212 | -0.2453 |
| 1 | 64 | 59 | 23.4 | 21 | 3.0 | 13284.8 | 12 | -0.1404 | -0.1903 |
| 1 | 80 | 67 | 23.4 | 22 | 3.6 | 12954.4 | 12 | -0.2060 | -0.2850 |
| 1 | 96 | 67 | 23.4 | 23 | 4.2 | 12839.9 | 12 | -0.2353 | -0.3255 |
| 1 | 120 | 127 | 11.7 | 24 | 5.0 | 8984.5 | 12 | -0.1692 | -0.2341 |
| 1 | 160 | 127 | 11.7 | 24 | 6.7 | 8956.6 | 12 | -0.1819 | -0.2521 |
| 2 | 24 | 43 | 51.1 | 5 | 4.8 | 15493.6 | 10 | -0.9671 | -0.9671 |
| 2 | 32 | 51 | 30.7 | 11 | 2.9 | 14782.4 | 10 | -0.7230 | -1.2091 |
| 2 | 40 | 55 | 25.5 | 13 | 3.1 | 14145.2 | 10 | -0.2107 | -0.2597 |
| 2 | 48 | 67 | 23.4 | 15 | 3.2 | 13927.1 | 10 | -0.1417 | -0.2893 |
| 2 | 64 | 127 | 11.7 | 21 | 3.0 | 9374.9 | 12 | -0.1277 | -0.1504 |
| 2 | 80 | 127 | 11.7 | 23 | 3.5 | 9145.0 | 12 | -0.1264 | -0.1751 |
| 2 | 96 | 127 | 11.7 | 23 | 4.2 | 9043.1 | 12 | -0.1576 | -0.2182 |
| 2 | 120 | 169 | 7.8 | 24 | 5.0 | 8991.3 | 12 | -0.1676 | -0.2319 |
| 2 | 160 | 169 | 7.8 | 24 | 6.7 | 8953.6 | 12 | -0.1855 | -0.2571 |

Richardson in ny over the last three meshes of each F:

| F | meshes | Pcr: order, extrapolated, error at the finest | b_min_t: same | crit b_t: same |
|---|---|---|---|---|
| 0.5 | 96-120-160 | p 5.5, 12776, +0.0 % | p 4.4, -0.2745, +1.5 % | p 4.5, -0.37989, +1.4 % |
| 1 | 96-120-160 | p 22.1, 8956.5, +0.0 % | not monotone | not monotone |
| 2 | 96-120-160 | p 2.3, 8912.5, +0.5 % | not monotone | not monotone |

**SS3-IR, case 6**

| F | ny | nx | dx max (mm) | n_c | ny/n_c | Pcr (N) | m | b_min_t | crit b_t |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 24 | 17 | 75.0 | 7 | 3.4 | 13627.4 | 10 | -0.0095 | -0.0140 |
| 0.5 | 32 | 17 | 75.0 | 16 | 2.0 | 10823.2 | 8 | -0.0547 | -0.0719 |
| 0.5 | 40 | 17 | 75.0 | 20 | 2.0 | 8859.9 | 7 | -0.0960 | -0.1163 |
| 0.5 | 48 | 17 | 75.0 | 24 | 2.0 | 7810.2 | 7 | -0.1410 | -0.1898 |
| 0.5 | 64 | 17 | 75.0 | 28 | 2.3 | 6837.2 | 10 | -0.1127 | -0.1541 |
| 0.5 | 80 | 25 | 50.0 | 29 | 2.8 | 5755.6 | 10 | -0.2842 | -0.3139 |
| 0.5 | 96 | 25 | 50.0 | 30 | 3.2 | 5621.7 | 10 | -0.4336 | -0.4572 |
| 0.5 | 120 | 33 | 37.5 | 30 | 4.0 | 5157.3 | 10 | -0.5227 | -0.6481 |
| 0.5 | 160 | 41 | 30.0 | 31 | 5.2 | 4889.5 | 10 | -0.6276 | -0.7490 |
| 1 | 24 | 17 | 75.0 | 7 | 3.4 | 13627.4 | 10 | -0.0095 | -0.0140 |
| 1 | 32 | 17 | 75.0 | 16 | 2.0 | 10823.2 | 8 | -0.0547 | -0.0719 |
| 1 | 40 | 25 | 50.0 | 20 | 2.0 | 8176.1 | 7 | -0.0827 | -0.1333 |
| 1 | 48 | 25 | 50.0 | 23 | 2.1 | 7291.3 | 9 | -0.1597 | -0.1404 |
| 1 | 64 | 33 | 37.5 | 26 | 2.5 | 5779.9 | 10 | -0.2388 | -0.2887 |
| 1 | 80 | 41 | 30.0 | 27 | 3.0 | 5166.5 | 10 | -0.4169 | -0.5726 |
| 1 | 96 | 49 | 25.0 | 29 | 3.3 | 4953.3 | 10 | -0.4590 | -0.5508 |
| 1 | 120 | 65 | 18.8 | 29 | 4.1 | 4574.2 | 12 | -0.1732 | -0.2390 |
| 1 | 160 | 81 | 15.0 | 30 | 5.3 | 4552.7 | 12 | -0.2198 | -0.3030 |
| 2 | 24 | 25 | 50.0 | 7 | 3.4 | 9875.8 | 10 | -0.0185 | -0.0222 |
| 2 | 32 | 33 | 37.5 | 14 | 2.3 | 7508.7 | 8 | -0.0098 | 0.0028 |
| 2 | 40 | 41 | 30.0 | 16 | 2.5 | 6181.2 | 10 | -0.0214 | -0.0333 |
| 2 | 48 | 49 | 25.0 | 18 | 2.7 | 5770.1 | 10 | -0.0796 | -0.1058 |
| 2 | 64 | 65 | 18.8 | 24 | 2.7 | 4907.5 | 12 | -0.0650 | -0.0892 |
| 2 | 80 | 81 | 15.0 | 27 | 3.0 | 4736.0 | 10 | -0.1597 | -0.1799 |
| 2 | 96 | 97 | 12.5 | 28 | 3.4 | 4633.0 | 12 | -0.1594 | -0.2202 |
| 2 | 120 | 121 | 10.0 | 29 | 4.1 | 4556.8 | 12 | -0.1929 | -0.2660 |
| 2 | 160 | 161 | 7.5 | 30 | 5.3 | 4517.5 | 12 | -0.2182 | -0.3012 |

Richardson in ny over the last three meshes of each F:

| F | meshes | Pcr: order, extrapolated, error at the finest | b_min_t: same | crit b_t: same |
|---|---|---|---|---|
| 0.5 | 96-120-160 | p 3.2, 4712.8, +3.7 % | p 0.4, -1.5966, +60.7 % | p 3.6, -0.80544, +7.0 % |
| 1 | 96-120-160 | p 13.0, 4552.2, +0.0 % | not monotone | not monotone |
| 2 | 96-120-160 | p 3.7, 4496.5, +0.5 % | p 2.1, -0.24816, +12.1 % | p 2.0, -0.3452, +12.7 % |

### Extension: SS3-IR at ny = 200 and 240, and F = 3 (2026-09-25)

12 more SS3-IR runs, all finished without error: ny = 200 and 240 at F = 2,
and F = 3 at ny = 120 and 160. Fine meshes at F >= 2 (time and peak memory
on one core):

| case | F | ny | nx | dx max (mm) | n_c | ny/n_c | m | Pcr (N) | b_min_t | crit b_t | time (h) | memory (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2 | 120 | 171 | 7.6 | 30 | 4.0 | 12 | 3915.0 | -0.1462 | -0.2021 | 0.77 | 7.6 |
| 0 | 2 | 160 | 221 | 5.5 | 30 | 5.3 | 10 | 3875.6 | -0.1572 | -0.2279 | 1.29 | 12.6 |
| 0 | 2 | 200 | 321 | 5.2 | 31 | 6.5 | 12 | 3862.5 | -0.1732 | -0.2394 | 2.73 | 23.3 |
| 0 | 2 | 240 | 321 | 5.2 | 31 | 7.7 | 12 | 3857.1 | -0.1762 | -0.2435 | 3.24 | 28.1 |
| 0 | 3 | 120 | 221 | 5.5 | 30 | 4.0 | 12 | 3913.8 | -0.1465 | -0.2026 | 0.80 | 9.9 |
| 0 | 3 | 160 | 321 | 5.2 | 30 | 5.3 | 10 | 3875.4 | -0.1576 | -0.2284 | 1.47 | 17.9 |
| 1 | 2 | 120 | 169 | 7.8 | 24 | 5.0 | 12 | 8991.3 | -0.1676 | -0.2319 | 0.44 | 7.5 |
| 1 | 2 | 160 | 169 | 7.8 | 24 | 6.7 | 12 | 8953.6 | -0.1855 | -0.2571 | 1.20 | 10.1 |
| 1 | 2 | 200 | 229 | 5.8 | 24 | 8.3 | 12 | 8914.0 | -0.1903 | -0.2636 | 1.75 | 16.5 |
| 1 | 2 | 240 | 275 | 4.7 | 24 | 10.0 | 12 | 8902.8 | -0.1920 | -0.2661 | 2.39 | 24.0 |
| 1 | 3 | 120 | 229 | 5.8 | 24 | 5.0 | 12 | 8966.5 | -0.1653 | -0.2290 | 1.44 | 10.2 |
| 1 | 3 | 160 | 275 | 4.7 | 24 | 6.7 | 12 | 8920.8 | -0.1827 | -0.2528 | 2.34 | 15.8 |
| 6 | 2 | 120 | 121 | 10.0 | 29 | 4.1 | 12 | 4556.8 | -0.1929 | -0.2660 | 0.30 | 5.3 |
| 6 | 2 | 160 | 161 | 7.5 | 30 | 5.3 | 12 | 4517.5 | -0.2182 | -0.3012 | 0.70 | 9.6 |
| 6 | 2 | 200 | 193 | 6.3 | 30 | 6.7 | 14 | 4505.5 | -0.2002 | -0.3151 | 2.62 | 15.2 |
| 6 | 2 | 240 | 233 | 5.2 | 30 | 8.0 | 12 | 4500.3 | -0.2326 | -0.3207 | 2.87 | 20.4 |
| 6 | 3 | 120 | 177 | 6.8 | 29 | 4.1 | 12 | 4552.0 | -0.1942 | -0.2675 | 0.93 | 7.9 |
| 6 | 3 | 160 | 233 | 5.2 | 30 | 5.3 | 12 | 4516.1 | -0.2185 | -0.3013 | 2.34 | 13.9 |

Error of each mesh at F = 2 against the Richardson estimate from ny = 160, 200 and 240 at F = 2:

| case | quantity | estimate (order) | ny = 120 | ny = 160 | ny = 200 | ny = 240 |
|---|---|---|---|---|---|---|
| 0 | Pcr | 3850.8 (p 3.4) | +1.7 % | +0.6 % | +0.3 % | +0.2 % |
| 0 | b_min_t | -0.17741 (p 7.0) | +17.6 % | +11.4 % | +2.4 % | +0.7 % |
| 0 | crit b_t | -0.2475 (p 3.9) | +18.3 % | +7.9 % | +3.3 % | +1.6 % |
| 1 | Pcr | 8895.5 (p 5.1) | +1.1 % | +0.7 % | +0.2 % | +0.1 % |
| 1 | b_min_t | -0.19349 (p 4.1) | +13.4 % | +4.1 % | +1.7 % | +0.8 % |
| 1 | crit b_t | -0.26868 (p 3.7) | +13.7 % | +4.3 % | +1.9 % | +1.0 % |
| 6 | Pcr | 4493.7 (p 3.1) | +1.4 % | +0.5 % | +0.3 % | +0.1 % |
| 6 | b_min_t | not monotone | -0.19292 | -0.21817 | -0.20016 | -0.23264 |
| 6 | crit b_t | -0.32711 (p 3.4) | +18.7 % | +7.9 % | +3.7 % | +2.0 % |

A positive error is a b less negative than the estimate: the coarser meshes
**underestimate** the imperfection sensitivity, the unconservative side.

**Findings**

1. **The axial mesh is converged at F = 2 for cases 0 and 6.** F = 3 against
   F = 2 at ny = 160 changes Pcr by 0.005 % (case 0) and 0.03 % (case 6), and
   b_min_t by 0.25 % and 0.1 %. Case 1 is more sensitive: F = 3 moves Pcr by
   0.4 % and b_min_t by 1.5 % at ny = 160 (nx 169 to 275), and at ny = 240,
   F = 2 already gives nx = 275.
2. **Pcr** converges at order 3-5 in ny and is within 0.7 % at ny = 160 and
   0.2 % at ny = 240 for all three designs.
3. **b of the critical cluster** converges at order 3.4-3.9 and is the most
   regular quantity: 7.9 %, 4.3 % and 7.9 % from its estimate at ny = 160,
   3.3 %, 1.9 % and 3.7 % at ny = 200, and 1.6 %, 1.0 % and 2.0 % at ny = 240
   (cases 0, 1 and 6).
4. **b_min_t of the full set** follows it for cases 0 and 1 (0.7 % and 0.8 %
   at ny = 240) but not for case 6, whose Koiter set changes from 12 to 14
   modes at ny = 200 (-0.218, -0.200, -0.233 at ny = 160, 200, 240): the set
   still depends on the mesh where near-equal multipliers of other wave
   numbers enter or leave it.
5. **Elements per wave.** ny = 240 is 7.7-10 elements per wave of n_c (30,
   24 and 30), and gives the critical cluster within 1-2 %; ny = 200 (6.5-8.3)
   within 2-4 %; ny = 160 (5.3-6.7) within 4-8 %. The rule of about 8
   elements per wave of n_c, found in the removed studies with the anchored
   SS3 edges, holds.
6. The combined fit f = f_inf + a/ny^p + c/nx^q over F >= 2 reproduces the
   Richardson estimates of Pcr (within 0.1 %), but not those of b: with 6 to
   8 meshes and five parameters it runs to the ends of its exponent grid
   (p or q = 0.5) for b_min_t. The Richardson estimates in ny at F = 2 are
   the reference.

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

Open. From the restarted convergence study (Sections *Results of the first
162 runs* and *Extension*):

- **Edge condition: SS3-IR**, the only one of the models since: free-IR
  does not converge to a buckling load of the shell (free-edge ovalization
  at 2.5-6.3 % of the SS3-IR load, still falling with the mesh), and SS3-IR
  gives the SS3 results.
- **Axial mesh: F = 2**, dx max = dy/2. Elements longer axially than around
  are worse (case 6 at F = 0.5), and F = 3 changes b by 0.1-1.5 % only.
- **Quantity:** b of the critical cluster converges regularly; b_min_t of
  the full set carries, on top of it, the changes of the Koiter set from mesh
  to mesh (case 6).
- **ny**, one of, for the error of b of the critical cluster on designs A, B
  and C and per run on one core:
  1. ny = 160: 4-8 %, Pcr 0.7 %, 0.7-1.3 h and 10-13 GB;
  2. ny = 200: 2-4 %, Pcr 0.3 %, 1.8-2.7 h and 15-23 GB;
  3. ny = 240: 1-2 %, Pcr 0.2 %, 2.4-3.2 h and 20-28 GB, about 8 elements
     per wave of n_c.
  The error is on the unconservative side (b less negative) and grows for
  designs with a larger n_c than 30.

The options of the previous version of this section, which rested on the
removed studies, are in git history.

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
    output; the edge condition of the models, SS3-IR, as `edges` in the
    RESULT line.
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
