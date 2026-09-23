# Production-size check of the vectorized Koiter tensors, HPC cluster

Cluster `hpc12` (Torque/PBS), queue `asm-small`, one core per run
(`nodes=1:ppn=1`, `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`),
Python 3.12.11, NumPy 2.3.1, SciPy 1.16.0, pypardiso 0.4.7, 2026-09-23.

- new: `vectorize-koiter` at `49b325a`, `~/bfsccylinder_models`
- baseline: version 0.3.2, `3251981`, `~/bfsccylinder_models_baseline`

Each run imported `bfsccylinder_models` from its own checkout through
`PYTHONPATH`, ahead of the 0.3.2 copy pip-installed in site-packages. The
`library` field of every JSON file confirms this.

## Test suite

Batch jobs, one core, with pypardiso blocked by an `ImportError` stub ahead
on `PYTHONPATH`. pytest 9.1.1 came from a `pip install --target` directory,
since the cluster Python has no pytest.

| checkout | result | time |
|---|---|---|
| baseline `3251981` | **34 passed**, 4 warnings | 691 s |
| new `49b325a` | **44 passed**, 4 warnings | 239 s |

Both suites give the same 4 warnings, lobpcg tolerance warnings of
`linbuck_VAFW.py` in `test_linBuck_VAFW.py`, code the refactor does not
touch. The logs are `tests_baseline_3251981.log` and
`tests_vectorize-koiter_49b325a.log`.

## Runs

`doc/verification/benchmark_koiter_vectorization_hpc.sh` was submitted from
the DOE09 directory with `run_case.py` (`koiter_num_modes=5`,
`num_eigvals=12`, `use_safe_solvers`, `use_distinct_modes`). Each run is one
job, `mem=16gb`. Time is wall-clock seconds on one core.

| run | case | pre-buckling | ny | elements | DOF | time_total (s) | time_koiter (s) | time_element_loop (s) | time_bordered_solves (s) | time_koiter/element (ms) | [t(m=5) − t(m=0)]/element (ms) | peak_mem_gb | Pcr (N) | b_0000 | b_iiii |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| new_m5 | 0 | NL | 160 | 25600 | 257600 | 1724 | 429.1 | 33.8 | 390.6 | **16.76** | 20.77 | 8.31 | 3872.22 | −0.23529 | −0.2353, −0.2353, −0.1957, −0.1957, −0.2306 |
| new_m0 | 0 | NL | 160 | 25600 | 257600 | 1192 | 0.5 | – | – | – | – | 6.20 | 3872.22 | – | – |
| new_m5 | 1 | NL | 160 | 20160 | 203200 | 1298 | 308.4 | 29.3 | 277.0 | 15.30 | 14.84 | 6.46 | 8961.55 | −0.27106 | −0.2711, −0.2520, −0.2644, −0.2633, −0.2619 |
| new_m0 | 1 | NL | 160 | 20160 | 203200 | 999 | 0.4 | – | – | – | – | 4.87 | 8961.55 | – | – |
| new_m5 | 6 | NL | 160 | 12800 | 129600 | 1167 | 171.5 | 18.2 | 152.0 | 13.40 | 11.78 | 4.06 | 4558.02 | −0.27504 | −0.2750, −0.2836, −0.2914, −0.2913, −0.2692 |
| new_m0 | 6 | NL | 160 | 12800 | 129600 | 1016 | 0.4 | – | – | – | – | 3.10 | 4558.02 | – | – |
| new_m5 | 0 | LIN | 160 | 25600 | 257600 | 754 | 405.2 | 38.0 | 364.1 | 15.83 | – | 7.24 | 4549.53 | 0.27601 | 0.2760, 0.2957, 0.2558, 0.3160, 0.2353 |
| new_m5 | 6 | NL | 40 | 960 | 10000 | 41 | 9.7 | 1.1 | 8.5 | 10.06 | – | 0.41 | 8190.34 | −0.12016 | −0.1202, −0.0834, −0.1597, −0.0826, −0.1487 |

The last row is the ny=40 setup check.

- **Koiter time per element:** 16.76 ms for case 0, NL, ny=160, against the
  0.7 s in `generate_qsubs.py`, 42× less. The element loop is now 8–11% of
  the Koiter time at ny=160. The rest is almost all the bordered solves
  (PARDISO LU + GMRES).
- **Pcr and b_0000:** both match the single-mode values in
  `DOE09_convergence.txt` at ny=160, case 0. NL: 3872.2235 N, −0.235286.
  LIN: 4549.5270 N, 0.276011.
- **The two time measures:** for case 0 NL, t(m=5) − t(m=0) (20.8 ms/element)
  exceeds the direct measurement (16.8 ms/element). The non-Koiter part of the
  m=5 run took 1295 s, against 1192 s for the m=0 run, although both do the
  same pre-buckling and eigenvalue work. This is a difference between nodes,
  not Koiter cost, so the direct measurement is the one to use.
- **Distinct modes:** `num_distinct` among the 12 eigenvectors was 12, 11, 12
  and 8 for NL cases 0, 1, 6 and LIN case 0, always ≥ 5.
- **Peak memory:** at most 8.3 GB, for case 0 NL m=5, 2.1 GB above the same
  run without Koiter.

### Baseline comparison

Pending: `base_m5`, case 0, NL, ny=160 (job 1032514) is still running (5+ h expected).

## Scaling with the number of elements, NL, ny=160

| case | elements | time_koiter/element (ms) |
|---|---|---|
| 6 | 12800 | 13.40 |
| 1 | 20160 | 15.30 |
| 0 | 25600 | 16.76 |

The per-element time is roughly constant: it grows 25% over a 2× range of
elements, following the bordered solves, which grow faster than linearly. A
power law fits the three points to within 1%:

    time_koiter = 6.56e-4 * num_elements**1.319   (s)

The requested form `c1*num_elements + c2*dof**p` cannot be identified from
three points, since dof ≈ 10·num_elements here. Every p from 1.2 to 2 fits
to within 1%, for instance p=2 with c1 = 9.9e-3 s and c2 = 2.6e-9 s. The
power law is used below.

## Core-hour estimate of the 20,000 DOE09 runs

`estimate_doe09_core_hours.py --doe-dir ~/DOE09` reads `generate_qsubs.py`
with `ast` and does not execute it. It uses ny=160 and 20,000 runs (LIN +
NL), with 19,903 elements per run on average and at most 47,680.

| koiter_time_per_element | pre-buckling + eigen (core-h) | Koiter (core-h) | total (core-h) |
|---|---|---|---|
| 0.7 s (current, old loop) | 13933 | 77402 | 91335 |
| **0.01676 s (new, case 0 NL ny=160)** | 13933 | 1853 | **15786** |
| 0.02078 s (new, t(m=5) − t(m=0), case 0 NL) | 13933 | 2298 | 16231 |
| power law above, per design | 13933 | 1724 | 15657 |

- **Total:** 83% fewer core-hours, from 91,335 to 15,786. The Koiter share
  falls from 85% to 12%.
- **Constant vs power law:** 13% of the designs have more elements than case
  0. The power law gives them up to 20.3 ms/element, at 47,680 elements. The
  smaller designs, the majority, fall below 16.76 ms/element. Over the whole
  DOE, the constant 16.76 ms overestimates the power law by 7%, so it is a
  safe value for `koiter_time_per_element`. For the per-job walltime of the
  largest designs, 0.021 s/element leaves margin.

## Notes on the setup

- git on the cluster is 1.8.3.1, which has no `git worktree`, so the
  baseline is a local clone detached at `3251981`.
- The first ny=40 check (job 1032502) ran against an older `~/DOE09`
  (`num_eigvals=4, koiter_num_modes=1` hardcoded, no `use_distinct_modes`,
  `estimate_nx` or `generate_qsubs.py`). It failed with
  `AttributeError: module 'run_case' has no attribute 'use_distinct_modes'`
  at `benchmark_koiter_vectorization.py:144`. After the user synced the
  workstation's DOE09 scripts, the check (job 1032505) and every run above
  succeeded, without a change to the library or to DOE09.
- `generate_qsubs.py` was read, never executed; no chunk files were written.
