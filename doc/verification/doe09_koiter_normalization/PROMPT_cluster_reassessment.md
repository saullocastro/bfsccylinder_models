# DOE09: settle what makes b_min_t of the NL cases not converge, on the cluster

## Context

- **Library.** Branch `doe09-koiter-normalization` of `bfsccylinder_models`, checked out at `/home/saullogiovanip/bfsccylinder_models`. The jobs put it on `PYTHONPATH`. It is not pip-installed; check the import with `python -c "import bfsccylinder_models; print(bfsccylinder_models.__file__)"` under the same `PYTHONPATH`.
- **Python.** `/home/saullogiovanip/miniconda3/bin/python3`, the one in `generate_qsubs_convergence.py`. It has `pypardiso`, which `run_case.py` patches through `use_safe_solvers`.
- **DOE09 working directory.** The directory with `DOE09.txt`, `run_case.py` and the outputs `DOE09_conv_*_k5g.out` of the last convergence study. Find it under `/home/saullogiovanip` if I have not given it. All jobs run from there.
- **Driver.** The copies in `doc/verification/doe09_koiter_normalization/scripts/` of this branch are the current ones.
  - Before changing anything, `diff` each of them against the DOE09 working directory.
  - If any differs, stop and show me the differences. Do not overwrite either side.
  - Afterwards, make every change in `scripts/` of the repository and copy the changed files into the DOE09 working directory.

**Read first:** `doc/verification/doe09_koiter_normalization/REPORT.md`, the section "Reassessment: why b_min_t of the NL cases does not converge", and option 8 of the Decision. It was written from the stored `_k5g` results, without new runs. In short:

1. **b_min depends on the number of wave-number clusters in the Koiter set.** A cluster is the 4 modes of one wave number n. b_min over one cluster is b1, over two about 1.5 b1, over three about 2.1 b1.
2. **Which clusters are in the set changes with the mesh.** Their buckling loads are 0.04–0.4 % apart, less than each one moves between meshes.
3. **The b_min of a single cluster converges.** At ny = 160 it is within 2.5 % (case 1 NL), 3 % (6 NL) and 10–12 % (0 NL) of the extrapolated value.
4. **The ny refinement is not uniform.** The axial mesh of case 1 is the same (nx = 127) at ny = 120, 160 and 200. The edge elements of case 0 stay 5.2 mm long from ny = 80 to 240.
5. **The expansion point is not measured.** lambda_b/lambda_c ranges from 0.9951 to 0.9983 across runs, and its effect on b is unknown.

The goal is to decide which quantity the DOE should report, one that is the same at every mesh, and then which mesh converges it.

## Constraints

- **Library.**
  - Do not change the library (`bfsccylinder_models/`).
  - Every change goes into the driver scripts and new check scripts.
  - `NLprebuck_eps1` is already an argument of `fkoiter_cylinder_CTS_circum`.
- **Defaults.** With no new option, `run_case.py` must give the same RESULT as today. `generate_qsubs.py` asserts `run_case.koiter_num_distinct == 5`, so the command line options must not change the module globals when `run_case` is imported, only when it runs as `__main__`.
- **Do not execute `generate_qsubs.py`.** It has `submit = True` and would submit the whole DOE.
- **Submitting jobs.** The new job generator starts with `submit = False`.
  - First show me the list of runs, their walltime, memory and an estimate of the core-hours.
  - Submit only after I confirm.
- **Git.**
  - Commit on `doe09-koiter-normalization` and push to it.
  - **No `Co-Authored-By` line, nor any other co-author or tool attribution, in the commit messages.**
  - Keep the style of the existing commit messages: a `DOC:`/`ENH:` prefix, a short title, and a body that says what was measured.
- **Style.** Match `run_case.py`: `#NOTE` comments that say why, and the same naming and docstring style.

## 1. Driver options (run_case.py)

Keep `python run_case.py ICASE LIN|NL [NY]` working. Add optional flags, parsed only under `__main__`, and record each of them in the RESULT line:

- **`--distinct K`:** `koiter_num_distinct`, default 5.
- **`--num-eigvals N`:** default 16. It must leave room for K distinct modes, their partners, the completion of the last group, and one more distinct mode, which `distinct_first` needs to know where the group ends. Use about `2*K + 8`: 22 for K = 7 and 26 for K = 9.
  - **ARPACK pitfall.** `eigsh` with k = 20 has stopped with ARPACK error -8 on DOE09 meshes before.
  - In the `eigsh` of `use_safe_solvers`, pass a larger `ncv`, for example `min(n, max(2*k + 1, k + 32))`. Retry once with a larger one if ARPACK fails.
  - Record `ncv` and whether a retry happened.
- **`--axial-factor F`:** default 1. The largest axial element length becomes dy/F instead of dy.
  - In `choose_nxt`, and likewise in `estimate_nx`, replace dy by dy/F in the formula for `nxt` and in the plateau test `plateau_dx(c, ...) > dy`.
  - Do not change the comparison of `fkoiter_cylinder_CTS_circum` itself, which uses the true dy.
  - Check with `checks/mesh_axial_resolution.py`, extended with the factor, that dx max ≤ dy/F for cases 0, 1 and 6 at ny = 120, 160 and 240, and that F = 1 reproduces the present nx exactly.
- **`--eps1 E`:** passed as `NLprebuck_eps1`, default 0.005. Record the lambda_b/lambda_c actually reached; it is `lambda_b/(Pcr/(Nxxunit*circ))`.
- **Subsets in the RESULT line.**
  - Group the Koiter modes by wave number (`mode_harmonics`). For each group of subsets below, record `ns`, the number of modes, whether every cluster in it is complete, `b_min_energy`, `crest_e` (the largest over the 8 rotations, as for the whole set, with `orbit_crest`) and `b_min_t`:
    - every prefix of the clusters, in the order of the Koiter set (increasing multiplier);
    - every window n_c, n_c ± 1, n_c ± 2, ... of wave numbers about the critical one n_c, for as long as all of its clusters are in the set.
  - b_min_energy of a subset is the minimum over the sub-block of the energy-normalized b_ijkl, as in `checks/cluster_subsets.py`.
  - The combined mode of the subset is sum e_k s_k u_k over its modes. `pair_rotation` works unchanged on it, since the planes of the other modes carry no component.
  - A subset taken from a larger run is not the same as a run on that subset alone: the orthogonality conditions of the second-order fields include every Koiter mode. Study (a) below measures that difference.

**Smoke test.** Run one short job before any study, with case 6 NL, ny = 60 and `--distinct 7`, and a second with `--axial-factor 2`. Use qsub, not the login node. Check:
- the RESULT has the new fields;
- no `selection` error;
- no PARDISO fallback;
- rerunning with no option gives `b_min_t` identical to an unchanged `run_case.py` on the same case.

## 2. Studies

Write `generate_qsubs_reassess.py`, modelled on `generate_qsubs_convergence.py`: one job per run, `python -u`, the same `PYTHONPATH` and thread settings, and `submit = False`.
- **Output names:** `DOE09_reassess_<study>_<icase:05d>_ny<ny:03d>_<LIN|NL>_<tag>.out`, so they cannot be mistaken for DOE or `_k5g` outputs.
- **Resuming:** skip a run whose output already has a RESULT with the options requested.
- **Walltime and memory:** the `_k5g` runs took up to 70 min and 14 GB (case 0 NL, ny = 200, 12 modes).
  - With m Koiter modes, the bordered solves grow as m(m+1)/2: 210 for m = 20, against 78 for m = 12.
  - The (N, m, m) arrays phi3, phi30 and cst take 3·N·m²·8 bytes, about 2.5 GB for case 0 at ny = 160 with m = 20.
  - Use 12 h and 24 GB for ny ≤ 160 and K = 9, and more for ny = 240. Adjust from the smoke test.

| study | runs | what it answers |
|---|---|---|
| (a) truncation | cases 0, 1, 6, NL, ny = 120 and 160, `--distinct` 5, 7, 9 (18 runs); add 0 LIN and 6 LIN at ny = 160 with 9 (2 runs) | whether b_min_energy and b_min_t settle as clusters are added, over the full set and over the prefix and window subsets |
| (b) axial refinement | cases 0, 1, 6, LIN and NL, ny = 160, `--axial-factor` 1.5 and 2 (12 runs); factor 1 is the existing `_k5g` run | the axial part of the discretization error, per cluster, above all for case 1 |
| (c) expansion point | cases 1 and 6, NL, ny = 120, `--eps1` 0.002, 0.001, 0.0005 (6 runs); 0.005 is the existing `_k5g` run | the sensitivity of the per-cluster and full-set b to lambda_b/lambda_c, and its extrapolation to 1 |
| (d) ny = 240 | cases 0, 1, 6, LIN and NL, present setup (6 runs) | per-cluster Richardson from 160, 200 and 240; case 1 changes nx at 240, so read it together with (b) |

If one of the smoke tests or runs shows that `--distinct 9` hits a selection error, raise `--num-eigvals` for that run, and say so. Do not shrink the set.

## 3. Post-processing

Write `checks/reassessment_post.py`, which reads the outputs and prints the tables below. Also archive the RESULT lines as `results/DOE09_reassess_<study>.jsonl.gz`, in the format of the existing files: one `{"file": ..., "result": ...}` per run, with `result` null for a run that wrote none.

- **(a)**
  - b_min_energy and b_min_t against the number of clusters, per case and ny, for the full set of each K and for the prefix and window subsets of the K = 9 run.
  - The difference between a subset of the K = 9 run and the full set of the run with that many clusters.
  - Where e_min puts its weight.
- **(b)** b_min of each complete cluster (fixed n) and of the full set against the axial factor, with nx, dx max and Pcr.
- **(c)** The same quantities against lambda_b/lambda_c, with a linear fit and its value at lambda_b/lambda_c = 1.
- **(d)**
  - Per-cluster b_min at ny = 120, 160, 200 and 240, with Richardson from the last three.
  - `checks/cluster_subsets.py` does the three-mesh version; generalize it rather than duplicating it.
  - Note which of the meshes changed nx.

## 4. Report and decision

Add a section to `REPORT.md` after the Reassessment, with the tables of Section 3, and answer, with numbers:

1. **Truncation.** Does b_min_t, of the full set or of a window about n_c, settle as clusters are added?
   - Say from how many clusters on, and to what tolerance. For example, less than 2 % from 7 to 9 distinct modes.
   - If it does not settle, say that the multi-mode b_min is not a usable DOE quantity for these designs.
2. **Mesh, per cluster.** What are the axial and the circumferential parts of the per-cluster error at ny = 160?
   - Which ny, and which axial factor, bring the critical cluster within 2–3 %?
   - What does that cost for the 20,000 runs, with `koiter_time_per_element` rescaled for the number of modes?
3. **Expansion point.** What is the sensitivity to lambda_b/lambda_c? If it exceeds 1 % at the default `NLprebuck_eps1`, recommend a value.
4. **Recommendation.**
   - Recommend the quantity for the DOE:
     - the b_min of the critical cluster;
     - b_min_t over a window;
     - or something else the data supports.
   - Recommend the mesh (ny and axial factor) and the Koiter set for it.
   - Update the Decision section accordingly.
   - Update the constants of `generate_qsubs.py` that the recommendation changes, such as `koiter_num_distinct` and `koiter_time_per_element`. Edit the file; do not run it.

Then commit, with no co-author line, and push to `doe09-koiter-normalization`. Give me a short summary of the answers and the core-hour estimate.
