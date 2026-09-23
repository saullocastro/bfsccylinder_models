# Production-size check of the vectorized Koiter tensors, on the HPC cluster

## Context

`bfsccylinder_models` computes Koiter post-buckling coefficients of
cylinders. The branch `vectorize-koiter` of
`git@github.com:saullocastro/bfsccylinder_models.git` replaces the Python
element loop that builds the Koiter tensors (repeated in
`koiter_cylinder.py`, `koiter_cylinder_sanders.py`, `koiter_cylinder_CTS.py`
and `koiter_cylinder_CTS_sanders.py`) with one vectorized loop,
`bfsccylinder_models/koiter_tensors.py`, shared by the four models. It also
solves the m(m+1)/2 bordered systems with i <= j instead of all m², since
uij = uji. Version 0.3.2, commit `3251981`, is the baseline.

This was already done on a Windows workstation, one core per run:
- `tests/test_koiter_tensors.py` checks the new element loop against the old
  one to 5e-16. The suite passes: 44 tests, against 34 on the baseline.
- On DOE09 case 6 at ny=40 with 5 Koiter modes, the element loop dropped
  from about 700 s to about 1 s. Most of the Koiter time left is the bordered
  solves of `run_case.py` (PARDISO LU plus GMRES).
- b_ijkl matches the baseline to 1e-13 on the DOE cases and to 9e-11 on
  the Waters shell. The Waters figure is the round-off floor of the bordered
  solve itself (`doc/verification/bordered_roundoff_floor.py`).

The one thing that could not be measured there is the **production size**:
DOE09 at ny=160, which needs about 10 GB and the cluster's own cores. That is
your task.

The DOE is 10,000 CTS designs, run twice each (`LIN` and `NL`), with
`run_case.py` in the DOE09 directory on the cluster. That script calls
`koiter_cylinder_CTS_sanders.fkoiter_cylinder_CTS_circum` with
`koiter_num_modes=5` and `num_eigvals=12`. The job generator
`generate_qsubs.py` estimates the cost as
`time_per_dof[prebuck]*dof + koiter_time_per_element*num_elements`, with
`koiter_time_per_element = 0.7` s, the old cost. The new value of that constant
is what you are measuring.

## Constraints

- **Do not run `generate_qsubs.py`.** It has `submit = True` and submits about
  2,000 jobs. Read it; never execute it.
- **Do not modify any code of the library or of the DOE09 directory.** If
  something fails, report it with the traceback; do not fix it.
- **Solvers.**
  - The library silently uses `pypardiso` whenever it can import it, and
    pypardiso gives wrong K_C0 solves (Pcr 602 N instead of 4228 N).
  - `run_case.py` needs pypardiso, and replaces the library's solvers itself
    (`use_safe_solvers`), so the DOE runs are fine with it installed.
  - The test suite must NOT see pypardiso. If `python -c "import pypardiso"`
    works in the cluster Python, run pytest with a blocker on the path:
    ```
    mkdir -p /tmp/nopardiso/pypardiso
    echo 'raise ImportError("blocked for the tests")' > /tmp/nopardiso/pypardiso/__init__.py
    PYTHONPATH=/tmp/nopardiso:<checkout> python -m pytest tests/
    ```
- **Environment.**
  - One thread per run: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
  - `bfsccylinder_models` must be imported from the intended checkout, with
    `PYTHONPATH=<checkout>`. Confirm this with
    `python -c "import bfsccylinder_models; print(bfsccylinder_models.__file__)"`.
    Never pip-install the library.
  - Run the heavy work through the batch system (PBS `qsub`, as
    `generate_qsubs.py` does), not on the login node.
- **Git.** Commit only on `vectorize-koiter`. Push only that branch. Do not
  merge, rebase or touch `main`/`master`.

## Steps

1. **Checkouts.**
   ```
   git clone -b vectorize-koiter git@github.com:saullocastro/bfsccylinder_models.git ~/bfsccylinder_models
   cd ~/bfsccylinder_models && git worktree add ../bfsccylinder_models_baseline 3251981
   ```
   If `~/bfsccylinder_models` already exists, fetch and check out the branch
   there instead, after looking at what the directory holds.

2. **Test suite, both checkouts, as a batch job.** Single-threaded, with the
   pypardiso blocker if needed. The baseline is expected to pass all 34
   tests, and the new version all 44. Record any failure with its output.
   Failures that happen on the baseline too are not the refactor's.

3. **Production runs, as batch jobs.** Use
   `doc/verification/benchmark_koiter_vectorization_hpc.sh`, submitted from
   the DOE09 directory. Its header explains the variables. Set `PYTHON`,
   `NEW` and `BASE`, and `EXTRA_PYTHONPATH` if pypardiso lives in a separate
   directory. Submit:
   - `RUN=new_m5` and `RUN=new_m0` for `CASE=0`, `PREBUCK=NL`, `NY=160`.
     This is the required measurement.
   - `RUN=new_m5` and `RUN=new_m0` for `CASE=1` and `CASE=6`, NL, ny=160, to
     check that the Koiter time scales with the number of elements.
   - `RUN=new_m5` for `CASE=0`, `PREBUCK=LIN`, ny=160.
   - Optionally, if the queue allows: `RUN=base_m5` for `CASE=0`, NL,
     ny=160. It takes 5+ hours, and serves to confirm, at production size,
     the speedup and that Pcr, load_mult, a_ijk and b_ijkl agree with the new
     version.

   Before submitting the ny=160 jobs, check that the setup works with one
   quick job: `NY=40`, `CASE=6`, `RUN=new_m5`. It should take a few minutes.

4. **Report.** From the JSON files, write a table with one row per run:
   - case, pre-buckling, ny, number of elements, DOF;
   - `time_total`, `time_koiter`, `time_element_loop` and
     `time_bordered_solves`, and `time_koiter/num_elements`;
   - t(m=5) − t(m=0) per element;
   - `peak_mem_gb`;
   - Pcr, `b_ijkl[0][0][0][0]` and the `b_iiii`.

   If the baseline run exists, add the maximum relative difference of Pcr,
   load_mult, a_ijk and b_ijkl against the new run, each measured against
   the largest |value| of that quantity.

   Then compute the core-hour estimate of the 20,000 DOE runs with the new
   Koiter time per element of case 0, NL, ny=160:
   ```
   PYTHONPATH=~/bfsccylinder_models python ~/bfsccylinder_models/doc/verification/estimate_doe09_core_hours.py --doe-dir <DOE09 dir> <new koiter_time_per_element>
   ```
   It reads `generate_qsubs.py` with `ast` and does not execute it.

   Also say whether the per-element Koiter time of cases 0, 1 and 6 is
   roughly constant. The bordered solves grow faster than the number of
   elements, so a single per-element constant may under- or overestimate
   the large designs. If it varies a lot, fit
   `koiter_time = c1*num_elements + c2*dof**p` and report the fit.

5. **Deliver.**
   - Put the JSON files, the job logs (trim the lines starting with
     `# $b_` if they are huge) and a `REPORT.md` with the tables and the
     estimate under `doc/verification/cluster_results/` in the branch.
   - Commit, then push `vectorize-koiter` only.
   - Print the report in your final answer too.
