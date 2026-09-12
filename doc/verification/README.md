# Verification scripts

Every number quoted in `doc/nlprebuck_implementation.tex` that is not a
literature value comes from one of the scripts in this directory or from the
test suite in `tests/`. They are listed, with the section each one feeds, in
the appendix of that document.

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
| `kinematics_vs_element.py` | the strain and its first derivative assumed by the Koiter expansion match the element to machine precision, for both kinematics; a control with `-Sv/R` removed fails, so the check is sharp | seconds |
| `tangent_consistency.py` | `KC0 + KCNL + KG` is the derivative of `fint`, and equals the Koiter second variation; identifies the stress resultants the element's `KG` actually uses | seconds |
| `cts_vs_constant_stiffness.py` | the CTS models reproduce the `newton_raphson` models in the constant-stiffness limit, `NLprebuck` on and off | ~10 min |
| `cts_mesh_convergence.py` | mesh convergence of `Pcr` and `b_1111` for a steered design; pass `nl` for the non-linear pre-buckling variant | ~12 min / ~35 min |
| `cts_mode_spectrum.py` | circumferential harmonic content of the lowest modes against mesh; pass `nl` for non-linear pre-buckling | ~20 min |
| `reference_mode_cluster.py` | size of the near-critical cluster of the Sun and Arbocz shells, which is what sizes a multi-mode expansion | ~5 min |

## Literature verification cases

These live in `tests/` because they assert rather than report:

| test | shell | reference |
| --- | --- | --- |
| `test_koiter_cylinder_newton_raphson.py` | Sun et al. §3.1 and NASA AW-CYL-1-1 | Sun et al. 2020; Arbocz, Starnes & Nemeth 2001 (ANILISA, STAGS-A) |
| `test_koiter_cylinder_Waters.py`, `_sanders.py` | Waters shell | Arbocz & Starnes 2002 |
| `test_koiter_cylinder_CTS.py`, `_sanders.py` | CTS cylinder, constant-stiffness limit | cross-check against the `newton_raphson` models |
| `test_buckling_mode_cluster.py` | Sun et al. §3.1 | degeneracy and cluster structure |
| `test_linBuck_VAFW.py`, `test_Zhihua_error.py` | VAFW cylinders | linear buckling |
| `test_cts_shares_nlprebuck_algorithm.py` | — | source parity between the CTS and `newton_raphson` models |
