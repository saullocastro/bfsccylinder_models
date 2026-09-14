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
