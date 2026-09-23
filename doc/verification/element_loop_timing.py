"""Time per element of the Koiter element loop, version 0.3.2 against now

The loop of version 0.3.2 is reference_element_tensors of
tests/test_koiter_tensors.py; the current one is koiter_element_tensors. Both
integrate the same four Sanders elements with random stiffness, pre-buckling
state and modes, NLprebuck on, with m Koiter modes and m + 2 directions of
the null space of phi2, as in the DOE runs (num_cond = 7 for m = 5). Only the
element loop is timed: the bordered solves, which the models add after it,
depend on the mesh and on the solver, see "Vectorization of the Koiter
tensors" in doc/nlprebuck_implementation.tex.
Run it on an otherwise idle machine, one BLAS thread:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python element_loop_timing.py [modes]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, 'tests'))

import numpy as np
from bfsccylinder.quadrature import get_points_weights

from bfsccylinder_models.koiter_tensors import koiter_element_tensors
from bfsccylinder_models import koiter_cylinder_CTS_sanders as model
from test_koiter_tensors import _elements, reference_element_tensors


def median_time(func, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    modes = [int(a) for a in sys.argv[1:]] or [1, 2, 5, 8]
    points, weights = get_points_weights(nint=4)
    rng = np.random.default_rng(0)
    elements, N = _elements(True, rng)
    ne = len(elements)
    u0, u0dot, u0ddot = 1e-4*rng.standard_normal((3, N))
    print('| m | directions | old loop (ms/element) | new loop (ms/element) '
          '| speedup |')
    print('|---|---|---|---|---|')
    for m in modes:
        nc = m + 2
        Ucond = rng.standard_normal((N, nc))
        ucond = {k: Ucond[:, k] for k in range(nc)}
        new = median_time(lambda: koiter_element_tensors(elements, points,
                weights, u0, u0dot, u0ddot, Ucond, m, True,
                model.nonlinear_rows), 21)/ne
        old = median_time(lambda: reference_element_tensors(elements, points,
                weights, u0, u0dot, u0ddot, ucond, m, True, True),
                3 if m < 8 else 1)/ne
        print('| %d | %d | %.1f | %.3f | %.0f |' % (m, nc, 1e3*old, 1e3*new,
                                                  old/new), flush=True)


if __name__ == '__main__':
    main()
