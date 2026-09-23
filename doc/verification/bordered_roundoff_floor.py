"""Round-off floor of the second order fields and of b_ijkl, Waters shell

The bordered system of the second order fields amplifies round-off in its
right-hand side by about 1e8 on this shell, solved with SciPy's SuperLU. Any
change to the order in which the element tensors are summed, such as the
vectorization of koiter_tensors.py, therefore moves uij far more than the
tensors themselves, and b_ijkl by up to about 1e-10 of its largest entry.
This measures that floor: the model is run twice, the second time with the
right-hand side of every bordered solve multiplied by (1 + eps r), r standard
normal and eps = 4e-16 the round-off of the element tensors measured by
tests/test_koiter_tensors.py, and it reports how far uij and b_ijkl move.
The first spsolve of the model is the static solve on the nu unknown DOFs;
the bordered systems are the larger ones.

usage: python bordered_roundoff_floor.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir))

import numpy as np
from composites import laminated_plate
import bfsccylinder_models.koiter_cylinder_sanders as model

ny = 60
m = 2
spsolve = model.spsolve


def run(NLprebuck, eps):
    rng = np.random.default_rng(0)
    nu = []

    def perturbed(A, b):
        if not nu:
            nu.append(b.shape[0])
        elif b.shape[0] > nu[0]:
            b = b*(1 + eps*rng.standard_normal(b.shape[0]))
        return spsolve(A, b)

    model.spsolve = perturbed
    try:
        L = 0.3556
        R = 0.20318603
        laminaprop = (127.629e9, 11.3074e9, 0.300235, 6.00257e9, 6.00257e9,
                      6.00257e9)
        prop = laminated_plate(stack=[45, -45, 0, 90, 90, 0, -45, 45],
                laminaprop=laminaprop, plyt=0.00012692375, offset=0, rho=1611)
        nx = int(ny*L/(2*np.pi*R))
        if nx % 2 == 0:
            nx += 1
        out = model.fkoiter_cyl_SS3(L, R, nx, ny, prop, num_eigvals=12,
                koiter_num_modes=m, NLprebuck=NLprebuck)
    finally:
        model.spsolve = spsolve
    k = out['koiter']
    b = np.array([k['b_ijkl'][idx] for idx in np.ndindex(*(m,)*4)])
    return k['uij'], b


for NLprebuck in [False, True]:
    u0, b0 = run(NLprebuck, 0.)
    u1, b1 = run(NLprebuck, 4e-16)
    scale = max(np.abs(v).max() for v in u0.values())
    du = max(np.abs(u1[key] - u0[key]).max() for key in u0)/scale
    db = np.abs(b1 - b0).max()/np.abs(b0).max()
    print('NLprebuck=%s: rhs perturbed by 4e-16 relative -> uij moves '
          '%.1e, b_ijkl moves %.1e' % (NLprebuck, du, db))
