"""Where does the gap between b_1111 and ANILISA come from?

Single-mode b_1111 of NASA AW-CYL-1-1 (Arbocz, Starnes and Nemeth 2001),
von Karman kinematics as in the test suite, against the mesh, the distance of
the expansion point to the bifurcation point, and the member of the
degenerate pair. For each run it reports, besides b_1111 as the model
normalizes it (largest nodal translation equal to h):

- b_env, b_1111 with the mode normalized by its crest amplitude instead, the
  envelope max sqrt(w1**2 + w2**2) of the pair, which does not depend on
  where the crest of a skewed mode falls between nodes
- the lowest multiplier of every circumferential harmonic k, relative to the
  critical one, from the eigenproblem restricted to the invariant subspace of
  harmonic k; the second order field carries the harmonics 0 and 2n, and a
  multiplier of those close to the critical one would amplify it

Reported in: Sections "The modes come in degenerate pairs", "Normalising the
modes" and "The gap to ANILISA".
Runtime: from 2 min (ny=60, nx=25) to 25 min (ny=120, nx=51) per run on an
unloaded machine. Every run is independent, so the configurations are given
on the command line and can be run in parallel:

    python reference_b_convergence.py NY NX [EPS1] [ROT_DEG] [sanders | sun]
                                      [linear]

EPS1 is NLprebuck_eps1, 0.005 by default; ROT_DEG rotates the critical mode
inside its pair; 'sanders' keeps AW-CYL-1-1 and its load but replaces the
von Karman kinematics by the Sanders ones; 'sun' selects the Sun et al. 3.1
shell with Sanders kinematics, as in the test suite, instead. 'linear'
expands about the linear pre-buckling state, and '60 17 sanders linear' is
the Waters shell of tests/test_koiter_cylinder_Waters_sanders.py.
"""

import json
import os
import sys
import time
#NOTE the repository root goes FIRST on sys.path: an installed
#     bfsccylinder_models would otherwise shadow the working tree, silently
#     verifying a different version of the code than the one being edited
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir))

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, csc_matrix
from composites import laminated_plate
import bfsccylinder
import bfsccylinder.sanders
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD

import bfsccylinder_models.koiter_cylinder as vk_model
import bfsccylinder_models.koiter_cylinder_sanders as sa_model
from bfsccylinder_models.cyclic_symmetry import (degenerate_partner,
        mesh_order, rotated)

DOF = 10
E11, E22, G12, nu12 = 127.629e9, 11.3074e9, 6.00257e9, 0.300235
STACK = (45, -45, 0, 90, 90, 0, -45, 45)
H = 0.00101539
L = 0.3556


def operators(out, prop, R, lib, element, nonlinear=True):
    """KC0 + KCNL(u0) and KG(u0), the operators of the eigenproblem; KC0
    alone in place of the first for a linear pre-buckling state"""
    points, weights = get_points_weights(nint=4)
    x = out['x']
    nid_pos = out['nid_pos']
    elements = []
    for k, (n1, n2, n3, n4) in enumerate(zip(out['n1s'], out['n2s'],
                                             out['n3s'], out['n4s'])):
        elem = element(4)
        elem.n1, elem.n2, elem.n3, elem.n4 = n1, n2, n3, n4
        elem.c1 = DOF*nid_pos[n1]
        elem.c2 = DOF*nid_pos[n2]
        elem.c3 = DOF*nid_pos[n3]
        elem.c4 = DOF*nid_pos[n4]
        elem.R = R
        elem.lex = x[nid_pos[n2]] - x[nid_pos[n1]]
        elem.ley = 2*np.pi*R/out['ny']
        assign_constant_ABD(elem, prop)
        elem.init_k_KC0 = k*lib.KC0_SPARSE_SIZE
        elem.init_k_KCNL = k*lib.KCNL_SPARSE_SIZE
        elem.init_k_KG = k*lib.KG_SPARSE_SIZE
        elements.append(elem)
    N = DOF*x.shape[0]
    u0 = out['koiter']['u0']

    def assemble(update, size, *u):
        r = np.zeros(len(elements)*size, dtype=lib.INT)
        c = np.zeros(len(elements)*size, dtype=lib.INT)
        v = np.zeros(len(elements)*size, dtype=lib.DOUBLE)
        for elem in elements:
            update(*u, elem, points, weights, r, c, v)
        return coo_matrix((v, (r, c)), shape=(N, N)).tocsc()

    KC = assemble(lib.update_KC0, lib.KC0_SPARSE_SIZE)
    if nonlinear:
        KC = KC + assemble(lib.update_KCNL, lib.KCNL_SPARSE_SIZE, u0)
    KG = assemble(lib.update_KG, lib.KG_SPARSE_SIZE, u0)
    return KC, KG


def harmonic_multipliers(out, KC, KG, R):
    """Lowest multiplier of each harmonic k = 0, ..., ny/2

    The fields whose every nodal degree of freedom varies as cos(k y/R) and
    sin(k y/R) are invariant under the cyclic shift, which commutes with KC
    and KG. The single node constraint on the axial translation is not
    cyclic, and is applied to the axisymmetric harmonic only, where it
    removes the rigid body translation.
    """
    nx, ny = out['nx'], out['ny']
    order = mesh_order(out['x'], out['y'], nx, ny)
    theta = out['y'][order[0]]/R
    imid = np.argmin(np.abs(out['x'][order[:, 0]] - L/2.))
    res = {}
    for k in range(ny//2 + 1):
        rows, cols, vals = [], [], []
        for i in range(nx):
            for d in range(DOF):
                if i in (0, nx - 1) and d in (3, 6):
                    continue
                if k == 0 and i == imid and d == 0:
                    continue
                for trig in ((np.cos,) if k == 0 else (np.cos, np.sin)):
                    v = trig(k*theta)
                    #NOTE sin(k y/R) vanishes at every node for k = ny/2
                    if np.abs(v).max() < 1e-8:
                        continue
                    rows.append(DOF*order[i] + d)
                    cols.append(np.full(ny, len(rows) - 1))
                    vals.append(v)
        B = csc_matrix((np.concatenate(vals),
                        (np.concatenate(rows), np.concatenate(cols))),
                       shape=(KC.shape[0], len(rows)))
        th = eigh((B.T @ KG @ B).toarray(), (B.T @ KC @ B).toarray(),
                  eigvals_only=True)
        res[k] = float(np.min(-1/th[th < 0]))
    return res


def main(ny, nx, eps1=0.005, rot_deg=0., case='arbocz', sanders=False,
         linear=False):
    if case == 'sun' or sanders:
        model, lib = sa_model, bfsccylinder.sanders
        element = lib.BFSCCylinderSanders
    else:
        model, lib, element = vk_model, bfsccylinder, bfsccylinder.BFSCCylinder
    if case == 'sun':
        R, Nxxunit = 0.2032, 20000.
    else:
        R, Nxxunit = 0.20318603, 10000.
    prop = laminated_plate(stack=STACK,
            laminaprop=(E11, E22, nu12, G12, G12, G12), plyt=H/len(STACK))

    canonical_modes = model.canonical_modes

    def modes(mu, eigvecsu, bu, axi_order, DOF, *args, **kwargs):
        v = canonical_modes(mu, eigvecsu, bu, axi_order, DOF, *args, **kwargs)
        if rot_deg:
            phi = np.zeros(bu.shape[0])
            phi[bu] = v[:, 0]
            psi = degenerate_partner(phi, bu, axi_order, DOF)
            a = np.deg2rad(rot_deg)
            v[:, 0] = (np.cos(a)*phi + np.sin(a)*psi)[bu]
        return v

    model.canonical_modes = modes
    t0 = time.time()
    try:
        out = model.fkoiter_cyl_SS3(L, R, nx, ny, prop, nint=4,
                num_eigvals=2, koiter_num_modes=1, Nxxunit=Nxxunit,
                NLprebuck=not linear, NLprebuck_eps1=eps1,
                NLprebuck_maxiter=30)
    finally:
        model.canonical_modes = canonical_modes
    Ncl = E11*H**2/(R*np.sqrt(3*(1 - nu12**2)))
    mu = out['mu']
    b = out['koiter']['b_ijkl'][(0, 0, 0, 0)]

    order = mesh_order(out['x'], out['y'], nx, ny)
    u1 = out['koiter']['ui'][0]
    U1 = u1.reshape(-1, DOF)[order]
    W = U1[:, :, 6]
    n = int(np.argmax((np.abs(np.fft.rfft(W, axis=1))**2).sum(axis=0)))
    psi = rotated(u1, order, DOF)
    psi -= (psi @ u1)/(u1 @ u1)*u1
    psi *= np.linalg.norm(u1)/np.linalg.norm(psi)
    P1 = psi.reshape(-1, DOF)[order]
    #NOTE the crest over the circumference of a single harmonic is the
    #     envelope of its pair at every axial position, exact at the nodes of
    #     a column. Between two stations w follows the cubic Hermite
    #     interpolation of the element in w and w_x, degrees of freedom 6
    #     and 7, which is where a crest missed by the nodes is found
    crest_nodes = np.sqrt(W**2 + P1[:, :, 6]**2).max()/H
    xs = out['x'][order[:, 0]]
    t = np.linspace(0, 1, 41)[:, None]
    H00, H10 = 2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t
    H01, H11 = -2*t**3 + 3*t**2, t**3 - t**2
    crest = 0.
    for i in range(nx - 1):
        dx = xs[i + 1] - xs[i]
        env2 = 0.
        for F in (U1, P1):
            env2 = env2 + (H00*F[i, :, 6] + H10*dx*F[i, :, 7]
                           + H01*F[i + 1, :, 6] + H11*dx*F[i + 1, :, 7])**2
        crest = max(crest, np.sqrt(env2).max())
    crest /= H

    KC, KG = operators(out, prop, R, lib, element, nonlinear=not linear)
    mu_k = harmonic_multipliers(out, KC, KG, R)
    lowest = sorted(mu_k, key=mu_k.get)
    res = dict(case=case,
               kinematics='Sanders' if lib is bfsccylinder.sanders
                          else 'von Karman',
               NLprebuck=not linear,
               ny=ny, nx=nx, eps1=eps1, rot_deg=rot_deg, n=n,
               lambda_c=float(out['load_mult'][0]*Nxxunit/Ncl),
               lambda_b_over_lambda_c=float(1/mu[0]), b=float(b),
               max_nodal_w_over_h=float(np.abs(W).max()/H),
               crest_at_nodes_over_h=float(crest_nodes),
               b_crest_at_nodes=float(b/crest_nodes**2),
               crest_over_h=float(crest), b_env=float(b/crest**2),
               mu_0=mu_k[0]/mu[0], mu_2n=mu_k.get(2*n, np.nan)/mu[0],
               lowest_harmonics={k: mu_k[k]/mu[0] for k in lowest[:4]},
               time=time.time() - t0)
    print('RESULT ' + json.dumps(res))
    return res


if __name__ == '__main__':
    args = sys.argv[1:]
    case = 'sun' if 'sun' in args else 'arbocz'
    sanders = 'sanders' in args
    linear = 'linear' in args
    args = [a for a in args if a not in ('sun', 'sanders', 'linear')]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    main(int(args[0]), int(args[1]),
         float(args[2]) if len(args) > 2 else 0.005,
         float(args[3]) if len(args) > 3 else 0., case, sanders, linear)
