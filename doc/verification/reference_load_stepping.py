"""How does the load stepping reach the expansion point, and what does it
pass through on the way?

On the meshes of the test suite, Sun et al. 3.1 with ny=40 (Sanders) and
AW-CYL-1-1 with ny=40 (von Karman), and on AW-CYL-1-1 with ny=60, it reports

- at the reference load, where the first eigenvalue problem is solved, the
  largest radial deflection over h and the Frobenius norm of KCNL over that
  of KC0
- per load step, lambda_b/lambda_c, the sensitivity s = dlambda_c/dlambda_b
  over the step that led to it with the largest eta = 1/(1 - s) that would
  not have overshot, and the residuals of the Newton-Raphson that takes the
  state on to the next step, with any back-off or overshoot
- at the converged state, the multipliers the eigen solver returned with the
  circumferential wave number of each, the number of vectors spanning the
  null space of phi2, and a_111
- the reduced matrix of the axisymmetric Newton-Raphson at the converged
  state, its condition number without and with the symmetric scaling, and the
  relative difference the scaling makes to the solve for the rate of the
  pre-buckling state
- the residual of phi2 along the buckling mode relative to that of KC, with
  phi2 = KC + mu KG built from the operators of the eigenvalue problem, and
  with KCNL evaluated at lambda_c u0 instead

and, for AW-CYL-1-1 with ny=40 asked for four multipliers instead of two,
what the eigen solver returns.

Reported in: Sections "Conditioning of the reduced matrix", "Why the load
level matters", "Advancing the load level", "Fixing the member", "Consistency
of the operators", "The bordered system", "What the weighted condition does,
and does not, change" and "What the tangent correction changed".
Runtime: about 6 minutes.
"""

import contextlib
import io
import os
import re
import sys
#NOTE the repository root goes FIRST on sys.path: an installed
#     bfsccylinder_models would otherwise shadow the working tree, silently
#     verifying a different version of the code than the one being edited
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir))

import numpy as np
from scipy.sparse.linalg import norm as spnorm
from composites import laminated_plate
import bfsccylinder
import bfsccylinder.sanders

import bfsccylinder_models.koiter_cylinder as vk_model
import bfsccylinder_models.koiter_cylinder_sanders as sa_model
from bfsccylinder_models.cyclic_symmetry import axisymmetric_basis, mesh_order
from reference_b_convergence import (DOF, E11, E22, G12, nu12, STACK, H, L,
                                     operators)

# tag, model, library, element, R, Nxxunit, ny, nx per ny, multipliers,
# whether to carry on to the Koiter expansion
CASES = [
    ('Sun et al. 3.1, ny=40', sa_model, bfsccylinder.sanders,
     bfsccylinder.sanders.BFSCCylinderSanders, 0.2032, 20000., 40, 1., 4,
     True),
    ('AW-CYL-1-1, ny=40', vk_model, bfsccylinder, bfsccylinder.BFSCCylinder,
     0.20318603, 10000., 40, 1.5, 2, True),
    ('AW-CYL-1-1, ny=60', vk_model, bfsccylinder, bfsccylinder.BFSCCylinder,
     0.20318603, 10000., 60, 1.5, 2, True),
    ('AW-CYL-1-1, ny=40, four multipliers', vk_model, bfsccylinder,
     bfsccylinder.BFSCCylinder, 0.20318603, 10000., 40, 1.5, 4, False),
]


def unknown_dofs(out):
    """The boundary conditions of fkoiter_cyl_SS3"""
    x, y = out['x'], out['y']
    bk = np.zeros(DOF*x.shape[0], dtype=bool)
    edges = np.isclose(x, 0) | np.isclose(x, L)
    bk[3::DOF] = edges
    bk[6::DOF] = edges
    bk[0::DOF] = np.isclose(x, L/2.) & np.isclose(y, 0)
    return ~bk


def wave_number(vec, nx, ny):
    w = vec.reshape(nx, ny, DOF)[:, :, 6]
    return int(np.argmax((np.abs(np.fft.rfft(w, axis=1))**2).sum(axis=0)))


def load_steps(text):
    steps = []
    prev = None
    for line in text.splitlines():
        m = re.match(r'#    iteration (\d+) lambda_b (\S+) lambda_c (\S+) '
                     r'lambda_b/lambda_c (\S+)$', line)
        if m:
            lb, lc = float(m.group(2)), float(m.group(3))
            s = None
            if prev is not None and lb != prev[0]:
                s = (lc - prev[1])/(lb - prev[0])
            steps.append(dict(step=int(m.group(1)), ratio=lb/lc, s=s,
                              residuals=[], events=[]))
            prev = (lb, lc)
            continue
        if not steps:
            continue
        m = re.match(r'#        iteration \d+ crisfield_test (\S+)$', line)
        if m:
            steps[-1]['residuals'].append(float(m.group(1)))
        elif re.search(r'backing off|overshot|keeping the best|WARNING', line):
            steps[-1]['events'].append(line.lstrip('# '))
    return steps


def run(tag, model, lib, element, R, Nxxunit, ny, fx, num_eigvals, koiter):
    nx = int(fx*ny*L/(2*np.pi*R))
    nx += 1 - nx % 2
    prop = laminated_plate(stack=STACK,
            laminaprop=(E11, E22, nu12, G12, G12, G12), plyt=H/len(STACK))
    #NOTE the first sparse solve of the model is the linear static one,
    #     KC0 u = fext, and the first eigenvalue problem is posed at that
    #     state with KC0 + KCNL(u) as its second matrix
    first = {}
    spsolve, eigsh = model.spsolve, model.eigsh

    def spy_spsolve(A, b):
        x = spsolve(A, b)
        first.setdefault('KC0', A)
        first.setdefault('u', x)
        return x

    def spy_eigsh(**kwargs):
        first.setdefault('KC', kwargs['M'])
        return eigsh(**kwargs)

    model.spsolve, model.eigsh = spy_spsolve, spy_eigsh
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            out = model.fkoiter_cyl_SS3(L, R, nx, ny, prop,
                    nint=4, num_eigvals=num_eigvals,
                    koiter_num_modes=1 if koiter else 0, Nxxunit=Nxxunit,
                    NLprebuck=True)
    finally:
        model.spsolve, model.eigsh = spsolve, eigsh
    text = log.getvalue()

    print('\n=== %s  nx %d  ny %d' % (tag, nx, ny))
    bu = unknown_dofs(out)
    u = np.zeros(bu.shape[0])
    u[bu] = first['u']
    print('  reference load: max |w|/h %.4f   |KCNL|/|KC0| %.1e'
          % (np.abs(u[6::DOF]).max()/H,
             spnorm(first['KC'] - first['KC0'])/spnorm(first['KC0'])))

    print('  step  lambda_b/lambda_c        s  1/(1-s)  Newton-Raphson '
          'residuals to the next step')
    for st in load_steps(text):
        if st['s'] is None:
            sens = '%8s %8s' % ('-', '-')
        else:
            sens = '%8.3f %8.3f' % (st['s'], 1/(1 - st['s']))
        print('  %4d  %17.5f %s  %s' % (st['step'], st['ratio'], sens,
              ' '.join('%.1e' % r for r in st['residuals'])))
        for event in st['events']:
            print('          %s' % event)

    lm = out['load_mult']
    print('  multipliers returned, over the critical one: %s'
          % ', '.join('%.5f (n=%d)' % (m/lm[0],
                                        wave_number(out['eigvecs'][:, k],
                                                    nx, ny))
                      for k, m in enumerate(lm)))
    if not koiter:
        return
    nsp = int(re.search(r'deflated with (\d+) vectors', text).group(1))
    print('  null space of phi2 spanned by %d vectors   a_111 %.1e'
          % (nsp, out['koiter']['a_ijk'][(0, 0, 0)]))

    KC, KG = operators(out, prop, R, lib, element)
    KT = KC + KG
    order = mesh_order(out['x'], out['y'], nx, ny)
    Baxi, buaxi = axisymmetric_basis(order, bu, DOF)
    Krf = (Baxi.T @ KT @ Baxi).toarray()[np.ix_(buaxi, buaxi)]
    d = 1/np.sqrt(np.abs(Krf.diagonal()))
    Krs = d[:, None]*Krf*d[None, :]
    r = (Baxi.T @ (KT @ out['koiter']['u0dot']))[buaxi]
    a_plain = np.linalg.solve(Krf, r)
    a_scaled = d*np.linalg.solve(Krs, d*r)
    print('  reduced tangent, %d unknowns: condition number %.1e, scaled '
          '%.1e; the scaling changes the solve by %.1e'
          % (Krf.shape[0], np.linalg.cond(Krf), np.linalg.cond(Krs),
             np.linalg.norm(a_plain - a_scaled)/np.linalg.norm(a_scaled)))

    mu0 = out['mu'][0]
    phi = out['eigvecs'][:, 0][bu]
    KGuu = KG[bu, :][:, bu]
    bad = dict(out, koiter=dict(out['koiter'],
                                u0=lm[0]*out['koiter']['u0']))
    for label, K in (('KC of the eigenproblem', KC),
                     ('KCNL(lambda_c u0)', operators(bad, prop, R, lib,
                                                     element)[0])):
        Kuu = K[bu, :][:, bu]
        print('  |(KC + mu KG) phi|/|KC phi| with %-22s %.1e'
              % (label, np.linalg.norm((Kuu + mu0*KGuu) @ phi)
                 /np.linalg.norm(Kuu @ phi)))


if __name__ == '__main__':
    for case in CASES:
        run(*case)
        sys.stdout.flush()
