"""Does the nonlinear kinematics assumed by the Koiter expansion agree with
the one the finite element itself integrates?

The Koiter code assumes, on one Gauss point,

    eps   = Bm u + [g1**2/2, g2**2/2, g1 g2]        (the strain)
    eps,a = Bm + [g1 G1, g2 G2, g1 G2 + g2 G1]      (first derivative)
    eps,ab = [G1' G1, G2' G2, G1' G2 + G2' G1]      (second derivative)

with (G1, G2) = (Sw_x, Sw_y) for von Karman and (Sw_x, Sw_y - Sv/R) for
Sanders. Those three are exactly what fint and KG are built from:

    fint_a = int  N_i eps,a  +  M_i kappa,a
    KG_ab  = int  N_i eps,ab +  M_i kappa,ab,  kappa,ab = 0

so rebuilding both from the Python side and comparing against update_fint and
update_KG tests the assumed kinematics against the element, independently of
anything in bfsccylinder_models.

Reported in: Section "Consistency of the operators".
Runtime: seconds.
"""

import os
import sys
#NOTE the repository root goes FIRST on sys.path: an installed
#     bfsccylinder_models would otherwise shadow the working tree, silently
#     verifying a different version of the code than the one being edited
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir))

import numpy as np
from scipy.sparse import coo_matrix

import bfsccylinder as vk
import bfsccylinder.sanders as sa

DOF = 10
num_nodes = 4
R = 0.0680
lex, ley = 0.02, 0.015

#NOTE a fully populated, non-symmetric, non-balanced laminate, so that every
#     coupling term participates and no error can hide behind a zero
rng = np.random.default_rng(3)
A = np.array([[6.1e7, 2.0e7, 9.0e6],
              [2.0e7, 4.3e7, 7.0e6],
              [9.0e6, 7.0e6, 2.4e7]])
B = np.array([[3.1e4, 1.2e4, 5.0e3],
              [1.2e4, 2.2e4, 4.0e3],
              [5.0e3, 4.0e3, 1.4e4]])
D = np.array([[52.0, 17.0, 8.0],
              [17.0, 39.0, 6.0],
              [8.0, 6.0, 21.0]])


def build(mod, cls):
    elem = cls(4)
    elem.n1, elem.n2, elem.n3, elem.n4 = 1, 2, 3, 4
    elem.c1, elem.c2, elem.c3, elem.c4 = 0, DOF, 2*DOF, 3*DOF
    elem.R, elem.lex, elem.ley = R, lex, ley
    elem.init_k_KC0 = elem.init_k_KCNL = elem.init_k_KG = 0
    for nm, M in (('A', A), ('B', B), ('D', D)):
        for (a, b), idx in {(0, 0): '11', (0, 1): '12', (0, 2): '16',
                            (1, 1): '22', (1, 2): '26', (2, 2): '66'}.items():
            getattr(elem, nm + idx)[:, :] = M[a, b]
    return elem


def check(name, mod, cls, sanders):
    from bfsccylinder.quadrature import get_points_weights
    points, weights = get_points_weights(nint=4)
    elem = build(mod, cls)
    N = num_nodes*DOF
    u = 1e-4*rng.standard_normal(N)

    # ---- what the element says
    fint_elem = np.zeros(N)
    mod.update_fint(u, elem, points, weights, fint_elem)

    KGr = np.zeros(mod.KG_SPARSE_SIZE, dtype=mod.INT)
    KGc = np.zeros(mod.KG_SPARSE_SIZE, dtype=mod.INT)
    KGv = np.zeros(mod.KG_SPARSE_SIZE, dtype=mod.DOUBLE)
    mod.update_KG(u, elem, points, weights, KGr, KGc, KGv)
    KG_elem = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).toarray()

    # ---- what the Koiter kinematics says
    fint_py = np.zeros(N)
    KG_py = np.zeros((N, N))
    for i in range(4):
        xi, wi = points[i], weights[i]
        for j in range(4):
            eta, wj = points[j], weights[j]
            w = wi*wj*(lex*ley/4.)

            elem.update_Sw_x(xi, eta)
            elem.update_Sw_y(xi, eta)
            elem.update_Bm(xi, eta)
            elem.update_Bb(xi, eta)
            Sw_x = np.atleast_2d(elem.Sw_x)
            Sw_y = np.atleast_2d(elem.Sw_y)

            G1 = Sw_x
            if sanders:
                elem.update_Sv(xi, eta)
                G2 = Sw_y - np.atleast_2d(elem.Sv)/R
            else:
                G2 = Sw_y

            Bm = np.asarray(elem.Bm)
            Bb = np.asarray(elem.Bb)
            g1 = G1[0] @ u
            g2 = G2[0] @ u

            eps = Bm @ u + np.array([g1**2/2., g2**2/2., g1*g2])
            kap = Bb @ u
            Ni = A @ eps + B @ kap
            Mi = B @ eps + D @ kap

            eps_a = Bm + np.array([g1*G1[0], g2*G2[0], g1*G2[0] + g2*G1[0]])
            eps_ab = np.array([G1.T @ G1, G2.T @ G2, G1.T @ G2 + G2.T @ G1])

            fint_py += w*(Ni @ eps_a + Mi @ Bb)
            KG_py += w*np.einsum('i,iab->ab', Ni, eps_ab)

    rel = lambda a, b: np.abs(a - b).max()/max(np.abs(b).max(), 1e-300)
    print('%-11s fint  rel %.3e     KG  rel %.3e'
          % (name, rel(fint_py, fint_elem), rel(KG_py, KG_elem)))
    return rel(fint_py, fint_elem), rel(KG_py, KG_elem)


print('kinematics assumed by the Koiter expansion vs. the element:\n')
r1 = check('von Karman', vk, vk.BFSCCylinder, sanders=False)
r2 = check('Sanders', sa, sa.BFSCCylinderSanders, sanders=True)

print('\ncontrol: Sanders element with von Karman kinematics (G2 = Sw_y),')
print('i.e. what a MISSING -Sv/R would give')
r3 = check('sa w/ vK', sa, sa.BFSCCylinderSanders, sanders=False)
