"""Is KC0 + KCNL(u) + KG(u) the second variation that the Koiter tensors
describe, and is it the derivative of fint?

    d fint_a / du_b = int  eps,a' A eps,b + eps,a' B kap,b
                         + kap,a' B eps,b + kap,a' D kap,b
                         + N_i eps,ab_i

The last term is the one the Koiter code calls KG. Whether the element splits
the first four between KC0 and KCNL the same way is a convention; what must
hold is that the SUM equals d fint/du, because phi2 is built as KC + KG.

Reported in: Section "The tangent stiffness matrix".
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
from bfsccylinder.quadrature import get_points_weights

DOF = 10
num_nodes = 4
R = 0.0680
lex, ley = 0.02, 0.015

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


def build(cls):
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


def run(name, mod, cls, sanders):
    points, weights = get_points_weights(nint=4)
    elem = build(cls)
    N = num_nodes*DOF
    u = 1e-4*rng.standard_normal(N)

    def asm(fn, size):
        r = np.zeros(size, dtype=mod.INT)
        c = np.zeros(size, dtype=mod.INT)
        v = np.zeros(size, dtype=mod.DOUBLE)
        if fn is mod.update_KC0:
            fn(elem, points, weights, r, c, v)
        else:
            fn(u, elem, points, weights, r, c, v)
        return coo_matrix((v, (r, c)), shape=(N, N)).toarray()

    KC0 = asm(mod.update_KC0, mod.KC0_SPARSE_SIZE)
    KCNL = asm(mod.update_KCNL, mod.KCNL_SPARSE_SIZE)
    KG = asm(mod.update_KG, mod.KG_SPARSE_SIZE)
    KT_elem = KC0 + KCNL + KG

    def fint_of(uu):
        f = np.zeros(N)
        mod.update_fint(uu, elem, points, weights, f)
        return f

    # numerical d fint / du
    KT_fd = np.zeros((N, N))
    h = 1e-8
    for b in range(N):
        up = u.copy(); up[b] += h
        um = u.copy(); um[b] -= h
        KT_fd[:, b] = (fint_of(up) - fint_of(um))/(2*h)

    # the Koiter-side second variation
    KT_py = np.zeros((N, N))
    KG_py = np.zeros((N, N))
    for i in range(4):
        xi, wi = points[i], weights[i]
        for j in range(4):
            eta, wj = points[j], weights[j]
            w = wi*wj*(lex*ley/4.)
            elem.update_Sw_x(xi, eta); elem.update_Sw_y(xi, eta)
            elem.update_Bm(xi, eta); elem.update_Bb(xi, eta)
            G1 = np.atleast_2d(elem.Sw_x)
            G2 = np.atleast_2d(elem.Sw_y)
            if sanders:
                elem.update_Sv(xi, eta)
                G2 = G2 - np.atleast_2d(elem.Sv)/R
            Bm = np.asarray(elem.Bm); Bb = np.asarray(elem.Bb)
            g1 = G1[0] @ u; g2 = G2[0] @ u
            eps = Bm @ u + np.array([g1**2/2., g2**2/2., g1*g2])
            kap = Bb @ u
            Ni = A @ eps + B @ kap
            eps_a = Bm + np.array([g1*G1[0], g2*G2[0], g1*G2[0] + g2*G1[0]])
            eps_ab = np.array([G1.T @ G1, G2.T @ G2, G1.T @ G2 + G2.T @ G1])
            KG_py += w*np.einsum('i,iab->ab', Ni, eps_ab)
            KT_py += w*(eps_a.T @ A @ eps_a + eps_a.T @ B @ Bb
                        + Bb.T @ B @ eps_a + Bb.T @ D @ Bb
                        + np.einsum('i,iab->ab', Ni, eps_ab))

    rel = lambda a, b: np.abs(a - b).max()/max(np.abs(b).max(), 1e-300)
    print('  %-10s KT(elem) vs d fint/du   rel %.3e' % (name, rel(KT_elem, KT_fd)))
    print('  %-10s KT(Koiter) vs KT(elem)  rel %.3e' % ('', rel(KT_py, KT_elem)))
    print('  %-10s KG(elem) vs N.eps_ab    rel %.3e' % ('', rel(KG, KG_py)))
    # what N does the element's KG actually use?
    KG_lin = np.zeros((N, N))
    for i in range(4):
        xi, wi = points[i], weights[i]
        for j in range(4):
            eta, wj = points[j], weights[j]
            w = wi*wj*(lex*ley/4.)
            elem.update_Sw_x(xi, eta); elem.update_Sw_y(xi, eta)
            elem.update_Bm(xi, eta); elem.update_Bb(xi, eta)
            G1 = np.atleast_2d(elem.Sw_x)
            G2 = np.atleast_2d(elem.Sw_y)
            if sanders:
                elem.update_Sv(xi, eta)
                G2 = G2 - np.atleast_2d(elem.Sv)/R
            Bm = np.asarray(elem.Bm); Bb = np.asarray(elem.Bb)
            Ni_lin = A @ (Bm @ u) + B @ (Bb @ u)   # LINEAR strain only
            eps_ab = np.array([G1.T @ G1, G2.T @ G2, G1.T @ G2 + G2.T @ G1])
            KG_lin += w*np.einsum('i,iab->ab', Ni_lin, eps_ab)
    print('  %-10s KG(elem) vs N_lin.eps_ab rel %.3e' % ('', rel(KG, KG_lin)))


print('von Karman'); run('vK', vk, vk.BFSCCylinder, False)
print('Sanders');    run('sa', sa, sa.BFSCCylinderSanders, True)
