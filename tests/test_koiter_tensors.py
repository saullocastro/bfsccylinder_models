"""koiter_tensors.py against the loops it replaced

Until version 0.3.2 the four models integrated the Koiter tensors with Python
loops over every pair, and for phi4 every quadruple, of Koiter modes at every
integration point, and computed a_ijk and b_ijkl in loops over their indices.
reference_element_tensors and reference_coefficients below are those loops,
transcribed from koiter_cylinder_CTS_sanders.py of version 0.3.2 with only the
gathers and the kinematics factored out. The vectorized functions must
reproduce them to round off: the element tensors for both kinematics and with
and without the nonlinear pre-buckling terms, on elements with random
stiffness, pre-buckling state and modes, where no symmetry of a real cylinder
can hide a wrong index, and the coefficients on random contractions.
"""
import os
import sys
from functools import partial
from collections import defaultdict

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir))

from bfsccylinder import BFSCCylinder, DOF
from bfsccylinder.sanders import BFSCCylinderSanders
from bfsccylinder.quadrature import get_points_weights

from bfsccylinder_models.koiter_tensors import (koiter_element_tensors,
        a_coefficients, b_coefficients)
from bfsccylinder_models import koiter_cylinder, koiter_cylinder_CTS_sanders

num_nodes = 4
nint = 4
R = 0.4


def _elements(sanders, rng):
    """Four elements of a 3 x 3 node patch, with random ABD at every point"""
    nodes = np.arange(9).reshape(3, 3)
    elements = []
    for i in range(2):
        for j in range(2):
            elem = BFSCCylinderSanders(nint) if sanders else BFSCCylinder(nint)
            elem.c1 = DOF*nodes[i, j]
            elem.c2 = DOF*nodes[i+1, j]
            elem.c3 = DOF*nodes[i+1, j+1]
            elem.c4 = DOF*nodes[i, j+1]
            elem.R = R
            elem.lex = 0.05*(1 + i)
            elem.ley = 0.03*(1 + j)
            for name in ['A11', 'A12', 'A16', 'A22', 'A26', 'A66']:
                np.asarray(getattr(elem, name))[:] = 1e7*(1 + rng.random((nint, nint)))
            for name in ['B11', 'B12', 'B16', 'B22', 'B26', 'B66']:
                np.asarray(getattr(elem, name))[:] = 1e3*rng.standard_normal((nint, nint))
            elements.append(elem)
    return elements, DOF*9


def reference_element_tensors(elements, points, weights, u0, u0dot, u0ddot,
        ucond, koiter_num_modes, flag, sanders):
    """The element loop of version 0.3.2"""
    es = partial(np.einsum, optimize='greedy', casting='no')
    N = u0.shape[0]
    num_cond = len(ucond)
    phi4 = defaultdict(lambda: 0)
    phi3_ab = {}
    phi30_ab = {}
    cst_ab = {}
    phi3e_ab = {}
    phi30e_ab = {}
    cste_ab = {}
    phi20e_a = {}
    phi20_a = {}
    phi200_ab = {}
    for modei in range(num_cond):
        phi20_a[modei] = np.zeros(N)
        phi20e_a[modei] = np.zeros(num_nodes*DOF)
    for modei in range(koiter_num_modes):
        for modej in range(koiter_num_modes):
            phi200_ab[(modei, modej)] = 0
            phi3_ab[(modei, modej)] = np.zeros(N)
            phi30_ab[(modei, modej)] = np.zeros(N)
            cst_ab[(modei, modej)] = np.zeros(N)
            phi3e_ab[(modei, modej)] = np.zeros(num_nodes*DOF)
            phi30e_ab[(modei, modej)] = np.zeros(num_nodes*DOF)
            cste_ab[(modei, modej)] = np.zeros(num_nodes*DOF)

    for elem in elements:
        eiab = np.zeros((3, num_nodes*DOF, num_nodes*DOF))
        indices = []
        for ci in [elem.c1, elem.c2, elem.c3, elem.c4]:
            for i in range(DOF):
                indices.append(ci + i)
        u0e = u0[indices]
        u0dote = u0dot[indices]
        u0ddote = u0ddot[indices]
        uae = {modei: ucond[modei][indices] for modei in range(num_cond)}
        ube = uce = ude = uae
        lex = elem.lex
        ley = elem.ley
        for modei in range(num_cond):
            phi20e_a[modei] *= 0
        for modei in range(koiter_num_modes):
            for modej in range(koiter_num_modes):
                phi3e_ab[(modei, modej)] *= 0
                phi30e_ab[(modei, modej)] *= 0
                cste_ab[(modei, modej)] *= 0

        for i in range(nint):
            xi = points[i]
            weight_xi = weights[i]
            for j in range(nint):
                Aij = np.array([
                    [elem.A11[i, j], elem.A12[i, j], elem.A16[i, j]],
                    [elem.A12[i, j], elem.A22[i, j], elem.A26[i, j]],
                    [elem.A16[i, j], elem.A26[i, j], elem.A66[i, j]]])
                Bij = np.array([
                    [elem.B11[i, j], elem.B12[i, j], elem.B16[i, j]],
                    [elem.B12[i, j], elem.B22[i, j], elem.B26[i, j]],
                    [elem.B16[i, j], elem.B26[i, j], elem.B66[i, j]]])
                eta = points[j]
                weight_eta = weights[j]
                weight = weight_xi * weight_eta

                elem.update_Sw_x(xi, eta)
                elem.update_Sw_y(xi, eta)
                elem.update_Bm(xi, eta)
                elem.update_Bb(xi, eta)
                Sw_x = np.atleast_2d(elem.Sw_x)
                Sw_y = np.atleast_2d(elem.Sw_y)
                if sanders:
                    elem.update_Sv(xi, eta)
                    Sv = np.atleast_2d(elem.Sv)
                    G1 = Sw_x
                    G2 = Sw_y - Sv/R
                else:
                    G1 = Sw_x
                    G2 = Sw_y

                g1_s = G1[0] @ u0e
                g2_s = G2[0] @ u0e
                g1_d = G1[0] @ u0dote
                g2_d = G2[0] @ u0dote
                g1_dd = G1[0] @ u0ddote
                g2_dd = G2[0] @ u0ddote

                Bm = np.asarray(elem.Bm)
                Bb = np.asarray(elem.Bb)

                ei0 = ej0 = Bm @ u0dote + flag*np.array([g1_s*g1_d,
                                                         g2_s*g2_d,
                                                         g1_s*g2_d + g2_s*g1_d])
                ki0 = kj0 = Bb @ u0dote
                ei00 = ej00 = Bm @ u0ddote + flag*np.array([
                        g1_d**2 + g1_s*g1_dd,
                        g2_d**2 + g2_s*g2_dd,
                        2*g1_d*g2_d + g1_s*g2_dd + g2_s*g1_dd])
                ki00 = kj00 = Bb @ u0ddote
                Ni0 = Aij@ej0 + Bij@kj0
                Ni00 = Aij@ej00 + Bij@kj00
                eia = eib = eic = Bm + flag*np.array([g1_s*G1[0],
                                                      g2_s*G2[0],
                                                      g1_s*G2[0] + g2_s*G1[0]])
                kia = kib = kic = Bb
                Nia = Nib = Nic = es('ij,ja->ia', Aij, eia) + es('ij,ja->ia', Bij, kia)
                eia0 = eib0 = eic0 = flag*np.array([g1_d*G1[0],
                                                    g2_d*G2[0],
                                                    g1_d*G2[0] + g2_d*G1[0]])
                Nia0 = Nib0 = Nic0 = es('ij,ja->ia', Aij, eia0)
                Mia0 = Mib0 = es('ij,ja->ia', Bij, eia0)
                eiab[0] = G1.T @ G1
                eiab[1] = G2.T @ G2
                eiab[2] = G1.T @ G2 + G2.T @ G1
                eicd = eibd = eibc = eiad = eiac = eiab
                Niab = Niac = Niad = Nibc = Nibd = Nicd = es('ij,jab->iab', Aij, eiab)
                Miab = Miac = Mibc = es('ij,jab->iab', Bij, eiab)

                for modei in range(num_cond):
                    ua1 = uae[modei]
                    phi20e_a[modei] += 1/2.*weight*(lex*ley/4.)*(
                            (ei0 @ (Niab @ ua1))
                         +  ((Nia0 @ ua1) @ eib)
                         +  ((Nia @ ua1) @ eib0)
                         +  ((eia @ ua1) @ Nib0)
                         +  ((eia0 @ ua1) @ Nib)
                         +  (Ni0 @ (eiab @ ua1))
                         +  (ki0 @ (Miab @ ua1))
                         +  ((Mia0 @ ua1) @ kib)
                         +  ((kia @ ua1) @ Mib0)
                    )

                for modei in range(koiter_num_modes):
                    ua1 = uae[modei]
                    for modej in range(koiter_num_modes):
                        ub2 = ube[modej]
                        phi200_ab[(modei, modej)] += 1/2.*weight*(lex*ley/4.)*(
                                es('iab,i,a,b', Niab, ei00, ua1, ub2)
                            + 2*es('ia,ib,a,b', Nia0, eib0, ua1, ub2)
                            + 2*es('ib,ia,a,b', Nib0, eia0, ua1, ub2)
                              + es('i,iab,a,b', Ni00, eiab, ua1, ub2)
                            )
                        phi3e_ab[(modei, modej)] += 1/2.*weight*(lex*ley/4.)*(
                              (((Niab @ ub2) @ ua1) @ eic)
                            + ((eib @ ub2) @ (Niac @ ua1))
                            + ((Nia @ ua1) @ (eibc @ ub2))
                            + ((eia @ ua1) @ (Nibc @ ub2))
                            + ((Nib @ ub2) @ (eiac @ ua1))
                            + (((eiab @ ub2) @ ua1) @ Nic)
                            + (((Miab @ ub2) @ ua1) @ kic)
                            + ((kib @ ub2) @ (Miac @ ua1))
                            + ((kia @ ua1) @ (Mibc @ ub2))
                            )
                        phi30e_ab[(modei, modej)] += 1/2.*weight*(lex*ley/4.)*(
                              es('iab,ic,a,b', Niab, eic0, ua1, ub2)
                            + es('iac,ib,a,b', Niac, eib0, ua1, ub2)
                            + es('ia,ibc,a,b', Nia0, eibc, ua1, ub2)
                            + es('ibc,ia,a,b', Nibc, eia0, ua1, ub2)
                            + es('ib,iac,a,b', Nib0, eiac, ua1, ub2)
                            + es('ic,iab,a,b', Nic0, eiab, ua1, ub2)
                            )
                        cste_ab[(modei, modej)] += 1/2.*weight*(lex*ley/4.)*(
                              es('iab,ic,a,b', Niab, eic0, ua1, ub2))

                def fphi4(ua, ub, uc, ud):
                    return 1/2.*weight*(lex*ley/4.)*(
                          ((Niab @ ub) @ ua) @ ((eicd @ ud) @ uc)
                        + ((Niac @ uc) @ ua) @ ((eibd @ ud) @ ub)
                        + ((Niad @ ud) @ ua) @ ((eibc @ uc) @ ub)
                        + ((Nibc @ uc) @ ub) @ ((eiad @ ud) @ ua)
                        + ((Nibd @ ud) @ ub) @ ((eiac @ uc) @ ua)
                        + ((Nicd @ ud) @ uc) @ ((eiab @ ub) @ ua)
                        )

                for modei in range(koiter_num_modes):
                    for modej in range(koiter_num_modes):
                        for modek in range(koiter_num_modes):
                            for model in range(koiter_num_modes):
                                phi4[(modei, modej, modek, model)] += fphi4(uae[modei], ube[modej], uce[modek], ude[model])

        for modei in range(num_cond):
            phi20_a[modei][indices] += phi20e_a[modei]
        for modei in range(koiter_num_modes):
            for modej in range(koiter_num_modes):
                phi3_ab[(modei, modej)][indices] += phi3e_ab[(modei, modej)]
                phi30_ab[(modei, modej)][indices] += phi30e_ab[(modei, modej)]
                cst_ab[(modei, modej)][indices] += cste_ab[(modei, modej)]

    return phi20_a, phi3_ab, phi30_ab, cst_ab, phi200_ab, phi4


def _assert_close(new, ref, name):
    scale = np.abs(ref).max()
    assert scale > 0, name
    err = np.abs(new - ref).max()/scale
    assert err <= 1e-12, '%s: relative difference %.1e' % (name, err)


@pytest.mark.parametrize('flag', [False, True])
@pytest.mark.parametrize('sanders', [False, True])
def test_matches_the_loop_it_replaced(sanders, flag):
    rng = np.random.default_rng(1 + 2*sanders + flag)
    m = 3
    num_cond = 5
    elements, N = _elements(sanders, rng)
    points, weights = get_points_weights(nint=nint)
    u0, u0dot, u0ddot = 1e-4*rng.standard_normal((3, N))
    Ucond = rng.standard_normal((N, num_cond))
    ucond = {k: Ucond[:, k] for k in range(num_cond)}

    ref = reference_element_tensors(elements, points, weights, u0, u0dot,
            u0ddot, ucond, m, flag, sanders)
    model = koiter_cylinder_CTS_sanders if sanders else koiter_cylinder
    new = koiter_element_tensors(elements, points, weights, u0, u0dot,
            u0ddot, Ucond, m, flag, model.nonlinear_rows)
    phi20, phi3, phi30, cst, phi200, phi4 = new
    phi20_a, phi3_ab, phi30_ab, cst_ab, phi200_ab, phi4_ref = ref

    idx = np.ndindex(*(m,)*4)
    _assert_close(phi4, np.array([phi4_ref[k] for k in idx]).reshape((m,)*4),
                  'phi4')
    _assert_close(phi20, np.column_stack([phi20_a[k] for k in range(num_cond)]),
                  'phi20')
    for name, a, b in [('phi3', phi3, phi3_ab), ('phi30', phi30, phi30_ab),
                       ('cst', cst, cst_ab)]:
        if not flag and name != 'phi3':
            #NOTE made of eia0 only, which vanishes without NLprebuck
            assert np.abs(a).max() == 0 and max(np.abs(v).max()
                    for v in b.values()) == 0, name
            continue
        ref_ab = np.stack([np.column_stack([b[(i, j)] for j in range(m)])
                           for i in range(m)], axis=1)
        _assert_close(a, ref_ab, name)
    _assert_close(phi200, np.array([[phi200_ab[(i, j)] for j in range(m)]
                                    for i in range(m)]), 'phi200')


def reference_coefficients(phi3U, phi4, phi3uab, phi30U, phi200, lam, d):
    """The loops over the indices of a_ijk and b_ijkl of version 0.3.2, the
    contractions with the modes given"""
    m = lam.shape[0]
    a_abc = {}
    for modei in range(m):
        lambda_i = lam[modei]
        for modej in range(m):
            for modek in range(m):
                a_abc[(modei, modej, modek)] = -1./(2*lambda_i)*phi3U[modei, modej, modek]/d[modei]
    b_ijkl = {}
    for modei in range(m):
        lambda_i = lam[modei]
        for modej in range(m):
            for modek in range(m):
                for model in range(m):
                    b_ijkl[(modei, modej, modek, model)] = -1/(6*lambda_i*d[modei])*(
                            phi4[modei, modej, modek, model]
                            + 3*phi3uab[modei, modej, modek, model]
                            + 3*phi3uab[modei, model, modej, modek]
                            + lambda_i*(
                                a_abc[(modei, modei, modej)]*phi30U[modei, modek, model]
                               +a_abc[(modei, modej, modek)]*phi30U[modei, model, modei]
                               +a_abc[(modei, modek, model)]*phi30U[modei, modei, modej]
                                )
                            + phi200[modei, modei]*lambda_i**2*(
                                a_abc[(modei, modei, modej)]*a_abc[(modei, modek, model)]
                               +a_abc[(modei, modej, modek)]*a_abc[(modei, model, modei)]
                               +a_abc[(modei, modek, model)]*a_abc[(modei, modei, modej)]
                                )
                            )
    return a_abc, b_ijkl


def test_coefficients_match_the_loops_they_replaced():
    """Random contractions, so that every term of b_ijkl, those in a_ijk
    included, which vanish on a cylinder, is exercised with distinct
    values at every index"""
    rng = np.random.default_rng(7)
    m = 4
    phi3U = rng.standard_normal((m,)*3)
    phi4 = rng.standard_normal((m,)*4)
    phi3uab = rng.standard_normal((m,)*4)
    phi30U = rng.standard_normal((m,)*3)
    phi200 = rng.standard_normal((m, m))
    lam = 1 + rng.random(m)
    d = 1 + rng.random(m)
    a_ref, b_ref = reference_coefficients(phi3U, phi4, phi3uab, phi30U,
                                          phi200, lam, d)
    a = a_coefficients(phi3U, lam, d)
    b = b_coefficients(phi4, phi3uab, phi30U, phi200, a, lam, d)
    _assert_close(a, np.array([a_ref[k] for k in np.ndindex(a.shape)]
                              ).reshape(a.shape), 'a_ijk')
    _assert_close(b, np.array([b_ref[k] for k in np.ndindex(b.shape)]
                              ).reshape(b.shape), 'b_ijkl')


if __name__ == '__main__':
    for sanders in [False, True]:
        for flag in [False, True]:
            test_matches_the_loop_it_replaced(sanders, flag)
    test_coefficients_match_the_loops_they_replaced()
    print('ok')
