"""Koiter tensors and coefficients, shared by the four models

koiter_cylinder.py, koiter_cylinder_sanders.py, koiter_cylinder_CTS.py and
koiter_cylinder_CTS_sanders.py call the functions below with the same
arguments; the only thing each model supplies is ``nonlinear_rows``, its
kinematics, von Karman or Sanders. See "Vectorization of the Koiter tensors"
in doc/nlprebuck_implementation.tex.

Vectorization strategy
----------------------
Up to version 0.3.2 every model integrated the tensors with Python loops over
the Koiter modes at every integration point: over every pair of modes for
phi3, phi30, phi200 and the constants of the orthogonality conditions, and
over every quadruple for phi4, about 1e5 small NumPy calls per element with 5
modes. That loop is kept in tests/test_koiter_tensors.py, which checks the
functions below against it. Here the loops over the modes, and over the 16
integration points of an element, become axes of the arrays:

1. Axes. Within an element the arrays are indexed by
       p  the integration point, P = nint**2 of them
       i  the strain component, 3
       c  the element DOF, 40
       a, b, ...  the directions: the Koiter modes, or all the directions of
          the null space of phi2 (suffix c in the names, for Ucond)
   so that, for instance, EUc[p, i, a] is eps_i,u(u_a) at point p and
   eabUc[p, i, c, a] is the row c of d2eps_i/du2 applied to u_a.

2. One contraction per term and element. Every term of the old loops is a
   sum over p and i of the product of two such arrays, which sum_pi computes
   as a single matrix product; the weight of the point, w = 1/2 weight
   lex ley/4, is folded into the smaller operand beforehand. Which term of
   the old loop each call reproduces is noted next to it.

3. d2eps/du2 is never formed. It is [G1^T G1, G2^T G2, G1^T G2 + G2^T G1],
   so applied to a direction u it is G1^T and G2^T scaled by (G1 u, G2 u):
   eabUc and the second-order strain E[p, i, a, b] = u_a . d2eps_i/du2 . u_b
   are outer products of G1, G2 and q = G u, for every direction at once.

4. Pairs of terms. The terms of phi3, phi30 and phi200 come in pairs that are
   the same contraction with the mode indices a and b swapped. Each pair is
   computed once, the second member being a transpose, T().

5. phi4 once. Its six terms are permutations of the single quantity
   P[a, b, c, d] = sum w (A E_ab) . E_cd, which is accumulated over the
   points and elements; the permutations are applied once at the end. With A
   symmetric the six terms are three equal pairs; all six are kept.

6. Gather and scatter. The element vectors are taken from the global ones by
   fancy indexing with the 40 DOFs of the element, and the element tensors
   are added back the same way, those DOFs being distinct.

The coefficients a_ijk and b_ijkl follow the same idea: they are array
expressions over all their indices (a_coefficients, b_coefficients), from the
contractions of the global tensors with the modes, each one tensordot. The
dicts keyed by index tuples that the models return are built from them.

The result equals the old loop to round-off, 5e-16 of the largest entry of
every tensor, and the element loop costs about 1 ms per element, nearly
independent of the number of modes, against 1 s for 5 modes and 6 s for 8
before (doc/verification/element_loop_timing.py).
"""
import numpy as np

from bfsccylinder import DOF

num_nodes = 4


def calc_AB(elem):
    """Membrane and coupling stiffness at the integration points of elem

    Two (nint, nint, 3, 3) arrays indexed as elem.A11[i, j]. The CTS models
    fill elem.A11 and the like per integration point, as the laminate varies
    along x; the constant-stiffness models fill them with the same laminate
    everywhere (assign_constant_ABD), bit for bit prop.A and prop.B
    """
    A = np.array([
        [elem.A11, elem.A12, elem.A16],
        [elem.A12, elem.A22, elem.A26],
        [elem.A16, elem.A26, elem.A66]])
    B = np.array([
        [elem.B11, elem.B12, elem.B16],
        [elem.B12, elem.B22, elem.B26],
        [elem.B16, elem.B26, elem.B66]])
    return A.transpose(2, 3, 0, 1), B.transpose(2, 3, 0, 1)


def sum_pi(X, Y):
    """Sum over the integration points p and the strain components i

    Returns S[..., ...] = sum_{p,i} X[p, i, ...]*Y[p, i, ...], the remaining
    axes of X followed by those of Y, as one matrix product. np.einsum would
    do the same, but its per call overhead is what this module is meant to
    remove
    """
    n = X.shape[0]*X.shape[1]
    S = X.reshape(n, -1).T @ Y.reshape(n, -1)
    return S.reshape(X.shape[2:] + Y.shape[2:])


def T(S):
    """S[..., a, b] -> S[..., b, a], the second member of a pair of terms"""
    return S.swapaxes(-1, -2)


def koiter_element_tensors(elements, points, weights, u0, u0dot, u0ddot,
        Ucond, koiter_num_modes, flag, nonlinear_rows):
    """Koiter tensors assembled over all elements

    Parameters
    ----------
    elements : list
        The BFSC elements of the model.
    points, weights : array-like
        Gauss-Legendre points and weights of each direction.
    u0, u0dot, u0ddot : (N,) array
        Pre-buckling state and its first and second rates with respect to
        the load parameter.
    Ucond : (N, num_cond) array
        The Koiter modes in the first ``koiter_num_modes`` columns, then the
        remaining directions of the null space of phi2, for which only phi20
        is needed.
    koiter_num_modes : int
        Number of Koiter modes.
    flag : bool
        NLprebuck, multiplying every nonlinear contribution of the
        pre-buckling state, see the models.
    nonlinear_rows : callable
        ``nonlinear_rows(elem, xi, eta)`` returns (G1, G2), the rows of the
        nonlinear membrane strains eps_xx^NL = 1/2 (G1 u)**2,
        eps_yy^NL = 1/2 (G2 u)**2, gamma_xy^NL = (G1 u) (G2 u).

    Returns
    -------
    phi20 : (N, num_cond) array
        phi20_a[k] = phi20[:, k].
    phi3, phi30, cst : (N, m, m) arrays
        phi3_ab[(a, b)] = phi3[:, a, b], and likewise.
    phi200 : (m, m) array
    phi4 : (m, m, m, m) array
    """
    m = koiter_num_modes
    N = u0.shape[0]
    num_cond = Ucond.shape[1]
    num_elements = len(elements)
    nint = len(points)
    P = nint*nint

    #NOTE the integration points in the order of the old loops, xi outer and
    #     eta inner, so that calc_AB(elem).reshape(P, 3, 3) matches them
    xis = np.repeat(points, nint)
    etas = np.tile(points, nint)
    wpoints = np.repeat(weights, nint)*np.tile(weights, nint)

    Z = np.column_stack((u0, u0dot, u0ddot))
    offsets = np.arange(DOF)

    phi20 = np.zeros((N, num_cond))
    phi3 = np.zeros((N, m, m))
    phi30 = np.zeros((N, m, m))
    cst = np.zeros((N, m, m))
    phi200 = np.zeros((m, m))
    #NOTE sum over the integration points of w NE[:, a, b] E[:, c, d], out of
    #     which the six terms of phi4 are permutations, see the end
    Pabcd = np.zeros((m, m, m, m))

    G = np.zeros((P, 2, num_nodes*DOF))
    Bm = np.zeros((P, 3, num_nodes*DOF))
    Bb = np.zeros((P, 3, num_nodes*DOF))
    for count, elem in enumerate(elements):
        if count % max(1, num_elements//5) == 0:
            print('#    count', count+1, num_elements)
        #NOTE gather, strategy item 6
        indices = np.concatenate([c + offsets for c in
                                  (elem.c1, elem.c2, elem.c3, elem.c4)])
        Ze = Z[indices]
        u0dote = Ze[:, 1]
        u0ddote = Ze[:, 2]
        Uce = Ucond[indices]

        #NOTE elem.Bm and the like are views of buffers that the next update
        #     overwrites, hence the copies into G, Bm and Bb
        for p in range(P):
            G[p] = nonlinear_rows(elem, xis[p], etas[p])
            elem.update_Bm(xis[p], etas[p])
            elem.update_Bb(xis[p], etas[p])
            Bm[p] = elem.Bm
            Bb[p] = elem.Bb
        A, B = calc_AB(elem)
        A = A.reshape(P, 3, 3)
        B = B.reshape(P, 3, 3)
        #NOTE 1/2 weight (lex ley/4), the factor of every term of the old loops
        w = 1/2.*wpoints*(elem.lex*elem.ley/4.)

        G1 = G[:, 0]
        G2 = G[:, 1]
        #NOTE the pre-buckling STATE and its first and second RATES with
        #     respect to the load parameter are independent fields, see the
        #     models
        g = G @ Ze
        g1_s, g2_s = g[:, 0, 0], g[:, 1, 0]
        g1_d, g2_d = g[:, 0, 1], g[:, 1, 1]
        g1_dd, g2_dd = g[:, 0, 2], g[:, 1, 2]

        #NOTE eps_dot, the first derivative of the pre-buckling strain with
        #     respect to the load parameter, d/dl of
        #     Bm u + [g1**2/2, g2**2/2, g1 g2]
        ei0 = Bm @ u0dote + flag*np.column_stack((g1_s*g1_d,
                                                  g2_s*g2_d,
                                                  g1_s*g2_d + g2_s*g1_d))
        ki0 = Bb @ u0dote

        #NOTE eps_dot_dot, the second derivative
        ei00 = Bm @ u0ddote + flag*np.column_stack((
                g1_d**2 + g1_s*g1_dd,
                g2_d**2 + g2_s*g2_dd,
                2*g1_d*g2_d + g1_s*g2_dd + g2_s*g1_dd))
        ki00 = Bb @ u0ddote

        Ni0 = (A @ ei0[:, :, None] + B @ ki0[:, :, None])[:, :, 0]
        Ni00 = (A @ ei00[:, :, None] + B @ ki00[:, :, None])[:, :, 0]

        #NOTE d(eps)/d(u_a) at the pre-buckling state
        eia = Bm + flag*np.stack((g1_s[:, None]*G1,
                                  g2_s[:, None]*G2,
                                  g1_s[:, None]*G2 + g2_s[:, None]*G1), axis=1)
        kia = Bb
        Nia = A @ eia + B @ kia

        #NOTE d2(eps)/dl d(u_a)
        eia0 = flag*np.stack((g1_d[:, None]*G1,
                              g2_d[:, None]*G2,
                              g1_d[:, None]*G2 + g2_d[:, None]*G1), axis=1)
        Nia0 = A @ eia0
        Mia0 = B @ eia0

        #NOTE strategy item 3: eiab = [G1^T G1, G2^T G2, G1^T G2 + G2^T G1]
        #     is never formed; applied to a direction u it is G^T scaled by
        #     (G1 u, G2 u), so that
        #       eabUc[p, i, :, a] = eiab[p, i] @ u_a
        #       E[p, i, a, b]     = u_a @ eiab[p, i] @ u_b
        #     for every direction at once. Niab and Miab are Aij and Bij
        #     applied to the first axis of eiab
        q1 = G1 @ Uce
        q2 = G2 @ Uce
        eabUc = np.stack((G1[:, :, None]*q1[:, None, :],
                          G2[:, :, None]*q2[:, None, :],
                          G1[:, :, None]*q2[:, None, :]
                        + G2[:, :, None]*q1[:, None, :]), axis=1)
        shape = eabUc.shape
        NabUc = (A @ eabUc.reshape(P, 3, -1)).reshape(shape)
        MabUc = (B @ eabUc.reshape(P, 3, -1)).reshape(shape)

        #NOTE (ua @ eia) and the like, for every direction of Ucond, the
        #     weight w folded into the copies prefixed with w
        EUc = eia @ Uce
        E0Uc = eia0 @ Uce
        KUc = kia @ Uce
        NUc = Nia @ Uce
        N0Uc = Nia0 @ Uce
        M0Uc = Mia0 @ Uce
        wEUc, wE0Uc, wKUc, wNUc, wN0Uc, wM0Uc = (w[:, None, None]*X
                for X in (EUc, E0Uc, KUc, NUc, N0Uc, M0Uc))

        #NOTE phi20e_a[k] = phi20e[:, k], the terms in the order of the old
        #     loop
        phi20e = (sum_pi(w[:, None]*ei0, NabUc)  # ei0 @ (Niab @ ua1)
                + sum_pi(eia, wN0Uc)             # (Nia0 @ ua1) @ eib
                + sum_pi(eia0, wNUc)             # (Nia @ ua1) @ eib0
                + sum_pi(Nia0, wEUc)             # (eia @ ua1) @ Nib0
                + sum_pi(Nia, wE0Uc)             # (eia0 @ ua1) @ Nib
                + sum_pi(w[:, None]*Ni0, eabUc)  # Ni0 @ (eiab @ ua1)
                + sum_pi(w[:, None]*ki0, MabUc)  # ki0 @ (Miab @ ua1)
                + sum_pi(kia, wM0Uc)             # (Mia0 @ ua1) @ kib
                + sum_pi(Mia0, wKUc))            # (kia @ ua1) @ Mib0

        #NOTE the same quantities for the Koiter modes only, the first m
        #     directions of Ucond
        eabU = eabUc[..., :m]
        NabU = NabUc[..., :m]
        MabU = MabUc[..., :m]
        E0U = E0Uc[..., :m]
        wEU, wE0U, wKU, wNU, wN0U = (X[..., :m]
                for X in (wEUc, wE0Uc, wKUc, wNUc, wN0Uc))
        E = np.stack((q1[:, :m, None]*q1[:, None, :m],
                      q2[:, :m, None]*q2[:, None, :m],
                      q1[:, :m, None]*q2[:, None, :m]
                    + q2[:, :m, None]*q1[:, None, :m]), axis=1)
        NE = (A @ E.reshape(P, 3, m*m)).reshape(P, 3, m, m)
        ME = (B @ E.reshape(P, 3, m*m)).reshape(P, 3, m, m)
        wE = w[:, None, None, None]*E
        wNE = w[:, None, None, None]*NE
        wME = w[:, None, None, None]*ME

        #NOTE phi3e_ab[(a, b)] = phi3e[:, a, b], the nine terms of the old
        #     loop. Strategy item 4: the second, fifth and eighth are the
        #     fourth, third and ninth with a and b swapped
        t2 = sum_pi(NabU, wEU)   # ((eib @ ub2) @ (Niac @ ua1))
        t5 = sum_pi(eabU, wNU)   # ((Nib @ ub2) @ (eiac @ ua1))
        t8 = sum_pi(MabU, wKU)   # ((kib @ ub2) @ (Miac @ ua1))
        phi3e = (sum_pi(eia, wNE)   # (((Niab @ ub2) @ ua1) @ eic)
                 + t2
                 + T(t5)            # ((Nia @ ua1) @ (eibc @ ub2))
                 + T(t2)            # ((eia @ ua1) @ (Nibc @ ub2))
                 + t5
                 + sum_pi(Nia, wE)  # (((eiab @ ub2) @ ua1) @ Nic)
                 + sum_pi(kia, wME) # (((Miab @ ub2) @ ua1) @ kic)
                 + t8
                 + T(t8))           # ((kia @ ua1) @ (Mibc @ ub2))

        #NOTE 1/2 <N[L2(ua, ub)], L11(u0_dot, .)>, the constant of the
        #     orthogonality conditions, which equals the first, and the last,
        #     of the six terms of phi30e
        cste = sum_pi(eia0, wNE)
        t2 = sum_pi(NabU, wE0U)  # es('iac,ib,a,b', Niac, eib0, ua1, ub2)
        t5 = sum_pi(eabU, wN0U)  # es('ib,iac,a,b', Nib0, eiac, ua1, ub2)
        phi30e = (cste               # es('iab,ic,a,b', Niab, eic0, ua1, ub2)
                  + t2
                  + T(t5)            # es('ia,ibc,a,b', Nia0, eibc, ua1, ub2)
                  + T(t2)            # es('ibc,ia,a,b', Nibc, eia0, ua1, ub2)
                  + t5
                  + sum_pi(Nia0, wE))# es('ic,iab,a,b', Nic0, eiab, ua1, ub2)

        t2 = sum_pi(wN0U, E0U)   # es('ia,ib,a,b', Nia0, eib0, ua1, ub2)
        phi200 += (sum_pi(w[:, None]*ei00, NE)  # es('iab,i,a,b', Niab, ei00, ua1, ub2)
                   + 2*t2
                   + 2*T(t2)                    # es('ib,ia,a,b', Nib0, eia0, ua1, ub2)
                   + sum_pi(w[:, None]*Ni00, E))# es('i,iab,a,b', Ni00, eiab, ua1, ub2)

        #NOTE strategy item 5
        Pabcd += sum_pi(wNE, E)

        #NOTE scatter, strategy item 6: the 40 DOFs of an element are
        #     distinct, so the fancy indexed additions do not lose repeated
        #     indices
        phi20[indices] += phi20e
        phi3[indices] += phi3e
        phi30[indices] += phi30e
        cst[indices] += cste

    #NOTE the six terms of fphi4(ua, ub, uc, ud) of the old loop,
    #       ((Niab @ ub) @ ua) @ ((eicd @ ud) @ uc) = P[a, b, c, d]
    #       ((Niac @ uc) @ ua) @ ((eibd @ ud) @ ub) = P[a, c, b, d]
    #       ((Niad @ ud) @ ua) @ ((eibc @ uc) @ ub) = P[a, d, b, c]
    #       ((Nibc @ uc) @ ub) @ ((eiad @ ud) @ ua) = P[b, c, a, d]
    #       ((Nibd @ ud) @ ub) @ ((eiac @ uc) @ ua) = P[b, d, a, c]
    #       ((Nicd @ ud) @ uc) @ ((eiab @ ub) @ ua) = P[c, d, a, b]
    #     every one of them linear in the integrand, so they are permuted
    #     once here rather than at every integration point. With Aij
    #     symmetric P[a, b, c, d] = P[c, d, a, b] and the six terms are three
    #     equal pairs; all six are kept, which costs nothing and does not rely
    #     on it
    phi4 = (Pabcd
            + Pabcd.transpose(0, 2, 1, 3)
            + Pabcd.transpose(0, 2, 3, 1)
            + Pabcd.transpose(2, 0, 1, 3)
            + Pabcd.transpose(2, 0, 3, 1)
            + Pabcd.transpose(2, 3, 0, 1))

    return phi20, phi3, phi30, cst, phi200, phi4


def a_coefficients(phi3U, lam, d):
    """a_ijk of every i, j and k

    a_ijk = -1/(2 lambda_i) (phi3_ij . u_k)/(phi20_i . u_i)

    Parameters
    ----------
    phi3U : (m, m, m) array
        phi3U[i, j, k] = phi3_ab[(i, j)] @ ua[k].
    lam : (m,) array
        lambda_i.
    d : (m,) array
        d[i] = phi20_a[i] @ ua[i].
    """
    return -1./(2*lam[:, None, None])*phi3U/d[:, None, None]


def b_coefficients(phi4, phi3uab, phi30U, phi200, a, lam, d):
    """b_ijkl of every i, j, k and l, as the models have always computed them

        b_ijkl = -1/(6 lambda_i d_i) (phi4_ijkl
                 + 3 phi3_ij.u_kl + 3 phi3_il.u_jk
                 + lambda_i (a_iij phi30_ik.u_l + a_ijk phi30_il.u_i
                             + a_ikl phi30_ii.u_j)
                 + phi200_ii lambda_i**2 (a_iij a_ikl + a_ijk a_ili
                                          + a_ikl a_iij))

    with d_i = phi20_i . u_i. Every term is written out below as a
    broadcast over the four indices, in the order of the formula. The
    formula is not symmetric in j, k and l, nor in i, see "Known limitations"
    in doc/nlprebuck_implementation.tex and
    doc/verification/multimode_b_symmetry.py; it is reproduced, not changed.

    Parameters
    ----------
    phi4 : (m, m, m, m) array
    phi3uab : (m, m, m, m) array
        phi3uab[i, j, k, l] = phi3_ab[(i, j)] @ uab[(k, l)].
    phi30U : (m, m, m) array
        phi30U[i, k, l] = phi30_ab[(i, k)] @ ua[l].
    phi200 : (m, m) array
    a : (m, m, m) array
        a_ijk, a_coefficients.
    lam, d : (m,) arrays
        As in a_coefficients.
    """
    m = lam.shape[0]
    i = np.arange(m)
    aii = a[i, i, :]              # aii[i, j] = a_iij
    ali = a[i, :, i]              # ali[i, l] = a_ili
    P30li = phi30U[i, :, i]       # P30li[i, l] = phi30_il . u_i
    P30ii = phi30U[i, i, :]       # P30ii[i, j] = phi30_ii . u_j
    li = lam[:, None, None, None]
    return -1/(6*li*d[:, None, None, None])*(
            phi4
            + 3*phi3uab                                  # 3 phi3_ij . u_kl
            + 3*phi3uab.transpose(0, 2, 3, 1)            # 3 phi3_il . u_jk
            + li*(
                aii[:, :, None, None]*phi30U[:, None, :, :]  # a_iij phi30_ik.u_l
               + a[:, :, :, None]*P30li[:, None, None, :]    # a_ijk phi30_il.u_i
               + a[:, None, :, :]*P30ii[:, :, None, None]    # a_ikl phi30_ii.u_j
                )
            + phi200[i, i][:, None, None, None]*li**2*(
                aii[:, :, None, None]*a[:, None, :, :]       # a_iij a_ikl
               + a[:, :, :, None]*ali[:, None, None, :]      # a_ijk a_ili
               + a[:, None, :, :]*aii[:, :, None, None]      # a_ikl a_iij
                )
            )
