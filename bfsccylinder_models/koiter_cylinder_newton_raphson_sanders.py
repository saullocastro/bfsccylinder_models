import gc
from functools import partial
from collections import defaultdict

try:
    from pypardiso import spsolve
except ImportError:
    from scipy.sparse.linalg import spsolve

import numpy as np
from numpy import isclose
from scipy.sparse import coo_matrix, bmat, csc_matrix
from scipy.sparse.linalg import eigsh
from bfsccylinder.sanders import (BFSCCylinderSanders, update_KC0, update_KCNL,
        update_KG, update_fint, DOF, DOUBLE, INT, KC0_SPARSE_SIZE,
        KCNL_SPARSE_SIZE, KG_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD

num_nodes = 4


def fkoiter_cyl_SS3(L, R, nx, ny, prop, cg_x0=None, nint=4,
        num_eigvals=2, koiter_num_modes=1, Nxxunit=1., NLprebuck=False,
        NLprebuck_eps1=0.005, NLprebuck_maxiter=12, NR_maxiter=40,
        NR_eps=1.e-4, NR_eps_accept=1.e-3):

    circ = 2*np.pi*R
    out = {}

    out['nx'] = nx
    out['ny'] = ny
    nids = 1 + np.arange(nx*(ny+1))
    nids_mesh = nids.reshape(nx, ny+1)
    # closing the cylinder by reassigning last row of node-ids
    nids_mesh[:, nids_mesh.shape[1]-1] = nids_mesh[:, 0]
    nids = np.unique(nids_mesh)
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    out['nid_pos'] = nid_pos

    xlin = np.linspace(0, L, nx)
    ytmp = np.linspace(0, circ, ny+1)
    ylin = np.linspace(0, circ-(ytmp[ytmp.shape[0]-1] - ytmp[ytmp.shape[0]-2]), ny)
    xmesh, ymesh = np.meshgrid(xlin, ylin)
    xmesh = xmesh.T
    ymesh = ymesh.T

    # getting nodes
    ncoords = np.vstack((xmesh.flatten(), ymesh.flatten(), np.zeros_like(xmesh.flatten()))).T
    x = ncoords[:, 0]
    y = ncoords[:, 1]
    out['ncoords'] = ncoords
    out['x'] = x
    out['y'] = y

    i = nids_mesh.shape[0] - 1
    j = nids_mesh.shape[1] - 1
    n1s = nids_mesh[:i, :j].flatten()
    n2s = nids_mesh[1:, :j].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:i, 1:].flatten()
    out['n1s'] = n1s
    out['n2s'] = n2s
    out['n3s'] = n3s
    out['n4s'] = n4s

    points, weights = get_points_weights(nint=nint)

    num_elements = len(n1s)
    print('# nx', nx)
    print('# ny', ny)
    print('# number of elements', num_elements)

    elements = []
    N = DOF*nx*ny
    print('# numbers of DOF', N)
    init_k_KC0 = 0
    init_k_KCNL = 0
    init_k_KG = 0
    print('# starting element assembly')
    volume = 0
    mass = 0
    havg = prop.h # average shell thickness h
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        elem = BFSCCylinderSanders(nint)
        elem.n1 = n1
        elem.n2 = n2
        elem.n3 = n3
        elem.n4 = n4
        elem.c1 = DOF*nid_pos[n1]
        elem.c2 = DOF*nid_pos[n2]
        elem.c3 = DOF*nid_pos[n3]
        elem.c4 = DOF*nid_pos[n4]
        elem.R = R
        x1 = x[nid_pos[n1]]
        x2 = x[nid_pos[n2]]
        elem.lex = x2 - x1
        elem.ley = circ/ny
        volume += elem.lex*elem.ley*prop.h
        mass += elem.lex*elem.ley*prop.intrho
        assign_constant_ABD(elem, prop)
        elem.init_k_KC0 = init_k_KC0
        elem.init_k_KCNL = init_k_KCNL
        elem.init_k_KG = init_k_KG
        init_k_KC0 += KC0_SPARSE_SIZE
        init_k_KCNL += KCNL_SPARSE_SIZE
        init_k_KG += KG_SPARSE_SIZE
        elements.append(elem)

    out['volume'] = volume
    out['mass'] = mass
    out['havg'] = havg
    KC0r = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for elem in elements:
        update_KC0(elem, points, weights, KC0r, KC0c, KC0v)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    del KC0v, KC0r, KC0c
    gc.collect()

    print('# finished element assembly')

    # applying boundary conditions
    bk = np.zeros(N, dtype=bool)

    checkSS = isclose(x, 0) | isclose(x, L)
    bk[3::DOF] = checkSS
    bk[6::DOF] = checkSS
    check = isclose(x, L/2.) & isclose(y, 0)
    assert check.sum() == 1
    bk[0::DOF] = check
    bu = ~bk # same as np.logical_not, defining unknown DOFs
    u0 = np.zeros(N, dtype=DOUBLE)

    print('# starting static analysis')

    # axially compressive load applied at x=0 and x=L
    fext = np.zeros(N)
    # applying load
    for elem in elements:
        pos1 = nid_pos[elem.n1]
        pos2 = nid_pos[elem.n2]
        pos3 = nid_pos[elem.n3]
        pos4 = nid_pos[elem.n4]
        if isclose(x[pos3], L):
            Nxx = -Nxxunit
            xi = +1
        elif isclose(x[pos1], 0):
            Nxx = +Nxxunit
            xi = -1
        else:
            continue
        lex = elem.lex
        ley = elem.ley
        indices = []
        c1 = DOF*pos1
        c2 = DOF*pos2
        c3 = DOF*pos3
        c4 = DOF*pos4
        cs = [c1, c2, c3, c4]
        for ci in cs:
            for i in range(DOF):
                indices.append(ci + i)
        fe = np.zeros(num_nodes*DOF, dtype=float)
        for j in range(nint):
            eta = points[j]
            elem.update_Su(xi, eta)
            fe += ley/2.*weights[j]*elem.Su*Nxx
        fext[indices] += fe
    assert isclose(fext.sum(), 0)

    # sub-matrices corresponding to unknown DOFs
    KC0uu = KC0[bu, :][:, bu]

    KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)

    # solving
    uu = spsolve(KC0uu, fext[bu])
    cg_x0 = uu.copy()

    u0[bu] = uu
    u0_lin = u0.copy()

    def assemble_KG(u):
        KGv[:] = 0
        for elem in elements:
            update_KG(u, elem, points, weights, KGr, KGc, KGv)
        return coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()

    def solve_eig(KCuu, KGuu):
        """Buckling multipliers of the stress state currently stored in KG

        The starting vector is fixed on purpose. ARPACK keeps its random seed
        in a SAVEd variable, so without one the basis it returns for a
        degenerate eigenspace, and the buckling modes of a cylinder are
        degenerate pairs, depends on how many eigenvalue problems were solved
        before in the same process. The second order field is obtained by
        deflating that basis, so b_ijkl would otherwise change with the order
        in which the analyses run.
        """
        v0 = np.random.default_rng(0).random(KCuu.shape[0])
        eigvals, eigvecsu = eigsh(A=KGuu, k=num_eigvals, which='LM', M=KCuu,
                tol=1e-6, v0=v0)
        return eigvals, eigvecsu, -1/eigvals

    def bordered_solve(Auu, rhs, Q):
        """Solve Auu x = rhs with x orthogonal to the columns of Q

        Used wherever Auu is singular, or nearly so, along Q. The bordered
        system is non singular there and the Lagrange multipliers absorb the
        component of rhs that lies in that space.
        """
        if Q is None or Q.shape[1] == 0:
            return spsolve(Auu, rhs)
        n = Auu.shape[0]
        Qs = csc_matrix(Q)
        A = bmat([[Auu, Qs], [Qs.T, None]], format='csc')
        b = np.zeros(n + Q.shape[1])
        b[:n] = rhs
        return spsolve(A, b)[:n]

    if NLprebuck:
        print('#    initiating nonlinear pre-buckling state')
        KCNLr = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=INT)
        KCNLc = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=INT)
        KCNLv = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=DOUBLE)

        def assemble_KCNL(u):
            KCNLv[:] = 0
            for elem in elements:
                update_KCNL(u, elem, points, weights, KCNLr, KCNLc, KCNLv)
            return coo_matrix((KCNLv, (KCNLr, KCNLc)), shape=(N, N)).tocsc()

        def calc_KT(u):
            return KC0 + assemble_KCNL(u) + assemble_KG(u)

        def calc_fint(u, fint):
            fint *= 0
            for elem in elements:
                update_fint(u, elem, points, weights, fint)
            return fint

        # solving using Modified Newton-Raphson method
        def scaling(vec, D):
            """
                A. Peano and R. Riccioni, Automated discretisatton error
                control in finite element analysis. In Finite Elements m
                the Commercial Enviror&ent (Editei by J. 26.  Robinson),
                pp. 368-387. Robinson & Assoc., Verwood.  England (1978)
            """
            return np.sqrt((vec*np.abs(1/D))@vec)

        D = KC0uu.diagonal() # at beginning of load increment
        epsilon = NR_eps

        def solve_prebuckling(lambda_b, ui, Qdefl=None):
            """Pre-buckling state at the load level lambda_b*Nxxunit

            Qdefl holds the buckling modes estimated so far. The fundamental
            path of a perfect cylinder under axial compression has no
            component along them, they are not axisymmetric, but close to the
            bifurcation point the tangent stiffness matrix is nearly singular
            in those directions, so any round off component of the residual
            there is amplified and the iteration diverges. Deflating them
            keeps the correction on the fundamental path.
            """
            fext_b = lambda_b*fext
            fint = np.zeros(N)
            fint = calc_fint(ui, fint)
            Ri = fint - fext_b
            du = np.zeros(N)
            u = ui.copy()
            KT = calc_KT(ui)
            KTuu = KT[bu, :][:, bu]
            iteration = 0
            best_test = np.inf
            u_best = ui.copy()
            stall = 0

            def give_up(reason):
                #NOTE close to the bifurcation point the residual of the
                #     deflated iteration reaches a floor above NR_eps and then
                #     starts growing again. The best iterate is still a
                #     perfectly usable equilibrium state whenever that floor
                #     is below NR_eps_accept, so it is kept instead of
                #     throwing the load step away
                if best_test < NR_eps_accept:
                    print('#        %s, keeping the best iterate, '
                            'crisfield_test %r' % (reason, best_test))
                    return u_best, calc_KT(u_best)
                raise RuntimeError('Newton-Raphson %s at lambda_b=%r, the '
                        'pre-buckling state is probably too close to the '
                        'bifurcation point' % (reason, lambda_b))

            while True:
                duu = bordered_solve(KTuu, -Ri[bu], Qdefl)
                du[bu] = duu
                u = ui + du
                fint = calc_fint(u, fint)
                Ri = fint - fext_b
                crisfield_test = scaling(Ri[bu], D)/max(
                        scaling(fext_b[bu], D), scaling(fint[bu], D))
                print('#        iteration', iteration, 'crisfield_test',
                        crisfield_test)
                if crisfield_test < epsilon:
                    return u, calc_KT(u)
                if crisfield_test < best_test:
                    best_test = crisfield_test
                    u_best = u.copy()
                    stall = 0
                else:
                    stall += 1
                    if stall >= 3:
                        return give_up('stalled')
                #NOTE bailing out as soon as the iteration is clearly running
                #     away, so that the load stepping can back off without
                #     spending NR_maxiter iterations first
                if crisfield_test > 0.5:
                    return give_up('diverged')
                iteration += 1
                if iteration > NR_maxiter:
                    return give_up('did not converge')
                KT = calc_KT(u)
                KTuu = KT[bu, :][:, bu]
                ui = u.copy()

        #NOTE Iterative eigenvalue algorithm of Sun et al. (2020), Eqs. (44)
        #     to (46), Step 1 of their Fig. 2. The asymptotic expansion is
        #     only valid about a pre-buckling state in the neighbourhood of
        #     the bifurcation point, lambda_b/lambda_c approximately 0.995.
        #     Expanding about the reference load Nxxunit instead leaves
        #     lambda_b/lambda_c of the order of 0.1, where the nonlinear
        #     pre-buckling deformation caused by the edge restraint has not
        #     developed yet, and gives back the membrane pre-buckling result
        def calc_u0dot(KT, Qdefl=None):
            """Eq. (29), rate of the pre-buckling state with respect to the
            load parameter. Assuming instead that the pre-buckling path is
            linear in lambda, u0(lambda) = lambda*u0, would make this equal
            to the pre-buckling state itself. The buckling modes are deflated
            for the same reason as in solve_prebuckling"""
            u0dot = np.zeros(N, dtype=DOUBLE)
            u0dot[bu] = bordered_solve(KT[bu, :][:, bu], fext[bu], Qdefl)
            return u0dot

        lambda_b = 1.
        u0 = u0_lin.copy()
        KT = None
        #NOTE the previous converged load step, kept for the backward
        #     difference of Eq. (35)
        lambda_prev = None
        u0dot_prev = None
        KC = KC0 + assemble_KCNL(u0)
        KG = assemble_KG(u0)
        print('# starting iterative eigenvalue analysis')
        converged = False
        for iteration in range(1, NLprebuck_maxiter+1):
            KCuu = KC[bu, :][:, bu]
            KGuu = KG[bu, :][:, bu]
            eigvals, eigvecsu, mu = solve_eig(KCuu, KGuu)
            #NOTE the near critical eigenvectors, deflated from every solve
            #     with the tangent stiffness matrix from here on
            crit = [j for j in range(num_eigvals)
                    if abs(mu[j] - mu[0]) <= 1.e-2*abs(mu[0])]
            Qdefl = np.linalg.qr(eigvecsu[:, crit])[0]
            lambda_c = lambda_b*mu[0] # Eq. (46)
            print('#    iteration', iteration, 'lambda_b', lambda_b,
                    'lambda_c', lambda_c, 'lambda_b/lambda_c',
                    lambda_b/lambda_c)
            if (lambda_c - lambda_b)/lambda_c <= NLprebuck_eps1: # Eq. (45)
                print('#    converged')
                converged = True
                break
            #NOTE eta=0.8 in the first iteration to approach the neighbourhood
            #     of the buckling load quickly, eta=0.5 afterwards to avoid
            #     overshooting it and getting a negative eigenvalue
            eta = 0.8 if iteration == 1 else 0.5
            #NOTE the load stepping is load controlled, so the last steps take
            #     the state very close to a singular tangent stiffness matrix.
            #     A tangent predictor keeps the Newton-Raphson in its
            #     convergence radius there, and the step is halved whenever it
            #     is not enough
            while True:
                lambda_b_new = lambda_b + eta*(lambda_c - lambda_b) # Eq. (44)
                if KT is None:
                    guess = u0*(lambda_b_new/lambda_b)
                else:
                    guess = u0 + calc_u0dot(KT, Qdefl)*(lambda_b_new
                            - lambda_b)
                try:
                    u0_new, KT_new = solve_prebuckling(lambda_b_new, guess,
                            Qdefl)
                    break
                except RuntimeError:
                    eta *= 0.5
                    print('#    Newton-Raphson failed, backing off to eta',
                            eta)
                    if eta < 1.e-2:
                        u0_new = KT_new = None
                        break
            if u0_new is None:
                #NOTE no further progress possible with load control
                break
            lambda_prev = lambda_b
            u0dot_prev = u0_lin if KT is None else calc_u0dot(KT, Qdefl)
            u0, KT = u0_new, KT_new
            lambda_b = lambda_b_new
            KC = KC0 + assemble_KCNL(u0)
            KG = assemble_KG(u0)
        #NOTE the load stepping is load controlled, so it cannot always be
        #     pushed all the way to 1 - NLprebuck_eps1. The state reached is
        #     still far better than the reference load one, so it is kept and
        #     the ratio actually achieved is reported, rather than throwing
        #     the whole analysis away
        if not converged:
            print('# WARNING: the iterative eigenvalue algorithm stopped at '
                    'lambda_b/lambda_c = %r, short of the %r requested'
                    % (lambda_b/lambda_c, 1 - NLprebuck_eps1))

        if KT is None:
            KT = calc_KT(u0)
        u0dot = calc_u0dot(KT, Qdefl)

        #NOTE Eqs. (34) and (35), second derivative by a backward difference
        #     against the load step preceding the converged one, which is what
        #     Eq. (35) prescribes. Reusing a step the load stepping already
        #     converged on costs nothing and, unlike an extra step solved
        #     ahead of lambda_b, cannot fall on the far side of the
        #     bifurcation point where the Newton-Raphson no longer converges
        if lambda_prev is None:
            u0ddot = np.zeros(N, dtype=DOUBLE)
        else:
            u0ddot = (u0dot - u0dot_prev)/(lambda_b - lambda_prev)

        del KCNLv, KCNLr, KCNLc
        gc.collect()

    else:
        lambda_b = 1.
        KC = KC0
        KCuu = KC0uu
        KG = assemble_KG(u0)
        KGuu = KG[bu, :][:, bu]
        print('# starting eigenvalue analysis')
        eigvals, eigvecsu, mu = solve_eig(KCuu, KGuu)
        lambda_c = lambda_b*mu[0]
        #NOTE a linear pre-buckling state is exactly linear in lambda
        u0dot = u0.copy()
        u0ddot = np.zeros(N, dtype=DOUBLE)

    print('# finished eigenvalue analysis')
    print('# finished static analysis')

    #NOTE u0 is the pre-buckling state about which the expansion is made and
    #     mu the buckling multipliers of that state, so that the load factors
    #     with respect to Nxxunit are lambda_b*mu. mu[0] equals 1 within
    #     NLprebuck_eps1 once the iterative eigenvalue algorithm converged
    load_mult = lambda_b*mu

    Pcr = load_mult[0]*Nxxunit*circ
    print('# load_mult', load_mult)
    print('# critical buckling load', Pcr)

    out['Pcr'] = Pcr
    out['cg_x0'] = cg_x0
    out['eigvals'] = eigvals
    out['load_mult'] = load_mult
    out['lambda_b'] = lambda_b
    out['mu'] = mu
    eigvecs = np.zeros((N, num_eigvals))
    eigvecs[bu, :] = eigvecsu
    out['eigvecs'] = eigvecs
    out['koiter'] = None

    if koiter_num_modes == 0:
        return out

    lambda_a = {}
    for modei in range(koiter_num_modes):
        lambda_a[modei] = load_mult[modei]

    es = partial(np.einsum, optimize='greedy', casting='no')
    #from opt_einsum import contract
    #es = partial(contract)

    #NOTE making the maximum amplitude of the eigenmode equal to h
    #normalizing amplitude of eigenvector according to shell thickness
    ua = {}
    for modei in range(koiter_num_modes):
        ua[modei] = eigvecs[:, modei].copy()
        #NOTE normalizing as Abaqus does, assuming nonzero translations
        ampl = np.sqrt(ua[modei][0::DOF]**2 + ua[modei][3::DOF]**2 + ua[modei][6::DOF]**2).max()
        #NOTE using ampl = np.linalg.norm(ua[modei]) does not work
        ua[modei] /= ampl
        ua[modei] *= havg

    phi4 = defaultdict(lambda: 0)
    phi3_ab = {}
    phi30_ab = {}
    phi3e_ab = {}
    phi30e_ab = {}
    phi20e_a = {}
    phi20_a = {}
    #phi2 = np.zeros((N, N))
    phi200_ab = {}
    for modei in range(koiter_num_modes):
        phi20_a[modei] = np.zeros(N)
        phi20e_a[modei] = np.zeros(num_nodes*DOF)
        for modej in range(koiter_num_modes):
            phi200_ab[(modei, modej)] = 0
            phi3_ab[(modei, modej)] = np.zeros(N)
            phi30_ab[(modei, modej)] = np.zeros(N)
            phi3e_ab[(modei, modej)] = np.zeros(num_nodes*DOF)
            phi30e_ab[(modei, modej)] = np.zeros(num_nodes*DOF)


    #NOTE this flag multiplies every nonlinear contribution to the
    #     pre-buckling strains and to their derivatives. With NLprebuck=False
    #     the pre-buckling state is the linear elastic solution, the
    #     pre-buckling path is exactly linear in the load parameter, and those
    #     contributions have to be absent; with NLprebuck=True they are the
    #     terms that carry the nonlinear pre-buckling behaviour
    flag = NLprebuck

    # higher-order tensors for elements

    u0e = np.zeros(num_nodes*DOF, dtype=np.float64)
    u0dote = np.zeros(num_nodes*DOF, dtype=np.float64)
    u0ddote = np.zeros(num_nodes*DOF, dtype=np.float64)
    Aij = prop.A
    Bij = prop.B
    #Dij = prop.D
    for count, elem in enumerate(elements):
        if count % (num_elements//5) == 0:
            print('#    count', count+1, num_elements)
        eiab = np.zeros((3, num_nodes*DOF, num_nodes*DOF))

        c1 = elem.c1
        c2 = elem.c2
        c3 = elem.c3
        c4 = elem.c4

        u0e *= 0
        u0dote *= 0
        u0ddote *= 0
        for i in range(DOF):
            u0e[0*DOF + i] = u0[c1 + i]
            u0e[1*DOF + i] = u0[c2 + i]
            u0e[2*DOF + i] = u0[c3 + i]
            u0e[3*DOF + i] = u0[c4 + i]
            u0dote[0*DOF + i] = u0dot[c1 + i]
            u0dote[1*DOF + i] = u0dot[c2 + i]
            u0dote[2*DOF + i] = u0dot[c3 + i]
            u0dote[3*DOF + i] = u0dot[c4 + i]
            u0ddote[0*DOF + i] = u0ddot[c1 + i]
            u0ddote[1*DOF + i] = u0ddot[c2 + i]
            u0ddote[2*DOF + i] = u0ddot[c3 + i]
            u0ddote[3*DOF + i] = u0ddot[c4 + i]

        uae = {}
        for modei in range(koiter_num_modes):
            uae[modei] = np.zeros(num_nodes*DOF, dtype=np.float64)
            for i in range(DOF):
                uae[modei][0*DOF + i] = ua[modei][c1 + i]
                uae[modei][1*DOF + i] = ua[modei][c2 + i]
                uae[modei][2*DOF + i] = ua[modei][c3 + i]
                uae[modei][3*DOF + i] = ua[modei][c4 + i]

        ube = uce = ude = uae

        indices = []
        cs = [c1, c2, c3, c4]
        for ci in cs:
            for i in range(DOF):
                indices.append(ci + i)

        lex = elem.lex
        ley = elem.ley

        for modei in range(koiter_num_modes):
            phi20e_a[modei] *= 0
            for modej in range(koiter_num_modes):
                phi3e_ab[(modei, modej)] *= 0
                phi30e_ab[(modei, modej)] *= 0

        #phi2e = np.zeros((num_nodes*DOF, num_nodes*DOF))

        for i in range(nint):
            xi = points[i]
            weight_xi = weights[i]
            for j in range(nint):
                eta = points[j]
                weight_eta = weights[j]
                weight = weight_xi * weight_eta

                elem.update_Sw_x(xi, eta)
                elem.update_Sw_y(xi, eta)
                elem.update_Sv(xi, eta)
                elem.update_Bm(xi, eta)
                elem.update_Bb(xi, eta)

                Sw_x = np.atleast_2d(elem.Sw_x)
                Sw_y = np.atleast_2d(elem.Sw_y)
                Sv = np.atleast_2d(elem.Sv)

                #NOTE Sanders kinematics, Eqs. 39-43 of Castro and Jansen
                #     (AIAA SciTech 2022). Every nonlinear term of the Sanders
                #     membrane strains is recovered from the von Karman one by
                #     the substitution (Sw_x, Sw_y) -> (G1, G2), with
                #     G1 = Sw_x and G2 = Sw_y - Sv/R, because
                #       eps_yy^NL  = 1/2 (w,y - v/R)**2
                #       gamma_xy^NL = w,x (w,y - v/R)
                #     Using (Sw_x, Sw_y) here would silently fall back to von
                #     Karman kinematics in phi''', phi^iv and phi_dot''.
                G1 = Sw_x
                G2 = Sw_y - Sv/R

                #NOTE the pre-buckling STATE and its first and second RATES
                #     with respect to the load parameter are independent
                #     fields. Writing g1_s = lambda*g1_d, as an exactly linear
                #     pre-buckling path would allow, is what makes the
                #     nonlinear pre-buckling behaviour disappear
                g1_s = G1[0] @ u0e # w0,x
                g2_s = G2[0] @ u0e # w0,y - v0/R
                g1_d = G1[0] @ u0dote
                g2_d = G2[0] @ u0dote
                g1_dd = G1[0] @ u0ddote
                g2_dd = G2[0] @ u0ddote

                Bm = np.asarray(elem.Bm)
                Bb = np.asarray(elem.Bb)

                #NOTE eps_dot, the first derivative of the pre-buckling strain
                #     with respect to the load parameter, d/dl of
                #     Bm u + [g1**2/2, g2**2/2, g1 g2]
                ei0 = ej0 = Bm @ u0dote + flag*np.array([g1_s*g1_d,
                                                         g2_s*g2_d,
                                                         g1_s*g2_d + g2_s*g1_d])
                ki0 = kj0 = Bb @ u0dote

                #NOTE eps_dot_dot, the second derivative
                ei00 = ej00 = Bm @ u0ddote + flag*np.array([
                        g1_d**2 + g1_s*g1_dd,
                        g2_d**2 + g2_s*g2_dd,
                        2*g1_d*g2_d + g1_s*g2_dd + g2_s*g1_dd])
                ki00 = kj00 = Bb @ u0ddote

                Ni0 = Aij@ej0 + Bij@kj0
                Ni00 = Aij@ej00 + Bij@kj00

                #NOTE d(eps)/d(u_a) at the pre-buckling state
                eia = eib = eic = Bm + flag*np.array([g1_s*G1[0],
                                                      g2_s*G2[0],
                                                      g1_s*G2[0] + g2_s*G1[0]])

                kia = kib = kic = Bb

                Nia = Nib = Nic = es('ij,ja->ia', Aij, eia) + es('ij,ja->ia', Bij, kia)
                #Mia = Mib = es('ij,ja->ia', Bij, eia) + es('ij,ja->ia', Dij, kia)

                #NOTE d2(eps)/dl d(u_a)
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

                #phi2e += 1/2.*weight*(lex*ley/4.)*(
                           #  es('iab,i->ab', Niab, ei) #NOTE this is KG
                           #+ es('ia,ib->ab', Nia, eib)
                           #+ es('ib,ia->ab', Nib, eia)
                           #+ es('i,iab->ab', Ni, eiab) #NOTE this is KG
                           #+ es('iab,i->ab', Miab, ki) #NOTE this is KG
                           #+ es('ia,ib->ab', Mia, kib)
                           #+ es('ib,ia->ab', Mib, kia)
                        #)

                for modei in range(koiter_num_modes):
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

        #tmp = np.zeros((N, num_nodes*DOF))
        #tmp[indices] = phi2e
        #phi2[:, indices] += tmp
        for modei in range(koiter_num_modes):
            phi20_a[modei][indices] += phi20e_a[modei]
            for modej in range(koiter_num_modes):
                phi3_ab[(modei, modej)][indices] += phi3e_ab[(modei, modej)]
                phi30_ab[(modei, modej)][indices] += phi30e_ab[(modei, modej)]

    #NOTE phi2 must be the SAME operator whose null vector is the buckling
    #     mode, so it is built from the KC and KG of the eigenvalue analysis,
    #     both evaluated at the converged pre-buckling state, and scaled by
    #     the multiplier mu of that state rather than by the load factor
    #     lambda_c. Rebuilding KC from KCNL(lambda_c*u0) here, while the
    #     eigenproblem used KCNL(u0), leaves phi2 non singular in the
    #     direction of the buckling mode and corrupts the second order field
    phi2 = KC + KG*mu[0]
    phi2uu = KCuu + KGuu*mu[0]

    phi2_ab = {}
    for modei in range(koiter_num_modes):
        left = ua[modei] @ phi2
        for modej in range(koiter_num_modes):
            phi2_ab[(modei, modej)] = left @ ua[modej]

    print('# a_ijk factors')
    a_abc = {}
    for modei in range(koiter_num_modes):
        lambda_i = lambda_a[modei]
        for modej in range(koiter_num_modes):
            for modek in range(koiter_num_modes):
                a_ijk = -1./(2*lambda_i)*(phi3_ab[(modei, modej)] @ ua[modek])/(phi20_a[modei] @ ua[modei])
                a_abc[(modei, modej, modek)] = a_ijk
                print('# $a_%d%d%d$' % (modei+1, modej+1, modek+1), a_ijk)
    force2ndorder_ij = {}
    for modei in range(koiter_num_modes):
        for modej in range(koiter_num_modes):
            #NOTE phi3_ij = phi3_ji even in the asym case
            force2ndorder_ij[(modei, modej)] = -1/2.*phi3_ab[(modei, modej)]
            #NOTE the a_ijk contribution below is kept. For the symmetric
            #     bifurcation of a cylinder under axial compression a_ijk is
            #     zero to within round off, of the order of 1e-5 against a
            #     b_ijkl of order 1, so it changes nothing here; it matters
            #     only for an asymmetric bifurcation
            for modek in range(koiter_num_modes):
                lambda_k = lambda_a[modek]
                a_kij = a_abc[(modek, modei, modej)]
                force2ndorder_ij[(modei, modej)] += (
                        - (1/koiter_num_modes)*a_kij*lambda_k*phi20_a[modek]
                        )

    #NOTE phi2 is singular by construction, the buckling modes span its null
    #     space, so the second order fields cannot be obtained from a plain
    #     spsolve followed by a Gram-Schmidt projection. They come from the
    #     bordered system
    #         [phi2  Nsp] [uab  ]   [force2ndorder]
    #         [W^T   0  ] [alpha] = [    -cst     ]
    #     whose column border Nsp spans the null space of phi2, which makes it
    #     non singular, and whose row border W carries the orthogonality
    #     conditions imposed on the second order fields.
    #
    #     Those conditions are the ones of Sun et al. Eq. (18), in finite
    #     element form Eq. (33),
    #         q_k^T [KD(qb, qb_dot) + KG(sigma_b_dot)] uab
    #             + 1/2 q_k^T BNL^T(q_k) H BNL(q_k) qb_dot = 0
    #     The first term is the load parameter derivative of phi2 contracted
    #     with mode k on one slot, which is exactly phi20_a[k], and the second
    #     is a constant, so the condition is weighted by the stiffness
    #     matrices and inhomogeneous. A Gram-Schmidt projection, or an
    #     Euclidean border Q^T uab = 0, imposes neither
    nu = int(bu.sum())

    #NOTE the buckling modes of a cylinder come in degenerate pairs, one for
    #     each sign of the circumferential wave number, so the null space of
    #     phi2 is spanned by every eigenvector whose multiplier equals the
    #     critical one, not only by the Koiter modes
    cols = [ua[modek][bu] for modek in range(koiter_num_modes)]
    for j in range(num_eigvals):
        if abs(mu[j] - mu[0]) <= 1.e-3*abs(mu[0]):
            cols.append(eigvecsu[:, j])
    Qo, r = np.linalg.qr(np.asarray(cols).T)
    keep = np.abs(np.diag(r)) > 1.e-10*np.abs(np.diag(r)).max()
    Nsp = Qo[:, keep]
    print('# null space of phi2 deflated with %d vectors' % Nsp.shape[1])

    #NOTE one row of Eq. (33) per Koiter mode. Whatever is left of the null
    #     space, the degenerate partners of those modes, is outside the single
    #     mode theory and keeps the Euclidean condition
    W = np.zeros_like(Nsp)
    for modek in range(koiter_num_modes):
        W[:, modek] = phi20_a[modek][bu]
    extra = Nsp.shape[1] - koiter_num_modes
    if extra > 0:
        P = Nsp.copy()
        for modek in range(koiter_num_modes):
            v = ua[modek][bu]
            P = P - np.outer(v, v @ P)/(v @ v)
        W[:, koiter_num_modes:] = np.linalg.svd(
                P, full_matrices=False)[0][:, :extra]

    bordered = bmat([[phi2uu, csc_matrix(Nsp)],
                     [csc_matrix(W).T, None]], format='csc')

    uab = {}
    for modei in range(koiter_num_modes):
        for modej in range(koiter_num_modes):
            rhs = np.zeros(nu + Nsp.shape[1])
            rhs[:nu] = force2ndorder_ij[(modei, modej)][bu]
            for modek in range(koiter_num_modes):
                #NOTE the constant of Eq. (33). With every mode index equal,
                #     the six contributions summed into phi30_ab coincide, so
                #     that with the 1/2 carried by the quadrature weight
                #     phi30_ab @ ua is three times
                #     <N[L2(u1)], L11(u0_dot, u1)>, and Eq. (33) takes one
                #     half of it
                #TODO for koiter_num_modes > 1 the symmetrization of phi30
                #     over distinct modes still has to be worked out, the
                #     expression below is only exact when modei, modej and
                #     modek coincide
                rhs[nu + modek] = -(phi30_ab[(modei, modej)]
                        @ ua[modek])/6.
            sol = spsolve(bordered, rhs)
            uijbar = np.zeros(N)
            uijbar[bu] = sol[:nu]
            uab[(modei, modej)] = uijbar

    print('# b_ijkl factors')
    b_ijkl = {}
    for modei in range(koiter_num_modes):
        phi20_i = phi20_a[modei]
        lambda_i = lambda_a[modei]
        for modej in range(koiter_num_modes):
            for modek in range(koiter_num_modes):
                for model in range(koiter_num_modes):
                    b_ijkl[(modei, modej, modek, model)] = -1/(6*lambda_i*(phi20_i @ ua[modei]))*(
                            phi4[(modei, modej, modek, model)]
                            + 3*(phi3_ab[(modei, modej)] @ uab[(modek, model)])
                            + 3*(phi3_ab[(modei, model)] @ uab[(modej, modek)])
                            + lambda_i*(
                                a_abc[(modei, modei, modej)]*(phi30_ab[(modei, modek)] @ ua[model])
                               +a_abc[(modei, modej, modek)]*(phi30_ab[(modei, model)] @ ua[modei])
                               +a_abc[(modei, modek, model)]*(phi30_ab[(modei, modei)] @ ua[modej])
                                )
                            + phi200_ab[(modei, modei)]*lambda_i**2*(
                                a_abc[(modei, modei, modej)]*a_abc[(modei, modek, model)]
                               +a_abc[(modei, modej, modek)]*a_abc[(modei, model, modei)]
                               +a_abc[(modei, modek, model)]*a_abc[(modei, modei, modej)]
                                )
                            )
                    print('# $b_{%d%d%d%d}$, %f' % (modei+1, modej+1,
                        modek+1, model+1, b_ijkl[(modei, modej, modek, model)]))

    koiter = dict(
        a_ijk=a_abc,
        b_ijkl=b_ijkl,
        koiter_num_modes=koiter_num_modes,
        lambda_i=lambda_a,
        u0=u0,
        ui=ua,
        uij=uab,
            )
    out['koiter'] = koiter

    return out
