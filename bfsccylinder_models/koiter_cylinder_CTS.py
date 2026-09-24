import gc

try:
    from pypardiso import spsolve
except ImportError:
    from scipy.sparse.linalg import spsolve

import numpy as np
from numpy import isclose
from scipy.sparse import coo_matrix, bmat, csc_matrix
from scipy.sparse.linalg import eigsh
from composites import laminated_plate
from bfsccylinder import (BFSCCylinder, update_KC0, update_KCNL, update_KG,
        update_fint, DOF, DOUBLE, INT, KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE,
        KG_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder_models.cyclic_symmetry import (mesh_order,
        axisymmetric_basis, project_axisymmetric, canonical_modes,
        degenerate_partner)
from bfsccylinder_models.koiter_tensors import (koiter_element_tensors,
        a_coefficients, b_coefficients)

num_nodes = 4


def nonlinear_rows(elem, xi, eta):
    """Rows (G1, G2) of the nonlinear membrane strains at (xi, eta)

    eps_xx^NL = 1/2 (G1 u)**2, eps_yy^NL = 1/2 (G2 u)**2 and
    gamma_xy^NL = (G1 u) (G2 u), see koiter_element_tensors. Von Karman
    kinematics
    """
    elem.update_Sw_x(xi, eta)
    elem.update_Sw_y(xi, eta)
    Sw_x = np.asarray(elem.Sw_x)
    Sw_y = np.asarray(elem.Sw_y)
    G1 = Sw_x
    G2 = Sw_y
    return G1, G2


def fkoiter_cylinder_CTS_circum(L, R, rCTS, nxt, ny, E11, E22, nu12, G12, rho,
        h_tow, param_n, c2_ratio, thetadeg_c1, thetadeg_c2,
        ny_nx_aspect_ratio=1, cg_x0=None,
        idealistic_CTS=False, mesh_only=False, nint=4, num_eigvals=2,
        koiter_num_modes=1, Nxxunit=1., NLprebuck=False,
        NLprebuck_eps1=0.005, NLprebuck_maxiter=30, NR_maxiter=40,
        NR_eps=1.e-4, NR_eps_accept=1.e-3,
        max_ny_nx_aspect_ratio=2, zero_offset=False,
        c1_threshold_factor=0.01, c2_threshold_factor=0.01):

    c1_threshold = c1_threshold_factor*L
    c2_threshold = c2_threshold_factor*L
    circ = 2*np.pi*R
    out = {}

    assert nxt >= 2, 'At least two nodes are required in the transition zone.'
    assert thetadeg_c1 >= 0
    assert thetadeg_c2 >= 0

    if param_n == 0 or isclose(thetadeg_c1, thetadeg_c2):
        print('# constant stiffness')
        param_n = 0
        c2 = 0
        c1 = L
        t = 0
        if ny is not None:
            nx = int(ny*L/circ)
            if nx % 2 == 0:
                nx += 1
        else:
            print('# assuming nx=nxt')
            nx = nxt
            ny = int(round(nx*circ/L*ny_nx_aspect_ratio, 0))
        nxc = nx
        nxs = 0
    else:
        t = rCTS*np.sin(abs(np.deg2rad(thetadeg_c2 - thetadeg_c1)))
        nmax = L/(2*t)
        print('# nmax', nmax)
        if param_n > nmax:
            print('# param_n changed from ', param_n)
            print('#                 to   ', int(nmax))
            param_n = int(nmax)
        c2_max = (L - 2*t*param_n)/param_n
        c2 = c2_ratio*c2_max
        if c2 < c2_threshold:
            c2 = 0
            c2_ratio = 0
        c1 = (L - (2*t + c2)*param_n)/(param_n + 1)
        if c1 < c1_threshold:
            c1 = 0
        #NOTE recalculating t to accommodate cases where
        #     c1 < c1_threshold or c2 < c2_threshold
        t = ((L - c1*(param_n+1))/param_n - c2)/2

        nxc = max(2, int(round(c1/t*nxt, 0)))
        nxs = max(2, int(round(c2/t*nxt, 0)))
        dx = t/(nxt-1)
        if ny is None:
            ny = int(round(circ/dx*ny_nx_aspect_ratio, 0))
        dy = circ/ny
        if dy/dx > max_ny_nx_aspect_ratio:
            dxtmp = dy/max_ny_nx_aspect_ratio
            nxc = max(2, int(round(c1/dxtmp, 0)))
            nxs = max(2, int(round(c2/dxtmp, 0)))
    print('# param_t', t)
    print('# param_c1', c1)
    print('# param_c2', c2)
    print('# nxt', nxt)
    print('# nxc', nxc)
    print('# nxs', nxs)
    print('# rCTS', rCTS)
    print('# param_n', param_n)
    print('# c2_ratio', c2_ratio)
    print('# thetadeg_c1', thetadeg_c1)
    print('# thetadeg_c2', thetadeg_c2)
    assert isclose((2*t + c2)*param_n + c1*(param_n+1) - L, 0)
    if np.isclose(c1, 0):
        xlin = []
        thetalin = []
    else:
        ntmp = nxc-1
        if isclose(c1/2, L/2) and (nxc % 2) != 0:
            ntmp += 1
        if param_n == 0:
            endpoint = True
        else:
            endpoint = False
        xlin = np.linspace(0, c1, ntmp, endpoint=endpoint)
        thetalin = np.ones(ntmp)*thetadeg_c1
    for i in range(param_n):
        start = c1 + i*(c1 + 2*t + c2)
        xlin = np.concatenate((xlin, np.linspace(start, start+t, nxt-1, endpoint=False)))
        thetalin = np.concatenate((thetalin, thetadeg_c1 + np.linspace(0, 1, nxt-1, endpoint=False)*(thetadeg_c2 - thetadeg_c1)))
        if not isclose(c2, 0):
            #NOTE to keep always a node in the middle of the cylinder
            ntmp = nxs-1
            if isclose(0.5*(start+t) + 0.5*(start+t+c2), L/2) and (nxs % 2) == 0:
                ntmp += 1
            xlin = np.concatenate((xlin, np.linspace(start+t, start+t+c2, ntmp, endpoint=False)))
            thetalin = np.concatenate((thetalin, np.ones(ntmp)*thetadeg_c2))
        if i == param_n-1 and np.isclose(c1, 0):
            xlin = np.concatenate((xlin, np.linspace(start+t+c2, start+t+c2+t, nxt, endpoint=True)))
            thetalin = np.concatenate((thetalin, thetadeg_c2 + np.linspace(0, 1, nxt, endpoint=True)*(thetadeg_c1 - thetadeg_c2)))
        else:
            xlin = np.concatenate((xlin, np.linspace(start+t+c2, start+t+c2+t, nxt-1, endpoint=False)))
            thetalin = np.concatenate((thetalin, thetadeg_c2 + np.linspace(0, 1, nxt-1, endpoint=False)*(thetadeg_c1 - thetadeg_c2)))
        if i == param_n-1:
            endpoint = True
            neff = nxc
        else:
            endpoint = False
            neff = nxc-1
        if not np.isclose(c1, 0):
            ntmp = neff
            if isclose(0.5*(start+t+c2+t) + 0.5*(start+t+c2+t+c1), L/2) and (nxc % 2) == 0:
                ntmp += 1
            xlin = np.concatenate((xlin, np.linspace(start+t+c2+t, start+t+c2+t+c1, ntmp, endpoint=endpoint)))
            thetalin = np.concatenate((thetalin, np.ones(ntmp)*thetadeg_c1))

    assert np.isclose(xlin.min(), 0)
    assert np.isclose(xlin.max(), L)
    nx = xlin.shape[0]
    out['nx'] = nx
    out['ny'] = ny
    out['xlin'] = xlin
    out['thetalin'] = thetalin
    nids = 1 + np.arange(nx*(ny+1))
    nids_mesh = nids.reshape(nx, ny+1)
    # closing the cylinder by reassigning last row of node-ids
    nids_mesh[:, nids_mesh.shape[1]-1] = nids_mesh[:, 0]
    nids = np.unique(nids_mesh)
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    out['nid_pos'] = nid_pos

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
    connectivity = []
    N = DOF*nx*ny
    print('# numbers of DOF', N)
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    init_k_KC0 = 0
    init_k_KCNL = 0
    init_k_KG = 0
    print('# starting element assembly')
    volume = 0
    mass = 0
    thetadegavg_elements = []
    havg_elements = []
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        elem = BFSCCylinder(nint)
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
        havg_elem = 0
        thetadegavg_elem = 0
        for i in range(nint):
            wi = weights[i]
            xi = points[i]
            xlocal = x1 + (x2 - x1)*(xi + 1)/2.
            assert xlocal > x1 and xlocal < x2
            theta_local = np.interp(xlocal, xlin, thetalin)
            if idealistic_CTS:
                # Lincoln, R. L., Weaver, P. M., Pirrera, A., and Groh, R. M.
                # J., 2021, “Imperfection-Insensitive Continuous Tow-Sheared
                # Cylinders,” Compos. Struct., 260, p. 113445.
                #NOTE in the idealistic_CTS, there is thickness increase only
                #     when the steering occurs out of a reference angle
                steering_angle = theta_local - thetadeg_c1
            else:
                #NOTE in the real CTS, there is thickness increase for any
                #     angle other than 0, given that the shift direction is the
                #     circumferential direction
                steering_angle = theta_local
            plyt_local = h_tow / np.cos(np.deg2rad(steering_angle))

            # forcing balanced laminates
            stack = (theta_local, -theta_local)
            plyts = (plyt_local, plyt_local)

            offset = sum(plyts)/2.
            if zero_offset:
                offset = 0
            prop = laminated_plate(stack=stack, plyts=plyts, laminaprop=laminaprop, offset=offset, rho=rho)
            for j in range(nint):
                wj = weights[j]
                weight = wi*wj
                volume += weight*elem.lex*elem.ley/4.*prop.h
                mass += weight*elem.lex*elem.ley/4.*prop.intrho
                havg_elem += weight/4.*sum(plyts)
                thetadegavg_elem += weight/4.*theta_local

                elem.A11[i, j] = prop.A11
                elem.A12[i, j] = prop.A12
                elem.A16[i, j] = prop.A16
                elem.A22[i, j] = prop.A22
                elem.A26[i, j] = prop.A26
                elem.A66[i, j] = prop.A66
                elem.B11[i, j] = prop.B11
                elem.B12[i, j] = prop.B12
                elem.B16[i, j] = prop.B16
                elem.B22[i, j] = prop.B22
                elem.B26[i, j] = prop.B26
                elem.B66[i, j] = prop.B66
                elem.D11[i, j] = prop.D11
                elem.D12[i, j] = prop.D12
                elem.D16[i, j] = prop.D16
                elem.D22[i, j] = prop.D22
                elem.D26[i, j] = prop.D26
                elem.D66[i, j] = prop.D66
        havg_elements.append(havg_elem)
        thetadegavg_elements.append(thetadegavg_elem)
        elem.init_k_KC0 = init_k_KC0
        elem.init_k_KCNL = init_k_KCNL
        elem.init_k_KG = init_k_KG
        init_k_KC0 += KC0_SPARSE_SIZE
        init_k_KCNL += KCNL_SPARSE_SIZE
        init_k_KG += KG_SPARSE_SIZE
        elements.append(elem)
        connectivity.append([n1, n2, n3, n4])

    havg_elements = np.asarray(havg_elements)
    havg = havg_elements.mean()
    out['connectivity'] = connectivity
    out['volume'] = volume
    out['mass'] = mass
    out['thetadegavg_elements'] = thetadegavg_elements
    out['havg_elements'] = havg_elements
    out['havg'] = havg

    if mesh_only:
        return out

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
        #NOTE not named c1..c4, which hold the axial lengths of the CTS
        #     parameterization reported in out
        cs = [DOF*pos1, DOF*pos2, DOF*pos3, DOF*pos4]
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

    #NOTE the CTS parameterization of dos Santos and Castro, Materials, 2022,
    #     15(12), 4117, doi:10.3390/ma15124117, steers the tows along x only
    #     and thickens the laminate as h_tow/cos(theta(x)), so the laminate is
    #     uniform around the circumference as in the constant stiffness
    #     models. Solving for the pre-buckling state in the axisymmetric
    #     subspace then keeps the Newton-Raphson clear of the whole near
    #     critical cluster up to lambda_b/lambda_c close to 1, as AXBIF and
    #     ANILISA do. Only the pre-buckling state is restricted, the eigenvalue
    #     problem being solved on the FULL space. See "Why the axisymmetric
    #     reduction still applies" in doc/nlprebuck_implementation.tex
    axi_order = mesh_order(x, y, nx, ny)
    #NOTE unlike the constant stiffness models, xlin is not uniformly spaced
    #     here, so the axial stations that mesh_order sorts the nodes into are
    #     checked before the axisymmetric basis is built on them
    assert np.all(x[axi_order] == x[axi_order][:, :1])
    axi_imid = np.argmin(np.abs(xlin - L/2.))

    Baxi, buaxi = axisymmetric_basis(axi_order, bu, DOF)

    def project_axi(u):
        return project_axisymmetric(u, axi_order, axi_imid, DOF)

    def solve_axi(KT, rhs):
        """Solve KT du = rhs for the axisymmetric du

        The Galerkin projection onto the axisymmetric basis, and not the full
        solve followed by a projection, for the reason given in
        axisymmetric_basis. The reduced matrix is small enough to be dense
        """
        Kr = (Baxi.T @ KT @ Baxi).toarray()
        rr = Baxi.T @ rhs
        a = np.zeros(Baxi.shape[1], dtype=DOUBLE)
        Krf = Kr[np.ix_(buaxi, buaxi)]
        #NOTE nodal displacements next to nodal derivatives give the reduced
        #     matrix a condition number of 4e12, which the symmetric scaling
        #     brings down to 3e2. The sparse solver used elsewhere
        #     equilibrates internally, this dense one does not. Iterates are
        #     unchanged to ten digits on ny=60, so it is a guard, not a fix
        d = 1/np.sqrt(np.abs(Krf.diagonal()))
        a[buaxi] = d*np.linalg.solve(d[:, None]*Krf*d[None, :], d*rr[buaxi])
        return Baxi @ a

    u0_lin = project_axi(u0)
    u0 = u0_lin.copy()

    def assemble_KG(u):
        KGv[:] = 0
        for elem in elements:
            update_KG(u, elem, points, weights, KGr, KGc, KGv)
        return coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()

    def solve_eig(KCuu, KGuu):
        """Buckling multipliers of the stress state currently stored in KG

        The starting vector is fixed on purpose. ARPACK keeps its random seed
        in a SAVEd variable, so without one the basis it returns for a
        degenerate eigenspace depends on how many eigenvalue problems were
        solved before in the same process. That alone does not pin the basis
        down, round off inside the eigen solver being enough to rotate it, or
        to return a single member of a pair, which is what canonical_modes is
        for
        """
        v0 = np.random.default_rng(0).random(KCuu.shape[0])
        eigvals, eigvecsu = eigsh(A=KGuu, k=num_eigvals, which='LM', M=KCuu,
                tol=1e-6, v0=v0)
        mu = -1/eigvals
        return eigvals, canonical_modes(mu, eigvecsu, bu,
                axi_order, DOF), mu

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

        # Newton-Raphson, the tangent updated at every iteration
        def scaling(vec, D):
            """
                A. Peano and R. Riccioni, Automated discretisation error
                control in finite element analysis. In Finite Elements in
                the Commercial Environment (Edited by J. Robinson),
                pp. 368-387. Robinson & Assoc., Verwood, England (1978)
            """
            return np.sqrt((vec*np.abs(1/D))@vec)

        D = KC0uu.diagonal() # fixed for the whole analysis
        epsilon = NR_eps

        def solve_prebuckling(lambda_b, ui):
            """Pre-buckling state at the load level lambda_b*Nxxunit

            The correction is solved for in the axisymmetric subspace, where
            the fundamental path of a perfect cylinder under axial
            compression lives and where the tangent stiffness matrix stays
            regular right up to the bifurcation point, every buckling mode
            that makes it singular being non-axisymmetric. The convergence
            test below is still the one of the full residual.
            """
            fext_b = lambda_b*fext
            fint = np.zeros(N)
            fint = calc_fint(ui, fint)
            Ri = fint - fext_b
            u = ui.copy()
            KT = calc_KT(ui)
            iteration = 0
            best_test = np.inf
            u_best = ui.copy()
            stall = 0

            def give_up(reason):
                #NOTE close to the bifurcation point the residual can reach a
                #     floor above NR_eps and grow again. The best iterate is
                #     kept whenever that floor is below NR_eps_accept, rather
                #     than throwing the load step away
                if best_test < NR_eps_accept:
                    print('#        %s, keeping the best iterate, '
                            'crisfield_test %r' % (reason, best_test))
                    return u_best, calc_KT(u_best)
                raise RuntimeError('Newton-Raphson %s at lambda_b=%r, the '
                        'pre-buckling state is probably too close to the '
                        'bifurcation point' % (reason, lambda_b))

            while True:
                u = ui + solve_axi(KT, -Ri)
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
                ui = u.copy()

        #NOTE Iterative eigenvalue algorithm of Sun et al. (2020), Eqs. (44)
        #     to (46), Step 1 of their Fig. 2. The asymptotic expansion is
        #     only valid about a pre-buckling state in the neighbourhood of
        #     the bifurcation point, lambda_b/lambda_c approximately 0.995.
        #     Expanding about the reference load Nxxunit instead leaves
        #     lambda_b/lambda_c of the order of 0.1, where the nonlinear
        #     pre-buckling deformation caused by the edge restraint has not
        #     developed yet, and gives back the membrane pre-buckling result
        def calc_u0dot(KT):
            """Eq. (29), rate of the pre-buckling state with respect to the
            load parameter. Assuming instead that the pre-buckling path is
            linear in lambda, u0(lambda) = lambda*u0, would make this equal
            to the pre-buckling state itself. Solved in the axisymmetric
            subspace for the same reason as in solve_prebuckling"""
            return solve_axi(KT, fext)

        lambda_b = 1.
        u0 = u0_lin.copy()
        KT = None
        #NOTE the previous converged load step, kept for the backward
        #     difference of Eq. (35)
        lambda_prev = None
        u0dot_prev = None
        #NOTE the buckling load of the previous load step, for the
        #     sensitivity dlambda_c/dlambda_b below, and the whole state of
        #     that step, to fall back on if a step lands past the bifurcation
        #     point anyway
        lambda_c_prev = None
        back = None
        eta_cap = 1.
        KC = KC0 + assemble_KCNL(u0)
        KG = assemble_KG(u0)
        print('# starting iterative eigenvalue analysis')
        converged = False
        for iteration in range(1, NLprebuck_maxiter+1):
            KCuu = KC[bu, :][:, bu]
            KGuu = KG[bu, :][:, bu]
            eigvals, eigvecsu, mu = solve_eig(KCuu, KGuu)
            lambda_c = lambda_b*mu[0] # Eq. (46)
            print('#    iteration', iteration, 'lambda_b', lambda_b,
                    'lambda_c', lambda_c, 'lambda_b/lambda_c',
                    lambda_b/lambda_c)
            #NOTE Eq. (45) measures the distance to the bifurcation point
            #     from below only. Once lambda_b passes lambda_c its left hand
            #     side turns negative and the test accepts the state whatever
            #     the overshoot, so the distance is taken in absolute value
            gap = (lambda_c - lambda_b)/lambda_c
            if abs(gap) <= NLprebuck_eps1: # Eq. (45)
                print('#    converged')
                converged = True
                break
            if gap < 0:
                #NOTE past the bifurcation point by more than the tolerance.
                #     The fundamental path is still there, the pre-buckling
                #     solve being restricted to the axisymmetric subspace,
                #     but the expansion is meant to be made on the near side
                #     of it, so the load step is taken again shorter
                print('#    overshot to lambda_b/lambda_c %r, stepping back'
                        % (lambda_b/lambda_c))
                eta_cap *= 0.5
                if back is None or eta_cap < 1.e-2:
                    break
                (lambda_b, u0, KT, lambda_prev, u0dot_prev,
                        lambda_c_prev) = back
                back = None
                KC = KC0 + assemble_KCNL(u0)
                KG = assemble_KG(u0)
                continue
            #NOTE eta=0.8 in the first iteration to approach the neighbourhood
            #     of the buckling load quickly, eta=0.5 afterwards
            eta = 0.8 if iteration == 1 else 0.5
            #NOTE lambda_c falls as lambda_b rises, and with
            #     s = dlambda_c/dlambda_b the step of Eq. (44) lands past the
            #     bifurcation point whenever eta > 1/(1 - s); s reaches -1.09
            #     on the ny=60 Arbocz and Starnes mesh. Capping eta at
            #     0.7/(1 - s), s from the two previous load steps, makes
            #     Eq. (44) a secant iteration dividing the distance by a fixed
            #     factor per step; 0.7 because the backward estimate of s lags
            #     while it steepens. See "Advancing the load level" in
            #     doc/nlprebuck_implementation.tex
            if lambda_c_prev is not None and lambda_b != lambda_prev:
                s = (lambda_c - lambda_c_prev)/(lambda_b - lambda_prev)
                if s < 0:
                    eta = min(eta, 0.7/(1 - s))
            eta = min(eta, eta_cap)
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
                    guess = u0 + calc_u0dot(KT)*(lambda_b_new - lambda_b)
                try:
                    u0_new, KT_new = solve_prebuckling(lambda_b_new, guess)
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
            back = (lambda_b, u0, KT, lambda_prev, u0dot_prev, lambda_c_prev)
            lambda_prev = lambda_b
            u0dot_prev = u0_lin if KT is None else calc_u0dot(KT)
            lambda_c_prev = lambda_c
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
        u0dot = calc_u0dot(KT)

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
    out['rCTS'] = rCTS
    out['param_n'] = param_n
    out['c2_ratio'] = c2_ratio
    out['thetadeg_c1'] = thetadeg_c1
    out['thetadeg_c2'] = thetadeg_c2
    out['t'] = t
    out['c1'] = c1
    out['c2'] = c2
    out['koiter'] = None

    #NOTE koiter_num_modes may be a callable of the multipliers and of the
    #     eigenvectors over every degree of freedom, returning the number of
    #     Koiter modes, so that the expansion can be made on a set chosen
    #     from the computed modes, e.g. one that does not cut a group of
    #     modes of equal multiplier. It is called once, after the last
    #     eigenvalue analysis
    if callable(koiter_num_modes):
        koiter_num_modes = int(koiter_num_modes(mu, eigvecs))
        if not 0 <= koiter_num_modes <= num_eigvals:
            raise ValueError('koiter_num_modes returned %d, outside 0 to '
                    'num_eigvals=%d' % (koiter_num_modes, num_eigvals))
    out['koiter_num_modes'] = koiter_num_modes

    if koiter_num_modes == 0:
        return out

    lambda_a = {}
    for modei in range(koiter_num_modes):
        lambda_a[modei] = load_mult[modei]

    #NOTE the largest nodal translation of every mode made equal to h; the
    #     reference coefficients use the crest amplitude of w instead, see
    #     "Normalising the modes" in doc/nlprebuck_implementation.tex
    ua = {}
    for modei in range(koiter_num_modes):
        ua[modei] = eigvecs[:, modei].copy()
        ampl = np.sqrt(ua[modei][0::DOF]**2 + ua[modei][3::DOF]**2 + ua[modei][6::DOF]**2).max()
        ua[modei] /= ampl
        ua[modei] *= havg

    #NOTE the null space of phi2, against which the second order fields are
    #     constrained, see the bordered system below. The buckling modes of a
    #     cylinder come in degenerate pairs, so it is spanned by every
    #     eigenvector whose multiplier equals the critical one AND by the
    #     partner of each of them, not only by the Koiter modes. Whether the
    #     eigen solver returns a partner is decided by round off, and one left
    #     out of the column border is a null vector of the whole bordered
    #     matrix, so it is rebuilt from the cyclic symmetry when missing
    cols = [ua[modek][bu] for modek in range(koiter_num_modes)]
    for j in range(num_eigvals):
        if abs(mu[j] - mu[0]) > 1.e-3*abs(mu[0]):
            continue
        cols.append(eigvecsu[:, j])
        group = [k for k in range(num_eigvals)
                 if abs(mu[k] - mu[j]) <= 1.e-5*abs(mu[j])]
        if len(group) == 1:
            partner = degenerate_partner(eigvecs[:, j], bu, axi_order, DOF)
            if partner is not None:
                cols.append(partner[bu])
    #NOTE the Koiter modes first, then what the other vectors add to them.
    #     Every eigenvector that is also a Koiter mode is in cols twice, and a
    #     QR factorization without pivoting would give the round off left of
    #     the duplicate a column of its own and project the next vector
    #     against it, so the complement is taken from an SVD instead
    C = np.asarray(cols).T
    C = C/np.linalg.norm(C, axis=0)
    Qm, Rm = np.linalg.qr(C[:, :koiter_num_modes])
    assert np.abs(np.diag(Rm)).min() > 1.e-6, 'linearly dependent Koiter modes'
    for _ in range(2):
        C = C - Qm @ (Qm.T @ C)
    Uc, sc = np.linalg.svd(C, full_matrices=False)[:2]
    Nsp = np.hstack((Qm, Uc[:, sc > 1.e-8]))
    print('# null space of phi2 deflated with %d vectors' % Nsp.shape[1])

    #NOTE the directions along which the orthogonality conditions are
    #     imposed, one per column of Nsp and spanning the same space: the
    #     Koiter modes, then the rest of the null space
    ucond = {}
    for modek in range(koiter_num_modes):
        ucond[modek] = ua[modek]
    for col in range(koiter_num_modes, Nsp.shape[1]):
        v = np.zeros(N)
        v[bu] = Nsp[:, col]
        ucond[col] = v
    num_cond = len(ucond)

    #NOTE this flag multiplies every nonlinear contribution to the
    #     pre-buckling strains and to their derivatives. With NLprebuck=False
    #     the pre-buckling state is the linear elastic solution, the
    #     pre-buckling path is exactly linear in the load parameter, and those
    #     contributions have to be absent; with NLprebuck=True they are the
    #     terms that carry the nonlinear pre-buckling behaviour
    flag = NLprebuck

    # higher-order tensors for elements

    #NOTE the Koiter tensors and coefficients are computed with the modes as
    #     axes of the arrays, never in loops over them, by the functions of
    #     koiter_tensors.py, common to the four models, whose docstring
    #     describes the strategy; the model supplies only its kinematics,
    #     nonlinear_rows. The stiffness at the integration points is read
    #     from the elements, where the CTS models store it per point and the
    #     constant-stiffness ones the same laminate everywhere.
    #     phi3_ab[(a, b)] is phi3[:, a, b], phi20_a[k] is phi20[:, k], and
    #     likewise for phi30, cst, phi200 and phi4
    U = np.column_stack([ua[modek] for modek in range(koiter_num_modes)])
    Ucond = np.column_stack([ucond[modek] for modek in range(num_cond)])
    lam = np.array([lambda_a[modei] for modei in range(koiter_num_modes)])
    phi20, phi3, phi30, cst, phi200, phi4 = koiter_element_tensors(elements,
            points, weights, u0, u0dot, u0ddot, Ucond, koiter_num_modes, flag,
            nonlinear_rows)

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
    #NOTE phi3U[i, j, k] = phi3_ab[(i, j)] @ ua[k], in one product, and
    #     d[i] = phi20_a[i] @ ua[i], the denominator of a_ijk and b_ijkl
    phi3U = np.tensordot(phi3, U, axes=(0, 0))
    d = np.array([phi20[:, modei] @ ua[modei]
                  for modei in range(koiter_num_modes)])
    a = a_coefficients(phi3U, lam, d)
    a_abc = {}
    for idx in np.ndindex(a.shape):
        a_abc[idx] = a[idx]
        print('# $a_%d%d%d$' % tuple(i+1 for i in idx), a[idx])
    #NOTE the second order fields solve the terms of order xi_a xi_b of the
    #     equilibrium equations,
    #         phi2 uab + 1/2 phi3_ab + sum_l z_l phi20_a[l] = 0
    #     with z_l the coefficient of xi_a xi_b in (lambda - lambda_l) xi_l.
    #     The same terms projected onto the Koiter modes, along which phi2
    #     vanishes, are the amplitude equations that give z,
    #         sum_l Tkl[k, l] z_l = -1/2 phi3_ab @ ua[k]
    #     with Tkl[k, l] = phi20_a[k] @ ua[l], so the right hand side is
    #     orthogonal to every Koiter mode and the column border of the
    #     bordered system below has nothing to absorb. For T-orthogonal modes
    #     z_l = lambda_l a_lab; on a cylinder phi3_ab @ ua[k] vanishes and this
    #     matters only for an asymmetric bifurcation. See "The singular
    #     system" in doc/nlprebuck_implementation.tex
    Tkl = phi20[:, :koiter_num_modes].T @ U

    def force2ndorder_ij(modei, modej):
        #NOTE phi3_ij = phi3_ji even in the asym case
        z = np.linalg.solve(Tkl, -1/2.*phi3U[modei, modej])
        return -1/2.*phi3[:, modei, modej] - phi20[:, :koiter_num_modes] @ z

    #NOTE phi2 is singular by construction, the buckling modes spanning its
    #     null space, so the second order fields come from the bordered system
    #         [phi2  Nsp] [uab  ]   [force2ndorder]
    #         [W^T   0  ] [alpha] = [    -cst     ]
    #     whose column border Nsp spans that null space, which makes it non
    #     singular, and whose row border W imposes along every direction q_k
    #     of the null space the orthogonality condition of Sun et al.
    #     Eq. (33), generalized to distinct indices a, b,
    #         phi20_a[k] @ uab = -cst_ab[(a, b)] @ q_k
    #                          = -1/2 <N[L2(ua, ub)], L11(u0_dot, q_k)>
    #     It is weighted by the stiffness matrices and inhomogeneous, which
    #     neither a Gram-Schmidt projection nor an Euclidean border is. Its
    #     constant is symmetric in a and b but not in k, and equals
    #     phi30_ab @ q_k/6, the average over the three slots, only when a, b
    #     and k coincide. Along the directions outside the Koiter modes it
    #     keeps the second order field from bringing back amplitude the
    #     expansion gave zero. See "Second-order fields and the orthogonality
    #     condition" in doc/nlprebuck_implementation.tex
    nu = int(bu.sum())
    W = phi20[bu]

    bordered = bmat([[phi2uu, csc_matrix(Nsp)],
                     [csc_matrix(W).T, None]], format='csc')

    #NOTE cstq[i, j, k] = cst_ab[(i, j)] @ ucond[k]
    cstq = np.tensordot(cst, Ucond, axes=(0, 0))
    #NOTE phi3_ab, cst_ab and hence the whole right hand side are symmetric
    #     in a and b, so uab = uba and only the fields with a <= b are solved
    #     for, m (m + 1)/2 solves instead of m**2. Solving for both gave
    #     fields that differ by round off only, see "Vectorization of the
    #     Koiter tensors" in doc/nlprebuck_implementation.tex
    uab = {}
    for modei in range(koiter_num_modes):
        for modej in range(koiter_num_modes):
            if modej < modei:
                uab[(modei, modej)] = uab[(modej, modei)]
                continue
            rhs = np.zeros(nu + num_cond)
            rhs[:nu] = force2ndorder_ij(modei, modej)[bu]
            rhs[nu:] = -cstq[modei, modej]
            sol = spsolve(bordered, rhs)
            uijbar = np.zeros(N)
            uijbar[bu] = sol[:nu]
            uab[(modei, modej)] = uijbar

    print('# b_ijkl factors')
    #NOTE phi3uab[i, j, k, l] = phi3_ab[(i, j)] @ uab[(k, l)] and
    #     phi30U[i, k, l] = phi30_ab[(i, k)] @ ua[l], as matrix products
    phi3uab = np.zeros((koiter_num_modes,)*4)
    for (modek, model), uijbar in uab.items():
        if model < modek:
            phi3uab[:, :, modek, model] = phi3uab[:, :, model, modek]
            continue
        phi3uab[:, :, modek, model] = np.tensordot(phi3, uijbar,
                axes=(0, 0))
    phi30U = np.tensordot(phi30, U, axes=(0, 0))
    #NOTE the formula of b_ijkl, not index-symmetric, is in b_coefficients
    b = b_coefficients(phi4, phi3uab, phi30U, phi200, a, lam, d)
    b_ijkl = {}
    for idx in np.ndindex(b.shape):
        b_ijkl[idx] = b[idx]
        print('# $b_{%d%d%d%d}$, %f' % (tuple(i+1 for i in idx) + (b[idx],)))

    koiter = dict(
        a_ijk=a_abc,
        b_ijkl=b_ijkl,
        koiter_num_modes=koiter_num_modes,
        lambda_i=lambda_a,
        u0=u0,
        ui=ua,
        uij=uab,
        u0dot=u0dot,
        ucond=ucond,
            )
    out['koiter'] = koiter

    return out
