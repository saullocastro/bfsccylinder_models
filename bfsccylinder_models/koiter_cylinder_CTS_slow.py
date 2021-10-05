from functools import partial
from collections import defaultdict

try:
    from pypardiso import spsolve
except:
    from scipy.sparse.linalg import spsolve

import numpy as np
from numpy import isclose
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import eigsh
from composites import laminated_plate
from bfsccylinder import (BFSCCylinder, update_KC0, update_KCNL, update_KG,
        update_fint, DOF, DOUBLE, INT, KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE,
        KG_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights

num_nodes = 4

def fkoiter_cylinder_CTS_circum(L, R, rCTS, nxt, ny, E11, E22, nu12, G12, rho,
        h_tow, param_n, param_f, thetadeg_c, thetadeg_s, cg_x0=None,
        mesh_only=False, nint=4, num_eigvals=2, koiter_num_modes=1, load=1000,
        NLprebuck=False):

    circ = 2*np.pi*R
    out = {}

    if param_n == 0 or param_f == 0:
        print('# constant stiffness case')
        assert ny is not None
        nx = int(ny*L/circ)
        if nx % 2 == 0:
            nx += 1
        xlin = np.linspace(0, L, nx)
        thetalin = np.ones_like(xlin)*thetadeg_c
        t = None
        c = None
        s = None

    else:
        assert abs(thetadeg_s) > abs(thetadeg_c), 'thetadeg_s must be larger than thetadeg_c'
        t = rCTS*np.sin(np.deg2rad(thetadeg_s - thetadeg_c))
        nmax = L/(2*t)
        print('# nmax', nmax)
        assert param_n <= nmax
        cmax = (L - 2*t*param_n)/(param_n+1)
        if not isclose(cmax, 0):
            s = param_f/(param_n*(param_f + 1))*(L - 2*t*param_n)
            c = 1/(param_f*param_n + param_f + param_n + 1)*(L-2*t*param_n)
        else:
            s = 0
            c = 0
        assert isclose((2*t + s)*param_n + c*(param_n+1) - L, 0)
        print('# param_t', t)
        print('# param_s', s)
        print('# param_c', c)

        dx = t/(nxt-1)
        if ny is None:
            ny = int(round(circ/dx, 0))
        nxc = max(2, int(round(c/t*nxt, 0)))
        nxs = max(2, int(round(s/t*nxt, 0)))
        print('# nxc', nxc)
        print('# nxs', nxs)
        xlin = np.linspace(0, c, nxc-1, endpoint=False)
        thetalin = np.ones(nxc-1)*thetadeg_c
        for i in range(param_n):
            start = c + i*(c + 2*t + s)
            xlin = np.concatenate((xlin, np.linspace(start, start+t, nxt-1, endpoint=False)))
            thetalin = np.concatenate((thetalin, thetadeg_c + np.linspace(0, 1, nxt-1, endpoint=False)*(thetadeg_s - thetadeg_c)))
            if not isclose(s, 0):
                xlin = np.concatenate((xlin, np.linspace(start+t, start+t+s, nxs-1, endpoint=False)))
                thetalin = np.concatenate((thetalin, np.ones(nxs-1)*thetadeg_s))
            xlin = np.concatenate((xlin, np.linspace(start+t+s, start+t+s+t, nxt-1, endpoint=False)))
            thetalin = np.concatenate((thetalin, thetadeg_s + np.linspace(0, 1, nxt-1, endpoint=False)*(thetadeg_c - thetadeg_s)))
            if i == param_n-1:
                endpoint = True
                neff = nxc
            else:
                endpoint = False
                neff = nxc-1
            xlin = np.concatenate((xlin, np.linspace(start+t+s+t, start+t+s+t+c, neff, endpoint=endpoint)))
            thetalin = np.concatenate((thetalin, np.ones(neff)*thetadeg_c))

    nx = xlin.shape[0]
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

    points, weights = get_points_weights(nint=nint)

    num_elements = len(n1s)
    print('# nx', nx)
    print('# ny', ny)
    print('# number of elements', num_elements)

    elements = []
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
            steering_angle = theta_local - thetadeg_c
            plyt_local = h_tow / np.cos(np.deg2rad(steering_angle))

            # forcing balanced laminates
            stack = (theta_local, -theta_local)
            plyts = (plyt_local, plyt_local)

            offset = sum(plyts)/2.
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

    havg_elements = np.asarray(havg_elements)
    havg = havg_elements.mean()
    out['volume'] = volume
    out['mass'] = mass
    out['thetadegavg_elements'] = thetadegavg_elements
    out['havg_elements'] = havg_elements
    out['havg'] = havg
    out['elements'] = elements

    if mesh_only:
        return out

    KC0r = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for elem in elements:
        update_KC0(elem, points, weights, KC0r, KC0c, KC0v)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    print('# finished element assembly')

    # applying boundary conditions
    bk = np.zeros(N, dtype=bool)

    checkSS = isclose(x, 0) | isclose(x, L)
    bk[3::DOF] = checkSS
    bk[6::DOF] = checkSS
    check = isclose(x, L/2.)
    bk[0::DOF] = check
    bu = ~bk # same as np.logical_not, defining unknown DOFs
    u0 = np.zeros(N, dtype=DOUBLE)
    uk = u0[bk]

    print('# starting static analysis')

    # axially compressive load applied at x=0 and x=L
    checkTopEdge = isclose(x, L)
    checkBottomEdge = isclose(x, 0)
    fext = np.zeros(N)
    fext[0::DOF][checkBottomEdge] = +load/ny
    assert isclose(fext.sum(), load)
    fext[0::DOF][checkTopEdge] = -load/ny
    assert isclose(fext.sum(), 0)

    # sub-matrices corresponding to unknown DOFs
    KC0uu = KC0[bu, :][:, bu]
    KC0uk = KC0[bu, :][:, bk]

    KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)

    Nu = N - bk.sum()

    # solving
    uu = spsolve(KC0uu, fext[bu])
    cg_x0 = uu.copy()

    u0[bu] = uu

    if NLprebuck:
        print('#    initiating nonlinear pre-buckling state')
        KCNLr = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=INT)
        KCNLc = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=INT)
        KCNLv = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=DOUBLE)

        def calc_KT(u, KCNLv, KGv):
            KCNLv *= 0
            KGv *= 0
            for elem in elements:
                update_KCNL(u, elem, points, weights, KCNLr, KCNLc, KCNLv)
                update_KG(u, elem, points, weights, KGr, KGc, KGv)
            KCNL = coo_matrix((KCNLv, (KCNLr, KCNLc)), shape=(N, N)).tocsc()
            KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
            return KC0 + KCNL + KG

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

        iteration = 0
        fint = np.zeros(N)
        fint = calc_fint(u0, fint)
        Ri = fint - fext
        du = np.zeros(N)
        ui = u0.copy()
        epsilon = 1.e-4
        KT = calc_KT(u0, KCNLv, KGv)
        KTuu = KT[bu, :][:, bu]
        D = KC0uu.diagonal() # at beginning of load increment
        while True:
            print('#    iteration', iteration)
            duu = spsolve(KTuu, -Ri[bu])
            du[bu] = duu
            u = ui + du
            fint = calc_fint(u, fint)
            Ri = fint - fext
            crisfield_test = scaling(Ri[bu], D)/max(scaling(fext[bu], D), scaling(fint[bu], D))
            print('#        crisfield_test, max(R)', crisfield_test, np.abs(Ri).max())
            if crisfield_test < epsilon:
                print('#    converged')
                break
            iteration += 1
            KT = calc_KT(u, KCNLv, KGv)
            KTuu = KT[bu, :][:, bu]
            ui = u.copy()
        u0 = u.copy()


        KCNLv *= 0
        for elem in elements:
            update_KCNL(u0, elem, points, weights, KCNLr, KCNLc, KCNLv)
        KCNL = coo_matrix((KCNLv, (KCNLr, KCNLc)), shape=(N, N)).tocsc()
        KC = KC0 + KCNL
        KCuu = KC[bu, :][:, bu]

    else:
        KC = KC0
        KCuu = KC0uu

    print('# finished static analysis')

    KGv *= 0
    for elem in elements:
        update_KG(u0, elem, points, weights, KGr, KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]

    print('# starting eigenvalue analysis')
    eigvals, eigvecsu = eigsh(A=KCuu, k=num_eigvals, which='SM', M=KGuu,
            tol=1e-8, sigma=1., mode='buckling')
    load_mult = -eigvals
    print('# finished eigenvalue analysis')

    Pcr = load_mult[0]*load
    print('# eigvals', load_mult)
    print('# critical buckling load', Pcr)

    out['P0'] = load
    out['Pcr'] = Pcr
    out['cg_x0'] = cg_x0
    out['eigvals'] = load_mult
    eigvecs = np.zeros((N, num_eigvals))
    eigvecs[bu, :] = eigvecsu
    out['eigvecs'] = eigvecs
    out['t'] = t
    out['s'] = s
    out['c'] = c
    out['koiter'] = None

    if koiter_num_modes == 0:
        return out

    lambda_a = {}
    for modei in range(koiter_num_modes):
        lambda_a[modei] = eigvals[modei]

    es = partial(np.einsum, optimize='greedy', casting='no')
    #from opt_einsum import contract
    #es = partial(contract)

    #NOTE making the maximum amplitude of the eigenmode equal to h
    #normalizing amplitude of eigenvector according to shell thickness
    ua = {}
    for modei in range(koiter_num_modes):
        ua[modei] = eigvecs[:, modei].copy()
        if isclose(abs(ua[modei][6::DOF].max()), abs(ua[modei][6::DOF].min())):
            ua[modei] /= abs(ua[modei][6::DOF].max())
        elif abs(ua[modei][6::DOF].max()) >= abs(ua[modei][6::DOF].min()):
            ua[modei] /= ua[modei][6::DOF].max()
        else:
            ua[modei] /= ua[modei][6::DOF].min()
        ua[modei] *= havg

    phi4 = defaultdict(lambda: 0)
    phi3_ab = {}
    phi30_ab = {}
    phi3e_ab = {}
    phi30e_ab = {}
    phi20e_a = {}
    phi20_a = {}
    phi2 = np.zeros((N, N))
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

    # higher-order tensors for elements

    u0e = np.zeros(num_nodes*DOF, dtype=np.float64)
    for count, elem in enumerate(elements):
        if count % (num_elements//5) == 0:
            print('#    count', count+1, num_elements)
        eiab = np.zeros((3, num_nodes*DOF, num_nodes*DOF))
        eiab0 = np.zeros((3, num_nodes*DOF, num_nodes*DOF))
        eiab00 = np.zeros((3, num_nodes*DOF, num_nodes*DOF))

        c1 = elem.c1
        c2 = elem.c2
        c3 = elem.c3
        c4 = elem.c4

        u0e *= 0
        for i in range(DOF):
            u0e[0*DOF + i] = u0[c1 + i]
            u0e[1*DOF + i] = u0[c2 + i]
            u0e[2*DOF + i] = u0[c3 + i]
            u0e[3*DOF + i] = u0[c4 + i]

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
        rows = []
        cols = []
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

        phi2e = np.zeros((num_nodes*DOF, num_nodes*DOF))

        for i in range(nint):
            xi = points[i]
            weight_xi = weights[i]
            for j in range(nint):
                #TODO use Cython later
                Aij = np.array([
                    [elem.A11[i, j], elem.A12[i, j], elem.A16[i, j]],
                    [elem.A12[i, j], elem.A22[i, j], elem.A26[i, j]],
                    [elem.A16[i, j], elem.A26[i, j], elem.A66[i, j]]])
                Bij = np.array([
                    [elem.B11[i, j], elem.B12[i, j], elem.B16[i, j]],
                    [elem.B12[i, j], elem.B22[i, j], elem.B26[i, j]],
                    [elem.B16[i, j], elem.B26[i, j], elem.B66[i, j]]])
                Dij = np.array([
                    [elem.D11[i, j], elem.D12[i, j], elem.D16[i, j]],
                    [elem.D12[i, j], elem.D22[i, j], elem.D26[i, j]],
                    [elem.D16[i, j], elem.D26[i, j], elem.D66[i, j]]])
                eta = points[j]
                weight_eta = weights[j]
                weight = weight_xi * weight_eta

                elem.update_Nw_x(xi, eta)
                elem.update_Nw_y(xi, eta)
                elem.update_Bm(xi, eta)
                elem.update_Bb(xi, eta)

                Nw_x = np.atleast_2d(elem.Nw_x)
                Nw_y = np.atleast_2d(elem.Nw_y)

                w0_x = Nw_x[0] @ u0e
                w0_y = Nw_y[0] @ u0e

                Bm = np.asarray(elem.Bm)
                Bb = np.asarray(elem.Bb)

                ei0 = ej0 = Bm @ u0e #NOTE ignoring NL terms
                ki0 = kj0 = Bb @ u0e

                #TODO why lambda_i[0]?
                ei = ei0*lambda_a[0]
                ki = ki0*lambda_a[0]

                ei00 = ej00 = np.array([w0_x**2, w0_y**2, 2*w0_x*w0_y])
                ki00 = 0

                Ni0 = Aij@ej0 + Bij@kj0
                Ni00 = Aij@ej00

                #TODO why lambda_a[0]?
                Ni = Ni0*lambda_a[0]

                eia = eib = eic = Bm #NOTE ignoring NL terms
                kia = kib = kic = Bb

                Nia = Nib = Nic = es('ij,ja->ia', Aij, eia) + es('ij,ja->ia', Bij, kia)
                Mia = Mib = es('ij,ja->ia', Bij, eia) + es('ij,ja->ia', Dij, kia)

                eia0 = eib0 = eic0 = [w0_x*Nw_x[0],
                                      w0_y*Nw_y[0],
                                      w0_x*Nw_y[0] + w0_y*Nw_x[0]]

                Nia0 = Nib0 = Nic0 = es('ij,ja->ia', Aij, eia0)
                Mia0 = Mib0 = es('ij,ja->ia', Bij, eia0)


                eiab[0] = Nw_x.T @ Nw_x
                eiab[1] = Nw_y.T @ Nw_y
                eiab[2] = Nw_x.T @ Nw_y + Nw_y.T @ Nw_x

                eicd = eibd = eibc = eiad = eiac = eiab

                Niab = Niac = Niad = Nibc = Nibd = Nicd = es('ij,jab->iab', Aij, eiab)
                Miab = Miac = Miad = Miad = Mibc = Mibd = Micd = es('ij,jab->iab', Bij, eiab)

                Niab = Niac = Niad = Nibc = Nibd = Nicd = es('ij,jab->iab', Aij, eiab)
                Miab = Miac = Miad = Miad = Mibc = Mibd = Micd = es('ij,jab->iab', Bij, eiab)

                phi2e += 1/2.*weight*(lex*ley/4.)*(
                             es('iab,i->ab', Niab, ei)
                           + es('ia,ib->ab', Nia, eib)
                           + es('ib,ia->ab', Nib, eia)
                           + es('i,iab->ab', Ni, eiab)
                           + es('iab,i->ab', Miab, ki)
                           + es('ia,ib->ab', Mia, kib)
                           + es('ib,ia->ab', Mib, kia)
                        )

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

        tmp = np.zeros((N, num_nodes*DOF))
        tmp[indices] = phi2e
        phi2[:, indices] += tmp
        for modei in range(koiter_num_modes):
            phi20_a[modei][indices] += phi20e_a[modei]
            for modej in range(koiter_num_modes):
                phi3_ab[(modei, modej)][indices] += phi3e_ab[(modei, modej)]
                phi30_ab[(modei, modej)][indices] += phi30e_ab[(modei, modej)]

    phi2uu = phi2[bu, :][:, bu]

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
            #NOTE I tried to add the contributions of a_ijk, but the correlation of b-factor in Diana was worse
            for modek in range(koiter_num_modes):
                lambda_k = lambda_a[modek]
                a_kij = a_abc[(modek, modei, modej)]
                force2ndorder_ij[(modei, modej)] += (
                        - (1/koiter_num_modes)*a_kij*lambda_k*phi20_a[modek]
                        )

    uab = {}
    for modei in range(koiter_num_modes):
        for modej in range(koiter_num_modes):
            uijbar = np.zeros(N)
            uijbar[bu] = spsolve(csc_matrix(phi2uu), force2ndorder_ij[(modei, modej)][bu])
            uab[(modei, modej)] = uijbar.copy()
            # Gram-Schmidt orthogonalization
            #NOTE uab are orthogonal to all buckling modes, but not mutually
            #     orthogonal (with respect to other second-order modes)
            for modek in range(koiter_num_modes):
                ui = ua[modek]
                uab[(modei, modej)] -= ui*np.dot(uijbar, ui)/np.dot(ui, ui)

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
