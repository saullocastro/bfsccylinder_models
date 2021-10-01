#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: infer_types=False
from functools import partial
from collections import defaultdict

import numpy as np
cimport numpy as np
from numpy import isclose, pi
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import eigsh, spsolve, cg, lobpcg, LinearOperator, spilu
from composites import laminated_plate
from bfsccylinder import BFSCCylinder, update_KC0, update_KG, KC0_SPARSE_SIZE, KG_SPARSE_SIZE
from bfsccylinder.quadrature import get_points_weights

ctypedef np.int64_t cINT
INT = np.int64
ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
cdef cINT DOF = 10
cdef cINT num_nodes = 4

def fkoiter_cylinder_CTS_circum(L, R, rCTS, nxt, ny, E11, E22, nu12, G12, rho, h_tow, param_n,
        param_f, thetadeg_c, thetadeg_s, clamped=True, cg_x0=None, lobpcg_X=None,
        int nint=4, int num_eigvals=2, int koiter_num_modes=1):


    cdef int i, j, k, m, n
    cdef int modei, modej, modek, model
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double w0_x, w0_y
    cdef double ei0[3]
    cdef double ej0[3]
    cdef double ei[3]
    cdef double ki0[3]
    cdef double kj0[3]
    cdef double ki[3]
    cdef double ei00[3]
    cdef double ej00[3]
    cdef double Ni[3]
    cdef double Ni0[3]
    cdef double Ni00[3]
    cdef np.ndarray[cDOUBLE, ndim=1] u0e, u0
    cdef np.ndarray[cDOUBLE, ndim=2] eia0, eib0, eic0
    cdef np.ndarray[cDOUBLE, ndim=2] eia, eib, eic, kia, kib, kic

    cdef np.ndarray[cDOUBLE, ndim=2] Nia, Nib, Nic, Mia, Mib
    cdef np.ndarray[cDOUBLE, ndim=2] Nia0, Nib0, Nic0, Mia0, Mib0

    cdef double[:, :, :] eiab, eicd, eibd, eibc, eiad, eiac
    cdef double[:, :, :] Niab, Niac, Niad, Nibc, Nibd, Nicd
    cdef double[:, :, :] Miab, Miac, Miad, Mibc

    eia0 = np.zeros((3, num_nodes*DOF), dtype=DOUBLE)

    Nia = np.zeros((3, num_nodes*DOF), dtype=DOUBLE)
    Mia = np.zeros((3, num_nodes*DOF), dtype=DOUBLE)
    Nia0 = np.zeros((3, num_nodes*DOF), dtype=DOUBLE)
    Mia0 = np.zeros((3, num_nodes*DOF), dtype=DOUBLE)

    eiab = np.zeros((3, num_nodes*DOF, num_nodes*DOF), dtype=DOUBLE)
    Niab = np.zeros((3, num_nodes*DOF, num_nodes*DOF), dtype=DOUBLE)
    Miab = np.zeros((3, num_nodes*DOF, num_nodes*DOF), dtype=DOUBLE)

    circ = 2*pi*R # [m]

    if param_n == 0 or param_f == 0:
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
        print('nmax', nmax)
        assert param_n <= nmax
        cmax = (L - 2*t*param_n)/(param_n+1)
        if not np.isclose(cmax, 0):
            s = param_f/(param_n*(param_f + 1))*(L - 2*t*param_n)
            c = 1/(param_f*param_n + param_f + param_n + 1)*(L-2*t*param_n)
        else:
            s = 0
            c = 0
        assert np.isclose((2*t + s)*param_n + c*(param_n+1) - L, 0)

        dx = t/(nxt-1)
        if ny is None:
            ny = int(round(circ/dx, 0))
        nxc = max(1, int(round(c/t*nxt, 0)))
        nxs = max(1, int(round(s/t*nxt, 0)))
        xlin = np.linspace(0, c, nxc-1, endpoint=False)
        thetalin = np.ones(nxc-1)*thetadeg_c
        for i in range(param_n):
            start = c + i*(c+2*t+s)
            xlin = np.concatenate((xlin, np.linspace(start, start+t, nxt-1, endpoint=False)))
            thetalin = np.concatenate((thetalin, thetadeg_c + np.linspace(0, 1, nxt-1, endpoint=False)*(thetadeg_s - thetadeg_c)))
            if nxs > 1:
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

    ytmp = np.linspace(0, circ, ny+1)
    ylin = np.linspace(0, circ-(ytmp[ytmp.shape[0]-1] - ytmp[ytmp.shape[0]-2]), ny)
    xmesh, ymesh = np.meshgrid(xlin, ylin)
    xmesh = xmesh.T
    ymesh = ymesh.T

    # getting nodes
    ncoords = np.vstack((xmesh.flatten(), ymesh.flatten())).T
    x = ncoords[:, 0]
    y = ncoords[:, 1]

    i = nids_mesh.shape[0] - 1
    j = nids_mesh.shape[1] - 1
    n1s = nids_mesh[:i, :j].flatten()
    n2s = nids_mesh[1:, :j].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:i, 1:].flatten()

    points, weights = get_points_weights(nint=nint)

    num_elements = len(n1s)
    print('# number of elements,', num_elements)

    elements = []
    N = DOF*nx*ny
    print('# number of DOF,', N)
    init_k_KC0 = 0
    init_k_KG = 0
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    mass = 0
    print('# starting element assembly')
    havg = 0 # average shell thickness h
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        shell = BFSCCylinder(nint)
        shell.n1 = n1
        shell.n2 = n2
        shell.n3 = n3
        shell.n4 = n4
        shell.c1 = DOF*nid_pos[n1]
        shell.c2 = DOF*nid_pos[n2]
        shell.c3 = DOF*nid_pos[n3]
        shell.c4 = DOF*nid_pos[n4]
        shell.R = R
        shell.lex = L/(nx-1) #TODO approximation, assuming evenly distributed element sizes
        shell.ley = circ/ny
        for i in range(nint):
            wi = weights[i]
            x1 = ncoords[nid_pos[n1]][0]
            x2 = ncoords[nid_pos[n2]][0]
            xi = points[i]
            xlocal = x1 + (x2 - x1)*(xi + 1)/2.
            assert xlocal > x1 and xlocal < x2

            theta_local = np.interp(xlocal, xlin, thetalin)
            steering_angle = theta_local - thetadeg_c
            plyt_local = h_tow / np.cos(np.deg2rad(steering_angle))

            #balanced laminate
            stack = (theta_local, -theta_local)
            plyts = (plyt_local, plyt_local)

            offset = sum(plyts)/2.
            prop = laminated_plate(stack=stack, plyts=plyts, laminaprop=laminaprop, offset=offset, rho=rho)
            for j in range(nint):
                wj = weights[j]
                weight = wi*wj
                mass += weight*shell.lex*shell.ley/4.*prop.intrho
                havg += weight/4.*sum(plyts)

                shell.A11[i, j] = prop.A11
                shell.A12[i, j] = prop.A12
                shell.A16[i, j] = prop.A16
                shell.A22[i, j] = prop.A22
                shell.A26[i, j] = prop.A26
                shell.A66[i, j] = prop.A66
                shell.B11[i, j] = prop.B11
                shell.B12[i, j] = prop.B12
                shell.B16[i, j] = prop.B16
                shell.B22[i, j] = prop.B22
                shell.B26[i, j] = prop.B26
                shell.B66[i, j] = prop.B66
                shell.D11[i, j] = prop.D11
                shell.D12[i, j] = prop.D12
                shell.D16[i, j] = prop.D16
                shell.D22[i, j] = prop.D22
                shell.D26[i, j] = prop.D26
                shell.D66[i, j] = prop.D66
        shell.init_k_KC0 = init_k_KC0
        shell.init_k_KG = init_k_KG
        init_k_KC0 += KC0_SPARSE_SIZE
        init_k_KG += KG_SPARSE_SIZE
        elements.append(shell)

    Kr = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kc = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kv = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for shell in elements:
        update_KC0(shell, points, weights, Kr, Kc, Kv)

    KC0 = coo_matrix((Kv, (Kr, Kc)), shape=(N, N)).tocsc()

    print('# finished element assembly')

    # applying boundary conditions
    bk = np.zeros(N, dtype=bool)

    checkSS = isclose(x, 0) | isclose(x, L)
    bk[0::DOF] = checkSS
    bk[3::DOF] = checkSS
    bk[6::DOF] = checkSS
    if clamped:
        bk[7::DOF] = checkSS
    bu = ~bk # same as np.logical_not, defining unknown DOFs

    print('# starting static analysis')

    # axial compression applied at x=L
    u0 = np.zeros(N, dtype=DOUBLE)

    compression = -0.0005
    checkTopEdge = isclose(x, L)
    u0[0::DOF] += checkTopEdge*compression
    uk = u0[bk]

    # sub-matrices corresponding to unknown DOFs
    KC0uu = KC0[bu, :][:, bu]
    KC0uk = KC0[bu, :][:, bk]
    KC0kk = KC0[bk, :][:, bk]

    fu = -KC0uk*uk

    Nu = N - bk.sum()

    # solving
    PREC = 1./KC0uu.diagonal().max()

    uu, info = cg(PREC*KC0uu, PREC*fu, x0=cg_x0, atol=0)
    if info != 0:
        print('#   failed with cg()')
        print('#   trying spsolve()')
        uu = spsolve(KC0uu, fu)
    cg_x0 = uu.copy()

    u0[bu] = uu

    print('# finished static analysis')

    KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for shell in elements:
        update_KG(u0, shell, points, weights, KGr, KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]

    # A * x[i] = lambda[i] * M * x[i]
    #NOTE this works and seems to be the fastest option

    print('# starting spilu')
    PREC2 = spilu(PREC*KC0uu, diag_pivot_thresh=0, drop_tol=1e-8,
            fill_factor=50)
    print('# finished spilu')
    def matvec(x):
        return PREC2.solve(x)
    Kuuinv = LinearOperator(matvec=matvec, shape=(Nu, Nu))

    print('# starting linear buckling analysis')

    maxiter = 1000
    if lobpcg_X is None:
        Xu = np.random.rand(Nu, num_eigvals)
        Xu /= np.linalg.norm(Xu, axis=0)
    else:
        Xu = lobpcg_X

    #NOTE default tolerance is too large
    tol = 1e-5
    eigvals, eigvecsu, hist = lobpcg(A=PREC*KC0uu, B=-PREC*KGuu, X=Xu, M=Kuuinv, largest=False,
            maxiter=maxiter, retResidualNormsHistory=True, tol=tol)
    load_mult = eigvals
    if not len(hist) <= maxiter:
        print('#   failed with lobpcg()')
        print('#   trying eigsh()')
        eigvals, eigvecsu = eigsh(A=KC0uu, k=num_eigvals, which='SM', M=KGuu,
                tol=1e-7, sigma=1., mode='buckling')
        load_mult = -eigvals

    print('# finished linear buckling analysis')

    force = np.zeros(N)
    fk = KC0uk.T*uu + KC0kk*uk
    force[bk] = fk
    Pcr = load_mult[0]*(force[0::DOF][checkTopEdge]).sum()
    print('# critical buckling load', Pcr)

    out = {}
    out['Pcr'] = Pcr
    out['cg_x0'] = cg_x0
    out['lobpcg_X'] = Xu
    out['mass'] = mass
    eigvecs = np.zeros((N, num_eigvals))
    eigvecs[bu, :] = eigvecsu
    out['eigvals'] = eigvals
    out['eigvecs'] = eigvecs
    out['xlin'] = xlin
    out['thetalin'] = thetalin
    out['t'] = t
    out['s'] = s
    out['c'] = c
    out['koiter'] = None

    if koiter_num_modes == 0:
        return out

    lambda_a = {}
    for modei in range(koiter_num_modes):
        lambda_a[modei] = eigvals[modei]

    es = partial(np.einsum, optimize='greedy')

    #NOTE making the maximum amplitude of the eigenmode equal to h
    #normalizing amplitude of eigenvector according to shell thickness
    ua = {}
    for modei in range(koiter_num_modes):
        ua[modei] = eigvecs[:, modei].copy()
        if np.isclose(abs(ua[modei][6::DOF].max()), abs(ua[modei][6::DOF].min())):
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
                eta = points[j]
                weight_eta = weights[j]
                weight = weight_xi * weight_eta

                elem.update_Nw_x(xi, eta)
                elem.update_Nw_y(xi, eta)
                elem.update_Bm(xi, eta)
                elem.update_Bb(xi, eta)

                w0_x = 0
                w0_y = 0
                for k in range(40):
                    w0_x += elem.Nw_x[k]*u0e[k]
                    w0_y += elem.Nw_y[k]*u0e[k]
                for k in range(40):
                    eia0[0, k] = w0_x*elem.Nw_x[k]
                    eia0[1, k] = w0_y*elem.Nw_y[k]
                    eia0[2, k] = w0_x*elem.Nw_y[k] + w0_y*elem.Nw_x[k]
                eib0 = eic0 = eia0

                for k in range(3):
                    ei0[k] = 0
                    ki0[k] = 0
                    for m in range(40):
                        ei0[k] += elem.Bm[k, m]*u0e[m] #NOTE ignoring NL terms
                        ki0[k] += elem.Bb[k, m]*u0e[m]
                    #TODO why lambda_i[0]?
                    ej0[k] = ei0[k]
                    kj0[k] = ki0[k]
                    ei[k] = ei0[k]*lambda_a[0]
                    ki[k] = ki0[k]*lambda_a[0]

                ei00[0] = ej00[0] = w0_x**2
                ei00[1] = ej00[1] = w0_y**2
                ei00[2] = ej00[2] = 2*w0_x*w0_y

                ki00 = 0

                A11 = elem.A11[i, j]
                A12 = elem.A12[i, j]
                A16 = elem.A16[i, j]
                A22 = elem.A22[i, j]
                A26 = elem.A26[i, j]
                A66 = elem.A66[i, j]
                B11 = elem.B11[i, j]
                B12 = elem.B12[i, j]
                B16 = elem.B16[i, j]
                B22 = elem.B22[i, j]
                B26 = elem.B26[i, j]
                B66 = elem.B66[i, j]
                D11 = elem.D11[i, j]
                D12 = elem.D12[i, j]
                D16 = elem.D16[i, j]
                D22 = elem.D22[i, j]
                D26 = elem.D26[i, j]
                D66 = elem.D66[i, j]

                Ni0[0] = (A11*ej0[0] * A12*ej0[1] + A16*ej0[2]
                        + B11*kj0[0] * B12*kj0[1] + B16*kj0[2])
                Ni0[1] = (A12*ej0[0] * A22*ej0[1] + A26*ej0[2]
                        + B12*kj0[0] * B22*kj0[1] + B26*kj0[2])
                Ni0[2] = (A16*ej0[0] * A26*ej0[1] + A66*ej0[2]
                        + B16*kj0[0] * B26*kj0[1] + B66*kj0[2])

                #TODO why lambda_a[0]?
                Ni[0] = Ni0[0]*lambda_a[0]
                Ni[1] = Ni0[1]*lambda_a[0]
                Ni[2] = Ni0[2]*lambda_a[0]

                Ni00[0] = A11*ej00[0] * A12*ej00[1] + A16*ej00[2]
                Ni00[1] = A12*ej00[0] * A22*ej00[1] + A26*ej00[2]
                Ni00[2] = A16*ej00[0] * A26*ej00[1] + A66*ej00[2]

                eia = eib = eic = np.asarray(elem.Bm) #NOTE ignoring NL terms
                kia = kib = kic = np.asarray(elem.Bb)

                Nia *= 0
                Mia *= 0
                Nia0 *= 0
                Mia0 *= 0
                for m in range(num_nodes*DOF):
                    Nia[0, m] += (A11*eia[0, m] + A12*eia[1, m] + A16*eia[2, m]
                                + B11*kia[0, m] + B12*kia[1, m] + B16*kia[2, m])
                    Nia[1, m] += (A12*eia[0, m] + A22*eia[1, m] + A26*eia[2, m]
                                + B12*kia[0, m] + B22*kia[1, m] + B26*kia[2, m])
                    Nia[2, m] += (A16*eia[0, m] + A26*eia[1, m] + A66*eia[2, m]
                                + B16*kia[0, m] + B26*kia[1, m] + B66*kia[2, m])
                    Mia[0, m] += (B11*eia[0, m] + B12*eia[1, m] + B16*eia[2, m]
                                + D11*kia[0, m] + D12*kia[1, m] + D16*kia[2, m])
                    Mia[1, m] += (B12*eia[0, m] + B22*eia[1, m] + B26*eia[2, m]
                                + D12*kia[0, m] + D22*kia[1, m] + D26*kia[2, m])
                    Mia[2, m] += (B16*eia[0, m] + B26*eia[1, m] + B66*eia[2, m]
                                + D16*kia[0, m] + D26*kia[1, m] + D66*kia[2, m])

                    Nia0[0, m] += A11*eia0[0, m] + A12*eia0[1, m] + A16*eia0[2, m]
                    Nia0[1, m] += A12*eia0[0, m] + A22*eia0[1, m] + A26*eia0[2, m]
                    Nia0[2, m] += A16*eia0[0, m] + A26*eia0[1, m] + A66*eia0[2, m]
                    Mia0[0, m] += B11*eia0[0, m] + B12*eia0[1, m] + B16*eia0[2, m]
                    Mia0[1, m] += B12*eia0[0, m] + B22*eia0[1, m] + B26*eia0[2, m]
                    Mia0[2, m] += B16*eia0[0, m] + B26*eia0[1, m] + B66*eia0[2, m]
                Nib = Nic = Nia
                Mib = Mia

                Nib0 = Nic0 = Nia0
                Mib0 = Mia0

                #eiab[0, :, :] = es('i,j->ij', elem.Nw_x, elem.Nw_x, out=eiab[0])
                #eiab[1, :, :] = es('i,j->ij', elem.Nw_y, elem.Nw_y, out=np.asarray(eiab[1]))
                #eiab[2, :, :] = es('i,j->ij', elem.Nw_x, elem.Nw_y) + es('i,j->ij', elem.Nw_y, elem.Nw_x)
                es('i,j->ij', elem.Nw_x, elem.Nw_x, out=np.asarray(eiab[0]))
                es('i,j->ij', elem.Nw_y, elem.Nw_y, out=np.asarray(eiab[1]))
                es('i,j->ij', elem.Nw_x, elem.Nw_y, out=np.asarray(eiab[2]))
                es('i,j->ij', elem.Nw_y, elem.Nw_x, out=np.asarray(eiab[2]))
                eicd = eibd = eibc = eiad = eiac = eiab

                Niab[...] = 0
                Miab[...] = 0
                for m in range(num_nodes*DOF):
                    for n in range(num_nodes*DOF):
                        Niab[0, m, n] += A11*eiab[0, m, n] + A12*eiab[1, m, n] + A16*eiab[2, m, n]
                        Niab[1, m, n] += A12*eiab[0, m, n] + A22*eiab[1, m, n] + A26*eiab[2, m, n]
                        Niab[2, m, n] += A16*eiab[0, m, n] + A26*eiab[1, m, n] + A66*eiab[2, m, n]
                        Miab[0, m, n] += B11*eiab[0, m, n] + B12*eiab[1, m, n] + B16*eiab[2, m, n]
                        Miab[1, m, n] += B12*eiab[0, m, n] + B22*eiab[1, m, n] + B26*eiab[2, m, n]
                        Miab[2, m, n] += B16*eiab[0, m, n] + B26*eiab[1, m, n] + B66*eiab[2, m, n]

                Niac = Niad = Nibc = Nibd = Nicd = Niab
                Miac = Miad = Mibc = Miab

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

    #phi2 = KC0
    phi2uu = phi2[bu, :][:, bu]

    phi2_ab = {}
    for modei in range(koiter_num_modes):
        left = ua[modei] @ phi2
        for modej in range(koiter_num_modes):
            phi2_ab[(modei, modej)] = left @ ua[modej]

    print()
    a_abc = {}
    for modei in range(koiter_num_modes):
        lambda_i = lambda_a[modei]
        for modej in range(koiter_num_modes):
            for modek in range(koiter_num_modes):
                a_ijk = -1./(2*lambda_i)*(phi3_ab[(modei, modej)] @ ua[modek])/(phi20_a[modei] @ ua[modei])
                a_abc[(modei, modej, modek)] = a_ijk
                print('# $a_%d%d%d$' % (modei+1, modej+1, modek+1), a_ijk)
    print()

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
