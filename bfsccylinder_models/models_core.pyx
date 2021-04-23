import numpy as np
from numpy import isclose, pi
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh, spsolve, cg, lobpcg, LinearOperator, spilu
from composites import laminated_plate
from bfsccylinder import (BFSCCylinder, update_KC0, update_KG, DOF, DOUBLE, INT,
KC0_SPARSE_SIZE, KG_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights

from .vatfunctions import theta_VAT_P_x


def flinearBucklingVATCylinder_x(L, R, nx, ny, E11, E22, nu12, G12, rho,
        h_tow, desvars, clamped=True, cg_x0=None, lobpcg_X=None, nint=4,
        num_eigvals=2):
    # geometry our FW cylinders
    circ = 2*pi*R # m

    nids = 1 + np.arange(nx*(ny+1))
    nids_mesh = nids.reshape(nx, ny+1)
    # closing the cylinder by reassigning last row of node-ids
    nids_mesh[:, -1] = nids_mesh[:, 0]
    nids = np.unique(nids_mesh)
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    xlin = np.linspace(0, L, nx)
    ytmp = np.linspace(0, circ, ny+1)
    ylin = np.linspace(0, circ-(ytmp[-1] - ytmp[-2]), ny)
    xmesh, ymesh = np.meshgrid(xlin, ylin)
    xmesh = xmesh.T
    ymesh = ymesh.T

    # getting nodes
    ncoords = np.vstack((xmesh.flatten(), ymesh.flatten())).T
    x = ncoords[:, 0]
    y = ncoords[:, 1]

    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

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
        shell.lex = L/(nx-1)
        shell.ley = circ/ny
        for i in range(nint):
            wi = weights[i]
            x1 = ncoords[nid_pos[n1]][0]
            x2 = ncoords[nid_pos[n2]][0]
            xi = points[i]
            xlocal = x1 + (x2 - x1)*(xi + 1)/2
            assert xlocal > x1 and xlocal < x2

            stack = []
            plyts = []
            for thetas in desvars:
                #NOTE min(thetas) is not strictly correct
                #     I kept it here for verification purposes against ABAQUS
                #     a better model is to do min( theta(x) )
                theta_min = min(thetas)

                theta_local = theta_VAT_P_x(xlocal, L, thetas)

                #balanced laminate
                stack.append(theta_local)
                stack.append(-theta_local)

                steering_angle = abs(theta_min - theta_local)
                plyt_local = h_tow/np.cos(np.deg2rad(steering_angle))

                plyts.append(plyt_local)
                plyts.append(plyt_local)

            offset = sum(plyts)/2.
            prop = laminated_plate(stack=stack,
                    plyts=plyts, laminaprop=laminaprop, offset=offset)
            for j in range(nint):
                wj = weights[j]
                weight = wi*wj
                mass += weight*shell.lex*shell.ley/4*prop.intrho

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
    u = np.zeros(N, dtype=DOUBLE)

    compression = -0.0005
    checkTopEdge = isclose(x, L)
    u[0::DOF] += checkTopEdge*compression
    uk = u[bk]

    # sub-matrices corresponding to unknown DOFs
    Kuu = KC0[bu, :][:, bu]
    Kuk = KC0[bu, :][:, bk]
    Kkk = KC0[bk, :][:, bk]

    fu = -Kuk*uk

    Nu = N - bk.sum()

    # solving
    PREC = 1/Kuu.diagonal().max()

    uu, info = cg(PREC*Kuu, PREC*fu, x0=cg_x0, atol=0)
    if info != 0:
        print('#   failed with cg()')
        print('#   trying spsolve()')
        uu = spsolve(Kuu, fu)
    cg_x0 = uu.copy()

    u[bu] = uu

    print('# finished static analysis')

    KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for shell in elements:
        update_KG(u, shell, points, weights, KGr, KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]

    # A * x[i] = lambda[i] * M * x[i]
    #NOTE this works and seems to be the fastest option

    print('# starting spilu')
    PREC2 = spilu(PREC*Kuu, diag_pivot_thresh=0, drop_tol=1e-8,
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
    eigvals, eigvecsu, hist = lobpcg(A=PREC*Kuu, B=-PREC*KGuu, X=Xu, M=Kuuinv, largest=False,
            maxiter=maxiter, retResidualNormsHistory=True, tol=tol)
    load_mult = eigvals
    if not len(hist) <= maxiter:
        print('#   failed with lobpcg()')
        print('#   trying eigsh()')
        eigvals, eigvecsu = eigsh(A=Kuu, k=num_eigvals, which='SM', M=KGuu,
                tol=1e-7, sigma=1., mode='buckling')
        load_mult = -eigvals

    print('# finished linear buckling analysis')

    force = np.zeros(N)
    fk = Kuk.T*uu + Kkk*uk
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
    out['eigvecs'] = eigvecs

    return out


def flinearBucklingCTS_circum(L, R, rCTS, nxt, ny, E11, E22, nu12, G12, rho,
        h_tow, n, f, thetadeg_c, thetadeg_s, clamped=True, cg_x0=None, lobpcg_X=None, nint=4):
    assert abs(thetadeg_s) > abs(thetadeg_c)
    assert f > 0
    t = rCTS*np.sin(np.deg2rad(thetadeg_s - thetadeg_s))
    nmax = L/(2*t)
    assert n >= nmax
    cmax = (L - 2*t*n)/(n+1)
    if not np.isclose(cmax, 0):
        s = f/(n*(f + 1))*(L-2*t*n)
        c = 1/(f*n + f + n + 1)*(L-2*t*n)
    else:
        s = 0
        c = 0
    assert np.isclose((2*t + s)*n + c*(n+1) - L, 0)

    # geometry our FW cylinders
    circ = 2*pi*R # m

    nxt = 3
    nxs = max(1, int(round(s/t*nxt, 0)))
    nxc = max(1, int(round(c/t*nxt, 0)))

    dx = t/(nxt-1)
    if ny is None:
        ny = int(round(circ/dx, 0))

    xlin = np.linspace(0, c, nxc-1, endpoint=False)
    thetalin = np.ones(nxc-1)*thetadeg_c
    for i in range(n):
        start = c + i*(c+2*t+s)
        xlin = np.concatenate((xlin, np.linspace(start, start+t, nxt-1, endpoint=False)))
        thetalin = np.concatenate((thetalin, thetadeg_c + np.linspace(0, 1, nxt-1, endpoint=False)*(thetadeg_s - thetadeg_c)))
        if nxs > 1:
            xlin = np.concatenate((xlin, np.linspace(start+t, start+t+s, nxs-1, endpoint=False)))
            thetalin = np.concatenate((thetalin, np.ones(nxs-1)*thetadeg_s))
        xlin = np.concatenate((xlin, np.linspace(start+t+s, start+t+s+t, nxt-1, endpoint=False)))
        thetalin = np.concatenate((thetalin, thetadeg_s + np.linspace(0, 1, nxt-1, endpoint=False)*(thetadeg_c - thetadeg_s)))
        if i == n-1:
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
    nids_mesh[:, -1] = nids_mesh[:, 0]
    nids = np.unique(nids_mesh)
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    ytmp = np.linspace(0, circ, ny+1)
    ylin = np.linspace(0, circ-(ytmp[-1] - ytmp[-2]), ny)
    xmesh, ymesh = np.meshgrid(xlin, ylin)
    xmesh = xmesh.T
    ymesh = ymesh.T

    # getting nodes
    ncoords = np.vstack((xmesh.flatten(), ymesh.flatten())).T
    x = ncoords[:, 0]
    y = ncoords[:, 1]

    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

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
            xlocal = x1 + (x2 - x1)*(xi + 1)/2
            assert xlocal > x1 and xlocal < x2

            theta_local = np.interp(xlocal, xlin, thetalin)
            steering_angle = theta_local - thetadeg_c
            plyt_local = h_tow / np.cos(np.deg2rad(steering_angle))

            #balanced laminate
            stack = (theta_local, -theta_local)
            plyts = (plyt_local, plyt_local)

            offset = sum(plyts)/2.
            prop = laminated_plate(stack=stack, plyts=plyts, laminaprop=laminaprop, offset=offset)
            for j in range(nint):
                wj = weights[j]
                weight = wi*wj
                mass += weight*shell.lex*shell.ley/4*prop.intrho

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
    u = np.zeros(N, dtype=DOUBLE)

    compression = -0.0005
    checkTopEdge = isclose(x, L)
    u[0::DOF] += checkTopEdge*compression
    uk = u[bk]

    # sub-matrices corresponding to unknown DOFs
    Kuu = KC0[bu, :][:, bu]
    Kuk = KC0[bu, :][:, bk]
    Kkk = KC0[bk, :][:, bk]

    fu = -Kuk*uk

    Nu = N - bk.sum()

    # solving
    PREC = 1/Kuu.diagonal().max()

    uu, info = cg(PREC*Kuu, PREC*fu, x0=cg_x0, atol=0)
    if info != 0:
        print('#   failed with cg()')
        print('#   trying spsolve()')
        uu = spsolve(Kuu, fu)
    cg_x0 = uu.copy()

    u[bu] = uu

    print('# finished static analysis')

    KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for shell in elements:
        update_KG(u, shell, points, weights, KGr, KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]

    # A * x[i] = lambda[i] * M * x[i]
    num_eigvals = 2
    #NOTE this works and seems to be the fastest option

    print('# starting spilu')
    PREC2 = spilu(PREC*Kuu, diag_pivot_thresh=0, drop_tol=1e-8,
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
    eigvals, eigvecsu, hist = lobpcg(A=PREC*Kuu, B=-PREC*KGuu, X=Xu, M=Kuuinv, largest=False,
            maxiter=maxiter, retResidualNormsHistory=True, tol=tol)
    load_mult = eigvals
    if not len(hist) <= maxiter:
        print('#   failed with lobpcg()')
        print('#   trying eigsh()')
        eigvals, eigvecsu = eigsh(A=Kuu, k=num_eigvals, which='SM', M=KGuu,
                tol=1e-7, sigma=1., mode='buckling')
        load_mult = -eigvals

    print('# finished linear buckling analysis')

    force = np.zeros(N)
    fk = Kuk.T*uu + Kkk*uk
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
    out['eigvecs'] = eigvecs
    out['xlin'] = xlin
    out['thetalin'] = thetalin
    out['t'] = t
    out['s'] = s
    out['c'] = c

    return out
