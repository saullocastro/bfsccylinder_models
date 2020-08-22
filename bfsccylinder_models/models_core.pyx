import numpy as np
from numpy import isclose, pi
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh, spsolve, cg, lobpcg, LinearOperator, spilu
from composites.laminate import read_stack
from bfsccylinder import (BFSCCylinder, update_KC0, update_KG, DOF, DOUBLE, INT,
KC0_SPARSE_SIZE, KG_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights

from .vatfunctions import theta_VAT_P_x


def flinearBucklingVATCylinder_x(L, R, nx, ny, E11, E22, nu12, G12, tow_thick, desvars,
            clamped=True, cg_x0=None, lobpcg_X=None):
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

    nint = 4
    points, weights = get_points_weights(nint=nint)

    num_elements = len(n1s)
    print('# number of elements,', num_elements)

    elements = []
    N = DOF*nx*ny
    print('# number of DOF,', N)
    init_k_KC0 = 0
    init_k_KG = 0
    laminaprop = (E11, E22, nu12, G12, G12, G12)
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
                plyt_local = tow_thick/np.cos(np.deg2rad(steering_angle))

                plyts.append(plyt_local)
                plyts.append(plyt_local)

            lam = read_stack(stack=stack,
                    plyts=plyts, laminaprop=laminaprop)
            for j in range(nint):
                shell.A11[i, j] = lam.ABD[0, 0]
                shell.A12[i, j] = lam.ABD[0, 1]
                shell.A16[i, j] = lam.ABD[0, 2]
                shell.A22[i, j] = lam.ABD[1, 1]
                shell.A26[i, j] = lam.ABD[1, 2]
                shell.A66[i, j] = lam.ABD[2, 2]
                shell.B11[i, j] = lam.ABD[0, 3]
                shell.B12[i, j] = lam.ABD[0, 4]
                shell.B16[i, j] = lam.ABD[0, 5]
                shell.B22[i, j] = lam.ABD[1, 4]
                shell.B26[i, j] = lam.ABD[1, 5]
                shell.B66[i, j] = lam.ABD[2, 5]
                shell.D11[i, j] = lam.ABD[3, 3]
                shell.D12[i, j] = lam.ABD[3, 4]
                shell.D16[i, j] = lam.ABD[3, 5]
                shell.D22[i, j] = lam.ABD[4, 4]
                shell.D26[i, j] = lam.ABD[4, 5]
                shell.D66[i, j] = lam.ABD[5, 5]
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
    PREC = 1/Kuu.diagonal().mean()

    uu, info = cg(PREC*Kuu, PREC*fu, x0=cg_x0, atol=0)
    cg_x0 = uu.copy()
    assert info == 0

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
    assert len(hist) <= maxiter
    load_mult = eigvals

    print('# finished linear buckling analysis')

    f = np.zeros(N)
    fk = Kuk.T*uu + Kkk*uk
    f[bk] = fk
    Pcr = load_mult[0]*(f[0::DOF][checkTopEdge]).sum()
    print('critical buckling load,', Pcr)

    out = {}
    out['cg_x0'] = cg_x0
    out['lobpcg_X'] = Xu

    return Pcr, out
