import numpy as np
from numpy import isclose, pi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from composites import laminated_plate
from bfsccylinder import (BFSCCylinder, update_KC0, update_KG, update_M, DOF,
        DOUBLE, INT, KC0_SPARSE_SIZE, KG_SPARSE_SIZE, M_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder_models.edges import edge_space, mass_matrix

num_nodes = 4


def flinBuck_VAFW(L, R, nx, ny, E11, E22, nu12, G12, rho,
        h_tow, desvars, funcVAT, nint=4, num_eigvals=2, Nxxunit=1.):
    """
    Linear buckling analysis of a VAT cylinder with properties changing over
    the axial direction (x)

    Assumptions:
    - classical shell theory (when BFS element is used)
    - monolithic laminated properties (only one material for the whole laminate)
    - SS3 edges with inertia relief, see ``bfsccylinder_models.edges``:
      v = w = 0 along both edges, the axial load Nxxunit applied on both
      edges, no node anchored, the axial translation removed by inertia
      relief
    - linear pre-buckling state
    - returns the critical buckling load in consistent force units

    Parameters
    ----------
    L : float
        Cylinder length.
    R : float
        Cylinder radius.
    nx : int
        Number of nodes along axial direction (odd number recommended).
    ny : int
        Number of nodes along circumferential direction (even number
        recommended).
    E11, E22, nu12, G12 : float
        Orthotropic material properties.
    rho : float
        Density of orthotropic material.
    h_tow : float
        FW tow thickness.
    desvars : list
        Each element of desvars is another list containing the variables
        compatible with the VAT function ``funcVAT`` being used.
    funcVAT : function
        VAT function in the form ``f(x, xmax, thetas)``, with ``x`` being the
        axial direction, ``xmax`` the maximum value of ``x`` in the domain, and
        ``thetas`` the angle values at the control points, such that the
        ``desvars`` parameter is a sequence of ``thetas``.
    nint : int, optional
        Number of integration points per direction.
    num_eigvals : int, optional
        Number of eigenvalues to extract.
    Nxxunit : float, optional
        Axial compressive load per unit circumferential length of the
        pre-buckling state, the load of load multiplier 1.

    Returns
    -------
    out : dict
        out['Pcr'] = critical buckling load
        out['load_mult'] = load multipliers of Nxxunit, ascending
        out['mass'] = mass
        out['eigvecs'] = buckling modes, as columns
        out['x'], out['y'] = nodal coordinates

    """
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
    havg_elements = []
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
        havg_elem = 0
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

                theta_local = funcVAT(xlocal, L, thetas)

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
                havg_elem += weight/4*sum(plyts)

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
        havg_elements.append(havg_elem)
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

    #NOTE SS3 edges with inertia relief, no node anchored, see edges.py: the
    #     axial translation is removed in the metric of the consistent mass
    #     matrix, at unit density, the density being uniform, which only
    #     scales it
    space = edge_space(x, L, DOF)
    space.set_mass(mass_matrix(elements, update_M, M_SPARSE_SIZE, N,
            havg_elements, [1.]*num_elements))

    print('# starting static analysis')

    # axially compressive load applied at x=0 and x=L
    fext = np.zeros(N)
    for shell in elements:
        pos1 = nid_pos[shell.n1]
        pos3 = nid_pos[shell.n3]
        if isclose(x[pos3], L):
            Nxx = -Nxxunit
            xi = +1
        elif isclose(x[pos1], 0):
            Nxx = +Nxxunit
            xi = -1
        else:
            continue
        indices = []
        for ci in [shell.c1, shell.c2, shell.c3, shell.c4]:
            for i in range(DOF):
                indices.append(ci + i)
        fe = np.zeros(num_nodes*DOF, dtype=float)
        for j in range(nint):
            eta = points[j]
            shell.update_Su(xi, eta)
            fe += shell.ley/2.*weights[j]*np.asarray(shell.Su)*Nxx
        fext[indices] += fe
    assert isclose(fext.sum(), 0)

    KC0uu = space.matrix(KC0)
    u = space.expand(space.solve(KC0uu, space.force(fext)))

    print('# finished static analysis')

    KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for shell in elements:
        update_KG(u, shell, points, weights, KGr, KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = space.matrix(KG)

    print('# starting linear buckling analysis')
    #NOTE KG q = theta KC0 q on the null space of the inertia relief
    #     condition, the largest theta in magnitude being the smallest load
    #     multipliers, -1/theta
    v0 = np.random.default_rng(0).random(space.size)
    eigvals, eigvecsu = space.eigsh(KGuu, KC0uu, num_eigvals, v0, 1e-7,
            eigsh)
    load_mult = -1/eigvals
    order = np.argsort(load_mult)
    load_mult = load_mult[order]
    eigvecsu = eigvecsu[:, order]
    print('# finished linear buckling analysis')

    Pcr = load_mult[0]*Nxxunit*circ
    print('# critical buckling load', Pcr)

    out = {}
    out['Pcr'] = Pcr
    out['load_mult'] = load_mult
    out['mass'] = mass
    out['eigvecs'] = space.expand(eigvecsu)
    out['x'] = x
    out['y'] = y

    return out
