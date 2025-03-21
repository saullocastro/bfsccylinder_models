import gc
from functools import partial
from collections import defaultdict

try:
    from pypardiso import spsolve
except ImportError:
    from scipy.sparse.linalg import spsolve

import numpy as np
from numpy import isclose
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from bfsccylinder import (BFSCCylinder, update_KC0, update_KCNL, update_KG,
        update_fint, DOF, DOUBLE, INT, KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE,
        KG_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD

num_nodes = 4


def fkoiter_cyl_SS3(L, R, nx, ny, prop, cg_x0=None, nint=4,
        num_eigvals=2, koiter_num_modes=1, Nxxunit=1., NLprebuck=False):

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
            elem.update_Nu(xi, eta)
            fe += ley/2.*weights[j]*elem.Nu*Nxx
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
        del KCNLv, KCNLr, KCNLc
        gc.collect()

        KC = KC0 + KCNL
        KCuu = KC[bu, :][:, bu]

    else:
        KC = KC0
        KCuu = KC0uu

    print('# finished static analysis')

    #NOTE u0 represents the latest linear or nonlinear pre-buckling state

    KGv *= 0
    for elem in elements:
        update_KG(u0, elem, points, weights, KGr, KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]

    print('# starting eigenvalue analysis')
    #eigvals, eigvecsu = eigsh(A=KCuu, k=num_eigvals, which='SM', M=KGuu,
            #tol=1e-8, sigma=1., mode='buckling')
    #load_mult = eigvals
    eigvals, eigvecsu = eigsh(A=KGuu, k=num_eigvals, which='LM', M=KCuu,
            tol=1e-6)
    load_mult = -1/eigvals
    print('# finished eigenvalue analysis')

    Pcr = load_mult[0]*Nxxunit*circ
    print('# load_mult', load_mult)
    print('# critical buckling load', Pcr)

    out['Pcr'] = Pcr
    out['cg_x0'] = cg_x0
    out['eigvals'] = eigvals
    out['load_mult'] = load_mult
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


    #NOTE I noticed that, in order to get DIANA's result, the linear
    #     pre-buckling state ignores all nonlinear quantities in the
    #     calculations of the strains and its derivatives
    #     Therefore, I am using this pythflag that multiply the referred nonlinear
    #     terms
    flag = NLprebuck

    # higher-order tensors for elements

    u0e = np.zeros(num_nodes*DOF, dtype=np.float64)
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

                #NOTE, added NL terms
                ei0 = ej0 = Bm @ u0e + flag*np.array([lambda_a[0]*w0_x**2,
                                                      lambda_a[0]*w0_y**2,
                                                      lambda_a[0]*2*w0_x*w0_y])
                ki0 = kj0 = Bb @ u0e

                ##TODO why lambda_i[0]?
                #ei = ei0*lambda_a[0]
                #ki = ki0*lambda_a[0]

                ei00 = ej00 = flag*np.array([w0_x**2,
                                             w0_y**2,
                                             2*w0_x*w0_y])

                Ni0 = Aij@ej0 + Bij@kj0
                Ni00 = Aij@ej00

                ##TODO why lambda_a[0]?
                #Ni = Ni0*lambda_a[0]

                #NOTE, added NL terms
                eia = eib = eic = Bm + flag*lambda_a[0]*np.array([w0_x*Nw_x[0],
                                                                  w0_y*Nw_y[0],
                                                                  w0_x*Nw_y[0] + w0_y*Nw_x[0]])

                kia = kib = kic = Bb

                Nia = Nib = Nic = es('ij,ja->ia', Aij, eia) + es('ij,ja->ia', Bij, kia)
                #Mia = Mib = es('ij,ja->ia', Bij, eia) + es('ij,ja->ia', Dij, kia)

                eia0 = eib0 = eic0 = flag*np.array([w0_x*Nw_x[0],
                                                    w0_y*Nw_y[0],
                                                    w0_x*Nw_y[0] + w0_y*Nw_x[0]])

                Nia0 = Nib0 = Nic0 = es('ij,ja->ia', Aij, eia0)
                Mia0 = Mib0 = es('ij,ja->ia', Bij, eia0)

                eiab[0] = Nw_x.T @ Nw_x
                eiab[1] = Nw_y.T @ Nw_y
                eiab[2] = Nw_x.T @ Nw_y + Nw_y.T @ Nw_x

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

    #TODO phi2uu = phi2[bu, :][:, bu]
    #if NLprebuck:
        #phi2 = KC + KG #TODO with KG?
        #phi2uu = KCuu + KGuu #TODO with KGuu?
    #else:
    if flag == 1:
        KCNLr = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=INT)
        KCNLc = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=INT)
        KCNLv = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        KCNLv *= 0
        for elem in elements:
            update_KCNL(u0*lambda_a[0], elem, points, weights, KCNLr, KCNLc, KCNLv)
        KCNL = coo_matrix((KCNLv, (KCNLr, KCNLc)), shape=(N, N)).tocsc()
        KC = KC0 + KCNL
        KCuu = KC[bu, :][:, bu]

    #NOTE I checked and phi2 can be really defined using K + KNL + KG
    phi2 = KC + KG*lambda_a[0]
    phi2uu = KCuu + KGuu*lambda_a[0]

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
            uijbar[bu] = spsolve(phi2uu, force2ndorder_ij[(modei, modej)][bu])
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
