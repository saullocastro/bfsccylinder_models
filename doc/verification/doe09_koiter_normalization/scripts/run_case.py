"""Runs one case of DOE09.txt with a linear (LIN) or a nonlinear (NL)
pre-buckling state

usage: python run_case.py ICASE LIN|NL [NY]

NY overrides ny below, for the convergence study. The last line printed is
"RESULT {json}", read by post.py and post_convergence.py
"""
import os
import sys
import json
import time
import traceback

import numpy as np
#NOTE the elision of large temporaries by numpy calls backtrace(), which
#     dlopens libgcc_s the first time. With the MKL 2024 libiomp5 of pip over
#     the MKL 2023 of conda, that dlopen segfaulted on the glibc 2.17 of the
#     cluster once PARDISO had run (ny >= 80). Loading libgcc_s here, before
#     MKL runs, avoids it
np.ones(10**6)*2 + 1.
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import triu
import bfsccylinder_models
import bfsccylinder_models.koiter_cylinder_CTS_sanders as model
from bfsccylinder_models.koiter_cylinder_CTS_sanders import fkoiter_cylinder_CTS_circum
from bfsccylinder_models.cyclic_symmetry import mesh_order, rotated

import koiter_post

DOE_name = 'DOE09'
DOF = 10

#NOTE mesh, see the convergence study (DOE09_convergence.txt): with
#     NLprebuck=True, Pcr at ny=160 is within 0.2, 0.1 and 0.5 per cent of
#     ny=200 for cases 0, 1 and 6, while b_factor still changes by 6, 7 and
#     20 per cent
ny = 160 #NOTE number of elements around circumference, nxt from choose_nxt

#NOTE relative residual above which a PARDISO solution is rejected
max_residual = 1.e-8

#NOTE multi-mode Koiter expansion on koiter_num_distinct modes distinct up to
#     the rotation of the cylinder, see use_distinct_modes. num_eigvals must
#     leave room for the rotated partner of each of them. With
#     koiter_rotation_closed the rotated partner of each distinct mode is
#     added to the expansion, koiter_num_modes = 2*koiter_num_distinct
koiter_num_distinct = 5
koiter_rotation_closed = True
koiter_num_modes = (2 if koiter_rotation_closed else 1)*koiter_num_distinct
num_eigvals = 12


def use_distinct_modes():
    """Koiter modes distinct up to the rotation of the cylinder

    The stiffness of these cylinders is uniform around the circumference, so
    every buckling mode is a single circumferential harmonic and, but for the
    axisymmetric ones, comes with a rotated partner of the same multiplier,
    which carries no new information into the expansion. The eigen solver
    returns both members of a pair, one member, or a member and the next
    mode, as decided by round off, so taking the first koiter_num_modes
    eigenvectors retains a random mixture of distinct modes and partners.

    canonical_modes of the model is wrapped so that, after it has fixed the
    rotation of every pair, the modes are reordered: first those not in the
    span of an earlier mode and of its rotation (degenerate_partner), in
    increasing multiplier, then the others. The multipliers are reordered in
    place with them, mu being the array the model keeps using. The number of
    distinct modes of the last eigenvalue analysis is returned in a list.

    With koiter_rotation_closed, every distinct mode is followed by its
    rotated partner instead, so that the Koiter modes span the rotations of
    each other. The partners do carry information into a multi-mode
    expansion: the interaction of two distinct modes depends on their
    relative rotation, which canonical_modes fixes by a convention, and when
    the two members of a pair of symmetric and antisymmetric modes, localized
    at the two ends, share a multiplier to round off, as for DOE09 case 1
    with NLprebuck, their rotations span a four dimensional eigenspace of
    which the eigen solver returns an arbitrary two dimensional slice. The
    expansion over the closed set is independent of that slice, and of the
    convention. An axisymmetric mode has no partner and takes one column
    """
    canonical_modes = model.canonical_modes
    num_distinct = [None]
    #NOTE with koiter_rotation_closed, True for the Koiter modes that are
    #     distinct modes and False for the partners, a list in num_distinct[1]
    num_distinct.append(None)
    #NOTE KCuu of the last eigenvalue analysis, the metric in which the
    #     closed set of Koiter modes is made orthonormal
    eigsh = model.eigsh
    KCuu = [None]

    def keep_KC(A, k, which, M, tol, v0):
        KCuu[0] = M
        return eigsh(A=A, k=k, which=which, M=M, tol=tol, v0=v0)

    model.eigsh = keep_KC

    def orthonormal(v, cols):
        """v made KCuu-orthonormal to cols, and of unit KCuu norm"""
        M = KCuu[0]
        for c in cols:
            v = v - (c @ (M @ v))*c
        return v/np.sqrt(v @ (M @ v))

    def distinct_first(mu, eigvecsu, bu, axi_order, DOF, deg_rtol=1.e-5):
        eigvecsu = canonical_modes(mu, eigvecsu, bu, axi_order, DOF,
                deg_rtol=deg_rtol)
        basis = np.zeros((bu.shape[0], 0))
        distinct = []
        twins = []
        for k in range(eigvecsu.shape[1]):
            phi = np.zeros(bu.shape[0])
            phi[bu] = eigvecsu[:, k]
            if basis.shape[1] > 0:
                coef = np.linalg.lstsq(basis, phi, rcond=None)[0]
                if (np.linalg.norm(phi - basis @ coef)
                        < 1.e-3*np.linalg.norm(phi)):
                    twins.append(k)
                    continue
            distinct.append(k)
            vecs = [phi]
            psi = model.degenerate_partner(phi, bu, axi_order, DOF)
            if psi is not None:
                vecs.append(psi)
            basis = np.column_stack([basis] + vecs)
        num_distinct[0] = len(distinct)
        if not koiter_rotation_closed:
            perm = distinct + twins
            mu[:] = np.asarray(mu)[perm]
            return eigvecsu[:, perm]
        #NOTE the expansion of the models takes the Koiter modes orthogonal in
        #     the metric of the load term, d_ij = 0 for i != j, which the
        #     eigenvectors are, but a partner built by degenerate_partner is
        #     orthogonal to its mode only, and not to the other members of a
        #     four dimensional eigenspace. Every column is therefore made
        #     KCuu-orthonormal to the previous ones, which for eigenvectors of
        #     one eigenvalue is the orthogonality in both matrices, and the
        #     partner is built from the mode so made, whose rotations are
        #     then orthogonal to the previous columns as well
        cols, mus, used, flags = [], [], [], []
        for k in distinct:
            if len(cols) + 1 > koiter_num_modes:
                break
            phi = np.zeros(bu.shape[0])
            phi[bu] = orthonormal(eigvecsu[:, k], cols)
            psi = model.degenerate_partner(phi, bu, axi_order, DOF)
            pair = [phi[bu]]
            if psi is not None:
                pair.append(orthonormal(psi[bu], cols + pair))
            if len(cols) + len(pair) > koiter_num_modes:
                break
            cols += pair
            mus += [mu[k]]*len(pair)
            used.append(k)
            flags += [True] + [False]*(len(pair) - 1)
        num_distinct[1] = flags
        #NOTE the rest of the columns, not Koiter modes, keep the width
        rest = [k for k in distinct + twins if k not in used]
        rest = rest[:eigvecsu.shape[1] - len(cols)]
        mu[:] = np.array(mus + [mu[k] for k in rest])
        return np.column_stack(cols + [eigvecsu[:, k] for k in rest])

    model.canonical_modes = distinct_first
    return num_distinct


def use_safe_solvers():
    """Linear solvers of fkoiter_cylinder_CTS_circum, returns their name

    The model imports pypardiso.spsolve whenever pypardiso is installed.
    Its default, the nonsymmetric factorization, returns a wrong solution for
    the pre-buckling system KC0uu u = f, whose diagonal spans 12 orders of
    magnitude (relative residual 2 for DOE09 case 0 at ny=80, against 1e-11
    with SuperLU). The Cholesky factorization of PARDISO is accurate for it
    (2e-13). The nonsymmetric one is not reliable for the bordered system of
    the second order fields either, 2e-15 for case 6 but 6e-3 for case 0,
    see lu_gmres. SuperLU is accurate for both, but its fill in limits ny:
    15 GB at ny=80 and 17.6 GB for the factorization of KCuu alone, done by
    eigsh, at ny=160, for case 0.

    Here symmetric matrices are factorized with Cholesky, the others with
    lu_gmres, eigsh uses the Cholesky factors of KCuu, and every PARDISO
    solution is checked against its residual, SuperLU being the fallback
    """
    try:
        import pypardiso
    except ImportError:
        model.spsolve = scipy.sparse.linalg.spsolve
        return 'superlu'

    def residual(A, x, b):
        return np.linalg.norm(A @ x - b)/np.linalg.norm(b)

    def is_symmetric(A):
        return scipy.sparse.linalg.norm(A - A.T) <= 1.e-12*scipy.sparse.linalg.norm(A)

    def cholesky(A):
        solver = pypardiso.PyPardisoSolver(mtype=2)
        Au = triu(A, format='csr')
        solver.factorize(Au)
        return solver, Au

    def lu_gmres(A, b):
        """Nonsymmetric PARDISO on the symmetrically scaled matrix, as the
        preconditioner of GMRES

        For the bordered system of DOE09 case 0 at ny=80, PARDISO alone leaves
        a relative residual of 6e-3 (8e-4 scaled), and the scaled factors as
        a preconditioner bring it to 8e-13 in 128 GMRES iterations
        """
        dg = np.abs(A.diagonal())
        dg[dg == 0] = 1.
        s = 1/np.sqrt(dg)
        D = scipy.sparse.diags(s)
        As = (D @ A @ D).tocsr()
        solver = pypardiso.PyPardisoSolver(mtype=11)
        solver.factorize(As)
        def precond(r):
            return s*solver.solve(As, np.ascontiguousarray(s*r))
        M = scipy.sparse.linalg.LinearOperator(A.shape, matvec=precond,
                dtype=np.float64)
        x0 = precond(b)
        try:
            x, info = scipy.sparse.linalg.gmres(A, b, x0=x0, M=M, rtol=1.e-12,
                    atol=0., restart=30, maxiter=20)
        except TypeError:
            #NOTE scipy < 1.12
            x, info = scipy.sparse.linalg.gmres(A, b, x0=x0, M=M, tol=1.e-12,
                    atol=0., restart=30, maxiter=20)
        solver.free_memory(everything=True)
        return x

    def spsolve(A, b):
        if is_symmetric(A):
            solver, Au = cholesky(A)
            x = solver.solve(Au, b)
            solver.free_memory(everything=True)
        else:
            x = lu_gmres(A, b)
        if residual(A, x, b) > max_residual:
            print('# WARNING: PARDISO residual %.1e, solving again with SuperLU'
                    % residual(A, x, b))
            x = scipy.sparse.linalg.spsolve(A, b)
        return x

    def eigsh(A, k, which, M, tol, v0):
        solver, Mu = cholesky(M)
        Minv = scipy.sparse.linalg.LinearOperator(M.shape, dtype=np.float64,
                matvec=lambda x: solver.solve(Mu, np.ascontiguousarray(x)))
        x = solver.solve(Mu, v0)
        if residual(M, x, v0) > max_residual:
            solver.free_memory(everything=True)
            print('# WARNING: PARDISO Cholesky residual %.1e in eigsh, using '
                    'SuperLU' % residual(M, x, v0))
            return scipy.sparse.linalg.eigsh(A=A, k=k, which=which, M=M,
                    tol=tol, v0=v0)
        out = scipy.sparse.linalg.eigsh(A=A, k=k, which=which, M=M, Minv=Minv,
                tol=tol, v0=v0)
        solver.free_memory(everything=True)
        return out

    model.spsolve = spsolve
    model.eigsh = eigsh
    return 'pardiso'


def choose_nxt(L, R, ny, rCTS, param_n, c2_ratio, thetadeg_c1, thetadeg_c2,
        c1_threshold_factor=0.01, c2_threshold_factor=0.01):
    """Nodes along each transition region, and max_ny_nx_aspect_ratio

    Chosen so that no element of the transition or plateau regions is longer
    than the circumferential element length dy. fkoiter_cylinder_CTS_circum
    gives a plateau of length c round(c/t*nxt) nodes, at least 2, or
    round(c/(dy/max_ny_nx_aspect_ratio)) nodes, at least 2, when the
    transition elements are more than max_ny_nx_aspect_ratio times shorter
    than dy. Either way a short plateau can get a single element, which is
    what the two parameters returned avoid. t, c1 and c2 are computed as in
    fkoiter_cylinder_CTS_circum
    """
    dy = 2*np.pi*R/ny
    if param_n == 0 or np.isclose(thetadeg_c1, thetadeg_c2):
        #NOTE constant stiffness, meshed from ny alone
        return 3, 2
    t = rCTS*np.sin(abs(np.deg2rad(thetadeg_c2 - thetadeg_c1)))
    param_n = min(param_n, int(L/(2*t)))
    c2 = c2_ratio*(L - 2*t*param_n)/param_n
    if c2 < c2_threshold_factor*L:
        c2 = 0
    c1 = (L - (2*t + c2)*param_n)/(param_n + 1)
    if c1 < c1_threshold_factor*L:
        c1 = 0
    t = ((L - c1*(param_n+1))/param_n - c2)/2
    plateaus = [c for c in [c1, c2] if c > 0]
    nxt = max(3, int(np.ceil(t/dy)) + 1)
    for c in plateaus:
        nxt = max(nxt, int(np.ceil((np.ceil(c/dy) + 0.5)*t/c)))

    def plateau_dx(c, max_ny_nx_aspect_ratio):
        if dy/(t/(nxt - 1)) > max_ny_nx_aspect_ratio:
            nodes = max(2, int(round(c/(dy/max_ny_nx_aspect_ratio), 0)))
        else:
            nodes = max(2, int(round(c/t*nxt, 0)))
        return c/(nodes - 1)

    max_ny_nx_aspect_ratio = 2
    while any(plateau_dx(c, max_ny_nx_aspect_ratio) > dy for c in plateaus):
        max_ny_nx_aspect_ratio += 1
    return nxt, max_ny_nx_aspect_ratio


def estimate_nx(L, R, ny, rCTS, param_n, c2_ratio, thetadeg_c1, thetadeg_c2,
        c1_threshold_factor=0.01, c2_threshold_factor=0.01):
    """Axial stations of the mesh of design_function, to within a few

    Used by generate_qsubs.py to estimate the time and memory of each run
    without assembling the model. t, c1, c2 and the plateau nodes as in
    choose_nxt
    """
    circ = 2*np.pi*R
    dy = circ/ny
    if param_n == 0 or np.isclose(thetadeg_c1, thetadeg_c2):
        nx = int(ny*L/circ)
        return nx + 1 if nx % 2 == 0 else nx
    nxt, max_ny_nx_aspect_ratio = choose_nxt(L, R, ny, rCTS, param_n,
            c2_ratio, thetadeg_c1, thetadeg_c2)
    t = rCTS*np.sin(abs(np.deg2rad(thetadeg_c2 - thetadeg_c1)))
    param_n = min(param_n, int(L/(2*t)))
    c2 = c2_ratio*(L - 2*t*param_n)/param_n
    if c2 < c2_threshold_factor*L:
        c2 = 0
    c1 = (L - (2*t + c2)*param_n)/(param_n + 1)
    if c1 < c1_threshold_factor*L:
        c1 = 0
    t = ((L - c1*(param_n+1))/param_n - c2)/2

    def plateau_nodes(c):
        if dy/(t/(nxt - 1)) > max_ny_nx_aspect_ratio:
            return max(2, int(round(c/(dy/max_ny_nx_aspect_ratio), 0)))
        return max(2, int(round(c/t*nxt, 0)))

    nx = 1 + 2*param_n*(nxt - 1)
    if c1 > 0:
        nx += (param_n + 1)*(plateau_nodes(c1) - 1)
    if c2 > 0:
        nx += param_n*(plateau_nodes(c2) - 1)
    return nx


def design_function(variables, constants):
    rCTS = variables['rCTS']
    param_n = variables['param_n']
    c2_ratio = variables['c2_ratio']
    thetadeg_c1 = variables['thetadeg_c1']
    thetadeg_c2 = variables['thetadeg_c2']
    #NOTE R and L could be variables
    L = constants['L']
    R = constants['R']
    ny = constants['ny']
    E11 = constants['E11']
    E22 = constants['E22']
    nu12 = constants['nu12']
    G12 = constants['G12']
    tow_thick = constants['tow_thick']
    rho = constants['rho']
    mesh_only = constants['mesh_only']
    Nxxunit = constants['Nxxunit']
    NLprebuck = constants['NLprebuck']

    #NOTE a fixed nxt gives elements 93 mm long next to elements 0.03 mm
    #     long across DOE09
    nxt, max_ny_nx_aspect_ratio = choose_nxt(L, R, ny, rCTS, param_n,
            c2_ratio, thetadeg_c1, thetadeg_c2)

    out = fkoiter_cylinder_CTS_circum(L, R, rCTS, nxt, ny, E11, E22, nu12, G12,
            rho, tow_thick, param_n, c2_ratio, thetadeg_c1, thetadeg_c2,
            mesh_only=mesh_only, Nxxunit=Nxxunit,
            num_eigvals=num_eigvals, koiter_num_modes=koiter_num_modes,
            NLprebuck=NLprebuck,
            max_ny_nx_aspect_ratio=max_ny_nx_aspect_ratio)
    out['nxt'] = nxt
    out['max_ny_nx_aspect_ratio'] = max_ny_nx_aspect_ratio
    return out


def mode_harmonics(out, k):
    """Circumferential wave number of mode k and its axisymmetric share"""
    w = out['eigvecs'][:, k].reshape(out['nx'], out['ny'], DOF)[:, :, 6]
    P = (np.abs(np.fft.rfft(w, axis=1))**2).sum(axis=0)
    P = P/P.sum()
    return int(np.argmax(P)), float(P[0])


def mode_amplitudes(out, k):
    """Crest and RMS radial amplitude of Koiter mode k, over its nodal one

    The models scale every Koiter mode so that its largest nodal translation
    equals the thickness, which misses the crest of a mode falling between
    two axial stations and makes b_ijkl oscillate with the mesh, see
    "Normalising the modes" in doc/nlprebuck_implementation.tex of
    bfsccylinder_models. Both amplitudes are returned in units of that
    largest nodal translation, so that the mode rescaled by s_k = 1/crest or
    s_k = 1/rms has a crest, or an RMS of w over the surface, equal to the
    thickness, see koiter_post.py.

    Between two axial stations w follows the cubic Hermite interpolation of
    the element in w and w_x, degrees of freedom 6 and 7, as in
    doc/verification/reference_b_convergence.py. The crest over the
    circumference of a single harmonic is the envelope of the mode and of
    its rotated partner, the stiffness of these cylinders being uniform
    around it, and the mean of w**2 over the ny nodes of an axial station is
    exact for such a harmonic
    """
    nx, ny = out['nx'], out['ny']
    order = mesh_order(out['x'], out['y'], nx, ny)
    u = out['koiter']['ui'][k]
    U = u.reshape(-1, DOF)[order]
    nodal = np.sqrt(U[:, :, 0]**2 + U[:, :, 3]**2 + U[:, :, 6]**2).max()
    psi = rotated(u, order, DOF)
    psi -= (psi @ u)/(u @ u)*u
    fields = [U]
    #NOTE an axisymmetric mode is its own rotation, and its envelope is |w|
    if np.linalg.norm(psi) > 1.e-8*np.linalg.norm(u):
        psi *= np.linalg.norm(u)/np.linalg.norm(psi)
        fields.append(psi.reshape(-1, DOF)[order])
    xs = out['x'][order[:, 0]]
    #NOTE 41 points per element for the crest; for the RMS 4 Gauss points,
    #     exact for the square of the cubic
    t = np.linspace(0, 1, 41)[:, None]
    tg, wg = np.polynomial.legendre.leggauss(4)
    tg, wg = (tg[:, None] + 1)/2, wg/2

    def hermite(F, i, t, dx):
        H00, H10 = 2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t
        H01, H11 = -2*t**3 + 3*t**2, t**3 - t**2
        return (H00*F[i, :, 6] + H10*dx*F[i, :, 7]
                + H01*F[i + 1, :, 6] + H11*dx*F[i + 1, :, 7])

    crest = 0.
    int_w2 = 0.
    for i in range(nx - 1):
        dx = xs[i + 1] - xs[i]
        env2 = sum(hermite(F, i, t, dx)**2 for F in fields)
        crest = max(crest, np.sqrt(env2).max())
        int_w2 += dx*(wg @ (hermite(U, i, tg, dx)**2).mean(axis=1))
    rms = np.sqrt(int_w2/(xs[-1] - xs[0]))
    return float(crest/nodal), float(rms/nodal)


def use_koiter_denominators():
    """lambda_i*d_i of the Koiter modes, d_i = phi20_i . u_i

    b_coefficients of the model is wrapped to keep lambda_i and d_i, which
    the model does not return. lambda_i*d_i is the counterpart of
    lambda_I q_I^T [dK_D + dK_G] q_I of Rahman (2009), Eq. (3.21), which his
    multi-mode analysis makes equal to one, see koiter_post.py. The values
    of the last Koiter analysis are returned in a list
    """
    b_coefficients = model.b_coefficients
    lambda_d = [None]

    def keep(phi4, phi3uab, phi30U, phi200, a, lam, d):
        lambda_d[0] = [float(v) for v in np.asarray(lam)*np.asarray(d)]
        return b_coefficients(phi4, phi3uab, phi30U, phi200, a, lam, d)

    model.b_coefficients = keep
    return lambda_d


def field_crest(out, u, num_points=11):
    """Largest |w| of the field u, between the nodes as well, over the
    largest nodal translation of Koiter mode 0

    A combination of modes of different circumferential wave numbers is no
    longer a single harmonic, whose crest mode_amplitudes finds from the
    envelope of a pair, so w is followed over every element with the bicubic
    Hermite interpolation of the element in w, w_x, w_y and w_xy, degrees of
    freedom 6 to 9, y being the arc length. The largest nodal translation of
    mode 0 is the thickness, to which the models scale every mode
    """
    nx, ny = out['nx'], out['ny']
    order = mesh_order(out['x'], out['y'], nx, ny)
    U0 = out['koiter']['ui'][0].reshape(-1, DOF)[order]
    nodal = np.sqrt(U0[:, :, 0]**2 + U0[:, :, 3]**2 + U0[:, :, 6]**2).max()
    U = u.reshape(-1, DOF)[order][:, :, 6:10]
    dx = np.diff(out['x'][order[:, 0]])[:, None]
    dy = out['y'].max()/(ny - 1)
    t = np.linspace(0, 1, num_points)
    H = np.array([2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t,
                  -2*t**3 + 3*t**2, t**3 - t**2])
    #NOTE nodes (i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1) of the
    #     elements, closing the circumference
    corners = [(U[:-1], 0, 0), (U[1:], 2, 0),
               (np.roll(U[:-1], -1, axis=1), 0, 2),
               (np.roll(U[1:], -1, axis=1), 2, 2)]
    w = 0.
    for F, a, b in corners:
        #NOTE w, dx*w_x, dy*w_y and dx*dy*w_xy against Hx*Hy, Gx*Hy, Hx*Gy
        #     and Gx*Gy
        w = w + (np.einsum('ij,p,q->ijpq', F[:, :, 0], H[a], H[b])
                 + np.einsum('ij,p,q->ijpq', dx*F[:, :, 1], H[a + 1], H[b])
                 + np.einsum('ij,p,q->ijpq', dy*F[:, :, 2], H[a], H[b + 1])
                 + np.einsum('ij,p,q->ijpq', dx*dy*F[:, :, 3], H[a + 1],
                             H[b + 1]))
    return float(np.abs(w).max()/nodal)


if __name__ == '__main__':
    icase = int(sys.argv[1])
    NLprebuck = dict(LIN=False, NL=True)[sys.argv[2]]
    if len(sys.argv) > 3:
        ny = int(sys.argv[3])

    DOE_vars = np.loadtxt(DOE_name + '.txt', skiprows=1)
    v1, v2, v3, v4, v5 = DOE_vars[icase]

    #NOTE parameters to be optimized
    variables = dict(
            rCTS=v1, # from 0.050 to 0.200
            param_n=int(v2), # from 1 to 12
            c2_ratio=v3, # from 0.05 to 0.95
            thetadeg_c1=v4, # from 0 to 75
            thetadeg_c2=v5, # from 0 to 75
            )

    constants = dict(
        L = 1.2,
        R = 0.4,
        ny = ny,
        E11 = 122e9,
        E22 = 7.32e9,
        nu12 = 0.31,
        G12 = 4.9e9,
        tow_thick = 0.13e-3,
        rho = 1540,
        mesh_only = False,
        Nxxunit = 1000.,
        NLprebuck = NLprebuck,
        )

    solvers = use_safe_solvers()
    num_distinct = use_distinct_modes()
    lambda_d = use_koiter_denominators()
    result = dict(case=icase, NLprebuck=NLprebuck, v1=v1, v2=v2, v3=v3, v4=v4,
                  v5=v5, library=bfsccylinder_models.__file__,
                  solvers=solvers, koiter_num_modes=koiter_num_modes,
                  koiter_rotation_closed=koiter_rotation_closed,
                  num_eigvals=num_eigvals)
    t0 = time.time()
    try:
        out = design_function(variables, constants)
        if num_distinct[0] < koiter_num_distinct:
            raise RuntimeError('%d distinct modes among %d eigenvectors, '
                    'fewer than koiter_num_distinct=%d, raise num_eigvals'
                    % (num_distinct[0], num_eigvals, koiter_num_distinct))
        n0, axi0 = mode_harmonics(out, 0)
        mu = out['mu']
        koiter = out['koiter']
        m = koiter_num_modes
        #NOTE b_ijkl[i][j][k][l] and a_ijk[i][j][k], the modes ordered by
        #     increasing multiplier, mode 0 being the critical one
        b_ijkl = [[[[float(koiter['b_ijkl'][(i, j, k, l)]) for l in range(m)]
                    for k in range(m)] for j in range(m)] for i in range(m)]
        a_ijk = [[[float(koiter['a_ijk'][(i, j, k)]) for k in range(m)]
                  for j in range(m)] for i in range(m)]
        result.update(
            num_distinct=num_distinct[0], koiter_distinct=num_distinct[1],
            b_ijkl=b_ijkl, a_ijk=a_ijk,
            b_iiii=[b_ijkl[i][i][i][i] for i in range(m)],
            )
        #NOTE b_ijkl and a_ijk above are for the nodal normalization of the
        #     models; post.py rescales them with these, see mode_amplitudes
        amplitudes = [mode_amplitudes(out, k) for k in range(m)]
        result.update(
            crest_w=[c for c, _ in amplitudes],
            rms_w=[r for _, r in amplitudes],
            lambda_d=lambda_d[0],
            )
        #NOTE the energy normalization of Rahman (2009), its most imperfection
        #     sensitive direction and b_min_t for the combined mode scaled to
        #     a crest of w equal to the thickness, see koiter_post.py
        s = koiter_post.energy_scales(lambda_d[0])
        b_energy, _ = koiter_post.rescaled(np.array(b_ijkl), np.array(a_ijk), s)
        b_min, e_min = koiter_post.min_direction(b_energy)
        combined = sum(e_min[k]*s[k]*koiter['ui'][k] for k in range(m))
        crest_e = field_crest(out, combined)
        result.update(
            b_min_energy=b_min, e_min=[float(v) for v in e_min],
            crest_e=crest_e, b_min_t=b_min/crest_e**2,
            #NOTE the bicubic crest of every mode, against the envelope one
            #     of mode_amplitudes, a check of field_crest
            crest_w_bicubic=[field_crest(out, koiter['ui'][k])
                             for k in range(m)],
            )
        result.update(
            #NOTE param_n and c2_ratio as analyzed, which differ from v2 and
            #     v3 when param_n exceeds nmax or c2 falls below its threshold
            param_n=int(out['param_n']), c2_ratio=float(out['c2_ratio']),
            t=float(out['t']), c1=float(out['c1']), c2=float(out['c2']),
            nx=int(out['nx']), ny=int(out['ny']), nxt=out['nxt'],
            max_ny_nx_aspect_ratio=out['max_ny_nx_aspect_ratio'],
            dx_max=float(np.diff(out['xlin']).max()),
            mass=float(out['mass']), Pcr=float(out['Pcr']),
            lambda_b=float(out['lambda_b']),
            b_factor=float(out['koiter']['b_ijkl'][(0, 0, 0, 0)]),
            n=n0, axi_share=axi0,
            mu_ratios=[float(m/mu[0]) for m in mu],
            modes_n=[mode_harmonics(out, k)[0] for k in range(len(mu))],
            )
    except Exception:
        traceback.print_exc()
        result['error'] = traceback.format_exc().splitlines()[-1]
    result['time_s'] = time.time() - t0
    #NOTE peak resident memory, which sets num_parallel in generate_qsubs.py
    try:
        import resource
        result['peak_mem_gb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
    except ImportError:
        try:
            import psutil
            result['peak_mem_gb'] = psutil.Process().memory_info().peak_wset/1e9
        except (ImportError, AttributeError):
            pass
    print('RESULT ' + json.dumps(result))
