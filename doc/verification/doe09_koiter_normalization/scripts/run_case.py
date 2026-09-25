"""Runs one case of DOE09.txt with a linear (LIN) or a nonlinear (NL)
pre-buckling state

usage: python run_case.py ICASE LIN|NL [NY] [--distinct K] [--num-eigvals N]
                         [--axial-factor F] [--eps1 E] [--nint P]
                         [--kinematics sanders|donnell] [--thickness-factor T]
                         [--nxxunit N]

NY overrides ny below, for the convergence study. The options, for the
reassessment studies (generate_qsubs_reassess.py), override
koiter_num_distinct, num_eigvals, axial_factor, NLprebuck_eps1, nint,
kinematics, thickness_factor and Nxxunit below, and
only when this file runs as a script, so that the globals seen by
generate_qsubs.py are those of the DOE. The last line printed is
"RESULT {json}", read by post.py, post_convergence.py and
checks/reassessment_post.py
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

from bfsccylinder.sanders import BFSCCylinderSanders
from bfsccylinder_models.cyclic_symmetry import mesh_order, rotated
from scipy.optimize import minimize

import koiter_post

#NOTE koiter_num_modes given as a callable needs bfsccylinder_models of the
#     branch doe09-koiter-normalization, which the job scripts put first on
#     PYTHONPATH
import inspect
if 'callable(koiter_num_modes)' not in inspect.getsource(
        model.fkoiter_cylinder_CTS_circum):
    raise ImportError('bfsccylinder_models at %s does not take koiter_num_modes '
            'as a callable, use the branch doe09-koiter-normalization'
            % bfsccylinder_models.__file__)
if 'mass_matrix' not in inspect.getsource(model.fkoiter_cylinder_CTS_circum):
    raise ImportError('bfsccylinder_models at %s has no inertia relief '
            'edges, use the branch doe09-koiter-normalization'
            % bfsccylinder_models.__file__)

DOE_name = 'DOE09'
DOF = 10

#NOTE mesh, see the convergence study (DOE09_convergence.txt): with
#     NLprebuck=True, Pcr at ny=160 is within 0.2, 0.1 and 0.5 per cent of
#     ny=200 for cases 0, 1 and 6, while b_factor still changes by 6, 7 and
#     20 per cent
ny = 160 #NOTE number of elements around circumference, nxt from choose_nxt

#NOTE relative residual above which a PARDISO solution is rejected
max_residual = 1.e-8

#NOTE multi-mode Koiter expansion on at least koiter_num_distinct modes
#     distinct up to the rotation of the cylinder, completed to the end of the
#     group of equal multipliers, to koiter_cluster_rtol, of the last one, and
#     on the rotated partner of each, see use_distinct_modes, which sets
#     koiter_num_modes to the callable that returns their number to the model
#     after its last eigenvalue analysis (bfsccylinder_models, branch
#     doe09-koiter-normalization). num_eigvals must leave room for the
#     partners and for the mode that ends the last group
koiter_num_distinct = 5
koiter_cluster_rtol = 1.e-5
koiter_num_modes = None
num_eigvals = 16

#NOTE the largest axial element length is dy/axial_factor, see choose_nxt;
#     1 is the mesh of the convergence studies
axial_factor = 1.
#NOTE tolerance of the iterative eigenvalue algorithm of the model on
#     lambda_b/lambda_c, the expansion point of NLprebuck, its default
NLprebuck_eps1 = 0.005
#NOTE Gauss-Legendre points per direction of every element, for the
#     stiffness, the laminate and the Koiter tensors alike, the default of
#     fkoiter_cylinder_CTS_circum; 4 integrates degree 7 exactly, less than
#     the degree of the products of four derivatives of w in phi4
nint = 4
#NOTE shell kinematics of the model, sanders (koiter_cylinder_CTS_sanders)
#     or donnell (koiter_cylinder_CTS); the two modules differ only in the
#     element and in the rotation G2 = w,y - v/R of the nonlinear strains,
#     the membrane strains, v,y + w/R, being the same
kinematics = 'sanders'
#NOTE factor on tow_thick, which changes R/h of a design and nothing else of
#     it, to measure how the mesh error of b depends on R/h
thickness_factor = 1.
#NOTE load unit in N/m, the load of lambda = 1, where the load stepping of
#     the nonlinear pre-buckling state starts; below it, Pcr/(2 pi R) less
#     than Nxxunit, the first step already overshoots the bifurcation and the
#     expansion point stays at lambda_b = 1 > lambda_c (thickness factor
#     0.7 of case 6, Pcr about 2470 N against 2513 N)
Nxxunit = 1000.
#NOTE the edge condition of the models, the only one they have, recorded in
#     the RESULT line: SS3 with inertia relief, v = w = 0 along both edges,
#     the load on both edges, no node anchored, see
#     bfsccylinder_models/edges.py
edges = 'SS3-IR'
#NOTE k and ncv of every call to eigsh, and the ARPACK error of a call
#     retried with a larger ncv, see use_safe_solvers
eigsh_calls = []


def eigsh_ncv(k, n):
    """Number of Lanczos vectors of eigsh for k eigenvalues of order n

    None, the default of scipy, min(n, max(2k + 1, 20)), for the
    num_eigvals=16 of the convergence studies, which it reproduces. With it
    ARPACK stopped with error -8 at k=20 on DOE09 meshes, so a larger k gets
    k + 32 at least
    """
    if k <= 16:
        return None
    return min(n, max(2*k + 1, k + 32))


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

    Every distinct mode is then followed by its rotated partner, so that the
    Koiter modes span the rotations of each other. The partners do carry information into a multi-mode
    expansion: the interaction of two distinct modes depends on their
    relative rotation, which canonical_modes fixes by a convention, and when
    the two members of a pair of symmetric and antisymmetric modes, localized
    at the two ends, share a multiplier to round off, as for DOE09 case 1
    with NLprebuck, their rotations span a four dimensional eigenspace of
    which the eigen solver returns an arbitrary two dimensional slice. The
    expansion over the closed set is independent of that slice, and of the
    convention. An axisymmetric mode has no partner and takes one column.

    The same holds for a group of distinct modes of equal multiplier: at
    least koiter_num_distinct distinct modes are taken, and then every
    further one whose multiplier equals that of the last taken to
    koiter_cluster_rtol, the tolerance of canonical_modes, so that no such
    group is cut. The 5th distinct mode cut a four dimensional eigenspace in
    10 of the 24 runs of the convergence study, leaving the other half of it
    out of the expansion as decided by round off. The number of Koiter modes
    then varies from run to run, and the global koiter_num_modes is set to
    the callable that returns it to the model after the last eigenvalue
    analysis, which raises the errors found in selecting the set
    """
    canonical_modes = model.canonical_modes
    num_distinct = [None]
    #NOTE True for the Koiter modes that are distinct modes and False for the
    #     partners, in num_distinct[1]; the relative gap between the
    #     multiplier of the last distinct Koiter mode and that of the next
    #     distinct mode in num_distinct[2]; the number of Koiter modes, or the
    #     error found in selecting them, in selection
    num_distinct += [None, None]
    selection = {}
    #NOTE KCuu of the last eigenvalue analysis, the metric in which the
    #     closed set of Koiter modes is made orthonormal
    eigsh = model.eigsh
    KCuu = [None]

    def keep_KC(A, k, which, M, tol, v0, **kwargs):
        KCuu[0] = M
        return eigsh(A=A, k=k, which=which, M=M, tol=tol, v0=v0, **kwargs)

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
        selection.clear()
        #NOTE the distinct modes of the expansion, complete groups
        take = distinct[:koiter_num_distinct]
        for k in distinct[koiter_num_distinct:]:
            if abs(mu[k] - mu[take[-1]]) > koiter_cluster_rtol*abs(mu[take[-1]]):
                break
            take.append(k)
        if len(take) < koiter_num_distinct or take[-1] == distinct[-1]:
            #NOTE fewer distinct modes than required, or the last one returned
            #     is in the set, and the end of its group is not known
            selection['error'] = ('%d distinct modes among %d eigenvectors do '
                    'not end a group of at least koiter_num_distinct=%d, raise '
                    'num_eigvals' % (len(distinct), eigvecsu.shape[1],
                                     koiter_num_distinct))
        nxt = distinct[len(take)] if len(distinct) > len(take) else take[-1]
        num_distinct[2] = float(abs(mu[nxt] - mu[take[-1]])/abs(mu[take[-1]]))
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
        for k in take:
            phi = np.zeros(bu.shape[0])
            phi[bu] = orthonormal(eigvecsu[:, k], cols)
            psi = model.degenerate_partner(phi, bu, axi_order, DOF)
            pair = [phi[bu]]
            if psi is not None:
                pair.append(orthonormal(psi[bu], cols + pair))
            cols += pair
            mus += [mu[k]]*len(pair)
            used.append(k)
            flags += [True] + [False]*(len(pair) - 1)
        num_distinct[1] = flags
        if len(cols) > eigvecsu.shape[1]:
            selection.setdefault('error', '%d Koiter modes, more than '
                    'num_eigvals=%d' % (len(cols), eigvecsu.shape[1]))
            cols, mus = cols[:eigvecsu.shape[1]], mus[:eigvecsu.shape[1]]
        selection['num_modes'] = len(cols)
        #NOTE the rest of the columns, not Koiter modes, keep the width
        rest = [k for k in distinct + twins if k not in used]
        rest = rest[:eigvecsu.shape[1] - len(cols)]
        mu[:] = np.array(mus + [mu[k] for k in rest])
        return np.column_stack(cols + [eigvecsu[:, k] for k in rest])

    def num_modes(mu, eigvecs):
        """Number of Koiter modes of the last eigenvalue analysis"""
        if 'error' in selection:
            raise RuntimeError(selection['error'])
        return selection['num_modes']

    global koiter_num_modes
    koiter_num_modes = num_modes
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

    Here symmetric matrices are factorized with Cholesky, the others, the
    bordered systems of the second order fields, with lu_gmres, and every
    PARDISO solution is checked against its residual, SuperLU being the
    fallback. The static solves and the inverse of KCuu in eigsh are those of
    the inertia relief solver of the model, bfsccylinder_models.edges, which
    uses the Cholesky factorization of PARDISO with the same check
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

    def arpack(A, k, which, M, tol, v0, **kwargs):
        """scipy eigsh with the ncv of eigsh_ncv, retried once with twice the
        Lanczos vectors, k + 64 at least, if ARPACK fails"""
        n = A.shape[0]
        ncv = eigsh_ncv(k, n)
        call = dict(k=k, ncv=min(n, max(2*k + 1, 20)) if ncv is None else ncv)
        eigsh_calls.append(call)
        try:
            return scipy.sparse.linalg.eigsh(A=A, k=k, which=which, M=M,
                    tol=tol, v0=v0, ncv=ncv, **kwargs)
        except scipy.sparse.linalg.ArpackError as e:
            call['error'] = str(e)
            call['ncv'] = min(n, max(2*call['ncv'], k + 64))
            print('# WARNING: %s, eigsh again with ncv=%d' % (e, call['ncv']))
            return scipy.sparse.linalg.eigsh(A=A, k=k, which=which, M=M,
                    tol=tol, v0=v0, ncv=call['ncv'], **kwargs)

    def eigsh(A, k, which, M, tol, v0, **kwargs):
        #NOTE the model gives the inverse, Minv, or the shifted inverse,
        #     sigma and OPinv, on the null space of the inertia relief
        #     condition, M being singular, see edges.py
        return arpack(A=A, k=k, which=which, M=M, tol=tol, v0=v0, **kwargs)

    model.spsolve = spsolve
    model.eigsh = eigsh
    return 'pardiso'


def choose_nxt(L, R, ny, rCTS, param_n, c2_ratio, thetadeg_c1, thetadeg_c2,
        c1_threshold_factor=0.01, c2_threshold_factor=0.01, axial_factor=1.):
    """Nodes along each transition region, and max_ny_nx_aspect_ratio

    Chosen so that no element of the transition or plateau regions is longer
    than dy/axial_factor, dy the circumferential element length, so that the
    axial mesh can be refined at fixed ny. fkoiter_cylinder_CTS_circum
    gives a plateau of length c round(c/t*nxt) nodes, at least 2, or
    round(c/(dy/max_ny_nx_aspect_ratio)) nodes, at least 2, when the
    transition elements are more than max_ny_nx_aspect_ratio times shorter
    than dy. Either way a short plateau can get a single element, which is
    what the two parameters returned avoid. t, c1 and c2 are computed as in
    fkoiter_cylinder_CTS_circum, which compares with the true dy the
    transition elements it is given, as plateau_dx does
    """
    dy = 2*np.pi*R/ny
    #NOTE dy itself for axial_factor=1, which reproduces the meshes of the
    #     convergence studies
    dx_max = dy/axial_factor
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
    nxt = max(3, int(np.ceil(t/dx_max)) + 1)
    for c in plateaus:
        nxt = max(nxt, int(np.ceil((np.ceil(c/dx_max) + 0.5)*t/c)))

    def plateau_dx(c, max_ny_nx_aspect_ratio):
        if dy/(t/(nxt - 1)) > max_ny_nx_aspect_ratio:
            nodes = max(2, int(round(c/(dy/max_ny_nx_aspect_ratio), 0)))
        else:
            nodes = max(2, int(round(c/t*nxt, 0)))
        return c/(nodes - 1)

    max_ny_nx_aspect_ratio = 2
    while any(plateau_dx(c, max_ny_nx_aspect_ratio) > dx_max
              for c in plateaus):
        max_ny_nx_aspect_ratio += 1
    return nxt, max_ny_nx_aspect_ratio


def estimate_nx(L, R, ny, rCTS, param_n, c2_ratio, thetadeg_c1, thetadeg_c2,
        c1_threshold_factor=0.01, c2_threshold_factor=0.01, axial_factor=1.):
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
            c2_ratio, thetadeg_c1, thetadeg_c2, axial_factor=axial_factor)
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
            c2_ratio, thetadeg_c1, thetadeg_c2, axial_factor=axial_factor)

    out = fkoiter_cylinder_CTS_circum(L, R, rCTS, nxt, ny, E11, E22, nu12, G12,
            rho, tow_thick, param_n, c2_ratio, thetadeg_c1, thetadeg_c2,
            mesh_only=mesh_only, Nxxunit=Nxxunit,
            num_eigvals=num_eigvals, koiter_num_modes=koiter_num_modes,
            NLprebuck=NLprebuck, NLprebuck_eps1=NLprebuck_eps1,
            max_ny_nx_aspect_ratio=max_ny_nx_aspect_ratio, nint=nint)
    out['nxt'] = nxt
    out['max_ny_nx_aspect_ratio'] = max_ny_nx_aspect_ratio
    return out


def mode_harmonics(out, k):
    """Circumferential wave number of mode k and its axisymmetric share"""
    w = out['eigvecs'][:, k].reshape(out['nx'], out['ny'], DOF)[:, :, 6]
    P = (np.abs(np.fft.rfft(w, axis=1))**2).sum(axis=0)
    P = P/P.sum()
    return int(np.argmax(P)), float(P[0])


class ElementField:
    """w of a field anywhere on the mesh, from the kinematics of the element

    w inside an element is Sw(xi, eta) @ q, q the 40 degrees of freedom of
    its four nodes, with Sw from update_Sw of BFSCCylinderSanders, the element
    the model is assembled with, so that no interpolation is assumed here. Sw
    depends on xi, eta, lex and ley only, and is evaluated once per element
    length. Amplitudes are returned over the largest nodal translation of
    Koiter mode 0, the thickness to which the models scale every mode
    """

    def __init__(self, out, num_grid=9, num_refine=20):
        pos = out['nid_pos']
        c = np.array([[DOF*pos[n] for n in nodes] for nodes in
                      zip(out['n1s'], out['n2s'], out['n3s'], out['n4s'])])
        self.idx = (c[:, :, None] + np.arange(DOF)).reshape(len(c), -1)
        x = out['x']
        #NOTE lex and ley as assigned by the model, y being the arc length at
        #     ny stations from 0 to circ - ley
        self.lex = x[c[:, 1]//DOF] - x[c[:, 0]//DOF]
        self.ley = out['y'].max()/(out['ny'] - 1)
        self.elem = BFSCCylinderSanders(4)
        self.elem.R = self.ley*out['ny']/(2*np.pi)
        self.elem.ley = self.ley
        self.num_refine = num_refine
        g = np.linspace(-1., 1., num_grid)
        self.grid = [(xi, eta) for xi in g for eta in g]
        #NOTE 4 Gauss points per direction integrate w**2 exactly, w being
        #     bicubic in the element
        tg, wg = np.polynomial.legendre.leggauss(4)
        self.gauss = [(xi, eta, wi*wj) for xi, wi in zip(tg, wg)
                      for eta, wj in zip(tg, wg)]
        self.groups = {}
        for e, le in enumerate(np.round(self.lex, 12)):
            self.groups.setdefault(le, []).append(e)
        self.S_grid = {le: self.Sw_points(le, self.grid) for le in self.groups}
        self.S_gauss = {le: self.Sw_points(le, [(p, q) for p, q, _ in
                        self.gauss]) for le in self.groups}
        U0 = out['koiter']['ui'][0].reshape(-1, DOF)
        self.nodal = np.sqrt(U0[:, 0]**2 + U0[:, 3]**2 + U0[:, 6]**2).max()
        self.area = self.lex.sum()*self.ley

    def Sw_points(self, le, points):
        self.elem.lex = le
        rows = []
        for xi, eta in points:
            self.elem.update_Sw(xi, eta)
            rows.append(np.atleast_2d(self.elem.Sw)[0].copy())
        return np.array(rows)

    def crest(self, u):
        """Largest |w| of u over the surface, over the nodal translation

        The largest |w| at a grid of points of every element, then a local
        maximization of |w| over (xi, eta) inside the num_refine elements of
        largest grid values, the crest of a mode falling anywhere inside an
        element
        """
        Q = u[self.idx]
        cand = []
        for le, sel in self.groups.items():
            W = np.abs(Q[sel] @ self.S_grid[le].T)
            k = np.argmax(W, axis=1)
            cand += zip(W[np.arange(len(sel)), k], sel, k)
        cand.sort(reverse=True)
        best = cand[0][0]
        for w0, e, kk in cand[:self.num_refine]:
            #NOTE |w| over its grid value, of order one, the default
            #     tolerances of L-BFGS-B being absolute
            q = Q[e]/w0
            le = self.lex[e]

            def f(p):
                self.elem.lex = le
                self.elem.update_Sw(*p)
                return -abs(np.atleast_2d(self.elem.Sw)[0] @ q)

            r = minimize(f, self.grid[kk], method='L-BFGS-B',
                         bounds=[(-1., 1.), (-1., 1.)],
                         options=dict(ftol=1.e-15, gtol=1.e-12))
            best = max(best, -r.fun*w0)
        return float(best/self.nodal)

    def rms(self, u):
        """RMS of w of u over the surface, over the nodal translation"""
        Q = u[self.idx]
        wg = np.array([w for _, _, w in self.gauss])
        int_w2 = 0.
        for le, sel in self.groups.items():
            W = Q[sel] @ self.S_gauss[le].T
            int_w2 += le*self.ley/4*(W**2 @ wg).sum()
        return float(np.sqrt(int_w2/self.area)/self.nodal)


def pair_rotation(out, koiter, distinct):
    """Rotation of the cylinder acting on the closed set of Koiter modes

    A shift of the circumferential node index by one element, rotated, is an
    exact symmetry of the mesh, and maps the plane of a distinct mode and of
    its partner onto itself, a rotation by n*dtheta in an orthonormal basis of
    the plane, n being the wave number and dtheta = 2 pi/ny. That rotation is
    extended to any angle theta, so that a combination of the Koiter modes
    can be followed along the continuous family of its rotations, of which
    the finite element crest is not constant. Returns rotate(v, t), v a
    combination of the Koiter modes and t = theta/dtheta, and the relative
    difference between rotate(v, 1) and the exact shift of a test vector
    """
    nx, ny = out['nx'], out['ny']
    order = mesh_order(out['x'], out['y'], nx, ny)
    ui = koiter['ui']
    m = len(distinct)
    planes = []
    for k in range(m):
        if not distinct[k] or k + 1 >= m or distinct[k + 1]:
            continue
        Qk, _ = np.linalg.qr(np.column_stack([ui[k], ui[k + 1]]))
        #NOTE the exact shift in the orthonormal basis of the plane
        M = Qk.T @ np.column_stack([rotated(Qk[:, j], order, DOF)
                                    for j in range(2)])
        n = mode_harmonics(out, k)[0]
        #NOTE the angle of the shift is n*dtheta, and M tells its sense
        sense = np.sign(M[1, 0]) if abs(M[1, 0]) > 1.e-12 else 1.
        planes.append((Qk, sense*n*2*np.pi/ny))

    def rotate(v, t):
        v = v.copy()
        for Qk, alpha in planes:
            z = Qk.T @ v
            c, s = np.cos(alpha*t), np.sin(alpha*t)
            v += Qk @ (np.array([[c, -s], [s, c]]) @ z - z)
        return v

    #NOTE on w only: a partner of degenerate_partner carries the axial rigid
    #     body translation that restores the constraint of a single node
    test = sum(ui[k]*(k + 1) for k in range(m))
    exact = rotated(test, order, DOF)[6::DOF]
    err = (np.linalg.norm(rotate(test, 1.)[6::DOF] - exact)
           /np.linalg.norm(exact))
    return rotate, float(err)


def orbit_crest(field, rotate, v, num_phases=8):
    """Largest element crest of v over its rotations by a fraction of an
    element, the finite element crest of a rotated field depending on where
    it falls between the nodes"""
    return max(field.crest(rotate(v, t))
               for t in np.arange(num_phases)/num_phases)


def cluster_subsets(modes_n, distinct):
    """Subsets of the Koiter modes made of whole wave-number clusters

    A cluster is the modes of one circumferential wave number n, the
    symmetric and antisymmetric modes and their rotated partners, 4 modes,
    complete when both distinct modes of n are in the set; a further
    distinct mode of n, another axial shape, and its partner are left out,
    as they enter the set of some meshes only. b_min over the
    set depends on how many clusters it holds, and which of them are in the
    set changes with the mesh, their multipliers lying closer than each
    moves between meshes (REPORT.md, Reassessment), so the set is followed
    through subsets defined by the wave numbers instead:

    - prefix: the first p clusters, in the order of the Koiter set, that is
      of increasing multiplier;
    - window: the wave numbers n_c - j to n_c + j about the critical one n_c,
      that of mode 0, for as long as all of them are in the set.

    Returns the modes of every wave number and the subsets as (kind, ns)
    """
    groups = {}
    seen = {}
    for k, n in enumerate(modes_n):
        seen[n] = seen.get(n, 0) + distinct[k]
        if seen[n] <= 2:
            groups.setdefault(n, []).append(k)
    order = list(groups)
    subsets = [('prefix', order[:p]) for p in range(1, len(order) + 1)]
    n_c = modes_n[0]
    j = 0
    while all(n in groups for n in range(n_c - j, n_c + j + 1)):
        subsets.append(('window', list(range(n_c - j, n_c + j + 1))))
        j += 1
    complete = {n: sum(distinct[k] for k in idx) >= 2
                for n, idx in groups.items()}
    return groups, complete, subsets


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='one case of DOE09.txt')
    parser.add_argument('icase', type=int)
    parser.add_argument('prebuck', choices=['LIN', 'NL'])
    parser.add_argument('ny', type=int, nargs='?', default=ny)
    parser.add_argument('--distinct', type=int, default=koiter_num_distinct,
            help='koiter_num_distinct, default %(default)s')
    parser.add_argument('--num-eigvals', type=int, default=None,
            help='num_eigvals, default %d for --distinct %d, 2K + 8 otherwise'
                 % (num_eigvals, koiter_num_distinct))
    parser.add_argument('--axial-factor', type=float, default=axial_factor,
            help='largest axial element length dy/F, default %(default)s')
    parser.add_argument('--eps1', type=float, default=NLprebuck_eps1,
            help='NLprebuck_eps1, default %(default)s')
    parser.add_argument('--nint', type=int, default=nint,
            help='integration points per direction, default %(default)s')
    parser.add_argument('--kinematics', choices=['sanders', 'donnell'],
            default=kinematics, help='default %(default)s')
    parser.add_argument('--thickness-factor', type=float,
            default=thickness_factor,
            help='factor on tow_thick, default %(default)s')
    parser.add_argument('--nxxunit', type=float, default=Nxxunit,
            help='load unit in N/m, default %(default)s')
    args = parser.parse_args()
    icase = args.icase
    NLprebuck = dict(LIN=False, NL=True)[args.prebuck]
    ny = args.ny
    #NOTE num_eigvals leaves room for the K distinct modes, their partners,
    #     the completion of the last group and one more distinct mode, which
    #     distinct_first needs to know where the group ends
    if args.num_eigvals is not None:
        num_eigvals = args.num_eigvals
    elif args.distinct != koiter_num_distinct:
        num_eigvals = 2*args.distinct + 8
    koiter_num_distinct = args.distinct
    axial_factor = args.axial_factor
    NLprebuck_eps1 = args.eps1
    nint = args.nint
    kinematics = args.kinematics
    thickness_factor = args.thickness_factor
    Nxxunit = args.nxxunit
    if kinematics == 'donnell':
        #NOTE before use_safe_solvers and the others, which patch the
        #     functions of model; ElementField keeps BFSCCylinderSanders,
        #     whose Sw is that of BFSCCylinder, w being bicubic in both
        import bfsccylinder_models.koiter_cylinder_CTS as model
        fkoiter_cylinder_CTS_circum = model.fkoiter_cylinder_CTS_circum

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
        tow_thick = 0.13e-3*thickness_factor,
        rho = 1540,
        mesh_only = False,
        Nxxunit = Nxxunit,
        NLprebuck = NLprebuck,
        )

    solvers = use_safe_solvers()
    num_distinct = use_distinct_modes()
    lambda_d = use_koiter_denominators()
    result = dict(case=icase, NLprebuck=NLprebuck, v1=v1, v2=v2, v3=v3, v4=v4,
                  v5=v5, library=bfsccylinder_models.__file__,
                  solvers=solvers, koiter_num_distinct=koiter_num_distinct,
                  koiter_cluster_rtol=koiter_cluster_rtol,
                  #NOTE see generate_qsubs.py
                  koiter_set='complete_clusters',
                  num_eigvals=num_eigvals, axial_factor=axial_factor,
                  NLprebuck_eps1=NLprebuck_eps1, nint=nint,
                  kinematics=kinematics, thickness_factor=thickness_factor,
                  Nxxunit=Nxxunit, edges=edges,
                  eigsh_calls=eigsh_calls)
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
        m = out['koiter_num_modes']
        #NOTE b_ijkl[i][j][k][l] and a_ijk[i][j][k], the modes ordered by
        #     increasing multiplier, mode 0 being the critical one
        b_ijkl = [[[[float(koiter['b_ijkl'][(i, j, k, l)]) for l in range(m)]
                    for k in range(m)] for j in range(m)] for i in range(m)]
        a_ijk = [[[float(koiter['a_ijk'][(i, j, k)]) for k in range(m)]
                  for j in range(m)] for i in range(m)]
        result.update(
            num_distinct=num_distinct[0], koiter_distinct=num_distinct[1],
            koiter_num_modes=m, koiter_gap=num_distinct[2],
            b_ijkl=b_ijkl, a_ijk=a_ijk,
            b_iiii=[b_ijkl[i][i][i][i] for i in range(m)],
            )
        #NOTE b_ijkl and a_ijk above are for the nodal normalization of the
        #     models; post.py rescales them with these, see ElementField
        #NOTE crest and RMS of w of every Koiter mode from the kinematics of
        #     the element, see ElementField and koiter_post.py
        field = ElementField(out)
        rotate, rotation_error = pair_rotation(out, koiter, num_distinct[1])
        result.update(
            #NOTE the largest over the rotations of each mode
            crest_w=[orbit_crest(field, rotate, koiter['ui'][k])
                     for k in range(m)],
            rotation_error=rotation_error,
            rms_w=[field.rms(koiter['ui'][k]) for k in range(m)],
            lambda_d=lambda_d[0],
            )
        #NOTE the energy normalization of Rahman (2009), its most imperfection
        #     sensitive direction and b_min_t for the combined mode scaled to
        #     a crest of w equal to the thickness, see koiter_post.py
        s = koiter_post.energy_scales(lambda_d[0])
        b_energy, _ = koiter_post.rescaled(np.array(b_ijkl), np.array(a_ijk), s)
        b_min, e_min = koiter_post.min_direction(b_energy)
        combined = sum(e_min[k]*s[k]*koiter['ui'][k] for k in range(m))
        #NOTE the minimum direction is defined up to a rotation of the
        #     cylinder, and the largest crest over the rotations is kept, with
        #     the smallest as a measure of the discretization
        crests = [field.crest(rotate(combined, t)) for t in np.arange(8)/8]
        crest_e = max(crests)
        result.update(crest_e_min=min(crests))
        result.update(
            b_min_energy=b_min, e_min=[float(v) for v in e_min],
            crest_e=crest_e, b_min_t=b_min/crest_e**2,
            #NOTE crest and RMS from ElementField, see generate_qsubs.py
            crest_method='element_orbit',
            )
        #NOTE b_min_energy, crest_e and b_min_t of subsets of whole
        #     wave-number clusters, see cluster_subsets. The b_ijkl of a
        #     subset are the sub-block of those of the set: the orthogonality
        #     conditions of the second order fields take every Koiter mode,
        #     so a subset is not a run on that subset alone. The combined
        #     mode of a subset has no component in the planes of the other
        #     modes, which pair_rotation leaves unchanged
        modes_n = [mode_harmonics(out, k)[0] for k in range(m)]
        groups, complete, subsets = cluster_subsets(modes_n, num_distinct[1])
        rows = []
        for kind, ns in subsets:
            idx = sum((groups[n] for n in ns), [])
            b_sub, e_sub = koiter_post.min_direction(
                    b_energy[np.ix_(idx, idx, idx, idx)])
            v = sum(e_sub[j]*s[k]*koiter['ui'][k] for j, k in enumerate(idx))
            crest_sub = orbit_crest(field, rotate, v)
            rows.append(dict(kind=kind, ns=ns, num_modes=len(idx),
                    complete=all(complete[n] for n in ns),
                    b_min_energy=b_sub, crest_e=crest_sub,
                    b_min_t=b_sub/crest_sub**2,
                    e=[float(x) for x in e_sub]))
        result.update(subsets=rows)
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
            #NOTE the expansion point reached, lambda_b/lambda_c
            lambda_ratio=float(out['lambda_b']/(out['Pcr']
                    /(constants['Nxxunit']*2*np.pi*constants['R']))),
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
