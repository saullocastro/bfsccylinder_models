"""
Edge conditions of the Koiter cylinder models
==============================================

The displacement vector of the models is written as ``u = T a``, with ``a``
the independent unknowns. Every degree of freedom that is either free or
fixed to zero is a column of the identity, or no column at all, so that
``T`` is a selection matrix for the SS3 edges; the SS4 edges tie the axial
displacement of every node of the loaded edge to one shared unknown, a column
of ``T`` carrying one at each of them. A stiffness matrix is reduced as
``T.T K T``, a force vector as ``T.T f``, and a displacement is expanded as
``T a`` and reduced back by reading it at one degree of freedom per unknown,
which is exact for every displacement ``T`` can produce.

The inertia relief edges, ``'SS3-IR'`` and ``'free-IR'``, anchor no node. The
axial load is applied at both edges, self-equilibrated, and every rigid body
mode the edges leave free is removed by the inertia relief condition
``C.T a = 0``, ``C = T.T M R``, with ``M`` the consistent mass matrix and ``R``
the rigid body modes: the displacement has no rigid body component in the
mass metric, its centre of mass does not move and it does not rotate as a
whole. The condition is imposed exactly, by :class:`ConstrainedSolver` in the
static solves, as the inverse of the eigen solver in the buckling analysis
(:meth:`EdgeSpace.eigsh`) and as a border of the bordered system of the
second order fields. Anchoring a statically determinate set of nodes and
subtracting the rigid body component afterwards would give the same result
only for the rigid body modes that are exact null vectors of every operator:
the axial translation and the rotation about the axis are, but the
transverse translations and rotations are not, their circumferential
variation not being in the finite element space, nor are the rotations null
vectors of the geometric stiffness matrix.

"""
import numpy as np
from numpy import isclose
import scipy.linalg
from scipy.sparse import csc_matrix, coo_matrix, triu
from scipy.sparse.linalg import LinearOperator, splu

EDGES = ('SS3', 'SS4', 'SS3-IR', 'free-IR')

#NOTE the six rigid body modes, in the order of the columns of
#     rigid_body_modes
RIGID = ('Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz')


def rigid_body_modes(x, y, R, DOF):
    """The six rigid body modes, one column each, in the element degrees of
    freedom

    The translations along and the rotations about the global axes, x along
    the axis of the cylinder, with the circumferential coordinate y = R theta
    and the local degrees of freedom u, u,x, u,y, v, v,x, v,y, w, w,x, w,y,
    w,xy, w positive outwards. The axial translation Tx and the rotation about
    the axis Rx are in the finite element space and are null vectors of the
    stiffness matrix to round off; the other four vary as cos(theta) and
    sin(theta) around the circumference, which the Hermite interpolation
    only approximates: their energy is 1e-9 (translations) and 2e-5
    (rotations) of that of a smooth deformation on a mesh of 40 elements
    around, 6e-11 and 4e-6 on one of 80.
    """
    th = y/R
    s, c = np.sin(th), np.cos(th)
    modes = np.zeros((x.shape[0], DOF, 6))
    # Tx
    modes[:, 0, 0] = 1.
    # Ty, d = (0, 1, 0): v = -sin, w = cos
    modes[:, 3, 1] = -s
    modes[:, 5, 1] = -c/R
    modes[:, 6, 1] = c
    modes[:, 8, 1] = -s/R
    # Tz, d = (0, 0, 1): v = cos, w = sin
    modes[:, 3, 2] = c
    modes[:, 5, 2] = -s/R
    modes[:, 6, 2] = s
    modes[:, 8, 2] = c/R
    # Rx, scaled by 1/R: v = 1
    modes[:, 3, 3] = 1.
    # Ry, d = e_y x X = (R sin, 0, -x)
    modes[:, 0, 4] = R*s
    modes[:, 2, 4] = c
    modes[:, 3, 4] = -x*c
    modes[:, 4, 4] = -c
    modes[:, 5, 4] = x*s/R
    modes[:, 6, 4] = -x*s
    modes[:, 7, 4] = -s
    modes[:, 8, 4] = -x*c/R
    modes[:, 9, 4] = -c/R
    # Rz, d = e_z x X = (-R cos, x, 0)
    modes[:, 0, 5] = -R*c
    modes[:, 2, 5] = s
    modes[:, 3, 5] = -x*s
    modes[:, 4, 5] = -s
    modes[:, 5, 5] = -x*c/R
    modes[:, 6, 5] = x*c
    modes[:, 7, 5] = c
    modes[:, 8, 5] = -x*s/R
    modes[:, 9, 5] = -s/R
    return modes.reshape(-1, 6)


def mass_matrix(elements, update_M, M_SPARSE_SIZE, N, h, rho):
    """Consistent mass matrix, h and rho the thickness and density of every
    element, which only the mass matrix reads from the elements"""
    for i, elem in enumerate(elements):
        elem.h = h[i]
        elem.rho = rho[i]
        elem.init_k_M = i*M_SPARSE_SIZE
    size = M_SPARSE_SIZE*len(elements)
    Mr = np.zeros(size, dtype=np.int64)
    Mc = np.zeros(size, dtype=np.int64)
    Mv = np.zeros(size, dtype=np.float64)
    for elem in elements:
        update_M(elem, Mr, Mc, Mv)
    return coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()


class Factor:
    """Factorization of a sparse symmetric positive definite matrix

    The Cholesky factorization of PARDISO when pypardiso is installed,
    checked against the residual of a solve, SuperLU otherwise or when the
    residual is above 1e-8
    """
    def __init__(self, A):
        A = csc_matrix(A)
        self.A = A
        self.pardiso = None
        try:
            import pypardiso
        except ImportError:
            pypardiso = None
        if pypardiso is not None:
            solver = pypardiso.PyPardisoSolver(mtype=2)
            Au = triu(A, format='csr')
            solver.factorize(Au)
            self.pardiso = (solver, Au)
            b = np.random.default_rng(0).random(A.shape[0])
            res = np.linalg.norm(A @ self.solve(b) - b)/np.linalg.norm(b)
            if res <= 1.e-8:
                return
            print('# WARNING: PARDISO Cholesky residual %.1e, using SuperLU'
                  % res)
            self.free()
        self.lu = splu(A)

    def solve(self, b):
        if self.pardiso is not None:
            solver, Au = self.pardiso
            return solver.solve(Au, np.ascontiguousarray(b))
        return self.lu.solve(b)

    def free(self):
        if self.pardiso is not None:
            self.pardiso[0].free_memory(everything=True)
            self.pardiso = None


class ConstrainedSolver:
    """Solution of ``K a + C lam = b``, ``C.T a = 0``, K symmetric and
    positive definite on the null space of C.T

    By elimination: the k unknowns ``pin``, chosen so that no combination of
    the rigid body modes vanishes on them, are eliminated last, so that the
    rest of K is positive definite and is factorized once, and ``[a_pin,
    lam]`` solve a dense system of size 2k. This is only a partition of the
    unknowns of the constrained system, whose solution is that of the
    inertia relief whatever pin is; the ``pin`` unknowns are not
    constrained. ``lam`` are the inertia forces, zero for a self-equilibrated
    load up to the approximation of the rigid body modes.
    """
    def __init__(self, K, C, pin):
        n, k = C.shape
        K = csc_matrix(K)
        f = np.ones(n, dtype=bool)
        f[pin] = False
        self.f, self.pin, self.C = f, pin, C
        Kf = K[f, :]
        self.Kfp = Kf[:, pin].toarray()
        Kpp = K[pin, :][:, pin].toarray()
        self.F = Factor(Kf[:, f])
        Cf, Cp = C[f], C[pin]
        Y = self.F.solve(np.column_stack((self.Kfp, Cf)))
        self.Y = Y
        S = np.block([[Kpp - self.Kfp.T @ Y[:, :k], Cp - self.Kfp.T @ Y[:, k:]],
                      [Cp.T - Cf.T @ Y[:, :k], -Cf.T @ Y[:, k:]]])
        self.S = scipy.linalg.lu_factor(S)
        self.k = k

    def solve(self, b):
        b = np.asarray(b, dtype=np.float64)
        z = self.F.solve(b[self.f])
        rhs = np.concatenate((b[self.pin] - self.Kfp.T @ z,
                              -self.C[self.f].T @ z))
        sol = scipy.linalg.lu_solve(self.S, rhs)
        a = np.zeros(b.shape[0])
        a[self.f] = z - self.Y @ sol
        a[self.pin] = sol[:self.k]
        return a

    def free(self):
        self.F.free()


class EdgeSpace:
    """The independent unknowns of the model, ``u = T a``

    Attributes
    ----------
    free : array-like
        Boolean mask of the degrees of freedom not fixed to zero, the tied
        ones included. This is what the cyclic symmetry helpers take as
        ``bu``, a rotation or an axisymmetric average of an admissible field
        keeping the tied displacements equal.
    T : csc_matrix or None
        The basis, None when it is the selection of ``free``.
    rep : array-like
        One degree of freedom per unknown, where it is read back.
    rigid : array-like or None
        The rigid body modes left free by the edges, as columns over every
        degree of freedom, to be removed by inertia relief; None for the
        edges that suppress them all.
    C : array-like or None
        The inertia relief condition ``C.T a = 0``, orthonormal columns over
        the unknowns, set by :meth:`set_mass`.

    """
    def __init__(self, free, tied=(), rigid=None, rigid_names=()):
        self.free = np.asarray(free, dtype=bool)
        N = self.free.shape[0]
        self.ties = [np.asarray(t, dtype=np.int64) for t in tied]
        self.rigid = rigid
        self.rigid_names = tuple(rigid_names)
        self.C = None
        indep = self.free.copy()
        for t in self.ties:
            assert self.free[t].all()
            indep[t] = False
        idx = np.flatnonzero(indep)
        self.rep = np.concatenate([idx] + [t[:1] for t in self.ties])
        self.size = self.rep.shape[0]
        if not self.ties:
            self.T = None
            self.indep = self.free
            return
        self.indep = indep
        rows = np.concatenate([idx] + self.ties)
        cols = np.concatenate([np.arange(idx.shape[0])]
                + [np.full(t.shape[0], idx.shape[0] + i)
                   for i, t in enumerate(self.ties)])
        self.T = csc_matrix((np.ones(rows.shape[0]), (rows, cols)),
                shape=(N, self.size))

    def matrix(self, K):
        """``T.T K T``, K a sparse matrix over every degree of freedom"""
        #NOTE the selection by indexing when there is no tie, which is what
        #     the models did before the ties, to the last digit
        if self.T is None:
            return K[self.free, :][:, self.free]
        return (self.T.T @ K @ self.T).tocsc()

    def force(self, f):
        """``T.T f``, f a vector, or vectors as columns, of forces"""
        if self.T is None:
            return f[self.free]
        return self.T.T @ f

    def expand(self, a):
        """``T a``, a vector, or vectors as columns, of unknowns"""
        if self.T is None:
            u = np.zeros((self.free.shape[0],) + np.shape(a)[1:])
            u[self.free] = a
            return u
        return self.T @ a

    def restrict(self, u):
        """The unknowns of the displacement ``u = T a``"""
        return u[self.rep]

    def set_mass(self, M):
        """The inertia relief condition, from the consistent mass matrix M
        over every degree of freedom"""
        Rr = self.restrict(self.rigid)
        C = self.force(M @ self.rigid)
        #NOTE orthonormal columns spanning the same condition, so that the
        #     border of the bordered systems is of unit scale
        self.C = np.linalg.qr(C)[0]
        self.Rr = Rr
        #NOTE the eliminated unknowns of ConstrainedSolver, among the nodal
        #     translations u, v and w, chosen by a pivoted QR on the rigid
        #     body modes, so that no combination of them vanishes there
        cand = np.flatnonzero(np.isin(self.rep % self.DOF, (0, 3, 6)))
        piv = scipy.linalg.qr(Rr[cand].T, pivoting=True, mode='r')[1]
        self.pin = np.sort(cand[piv[:Rr.shape[1]]])
        #NOTE the oblique projector along the rigid body modes onto the
        #     null space of C.T
        self.CtR = self.C.T @ Rr

    def remove_rigid(self, a):
        """a without its rigid body component, ``C.T a = 0``"""
        return a - self.Rr @ np.linalg.solve(self.CtR, self.C.T @ a)

    def solver(self, K):
        """:class:`ConstrainedSolver` of the unknowns"""
        return ConstrainedSolver(K, self.C, self.pin)

    def solve(self, K, b, spsolve):
        """K a = b on the unknowns, with the inertia relief condition when
        there is one"""
        if self.C is None:
            return spsolve(K, b)
        cs = self.solver(K)
        a = cs.solve(b)
        cs.free()
        return a

    def eigsh(self, KG, KC, k, v0, tol, eigsh):
        """Buckling multipliers on the null space of C.T, ARPACK in its
        generalized mode with the inverse of KC on that space

        The inverse of KC restricted to the null space of C.T is the
        constrained solve, so every Lanczos vector stays in it when the
        starting vector does; KC is positive definite there, the only place
        where ARPACK uses it as the inner product.
        """
        cs = self.solver(KC)
        Minv = LinearOperator(KC.shape, matvec=cs.solve, dtype=np.float64)
        out = eigsh(A=KG, k=k, which='LM', M=KC, Minv=Minv, tol=tol,
                    v0=self.remove_rigid(v0))
        cs.free()
        return out

    def border(self):
        """The columns that border the bordered system of the second order
        fields, and the rows, with the inertia relief condition"""
        return self.C

    def axisymmetric_condition(self, Baxi, buaxi):
        """The inertia relief condition in the axisymmetric subspace, the
        rigid body modes that are axisymmetric, Tx and Rx, the others being
        orthogonal to it; None when there is none"""
        if self.C is None:
            return None
        Ca = (Baxi.T @ self.expand(self.C))[buaxi]
        U, s = np.linalg.svd(Ca, full_matrices=False)[:2]
        keep = s > 1.e-8*s.max()
        if not keep.any():
            return None
        return U[:, keep]*s[keep]


def edge_space(x, L, DOF, edges='SS3', y=None, R=None):
    """Degrees of freedom fixed and tied at the edges x = 0 and x = L

    Parameters
    ----------
    x : array-like
        Axial coordinate of every node.
    L : float
        Length of the cylinder.
    DOF : int
        Degrees of freedom per node, ordered u, u,x, u,y, v, v,x, v,y, w,
        w,x, w,y, w,xy.
    edges : str, optional
        ``'SS3'``: v = w = 0 along both edges, the axial displacement free,
        and the axial rigid body translation removed at the node at x = L/2,
        y = 0. ``'SS4'``: v = w = 0 and in addition a uniform axial
        displacement along each edge, zero at x = 0 and the same unknown at
        every node of x = L, where the axial load is applied; force control,
        so the load is applied as before and its resultant is the reaction
        of the tied unknown. ``'SS3-IR'``: v = w = 0 along both edges and no
        node anchored, the axial translation removed by inertia relief.
        ``'free-IR'``: no degree of freedom fixed anywhere, the load on both
        edges, all six rigid body modes removed by inertia relief.
    y : array-like, optional
        Circumferential coordinate of every node, for ``'SS3'`` and the
        inertia relief edges.
    R : float, optional
        Radius, for the inertia relief edges.

    """
    if edges not in EDGES:
        raise ValueError('edges must be one of %r, not %r' % (EDGES, edges))
    N = DOF*x.shape[0]
    bk = np.zeros(N, dtype=bool)
    x0 = isclose(x, 0)
    xL = isclose(x, L)
    checkSS = x0 | xL
    tied = []
    if edges != 'free-IR':
        #NOTE every degree of freedom fixed at the edge nodes is fixed
        #     together with its derivative along the edge, d/dy, so that the
        #     Hermite interpolation along the edge makes it zero along the
        #     whole edge and not at the nodes only: v with v,y and w with
        #     w,y. Fixing v and w alone left the edges free to deflect between
        #     the nodes, an error that vanishes only as the circumferential
        #     element length does
        bk[3::DOF] = checkSS
        bk[5::DOF] = checkSS
        bk[6::DOF] = checkSS
        bk[8::DOF] = checkSS
    if edges == 'SS3':
        check = isclose(x, L/2.) & isclose(y, 0)
        assert check.sum() == 1
        bk[0::DOF] = check
    elif edges == 'SS4':
        #NOTE u,y = 0 at the nodes of both edges and u equal at all of them
        #     make u uniform along the whole edge, by the same Hermite
        #     argument; u = 0 at x = 0 also removes the axial rigid body
        #     translation
        bk[0::DOF] = x0
        bk[2::DOF] = checkSS
        tied.append(DOF*np.flatnonzero(xL))
    rigid = None
    names = ()
    if edges.endswith('-IR'):
        #NOTE the rigid body modes that the fixed degrees of freedom leave
        #     free, those that vanish on all of them: Tx alone with v = w = 0
        #     on the edges, all six on free edges
        modes = rigid_body_modes(x, y, R, DOF)
        keep = [j for j in range(6) if not np.any(modes[bk, j])]
        rigid = modes[:, keep]
        names = tuple(RIGID[j] for j in keep)
    space = EdgeSpace(~bk, tied, rigid, names)
    space.DOF = DOF
    return space
