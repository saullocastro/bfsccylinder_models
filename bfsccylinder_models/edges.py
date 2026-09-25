"""
Edge conditions of the cylinder models
======================================

Every model of this package supports the cylinder in the same way, SS3 with
inertia relief:

- v = w = 0 along both edges, x = 0 and x = L, the axial displacement u and
  the rotation free;
- the axial load applied on both edges, self-equilibrated;
- no node anchored: the axial rigid body translation, the only rigid body
  mode these edges leave, is removed by the inertia relief condition
  ``C.T a = 0``, ``C = M r``, with ``M`` the consistent mass matrix and ``r``
  the translation, so that the centre of mass does not move axially.

Every degree of freedom fixed at the edge nodes is fixed together with its
derivative along the edge, v with v,y and w with w,y, since the Hermite
interpolation builds v and w along the edge from both: fixing v and w alone
makes them vanish at the nodes only, the edge being free to deflect between
them.

The condition is imposed exactly, by :class:`ConstrainedSolver` in the static
solves, as the inverse of the eigen solver in the buckling analysis
(:meth:`EdgeSpace.eigsh`), as a border of the bordered system of the second
order fields and of the axisymmetric pre-buckling solves. The translation is
a null vector of every operator, so the buckling loads and the Koiter
coefficients are those of the same edges with u fixed at one node, to round
off; the displacements differ from them by a rigid translation.

"""
import warnings

import numpy as np
from numpy import isclose
import scipy.linalg
from scipy.sparse import csc_matrix, coo_matrix, triu, diags
from scipy.sparse.linalg import LinearOperator, splu


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


class NotPositiveDefinite(ValueError):
    """The matrix given to :class:`Factor` is not positive definite"""


class Factor:
    """Factorization of a sparse symmetric positive definite matrix A

    The symmetric diagonal scaling ``D A D``, ``D = diag(A)**-1/2``, is
    factorized, which brings the diagonal of these stiffness matrices, whose
    nodal displacements and nodal derivatives span about 12 orders of
    magnitude, to one; then every solve is refined, ``x += D (DAD)^-1 D
    (b - A x)``, until the relative residual is below ``rtol`` or stops
    decreasing. The Cholesky factorization of PARDISO when pypardiso is
    installed, SuperLU otherwise, in symmetric mode and without row
    interchanges, so that both detect a matrix that is not positive definite
    and raise :class:`NotPositiveDefinite`.
    """
    def __init__(self, A, rtol=1.e-10, max_refine=3):
        A = csc_matrix(A)
        diag = A.diagonal()
        if not np.all(diag > 0):
            raise NotPositiveDefinite('non-positive diagonal')
        self.A = A
        self.d = 1/np.sqrt(diag)
        D = diags(self.d)
        As = (D @ A @ D).tocsc()
        self.rtol, self.max_refine = rtol, max_refine
        self.pardiso = None
        try:
            import pypardiso
        except ImportError:
            pypardiso = None
        if pypardiso is not None:
            solver = pypardiso.PyPardisoSolver(mtype=2)
            Au = triu(As, format='csr')
            try:
                solver.factorize(Au)
            except pypardiso.pardiso_wrapper.PyPardisoError as e:
                solver.free_memory(everything=True)
                raise NotPositiveDefinite(str(e))
            self.pardiso = (solver, Au)
            return
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lu = splu(As, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.,
                      options=dict(SymmetricMode=True))
        if (not np.array_equal(lu.perm_r, lu.perm_c)
                or not np.all(lu.U.diagonal() > 0)):
            raise NotPositiveDefinite('negative pivot')
        self.lu = lu

    def _solve_scaled(self, b):
        if self.pardiso is not None:
            solver, Au = self.pardiso
            return solver.solve(Au, np.ascontiguousarray(b))
        return self.lu.solve(b)

    def solve(self, b):
        d = self.d if np.ndim(b) == 1 else self.d[:, None]
        x = d*self._solve_scaled(d*b)
        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            return x
        res = np.linalg.norm(b - self.A @ x)/norm_b
        for _ in range(self.max_refine):
            if res <= self.rtol:
                break
            dx = d*self._solve_scaled(d*(b - self.A @ x))
            res_new = np.linalg.norm(b - self.A @ (x + dx))/norm_b
            if res_new >= res:
                break
            x, res = x + dx, res_new
        return x

    def free(self):
        if self.pardiso is not None:
            self.pardiso[0].free_memory(everything=True)
            self.pardiso = None


class ConstrainedSolver:
    """Solution of ``K a + C lam = b``, ``C.T a = 0``, K symmetric and
    positive definite on the null space of C.T

    By elimination: the k unknowns ``pin``, chosen so that the rigid body
    modes do not vanish on them, are eliminated last, so that the rest of K
    is positive definite and is factorized once, and ``[a_pin, lam]`` solve a
    dense system of size 2k. This is only a partition of the unknowns of the
    constrained system, whose solution is that of the inertia relief whatever
    pin is; the ``pin`` unknowns are not constrained. ``lam`` are the inertia
    forces, zero for a self-equilibrated load.
    """
    def __init__(self, K, C, pin):
        n, k = C.shape
        K = csc_matrix(K)
        f = np.ones(n, dtype=bool)
        f[pin] = False
        self.f, self.pin, self.C, self.k = f, pin, C, k
        Kf = K[f, :]
        self.Kfp = Kf[:, pin].toarray()
        Kpp = K[pin, :][:, pin].toarray()
        self.F = Factor(Kf[:, f])
        Cf, Cp = C[f], C[pin]
        self.Y = self.F.solve(np.column_stack((self.Kfp, Cf)))
        Yk, Yc = self.Y[:, :k], self.Y[:, k:]
        S = np.block([[Kpp - self.Kfp.T @ Yk, Cp - self.Kfp.T @ Yc],
                      [Cp.T - Cf.T @ Yk, -Cf.T @ Yc]])
        self.S = scipy.linalg.lu_factor(S)

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
    """The unknowns of the model and the inertia relief condition on them

    Attributes
    ----------
    free : array-like
        Boolean mask of the unknown degrees of freedom, those not fixed at
        the edges.
    size : int
        Number of unknowns.
    C : array-like
        The inertia relief condition ``C.T a = 0``, one orthonormal column
        over the unknowns, set by :meth:`set_mass`.

    """
    def __init__(self, free, DOF):
        self.free = np.asarray(free, dtype=bool)
        self.size = int(self.free.sum())
        #NOTE the axial translation, u = 1 at every node
        r = np.zeros(self.free.shape[0])
        r[0::DOF] = 1.
        self.rigid = r[:, None]
        self.Rr = self.rigid[self.free]
        self.DOF = DOF
        self.C = None

    def matrix(self, K):
        """K over the unknowns, K a sparse matrix over every degree of
        freedom"""
        return K[self.free, :][:, self.free]

    def force(self, f):
        """f over the unknowns, f a vector, or vectors as columns"""
        return f[self.free]

    def expand(self, a):
        """The displacement over every degree of freedom, a a vector, or
        vectors as columns, over the unknowns"""
        u = np.zeros((self.free.shape[0],) + np.shape(a)[1:])
        u[self.free] = a
        return u

    def restrict(self, u):
        """The unknowns of the displacement u"""
        return u[self.free]

    def set_mass(self, M):
        """The inertia relief condition, from the consistent mass matrix M
        over every degree of freedom"""
        C = self.force(M @ self.rigid)
        self.C = C/np.linalg.norm(C)
        #NOTE the eliminated unknown of ConstrainedSolver, the axial
        #     displacement of the node where the condition weighs most
        cand = np.flatnonzero(self.Rr[:, 0])
        self.pin = cand[np.argmax(np.abs(self.C[cand, 0]))][None]
        self.CtR = self.C.T @ self.Rr

    def remove_rigid(self, a):
        """a without its rigid body component, ``C.T a = 0``"""
        return a - self.Rr @ np.linalg.solve(self.CtR, self.C.T @ a)

    def solver(self, K):
        """:class:`ConstrainedSolver` of the unknowns"""
        return ConstrainedSolver(K, self.C, self.pin)

    def solve(self, K, b):
        """K a = b on the unknowns, with the inertia relief condition"""
        cs = self.solver(K)
        a = cs.solve(b)
        cs.free()
        return a

    def eigsh(self, KG, KC, k, v0, tol, eigsh, mu_est=None, shift=0.9,
              max_tries=4):
        """The k eigenvalues theta of ``KG q = theta KC q`` of largest
        magnitude on the null space of C.T, the smallest buckling multipliers
        mu = -1/theta, in ascending order of theta, and their eigenvectors

        With an estimate mu_est of the smallest multiplier, ARPACK in
        shift-invert mode about theta = -1/sigma, sigma = shift*mu_est, below
        the critical multiplier: the operator is the inverse of KC + sigma KG
        on the null space of C.T, which is positive definite when no
        multiplier lies below sigma. Its factorization proves it, and a
        failed one lowers sigma by the factor 0.7, up to max_tries times.
        The shift maps the near critical cluster, whose multipliers are
        within a few per cent of each other, to eigenvalues 1/(mu - sigma)
        about ten times better separated than those of the unshifted
        operator, which is what ARPACK converges on.

        Without mu_est, or when every shift fails, ARPACK in its generalized
        mode with the inverse of KC on the null space of C.T.

        Either inverse is the constrained solve, so every Lanczos vector
        stays in the null space of C.T when the starting vector does; KC is
        positive definite there. The generalized mode uses KC as the inner
        product. The shift-invert mode does not: KC is singular along the
        translation, and components the round off leaves outside the null
        space of C.T, which its inner product does not see, grow until the
        Ritz vectors are almost parallel to C (residuals of 30 on the Waters
        shell at ny = 40). Its inner product is ``KC + rho C C.T`` instead,
        positive definite, and equal to KC on the null space of C.T. eigsh is
        the eigen solver, scipy.sparse.linalg.eigsh or one with its
        signature.
        """
        v0 = self.remove_rigid(v0)
        if mu_est is not None and mu_est > 0:
            sigma = shift*mu_est
            for _ in range(max_tries):
                try:
                    cs = self.solver(KC + sigma*KG)
                except NotPositiveDefinite:
                    print('# a buckling multiplier below the shift %r, '
                          'lowering it' % sigma)
                    sigma *= 0.7
                    continue
                #NOTE (KG + KC/sigma)^-1 = sigma (KC + sigma KG)^-1
                OPinv = LinearOperator(KC.shape, dtype=np.float64,
                        matvec=lambda b, cs=cs, s=sigma: s*cs.solve(b))
                C = self.C[:, 0]
                rho = KC.diagonal().mean()
                B = LinearOperator(KC.shape, dtype=np.float64,
                        matvec=lambda v: KC @ v + rho*C*(C @ v))
                theta, q = eigsh(A=KG, k=k, which='LM', M=B,
                        sigma=-1/sigma, OPinv=OPinv, tol=tol, v0=v0)
                cs.free()
                order = np.argsort(theta)
                return theta[order], q[:, order]
        cs = self.solver(KC)
        Minv = LinearOperator(KC.shape, matvec=cs.solve, dtype=np.float64)
        out = eigsh(A=KG, k=k, which='LM', M=KC, Minv=Minv, tol=tol, v0=v0)
        cs.free()
        return out

    def axisymmetric_condition(self, Baxi, buaxi):
        """The inertia relief condition in the coordinates of the
        axisymmetric basis Baxi, the translation being axisymmetric"""
        return (Baxi.T @ self.expand(self.C))[buaxi]


def edge_space(x, L, DOF):
    """The unknowns of the SS3 edges with inertia relief, see the module
    docstring

    Parameters
    ----------
    x : array-like
        Axial coordinate of every node.
    L : float
        Length of the cylinder.
    DOF : int
        Degrees of freedom per node, ordered u, u,x, u,y, v, v,x, v,y, w,
        w,x, w,y, w,xy.

    """
    N = DOF*x.shape[0]
    bk = np.zeros(N, dtype=bool)
    checkSS = isclose(x, 0) | isclose(x, L)
    bk[3::DOF] = checkSS
    bk[5::DOF] = checkSS
    bk[6::DOF] = checkSS
    bk[8::DOF] = checkSS
    return EdgeSpace(~bk, DOF)
