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

"""
import numpy as np
from numpy import isclose
from scipy.sparse import csc_matrix

EDGES = ('SS3', 'SS4')


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

    """
    def __init__(self, free, tied=()):
        self.free = np.asarray(free, dtype=bool)
        N = self.free.shape[0]
        self.ties = [np.asarray(t, dtype=np.int64) for t in tied]
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


def edge_space(x, L, DOF, edges='SS3', y=None):
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
        y = 0, which requires ``y``. ``'SS4'``: v = w = 0 and in addition a
        uniform axial displacement along each edge, zero at x = 0 and the
        same unknown at every node of x = L, where the axial load is
        applied; force control, so the load is applied as before and its
        resultant is the reaction of the tied unknown.
    y : array-like, optional
        Circumferential coordinate of every node, for ``'SS3'``.

    """
    if edges not in EDGES:
        raise ValueError('edges must be one of %r, not %r' % (EDGES, edges))
    N = DOF*x.shape[0]
    bk = np.zeros(N, dtype=bool)
    x0 = isclose(x, 0)
    xL = isclose(x, L)
    checkSS = x0 | xL
    #NOTE every degree of freedom fixed at the edge nodes is fixed together
    #     with its derivative along the edge, d/dy, so that the Hermite
    #     interpolation along the edge makes it zero along the whole edge and
    #     not at the nodes only: v with v,y and w with w,y. Fixing v and w
    #     alone left the edges free to deflect between the nodes, an error
    #     that vanishes only as the circumferential element length does
    bk[3::DOF] = checkSS
    bk[5::DOF] = checkSS
    bk[6::DOF] = checkSS
    bk[8::DOF] = checkSS
    tied = []
    if edges == 'SS3':
        check = isclose(x, L/2.) & isclose(y, 0)
        assert check.sum() == 1
        bk[0::DOF] = check
    else:
        #NOTE u,y = 0 at the nodes of both edges and u equal at all of them
        #     make u uniform along the whole edge, by the same Hermite
        #     argument; u = 0 at x = 0 also removes the axial rigid body
        #     translation
        bk[0::DOF] = x0
        bk[2::DOF] = checkSS
        tied.append(DOF*np.flatnonzero(xL))
    return EdgeSpace(~bk, tied)
