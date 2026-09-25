"""
Cyclic symmetry helpers for the structured cylinder meshes
==========================================================

The meshes used by the models of this package have ny elements around the
circumference and carry element degrees of freedom expressed in the local
shell frame, which turns with the circumferential coordinate. Shifting the
circumferential node index cyclically is therefore a symmetry of every
discrete operator assembled on them, and the functions below are the
consequences of that symmetry which the Koiter analyses need: the projection
onto the axisymmetric subspace, where the pre-buckling state of a perfect
cylinder under axial compression lives, and the removal of the arbitrary
rotation inside the degenerate buckling mode pairs.

"""
import numpy as np
from scipy.sparse import csc_matrix


def axisymmetric_basis(axi_order, bu, DOF):
    """Basis of the axisymmetric subspace, and its unknown coordinates

    One column per axial station and per degree of freedom, carrying one at
    that degree of freedom of every node of the station, so that an
    axisymmetric field is ``B a`` and the columns are orthogonal with squared
    norm ny.

    Solving the pre-buckling problem as the Galerkin projection onto this
    basis, ``B.T K B a = B.T f``, states directly what solving the full
    system and projecting the correction afterwards only arrives at
    indirectly, and it is far smaller: nx*DOF unknowns against the full
    count, 250 against 15000 on the ny=60 mesh of the Arbocz and Starnes
    case. It also makes the deflation of the buckling modes from the
    correction unnecessary, the whole near critical cluster being outside the
    subspace by construction.

    The two give the same iterates here, to eleven digits on the meshes of
    the test suite, so this is a simplification and a saving rather than a
    correction. Nor is it what made the Newton-Raphson converge on fine
    meshes, which was the consistent tangent stiffness matrix of
    bfsccylinder 0.6.0.

    Returns the basis and the boolean mask of its unknown coordinates. A
    reduced coordinate is known as soon as one of the nodes it spans is
    constrained; the edge conditions of the models constrain every node of an
    edge station alike.
    """
    nx, ny = axi_order.shape
    rows = np.empty((nx, ny, DOF), dtype=np.int64)
    cols = np.empty((nx, ny, DOF), dtype=np.int64)
    for d in range(DOF):
        rows[:, :, d] = DOF*axi_order + d
        cols[:, :, d] = DOF*np.arange(nx)[:, None] + d
    B = csc_matrix((np.ones(nx*ny*DOF), (rows.reshape(-1), cols.reshape(-1))),
                   shape=(bu.shape[0], nx*DOF))
    bkr = (~bu).reshape(-1, DOF)[axi_order].any(axis=1).reshape(-1)
    return B, ~bkr


def mesh_order(x, y, nx, ny):
    """Node positions arranged as (axial station, circumferential station)

    Returned as an integer array of shape (nx, ny) holding, at (i, j), the
    position of the node at the i-th axial and j-th circumferential station,
    so that ``u.reshape(-1, DOF)[mesh_order(...)]`` gives the displacement
    field laid out on the mesh.
    """
    return np.lexsort((y, x)).reshape(nx, ny)


def project_axisymmetric(u, axi_order, DOF):
    """Orthogonal projection of u onto the axisymmetric subspace

    The average over the orbit of the cyclic shift is the projector onto its
    invariant subspace. It keeps the inertia relief condition of the models,
    the mass being uniform around the circumference.
    """
    nx, ny = axi_order.shape
    U = u.reshape(-1, DOF)[axi_order]
    U = np.repeat(U.mean(axis=1, keepdims=True), ny, axis=1)
    uaxi = np.zeros((nx*ny, DOF), dtype=np.float64)
    uaxi[axi_order] = U
    return uaxi.reshape(-1)


def rotated(u, axi_order, DOF, shift=1):
    """The field rotated by shift circumferential element widths

    A cyclic shift of the circumferential node index is a symmetry of every
    operator assembled on these meshes, so it maps a buckling mode onto
    another buckling mode carrying the same multiplier.
    """
    nx, ny = axi_order.shape
    U = np.roll(u.reshape(-1, DOF)[axi_order], shift, axis=1)
    rot = np.zeros((nx*ny, DOF), dtype=np.float64)
    rot[axi_order] = U
    return rot.reshape(-1)


def degenerate_partner(phi, bu, axi_order, DOF):
    """The other member of the degenerate pair of the mode phi

    Built from the rotation of phi by one element, :func:`rotated`, the
    cyclic symmetry of the mesh making it another mode of the same pair,
    orthogonalized against phi and scaled to its norm.

    Returns None when there is no partner to build: an axisymmetric mode is
    its own rotation, and so, up to the sign, is the mode with ny/2
    circumferential waves; and a rotation that leaves anything on a
    constrained degree of freedom cannot be brought back into the admissible
    space without leaving the eigenspace.

    Parameters
    ----------
    phi : array-like
        The mode, over every degree of freedom.
    bu : array-like
        Boolean mask of the unknown degrees of freedom.
    axi_order : array-like
        Node positions on the mesh, from :func:`mesh_order`.
    DOF : int
        Degrees of freedom per node.

    """
    phi = np.asarray(phi, dtype=np.float64)
    phi_norm = np.sqrt(phi @ phi)
    psi = rotated(phi, axi_order, DOF)
    #NOTE the edge conditions and the inertia relief condition of the models
    #     are invariant under the rotation, so the rotated mode satisfies
    #     them; anything it still left on a constrained degree of freedom
    #     would have to be truncated, and the result would no longer be a
    #     mode
    if np.abs(psi[~bu]).max(initial=0.) > 1.e-10*np.abs(psi).max():
        return None
    psi -= (psi @ phi)/(phi @ phi)*phi
    psi_norm = np.sqrt(psi @ psi)
    if psi_norm <= 1.e-8*phi_norm:
        return None
    return psi*(phi_norm/psi_norm)


def canonical_modes(mu, eigvecsu, bu, axi_order, DOF, deg_rtol=1.e-5):
    """Fix the rotation left free inside each degenerate group of modes

    The buckling modes of a cylinder come in degenerate pairs, one for each
    sign of the circumferential wave number, and every rotation of a pair is
    again a pair of buckling modes. Which member of it comes out of the eigen
    solver is decided by round off, and b_ijkl is not invariant under that
    rotation: on the ny = 40 meshes of the test suite a rotation by 30 degrees
    moves b_1111 by 16% on the Sun et al. case and by 22% on the Arbocz and
    Starnes case, almost all of it through the normalization of the mode by
    its largest nodal translation, which misses the crest of these skewed
    modes. Each pair is therefore rotated to the member whose crest falls on
    the y = 0 generator, a property of the mesh and not of the arithmetic.
    That makes b_ijkl reproducible; it does not make it mesh converged.

    The partner of a mode need not be among the ones the eigen solver
    returned either: whether ARPACK returns both members of a pair, or one
    member and then the next multiplier, is decided by round off too. A
    missing partner is recovered here through :func:`rotated`, the cyclic
    symmetry of the mesh making the rotation of a mode another mode of the
    same pair. See Section "Buckling modes of a cylinder" of
    doc/nlprebuck_implementation.tex for the measurements.

    Parameters
    ----------
    mu : array-like
        Buckling multipliers, used only to find the degenerate groups.
    eigvecsu : array-like
        Eigenvectors as columns, over the unknown degrees of freedom.
    bu : array-like
        Boolean mask of the unknown degrees of freedom.
    axi_order : array-like
        Node positions on the mesh, from :func:`mesh_order`.
    DOF : int
        Degrees of freedom per node.
    deg_rtol : float, optional
        Two multipliers closer than this, relatively, belong to one group.

    """
    mu = np.asarray(mu)
    eigvecsu = np.array(eigvecsu, copy=True)
    num_eigvals = mu.shape[0]
    #NOTE radial degrees of freedom along the y = 0 generator
    rows = DOF*axi_order[:, 0] + 6
    done = np.zeros(num_eigvals, dtype=bool)
    for i in range(num_eigvals):
        if done[i]:
            continue
        grp = [j for j in range(i, num_eigvals)
               if not done[j] and abs(mu[j] - mu[i]) <= deg_rtol*abs(mu[i])]
        for j in grp:
            done[j] = True
        #NOTE a group of any other size is outside the pairing described above
        #     and is left as the eigen solver returned it
        if len(grp) > 2:
            continue
        i0 = grp[0]
        i1 = grp[1] if len(grp) == 2 else None
        phi = np.zeros(bu.shape[0], dtype=np.float64)
        phi[bu] = eigvecsu[:, i0]
        if i1 is not None:
            psi = np.zeros(bu.shape[0], dtype=np.float64)
            psi[bu] = eigvecsu[:, i1]
        else:
            psi = degenerate_partner(phi, bu, axi_order, DOF)
            #NOTE no partner, or none that could be kept admissible, so the
            #     mode is left alone rather than canonicalized with a
            #     corrupted one
            if psi is None:
                continue
        #NOTE of the two members of a pair one has a crest on the generator
        #     and the other a node, so the dominant eigenvector of the 2 by 2
        #     Gram matrix of their traces there is well separated
        Wg = np.column_stack((phi[rows], psi[rows]))
        c = np.linalg.eigh(Wg.T @ Wg)[1][:, -1]
        trace = Wg @ c
        #NOTE and pointing outwards, so that the mode itself, and not only
        #     b_ijkl, is reproducible
        if trace[np.argmax(np.abs(trace))] < 0:
            c = -c
        eigvecsu[:, i0] = (c[0]*phi + c[1]*psi)[bu]
        if i1 is not None:
            eigvecsu[:, i1] = (-c[1]*phi + c[0]*psi)[bu]
    return eigvecsu
