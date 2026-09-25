import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np
from scipy.sparse import csc_matrix
from composites import laminated_plate
from bfsccylinder.sanders import DOF

from bfsccylinder_models.edges import edge_space, ConstrainedSolver
from bfsccylinder_models.koiter_cylinder_sanders import fkoiter_cyl_SS3
from bfsccylinder_models.linbuck_VAFW import flinBuck_VAFW
from bfsccylinder_models.vatfunctions import func_VAT_P_x

L = 0.3556 # m
R = 0.20318603 # m
E11 = 127.629e9 # Pa
E22 = 11.3074e9 # Pa
G12 = 6.00257e9 # Pa
nu12 = 0.300235
plyt = 0.00012692375 # m


def _mesh(nx, ny):
    x, y = np.meshgrid(np.linspace(0, L, nx),
                       np.linspace(0, 2*np.pi*R, ny, endpoint=False),
                       indexing='ij')
    return x.ravel(), y.ravel()


def _waters(ny):
    """The shell of test_koiter_cylinder_Waters_sanders.py"""
    stack = [45, -45, 0, 90, 90, 0, -45, 45]
    nx = int(ny*L/(2*np.pi*R))
    if nx % 2 == 0:
        nx += 1
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    prop = laminated_plate(stack=stack, laminaprop=laminaprop, plyt=plyt,
            offset=0, rho=1611)
    return nx, prop


def test_edge_space():
    """v, v,y, w and w,y fixed on both edges, nothing else, and the
    condition along the axial translation"""
    x, y = _mesh(5, 8)
    space = edge_space(x, L, DOF)
    U = space.expand(np.arange(1., space.size + 1)).reshape(-1, DOF)
    edges = np.isclose(x, 0) | np.isclose(x, L)
    for d in range(DOF):
        fixed = d in (3, 5, 6, 8)
        assert np.all((U[edges, d] == 0) == fixed)
        assert np.all(U[~edges, d] != 0)
    #NOTE with a unit mass matrix the condition is the mean of u
    space.set_mass(csc_matrix(np.eye(DOF*x.shape[0])))
    a = np.random.default_rng(1).random(space.size)
    a = space.remove_rigid(a)
    assert abs(space.C[:, 0] @ a) < 1e-12
    assert abs(space.expand(a).reshape(-1, DOF)[:, 0].mean()) < 1e-12


def test_constrained_solver():
    """Against the dense solution of the bordered system, on a matrix
    singular along r, as the stiffness matrix is along the translation"""
    rng = np.random.default_rng(2)
    n = 12
    r = np.ones(n)/np.sqrt(n)
    A = rng.random((n, n))
    P = np.eye(n) - np.outer(r, r)
    K = P @ (A @ A.T) @ P
    C = rng.random((n, 1))
    C /= np.linalg.norm(C)
    b = rng.random(n)
    b -= r*(r @ b)
    dense = np.linalg.solve(np.block([[K, C], [C.T, np.zeros((1, 1))]]),
                            np.concatenate((b, [0.])))
    for pin in (0, 5):
        cs = ConstrainedSolver(csc_matrix(K), C, np.array([pin]))
        a = cs.solve(b)
        cs.free()
        assert np.allclose(a, dense[:n], rtol=1e-10, atol=1e-12)


def test_Waters_shell():
    """No node anchored: the buckling load and b of the SS3 edges with u
    fixed at the node x = L/2, y = 0, the translation being a null vector of
    every operator; the edge conditions hold on every field"""
    ny = 40
    nx, prop = _waters(ny)
    #NOTE the values of the model with u fixed at the node x = L/2, y = 0,
    #     up to commit 1fc65e2, for the linear and the nonlinear pre-buckling
    #     state
    reference = {False: (184158.23319749543, -0.044063121737014536),
                 True: (177473.77795087988, -0.052189040194737864)}
    for NLprebuck, (Pcr, b) in reference.items():
        out = fkoiter_cyl_SS3(L, R, nx, ny, prop, num_eigvals=4,
                koiter_num_modes=1, Nxxunit=1000., NLprebuck=NLprebuck,
                NLprebuck_eps1=0.0005)
        assert np.isclose(out['Pcr'], Pcr, rtol=1e-8)
        assert np.isclose(out['koiter']['b_ijkl'][(0, 0, 0, 0)], b,
                          rtol=1e-6)
        x = out['x']
        edges = np.isclose(x, 0) | np.isclose(x, L)
        for u in (out['koiter']['u0'], out['eigvecs'][:, 0],
                  out['koiter']['uij'][(0, 0)]):
            U = u.reshape(-1, DOF)
            #NOTE the mass is uniform along x on this shell, so the condition
            #     is close to a zero mean of the nodal u
            assert abs(U[:, 0].mean()) <= 1e-3*np.abs(U[:, 0]).max()
            assert np.abs(U[edges][:, [3, 5, 6, 8]]).max() == 0


def test_shift_invert_eigsh(monkeypatch):
    """The multipliers of EdgeSpace.eigsh in shift-invert mode are the
    smallest ones, degenerate pairs complete, also when the estimate is above
    the critical multiplier and the shift has to be lowered; the modes stay
    in the null space of the inertia relief condition"""
    import bfsccylinder_models.edges as edges
    from scipy.sparse.linalg import eigsh
    saved = {}
    eigsh_space = edges.EdgeSpace.eigsh

    def keep(self, KG, KC, k, v0, tol, eig, mu_est=None, **kwargs):
        saved.update(space=self, KG=KG, KC=KC, v0=v0)
        return eigsh_space(self, KG, KC, k, v0, tol, eig, mu_est=mu_est,
                           **kwargs)

    monkeypatch.setattr(edges.EdgeSpace, 'eigsh', keep)
    ny = 24
    nx, prop = _waters(ny)
    fkoiter_cyl_SS3(L, R, nx, ny, prop, num_eigvals=4, koiter_num_modes=0,
            Nxxunit=1000.)
    space, KG, KC, v0 = (saved[k] for k in ('space', 'KG', 'KC', 'v0'))
    #NOTE the dense solution on the null space of C.T. Either mode may
    #     return one member of a degenerate pair only, as decided by round
    #     off, which the models make up for, see canonical_modes; what is
    #     checked is that every multiplier returned is one of the spectrum,
    #     the first being the critical one
    import scipy.linalg
    Z = scipy.linalg.null_space(space.C.T)
    theta = scipy.linalg.eigh(Z.T @ (KG @ Z), Z.T @ (KC @ Z),
                              eigvals_only=True)
    spectrum = np.sort(-1/theta[theta < 0])
    for estimate in (1.02*spectrum[0], 3*spectrum[0]):
        theta_s, q = eigsh_space(space, KG, KC, 4, v0, 1e-8, eigsh,
                                 mu_est=estimate)
        mu = np.sort(-1/theta_s)
        assert np.isclose(mu[0], spectrum[0], rtol=1e-8)
        gap = np.abs(mu[:, None] - spectrum[None, :]).min(axis=1)
        assert gap.max() <= 1e-8*spectrum[0]
        assert np.abs(space.C[:, 0] @ q).max() <= 1e-10*np.abs(q).max()
        res = np.linalg.norm(KG @ q - (KC @ q)*theta_s, axis=0)
        assert res.max() <= 1e-6*np.linalg.norm(KG @ q, axis=0).max()


def test_linbuck_against_koiter_cylinder():
    """flinBuck_VAFW with a constant laminate is the linear buckling
    analysis of koiter_cylinder.fkoiter_cyl_SS3, same element and edges"""
    from bfsccylinder_models.koiter_cylinder import fkoiter_cyl_SS3 as fk
    ny = 40
    nx, _ = _waters(ny)
    #NOTE balanced plies as flinBuck_VAFW builds them, at their tow
    #     thickness, and its offset of half the laminate thickness
    thetas = [45., 0., 90.]
    desvars = [[t, t, t] for t in thetas]
    stack = sum([[t, -t] for t in thetas], [])
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    prop = laminated_plate(stack=stack, plyts=[plyt]*len(stack),
            laminaprop=laminaprop, offset=len(stack)*plyt/2, rho=1611)
    out = flinBuck_VAFW(L, R, nx, ny, E11, E22, nu12, G12, 1611, plyt,
            desvars, func_VAT_P_x, num_eigvals=4, Nxxunit=1000.)
    ref = fk(L, R, nx, ny, prop, num_eigvals=4, koiter_num_modes=0,
             Nxxunit=1000.)
    assert np.isclose(out['Pcr'], ref['Pcr'], rtol=1e-6)


if __name__ == '__main__':
    test_edge_space()
    test_constrained_solver()
    test_Waters_shell()
    test_linbuck_against_koiter_cylinder()
