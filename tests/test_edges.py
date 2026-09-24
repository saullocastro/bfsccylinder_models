import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np
from composites import laminated_plate
from bfsccylinder.sanders import DOF

from bfsccylinder_models.edges import edge_space
from bfsccylinder_models.koiter_cylinder_sanders import fkoiter_cyl_SS3

L = 0.3556 # m
R = 0.20318603 # m


def _mesh(nx, ny):
    x, y = np.meshgrid(np.linspace(0, L, nx),
                       np.linspace(0, 2*np.pi*R, ny, endpoint=False),
                       indexing='ij')
    return x.ravel(), y.ravel()


def test_SS3_space_is_the_selection():
    x, y = _mesh(5, 8)
    space = edge_space(x, L, DOF, edges='SS3', y=y)
    assert space.T is None
    a = np.arange(space.size, dtype=float)
    u = space.expand(a)
    assert np.all(u[~space.free] == 0)
    assert np.all(space.restrict(u) == a)


def test_SS4_space():
    x, y = _mesh(5, 8)
    space = edge_space(x, L, DOF, edges='SS4')
    xL = np.isclose(x, L)
    x0 = np.isclose(x, 0)
    a = np.random.default_rng(1).random(space.size)
    u = space.expand(a).reshape(-1, DOF)
    assert np.all(space.restrict(u.ravel()) == a)
    #NOTE u uniform at x = L, zero at x = 0, u,y, v, v,y, w, w,y zero on both
    assert np.ptp(u[xL, 0]) == 0 and u[xL, 0][0] != 0
    assert np.all(u[x0, 0] == 0)
    for d in (2, 3, 5, 6, 8):
        assert np.all(u[xL | x0, d] == 0)
    #NOTE T.T f of the tied unknown is the resultant of the nodal forces
    f = np.zeros(DOF*x.shape[0])
    f[DOF*np.flatnonzero(xL)] = 2.
    assert np.isclose(space.force(f)[-1], 2.*xL.sum())


def test_SS4_Waters_shell():
    #NOTE the shell of test_koiter_cylinder_Waters_sanders.py
    ny = 40
    E11 = 127.629e9 # Pa
    E22 = 11.3074e9 # Pa
    G12 = 6.00257e9 # Pa
    nu12 = 0.300235
    stack = [45, -45, 0, 90, 90, 0, -45, 45]
    plyt = 0.00012692375 # m
    nx = int(ny*L/(2*np.pi*R))
    if nx % 2 == 0:
        nx += 1
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    prop = laminated_plate(stack=stack, laminaprop=laminaprop, plyt=plyt,
            offset=0, rho=1611)
    Nxxunit = 1000.
    out = fkoiter_cyl_SS3(L, R, nx, ny, prop, num_eigvals=4,
            koiter_num_modes=1, Nxxunit=Nxxunit, edges='SS4')
    x = out['x']
    xL = np.isclose(x, L)
    x0 = np.isclose(x, 0)
    #NOTE the pre-buckling state, the mode and the second order field all
    #     have a uniform u along x = L and u = 0 along x = 0
    koiter = out['koiter']
    for u in (koiter['u0'], out['eigvecs'][:, 0], koiter['uij'][(0, 0)]):
        U = u.reshape(-1, DOF)
        scale = np.abs(U).max()
        assert np.ptp(U[xL, 0]) <= 1e-12*scale
        assert np.abs(U[x0, 0]).max() <= 1e-12*scale
        assert np.abs(U[xL | x0, 2]).max() <= 1e-12*scale
    #NOTE the edges stiffer than the SS3 ones, u being uniform along them
    ref = fkoiter_cyl_SS3(L, R, nx, ny, prop, num_eigvals=4,
            koiter_num_modes=0, Nxxunit=Nxxunit)
    assert out['Pcr'] > ref['Pcr']
    b = koiter['b_ijkl'][(0, 0, 0, 0)]
    print('Pcr', out['Pcr'], 'SS3', ref['Pcr'], 'b_1111', b)
    #NOTE regression values of the SS4 edges
    assert np.isclose(out['Pcr'], 194509.68706825865, rtol=1e-6)
    assert np.isclose(b, 0.13687095361424786, rtol=1e-4)


def _waters(ny):
    E11 = 127.629e9 # Pa
    E22 = 11.3074e9 # Pa
    G12 = 6.00257e9 # Pa
    nu12 = 0.300235
    stack = [45, -45, 0, 90, 90, 0, -45, 45]
    plyt = 0.00012692375 # m
    nx = int(ny*L/(2*np.pi*R))
    if nx % 2 == 0:
        nx += 1
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    prop = laminated_plate(stack=stack, laminaprop=laminaprop, plyt=plyt,
            offset=0, rho=1611)
    return nx, prop


def test_rigid_body_modes():
    #NOTE Tx and Rx are in the finite element space, and every mode is
    #     rigid: the stiffness matrix annihilates them to round off or to the
    #     interpolation error of cos and sin, see rigid_body_modes
    from bfsccylinder_models.edges import rigid_body_modes, mass_matrix
    x, y = _mesh(5, 8)
    modes = rigid_body_modes(x, y, R, DOF)
    assert modes.shape == (DOF*x.shape[0], 6)
    assert np.linalg.matrix_rank(modes) == 6


def test_SS3_IR_is_SS3():
    """No node anchored, the axial translation removed by inertia relief:
    the rigid translation is a null vector of every operator, so Pcr and b
    are those of SS3"""
    ny = 40
    nx, prop = _waters(ny)
    for NLprebuck in (False, True):
        out = {}
        for edges in ('SS3', 'SS3-IR'):
            out[edges] = fkoiter_cyl_SS3(L, R, nx, ny, prop, num_eigvals=4,
                    koiter_num_modes=1, Nxxunit=1000., NLprebuck=NLprebuck,
                    NLprebuck_eps1=0.0005, edges=edges)
        assert np.isclose(out['SS3-IR']['Pcr'], out['SS3']['Pcr'],
                          rtol=1e-9)
        b = [out[e]['koiter']['b_ijkl'][(0, 0, 0, 0)]
             for e in ('SS3', 'SS3-IR')]
        assert np.isclose(b[1], b[0], rtol=1e-6)
        #NOTE and the axial displacement has no mean, in the mass metric
        u0 = out['SS3-IR']['koiter']['u0'].reshape(-1, DOF)
        assert abs(u0[:, 0].mean()) <= 1e-10*np.abs(u0[:, 0]).max()


def test_free_IR_modes_are_not_rigid():
    """Free edges: the six rigid body modes removed, the lowest buckling
    modes the n = 2 ovalization of the free edges"""
    from bfsccylinder_models.edges import rigid_body_modes
    ny = 40
    nx, prop = _waters(ny)
    out = fkoiter_cyl_SS3(L, R, nx, ny, prop, num_eigvals=4,
            koiter_num_modes=0, Nxxunit=1000., edges='free-IR')
    modes = rigid_body_modes(out['x'], out['y'], R, DOF)
    for j in range(4):
        phi = out['eigvecs'][:, j]
        c = np.linalg.lstsq(modes, phi, rcond=None)[0]
        #NOTE to the tolerance of the eigen solver, 1e-6
        assert np.linalg.norm(modes @ c) <= 1e-6*np.linalg.norm(phi)
    W = out['eigvecs'][:, 0].reshape(-1, DOF)[:, 6].reshape(nx, ny)
    assert np.argmax(np.abs(np.fft.rfft(W, axis=1)).sum(axis=0)) == 2
    assert np.isclose(out['load_mult'][0], out['load_mult'][1], rtol=1e-6)


if __name__ == '__main__':
    test_SS3_space_is_the_selection()
    test_SS4_space()
    test_SS4_Waters_shell()
    test_rigid_body_modes()
    test_SS3_IR_is_SS3()
    test_free_IR_modes_are_not_rigid()
