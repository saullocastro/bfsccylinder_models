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


if __name__ == '__main__':
    test_SS3_space_is_the_selection()
    test_SS4_space()
    test_SS4_Waters_shell()
