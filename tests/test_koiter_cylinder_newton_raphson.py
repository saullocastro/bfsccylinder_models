import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np
from composites import isotropic_plate, laminated_plate

from bfsccylinder_models.koiter_cylinder_newton_raphson import fkoiter_cyl_SS3
from bfsccylinder_models.koiter_cylinder_newton_raphson_sanders import fkoiter_cyl_SS3 as fkoiter_cyl_SS3_sanders

def test_Sun_et_al():
    #Sun, Y., Tian, K., Li, R., and Wang, B., 2020, “Accelerated Koiter Method
    #for Post-Buckling Analysis of Thin-Walled Shells under Axial Compression,”
    #Thin-Walled Struct., 155, p. 106962.
    #Fig. 3 of that paper
    ny = 50
    R = 0.2032
    L = 0.3556
    nx = int(ny*L/(2*np.pi*R))
    if (nx % 2) == 0:
        nx += 1
    print('nx, ny', nx, ny)
    E11 = 127.629e9
    nu12 = 0.300235
    laminaprop = (E11, 11.3074e9, nu12, 6.00257e9, 6.00257e9, 6.00257e9)
    stack = (+45, -45, 0, 90, 90, 0 -45, +45)
    h = 0.00101539
    plyt = h/len(stack)
    Nxxunit = 20000.
    prop = laminated_plate(stack=stack, laminaprop=laminaprop, plyt=plyt)
    out = fkoiter_cyl_SS3_sanders(L, R, nx, ny, prop, cg_x0=None, nint=4,
            num_eigvals=4, koiter_num_modes=1, Nxxunit=Nxxunit, NLprebuck=True)
    print(out['eigvals'])
    Ncl = E11*h**2/(R*np.sqrt(3*(1-nu12**2)))
    print('normalized buckling load', out['Pcr']/Ncl)
    b_1111 = out['koiter']['b_ijkl'][(0, 0, 0, 0)]
    print('b_1111', b_1111)
    assert np.isclose(b_1111, -0.04192599251428145, rtol=0.02)


def test_Arbocz_Starnes_2002():
    #Arbocz, J., and Starnes, J. H., 2002, “On a High-Fidelity Hierarchical Approach to Buckling Load Calculations,” New Approaches to Structural Mechanics, Shells and Biological Structures, pp. 271–292.
    L = 0.3556 # m
    R = 0.20318603 # m
    ny = 40

    nx = int(1.5*ny*L/(2*np.pi*R))
    if (nx % 2) == 0:
        nx += 1
    print('nx, ny', nx, ny)

    E11 = 127.629e9
    E22 = 11.3074e9
    G12 = 6.00257e9
    nu12 = 0.3002
    stack = [45, -45, 0, 90, 90, 0, -45, 45]
    plyt =  0.00101539/len(stack)
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    prop = laminated_plate(stack=stack, laminaprop=laminaprop, plyt=plyt)
    Nxxunit = 10000. # N
    out = fkoiter_cyl_SS3(L, R, nx, ny, prop, cg_x0=None, nint=4,
            num_eigvals=2, koiter_num_modes=1, Nxxunit=Nxxunit, NLprebuck=True)
    ref = 391.990772951375*1000 # N/m
    ans = out['eigvals']*Nxxunit/ref
    print(ans)
    b_1111 = out['koiter']['b_ijkl'][(0, 0, 0, 0)]
    print('b_1111', b_1111)
    assert np.isclose(b_1111, -0.8046810677267613, rtol=0.02)

if __name__ == '__main__':
    test_Arbocz_Starnes_2002()

