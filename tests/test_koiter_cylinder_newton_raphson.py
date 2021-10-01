import sys
sys.path.append(r'..')
#sys.path.append(r'../../bfsccylinder')

import numpy as np
from composites import isotropic_plate, laminated_plate

from bfsccylinder_models.koiter_cylinder_newton_raphson import fkoiter_cyl_SS3

def test_Sun_et_al():
    #Sun, Y., Tian, K., Li, R., and Wang, B., 2020, “Accelerated Koiter Method for Post-Buckling Analysis of Thin-Walled Shells under Axial Compression,” Thin-Walled Struct., 155, p. 106962.
    L = 0.51 # m
    R = 0.25 # m
    ny = 40

    nx = int(ny*L/(2*np.pi*R))
    if (nx % 2) == 0:
        nx += 1
    print('nx, ny', nx, ny)

    E = 72e9
    nu = 0.31
    thickness = 0.0005
    prop = isotropic_plate(E=E, nu=nu, thickness=thickness)

    out = fkoiter_cyl_SS3(L, R, nx, ny, prop, cg_x0=None, lobpcg_X=None, nint=4,
        num_eigvals=2, koiter_num_modes=1)
    print(out['eigvals'])


    print()
    R = 0.2032
    L = 0.3556
    nx = int(ny*L/(2*np.pi*R))
    if (nx % 2) == 0:
        nx += 1
    print('nx, ny', nx, ny)
    laminaprop = (127.629e9, 11.3074e9, 0.300235, 6.00257e9, 6.00257e9, 6.00257e9)
    stack = (+45, -45, 0, 90, 90, 0 -45, +45)
    plyt = 0.00101539/len(stack)
    prop = laminated_plate(stack=stack, laminaprop=laminaprop, plyt=plyt)
    out = fkoiter_cyl_SS3(L, R, nx, ny, prop, cg_x0=None,
            lobpcg_X=None, nint=4, num_eigvals=2,
            koiter_num_modes=1)
    print(out['eigvals'])

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
    load = 50000. # N
    #load = 183366/1.5 # N
    out = fkoiter_cyl_SS3(L, R, nx, ny, prop, cg_x0=None,
            lobpcg_X=None, nint=4, num_eigvals=2,
            koiter_num_modes=1, load=load, NLprebuck=True)
    ref = 391.990772951375*1000 # N/m
    ans = out['eigvals']*load/2/np.pi/R/ref
    print(ans)
    print(out['koiter']['b_ijkl'])

if __name__ == '__main__':
    test_Arbocz_Starnes_2002()

