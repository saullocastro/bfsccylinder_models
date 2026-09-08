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
    #Thin-Walled Struct., 155, p. 106962. DOI: https://doi.org/10.1016/j.tws.2020.106962
    #
    #Composite cylindrical shell under axial compression of Section 3.1 and
    #Fig. 3 of that paper. Geometry, laminate and material properties are taken
    #from the text of Section 3.1, and the classical simply-supported boundary
    #condition SS-3 is used, as in the paper.
    ny = 50
    R = 0.2032 # m, R = 203.2 mm
    L = 0.3556 # m, L = 355.6 mm
    nx = int(ny*L/(2*np.pi*R))
    if (nx % 2) == 0:
        nx += 1
    print('nx, ny', nx, ny)
    E11 = 127.629e9 # Pa, E1 = 127.629 GPa
    E22 = 11.3074e9 # Pa, E2 = 11.3074 GPa
    G12 = 6.00257e9 # Pa, G12 = 6.00257 GPa
    nu12 = 0.300235
    #NOTE G13 and G23 are not given in the paper, assumed equal to G12
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    stack = (+45, -45, 0, 90, 90, 0, -45, +45) # [+-45/0/90]s
    h = 0.00101539 # m, total thickness t = 1.01539 mm
    plyt = h/len(stack)
    Nxxunit = 20000. # N/m
    prop = laminated_plate(stack=stack, laminaprop=laminaprop, plyt=plyt)
    out = fkoiter_cyl_SS3_sanders(L, R, nx, ny, prop, cg_x0=None, nint=4,
            num_eigvals=4, koiter_num_modes=1, Nxxunit=Nxxunit, NLprebuck=True)
    print(out['eigvals'])
    #NOTE reference buckling load of Eq. (47), a membrane stress resultant, such
    #     that the total axial force Pcr must be divided by the circumference
    Ncl = E11*h**2/(R*np.sqrt(3*(1-nu12**2))) # N/m
    Ncr = out['Pcr']/(2*np.pi*R) # N/m
    print('normalized buckling load', Ncr/Ncl)
    #NOTE the paper reports a classical (linear eigenvalue) buckling load of
    #     175.5 kN and a buckling load of 164.3 kN accounting for the nonlinear
    #     pre-buckling state, respectively 0.3508 and 0.3284 once normalized by
    #     Ncl. ANILISA and DIANA give 0.3286 and 0.3244 (Section 3.1).
    assert np.isclose(Ncr/Ncl, 0.3508, rtol=0.03)
    b_1111 = out['koiter']['b_ijkl'][(0, 0, 0, 0)]
    print('b_1111', b_1111)
    #NOTE this case runs the SANDERS model with NLprebuck=True. The reference
    #     was updated after fixing eps''_ab, eps_dot, eps_dot_dot and
    #     eps_dot'_a, which were falling back to von Karman kinematics (Eqs.
    #     40-43 of the SciTech 2022 paper), and again after fixing the stacking
    #     sequence, where a missing comma was merging plies 3 and 4 into a
    #     single -45 deg ply. Regression value; the paper reports b = -0.3772
    #     (ANILISA -0.3761, DIANA -0.3743, Table 2), which the present model
    #     does not reproduce.
    assert np.isclose(b_1111, -0.051636896760166424, rtol=0.02)


def test_Arbocz_Starnes_2002():
    #Arbocz, J., and Starnes, J. H., 2002, “On a High-Fidelity Hierarchical Approach to Buckling Load Calculations,” New Approaches to Structural Mechanics, Shells and Biological Structures, pp. 271–292.
    #
    #Values below are taken from the conference version of that work, Arbocz,
    #J., Starnes, J. H., and Nemeth, M. P., 2001, same title, paper
    #AIAA-2001-1392, whose Table 1 lists the geometric and material properties
    #of the NASA layered composite shell AW-CYL-1-1, laminate [+-45/0/90]s.
    L = 0.3556 # m, 14.0 in = 355.600 mm
    R = 0.20318603 # m, 7.99945 in = 203.18603 mm
    #NOTE ny=40 of the original version of this test was far from mesh
    #     converged: the buckling load converges from below and only settles
    #     at ny>=60, where it lands on the membrane pre-buckling result of the
    #     paper. See the NOTE on b_1111 below for the sensitivity of the
    #     post-buckling coefficient
    ny = 60

    nx = int(1.5*ny*L/(2*np.pi*R))
    if (nx % 2) == 0:
        nx += 1
    print('nx, ny', nx, ny)

    E11 = 127.629e9 # Pa, 18.5111e6 psi = 12.7629e4 N/mm2
    E22 = 11.3074e9 # Pa, 1.64e6 psi = 1.13074e4 N/mm2
    G12 = 6.00257e9 # Pa, 0.8706e6 psi = 6.00257e3 N/mm2
    nu12 = 0.300235
    stack = [45, -45, 0, 90, 90, 0, -45, 45] # [+-45/0/90]s
    h = 0.00101539 # m, total thickness 0.039976 in = 1.01539 mm
    #NOTE 8 plies of equal thickness (0.004997 in)
    plyt =  h/len(stack)
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    prop = laminated_plate(stack=stack, laminaprop=laminaprop, plyt=plyt)
    Nxxunit = 10000. # N/m
    out = fkoiter_cyl_SS3(L, R, nx, ny, prop, cg_x0=None, nint=4,
            num_eigvals=2, koiter_num_modes=1, Nxxunit=Nxxunit, NLprebuck=True)
    #NOTE normalizing stress resultant Ncl = E h**2/(c R), with
    #     c = sqrt(3(1 - nu12**2)), equal to the -2238.325 lb/in used to
    #     normalize every buckling load reported in the paper
    ref = E11*h**2/(R*np.sqrt(3*(1-nu12**2))) # N/m
    #NOTE out['eigvals'] are the raw eigenvalues of the shifted problem, the
    #     load multiplier is out['load_mult'] = -1/eigvals
    lambda_c = out['load_mult'][0]*Nxxunit/ref
    print('lambda_c', lambda_c)
    #NOTE this model reproduces the MEMBRANE pre-buckling results of the paper,
    #     lambda_c = 0.365992 (Level-1 AXBIF, m=1, n=7) and 0.364370 (n=11) or
    #     0.364715 (n=7) with Level-2 ANILISA and SS-3, and not the ones with
    #     rigorous nonlinear pre-buckling, lambda_c = 0.328594 (Level-2
    #     ANILISA, n=11) and 0.327759 (Level-3 STAGS-A, 161x201 mesh, n=11),
    #     which are about 11% lower because of the edge restraint effect
    assert np.isclose(lambda_c, 0.365992, rtol=0.01)
    b_1111 = out['koiter']['b_ijkl'][(0, 0, 0, 0)]
    print('b_1111', b_1111)
    #NOTE regression value. Consistently with lambda_c above, the comparable
    #     reference is the Level-1 BFACT result of the paper, b = -0.048844,
    #     obtained with membrane pre-buckling and an imperfection affine to the
    #     m=1, n=7 buckling mode; the Level-2 ANILISA value with nonlinear
    #     pre-buckling is b = -0.37605 (alpha = 0.46663, beta = -0.22174).
    #     b_1111 converges much more slowly than lambda_c, ny=60, 64 and 70
    #     give -0.0598, -0.0586 and -0.0593, with an outlier at ny=56, hence
    #     the loose tolerance
    assert np.isclose(b_1111, -0.0597, rtol=0.05)

if __name__ == '__main__':
    test_Arbocz_Starnes_2002()

