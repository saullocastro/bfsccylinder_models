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
    #NOTE ny=40 keeps this test at about 2 min. The finer ny=50 used before
    #     takes several times longer and gives lambda_c = 0.3367 with n=7
    #     circumferential waves and b_1111 = -0.0420
    ny = 40
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
    #
    #     Since the iterative eigenvalue algorithm of Sun et al. Eqs. (44) to
    #     (46) was implemented, the expansion really is made about a nonlinear
    #     pre-buckling state, lambda_b/lambda_c = 0.985 instead of the 0.142
    #     that solving at the reference load Nxxunit used to give, and this
    #     model no longer returns the membrane pre-buckling load of 0.3581.
    #     For reference, the ANILISA n-search with rigorous nonlinear
    #     pre-buckling and SS-3 gives, for the same shell, 0.337088 at n=7 and
    #     an absolute minimum of 0.328594 at n=11 (Table 3 of Arbocz, Starnes
    #     and Nemeth, AIAA-2001-1392). The ny=50 mesh reproduces the n=7 entry
    #     to 0.1%, at 0.3367. This coarser mesh overshoots the knockdown, so
    #     the value below is a REGRESSION value for this mesh, not a
    #     converged one
    assert np.isclose(Ncr/Ncl, 0.311554, rtol=0.01)
    b_1111 = out['koiter']['b_ijkl'][(0, 0, 0, 0)]
    print('b_1111', b_1111)
    #NOTE regression value for this mesh, NOT a converged one, and not
    #     comparable to the b = -0.3772 of Table 2 (ANILISA -0.3761, DIANA
    #     -0.3743), which belongs to the n=11 edge buckling mode. b_1111 is
    #     far more mesh sensitive than the buckling load
    assert np.isclose(b_1111, -0.059885, rtol=0.05)


def test_Arbocz_Starnes_2002():
    #Arbocz, J., and Starnes, J. H., 2002, “On a High-Fidelity Hierarchical Approach to Buckling Load Calculations,” New Approaches to Structural Mechanics, Shells and Biological Structures, pp. 271–292.
    #
    #Values below are taken from the conference version of that work, Arbocz,
    #J., Starnes, J. H., and Nemeth, M. P., 2001, same title, paper
    #AIAA-2001-1392, whose Table 1 lists the geometric and material properties
    #of the NASA layered composite shell AW-CYL-1-1, laminate [+-45/0/90]s.
    L = 0.3556 # m, 14.0 in = 355.600 mm
    R = 0.20318603 # m, 7.99945 in = 203.18603 mm
    #NOTE ny=40 keeps this test at about 4 min. The finer ny=60 takes more
    #     than an hour with the iterative eigenvalue algorithm and gives
    #     lambda_c = 0.3359 with n=10 circumferential waves and
    #     b_1111 = -0.4310, which is the result worth quoting, see the NOTE on
    #     b_1111 below
    ny = 40

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
    #NOTE before the iterative eigenvalue algorithm of Sun et al. Eqs. (44) to
    #     (46) was implemented, the expansion was made about the state at the
    #     reference load, lambda_b/lambda_c = 0.07, and this model returned the
    #     MEMBRANE pre-buckling results of the paper, lambda_c = 0.365992
    #     (Level-1 AXBIF, m=1, n=7) and 0.364370 (n=11) or 0.364715 (n=7) with
    #     Level-2 ANILISA and SS-3. It now reaches lambda_b/lambda_c = 0.93 and
    #     returns the rigorous nonlinear pre-buckling branch instead, where the
    #     edge restraint drives the critical mode to a high circumferential
    #     wave number and drops the load by about 10%. The Level-2 ANILISA
    #     n-search gives 0.329163 at n=10 and its absolute minimum 0.328594 at
    #     n=11, and Level-3 STAGS-A gives 0.327759 (n=11, 161x201 mesh). The
    #     ny=60 mesh gives 0.3359 with n=10, within 2% of the ANILISA n=10
    #     entry, the rest being the lambda_b/lambda_c = 0.93 that the load
    #     controlled solver stops at, with lambda_c still decreasing. This
    #     coarser mesh gives 0.304019, so the value below is a REGRESSION
    #     value for this mesh and not a converged one
    assert np.isclose(lambda_c, 0.304019, rtol=0.01)
    b_1111 = out['koiter']['b_ijkl'][(0, 0, 0, 0)]
    print('b_1111', b_1111)
    #NOTE regression value for this mesh, NOT a converged one. b_1111 is far
    #     more mesh sensitive than the buckling load, so it is the ny=60 value
    #     of about -0.43 that is worth comparing with the literature. That one
    #     is of the same order as the b = -0.37605 that Level-2 ANILISA reports
    #     for the n=11 mode with rigorous nonlinear pre-buckling (alpha =
    #     0.46663, beta = -0.22174) and as the -0.3772 of Sun et al. Table 2,
    #     whereas before the pre-buckling state was fixed this test gave
    #     -0.0598, close to the Level-1 BFACT value of -0.048844 obtained with
    #     MEMBRANE pre-buckling on the m=1, n=7 mode. The difference that
    #     remains at ny=60 comes from the mode being n=10 instead of n=11,
    #     from lambda_b/lambda_c stopping at 0.93, and from the orthogonality
    #     condition of the second order field still being the Euclidean one
    #     instead of Eq. (33).
    #
    #     The ny=50 and ny=60 numbers quoted in these NOTEs were measured
    #     before the ARPACK starting vector was fixed in solve_eig, so they
    #     may shift a little once re-measured, the coarse mesh values here
    #     moved by a few percent
    assert np.isclose(b_1111, -1.576047, rtol=0.05)

if __name__ == '__main__':
    test_Arbocz_Starnes_2002()

