import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np
from composites import isotropic_plate, laminated_plate

from bfsccylinder_models.koiter_cylinder import fkoiter_cyl_SS3
from bfsccylinder_models.koiter_cylinder_sanders import fkoiter_cyl_SS3 as fkoiter_cyl_SS3_sanders

def test_Sun_et_al():
    #Sun, Y., Tian, K., Li, R., and Wang, B., 2020, “Accelerated Koiter Method
    #for Post-Buckling Analysis of Thin-Walled Shells under Axial Compression,”
    #Thin-Walled Struct., 155, p. 106962. DOI: https://doi.org/10.1016/j.tws.2020.106962
    #
    #Composite cylindrical shell under axial compression of Section 3.1 and
    #Fig. 3 of that paper. Geometry, laminate and material properties are taken
    #from the text of Section 3.1, and the classical simply-supported boundary
    #condition SS-3 is used, as in the paper.
    #NOTE ny=50, about 70 s. With v and w fixed along the whole edge, ny=40
    #     (nx = 11) buckles with n=7, Ncr/Ncl = 0.354920 and b_1111 =
    #     -0.052152; with v and w fixed at the edge nodes only it buckled with
    #     n=10, 0.346127 and -0.230556, and ny=50 gave 0.347283 and -0.222760
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
    #NOTE the paper reports 175.5 kN from a linear eigenvalue analysis and
    #     164.3 kN with the nonlinear pre-buckling state, 0.3508 and 0.3284
    #     normalized by Ncl, and ANILISA and DIANA 0.3286 and 0.3244. The
    #     ANILISA n-search with rigorous nonlinear pre-buckling and SS-3 gives
    #     0.337088 at n=7 and its minimum 0.328594 at n=11 (Table 3 of Arbocz,
    #     Starnes and Nemeth, AIAA-2001-1392). This mesh buckles with n=10.
    #     REGRESSION value for this mesh, not a converged one
    assert np.isclose(Ncr/Ncl, 0.348965, rtol=0.01)
    b_1111 = out['koiter']['b_ijkl'][(0, 0, 0, 0)]
    print('b_1111', b_1111)
    #NOTE REGRESSION value for this mesh, not a converged one, and not
    #     comparable to the b = -0.3772 of Table 2 (ANILISA -0.3761, DIANA
    #     -0.3743), which belongs to the n=11 mode. The values of the rest of
    #     this note are those of ny=40 with v and w fixed at the edge nodes
    #     only, b_1111 = -0.230556. Rotating the critical mode by 30 degrees
    #     inside its pair
    #     moves it to -0.192752, almost all of it through the normalization by
    #     the largest nodal translation, which misses the crest of this skewed
    #     mode by 22%: normalized by the crest the two members give -0.15456
    #     and -0.15022. canonical_modes fixes the member, which makes the value
    #     reproducible, 2e-8 between a single and a multi threaded run. See
    #     "Buckling modes of a cylinder" in doc/nlprebuck_implementation.tex
    assert np.isclose(b_1111, -0.234711, rtol=0.05)


def test_Arbocz_Starnes_2002():
    #Arbocz, J., and Starnes, J. H., 2002, “On a High-Fidelity Hierarchical Approach to Buckling Load Calculations,” New Approaches to Structural Mechanics, Shells and Biological Structures, pp. 271–292.
    #
    #Values below are taken from the conference version of that work, Arbocz,
    #J., Starnes, J. H., and Nemeth, M. P., 2001, same title, paper
    #AIAA-2001-1392, whose Table 1 lists the geometric and material properties
    #of the NASA layered composite shell AW-CYL-1-1, laminate [+-45/0/90]s.
    L = 0.3556 # m, 14.0 in = 355.600 mm
    R = 0.20318603 # m, 7.99945 in = 203.18603 mm
    #NOTE ny=40 keeps this test at about 40 s. The ny=60 mesh, which needs
    #     bfsccylinder >= 0.6.0 to converge, gives lambda_c = 0.331066 with
    #     n=11 circumferential waves and b_1111 = -0.358347, the figures worth
    #     comparing with the literature below (0.330603 and -0.356457 with v
    #     and w fixed at the edge nodes only, and 0.331413 and -0.335975 at
    #     ny=40, against 0.336760 and -0.423807 now)
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
    #     load multiplier is out['load_mult'] = -lambda_b/eigvals
    lambda_c = out['load_mult'][0]*Nxxunit/ref
    print('lambda_c', lambda_c)
    #NOTE expanded about the state at the reference load instead,
    #     lambda_b/lambda_c = 0.07, this model returns the MEMBRANE
    #     pre-buckling result of the paper, lambda_c = 0.365992 (Level-1
    #     AXBIF, n=7). The Level-2 ANILISA n-search with rigorous nonlinear
    #     pre-buckling gives 0.329163 at n=10 and its minimum 0.328594 at n=11,
    #     and Level-3 STAGS-A 0.327759 (n=11, 161x201 mesh). This mesh buckles
    #     with n=10, 2.3% above the ANILISA n=10 entry; ny=60 buckles with n=11,
    #     0.8% above the n=11 one. REGRESSION values for their mesh
    assert np.isclose(lambda_c, 0.336760, rtol=0.01)
    b_1111 = out['koiter']['b_ijkl'][(0, 0, 0, 0)]
    print('b_1111', b_1111)
    #NOTE REGRESSION value for this mesh, not a converged one. Level-2
    #     ANILISA gives b = -0.37605 for the n=11 mode with rigorous nonlinear
    #     pre-buckling (alpha = 0.46663, beta = -0.22174), and Sun et al.
    #     Table 2 -0.3772. This mesh buckles with n=10; ny=60 buckles with n=11
    #     and gives -0.358347, 4.7% short of ANILISA. That gap is not the
    #     single-mode truncation, the reference coefficients being single-mode
    #     ones too; "The gap to ANILISA" in doc/nlprebuck_implementation.tex
    #     takes it apart, leaving about 4% of b unexplained by the
    #     normalization, the mesh and the expansion point.
    #
    #     A negative b is an imperfection sensitive shell, which this one is.
    #     b came out positive, +0.410664, until bfsccylinder 0.6.0 made the
    #     tangent stiffness matrix consistent with the internal force vector
    assert np.isclose(b_1111, -0.423807, rtol=0.05)

if __name__ == '__main__':
    test_Arbocz_Starnes_2002()

