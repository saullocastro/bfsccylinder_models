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
    #NOTE ny=40 keeps this test at about 2 min. The ny=50 mesh takes several
    #     times longer and gives Ncr/Ncl = 0.347244 with the same n=10
    #     circumferential waves and b_1111 = -0.222844
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
    #     The expansion is made about a CONVERGED nonlinear pre-buckling
    #     state at lambda_b/lambda_c = 0.996, reached in seven load steps
    #     taking a single Newton-Raphson iteration each, with no step back
    #     off, against the 0.142 that solving at the reference load Nxxunit
    #     used to give. For reference, the ANILISA n-search with rigorous
    #     nonlinear pre-buckling and SS-3 gives, for the same shell, 0.337088
    #     at n=7 and an absolute minimum of 0.328594 at n=11 (Table 3 of
    #     Arbocz, Starnes and Nemeth, AIAA-2001-1392). This mesh buckles with
    #     n=10, and so does ny=50, which gives 0.347244, 0.3% away. The value
    #     below is a REGRESSION value for this mesh, not a converged one
    assert np.isclose(Ncr/Ncl, 0.346127, rtol=0.01)
    b_1111 = out['koiter']['b_ijkl'][(0, 0, 0, 0)]
    print('b_1111', b_1111)
    #NOTE regression value for this mesh, NOT a converged one, and not
    #     directly comparable to the b = -0.3772 of Table 2 (ANILISA -0.3761,
    #     DIANA -0.3743), which belongs to the n=11 edge buckling mode where
    #     this mesh buckles with n=10.
    #
    #     b_1111 is more mesh sensitive than the buckling load, ny=50 giving
    #     -0.222844 against the value below, 3.4% away, where the buckling
    #     loads of the two meshes are 0.3% apart. It is also sensitive to
    #     which member of the degenerate buckling pair the expansion is made
    #     about: rotating the critical mode by 30 degrees inside its own
    #     eigenspace, which leaves the pre-buckling state and the buckling
    #     load untouched to eleven digits, moves it to -0.192752, 16% away.
    #     The second order field of a mode with n circumferential waves
    #     carries the harmonic 2n, which this mesh samples with two nodes per
    #     wave, and that is what the residual dependence measures.
    #     canonical_modes fixes the choice to the mesh aligned member, which
    #     is what makes the value below reproducible: 9e-9 between a single
    #     threaded and a multi threaded run
    assert np.isclose(b_1111, -0.230556, rtol=0.05)


def test_Arbocz_Starnes_2002():
    #Arbocz, J., and Starnes, J. H., 2002, “On a High-Fidelity Hierarchical Approach to Buckling Load Calculations,” New Approaches to Structural Mechanics, Shells and Biological Structures, pp. 271–292.
    #
    #Values below are taken from the conference version of that work, Arbocz,
    #J., Starnes, J. H., and Nemeth, M. P., 2001, same title, paper
    #AIAA-2001-1392, whose Table 1 lists the geometric and material properties
    #of the NASA layered composite shell AW-CYL-1-1, laminate [+-45/0/90]s.
    L = 0.3556 # m, 14.0 in = 355.600 mm
    R = 0.20318603 # m, 7.99945 in = 203.18603 mm
    #NOTE ny=40 keeps this test at about 4 min. The ny=60 mesh converges in
    #     four load steps with no step back off and gives lambda_c = 0.330603
    #     with n=11 circumferential waves and b_1111 = -0.356457, which are
    #     the figures worth comparing with the literature below. That mesh
    #     was unusable until bfsccylinder 0.6.0: with the tangent stiffness
    #     matrix inconsistent with the internal force vector the
    #     Newton-Raphson contracted by about 0.95 per iteration there,
    #     exhausted NR_maxiter at every load step beyond lambda_b/lambda_c =
    #     0.92, and the load stepping saturated near 0.94
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
    #     Level-2 ANILISA and SS-3. It now CONVERGES at lambda_b/lambda_c =
    #     0.999 and returns the rigorous nonlinear pre-buckling branch
    #     instead, the edge restraint driving the critical mode to a high
    #     circumferential wave number and dropping the load by about 9%.
    #
    #     The Level-2 ANILISA n-search gives 0.329163 at n=10 and its absolute
    #     minimum 0.328594 at n=11, and Level-3 STAGS-A gives 0.327759 (n=11,
    #     161x201 mesh). This mesh buckles with n=10 and gives the 0.331413
    #     below, 0.7% above the ANILISA n=10 entry, and ny=60 buckles with
    #     n=11 and gives 0.330603, 0.6% above the ANILISA n=11 one. Both are
    #     still REGRESSION values for their mesh
    assert np.isclose(lambda_c, 0.331413, rtol=0.01)
    b_1111 = out['koiter']['b_ijkl'][(0, 0, 0, 0)]
    print('b_1111', b_1111)
    #NOTE regression value for this mesh, NOT a converged one, but of the
    #     right sign and the right order now. Level-2 ANILISA reports
    #     b = -0.37605 for the n=11 mode with rigorous nonlinear pre-buckling
    #     (alpha = 0.46663, beta = -0.22174) and Sun et al. Table 2 gives
    #     -0.3772. This mesh buckles with n=10 and gives the -0.335975 below;
    #     ny=60 buckles with n=11 and gives -0.356457, 5% from the ANILISA
    #     value for that mode.
    #
    #     A negative b is an imperfection sensitive shell, which is what this
    #     one is. It used to come out POSITIVE here, +0.410664, and the sign
    #     was put down to a single mode expansion about a degenerate critical
    #     mode being unable to determine it. That was wrong. The tangent
    #     stiffness matrix of bfsccylinder was not the derivative of the
    #     internal force vector, by 1.4e-3 in a directional Taylor test, and
    #     the sign of b went with it, as did 9% of lambda_c and the
    #     convergence of the Newton-Raphson on finer meshes. Fixed in
    #     bfsccylinder 0.6.0, which requirements.txt now asks for.
    #
    #     Which member of the degenerate pair the expansion is made about
    #     still matters, by 16% on the Sun et al. case of this file, but no
    #     longer for the sign. canonical_modes fixes that choice.
    #
    #     Imposing the stiffness weighted orthogonality of Sun et al. Eq. (33)
    #     on the second order field, which is what the code does, was measured
    #     against the Euclidean condition before the tangent was fixed and
    #     moved the second order field by 6.5e-7 relative, a difference the
    #     two conditions are near-coincident on because the null space
    #     component of uab they disagree on is small to begin with. b_ijkl
    #     sees uab through phi3_ab @ uab, whose contraction with the buckling
    #     modes is the numerator of a_ijk, and a_ijk vanishes for this
    #     symmetric bifurcation. Not re-measured since
    assert np.isclose(b_1111, -0.335975, rtol=0.05)

if __name__ == '__main__':
    test_Arbocz_Starnes_2002()

