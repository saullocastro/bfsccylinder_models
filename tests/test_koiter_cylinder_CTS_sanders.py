import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np
import pytest
from composites import laminated_plate

from bfsccylinder_models.koiter_cylinder_CTS_sanders import fkoiter_cylinder_CTS_circum
from bfsccylinder_models.koiter_cylinder_sanders import fkoiter_cyl_SS3

#NOTE NLprebuck=True exercises the nonlinear pre-buckling algorithm: the
#     axisymmetric Newton-Raphson pre-buckling solve and the iterative
#     eigenvalue algorithm of Sun et al. that walks the expansion point up to
#     the bifurcation point. In the constant stiffness limit below the CTS
#     model discretizes exactly the same shell on exactly the same mesh as
#     fkoiter_cyl_SS3, so the two must return the same answer whichever
#     pre-buckling state the expansion is made about
@pytest.mark.parametrize('NLprebuck', [False, True])
def test_pm45(NLprebuck):
    L = 0.3 # m
    R = 0.136/2 # m

    ny = 30

    E11 = 90e9
    E22 = 7e9
    nu12 = 0.32
    G12 = 4.4e9
    tow_thick = 0.4e-3
    rho = 1611 # kg/m3

    rCTS = 0.2

    nxt = 2
    param_n = 0
    c2_ratio = 0
    thetadeg_c1 = 45
    thetadeg_c2 = 45

    Nxxunit = 1.
    out1 = fkoiter_cylinder_CTS_circum(L, R, rCTS, nxt, ny, E11, E22, nu12, G12,
            rho, tow_thick, param_n, c2_ratio, thetadeg_c1, thetadeg_c2,
            num_eigvals=5, koiter_num_modes=1, Nxxunit=Nxxunit, idealistic_CTS=True,
            NLprebuck=NLprebuck, zero_offset=False)
    print('cylinder_CTS eigvals', out1['eigvals'])
    print('cylinder_CTS koiter', out1['koiter'])

    nx = int(ny*L/(2*np.pi*R))
    if nx % 2 == 0:
        nx += 1
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    plyt = tow_thick
    stack = [45, -45]
    prop = laminated_plate(stack=stack, laminaprop=laminaprop, plyt=plyt,
            offset=plyt, rho=rho)
    out2 = fkoiter_cyl_SS3(L, R, nx, ny, prop, num_eigvals=5,
            koiter_num_modes=1, Nxxunit=Nxxunit, NLprebuck=NLprebuck)

    print('fkoiter_cyl_SS3 eigvals', out2['eigvals'])
    print('fkoiter_cyl_SS3 koiter', out2['koiter'])

    assert np.isclose(out1['volume'], out2['volume'])
    assert np.isclose(out1['mass'], out2['mass'])

    #NOTE only the first (critical) buckling eigenvalue is compared. The higher
    #     modes form near-degenerate clusters and ARPACK (eigsh) returns them in
    #     a run-dependent order/multiplicity, so an element-wise comparison of
    #     the whole spectrum is not reproducible.
    #
    #     With NLprebuck=True the eigenvalues belong to the pre-buckling state
    #     the load stepping stopped at, which is only pinned to within
    #     NLprebuck_eps1, so the buckling LOAD is what the two models must
    #     agree on, not the raw eigenvalue of the shifted problem
    if NLprebuck:
        assert np.isclose(out1['Pcr'], out2['Pcr'], rtol=0.01)
        assert abs(out1['mu'][0] - 1) <= 0.005
        assert abs(out2['mu'][0] - 1) <= 0.005
    else:
        assert np.isclose(out1['eigvals'][0], out2['eigvals'][0])
        assert np.isclose(out1['Pcr'], out2['Pcr'])

    #TODO I am unsure about the a factors
    #for k in out1['koiter']['a_ijk'].keys():
        #assert np.isclose(out1['koiter']['a_ijk'][k],
                          #out2['koiter']['a_ijk'][k],
                          #atol=1e-5)
    assert np.isclose(out1['koiter']['b_ijkl'][(0, 0, 0, 0)],
                      out2['koiter']['b_ijkl'][(0, 0, 0, 0)],
                      rtol=0.05)

if __name__ == '__main__':
    test_pm45(False)
    test_pm45(True)

