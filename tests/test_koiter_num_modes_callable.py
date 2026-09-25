import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np
import pytest
from composites import laminated_plate

from bfsccylinder_models.koiter_cylinder_CTS_sanders import fkoiter_cylinder_CTS_circum
from bfsccylinder_models.koiter_cylinder_sanders import fkoiter_cyl_SS3

#NOTE koiter_num_modes given as a callable of the multipliers and the
#     eigenvectors, called after the last eigenvalue analysis, must give the
#     same expansion as the number it returns given directly, and receive
#     the arrays the models return in out['mu'] and out['eigvecs']
L = 0.3 # m
R = 0.136/2 # m
ny = 30
E11 = 90e9
E22 = 7e9
nu12 = 0.32
G12 = 4.4e9
tow_thick = 0.4e-3
rho = 1611 # kg/m3


def cts(koiter_num_modes, NLprebuck):
    return fkoiter_cylinder_CTS_circum(L, R, 0.2, 2, ny, E11, E22, nu12, G12,
            rho, tow_thick, 0, 0, 45, 45, num_eigvals=5,
            koiter_num_modes=koiter_num_modes, Nxxunit=1., idealistic_CTS=True,
            NLprebuck=NLprebuck, zero_offset=False)


def ss3(koiter_num_modes, NLprebuck):
    nx = int(ny*L/(2*np.pi*R))
    if nx % 2 == 0:
        nx += 1
    prop = laminated_plate(stack=[45, -45],
            laminaprop=(E11, E22, nu12, G12, G12, G12), plyt=tow_thick,
            offset=tow_thick, rho=rho)
    return fkoiter_cyl_SS3(L, R, nx, ny, prop, num_eigvals=5,
            koiter_num_modes=koiter_num_modes, Nxxunit=1., NLprebuck=NLprebuck)


@pytest.mark.parametrize('model', [cts, ss3])
@pytest.mark.parametrize('NLprebuck', [False, True])
def test_callable_matches_number(model, NLprebuck):
    seen = {}

    def choose(mu, eigvecs):
        seen['mu'] = np.array(mu)
        seen['eigvecs'] = np.array(eigvecs)
        return 2

    out_c = model(choose, NLprebuck)
    out_n = model(2, NLprebuck)
    assert out_c['koiter_num_modes'] == out_n['koiter_num_modes'] == 2
    assert np.allclose(seen['mu'], out_c['mu'], rtol=0, atol=0)
    assert np.allclose(seen['eigvecs'], out_c['eigvecs'], rtol=0, atol=0)
    b_c = out_c['koiter']['b_ijkl']
    b_n = out_n['koiter']['b_ijkl']
    assert sorted(b_c) == sorted(b_n)
    scale = max(abs(v) for v in b_n.values())
    for idx in b_n:
        assert abs(b_c[idx] - b_n[idx]) <= 1.e-8*scale


@pytest.mark.parametrize('model', [cts, ss3])
def test_callable_zero_and_out_of_range(model):
    out = model(lambda mu, eigvecs: 0, False)
    assert out['koiter'] is None
    assert out['koiter_num_modes'] == 0
    with pytest.raises(ValueError):
        model(lambda mu, eigvecs: 6, False)
