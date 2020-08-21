import sys
sys.path.append(r'..')
from bfsccylinder_models.models import linearBucklingVATCylinder_x

def test_models():
    L = 0.3 # m
    R = 0.136/2 # m
    nx = 20 # axial
    E11 = 90e9
    E22 = 7e9
    nu12 = 0.32
    G12 = 4.4e9
    tow_thick = 0.4e-3
    theta_VP_1 = 45.4
    theta_VP_2 = 86.5
    theta_VP_3 = 85.8
    desvars = [[theta_VP_1, theta_VP_2, theta_VP_3]]
    Pcr, out = linearBucklingVATCylinder_x(L, R, nx, E11, E22, nu12, G12, tow_thick,
            desvars, clamped=True)
    theta_VP_1 = 55.4
    theta_VP_2 = 76.5
    theta_VP_3 = 75.8
    desvars = [[theta_VP_1, theta_VP_2, theta_VP_3]]
    Pcr, out = linearBucklingVATCylinder_x(L, R, nx, E11, E22, nu12, G12, tow_thick,
            desvars, clamped=True, cg_x0=out['cg_x0'], lobpcg_X=out['lobpcg_X'])

if __name__ == '__main__':
    test_models()

