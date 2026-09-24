import sys, time
import numpy as np
from scipy.optimize import minimize
import run_case as rc
icase, prebuck, ny = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rc.use_safe_solvers(); rc.use_distinct_modes(); rc.use_koiter_denominators()
v1, v2, v3, v4, v5 = np.loadtxt('DOE09.txt', skiprows=1)[icase]
out = rc.design_function(dict(rCTS=v1, param_n=int(v2), c2_ratio=v3, thetadeg_c1=v4, thetadeg_c2=v5),
    dict(L=1.2, R=0.4, ny=ny, E11=122e9, E22=7.32e9, nu12=0.31, G12=4.9e9, tow_thick=0.13e-3, rho=1540,
         mesh_only=False, Nxxunit=1000., NLprebuck=prebuck == 'NL'))
t0 = time.time(); F = rc.ElementField(out); t_init = time.time() - t0
ref = rc.ElementField(out, num_grid=41, num_refine=200)
print('elements', len(F.lex), 'distinct lex', len(F.groups), 'init %.1f s' % t_init)
# old edge-based RMS for comparison: Hermite along x at node columns, mean over y nodes
from bfsccylinder_models.cyclic_symmetry import mesh_order
order = mesh_order(out['x'], out['y'], out['nx'], out['ny'])
xs = out['x'][order[:, 0]]
tg, wg = np.polynomial.legendre.leggauss(4); tg, wg = (tg[:, None] + 1)/2, wg/2
def rms_old(u):
    U = u.reshape(-1, 10)[order]; I = 0.
    for i in range(out['nx'] - 1):
        dx = xs[i + 1] - xs[i]; t = tg
        h = (2*t**3 - 3*t**2 + 1)*U[i, :, 6] + (t**3 - 2*t**2 + t)*dx*U[i, :, 7] + (-2*t**3 + 3*t**2)*U[i + 1, :, 6] + (t**3 - t**2)*dx*U[i + 1, :, 7]
        I += dx*(wg @ (h**2).mean(axis=1))
    return np.sqrt(I/(xs[-1] - xs[0]))/F.nodal
print('mode  crest(9x9+20)  crest ref(41x41+200)  rel diff   rms elem   rms edge-mean   t_crest s')
for k in range(rc.koiter_num_modes):
    u = out['koiter']['ui'][k]
    t0 = time.time(); c = F.crest(u); tc = time.time() - t0
    cr = ref.crest(u)
    print('%2d    %.6f       %.6f            %.1e    %.6f   %.6f     %.2f' % (k, c, cr, abs(c/cr - 1), F.rms(u), rms_old(u), tc))
