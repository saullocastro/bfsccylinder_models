"""Crest of w of the Koiter modes: edge envelope, bicubic grid, and the
element's own shape functions (grid + local maximization)"""
import sys
import numpy as np
from scipy.optimize import minimize
from bfsccylinder.sanders import BFSCCylinderSanders
import run_case as rc

icase, prebuck, ny = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rc.use_safe_solvers(); rc.use_distinct_modes(); rc.use_koiter_denominators()
v1, v2, v3, v4, v5 = np.loadtxt('DOE09.txt', skiprows=1)[icase]
out = rc.design_function(dict(rCTS=v1, param_n=int(v2), c2_ratio=v3, thetadeg_c1=v4, thetadeg_c2=v5),
    dict(L=1.2, R=0.4, ny=ny, E11=122e9, E22=7.32e9, nu12=0.31, G12=4.9e9, tow_thick=0.13e-3, rho=1540,
         mesh_only=False, Nxxunit=1000., NLprebuck=prebuck == 'NL'))
DOF = 10
pos = out['nid_pos']
c = np.array([[DOF*pos[n] for n in ns] for ns in zip(out['n1s'], out['n2s'], out['n3s'], out['n4s'])])
x = out['x']
lex = x[c[:, 1]//DOF] - x[c[:, 0]//DOF]
ley = 2*np.pi*0.4/ny
elem = BFSCCylinderSanders(4); elem.R = 0.4; elem.ley = ley
elem.c1, elem.c2, elem.c3, elem.c4 = 0, DOF, 2*DOF, 3*DOF
def Sw(le, xi, eta):
    elem.lex = le; elem.update_Sw(xi, eta); return np.atleast_2d(elem.Sw)[0].copy()
idx = (c[:, :, None] + np.arange(DOF)[None, None, :]).reshape(len(c), -1)

# 1) the bicubic reconstruction of field_crest against Sw, random dofs, interior points
rng = np.random.default_rng(0)
q = rng.standard_normal(40); le = lex[0]
H = lambda t: np.array([2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t, -2*t**3 + 3*t**2, t**3 - t**2])
err = 0.
for xi, eta in rng.uniform(-1, 1, (20, 2)):
    t, s = (xi + 1)/2, (eta + 1)/2
    Hx, Hy = H(t), H(s)
    w = 0.
    for node, (a, b) in enumerate([(0, 0), (2, 0), (2, 2), (0, 2)]):
        Q = q[node*DOF + 6: node*DOF + 10]
        w += Q[0]*Hx[a]*Hy[b] + le*Q[1]*Hx[a+1]*Hy[b] + ley*Q[2]*Hx[a]*Hy[b+1] + le*ley*Q[3]*Hx[a+1]*Hy[b+1]
    err = max(err, abs(w - Sw(le, xi, eta) @ q))
print('bicubic reconstruction vs element Sw, max abs diff %.2e (|q|~1)' % err)

def element_crest(u, npts=9):
    Q = u[idx]
    best = (0., None, None)
    g = np.linspace(-1, 1, npts)
    for le in np.unique(np.round(lex, 12)):
        sel = np.where(np.isclose(lex, le, rtol=0, atol=1e-12))[0]
        S = np.array([Sw(le, xi, eta) for xi in g for eta in g])
        W = np.abs(Q[sel] @ S.T)
        k = np.unravel_index(np.argmax(W), W.shape)
        if W[k] > best[0]:
            best = (W[k], sel[k[0]], (g[k[1]//npts], g[k[1] % npts]))
    grid = best[0]
    # refine in the best element and its neighbours in the grid ranking
    W_all = []
    e, (xi0, eta0) = best[1], best[2]
    f = lambda p: -abs(Sw(lex[e], np.clip(p[0], -1, 1), np.clip(p[1], -1, 1)) @ Q[e])
    r = minimize(f, [xi0, eta0], method='Nelder-Mead', options=dict(xatol=1e-10, fatol=1e-14))
    return grid, -r.fun, (e, r.x)

U0 = out['koiter']['ui'][0].reshape(-1, DOF)
nodal = np.sqrt(U0[:, 0]**2 + U0[:, 3]**2 + U0[:, 6]**2).max()
print('mode  n   nodal|w|/h  envelope  bicubic11  element_grid9  element_max   (all over the nodal translation h)')
for k in range(rc.koiter_num_modes):
    u = out['koiter']['ui'][k]
    env, _ = rc.mode_amplitudes(out, k)
    bic = rc.field_crest(out, u)
    g9, emax, (e, p) = element_crest(u)
    wn = np.abs(u[6::DOF]).max()/nodal
    print('%2d  %3d   %.5f     %.5f   %.5f    %.5f        %.5f  at xi,eta %+.3f %+.3f' % (k, rc.mode_harmonics(out, k)[0], wn, env, bic, g9/nodal, emax/nodal, *p))
