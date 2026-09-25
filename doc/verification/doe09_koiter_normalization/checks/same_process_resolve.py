"""Same model twice in one process, modes 0 and 1 rotated by 30 deg in the
second; compare mode 2, u_22, d and b_ijkl"""
import sys
import numpy as np
import run_case as rc
icase, prebuck, ny = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rc.use_safe_solvers()
nd = rc.use_distinct_modes()
ld = rc.use_koiter_denominators()
inner = rc.model.canonical_modes
angle = [0.]
def outer(*a, **k):
    v = inner(*a, **k)
    t = np.deg2rad(angle[0]); v0, v1 = v[:, 0].copy(), v[:, 1].copy()
    v[:, 0], v[:, 1] = np.cos(t)*v0 + np.sin(t)*v1, -np.sin(t)*v0 + np.cos(t)*v1
    return v
rc.model.canonical_modes = outer
v1, v2, v3, v4, v5 = np.loadtxt('DOE09.txt', skiprows=1)[icase]
var = dict(rCTS=v1, param_n=int(v2), c2_ratio=v3, thetadeg_c1=v4, thetadeg_c2=v5)
con = dict(L=1.2, R=0.4, ny=ny, E11=122e9, E22=7.32e9, nu12=0.31, G12=4.9e9,
    tow_thick=0.13e-3, rho=1540, mesh_only=False, Nxxunit=1000., NLprebuck=prebuck == 'NL')
res = []
for ang in [0., 30.]:
    angle[0] = ang
    out = rc.design_function(var, con)
    K = out['koiter']
    B = np.array([[[[K['b_ijkl'][(i, j, k, l)] for l in range(5)] for k in range(5)] for j in range(5)] for i in range(5)])
    res.append(dict(mu=out['mu'][:5].copy(), u2=K['ui'][2].copy(), u0=K['ui'][0].copy(), u1=K['ui'][1].copy(),
                    u22=K['uij'][(2, 2)].copy(), ld=np.array(ld[0]), B=B, lam=np.array([K['lambda_i'][i] for i in range(5)]),
                    u0dot=K['u0dot'].copy(), u0c=K['u0'].copy()))
a, b = res
rel = lambda x, y: np.linalg.norm(x - y)/np.linalg.norm(x)
print('mu', a['mu']/a['mu'][0], b['mu']/b['mu'][0])
print('u0 prebuck rel diff %.2e, u0dot %.2e' % (rel(a['u0c'], b['u0c']), rel(a['u0dot'], b['u0dot'])))
print('mode 2 rel diff %.2e' % min(rel(a['u2'], b['u2']), rel(a['u2'], -b['u2'])))
print('u_22 rel diff %.2e' % rel(a['u22'], b['u22']))
print('lambda_d', a['ld'], b['ld'])
print('b_2222 %.6f %.6f' % (a['B'][2,2,2,2], b['B'][2,2,2,2]))
# expected mixed modes: u0' = c u0 + s u1 up to scale; check
t = np.deg2rad(30.)
for name, e in [('u0', np.cos(t)*a['u0']/np.abs(a['u0']).max() + np.sin(t)*a['u1']/np.abs(a['u1']).max())]:
    pass
