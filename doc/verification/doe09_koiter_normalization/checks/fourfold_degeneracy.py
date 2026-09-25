import sys
import numpy as np
import run_case as rc
from bfsccylinder_models.cyclic_symmetry import degenerate_partner
icase, prebuck, ny = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rc.use_safe_solvers(); rc.use_distinct_modes()
cap = {}
inner = rc.model.canonical_modes
def spy(mu, v, bu, axi, DOF, deg_rtol=1.e-5):
    v = inner(mu, v, bu, axi, DOF, deg_rtol=deg_rtol)
    cap.update(mu=np.array(mu), v=v.copy(), bu=bu, axi=axi); return v
rc.model.canonical_modes = spy
rc.koiter_num_modes = 0
v1, v2, v3, v4, v5 = np.loadtxt('DOE09.txt', skiprows=1)[icase]
out = rc.design_function(dict(rCTS=v1, param_n=int(v2), c2_ratio=v3, thetadeg_c1=v4, thetadeg_c2=v5),
    dict(L=1.2, R=0.4, ny=ny, E11=122e9, E22=7.32e9, nu12=0.31, G12=4.9e9, tow_thick=0.13e-3, rho=1540,
         mesh_only=False, Nxxunit=1000., NLprebuck=prebuck == 'NL'))
mu, v, bu, axi = cap['mu'], cap['v'], cap['bu'], cap['axi']
def full(x): p = np.zeros(bu.shape[0]); p[bu] = x; return p
ns = [rc.mode_harmonics(out, k)[0] for k in range(len(mu))]
print('mu-1', ['%.1e' % (m/mu[0] - 1) for m in mu]); print('n', ns)
for grp in [(0, 1), (2, 3)]:
    V = []
    for k in grp:
        p = full(v[:, k]); V += [p/np.linalg.norm(p)]
        q = degenerate_partner(p, bu, axi, 10)
        if q is not None: V += [q/np.linalg.norm(q)]
    sv = np.linalg.svd(np.array(V).T, compute_uv=False)
    print('modes', grp, 'n', [ns[k] for k in grp], 'singular values of {m, P m}', np.round(sv, 4))
    # axial mirror symmetry of each mode: overlap with its mirror image x -> L - x
    for k in grp:
        W = full(v[:, k]).reshape(-1, 10)[axi][:, :, 6]
        F = np.fft.rfft(W, axis=1)[:, ns[k]]
        print('   mode', k, 'mirror overlap |<F, F(L-x)>|/|F|^2 = %.3f' % (abs(np.vdot(F, F[::-1]))/np.vdot(F, F).real),
              ' end amplitudes left/right %.3f %.3f' % (np.abs(F[:len(F)//3]).max(), np.abs(F[-len(F)//3:]).max()))
