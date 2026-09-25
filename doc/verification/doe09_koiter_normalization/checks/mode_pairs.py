import sys, numpy as np
icase, prebuck, ny = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
import run_case as rc
m = rc.model
from bfsccylinder_models.cyclic_symmetry import degenerate_partner
cap = {}
orig = m.canonical_modes
def spy(mu, eigvecsu, bu, axi_order, DOF, deg_rtol=1.e-5):
    cap.update(mu=np.array(mu), raw=np.array(eigvecsu), bu=bu, axi=axi_order)
    return orig(mu, eigvecsu, bu, axi_order, DOF, deg_rtol=deg_rtol)
m.canonical_modes = spy
rc.use_safe_solvers()
nd = rc.use_distinct_modes()
v1, v2, v3, v4, v5 = np.loadtxt('DOE09.txt', skiprows=1)[icase]
variables = dict(rCTS=v1, param_n=int(v2), c2_ratio=v3, thetadeg_c1=v4, thetadeg_c2=v5)
constants = dict(L=1.2, R=0.4, ny=ny, E11=122e9, E22=7.32e9, nu12=0.31, G12=4.9e9,
    tow_thick=0.13e-3, rho=1540, mesh_only=False, Nxxunit=1000., NLprebuck=prebuck == 'NL')
rc.koiter_num_modes = 0  # modes only
out = rc.design_function(variables, constants)
mu, raw, bu, axi = cap['mu'], cap['raw'], cap['bu'], cap['axi']
DOF = 10; nx, nyy = axi.shape
def full(v):
    p = np.zeros(bu.shape[0]); p[bu] = v; return p
def harm(p):
    W = p.reshape(-1, DOF)[axi][:, :, 6]
    F = np.fft.rfft(W, axis=1)
    n = int(np.argmax((np.abs(F)**2).sum(0)))
    return n, F[:, n]
print('nx', nx, 'ny', nyy)
print('mu/mu0', np.round(mu/mu[0], 7))
H = [harm(full(raw[:, k])) for k in range(len(mu))]
for k in range(len(mu)):
    n, A = H[k]
    # axial half-wave count: sign changes of |A| profile relative phase
    ph = A*np.conj(A[np.argmax(np.abs(A))]); re = ph.real
    sc = int(np.sum(np.diff(np.sign(re[np.abs(re) > 1e-3*np.abs(re).max()])) != 0))
    print(k, 'n', n, 'axial sign changes', sc)
# pairwise axial-profile similarity for same n
for i in range(len(mu)):
    for j in range(i+1, len(mu)):
        if H[i][0] == H[j][0]:
            a, b = H[i][1], H[j][1]
            print('pair', i, j, 'n', H[i][0], '|<a,b>|/|a||b| = %.6f' % (abs(np.vdot(a, b))/np.linalg.norm(a)/np.linalg.norm(b)))
# does the mesh have exact cyclic symmetry? rotated mode residual in span of raw modes with same mu
p0 = full(raw[:, 0]); psi = degenerate_partner(p0, bu, axi, DOF)
print('partner of mode 0 None?', psi is None)
if psi is not None:
    B = np.column_stack([full(raw[:, k]) for k in range(len(mu))])
    c = np.linalg.lstsq(B, psi, rcond=None)[0]
    print('partner residual in span of all 12 modes %.2e' % (np.linalg.norm(psi - B@c)/np.linalg.norm(psi)), 'coefs', np.round(c, 3))
print('x of nodes, first station range', out['xlin'][:3], 'dy', 2*np.pi*0.4/ny)

x = np.sort(np.unique(out['xlin'])) if len(out['xlin']) == nx else None
for k in [0, 1]:
    n, A = H[k]
    amp = np.abs(A)/np.abs(A).max()
    print('mode', k, '|A(x)| every 5th station', np.round(amp[::5], 2))
    print('   mirror A(L-x) = s A(x): s=%+.4f' % (np.vdot(A, A[::-1]).real/np.vdot(A, A).real))
a0, a1 = H[0][1], H[1][1]
for name, c in [('sum', a0 + a1), ('diff', a0 - a1)]:
    print(name, np.round(np.abs(c)[::5]/np.abs(c).max(), 2))
