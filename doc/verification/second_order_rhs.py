"""Which right-hand side makes the second-order expansion consistent?

The second order fields of an m-mode Koiter expansion about a non-linear
pre-buckling state solve

    phi2 uij = -1/2 phi3_ij - sum_l z_l phi20_l

with z_l the coefficient of xi_i xi_j in (lambda - lambda_l) xi_l. Three
candidates for z are compared:

    'T^-1'    z = -1/2 T^-1 [phi3_ij . u_k],  T_kl = phi20_k . u_l,
              what the projection of the same terms onto the modes gives
    'a, 1/m'  z_l = lambda_l a_lij/m, what the models used to do
    'a'       z_l = lambda_l a_lij, the first without the 1/m, exact only for
              modes orthogonal with respect to T

on a small polynomial energy with two simultaneously critical modes, a
non-linear fundamental path, non-zero a_ijk and modes that are not
T-orthogonal, written in coordinates that are not orthonormal so that the
Euclidean column border of the bordered system cannot repair a wrong
right-hand side by accident.

The test is the order of the residual of the equilibrium equations along a
post-buckling ray of the perfect system, u(s) = u0(lambda) + s U e
+ s**2 sum e_i e_j uij with lambda = lambda_c + s lambda_1: it must be O(s**3)
if the second-order fields are right, and it is O(s**2) otherwise.

Reported in: Section "The singular system".
Runtime: seconds.
"""

import numpy as np
from scipy.optimize import brentq

rng = np.random.default_rng(7)
nx, ny = 2, 4
N = nx + ny
X = slice(0, nx)
Y = slice(nx, N)


def sym3(T):
    return (T + T.transpose(0, 2, 1) + T.transpose(1, 0, 2)
            + T.transpose(1, 2, 0) + T.transpose(2, 0, 1)
            + T.transpose(2, 1, 0))/6.


# energy in coordinates z = (x, y): 1/2 z K2 z + 1/6 K3[z, z, z] - lam F.z,
# with no term linear in x, so that the fundamental path has x = 0
K2 = np.zeros((N, N))
M = rng.standard_normal((ny, ny))
K2[Y, Y] = M @ M.T + ny*np.eye(ny)
K3 = np.zeros((N, N, N))
K3[Y, Y, Y] = 0.3*sym3(rng.standard_normal((ny, ny, ny)))
K3[X, X, X] = sym3(rng.standard_normal((nx, nx, nx)))
#NOTE 'orthogonal' in the command line makes the x block of the Hessian
#     diagonal all along the path, and so T diagonal, which is the case where
#     only the 1/m is wrong
T_orthogonal = 'orthogonal' in __import__('sys').argv
Bx = np.diag([2.0, 1.0]) if T_orthogonal else np.array([[2.0, 0.7],
                                                         [0.7, 1.0]])
beta = rng.standard_normal(ny)
E = 0.2*rng.standard_normal((nx, nx, ny))
E = (E + E.transpose(1, 0, 2))/2.
if T_orthogonal:
    E[0, 1] = E[1, 0] = 0.
xxy = Bx[:, :, None]*beta[None, None, :] + E
K3[X, X, Y] = xxy
K3[X, Y, X] = xxy.transpose(0, 2, 1)
K3[Y, X, X] = xxy.transpose(2, 0, 1)
F = np.zeros(N)
F[Y] = rng.standard_normal(ny)


def path_y(lam, y=None):
    y = np.zeros(ny) if y is None else y.copy()
    for _ in range(50):
        r = K2[Y, Y] @ y + 0.5*np.einsum('abc,a,b->c', K3[Y, Y, Y], y, y) \
            - lam*F[Y]
        J = K2[Y, Y] + np.einsum('abc,a->bc', K3[Y, Y, Y], y)
        dy = np.linalg.solve(J, -r)
        y += dy
        if np.abs(dy).max() < 1e-15*max(1, np.abs(y).max()):
            break
    return y


# the x block of the Hessian on the path is K2xx + K3[x, x, y(lam)]; making
# it vanish at lam_c makes both x directions critical there
lam_c = 1.0
yc = path_y(lam_c)
K2[X, X] = -np.einsum('ija,a->ij', K3[X, X, Y], yc)
# the sign of beta decides which side of lam_c is stable; flip if needed
yb = path_y(0.9*lam_c)
if np.linalg.eigvalsh(K2[X, X] + np.einsum('ija,a->ij', K3[X, X, Y], yb)).min() < 0:
    K3[X, X, Y] *= -1
    K3[X, Y, X] *= -1
    K3[Y, X, X] *= -1
    K2[X, X] *= -1

# the same energy in coordinates u = Q z that are not orthonormal
Q = np.eye(N) + 0.6*rng.standard_normal((N, N))
Qi = np.linalg.inv(Q)
K2u = Qi.T @ K2 @ Qi
K3u = np.einsum('abc,ai,bj,ck->ijk', K3, Qi, Qi, Qi)
Fu = Qi.T @ F


def grad(u, lam):
    return K2u @ u + 0.5*np.einsum('ijk,i,j->k', K3u, u, u) - lam*Fu


def hess(u):
    return K2u + np.einsum('ijk,i->jk', K3u, u)


def u_path(lam):
    z = np.zeros(N)
    z[Y] = path_y(lam, yc)
    return Q @ z


u0 = u_path(lam_c)
h = 1e-5
u0dot = (u_path(lam_c + h) - u_path(lam_c - h))/(2*h)
phi2 = hess(u0)
phi20 = np.einsum('ijk,i->jk', K3u, u0dot)
U = Q[:, X]
assert np.abs(phi2 @ U).max() < 1e-8*np.abs(phi2).max()
T = U.T @ phi20 @ U
print('T =\n', T)


def phi3(a, b):
    return np.einsum('ijk,i,j->k', K3u, a, b)


def second_order(variant):
    Nsp = np.linalg.qr(U)[0]
    W = phi20 @ U
    A = np.block([[phi2, Nsp], [W.T, np.zeros((nx, nx))]])
    uij, alpha = {}, {}
    for i in range(nx):
        for j in range(nx):
            p3 = phi3(U[:, i], U[:, j])
            pk = U.T @ p3
            if variant == 'T^-1':
                z = np.linalg.solve(T, -0.5*pk)
            else:
                a = -pk/(2*lam_c*np.diag(T))
                z = lam_c*a/(nx if variant == 'a, 1/m' else 1)
            g = -0.5*p3 - phi20 @ U @ z
            sol = np.linalg.solve(A, np.concatenate((g, np.zeros(nx))))
            uij[(i, j)] = sol[:N]
            alpha[(i, j)] = sol[N:]
    return uij, alpha


def rays():
    """Directions e with lambda_1 T e = -1/2 [phi3(Ue, Ue) . u_k]"""
    def cross(t):
        e = np.array([np.cos(t), np.sin(t)])
        p = U.T @ phi3(U @ e, U @ e)
        v = T @ e
        return v[0]*p[1] - v[1]*p[0]
    ts = np.linspace(0, np.pi, 721)
    f = [cross(t) for t in ts]
    out = []
    for k in range(len(ts) - 1):
        if f[k]*f[k + 1] < 0:
            t = brentq(cross, ts[k], ts[k + 1], xtol=1e-14)
            e = np.array([np.cos(t), np.sin(t)])
            p = U.T @ phi3(U @ e, U @ e)
            v = T @ e
            out.append((e, -0.5*(v @ p)/(v @ v)))
    return out


found = rays()
assert found, 'no post-buckling ray'
print('rays (e, lambda_1):', [(np.round(e, 4), round(l1, 5)) for e, l1 in found])
ss = np.array([4e-3, 2e-3, 1e-3, 5e-4])
for variant in ('T^-1', 'a', 'a, 1/m'):
    uij, alpha = second_order(variant)
    amax = max(np.abs(v).max() for v in alpha.values())
    orders = []
    for e, lam1 in found:
        res = []
        for s in ss:
            u2 = sum(e[i]*e[j]*uij[(i, j)] for i in range(nx) for j in range(nx))
            lam = lam_c + s*lam1
            u = u_path(lam) + s*(U @ e) + s**2*u2
            res.append(np.linalg.norm(grad(u, lam)))
        orders.append(np.polyfit(np.log(ss), np.log(res), 1)[0])
    print('%-8s  |alpha| max %.2e   residual order along each ray %s'
          % (variant, amax, ', '.join('%.2f' % o for o in orders)))
