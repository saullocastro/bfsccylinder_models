"""The orthogonality conditions imposed on the second order fields.

For every pair of Koiter modes a, b and along every direction q of the null
space of phi2, the bordered system of the Koiter models is meant to impose

    q . dKT/dlambda . uab + 1/2 <N[L2(ua, ub)], L11(u0_dot, q)> = 0

which for a = b = q is Eq. (33) of Sun et al. (2020). Both terms are
recomputed here without the tensors the models assemble: the first from a
central difference of the tangent stiffness matrix of the compiled element
along the pre-buckling rate, exact because KC0 + KCNL(u) + KG(u) is quadratic
in u, and the second by integrating the strains of the element directly.

On a cylinder the constant vanishes between genuine buckling modes, their
circumferential harmonics not combining, and so does the difference between
the weighted and an Euclidean condition along a degenerate partner. A check
made with genuine modes would pass whatever the models imposed along those
directions, so the modes are made generic here, by adding a fixed random
field to what the eigen solver returns. The conditions are an algebraic
property of the bordered system and must hold for any modes.

References
----------
Sun, Y., Tian, K., Li, R., and Wang, B., 2020, "Accelerated Koiter Method for
Post-Buckling Analysis of Thin-Walled Shells under Axial Compression,"
Thin-Walled Struct., 155, p. 106962. DOI: 10.1016/j.tws.2020.106962
"""
import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from composites import laminated_plate
import bfsccylinder
import bfsccylinder.sanders
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD

import bfsccylinder_models.koiter_cylinder as vk_model
import bfsccylinder_models.koiter_cylinder_sanders as sa_model
from bfsccylinder_models.cyclic_symmetry import degenerate_partner, mesh_order

DOF = 10
L = 0.3556
R = 0.2032
STACK = (45, -45, 0, 90, 90, 0, -45, 45)
H = 0.00101539

KINEMATICS = {
    'von Karman': (vk_model, bfsccylinder, bfsccylinder.BFSCCylinder, False),
    'Sanders': (sa_model, bfsccylinder.sanders,
                bfsccylinder.sanders.BFSCCylinderSanders, True),
}


def _prop():
    return laminated_plate(stack=STACK,
            laminaprop=(127.629e9, 11.3074e9, 0.300235, 6.00257e9, 6.00257e9,
                        6.00257e9), plyt=H/len(STACK))


def _run(model, ny, num_eigvals, koiter_num_modes):
    nx = int(ny*L/(2*np.pi*R))
    if nx % 2 == 0:
        nx += 1
    return model.fkoiter_cyl_SS3(L, R, nx, ny, _prop(), cg_x0=None, nint=4,
            num_eigvals=num_eigvals, koiter_num_modes=koiter_num_modes,
            Nxxunit=20000., NLprebuck=True)


def _unknown_dofs(out):
    """The boundary conditions of fkoiter_cyl_SS3"""
    x, y = out['x'], out['y']
    bk = np.zeros(DOF*x.shape[0], dtype=bool)
    edges = np.isclose(x, 0) | np.isclose(x, L)
    bk[3::DOF] = edges
    bk[6::DOF] = edges
    bk[0::DOF] = np.isclose(x, L/2.) & np.isclose(y, 0)
    return ~bk


def _elements(out, element, prop):
    x = out['x']
    nid_pos = out['nid_pos']
    elements = []
    for n1, n2, n3, n4 in zip(out['n1s'], out['n2s'], out['n3s'], out['n4s']):
        elem = element(4)
        elem.n1, elem.n2, elem.n3, elem.n4 = n1, n2, n3, n4
        elem.c1 = DOF*nid_pos[n1]
        elem.c2 = DOF*nid_pos[n2]
        elem.c3 = DOF*nid_pos[n3]
        elem.c4 = DOF*nid_pos[n4]
        elem.R = R
        elem.lex = x[nid_pos[n2]] - x[nid_pos[n1]]
        elem.ley = 2*np.pi*R/out['ny']
        assign_constant_ABD(elem, prop)
        elements.append(elem)
    return elements


def _dKT_dlambda(out, lib, elements):
    """Central difference of KC0 + KCNL(u) + KG(u) along u0_dot

    KC0 does not depend on u and cancels, so only KCNL and KG are assembled
    """
    points, weights = get_points_weights(nint=4)
    u0 = out['koiter']['u0']
    u0dot = out['koiter']['u0dot']
    N = u0.shape[0]
    num_elements = len(elements)
    for k, elem in enumerate(elements):
        elem.init_k_KCNL = k*lib.KCNL_SPARSE_SIZE
        elem.init_k_KG = k*lib.KG_SPARSE_SIZE

    def assemble(update, size, u):
        r = np.zeros(num_elements*size, dtype=lib.INT)
        c = np.zeros(num_elements*size, dtype=lib.INT)
        v = np.zeros(num_elements*size, dtype=lib.DOUBLE)
        for elem in elements:
            update(u, elem, points, weights, r, c, v)
        return coo_matrix((v, (r, c)), shape=(N, N)).tocsr()

    def KT_varying(u):
        return (assemble(lib.update_KCNL, lib.KCNL_SPARSE_SIZE, u)
                + assemble(lib.update_KG, lib.KG_SPARSE_SIZE, u))

    #NOTE the difference is exact for a quadratic whatever the step, and a
    #     step the size of the load level keeps the cancellation small
    h = out['lambda_b']
    return (KT_varying(u0 + h*u0dot) - KT_varying(u0 - h*u0dot))/(2*h)


def _constants(out, elements, sanders, prop):
    """1/2 <N[L2(ua, ub)], L11(u0_dot, q)> for every a, b and direction q"""
    points, weights = get_points_weights(nint=4)
    G1s, G2s, idxs, ws = [], [], [], []
    for elem in elements:
        idx = np.concatenate([c + np.arange(DOF)
                              for c in (elem.c1, elem.c2, elem.c3, elem.c4)])
        for i in range(4):
            for j in range(4):
                xi, eta = points[i], points[j]
                elem.update_Sw_x(xi, eta)
                elem.update_Sw_y(xi, eta)
                #NOTE copies: the element exposes its buffers, which the next
                #     update overwrites
                G1 = np.array(elem.Sw_x, dtype=np.float64).reshape(-1)
                G2 = np.array(elem.Sw_y, dtype=np.float64).reshape(-1)
                if sanders:
                    elem.update_Sv(xi, eta)
                    G2 = G2 - np.array(elem.Sv, dtype=np.float64).reshape(-1)/R
                G1s.append(G1)
                G2s.append(G2)
                idxs.append(idx)
                ws.append(weights[i]*weights[j]*elem.lex*elem.ley/4.)
    G1s, G2s, idxs, ws = map(np.asarray, (G1s, G2s, idxs, ws))

    def L2(a, b):
        g1a = (G1s*a[idxs]).sum(axis=1)
        g2a = (G2s*a[idxs]).sum(axis=1)
        g1b = (G1s*b[idxs]).sum(axis=1)
        g2b = (G2s*b[idxs]).sum(axis=1)
        return np.array([g1a*g1b, g2a*g2b, g1a*g2b + g2a*g1b])

    koiter = out['koiter']
    cst = {}
    for (a, b) in koiter['uij']:
        Nab = prop.A @ L2(koiter['ui'][a], koiter['ui'][b])
        for k, q in koiter['ucond'].items():
            cst[(a, b, k)] = 0.5*(ws*(Nab*L2(koiter['u0dot'], q))).sum()
    return cst


def _generic(canonical_modes):
    rng = np.random.default_rng(0)

    def modes(*args, **kwargs):
        v = canonical_modes(*args, **kwargs)
        noise = rng.standard_normal(v.shape)
        return v + 0.2*noise*(np.linalg.norm(v, axis=0)
                              /np.linalg.norm(noise, axis=0))
    return modes


def _one_member_per_pair(eigsh):
    """The eigen solver, returning a single member of every degenerate pair

    Whether ARPACK returns both members of a degenerate pair, or one member
    and then the next multiplier, is decided by round off, and it changed
    between CI runs of the same code. With both members of the critical pair
    as the two Koiter modes, the null space of phi2 is spanned by the Koiter
    modes alone and there is no direction beyond them to check. The partner
    that was not returned, which is the case the column border is built for,
    is therefore made the case tested here, by asking for more eigenpairs and
    keeping the first of every multiplier
    """
    def solve(A, k, M, **kwargs):
        eigvals, eigvecs = eigsh(A=A, k=2*k, M=M, **kwargs)
        mu = -1/eigvals
        keep = []
        for j in range(eigvals.shape[0]):
            if not any(abs(mu[j] - mu[i]) <= 1e-5*abs(mu[i]) for i in keep):
                keep.append(j)
        keep = keep[:k]
        return eigvals[keep], eigvecs[:, keep]
    return solve


@pytest.mark.parametrize('kinematics', list(KINEMATICS))
def test_conditions_hold_along_every_direction_of_the_null_space(
        kinematics, monkeypatch):
    model, lib, element, sanders = KINEMATICS[kinematics]
    monkeypatch.setattr(model, 'eigsh', _one_member_per_pair(model.eigsh))
    monkeypatch.setattr(model, 'canonical_modes',
                        _generic(model.canonical_modes))
    solves = []
    spsolve = model.spsolve

    def capture(A, b):
        solves.append((A.tocsr(), np.array(b, copy=True)))
        return spsolve(A, b)

    monkeypatch.setattr(model, 'spsolve', capture)
    out = _run(model, ny=24, num_eigvals=2, koiter_num_modes=2)
    koiter = out['koiter']
    m = koiter['koiter_num_modes']
    bu = _unknown_dofs(out)
    nu = int(bu.sum())

    #NOTE the right hand side of every second order field is orthogonal to
    #     the Koiter modes, as the amplitude equations T z = -1/2 phi3_ij . u_k
    #     make it, with modes that are not T-orthogonal, T_kl = phi20_k . u_l,
    #     where lambda_l a_lij in place of z would not do. The bordered solves
    #     are the last m*m of the model
    U = np.column_stack([koiter['ui'][k][bu] for k in range(m)])
    A, _ = solves[-1]
    W = A[nu:nu + m, :nu].toarray()
    T = W @ U
    assert abs(T[0, 1]) > 1e-3*np.sqrt(abs(T[0, 0]*T[1, 1]))
    for A, b in solves[-m*m:]:
        assert A.shape[0] == nu + len(koiter['ucond'])
        g = b[:nu]
        assert np.abs(U.T @ g).max() <= 1e-9*np.linalg.norm(U, axis=0).max()*np.linalg.norm(g)
    #NOTE more directions than Koiter modes, so that the rows beyond them are
    #     checked too
    assert len(koiter['ucond']) > koiter['koiter_num_modes']

    prop = _prop()
    elements = _elements(out, element, prop)
    T = _dKT_dlambda(out, lib, elements)
    cst = _constants(out, elements, sanders, prop)

    firsts = []
    for (a, b), uab in koiter['uij'].items():
        Tuab = T @ uab
        for k, q in koiter['ucond'].items():
            first = q @ Tuab
            firsts.append(abs(first))
            residual = first + cst[(a, b, k)]
            scale = abs(first) + abs(cst[(a, b, k)])
            assert abs(residual) <= 1e-6*scale, (
                    'condition (%d, %d) along direction %d: %r + %r = %r'
                    % (a, b, k, first, cst[(a, b, k)], residual))
    #NOTE and the constants are not negligible, or the check above would not
    #     see them
    assert max(abs(c) for c in cst.values()) > 1e-3*max(firsts)


def test_column_border_contains_the_degenerate_partner(monkeypatch):
    """The eigen solver does not return the partner, which is the case the
    column border used to miss"""
    monkeypatch.setattr(sa_model, 'eigsh', _one_member_per_pair(sa_model.eigsh))
    out = _run(sa_model, ny=30, num_eigvals=2, koiter_num_modes=1)
    mu = out['mu']
    assert not np.isclose(mu[0], mu[1], rtol=1e-5), (
            'the partner was returned, so this mesh no longer tests the case')
    koiter = out['koiter']
    order = mesh_order(out['x'], out['y'], out['nx'], out['ny'])
    partner = degenerate_partner(koiter['ui'][0], _unknown_dofs(out), order,
                                 DOF)
    assert partner is not None
    Q = np.linalg.qr(np.column_stack(list(koiter['ucond'].values())))[0]
    left = partner - Q @ (Q.T @ partner)
    assert np.linalg.norm(left) <= 1e-8*np.linalg.norm(partner)


if __name__ == '__main__':
    import _pytest.monkeypatch
    mp = _pytest.monkeypatch.MonkeyPatch()
    for kin in KINEMATICS:
        test_conditions_hold_along_every_direction_of_the_null_space(kin, mp)
        mp.undo()
    test_column_border_contains_the_degenerate_partner(mp)
    mp.undo()
    print('ok')
