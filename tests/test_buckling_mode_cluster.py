"""Structure of the near-critical buckling spectrum.

These are the measurements that size a multi-mode Koiter expansion, and they
are what bounds the agreement of the present SINGLE-mode b_ijkl with the
literature: the expansion retains one eigenvector out of a cluster of six to
eight.

References
----------
Sun, Y., Tian, K., Li, R., and Wang, B., 2020, "Accelerated Koiter Method for
Post-Buckling Analysis of Thin-Walled Shells under Axial Compression,"
Thin-Walled Struct., 155, p. 106962. DOI: 10.1016/j.tws.2020.106962

Arbocz, J., Starnes, J. H., and Nemeth, M. P., 2001, "On a High-Fidelity
Hierarchical Approach to Buckling Load Calculations," AIAA-2001-1392.
DOI: 10.2514/6.2001-1392

Both papers describe the same shell, the NASA layered composite cylinder
AW-CYL-1-1, laminate [+-45/0/90]s.

All runs below use koiter_num_modes=0, which returns straight after the
eigenvalue analysis and skips the expensive Koiter tensor assembly.
"""
import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np
import pytest
from composites import laminated_plate

from bfsccylinder_models.koiter_cylinder_newton_raphson_sanders import (
        fkoiter_cyl_SS3)

L = 0.3556      # m
R = 0.2032      # m
E11 = 127.629e9
E22 = 11.3074e9
G12 = 6.00257e9
nu12 = 0.300235
STACK = (45, -45, 0, 90, 90, 0, -45, 45)
H = 0.00101539  # m


def _run(ny, num_eigvals, NLprebuck):
    nx = int(ny*L/(2*np.pi*R))
    if nx % 2 == 0:
        nx += 1
    prop = laminated_plate(stack=STACK,
                           laminaprop=(E11, E22, nu12, G12, G12, G12),
                           plyt=H/len(STACK))
    return fkoiter_cyl_SS3(L, R, nx, ny, prop, cg_x0=None, nint=4,
                           num_eigvals=num_eigvals, koiter_num_modes=0,
                           Nxxunit=20000., NLprebuck=NLprebuck)


@pytest.mark.parametrize('NLprebuck', [False, True])
def test_multipliers_come_in_degenerate_pairs(NLprebuck):
    """Every buckling multiplier of a cylinder is doubly degenerate.

    One mode for each sign of the circumferential wave number. This is the
    cyclic symmetry of the mesh, so it holds to solver tolerance, and it is
    the reason canonical_modes exists: the member of a pair the eigen solver
    returns is decided by round off.
    """
    mu = _run(ny=40, num_eigvals=8, NLprebuck=NLprebuck)['mu']
    for k in range(0, 8, 2):
        assert np.isclose(mu[k], mu[k + 1], rtol=1e-6), (
                'multipliers %d and %d are not a degenerate pair: %r vs %r'
                % (k, k + 1, mu[k], mu[k + 1]))
    #NOTE and consecutive pairs are genuinely distinct, so the pairing above
    #     is not an artefact of everything being equal
    for k in (0, 2, 4):
        assert mu[k + 2] > mu[k]*(1 + 1e-5)


def test_nonlinear_prebuckling_opens_a_spectral_gap():
    """With a non-linear pre-buckling state the near-critical cluster becomes
    a well-defined set.

    Measured on the Sun et al. shell, ny=40: four degenerate pairs lie within
    one per cent of the critical multiplier and the fifth sits 2.3 per cent
    above the fourth. That gap is an order of magnitude wider than the
    spacing inside the cluster, which is what makes a multi-mode truncation
    of it well posed. With a LINEAR pre-buckling state the corresponding gap
    is 0.28 per cent, comparable to the internal spacing, and no truncation
    is defensible.

    Regression values for this mesh; the qualitative statement, gap >> internal
    spacing, is the part that carries meaning.
    """
    mu = _run(ny=40, num_eigvals=10, NLprebuck=True)['mu']
    ratio = mu/mu[0]

    n_within_1pct = int((ratio - 1 <= 0.01).sum())
    assert n_within_1pct == 8, (
            'expected 4 degenerate pairs within 1%% of critical, got %d modes'
            % n_within_1pct)

    gap = ratio[8]/ratio[7] - 1
    assert gap > 0.02, 'expected a spectral gap above 2%%, got %.4f' % gap

    #NOTE the spacing INSIDE the cluster, which the gap must dominate
    internal = max(ratio[k + 2]/ratio[k] - 1 for k in (0, 2, 4))
    assert gap > 5*internal, (
            'gap %.4f does not dominate the internal spacing %.4f'
            % (gap, internal))


def test_single_mode_expansion_is_a_truncation_of_that_cluster():
    """The count that a multi-mode expansion would have to reach.

    koiter_num_modes=1 retains one eigenvector; closing the one per cent
    cluster of the Sun et al. shell needs eight, and its first omission is
    the degenerate partner of the critical mode itself. Recorded here so the
    number is checked rather than remembered.
    """
    mu = _run(ny=40, num_eigvals=10, NLprebuck=True)['mu']
    needed = int(((mu/mu[0] - 1) <= 0.01).sum())
    assert needed == 8
    #NOTE the partner of the critical mode is degenerate with it, so even
    #     koiter_num_modes=2 is the minimum for a complete critical eigenspace
    assert np.isclose(mu[0], mu[1], rtol=1e-6)


if __name__ == '__main__':
    test_multipliers_come_in_degenerate_pairs(False)
    test_multipliers_come_in_degenerate_pairs(True)
    test_nonlinear_prebuckling_opens_a_spectral_gap()
    test_single_mode_expansion_is_a_truncation_of_that_cluster()
    print('ok')
