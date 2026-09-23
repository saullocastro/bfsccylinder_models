"""The CTS models must carry the SAME NLprebuck algorithm as the
constant-stiffness models.

The algorithm was spliced from koiter_cylinder.py and koiter_cylinder_sanders.py
into koiter_cylinder_CTS*.py rather than retyped, so that the two model families
cannot drift apart. This test asserts that character for character, on the
regions that are meant to be shared: the axisymmetric pre-buckling solver and
the iterative eigenvalue algorithm, the flag note, the kinematics of the Koiter
tensors and the call that integrates them, phi2, the a_ijk, the bordered system
for the second-order fields, and the b_ijkl block. The element integration of
the Koiter tensors itself, the pre-buckling state and its rates included, is in
koiter_tensors.py, which all four models call, so it is shared by construction.

What is deliberately NOT shared, and so is not compared: the mesh generation
of the CTS parameterization, the per-integration-point ABD of a
variable-stiffness laminate, and the note on why the axisymmetric reduction
applies to a CTS cylinder.
"""
import io
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, os.pardir, 'bfsccylinder_models')

#NOTE (start, end) of each shared region. Both markers must be unique in both
#     files of a pair, which is itself part of what is being asserted
REGIONS = [
    ('    def assemble_KG(u):', '    Pcr = load_mult[0]*Nxxunit*circ'),
    ('    #NOTE this flag multiplies', '    flag = NLprebuck'),
    ('    #NOTE the null space of phi2, against which',
     '    num_cond = len(ucond)'),
    ('def nonlinear_rows(elem, xi, eta):', '    return G1, G2'),
    ('    #NOTE integrated by koiter_element_tensors',
     '    Ucond = np.column_stack([ucond[modek] for modek in range(num_cond)])'),
    ('    phi20, phi3, phi30, cst, phi200, phi4 = koiter_element_tensors(',
     '            nonlinear_rows, calc_AB)'),
    ('    #NOTE phi2 must be the SAME operator', '    phi2uu = KCuu + KGuu*mu[0]'),
    ("    print('# a_ijk factors')",
     "                print('# $a_%d%d%d$' % (modei+1, modej+1, modek+1), a_ijk)"),
    ('    #NOTE the second order fields solve the terms of order',
     '        return -1/2.*phi3[:, modei, modej] - phi20[:, :koiter_num_modes] @ z'),
    ('    #NOTE phi2 is singular by construction',
     '            uab[(modei, modej)] = uijbar'),
    #NOTE not '    return out' as the end marker: the CTS models return early
    #     for mesh_only and for koiter_num_modes == 0
    ("    print('# b_ijkl factors')", "    out['koiter'] = koiter"),
]

PAIRS = [
    ('koiter_cylinder_CTS.py', 'koiter_cylinder.py'),
    ('koiter_cylinder_CTS_sanders.py', 'koiter_cylinder_sanders.py'),
]


def _read(name):
    with io.open(os.path.join(PKG, name), encoding='utf-8') as f:
        return f.read()


def _region(text, start, end, name):
    assert text.count(start) == 1, (
            '%s: %d occurrences of the start marker %r'
            % (name, text.count(start), start))
    assert text.count(end) == 1, (
            '%s: %d occurrences of the end marker %r'
            % (name, text.count(end), end))
    i = text.index(start)
    j = text.index(end, i) + len(end)
    assert j > i, '%s: end marker precedes the start marker' % name
    return text[i:j]


@pytest.mark.parametrize('cts, nr', PAIRS)
@pytest.mark.parametrize('start, end', REGIONS)
def test_shared_region_is_identical(cts, nr, start, end):
    a = _region(_read(cts), start, end, cts)
    b = _region(_read(nr), start, end, nr)
    assert a == b, (
            '%s and %s have diverged in the region starting at %r.\n'
            'The NLprebuck algorithm is meant to be identical in both; port '
            'the change to the other file rather than letting them drift.'
            % (cts, nr, start))


if __name__ == '__main__':
    for cts, nr in PAIRS:
        for start, end in REGIONS:
            test_shared_region_is_identical(cts, nr, start, end)
    print('ok')
