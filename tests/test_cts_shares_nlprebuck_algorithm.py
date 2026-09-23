"""The four models must carry the SAME NLprebuck and Koiter algorithm.

The algorithm was spliced from koiter_cylinder.py and koiter_cylinder_sanders.py
into koiter_cylinder_CTS*.py rather than retyped, so that the model families
cannot drift apart. This test asserts that character for character:

- across all four models, on the axisymmetric pre-buckling solver and the
  iterative eigenvalue algorithm, and on the whole Koiter section, from the
  normalization of the modes to the dict of coefficients returned; the Koiter
  tensors and coefficients themselves are computed by koiter_tensors.py,
  which all four call;
- between the models of the same kinematics, on nonlinear_rows, the rows of
  the nonlinear membrane strains, which is the one place where the von Karman
  and the Sanders models differ in the Koiter section.

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

MODELS = ['koiter_cylinder.py', 'koiter_cylinder_sanders.py',
          'koiter_cylinder_CTS.py', 'koiter_cylinder_CTS_sanders.py']

#NOTE (start, end) of each shared region. Both markers must be unique in both
#     files compared, which is itself part of what is being asserted
REGIONS = [
    ('    def assemble_KG(u):', '    Pcr = load_mult[0]*Nxxunit*circ'),
    #NOTE not '    return out' as the end marker: the CTS models return early
    #     for mesh_only
    ('    lambda_a = {}', "    out['koiter'] = koiter"),
]
KINEMATICS_REGIONS = [
    ('def nonlinear_rows(elem, xi, eta):', '    return G1, G2'),
]
SAME_KINEMATICS = [
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


def _assert_same(a_name, b_name, start, end):
    a = _region(_read(a_name), start, end, a_name)
    b = _region(_read(b_name), start, end, b_name)
    assert a == b, (
            '%s and %s have diverged in the region starting at %r.\n'
            'The algorithm is meant to be identical in both; port the change '
            'to the other files rather than letting them drift.'
            % (a_name, b_name, start))


@pytest.mark.parametrize('model', MODELS[:-1])
@pytest.mark.parametrize('start, end', REGIONS)
def test_shared_region_is_identical(model, start, end):
    _assert_same(MODELS[-1], model, start, end)


@pytest.mark.parametrize('a_name, b_name', SAME_KINEMATICS)
@pytest.mark.parametrize('start, end', KINEMATICS_REGIONS)
def test_kinematics_is_identical(a_name, b_name, start, end):
    _assert_same(a_name, b_name, start, end)


if __name__ == '__main__':
    for model in MODELS[:-1]:
        for start, end in REGIONS:
            test_shared_region_is_identical(model, start, end)
    for a_name, b_name in SAME_KINEMATICS:
        for start, end in KINEMATICS_REGIONS:
            test_kinematics_is_identical(a_name, b_name, start, end)
    print('ok')
