import json
import os

import numpy as np

DOE = 'DOE09'
size_of_DOE = np.loadtxt(DOE + '.txt', skiprows=1).shape[0]
#NOTE as in run_case.py and generate_qsubs.py: 5 distinct modes and their
#     rotated partners
koiter_num_modes = 10
prebucks = ['LIN', 'NL']

inputs = ['v1', 'v2', 'v3', 'v4', 'v5']
#NOTE param_n and c2_ratio as analyzed, see run_case.py
effective = ['param_n', 'c2_ratio']
#NOTE gap: lowest multiplier of a mode distinct from the critical pair over
#     the critical one, None if the 4 computed modes are all degenerate. The
#     single-mode b_factor is determinate only for gaps appreciably larger
#     than NLprebuck_eps1 = 0.005, see sec:convergence of the manuscript
#     b_factor is b_1111 of the critical mode; b_iiii the b_iiii of the
#     koiter_num_modes distinct modes of run_case.py, joined by '/', whose
#     full b_ijkl and a_ijk are in the RESULT line of each run
results = ['mass', 'Pcr', 'b_min_t', 'b_min_energy', 'b_factor', 'b_iiii',
           'num_distinct', 'lambda_b',
           'n', 'axi_share', 'mu1_ratio', 'gap', 'converged', 'solvers']
columns = (['run ID'] + inputs + effective
           + ['%s_%s' % (r, p) for p in ['LIN', 'NL'] for r in results])


def read_result(fname):
    """RESULT dict of a run, None if it did not finish"""
    if not os.path.isfile(fname):
        return None
    result = None
    #NOTE the iterative eigenvalue algorithm stopping short of
    #     1 - NLprebuck_eps1 is a failure, not a note, see "Recommendation"
    #     in doc/nlprebuck_implementation.tex of bfsccylinder_models
    warning = False
    with open(fname) as f:
        for line in f:
            if line.startswith('RESULT '):
                result = json.loads(line[len('RESULT '):])
            elif 'WARNING: the iterative eigenvalue algorithm stopped' in line:
                #NOTE not the WARNING of a solver falling back to SuperLU
                warning = True
    if result is None or 'error' in result:
        return None
    if (result.get('koiter_num_modes') != koiter_num_modes
            or 'b_min_energy' not in result):
        #NOTE a run of an earlier setup, see generate_qsubs.py
        return None
    #NOTE the multiplier of the second distinct Koiter mode, the first
    #     being followed by its rotated partner
    result['mu1_ratio'] = result['mu_ratios'][result['koiter_distinct'].index(
        True, 1)]
    result['gap'] = next((m for m in result['mu_ratios'] if m > 1 + 1.e-6),
                         None)
    result['converged'] = not warning
    result['b_iiii'] = '/'.join('%.6g' % b for b in
                                result.get('b_iiii', [result['b_factor']]))
    result.setdefault('num_distinct', None)
    return result


output = ['# ' + ', '.join(columns) + '\n']
num_missing = dict(LIN=0, NL=0)
#NOTE b_ijkl[i, p, j, k, l, m] and a_ijk[i, p, j, k, l] of run i, p indexing
#     prebucks, for the nodal normalization of the models, the modes ordered
#     by increasing multiplier, mode 0 being the critical one, see
#     run_case.py. NaN for a missing or failed run
m = koiter_num_modes
b_ijkl = np.full((size_of_DOE, len(prebucks)) + (m,)*4, np.nan)
a_ijk = np.full((size_of_DOE, len(prebucks)) + (m,)*3, np.nan)
mu_ratios = np.full((size_of_DOE, len(prebucks), m), np.nan)
modes_n = np.full((size_of_DOE, len(prebucks), m), -1)
#NOTE the scales of the other normalizations, see koiter_post.py: b_ijkl
#     and a_ijk of the modes with a crest or an RMS of w equal to the
#     thickness, or of the energy normalization of Rahman (2009), are
#     koiter_post.rescaled(b_ijkl, a_ijk, s) with s = 1/crest_w, 1/rms_w or
#     koiter_post.energy_scales(lambda_d). Not stored, 1.6 GB each
crest_w = np.full((size_of_DOE, len(prebucks), m), np.nan)
rms_w = np.full((size_of_DOE, len(prebucks), m), np.nan)
lambda_d = np.full((size_of_DOE, len(prebucks), m), np.nan)
distinct = np.zeros((size_of_DOE, len(prebucks), m), dtype=bool)
#NOTE b_min of the energy normalization, its direction e_min over the
#     Koiter modes, and b_min_t of the combined mode scaled to a crest of w
#     equal to the thickness, see koiter_post.py
b_min_energy = np.full((size_of_DOE, len(prebucks)), np.nan)
b_min_t = np.full((size_of_DOE, len(prebucks)), np.nan)
crest_e = np.full((size_of_DOE, len(prebucks)), np.nan)
e_min = np.full((size_of_DOE, len(prebucks), m), np.nan)
for i in range(size_of_DOE):
    runs = {}
    for p, prebuck in enumerate(prebucks):
        runs[prebuck] = r = read_result(DOE + '_%05d_%s.out' % (i, prebuck))
        if r is None:
            num_missing[prebuck] += 1
            continue
        b_ijkl[i, p] = r['b_ijkl']
        a_ijk[i, p] = r['a_ijk']
        mu_ratios[i, p] = r['mu_ratios'][:m]
        modes_n[i, p] = r['modes_n'][:m]
        crest_w[i, p] = r['crest_w']
        rms_w[i, p] = r['rms_w']
        lambda_d[i, p] = r['lambda_d']
        distinct[i, p] = r['koiter_distinct']
        b_min_energy[i, p] = r['b_min_energy']
        b_min_t[i, p] = r['b_min_t']
        crest_e[i, p] = r['crest_e']
        e_min[i, p] = r['e_min']
    first = runs['LIN'] or runs['NL']
    row = ['%s_%05d' % (DOE, i)]
    row += [str(first[k]) if first else 'None' for k in inputs + effective]
    for prebuck in ['LIN', 'NL']:
        r = runs[prebuck]
        row += [str(r[k]) if r else 'None' for k in results]
    output.append(', '.join(row) + '\n')

print('# missing or failed runs', num_missing)
with open(DOE + '_output.txt', 'w') as f:
    f.writelines(output)
#NOTE 10,000 b_ijkl per run, too many for the columns of the table above
np.savez_compressed(DOE + '_koiter.npz', run_id=['%s_%05d' % (DOE, i)
        for i in range(size_of_DOE)], prebuck=prebucks, b_ijkl=b_ijkl,
        a_ijk=a_ijk, crest_w=crest_w, rms_w=rms_w, lambda_d=lambda_d,
        distinct=distinct, b_min_energy=b_min_energy, b_min_t=b_min_t,
        crest_e=crest_e, e_min=e_min, mu_ratios=mu_ratios, modes_n=modes_n)
