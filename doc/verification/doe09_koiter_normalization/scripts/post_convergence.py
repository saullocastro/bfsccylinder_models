"""Table of the convergence study run by generate_qsubs_convergence.py

usage: python post_convergence.py [SUFFIX]

SUFFIX selects the study, '' for the single-mode one, '_k5' for the
five-mode one, '_k5c' (default) for the five distinct modes and their rotated
partners, and is appended to the name of the table written
"""
import glob
import json
import sys

DOE_name = 'DOE09'
suffix = sys.argv[1] if len(sys.argv) > 1 else '_k5c'

columns = ['case', 'NLprebuck', 'ny', 'nx', 'nxt', 'max_ny_nx_aspect_ratio',
           'DOF', 'dx_max_mm', 'Pcr', 'n', 'mu1_ratio', 'modes_n', 'mu_ratios',
           'b_min_t', 'b_min_energy', 'crest_e',
           'b_factor', 'b_iiii', 'b_iiii_crest', 'b_iiii_rms', 'crest_w',
           'num_distinct', 'lambda_b', 'converged',
           'time_s', 'peak_mem_gb', 'solvers', 'error']

rows = []
fnames = sorted(glob.glob(DOE_name + '_conv_*%s.out' % suffix))
if suffix == '':
    fnames = [f for f in fnames if f.endswith(('_LIN.out', '_NL.out'))]
for fname in fnames:
    result = None
    warning = False
    with open(fname) as f:
        for line in f:
            if line.startswith('RESULT '):
                result = json.loads(line[len('RESULT '):])
            elif 'WARNING: the iterative eigenvalue algorithm stopped' in line:
                #NOTE not the WARNING of a solver falling back to SuperLU
                warning = True
    if result is None:
        #NOTE killed, e.g. out of memory or walltime
        rows.append([fname] + ['None']*(len(columns) - 1))
        continue
    r = result
    if 'error' in r:
        rows.append([str(r['case']), str(r['NLprebuck'])]
                    + ['None']*(len(columns) - 6)
                    + [str(r['time_s']), str(r.get('peak_mem_gb')),
                       r['solvers'], r['error']])
        continue
    rows.append([str(v) for v in [
        r['case'], r['NLprebuck'], r['ny'], r['nx'], r['nxt'],
        r['max_ny_nx_aspect_ratio'], 10*r['nx']*r['ny'], 1e3*r['dx_max'],
        #NOTE the second distinct Koiter mode, see koiter_distinct
        r['Pcr'], r['n'], r['mu_ratios'][r['koiter_distinct'].index(True, 1)]
        if 'koiter_distinct' in r else r['mu_ratios'][1],
        '/'.join(str(n) for n in r['modes_n']),
        '/'.join('%.6f' % m for m in r['mu_ratios']),
        r.get('b_min_t'), r.get('b_min_energy'), r.get('crest_e'),
        r['b_factor'],
        '/'.join('%.6g' % b for b in r.get('b_iiii', [r['b_factor']])),
        #NOTE b_iiii of the modes with a crest, or an RMS of w, equal to the
        #     thickness, see ElementField in run_case.py
        '/'.join('%.6g' % (b/c**2) for b, c in
                 zip(r.get('b_iiii', []), r.get('crest_w', []))) or None,
        '/'.join('%.6g' % (b/c**2) for b, c in
                 zip(r.get('b_iiii', []), r.get('rms_w', []))) or None,
        '/'.join('%.5f' % c for c in r.get('crest_w', [])) or None,
        r.get('num_distinct'), r['lambda_b'], not warning, r['time_s'],
        r.get('peak_mem_gb'), r['solvers'], '']])

with open(DOE_name + '_convergence%s.txt' % suffix, 'w') as f:
    f.write('# ' + ', '.join(columns) + '\n')
    for row in rows:
        f.write(', '.join(row) + '\n')
print('# runs', len(rows))
