"""b_min over subsets of the wave-number clusters of the Koiter set

From the stored RESULT lines of a convergence study, no model run. The
Koiter modes of every run are grouped by circumferential wave number n
(modes_n), a cluster being the 2 or 4 modes of one n: the symmetric and
antisymmetric modes and their rotated partners. For every run are printed

- lambda_b/lambda_c, the expansion point reached by the load stepping;
- the weight sum e_i**2 of every cluster in the minimum direction e_min;
- b_min of the energy normalization, the minimum of the symmetrized b_ijkl
  over e.e = 1 (koiter_post.min_direction), restricted to every subset of
  one, two and three clusters, which only takes the sub-block of b_ijkl;
- the share of |b_ijkl|**2 in the entries that couple three or four
  different wave numbers.

Unlike the b_iiii of a single mode, the b_min of a whole cluster does not
depend on the slice of a degenerate eigenspace returned by the eigen solver,
and is followed across meshes as a measure of the discretization error;
see the Richardson estimate at the end.

usage, from doc/verification/doe09_koiter_normalization:

    python checks/cluster_subsets.py [results/DOE09_conv_k5g.jsonl.gz]
"""
import gzip
import itertools
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import koiter_post as kp

#NOTE Nxxunit of run_case.py, so that lambda_c = Pcr/(Nxxunit*circ)
Nxxunit = 1000.
circ = 2*np.pi*0.4


def clusters(r):
    groups = {}
    for k, n in enumerate(r['modes_n'][:r['koiter_num_modes']]):
        groups.setdefault(n, []).append(k)
    return groups


def energy_b(r):
    b = np.array(r['b_ijkl'])
    a = np.array(r['a_ijk'])
    return kp.rescaled(b, a, kp.energy_scales(r['lambda_d']))[0]


def subset_b(be, idx):
    return kp.min_direction(be[np.ix_(idx, idx, idx, idx)], num_starts=20)[0]


def main(path):
    runs = [json.loads(l)['result'] for l in gzip.open(path, 'rt')]
    runs = [r for r in runs if r and 'error' not in r]
    runs.sort(key=lambda r: (r['case'], not r['NLprebuck'], r['ny']))
    per_cluster = {}
    for r in runs:
        be = energy_b(r)
        groups = clusters(r)
        keys = sorted(groups)
        e = np.array(r['e_min'])
        pre = 'NL' if r['NLprebuck'] else 'LIN'
        print('case %d %s ny=%d nx=%d m=%d lambda_b/lambda_c=%.5f '
              'b_min_energy=%.4g crest_e=%.3f b_min_t=%.4f'
              % (r['case'], pre, r['ny'], r['nx'], r['koiter_num_modes'],
                 r['lambda_b']/(r['Pcr']/(Nxxunit*circ)), r['b_min_energy'],
                 r['crest_e'], r['b_min_t']))
        print('    weights in e_min: ' + ', '.join('n=%d (%d modes) %.2f'
              % (n, len(groups[n]), (e[groups[n]]**2).sum()) for n in keys))
        for size in range(1, len(keys) + 1):
            row = []
            for c in itertools.combinations(keys, size):
                idx = sum((groups[n] for n in c), [])
                bmin = subset_b(be, idx)
                row.append('%s: %.4g' % ('+'.join(map(str, c)), bmin))
                if size == 1 and len(groups[c[0]]) == 4:
                    per_cluster.setdefault((r['case'], pre, c[0]), {})[
                            r['ny']] = bmin
            print('    b_min over %d cluster(s): %s' % (size, ', '.join(row)))
        bs = kp.symmetrized(be)
        ns = r['modes_n'][:r['koiter_num_modes']]
        three = sum(bs[idx]**2 for idx in np.ndindex(bs.shape)
                    if len({ns[i] for i in idx}) >= 3)
        print('    entries coupling three or more wave numbers: %.0f %% of '
              '|b_ijkl|**2' % (100*three/(bs**2).sum()))

    print()
    print('b_min of single complete clusters (4 modes), Richardson from '
          'ny = 120, 160 and 200')
    for (case, pre, n), vals in sorted(per_cluster.items()):
        if not all(ny in vals for ny in (120, 160, 200)):
            continue
        h = 1/np.array([120., 160., 200.])
        f = np.array([vals[120], vals[160], vals[200]])
        line = 'case %d %s n=%d: %s' % (case, pre, n, ', '.join('%.4g' % v
                                                              for v in f))
        R = (f[1] - f[0])/(f[2] - f[1]) if f[2] != f[1] else np.inf
        g = lambda p: (h[0]**p - h[1]**p)/(h[1]**p - h[2]**p) - R
        try:
            p = brentq(g, 0.05, 30)
        except ValueError:
            print(line + ', not monotone')
            continue
        C = (f[1] - f[2])/(h[1]**p - h[2]**p)
        finf = f[2] - C*h[2]**p
        print(line + ', order %.1f, extrapolated %.4g, error %.1f %% at 160 '
              'and %.1f %% at 200' % (p, finf, 100*(f[1] - finf)/finf,
                                      100*(f[2] - finf)/finf))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join('results', 'DOE09_conv_k5g.jsonl.gz'))
