"""Buckling load of the lowest mode of every wave number, across meshes

From the stored RESULT lines, no model run: mu_ratios times Pcr for the
num_eigvals modes of every run, the lowest per circumferential wave number
(modes_n), marked * when that wave number is in the Koiter set. Shows how
far apart the clusters are, against how much each moves with the mesh.

usage, from doc/verification/doe09_koiter_normalization:

    python checks/multipliers_by_wavenumber.py [results/DOE09_conv_k5g.jsonl.gz] [NL|LIN]
"""
import gzip
import json
import os
import sys


def main(path, NLprebuck=True):
    runs = [json.loads(l)['result'] for l in gzip.open(path, 'rt')]
    runs = [r for r in runs if r and 'error' not in r
            and r['NLprebuck'] == NLprebuck]
    for case in sorted({r['case'] for r in runs}):
        table = {}
        for r in runs:
            if r['case'] != case:
                continue
            m = r['koiter_num_modes']
            best = {}
            for k, (n, q) in enumerate(zip(r['modes_n'], r['mu_ratios'])):
                P = q*r['Pcr']
                if n not in best or P < best[n][0]:
                    best[n] = (P, k < m)
            table[r['ny']] = best
        nys = sorted(table)
        print('case %d %s, buckling load in N of the lowest mode of each wave '
              'number n, * in the Koiter set' % (case,
                                                 'NL' if NLprebuck else 'LIN'))
        print('    n' + ''.join('%12s' % ('ny=%d' % ny) for ny in nys))
        for n in sorted(set().union(*table.values())):
            row = ''
            for ny in nys:
                if n in table[ny]:
                    P, in_set = table[ny][n]
                    row += '%11.1f%s' % (P, '*' if in_set else ' ')
                else:
                    row += '%12s' % '-'
            print('  %3d%s' % (n, row))


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
            'results', 'DOE09_conv_k5g.jsonl.gz')
    main(path, (sys.argv[2] if len(sys.argv) > 2 else 'NL') == 'NL')
