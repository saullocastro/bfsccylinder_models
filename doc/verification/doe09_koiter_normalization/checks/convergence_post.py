"""Tables of the mesh convergence study from a low ny,
scripts/generate_qsubs_convergence.py

From the RESULT lines, archived in results/DOE09_conv_ir.jsonl.gz, one
{"file": ..., "result": ...} per run, result null for a run that wrote none.
For every case (0, 1, 6), NL, eps1 = 0.0005, on the SS3 edges with inertia
relief of the models:

- per axial factor F, the ny sequence: nx, dx max, the critical wave number
  n_c and the elements per wave ny/n_c, Pcr, the number of Koiter modes m,
  b_min_t of the full set, of the critical cluster alone (crit b_t) and of
  the window n_c +- 1 (win1 b_t, * a cluster of it incomplete), and the
  Richardson extrapolation in ny over every three consecutive meshes;
- over the meshes with F >= 2, where nx grows with ny, and at least
  min_per_wave elements per wave of n_c, the fit f = f_inf + a/ny**p +
  c/nx**q, and the error of every mesh against f_inf split into its ny and
  nx parts. The meshes with F <= 1 are left out: for case 1 they keep
  nx = 67 up to ny = 96, which misses the critical mode.

A cluster is the modes of one wave number (cluster_subsets.py).

usage, from doc/verification/doe09_koiter_normalization:

    python checks/convergence_post.py [--archive DOE09_DIR] [--min-per-wave W]

--archive first writes the archive from the outputs in DOE09_DIR
"""
import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cluster_subsets as cs

here = os.path.join(os.path.dirname(__file__), '..')
archive_name = os.path.join(here, 'results', 'DOE09_conv_ir.jsonl.gz')
patterns = ['DOE09_conv_*_NL_SS3IR_F*_eps0p0005.out']


def archive(workdir):
    paths = sorted(sum([glob.glob(os.path.join(workdir, p))
                        for p in patterns], []))
    os.makedirs(os.path.dirname(archive_name), exist_ok=True)
    with gzip.open(archive_name, 'wt') as f:
        for path in paths:
            result = None
            with open(path) as g:
                for line in g:
                    if line.startswith('RESULT '):
                        result = json.loads(line[len('RESULT '):])
            f.write(json.dumps(dict(file=os.path.basename(path),
                                    result=result)) + '\n')
    print('# %d runs in %s' % (len(paths), archive_name))


def load():
    if not os.path.isfile(archive_name):
        return []
    rows = [json.loads(l) for l in gzip.open(archive_name, 'rt')]
    for row in rows:
        r = row['result']
        if r is None or 'error' in r:
            print('# %s: %s' % (row['file'], 'no RESULT' if r is None
                                else r['error']))
    return [row['result'] for row in rows
            if row['result'] and 'error' not in row['result']]


def pct(a, b):
    """relative difference of a from b, in per cent"""
    return 100*(a - b)/abs(b)


def subset(r, kind, ns):
    for s in r.get('subsets', []):
        if s['kind'] == kind and s['ns'] == list(ns):
            return s
    return None


def critical(r):
    """b_min_t of the critical cluster alone"""
    s = subset(r, 'window', [r['modes_n'][0]])
    return s['b_min_t'] if s else None


def window1(r):
    """b_min_t of the window n_c - 1 to n_c + 1, and whether it is complete"""
    n_c = r['modes_n'][0]
    s = subset(r, 'window', [n_c - 1, n_c, n_c + 1])
    return (s['b_min_t'], s['complete']) if s else None


QUANTITIES = [('Pcr', lambda r: r['Pcr']),
              ('b_min_t', lambda r: r['b_min_t']),
              ('crit b_t', critical)]


def sequence(rows):
    print('  %4s %4s %6s %4s %6s %9s %3s %9s %9s %10s' % ('ny', 'nx',
          'dx max', 'n_c', 'ny/n_c', 'Pcr', 'm', 'b_min_t', 'crit b_t',
          'win1 b_t'))
    for r in rows:
        n_c = r['modes_n'][0]
        c = critical(r)
        w = window1(r)
        print('  %4d %4d %6.1f %4d %6.1f %9.2f %3d %9.5f %9s %10s' % (
              r['ny'], r['nx'], 1e3*r['dx_max'], n_c, r['ny']/n_c, r['Pcr'],
              r['koiter_num_modes'], r['b_min_t'],
              '%.5f' % c if c is not None else '-',
              '%.5f%s' % (w[0], '' if w[1] else '*') if w else '-'))
    xs = [r['ny'] for r in rows]
    for name, get in QUANTITIES:
        vals = [get(r) for r in rows]
        line = '  %-9s' % name
        for i in range(len(rows) - 2):
            tri = vals[i:i + 3]
            if None in tri:
                continue
            fit = cs.richardson(xs[i:i + 3], tri)
            line += ' | %d-%d-%d: ' % tuple(xs[i:i + 3])
            if fit is None:
                line += 'not monotone'
            else:
                line += 'p %.1f, %.5g, %+.1f %%' % (fit[0], fit[1],
                                                    pct(tri[2], fit[1]))
        print(line)


def fit_grid(nys, nxs, f):
    """f = f_inf + a/ny**p + c/nx**q, least squares in f_inf, a and c for
    every p and q of a grid, the pair of smallest residual kept; returns
    p, q, f_inf, a, c and the rms residual"""
    nys, nxs, f = [np.asarray(v, dtype=float) for v in (nys, nxs, f)]
    best = None
    for p in np.arange(0.5, 8.01, 0.05):
        for q in np.arange(0.5, 8.01, 0.05):
            A = np.column_stack([np.ones_like(f), nys**-p, nxs**-q])
            coef, *_ = np.linalg.lstsq(A, f, rcond=None)
            rms = np.sqrt(np.mean((A @ coef - f)**2))
            if best is None or rms < best[-1]:
                best = (p, q, *coef, rms)
    return best


def grid(rows, min_per_wave, min_F=2.):
    rows = [r for r in rows if r['ny']/r['modes_n'][0] >= min_per_wave
            and r['axial_factor'] >= min_F]
    print('\n  fit f = f_inf + a/ny**p + c/nx**q over the %d meshes with '
          'ny/n_c >= %g and F >= %g' % (len(rows), min_per_wave, min_F))
    if len(rows) < 5:
        print('  too few meshes')
        return
    for name, get in QUANTITIES:
        pts = [(r, get(r)) for r in rows if get(r) is not None]
        if len(pts) < 5:
            continue
        p, q, f_inf, a, c, rms = fit_grid([r['ny'] for r, _ in pts],
                                          [r['nx'] for r, _ in pts],
                                          [v for _, v in pts])
        print('  %-9s p %.2f q %.2f f_inf %.5g, rms residual %.2g (%.2f %%)'
              % (name, p, q, f_inf, rms, 100*rms/abs(f_inf)))
        print('    %4s %4s %4s %7s %10s %8s %8s %8s' % ('ny', 'F', 'nx',
              'nx*ny', 'value', 'error', 'ny part', 'nx part'))
        for r, v in sorted(pts, key=lambda t: t[0]['nx']*t[0]['ny']):
            print('    %4d %4g %4d %7d %10.5g %+7.2f%% %+7.2f%% %+7.2f%%' % (
                r['ny'], r['axial_factor'], r['nx'], r['nx']*r['ny'], v,
                pct(v, f_inf), 100*a*r['ny']**-p/abs(f_inf),
                100*c*r['nx']**-q/abs(f_inf)))


def main(min_per_wave=4.):
    runs = load()
    for case in [0, 1, 6]:
        rows = [r for r in runs if r['case'] == case]
        if not rows:
            continue
        print('\ncase %d NL, eps1 = 0.0005' % case)
        for F in sorted(set(r['axial_factor'] for r in rows)):
            seq = sorted([r for r in rows if r['axial_factor'] == F],
                         key=lambda r: r['ny'])
            print('\n case %d, F = %g' % (case, F))
            sequence(seq)
        grid(rows, min_per_wave)


if __name__ == '__main__':
    if '--archive' in sys.argv:
        archive(sys.argv[sys.argv.index('--archive') + 1])
    min_per_wave = 4.
    if '--min-per-wave' in sys.argv:
        min_per_wave = float(sys.argv[sys.argv.index('--min-per-wave') + 1])
    main(min_per_wave)
