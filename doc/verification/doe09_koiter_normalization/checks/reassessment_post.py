"""Tables of the reassessment studies of DOE09, generate_qsubs_reassess.py

From the RESULT lines, archived in results/DOE09_reassess_<study>.jsonl.gz
as the other studies, one {"file": ..., "result": ...} per run, result null
for a run that wrote none, and from results/DOE09_conv_k5g.jsonl.gz for the
reference runs. A cluster is the modes of one wave number n, complete when
both distinct modes of n, the symmetric and the antisymmetric one, are in
the Koiter set (cluster_subsets.py); its b_min_energy is that of the
sub-block of the energy normalized b_ijkl, and its crest_e and b_min_t, of
the prefix and window subsets of run_case.cluster_subsets, are in the
RESULT line of the new runs only.

- (a) truncation: b_min_energy and b_min_t of the full set against the
  number of clusters, for --distinct 5, 7 and 9, and of the prefix and window
  subsets of the largest run; a subset against the full set of the run with
  the same clusters; the weights of e_min on the clusters;
- (b) axial refinement at ny = 160: every complete cluster and the full set
  against the axial factor, with nx, dx max and Pcr, and the Richardson
  estimate against dx max;
- (c) expansion point at ny = 120: the same against lambda_b/lambda_c, with
  a linear fit and its value at lambda_b/lambda_c = 1;
- (d) ny = 240: every complete cluster at ny = 120 to 240, Richardson from
  the last three meshes, the meshes whose nx changed marked;
- (e) the NL mesh sequences of (b) and (d) again with eps1 = 0.0005, so
  that lambda_b/lambda_c does not vary between the meshes: every complete
  cluster, the critical cluster and the window n_c +- 1, Richardson in ny
  and in dx max. The ny = 120 runs of (e) are added to (c);
- (f) added to the sequences of (e): the axial factors 2.5, 4 and 5 of case
  1 at ny = 160, ny = 280 of case 0, and the candidate DOE mesh, ny = 240
  with factor 1.5, of cases 1 and 6, printed against ny = 240 at factor 1;
- (g) case 1 NL on a grid of ny and nx, with the runs of (e) and (f) at
  eps1 = 0.0005: the critical cluster at every (ny, nx), and a fit of
  f = f_inf + a/ny**p + c/nx**q to all of them, with the error of each
  mesh against f_inf and its number of elements;
- (h) case 0 NL at ny = 160 with 4, 5 and 6 integration points per
  direction;
- (i) the ny sequences of (e) of cases 0 and 6 with Donnell kinematics,
  against Sanders;
- (j) case 6 NL at thickness factors 0.7, 1 and 2, the mesh error of b
  against the elements per wave of the critical wave number, ny/n_c, and
  R/h;
- (k) the ny sequences of (e) of cases 0, 1 and 6 with v = w = 0 along the
  whole edge, --ss-full, against v = w = 0 at the edge nodes only.

usage, from doc/verification/doe09_koiter_normalization:

    python checks/reassessment_post.py [--archive DOE09_DIR]

--archive first writes the archives from the outputs
DOE09_reassess_<study>_*.out in DOE09_DIR
"""
import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cluster_subsets as cs

studies = ['smoke', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
           'l', 'm']
here = os.path.join(os.path.dirname(__file__), '..')
results_dir = os.path.join(here, 'results')


def archive(workdir):
    for study in studies:
        paths = sorted(glob.glob(os.path.join(workdir, 'DOE09_reassess_%s_*.out'
                                              % study)))
        if not paths:
            continue
        name = os.path.join(results_dir, 'DOE09_reassess_%s.jsonl.gz' % study)
        with gzip.open(name, 'wt') as f:
            for path in paths:
                result = None
                with open(path) as g:
                    for line in g:
                        if line.startswith('RESULT '):
                            result = json.loads(line[len('RESULT '):])
                f.write(json.dumps(dict(file=os.path.basename(path),
                                        result=result)) + '\n')
        print('# %d runs in %s' % (len(paths), name))


def load(name):
    path = os.path.join(results_dir, name)
    if not os.path.isfile(path):
        return []
    rows = [json.loads(l) for l in gzip.open(path, 'rt')]
    for row in rows:
        r = row['result']
        if r is None or 'error' in r:
            print('# %s: %s' % (row['file'], 'no RESULT' if r is None
                                else r['error']))
    return [row['result'] for row in rows
            if row['result'] and 'error' not in row['result']]


def pre(r):
    return 'NL' if r['NLprebuck'] else 'LIN'


def pct(a, b):
    """relative difference of a from b, in per cent"""
    return 100*(a - b)/abs(b)


def ns_label(r):
    """wave numbers of the Koiter set in its order, * for an incomplete
    cluster"""
    groups = cs.clusters(r)
    return ' '.join('%d%s' % (n, '' if cs.complete(r, idx) else '*')
                    for n, idx in groups.items())


def subset(r, kind, ns):
    for s in r.get('subsets', []):
        if s['kind'] == kind and s['ns'] == list(ns):
            return s
    return None


def critical(r):
    """window subset of the critical wave number alone, the critical cluster"""
    return subset(r, 'window', [r['modes_n'][0]])


def window1(r):
    """window subset n_c - 1 to n_c + 1"""
    n_c = r['modes_n'][0]
    return subset(r, 'window', [n_c - 1, n_c, n_c + 1])


_cluster_cache = {}


def cluster_b(r):
    key = id(r)
    if key not in _cluster_cache:
        _cluster_cache[key] = cs.cluster_b(r)
    return _cluster_cache[key]


def weights(r):
    """weight of e_min on every wave number, all of its modes"""
    e = np.array(r['e_min'])
    groups = {}
    for k, n in enumerate(r['modes_n'][:r['koiter_num_modes']]):
        groups.setdefault(n, []).append(k)
    return ', '.join('n=%d %.2f' % (n, (e[idx]**2).sum())
                     for n, idx in groups.items())


def table_a(runs):
    print('\n(a) truncation: full set of every --distinct K, and the subsets '
          'of the largest')
    keys = sorted({(pre(r), r['case'], r['ny']) for r in runs},
                  key=lambda k: (k[0] != 'NL', k[1], k[2]))
    summary = []
    for p, case, ny in keys:
        group = sorted([r for r in runs if (pre(r), r['case'], r['ny'])
                        == (p, case, ny)], key=lambda r: r['koiter_num_distinct'])
        print('\ncase %d %s ny=%d nx=%d' % (case, p, ny, group[0]['nx']))
        print('  %2s %3s %3s %-28s %10s %8s %10s %8s' % ('K', 'm', 'nc', 'clusters',
              'b_min_e', 'crest_e', 'b_min_t', 'gap'))
        for r in group:
            print('  %2d %3d %3d %-28s %10.5g %8.4f %10.5g %8.1e'
                  % (r['koiter_num_distinct'], r['koiter_num_modes'],
                     len(cs.clusters(r)), ns_label(r), r['b_min_energy'],
                     r['crest_e'], r['b_min_t'], r['koiter_gap']))
        for r in group:
            print('  weights of e_min, K=%d: %s' % (r['koiter_num_distinct'],
                                                    weights(r)))
        big = group[-1]
        for kind in ['prefix', 'window']:
            print('  %s subsets of K=%d:' % (kind, big['koiter_num_distinct']))
            prev = None
            for s in big['subsets']:
                if s['kind'] != kind:
                    continue
                step = ('' if prev is None else '%+6.1f %%'
                        % pct(s['b_min_t'], prev))
                print('    %-24s m=%2d complete=%d b_min_e %10.5g crest_e '
                      '%7.4f b_min_t %10.5g %s' % (
                          ' '.join(map(str, s['ns'])), s['num_modes'],
                          s['complete'], s['b_min_energy'], s['crest_e'],
                          s['b_min_t'], step))
                prev = s['b_min_t']
        #NOTE a subset of a larger run against the run on those clusters,
        #     which differ by the orthogonality conditions of the second order
        #     fields on the other Koiter modes
        for small in group[:-1]:
            for large in group:
                if large['koiter_num_distinct'] <= small['koiter_num_distinct']:
                    continue
                ns = set(cs.clusters(small))
                for s in large['subsets']:
                    if s['kind'] == 'prefix' and set(s['ns']) == ns:
                        print('  K=%d prefix %s against the full K=%d set: '
                              'b_min_e %.5g against %.5g (%+.2f %%), b_min_t '
                              '%.5g against %.5g (%+.2f %%)' % (
                                  large['koiter_num_distinct'],
                                  ' '.join(map(str, s['ns'])),
                                  small['koiter_num_distinct'],
                                  s['b_min_energy'], small['b_min_energy'],
                                  pct(s['b_min_energy'], small['b_min_energy']),
                                  s['b_min_t'], small['b_min_t'],
                                  pct(s['b_min_t'], small['b_min_t'])))
                        break
        #NOTE the same window across K
        for j in range(3):
            row = []
            for r in group:
                n_c = r['modes_n'][0]
                s = subset(r, 'window', range(n_c - j, n_c + j + 1))
                if s is not None:
                    row.append('K=%d %.5g/%.5g%s' % (
                        r['koiter_num_distinct'], s['b_min_energy'],
                        s['b_min_t'], '' if s['complete'] else '*'))
            if row:
                print('  window n_c +- %d, b_min_e/b_min_t: %s'
                      % (j, ', '.join(row)))
        summary.append((p, case, ny, group))
    return summary


def axial_rows(case, prebuck, runs_b, runs_a, k5g):
    """runs at ny = 160 for every axial factor, factor 1 from (a) if there,
    else from _k5g"""
    ref = [r for r in runs_a if r['case'] == case and pre(r) == prebuck
           and r['ny'] == 160 and r['koiter_num_distinct'] == 5]
    ref = ref or [r for r in k5g if r['case'] == case and pre(r) == prebuck
                  and r['ny'] == 160]
    rows = ref[:1] + [r for r in runs_b if r['case'] == case
                      and pre(r) == prebuck]
    return sorted(rows, key=lambda r: r.get('axial_factor', 1.))


def cluster_table(rows, label, extra=None):
    """rows of runs with b_min_energy of every complete cluster"""
    ns = sorted(set().union(*[cluster_b(r) for r in rows]))
    head = '  %-12s %4s %6s %8s %9s %3s %10s %10s %10s %10s' % (label, 'nx',
            'dx max', 'l/lc', 'Pcr', 'm', 'b_min_e', 'b_min_t', 'crit b_t',
            'win1 b_t')
    print(head + ''.join('%10s' % ('n=%d' % n) for n in ns))
    for r, lab in zip(rows, extra):
        c = critical(r)
        w = window1(r)
        cb = cluster_b(r)
        lam = r['lambda_b']/(r['Pcr']/(r.get('Nxxunit', 1000.)*2*np.pi*0.4))
        print('  %-12s %4d %6.2f %8.5f %9.2f %3d %10.5g %10.5g %10s %10s' % (
            lab, r['nx'], 1e3*r['dx_max'], lam, r['Pcr'],
            r['koiter_num_modes'], r['b_min_energy'], r['b_min_t'],
            '%.5g' % c['b_min_t'] if c else '-',
            '%.5g%s' % (w['b_min_t'], '' if w['complete'] else '*')
            if w else '-')
              + ''.join('%10s' % ('%.5g' % cb[n] if n in cb else '-')
                        for n in ns))
    return ns


def table_b(runs_b, runs_a, k5g):
    print('\n(b) axial refinement at ny = 160, b_min_energy of every complete '
          'cluster; crit b_t is b_min_t of the critical cluster, from the '
          'subsets')
    out = {}
    for case in [0, 1, 6]:
        for prebuck in ['LIN', 'NL']:
            rows = axial_rows(case, prebuck, runs_b, runs_a, k5g)
            if len(rows) < 2:
                continue
            print('\ncase %d %s' % (case, prebuck))
            ns = cluster_table(rows, 'F', ['%g' % r.get('axial_factor', 1.)
                                           for r in rows])
            for n in ns:
                pts = [(r['dx_max'], cluster_b(r)[n]) for r in rows
                       if n in cluster_b(r)]
                if len(pts) < 2 or n not in cluster_b(rows[0]):
                    continue
                f = [v for _, v in pts]
                line = '  n=%d: change from F=1 to the finest %+.2f %%' % (
                        n, pct(f[-1], f[0]))
                if len(pts) >= 3:
                    fit = cs.richardson([1/dx for dx, _ in pts[-3:]], f[-3:])
                    if fit:
                        line += (', Richardson in dx max, order %.1f, '
                                 'extrapolated %.5g, axial error at F=1 %+.2f %%'
                                 % (fit[0], fit[1], pct(f[0], fit[1])))
                    else:
                        line += ', not monotone'
                print(line)
            out[(case, prebuck)] = rows
    return out


def table_c(runs_c, runs_a, k5g):
    print('\n(c) expansion point at ny = 120, against lambda_b/lambda_c, '
          'linear fit and its value at 1')
    for case in [0, 1, 6]:
        ref = [r for r in runs_a if r['case'] == case and pre(r) == 'NL'
               and r['ny'] == 120 and r['koiter_num_distinct'] == 5]
        ref = ref or [r for r in k5g if r['case'] == case and r['NLprebuck']
                      and r['ny'] == 120]
        rows = ref[:1] + [r for r in runs_c if r['case'] == case]
        if len(rows) < 2:
            continue
        for r in rows:
            r.setdefault('lambda_ratio', r['lambda_b']/(r['Pcr']
                         /(r.get('Nxxunit', 1000.)*2*np.pi*0.4)))
        rows.sort(key=lambda r: r['lambda_ratio'])
        print('\ncase %d NL' % case)
        ns = cluster_table(rows, 'eps1 (l/lc)', ['%g %.5f' % (
                r.get('NLprebuck_eps1', 0.005), r['lambda_ratio']) for r in rows])
        x = np.array([r['lambda_ratio'] for r in rows])
        default = [r for r in rows if r.get('NLprebuck_eps1', 0.005) == 0.005][0]
        series = [('Pcr', [r['Pcr'] for r in rows], default['Pcr']),
                  ('b_min_e', [r['b_min_energy'] for r in rows],
                   default['b_min_energy']),
                  ('b_min_t', [r['b_min_t'] for r in rows], default['b_min_t'])]
        crit = [critical(r) for r in rows]
        if all(crit):
            series.append(('crit b_t', [c['b_min_t'] for c in crit],
                           critical(default)['b_min_t']))
        for n in ns:
            if all(n in cluster_b(r) for r in rows):
                series.append(('n=%d' % n, [cluster_b(r)[n] for r in rows],
                               cluster_b(default)[n]))
        for name, y, y0 in series:
            slope, at1 = np.polyfit(x - 1, y, 1)
            resid = np.array(y) - (slope*(x - 1) + at1)
            print('  %-9s at lambda_b/lambda_c = 1: %.5g, slope %.4g, rms '
                  'residual %.2g, default eps1 = 0.005 off by %+.2f %%'
                  % (name, at1, slope, np.sqrt((resid**2).mean()),
                     pct(y0, at1)))


def table_d(runs_d, runs_a, k5g):
    print('\n(d) ny = 120 to 240 with the setup of _k5g, * where nx changed '
          'from the previous mesh')
    for case in [0, 1, 6]:
        for prebuck in ['LIN', 'NL']:
            rows = [r for r in k5g if r['case'] == case and pre(r) == prebuck
                    and r['ny'] >= 120]
            rows += [r for r in runs_d if r['case'] == case
                     and pre(r) == prebuck]
            #NOTE ny = 120 and 160 of (a), K = 5, which have the subsets
            for i, r in enumerate(rows):
                for q in runs_a:
                    if (q['case'], pre(q), q['ny'], q['koiter_num_distinct']) \
                            == (case, prebuck, r['ny'], 5):
                        rows[i] = q
            rows.sort(key=lambda r: r['ny'])
            if not any(r['ny'] == 240 for r in rows):
                continue
            print('\ncase %d %s' % (case, prebuck))
            labels = []
            for i, r in enumerate(rows):
                changed = i > 0 and r['nx'] != rows[i - 1]['nx']
                labels.append('%d%s' % (r['ny'], '*' if changed else ''))
            ns = cluster_table(rows, 'ny', labels)
            nys = [r['ny'] for r in rows]
            for n in ns + ['full']:
                vals = {r['ny']: (r['b_min_t'] if n == 'full'
                                  else cluster_b(r).get(n)) for r in rows}
                line = '  %s:' % ('b_min_t full set' if n == 'full'
                                  else 'n=%d' % n)
                for tri in [(120, 160, 200), (160, 200, 240)]:
                    if not all(vals.get(ny) is not None for ny in tri):
                        continue
                    f = [vals[ny] for ny in tri]
                    fit = cs.richardson(tri, f)
                    if fit is None:
                        line += ' from %s not monotone;' % (tri,)
                    else:
                        line += (' from %s order %.1f, extrapolated %.5g, '
                                 'error %+.1f %% at 160, %+.1f %% at 200;'
                                 % (tri, fit[0], fit[1],
                                    pct(vals[160], fit[1]),
                                    pct(vals[200], fit[1])))
                print(line)


def fits(rows, xs, quantities):
    """Richardson of every quantity over every three consecutive rows,
    x = 1/h"""
    for name, get in quantities:
        vals = [get(r) for r in rows]
        pts = [(x, v) for x, v in zip(xs, vals) if v is not None]
        if len(pts) < 3:
            continue
        line = '  %-10s' % name
        for tri in [pts[i:i + 3] for i in range(len(pts) - 2)]:
            fit = cs.richardson([x for x, _ in tri], [v for _, v in tri])
            f = [v for _, v in tri]
            if fit is None:
                line += ' | %s: not monotone' % ', '.join('%.5g' % v for v in f)
            else:
                line += (' | %s: order %.1f, extrapolated %.5g, error %+.1f %% '
                         'and %+.1f %%' % (', '.join('%.5g' % v for v in f),
                         fit[0], fit[1], pct(f[1], fit[1]), pct(f[2], fit[1])))
        print(line)


def table_e(runs_e):
    print('\n(e) and (f), eps1 = 0.0005, NL: ny = 120 to 280 at F = 1, the '
          'axial factors at ny = 160, and ny = 240 at F = 1.5; win1 is the '
          'window n_c +- 1, * incomplete')
    for case in [0, 1, 6]:
        rows = [r for r in runs_e if r['case'] == case]
        if not rows:
            continue
        ns = sorted(set().union(*[cluster_b(r) for r in rows]))
        quantities = ([('crit b_t', lambda r: critical(r)['b_min_t']
                        if critical(r) else None),
                       ('win1 b_t', lambda r: window1(r)['b_min_t']
                        if window1(r) and window1(r)['complete'] else None),
                       ('b_min_t', lambda r: r['b_min_t'])]
                      + [('n=%d' % n, lambda r, n=n: cluster_b(r).get(n))
                         for n in ns])
        circ = sorted([r for r in rows if r['axial_factor'] == 1.],
                      key=lambda r: r['ny'])
        print('\ncase %d NL, ny at F = 1, Richardson in ny, errors at the '
              'middle and last of each three' % case)
        labels = ['%d%s' % (r['ny'], '*' if i and r['nx'] != circ[i - 1]['nx']
                            else '') for i, r in enumerate(circ)]
        cluster_table(circ, 'ny', labels)
        fits(circ, [r['ny'] for r in circ], quantities)
        ax = sorted([r for r in rows if r['ny'] == 160],
                    key=lambda r: r['axial_factor'])
        if len(ax) > 1:
            print('case %d NL, axial factor at ny = 160, Richardson in 1/dx max'
                  % case)
            cluster_table(ax, 'F', ['%g' % r['axial_factor'] for r in ax])
            fits(ax, [1/r['dx_max'] for r in ax], quantities)
        other = sorted([r for r in rows if r['ny'] == 240],
                       key=lambda r: r['axial_factor'])
        if len(other) > 1:
            print('case %d NL, ny = 240, axial factor 1 and the candidate DOE '
                  'mesh' % case)
            cluster_table(other, 'F', ['%g' % r['axial_factor']
                                       for r in other])


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


def table_g(runs):
    print('\n(g) case 1 NL, eps1 = 0.0005, grid of ny and nx from (e), (f) '
          'and (g); fit f = f_inf + a/ny**p + c/nx**q')
    rows = sorted([r for r in runs if r['case'] == 1],
                  key=lambda r: (r['ny'], r['nx']))
    if len(rows) < 5:
        return
    n_c = rows[-1]['modes_n'][0]
    cluster_table(rows, 'ny/F', ['%d/%g' % (r['ny'], r['axial_factor'])
                                 for r in rows])
    quantities = [('crit b_t', lambda r: critical(r)['b_min_t']
                   if critical(r) else None),
                  ('n=%d' % n_c, lambda r: cluster_b(r).get(n_c))]
    for name, get in quantities:
        pts = [(r, get(r)) for r in rows if get(r) is not None]
        p, q, f_inf, a, c, rms = fit_grid([r['ny'] for r, _ in pts],
                                          [r['nx'] for r, _ in pts],
                                          [v for _, v in pts])
        print('  %-10s p %.2f q %.2f f_inf %.5g, rms residual %.2g (%.2f %%)'
              % (name, p, q, f_inf, rms, 100*rms/abs(f_inf)))
        print('    %4s %4s %7s %10s %8s %8s %8s' % ('ny', 'nx', 'nx*ny',
              'value', 'error', 'ny part', 'nx part'))
        for r, v in sorted(pts, key=lambda t: t[0]['nx']*t[0]['ny']):
            print('    %4d %4d %7d %10.5g %+7.2f%% %+7.2f%% %+7.2f%%' % (
                r['ny'], r['nx'], r['nx']*r['ny'], v, pct(v, f_inf),
                100*a*r['ny']**-p/abs(f_inf), 100*c*r['nx']**-q/abs(f_inf)))


def sequence_quantities(rows):
    ns = sorted(set().union(*[cluster_b(r) for r in rows]))
    return ([('crit b_t', lambda r: critical(r)['b_min_t']
              if critical(r) else None),
             ('crit b_e', lambda r: cluster_b(r).get(r['modes_n'][0]))]
            + [('n=%d' % n, lambda r, n=n: cluster_b(r).get(n)) for n in ns])


def table_h(runs):
    print('\n(h) case 0 NL, ny = 160, eps1 = 0.0005, integration points per '
          'direction')
    rows = sorted(runs, key=lambda r: r.get('nint', 4))
    if len(rows) < 2:
        return
    cluster_table(rows, 'nint', ['%d' % r.get('nint', 4) for r in rows])
    for name, get in sequence_quantities(rows):
        vals = [get(r) for r in rows]
        if None in vals:
            continue
        print('  %-10s %s' % (name, ', '.join('%+.3f %%' % pct(v, vals[-1])
                                            for v in vals[:-1])
                              + ' from the largest nint'))


def table_i(runs):
    print('\n(i) Donnell against Sanders kinematics, NL, eps1 = 0.0005, '
          'Richardson in ny')
    for case in [0, 6]:
        for kin in ['sanders', 'donnell']:
            rows = sorted([r for r in runs if r['case'] == case
                           and r.get('kinematics', 'sanders') == kin],
                          key=lambda r: r['ny'])
            if not rows:
                continue
            print('\ncase %d NL, %s' % (case, kin))
            cluster_table(rows, 'ny', ['%d' % r['ny'] for r in rows])
            fits(rows, [r['ny'] for r in rows], sequence_quantities(rows))


def table_j(runs):
    print('\n(j) case 6 NL, eps1 = 0.0005, thickness factor T, Richardson in '
          'ny; ny/n_c elements per wave of the critical wave number')
    #NOTE lambda_b/lambda_c above 1 is an expansion point past the
    #     bifurcation, where the load stepping started above Pcr
    for r in runs:
        if r['lambda_ratio'] > 1:
            print('# left out, lambda_b/lambda_c %.4f: T = %g, ny = %d, '
                  'Nxxunit %g' % (r['lambda_ratio'], r['thickness_factor'],
                                  r['ny'], r.get('Nxxunit', 1000.)))
    runs = [r for r in runs if r['lambda_ratio'] <= 1]
    for T in sorted(set(r.get('thickness_factor', 1.) for r in runs)):
        rows = sorted([r for r in runs
                       if r.get('thickness_factor', 1.) == T],
                      key=lambda r: r['ny'])
        print('\nT = %g, n_c %s' % (T, ', '.join('%d' % r['modes_n'][0]
                                                for r in rows)))
        cluster_table(rows, 'ny (ny/n_c)', ['%d (%.1f)' % (r['ny'],
                      r['ny']/r['modes_n'][0]) for r in rows])
        fits(rows, [r['ny'] for r in rows], sequence_quantities(rows))


def table_k(runs):
    print('\n(k) simply supported edges, v = w = 0 at the edge nodes only '
          '(nodes) or along the whole edge (edge), NL, eps1 = 0.0005, '
          'Richardson in ny')
    for case in [0, 1, 6]:
        for ss in [False, True]:
            rows = sorted([r for r in runs if r['case'] == case
                           and r.get('ss_edge_tangential', False) == ss],
                          key=lambda r: r['ny'])
            if not rows:
                continue
            print('\ncase %d NL, %s' % (case, 'edge' if ss else 'nodes'))
            cluster_table(rows, 'ny', ['%d' % r['ny'] for r in rows])
            fits(rows, [r['ny'] for r in rows], sequence_quantities(rows))


def table_l(runs):
    print('\n(l) four more designs of the DOE, NL, ny = 160 to 240, axial '
          'factor 1.5, eps1 = 0.0005, Nxxunit 500, Richardson in ny')
    for case in sorted(set(r['case'] for r in runs)):
        rows = sorted([r for r in runs if r['case'] == case],
                      key=lambda r: r['ny'])
        print('\ncase %d NL, n_c %s' % (case, ', '.join(
              '%d' % r['modes_n'][0] for r in rows)))
        cluster_table(rows, 'ny (ny/n_c)', ['%d (%.1f)' % (r['ny'],
                      r['ny']/r['modes_n'][0]) for r in rows])
        fits(rows, [r['ny'] for r in rows], sequence_quantities(rows))


def table_m(runs):
    print('\n(m) SS3 and SS4 edges from a low ny, NL, eps1 = 0.0005, '
          'Richardson in ny; ny/n_c elements per wave of the critical wave '
          'number')
    for case in [0, 1, 6]:
        for edges in ['SS3', 'SS4']:
            rows = sorted([r for r in runs if r['case'] == case
                           and r.get('edges', 'SS3') == edges],
                          key=lambda r: r['ny'])
            if not rows:
                continue
            print('\ncase %d NL, %s' % (case, edges))
            cluster_table(rows, 'ny (ny/n_c)', ['%d (%.1f)' % (r['ny'],
                          r['ny']/r['modes_n'][0]) for r in rows])
            fits(rows, [r['ny'] for r in rows], sequence_quantities(rows))


def main():
    k5g = load('DOE09_conv_k5g.jsonl.gz')
    runs = {s: load('DOE09_reassess_%s.jsonl.gz' % s) for s in studies}
    table_a(runs['a'])
    table_b(runs['b'], runs['a'], k5g)
    table_c(runs['c'] + [r for r in runs['e'] if r['ny'] == 120
                         and r['axial_factor'] == 1.], runs['a'], k5g)
    table_d(runs['d'], runs['a'], k5g)
    table_e(runs['e'] + runs['f'])
    table_g([r for r in runs['e'] + runs['f'] + runs['g']
             if r['NLprebuck_eps1'] == 0.0005])
    ref = [r for r in runs['e'] if r['axial_factor'] == 1.]
    table_h(runs['h'] + [r for r in ref if r['case'] == 0
                         and r['ny'] == 160])
    table_i(runs['i'] + [r for r in ref if r['ny'] >= 120])
    table_j(runs['j'] + [r for r in ref if r['case'] == 6
                         and r['ny'] >= 160])
    table_k(runs['k'] + [r for r in ref if r['ny'] >= 120])
    table_l(runs['l'])
    #NOTE the SS3 rows of (m) continue those of (k), same setup
    table_m(runs['m'] + runs['k'])


if __name__ == '__main__':
    if '--archive' in sys.argv:
        archive(sys.argv[sys.argv.index('--archive') + 1])
    main()
