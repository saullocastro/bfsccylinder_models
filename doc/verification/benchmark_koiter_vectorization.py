"""Cost and results of the Koiter tensors before and after their vectorization

Runs the same models with two checkouts of bfsccylinder_models, each in its
own subprocess with PYTHONPATH pointing at it and one BLAS thread, and
compares what they return and how long they take. Two kinds of cases:

- DOE09 designs of koiter_cylinder_CTS_sanders.py, built with
  run_case.design_function of the DOE, with its solvers (use_safe_solvers:
  PARDISO Cholesky, scaled PARDISO LU with GMRES, SuperLU fallback) and its
  Koiter modes distinct up to the rotation of the cylinder
  (use_distinct_modes). pypardiso and mkl must be importable from --pypardiso,
  a directory made with `pip install --target DIR pypardiso`, put on the path
  of these runs only;
- the Waters shell of tests/test_koiter_cylinder_Waters_sanders.py with
  koiter_cylinder_sanders.py, solved with SciPy's SuperLU, --pypardiso being
  left out of its path, as the library would pick pypardiso up by itself.

The Koiter time is reported twice: as t(m) - t(0), the total time with m
Koiter modes less the total time without the Koiter section, and as measured
directly, from the moment the model prints its critical buckling load, which
is the last thing it does before the Koiter section, to its return.

usage:
    python benchmark_koiter_vectorization.py run [options]
        runs every configuration not yet in --results, resumable
    python benchmark_koiter_vectorization.py report [--results DIR]
        prints the tables of doc/verification/README.md
    python benchmark_koiter_vectorization.py worker ...
        one run, used by `run`

The default checkouts are this repository and ../bfsccylinder_models_baseline
next to it, made with
    git worktree add ../bfsccylinder_models_baseline 3251981
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
BASELINE = os.path.abspath(os.path.join(REPO, os.pardir,
                                        'bfsccylinder_models_baseline'))
PYPARDISO = os.path.abspath(os.path.join(REPO, os.pardir, 'pypardiso_site'))
DOE_DIR = os.path.join(os.path.expanduser('~'), 'OneDrive - Delft University '
        'of Technology', 'author', '2026_paper_Rogerio', 'DOE09')
RESULTS = os.path.join(HERE, 'benchmark_koiter_vectorization')

MODES = [0, 1, 2, 5, 8]


#NOTE ARPACK stops with error -8, dsteqr failing on the Lanczos tridiagonal
#     matrix, for 20 eigenvalues on these meshes, in both versions, the
#     eigenvalue analysis being the same; the first of 16, 24 and 12 that
#     does not is used instead. With 12, fewer than 8 of the Koiter modes are
#     distinct up to the rotation of the cylinder, which num_distinct of the
#     results records; the Waters runs do not ask for distinct modes at all
EIGVALS_M8 = {('doe', 0, 'LIN', 80): 24, ('doe', 0, 'NL', 40): 12,
              ('doe', 1, 'NL', 40): 16, ('doe', 1, 'NL', 80): 16,
              ('waters', -1, 'LIN', 120): 12}


def num_eigvals(kind, case, prebuck, ny, m):
    """12 as in the DOE, 20 for 8 modes distinct up to the rotation of the
    cylinder, which need up to 16 eigenvectors"""
    if m <= 5:
        return 12
    return EIGVALS_M8.get((kind, case, prebuck, ny), 20)


def configurations(quick=False):
    """(kind, case, prebuck, ny, m, neig) of every run of the benchmark

    Every m > 0 comes with an m = 0 run at the same num_eigvals, so that
    t(m) - t(0) does not include a difference in the eigenvalue analysis
    """
    confs = []
    doe = [(0, 40), (1, 40), (6, 40)] if quick else [
            (case, ny) for ny in [40, 80] for case in [0, 1, 6]]
    const = [(-1, 60)] if quick else [(-1, 60), (-1, 120)]
    for kind, cases in [('doe', doe), ('waters', const)]:
        for case, ny in cases:
            for prebuck in ['LIN', 'NL']:
                neigs = [num_eigvals(kind, case, prebuck, ny, m)
                         for m in MODES]
                for neig in sorted(set(neigs[1:])):
                    confs.append((kind, case, prebuck, ny, 0, neig))
                for m, neig in zip(MODES[1:], neigs[1:]):
                    confs.append((kind, case, prebuck, ny, m, neig))
    return confs


def conf_name(version, conf, repeat):
    kind, case, prebuck, ny, m, neig = conf
    return ('%s_%s_%d_%s_ny%03d_m%d_neig%02d_r%d'
            % (version, kind, case, prebuck, ny, m, neig, repeat))


# ---------------------------------------------------------------- worker ---

class Clock:
    """stdout that remembers when the model printed its buckling load"""
    def __init__(self, stream):
        self.stream = stream
        self.t_pcr = None

    def write(self, text):
        if self.t_pcr is None and '# critical buckling load' in text:
            self.t_pcr = time.perf_counter()
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()


def peak_memory_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
    except ImportError:
        import psutil
        return psutil.Process().memory_info().peak_wset/1e9


def run_doe(case, prebuck, ny, m, neig, doe_dir):
    sys.path.insert(0, doe_dir)
    import run_case
    run_case.koiter_num_modes = m
    run_case.num_eigvals = neig
    v1, v2, v3, v4, v5 = np.loadtxt(os.path.join(doe_dir, 'DOE09.txt'),
                                    skiprows=1)[case]
    variables = dict(rCTS=v1, param_n=int(v2), c2_ratio=v3, thetadeg_c1=v4,
                     thetadeg_c2=v5)
    constants = dict(L=1.2, R=0.4, ny=ny, E11=122e9, E22=7.32e9, nu12=0.31,
                     G12=4.9e9, tow_thick=0.13e-3, rho=1540, mesh_only=False,
                     Nxxunit=1000., NLprebuck=(prebuck == 'NL'))
    solvers = run_case.use_safe_solvers()
    num_distinct = run_case.use_distinct_modes()
    assert solvers == 'pardiso', 'pypardiso not found, see --pypardiso'
    t0 = time.perf_counter()
    out = run_case.design_function(variables, constants)
    t1 = time.perf_counter()
    info = dict(num_distinct=num_distinct[0], solvers=solvers,
                library=run_case.model.__file__)
    return out, t0, t1, info


def run_waters(prebuck, ny, m, neig):
    """The shell of tests/test_koiter_cylinder_Waters_sanders.py"""
    from composites import laminated_plate
    import bfsccylinder_models.koiter_cylinder_sanders as model
    assert model.spsolve.__module__.startswith('scipy'), (
            'pypardiso on the path of a Waters run')
    L = 0.3556
    R = 0.20318603
    laminaprop = (127.629e9, 11.3074e9, 0.300235, 6.00257e9, 6.00257e9,
                  6.00257e9)
    prop = laminated_plate(stack=[45, -45, 0, 90, 90, 0, -45, 45],
            laminaprop=laminaprop, plyt=0.00012692375, offset=0, rho=1611)
    nx = int(ny*L/(2*np.pi*R))
    if nx % 2 == 0:
        nx += 1
    t0 = time.perf_counter()
    out = model.fkoiter_cyl_SS3(L, R, nx, ny, prop, num_eigvals=neig,
            koiter_num_modes=m, Nxxunit=1., NLprebuck=(prebuck == 'NL'))
    t1 = time.perf_counter()
    out['nx'] = nx
    out['ny'] = ny
    return out, t0, t1, dict(solvers='superlu', library=model.__file__)


def time_calls(module, name, clock, timings):
    """Wrap module.name so that the time spent in it after the buckling load
    was printed, which is the Koiter section, adds up in timings[name]"""
    func = getattr(module, name)
    timings[name] = 0.

    @wraps(func)
    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            if clock.t_pcr is not None:
                timings[name] += time.perf_counter() - t0

    setattr(module, name, timed)


def worker(args):
    clock = Clock(sys.stdout)
    sys.stdout = clock
    #NOTE the element loop (new version only) and the bordered solves, which
    #     are all the solves of the Koiter section. For the DOE runs spsolve
    #     is wrapped after use_safe_solvers replaced it, see run_doe
    timings = {}
    if args.kind == 'doe':
        sys.path.insert(0, args.doe_dir)
        import run_case
        model = run_case.model
        use_safe_solvers = run_case.use_safe_solvers

        def patched():
            out = use_safe_solvers()
            time_calls(model, 'spsolve', clock, timings)
            return out
        run_case.use_safe_solvers = patched
    else:
        import bfsccylinder_models.koiter_cylinder_sanders as model
        time_calls(model, 'spsolve', clock, timings)
    if hasattr(model, 'koiter_element_tensors'):
        time_calls(model, 'koiter_element_tensors', clock, timings)
    if args.kind == 'doe':
        out, t0, t1, info = run_doe(args.case, args.prebuck, args.ny, args.m,
                                    args.neig, args.doe_dir)
    else:
        out, t0, t1, info = run_waters(args.prebuck, args.ny, args.m,
                                       args.neig)
    sys.stdout = clock.stream
    m = args.m
    res = dict(info, kind=args.kind, case=args.case, prebuck=args.prebuck,
               ny=args.ny, m=m, neig=args.neig,
               num_elements=(out['nx'] - 1)*out['ny'],
               num_dof=int(out['eigvecs'].shape[0]),
               time_total=t1 - t0,
               time_koiter=t1 - clock.t_pcr,
               time_bordered_solves=timings.get('spsolve'),
               time_element_loop=timings.get('koiter_element_tensors'),
               peak_mem_gb=peak_memory_gb(),
               Pcr=float(out['Pcr']),
               lambda_b=float(out['lambda_b']),
               load_mult=[float(v) for v in out['load_mult']])
    if m > 0:
        k = out['koiter']
        res.update(
            num_cond=len(k['ucond']),
            a_ijk=[[[float(k['a_ijk'][(i, j, l)]) for l in range(m)]
                    for j in range(m)] for i in range(m)],
            b_ijkl=[[[[float(k['b_ijkl'][(i, j, l, n)]) for n in range(m)]
                      for l in range(m)] for j in range(m)] for i in range(m)])
    with open(args.out, 'w') as f:
        json.dump(res, f)
    if args.save_fields and m > 0:
        np.savez(args.save_fields, **{'u%d%d' % key: v
                 for key, v in out['koiter']['uij'].items()})


# ------------------------------------------------------------------- run ---

def launch(version, lib, conf, repeat, args, save_fields=False):
    name = conf_name(version, conf, repeat)
    path = os.path.join(args.results, name + '.json')
    if os.path.isfile(path):
        return name, 'done'
    kind, case, prebuck, ny, m, neig = conf
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1')
    pythonpath = [lib] + ([args.pypardiso] if kind == 'doe' else [])
    env['PYTHONPATH'] = os.pathsep.join(pythonpath)
    cmd = [sys.executable, '-u', os.path.abspath(__file__), 'worker',
           '--kind', kind, '--case', str(case), '--prebuck', prebuck,
           '--ny', str(ny), '--m', str(m), '--neig', str(neig),
           '--doe-dir', args.doe_dir, '--out', path + '.tmp']
    if save_fields:
        cmd += ['--save-fields', os.path.join(args.results, name + '.npz')]
    t0 = time.time()
    with open(os.path.join(args.results, name + '.log'), 'w') as log:
        #NOTE from the results directory, so that neither checkout is on
        #     sys.path through the working directory
        ret = subprocess.call(cmd, env=env, stdout=log,
                              stderr=subprocess.STDOUT, cwd=args.results)
    if ret == 0:
        os.replace(path + '.tmp', path)
        status = 'ok %.0f s' % (time.time() - t0)
    else:
        status = 'FAILED (%d), see %s.log' % (ret, name)
    print(name, status, flush=True)
    return name, status


def estimated_seconds(version, conf):
    """Rough cost, to start the longest runs first"""
    kind, case, prebuck, ny, m, neig = conf
    elements = {0: 2400, 1: 1800, 6: 960, -1: 960}[case]*(ny/(
            60. if kind == 'waters' else 40.))**1.8
    per_element = {0: 0, 1: 0.03, 2: 0.08, 5: 0.7, 8: 3.5}[m]
    if version == 'new':
        per_element = 0.002*(m > 0)
    return 10*(ny/40.)**2*(2 if prebuck == 'NL' else 1) + per_element*elements


def run(args):
    os.makedirs(args.results, exist_ok=True)
    tasks = []
    for conf in configurations(args.quick):
        kind, case, prebuck, ny, m, neig = conf
        if args.m8_exceptions and (m not in (0, 8) or
                EIGVALS_M8.get((kind, case, prebuck, ny)) != neig):
            continue
        for version, lib in [('base', args.baseline), ('new', args.new)]:
            if version not in args.versions:
                continue
            #NOTE the old element loop costs 3.5 s per element with 8 modes,
            #     5 to 8 hours per run on these meshes, and is exactly linear
            #     in the number of elements, so for m = 8 the baseline is run
            #     on the coarse meshes only and compared per element
            if (version == 'base' and m == 8
                    and ny > (60 if kind == 'waters' else 40)):
                continue
            repeats = args.repeats
            #NOTE a baseline run longer than --max-baseline-hours is made
            #     once only, its Koiter time being far above the noise
            if (version == 'base' and estimated_seconds(version, conf)
                    > 3600*args.max_baseline_hours):
                repeats = 1
            for repeat in range(repeats):
                tasks.append((version, lib, conf, repeat))
    tasks.sort(key=lambda t: -estimated_seconds(t[0], t[2]))
    print('# %d runs, %d jobs in parallel' % (len(tasks), args.jobs))
    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(lambda t: launch(*t, args), tasks))


# ---------------------------------------------------------------- fields ---

FIELD_CONFS = [(kind, case, prebuck, ny, m, 12)
               for kind, case, ny, ms in [('doe', 6, 40, [2, 5]),
                                          ('waters', -1, 60, [2])]
               for m in ms for prebuck in ['LIN', 'NL']]


def fields(args):
    """uij = uji, the second order fields solved for once

    The baseline solves the bordered system for every (i, j), the new version
    for i <= j only. Reported: how far apart the baseline's own uij and uji
    are, and how far the new uij are from the baseline's, both against the
    largest |uij|
    """
    os.makedirs(args.results, exist_ok=True)
    tasks = [(version, lib, conf, 0) for conf in FIELD_CONFS
             for version, lib in [('base', args.baseline), ('new', args.new)]]
    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(lambda t: launch(*t, args, save_fields=True), tasks))
    print('\n| case | pre-buckling | ny | m | baseline uij against uji | '
          'new uij against baseline uij | b_ijkl new against baseline |')
    print('|---|---|---|---|---|---|---|')
    for conf in FIELD_CONFS:
        kind, case, prebuck, ny, m, neig = conf
        u = {}
        b = {}
        for version in ['base', 'new']:
            name = os.path.join(args.results, conf_name(version, conf, 0))
            with np.load(name + '.npz') as f:
                u[version] = {(i, j): f['u%d%d' % (i, j)] for i in range(m)
                              for j in range(m)}
            with open(name + '.json') as f:
                b[version] = json.load(f)['b_ijkl']
        scale = max(np.abs(v).max() for v in u['base'].values())
        asym = max(np.abs(u['base'][(i, j)] - u['base'][(j, i)]).max()
                   for i in range(m) for j in range(i))/scale
        diff = max(np.abs(u['new'][key] - u['base'][key]).max()
                   for key in u['base'])/scale
        print('| %s | %s | %d | %d | %.1e | %.1e | %.1e |' % (
              'DOE09 %d' % case if kind == 'doe' else 'Waters', prebuck, ny,
              m, asym, diff, rel_diff(b['new'], b['base'])))


# ---------------------------------------------------------------- report ---

def load(results):
    runs = {}
    for name in sorted(os.listdir(results)):
        if name.endswith('.json'):
            with open(os.path.join(results, name)) as f:
                res = json.load(f)
            version = name.split('_')[0]
            key = (res['kind'], res['case'], res['prebuck'], res['ny'],
                   res['m'], res['neig'])
            runs.setdefault((version, key), []).append(res)
    return runs


def rel_diff(new, base):
    new = np.asarray(new, dtype=float)
    base = np.asarray(base, dtype=float)
    return float(np.abs(new - base).max()/np.abs(base).max())


def report(args):
    runs = load(args.results)
    med = lambda rs, q: float(np.median([r[q] for r in rs]))
    keys = sorted(set(k for (v, k) in runs if k[4] > 0))

    print('\n### Results, new against baseline\n')
    print('max relative difference, against the largest |value| of each '
          'quantity; a_ijk also as the largest |a_ijk| of the baseline\n')
    print('| case | pre-buckling | ny | m | load_mult | Pcr | a_ijk | '
          'max abs a_ijk | b_ijkl | null space vectors |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    worst = {}
    for key in keys:
        if ('base', key) not in runs or ('new', key) not in runs:
            continue
        b = runs[('base', key)][0]
        n = runs[('new', key)][0]
        d = dict(load_mult=rel_diff(n['load_mult'], b['load_mult']),
                 Pcr=abs(n['Pcr'] - b['Pcr'])/abs(b['Pcr']),
                 a_ijk=rel_diff(n['a_ijk'], b['a_ijk']),
                 b_ijkl=rel_diff(n['b_ijkl'], b['b_ijkl']))
        for q, v in d.items():
            worst[q] = max(worst.get(q, 0), v)
        kind, case, prebuck, ny, m, neig = key
        print('| %s | %s | %d | %d | %.1e | %.1e | %.1e | %.1e | %.1e | %s |'
              % ('DOE09 %d' % case if kind == 'doe' else 'Waters', prebuck,
                 ny, m, d['load_mult'], d['Pcr'], d['a_ijk'],
                 np.abs(b['a_ijk']).max(), d['b_ijkl'],
                 '%d = %d' % (b['num_cond'], n['num_cond'])
                 if b['num_cond'] == n['num_cond']
                 else '%d != %d' % (b['num_cond'], n['num_cond'])))
    print('\nworst:', ', '.join('%s %.1e' % kv for kv in worst.items()))

    print('\n### Time and peak memory, median of the repeats\n')
    print('Koiter time: t(m) - t(0) / measured directly; per element from '
          'the direct measurement\n')
    print('| case | pre-buckling | ny | elements | m | total base (s) | '
          'total new (s) | Koiter base (s) | Koiter new (s) | '
          'Koiter/element base (ms) | Koiter/element new (ms) | speedup '
          'Koiter | speedup total | peak mem base (GB) | peak mem new (GB) |'
          ' repeats base/new |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for key in keys:
        kind, case, prebuck, ny, m, neig = key
        key0 = (kind, case, prebuck, ny, 0, neig)
        row = []
        for version in ['base', 'new']:
            rs = runs.get((version, key))
            rs0 = runs.get((version, key0))
            if not rs or not rs0:
                row.append(None)
                continue
            t = med(rs, 'time_total')
            row.append(dict(t=t, dk=t - med(rs0, 'time_total'),
                            k=med(rs, 'time_koiter'),
                            mem=med(rs, 'peak_mem_gb'), n=len(rs),
                            ne=rs[0]['num_elements']))
        b, n = row
        if b is None and n is None:
            continue
        ne = (b or n)['ne']
        f = lambda r, q, fmt: fmt % r[q] if r else '-'
        print('| %s | %s | %d | %d | %d | %s | %s | %s | %s | %s | %s | %s |'
              ' %s | %s | %s | %s/%s |' % (
              'DOE09 %d' % case if kind == 'doe' else 'Waters', prebuck, ny,
              ne, m, f(b, 't', '%.0f'), f(n, 't', '%.1f'),
              '%.0f / %.0f' % (b['dk'], b['k']) if b else '-',
              '%.1f / %.1f' % (n['dk'], n['k']) if n else '-',
              '%.0f' % (1e3*b['k']/ne) if b else '-',
              '%.2f' % (1e3*n['k']/ne) if n else '-',
              '%.0f' % (b['k']/n['k']) if b and n else '-',
              '%.1f' % (b['t']/n['t']) if b and n else '-',
              f(b, 'mem', '%.2f'), f(n, 'mem', '%.2f'),
              b['n'] if b else 0, n['n'] if n else 0))

    print('\n### Scaling of the Koiter time with m, new version\n')
    print('direct Koiter time per element (ms), median over the cases\n')
    print('| ny | pre-buckling | ' + ' | '.join('m=%d' % m for m in MODES[1:])
          + ' |')
    print('|---|---|' + '---|'*len(MODES[1:]))
    for kind in ['doe', 'waters']:
        for ny in sorted(set(k[3] for k in keys if k[0] == kind)):
            for prebuck in ['LIN', 'NL']:
                cells = []
                for m in MODES[1:]:
                    vals = [med(rs, 'time_koiter')/rs[0]['num_elements']
                            for (v, k), rs in runs.items()
                            if v == 'new' and k[0] == kind and k[3] == ny
                            and k[2] == prebuck and k[4] == m]
                    cells.append('%.2f' % (1e3*np.median(vals))
                                 if vals else '-')
                print('| %s %d | %s | %s |' % ('DOE09' if kind == 'doe'
                      else 'Waters', ny, prebuck, ' | '.join(cells)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--baseline', default=BASELINE)
    p.add_argument('--new', default=REPO)
    p.add_argument('--pypardiso', default=PYPARDISO)
    p.add_argument('--doe-dir', default=DOE_DIR)
    p.add_argument('--results', default=RESULTS)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--max-baseline-hours', type=float, default=1.)
    p.add_argument('--versions', nargs='+', default=['base', 'new'])
    p.add_argument('--quick', action='store_true',
                   help='DOE09 at ny=40 and the Waters shell at ny=60 only')
    p.add_argument('--m8-exceptions', action='store_true',
                   help='only the m=8 runs of EIGVALS_M8 and their m=0 runs')
    p = sub.add_parser('report')
    p.add_argument('--results', default=RESULTS)
    p = sub.add_parser('worker')
    p.add_argument('--kind', choices=['doe', 'waters'], required=True)
    p.add_argument('--case', type=int, required=True)
    p.add_argument('--prebuck', choices=['LIN', 'NL'], required=True)
    p.add_argument('--ny', type=int, required=True)
    p.add_argument('--m', type=int, required=True)
    p.add_argument('--neig', type=int, required=True)
    p.add_argument('--doe-dir', default=DOE_DIR)
    p.add_argument('--out', required=True)
    p.add_argument('--save-fields', default=None,
                   help='.npz to save the second order fields uij in')
    p = sub.add_parser('fields')
    p.add_argument('--baseline', default=BASELINE)
    p.add_argument('--new', default=REPO)
    p.add_argument('--pypardiso', default=PYPARDISO)
    p.add_argument('--doe-dir', default=DOE_DIR)
    p.add_argument('--results', default=RESULTS + '_fields')
    p.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    dict(run=run, report=report, worker=worker, fields=fields)[args.command](
            args)


if __name__ == '__main__':
    main()
