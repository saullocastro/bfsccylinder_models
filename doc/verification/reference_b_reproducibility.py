"""Is b_1111 reproducible, and what does the column border add to it?

- b_1111 of Sun et al. 3.1, ny=40, as in the test suite, computed in two
  processes, one with a single BLAS thread and one with the default number
- b_1111 of AW-CYL-1-1, ny=60, as the model computes it and with the
  degenerate partner left out of the column border of the bordered system,
  which then holds the critical mode alone, as it used to. The eigen solver
  does not return the partner on that mesh, the model rebuilds it

Reported in: Sections "Fixing the member" and "The bordered system".
Runtime: about 7 minutes.
"""

import contextlib
import io
import os
import re
import subprocess
import sys
#NOTE the repository root goes FIRST on sys.path: an installed
#     bfsccylinder_models would otherwise shadow the working tree, silently
#     verifying a different version of the code than the one being edited
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir))

import numpy as np
from composites import laminated_plate

import bfsccylinder_models.koiter_cylinder as vk_model
import bfsccylinder_models.koiter_cylinder_sanders as sa_model

L = 0.3556
E11, E22, G12, nu12 = 127.629e9, 11.3074e9, 6.00257e9, 0.300235
STACK = (45, -45, 0, 90, 90, 0, -45, 45)
H = 0.00101539
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')


def quiet(fn, *args, **kwargs):
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        out = fn(*args, **kwargs)
    return out, log.getvalue()


def prop():
    return laminated_plate(stack=STACK,
            laminaprop=(E11, E22, nu12, G12, G12, G12), plyt=H/len(STACK))


def sun():
    R, ny = 0.2032, 40
    nx = int(ny*L/(2*np.pi*R))
    nx += 1 - nx % 2
    out, _ = quiet(sa_model.fkoiter_cyl_SS3, L, R, nx, ny, prop(),
                   nint=4, num_eigvals=4, koiter_num_modes=1,
                   Nxxunit=20000., NLprebuck=True)
    print('RESULT %.17g %.17g' % (out['koiter']['b_ijkl'][(0, 0, 0, 0)],
                                  out['load_mult'][0]))


def arbocz(ny=60):
    R = 0.20318603
    nx = int(1.5*ny*L/(2*np.pi*R))
    nx += 1 - nx % 2
    out, log = quiet(vk_model.fkoiter_cyl_SS3, L, R, nx, ny, prop(),
                     nint=4, num_eigvals=2, koiter_num_modes=1,
                     Nxxunit=10000., NLprebuck=True)
    nsp = int(re.search(r'deflated with (\d+) vectors', log).group(1))
    return out['koiter']['b_ijkl'][(0, 0, 0, 0)], out['mu'], nsp


if __name__ == '__main__':
    if sys.argv[1:] == ['sun']:
        sun()
        sys.exit(0)

    print('b_1111 of Sun et al. 3.1, ny=40, in two processes')
    bs = []
    for label, threads in (('one BLAS thread', '1'),
                           ('default threads', None)):
        env = {k: v for k, v in os.environ.items() if k not in THREADS}
        if threads:
            env.update({k: threads for k in THREADS})
        p = subprocess.run([sys.executable, os.path.abspath(__file__), 'sun'],
                           env=env, capture_output=True, text=True,
                           check=True)
        b, lc = map(float, re.search(r'^RESULT (\S+) (\S+)$', p.stdout,
                                     re.M).groups())
        bs.append(b)
        print('  %-16s b_1111 %.12f   load multiplier %.12f' % (label, b, lc))
    print('  relative difference %.1e' % (abs(bs[0] - bs[1])/abs(bs[1])))
    sys.stdout.flush()

    print('\nb_1111 of AW-CYL-1-1, ny=60, and the partner in the column border')
    b_with, mu, nsp_with = arbocz()
    partner = vk_model.degenerate_partner
    vk_model.degenerate_partner = lambda *args, **kwargs: None
    try:
        b_without, _, nsp_without = arbocz()
    finally:
        vk_model.degenerate_partner = partner
    print('  multipliers returned %s' % ', '.join('%.6f' % m for m in mu))
    print('  with the partner     b_1111 %.12f   column border of %d vectors'
          % (b_with, nsp_with))
    print('  without the partner  b_1111 %.12f   column border of %d vectors'
          % (b_without, nsp_without))
    print('  relative difference %.1e' % (abs(b_with - b_without)/abs(b_with)))
