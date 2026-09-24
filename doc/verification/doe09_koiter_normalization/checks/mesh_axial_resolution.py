"""Axial mesh that run_case.design_function builds for every ny and
axial_factor

The mesh of fkoiter_cylinder_CTS_circum with mesh_only=True, with the
element and the laminate replaced by stubs, since only xlin is needed and
the laminate of every integration point would take minutes at ny = 280.
nx, nxt and max_ny_nx_aspect_ratio come from choose_nxt of run_case.py.
Printed per case and ny: nx, the circumferential element length dy, the
axial element lengths at the edge, smallest and largest, the average
thickness and sqrt(R h), the length scale of the edge boundary layer and of
the axial waves of the modes.

Checked, and an AssertionError raised otherwise: that axial_factor = 1 gives
the nx of the convergence studies, and that the largest axial element is no
longer than dy/axial_factor for the factors of the reassessment at ny = 120,
160 and 240.

usage, from doc/verification/doe09_koiter_normalization, with the library
on PYTHONPATH:

    python checks/mesh_axial_resolution.py
"""
import contextlib
import io
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import run_case

#NOTE rows 0, 1 and 6 of DOE09.txt: rCTS, param_n, c2_ratio, thetadeg_c1,
#     thetadeg_c2
designs = {0: (0.090, 10, 0.94, 12.4, 17.0),
           1: (0.148, 4, 0.84, 50.0, 73.4),
           6: (0.171, 8, 0.55, 11.1, 69.1)}
L, R = 1.2, 0.4
#NOTE nx of the convergence studies, the RESULT lines of results/ for
#     ny <= 200 and REPORT.md, Reassessment, Section 4, for 240 and 280
present_nx = {0: [91, 121, 161, 191, 221, 321],
              1: [67, 127, 127, 127, 169, 169],
              6: [41, 65, 81, 97, 121, 137]}
nys = [80, 120, 160, 200, 240, 280]
ABD = ['%s%d%d' % (M, i, j) for M in 'ABD'
       for i, j in [(1, 1), (1, 2), (1, 6), (2, 2), (2, 6), (6, 6)]]


class Element:
    def __init__(self, nint):
        for k in ABD:
            setattr(self, k, np.zeros((nint, nint)))


def laminate(stack, plyts, **kwargs):
    return SimpleNamespace(h=sum(plyts), intrho=0., **{k: 0. for k in ABD})


run_case.model.BFSCCylinderSanders = Element
run_case.model.laminated_plate = laminate

for F in [1., 1.5, 2.]:
    #NOTE design_function takes the global of run_case, as run_case.py
    #     --axial-factor sets it
    run_case.axial_factor = F
    for icase, (rCTS, param_n, c2_ratio, th1, th2) in designs.items():
        for ny, nx0 in zip(nys, present_nx[icase]):
            variables = dict(rCTS=rCTS, param_n=param_n, c2_ratio=c2_ratio,
                             thetadeg_c1=th1, thetadeg_c2=th2)
            constants = dict(L=L, R=R, ny=ny, E11=122e9, E22=7.32e9,
                             nu12=0.31, G12=4.9e9, tow_thick=0.13e-3,
                             rho=1540, mesh_only=True, Nxxunit=1000.,
                             NLprebuck=False)
            with contextlib.redirect_stdout(io.StringIO()):
                out = run_case.design_function(variables, constants)
            dx = np.diff(out['xlin'])
            dy = 2*np.pi*R/ny
            h = out['havg']
            nx_est = run_case.estimate_nx(L, R, ny, rCTS, param_n, c2_ratio,
                                          th1, th2, axial_factor=F)
            print('F=%.1f case %d ny=%d nx=%d (estimate_nx %d) nxt=%d '
                  'aspect=%d | dy %.1f mm | dx: edge %.1f, min %.2f, max '
                  '%.1f mm, max/(dy/F) %.3f | h %.3f mm, sqrt(Rh) %.1f mm'
                  % (F, icase, ny, out['nx'], nx_est, out['nxt'],
                     out['max_ny_nx_aspect_ratio'], dy*1e3, dx[0]*1e3,
                     dx.min()*1e3, dx.max()*1e3, dx.max()/(dy/F), h*1e3,
                     np.sqrt(R*h)*1e3))
            if F == 1.:
                assert out['nx'] == nx0, 'nx %d, not the %d of the studies' % (
                        out['nx'], nx0)
            if ny in (120, 160, 240):
                assert dx.max() <= dy/F*(1 + 1.e-12), 'dx max above dy/F'
print('checks passed: nx of the studies for F = 1, dx max <= dy/F at ny = '
      '120, 160 and 240')
