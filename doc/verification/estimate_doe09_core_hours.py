"""Core-hour estimate of generate_qsubs.py WITHOUT executing it

generate_qsubs.py has submit = True and writes ~2000 job files when run, so
its constants are read with ast and its estimate is reproduced here:
    time = time_per_dof[prebuck]*dof + koiter_time_per_element*num_elements
summed over every case of DOE09.txt and both LIN and NL (done() ignored, all
20000 runs counted).

usage: python estimate_doe09_core_hours.py [--doe-dir DIR] [koiter_time_per_element ...]

DIR defaults to the DOE09 directory of the workstation; run_case.py imports
bfsccylinder_models, so put a checkout on PYTHONPATH
"""
import ast
import os
import sys

import numpy as np

DOE = os.path.join(os.path.expanduser('~'), 'OneDrive - Delft University of '
                   'Technology', 'author', '2026_paper_Rogerio', 'DOE09')
args = sys.argv[1:]
if args[:1] == ['--doe-dir']:
    DOE = args[1]
    args = args[2:]
sys.path.insert(0, DOE)
import run_case

with open(os.path.join(DOE, 'generate_qsubs.py')) as f:
    tree = ast.parse(f.read())
consts = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
            isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name in ('time_per_dof', 'mem_per_dof'):
            consts[name] = {kw.arg: ast.literal_eval(kw.value)
                            for kw in node.value.keywords}
        elif name == 'koiter_time_per_element':
            consts[name] = ast.literal_eval(node.value.body)
time_per_dof = consts['time_per_dof']
print('time_per_dof', time_per_dof, 'koiter_time_per_element in file',
      consts['koiter_time_per_element'])

DOE_vars = np.loadtxt(os.path.join(DOE, 'DOE09.txt'), skiprows=1)
L, R, ny = 1.2, 0.4, run_case.ny
dofs = []
elements = []
for v1, v2, v3, v4, v5 in DOE_vars:
    nx = run_case.estimate_nx(L, R, ny, v1, int(v2), v3, v4, v5)
    dofs.append(run_case.DOF*nx*ny)
    elements.append((nx - 1)*ny)
dofs = np.array(dofs)
elements = np.array(elements)
base = sum(time_per_dof[p]*dofs.sum() for p in ['LIN', 'NL'])/3600
print('runs %d, ny %d, elements per run: mean %.0f, max %d'
      % (2*len(dofs), ny, elements.mean(), elements.max()))
print('without Koiter element time: %.0f core-hours' % base)
for k in [consts['koiter_time_per_element']] + [float(a) for a in args]:
    koiter = 2*k*elements.sum()/3600
    print('koiter_time_per_element %.4g s: %.0f + %.0f = %.0f core-hours'
          % (k, base, koiter, base + koiter))
