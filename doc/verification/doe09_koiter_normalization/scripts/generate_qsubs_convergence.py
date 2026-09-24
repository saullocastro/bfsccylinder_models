"""Mesh convergence study of DOE09 from a low ny, one PBS job per run

Designs A, B and C of the manuscript, the DOE09 cases 0, 1 and 6, NL,
--eps1 0.0005, on the inertia relief edges of bfsccylinder_models/edges.py,
the axial load on both edges and no node anchored:

- SS3-IR: v = w = 0 along both edges, the axial translation removed by
  inertia relief;
- free-IR: no edge condition, the six rigid body modes removed by inertia
  relief.

Every edge condition and case at ny = 24, 32, 40, 48, 64, 80, 96, 120 and
160 and at the axial factors F = 0.5, 1 and 2, the largest axial element
length being dy/F: elements twice as long axially as around, square, and
half as long; 162 runs. The previous convergence studies, all with the SS3
edges anchored at one node, were removed from the branch; git history has
them.

usage, from the DOE09 working directory, with DOE09.txt and run_case.py:

    python generate_qsubs_convergence.py

prints the runs with their walltime, memory and estimated time and writes
the job scripts of the runs not done; set submit = True to submit them
"""
import json
import os
from subprocess import Popen

import run_case

DOE_name = 'DOE09'
python = '/home/saullogiovanip/miniconda3/bin/python3'
submit = False

cases = [0, 1, 6]
edges_list = ['SS3-IR', 'free-IR']
nys = [24, 32, 40, 48, 64, 80, 96, 120, 160]
axial_factors = [0.5, 1., 2.]
eps1 = 0.0005

#NOTE rows of DOE09.txt, for the mesh of every axial factor
designs = {0: (0.090, 10, 0.94, 12.4, 17.0),
           1: (0.148, 4, 0.84, 50.0, 73.4),
           6: (0.171, 8, 0.55, 11.1, 69.1)}
L, R = 1.2, 0.4

#NOTE time per element on one core, from the previous studies (12 Koiter
#     modes in the NL runs): about 60 ms of Koiter section and 50 ms of
#     pre-buckling and eigenvalue analyses; the inertia relief adds a few
#     solves per eigenvalue analysis
time_per_element = 0.050 + 0.060 # s
#NOTE peak memory against DOF = 10*nx*ny, 37e-6 GB per DOF
mem_per_dof = 37.e-6 # GB


def estimate(icase, ny, axial_factor):
    """nx, estimated time in s and peak memory in GB of a run"""
    nx = run_case.estimate_nx(L, R, ny, *designs[icase],
                              axial_factor=axial_factor)
    time = (nx - 1)*ny*time_per_element
    mem = 0.5 + mem_per_dof*10*nx*ny
    return nx, time, mem


def tag(r):
    return ('%s_F%g_eps0p0005' % (r['edges'], r['axial_factor'])
            ).replace('.', 'p').replace('-', '')


runs = []
for edges in edges_list:
    for icase in cases:
        for F in axial_factors:
            for ny in nys:
                runs.append(dict(edges=edges, icase=icase, ny=ny,
                                 axial_factor=F))
for r in runs:
    r['outname'] = DOE_name + ('_conv_%05d_ny%03d_NL_%s.out'
                               % (r['icase'], r['ny'], tag(r)))


def done(r):
    """True if the output has a RESULT, successful or not, of the options
    requested"""
    if not os.path.isfile(r['outname']):
        return False
    with open(r['outname']) as f:
        for line in f:
            if line.startswith('RESULT '):
                result = json.loads(line[len('RESULT '):])
                return (result.get('ny') == r['ny']
                        and result.get('axial_factor') == r['axial_factor']
                        and result.get('NLprebuck_eps1') == eps1
                        and result.get('edges') == r['edges']
                        and ('subsets' in result or 'error' in result))
    return False


if __name__ == '__main__':
    print('%-52s %4s %6s %6s %6s' % ('# output', 'nx', 'est h', 'wall h',
                                     'mem GB'))
    scripts = []
    total = 0.
    for r in runs:
        nx, time, mem = estimate(r['icase'], r['ny'], r['axial_factor'])
        walltime = 12 if 2*time < 12*3600 else 24
        mem_gb = 8 if 1.5*mem < 8 else (24 if 1.5*mem < 24 else 48)
        status = 'done' if done(r) else ''
        print('%-52s %4d %6.2f %6d %6d %s' % (r['outname'], nx, time/3600,
              walltime, mem_gb, status))
        if status:
            continue
        total += time
        options = ['--eps1 %g' % eps1, '--edges %s' % r['edges']]
        if r['axial_factor'] != run_case.axial_factor:
            options.append('--axial-factor %g' % r['axial_factor'])
        qsub_script = """#!/bin/sh
#
#PBS -l nodes=1:ppn=1,mem={mem}gb,walltime={walltime}:00:00
#
cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
#NOTE bfsccylinder_models of the branch doe09-koiter-normalization, see
#     run_case.py
export PYTHONPATH=/home/saullogiovanip/bfsccylinder_models
{python} -u run_case.py {icase} NL {ny} {options} > {outname} 2>&1
""".format(mem=mem_gb, walltime=walltime, python=python, icase=r['icase'],
           ny=r['ny'], options=' '.join(options), outname=r['outname'])
        qsub_script_name = r['outname'][:-4] + '.sub'
        with open(qsub_script_name, 'w') as f:
            f.write(qsub_script)
        scripts.append(qsub_script_name)
    print('# num_qsubs', len(scripts))
    print('# estimated core-hours %.0f' % (total/3600))

    if submit:
        for qsub_script_name in scripts:
            p = Popen('qsub %s' % qsub_script_name, shell=True)
            p.wait()
