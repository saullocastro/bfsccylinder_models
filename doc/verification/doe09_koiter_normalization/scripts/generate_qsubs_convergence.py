"""Mesh convergence study of DOE09, one PBS job per run

Designs A, B and C of the manuscript are the DOE09 cases 0, 1 and 6. Every
case is run with run_case.py at each ny, with NLprebuck=False and True
"""
import json
import os
from subprocess import Popen

DOE_name = 'DOE09'
python = '/home/saullogiovanip/miniconda3/bin/python3'
submit = True

cases = [0, 1, 6]
nys = [80, 120, 160, 200]
#NOTE memory per run, one run per job. With pypardiso, case 0 at ny=80 peaks
#     at 1.9 GB (LIN) and 2.3 GB (NL), and case 0 is the largest of the three;
#     without pypardiso run_case.py falls back to SuperLU, which needed
#     17.8 GB and 23.0 GB for the same runs and does not fit ny >= 120
mem_gb = {80: 4, 120: 8, 160: 16, 200: 24}
#NOTE koiter_num_modes=10 in run_case.py, 5 distinct modes and their rotated
#     partners, costs several times the figures below, measured with 5, and
#     this study measures it: 55 bordered solves against 15, and 10**4 against
#     625 terms of phi4 per integration point.
#     With koiter_num_modes=5 in run_case.py and the vectorized Koiter
#     tensors of bfsccylinder_models 0.4.0, the Koiter section adds about
#     20 ms per element on one core of the cluster (16.8 ms for case 0 at
#     ny=160, see generate_qsubs.py): 13 min for the 38200 elements of case 0
#     at ny=200, on top of the 1 h of the single-mode NL run. Version 0.3.2
#     took 0.7 s per element on a workstation, 7.4 h for the same run
walltime_hours = 12
#NOTE part of the output names, so that the single-mode study is kept
koiter_suffix = '_k5c'


def done(outname):
    """True if the run already wrote its RESULT, successful or not"""
    if not os.path.isfile(outname):
        return False
    with open(outname) as f:
        for line in f:
            if line.startswith('RESULT '):
                #NOTE b_min_energy, see koiter_post.py
                result = json.loads(line[len('RESULT '):])
                return 'b_min_energy' in result or 'error' in result
    return False


scripts = []
for ny in nys:
    for icase in cases:
        for prebuck in ['LIN', 'NL']:
            outname = DOE_name + ('_conv_%05d_ny%03d_%s%s.out'
                                  % (icase, ny, prebuck, koiter_suffix))
            if done(outname):
                continue
            qsub_script = """#!/bin/sh
#
#PBS -l nodes=1:ppn=1,mem={mem}gb,walltime={walltime}:00:00
#
cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
{python} -u run_case.py {icase} {prebuck} {ny} > {outname} 2>&1
""".format(mem=mem_gb[ny], walltime=walltime_hours, python=python,
           icase=icase, prebuck=prebuck, ny=ny, outname=outname)
            qsub_script_name = outname[:-4] + '.sub'
            with open(qsub_script_name, 'w') as f:
                f.write(qsub_script)
            scripts.append(qsub_script_name)
print('# num_qsubs', len(scripts))

if submit:
    for qsub_script_name in scripts:
        p = Popen('qsub %s' % qsub_script_name, shell=True)
        p.wait()
