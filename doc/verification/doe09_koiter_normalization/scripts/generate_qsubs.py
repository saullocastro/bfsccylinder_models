import json
import os
from subprocess import Popen

import numpy as np

import run_case

#NOTE multi-mode Koiter analysis of koiter_cylinder_CTS_sanders on at least 5
#     distinct modes, completed to whole groups of equal multiplier, and their
#     rotated partners, with the full b_ijkl and a_ijk of the Koiter modes in
#     the RESULT line of each run, read by post.py
koiter_num_distinct = 5
assert run_case.koiter_num_distinct == koiter_num_distinct, \
        'set koiter_num_distinct = %d in run_case.py' % koiter_num_distinct

DOE_name = 'DOE09'
python = '/home/saullogiovanip/miniconda3/bin/python3'
submit = True

#NOTE time and peak memory of one run on one core, against its DOF = 10*nx*ny,
#     the largest per-DOF values of the 24 runs of the convergence study
#     (DOE09_convergence.txt)
time_per_dof = dict(LIN=8.e-3, NL=17.e-3) # s
mem_per_dof = dict(LIN=28.e-6, NL=35.e-6) # GB
mem_base = 0.5 # GB
#NOTE with koiter_num_modes=5 the Koiter section adds 16.8 ms per element,
#     measured on one core of the cluster for case 0, NLprebuck, ny=160,
#     with the vectorized element loop of bfsccylinder_models (version
#     0.4.0, doc/verification/cluster_results/REPORT.md); 1.3 ms
#     of it is the element loop, the rest the bordered solves. It grows
#     slowly with the mesh, 13.4 ms for case 6 (12,800 elements) and up to
#     about 20 ms for the largest designs (47,680 elements). Version 0.3.2
#     of the library, with the Python element loop, took 0.7 s per element
#     on a workstation and 2.0 s on the cluster. It is not in the
#     single-mode values above
#NOTE with koiter_num_modes=10, 5 distinct modes and their rotated partners,
#     the Koiter section costs 30 ms per element more than with 5, the
#     median over the 24 runs of the convergence study with both
#     (DOE09_convergence_k5c.txt against k5_crest/), from 23 to 38 ms, so
#     16.8 + 30 ms in all
#     With the groups completed, 12 modes in 10 of the 24 runs of the
#     convergence study (DOE09_convergence_k5g.txt), 20 ms more per element
#     for them, the median of those 10 runs against the same runs with 10
#     modes (9.6 to 38.7 ms), so about 55 ms on average, 60 ms here
koiter_time_per_element = 0.060 # s

#NOTE one node per job, the runs executed side by side in it, one core each.
#     num_parallel is set per job from the largest run it contains
ppn = 10
mem_node_gb = 32
mem_margin = 0.85
walltime_hours = 24
target_time_seconds = 3600*16 # per job, leaving margin to the walltime


def done(outname):
    """True if the run already wrote its RESULT, successful or not, with
    koiter_num_modes modes

    An output of an earlier setup, single-mode, with 5 modes not closed
    under the rotation, with a group of equal multipliers cut, or without the
    normalizations of koiter_post.py, is run again
    """
    if not os.path.isfile(outname):
        return False
    with open(outname) as f:
        for line in f:
            if line.startswith('RESULT '):
                result = json.loads(line[len('RESULT '):])
                #NOTE koiter_set and crest_method of the present run_case.py
                if 'error' in result:
                    return result.get('koiter_set') == 'complete_clusters'
                return (result.get('koiter_set') == 'complete_clusters'
                        and result.get('koiter_num_distinct')
                            == koiter_num_distinct
                        and result.get('crest_method') == 'element_orbit')
    return False


DOE_vars = np.loadtxt(DOE_name + '.txt', skiprows=1)
size_of_DOE = DOE_vars.shape[0]
L, R, ny = 1.2, 0.4, run_case.ny
print('# size_of_DOE', size_of_DOE, 'ny', ny)

tasks = []
for i, (v1, v2, v3, v4, v5) in enumerate(DOE_vars):
    nx = run_case.estimate_nx(L, R, ny, v1, int(v2), v3, v4, v5)
    dof = run_case.DOF*nx*ny
    num_elements = (nx - 1)*ny
    for prebuck in ['LIN', 'NL']:
        outname = DOE_name + ('_%05d_%s.out' % (i, prebuck))
        if done(outname):
            continue
        tasks.append(dict(icase=i, prebuck=prebuck, outname=outname,
                          time=(time_per_dof[prebuck]*dof
                                + koiter_time_per_element*num_elements),
                          mem=mem_base + mem_per_dof[prebuck]*dof))
print('# runs to submit', len(tasks))
print('# estimated core-hours', sum(t['time'] for t in tasks)/3600)

#NOTE the largest runs first, so that every job holds runs of similar size and
#     num_parallel, set by the first run of the job, fits all of them
tasks.sort(key=lambda t: t['mem'], reverse=True)
chunks = []
chunk = []
for task in tasks:
    if chunk:
        num_parallel = chunk[0]['num_parallel']
        chunk_time = sum(t['time'] for t in chunk)/num_parallel
        if chunk_time + task['time']/num_parallel > target_time_seconds:
            chunks.append(chunk)
            chunk = []
    if not chunk:
        num_parallel = max(1, min(ppn, int(mem_margin*mem_node_gb/task['mem'])))
        task['num_parallel'] = num_parallel
    chunk.append(task)
if chunk:
    chunks.append(chunk)
print('# num_qsubs', len(chunks))

scripts = []
for i, qsub_cases in enumerate(chunks):
    num_parallel = qsub_cases[0]['num_parallel']
    tasks_name = DOE_name + ('_chunk_%04d.tasks' % i)
    with open(tasks_name, 'w') as f:
        for t in qsub_cases:
            f.write('%s -u run_case.py %d %s > %s 2>&1\n'
                    % (python, t['icase'], t['prebuck'], t['outname']))
    qsub_script = """#!/bin/sh
#
#PBS -l nodes=1:ppn={ppn},mem={mem}gb,walltime={walltime}:00:00
#
cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
#NOTE bfsccylinder_models of the branch doe09-koiter-normalization, see
#     run_case.py
export PYTHONPATH=/home/saullogiovanip/bfsccylinder_models
xargs -P {num_parallel} -I CMD sh -c 'CMD' < {tasks_name}
""".format(ppn=ppn, mem=mem_node_gb, walltime=walltime_hours,
           num_parallel=num_parallel, tasks_name=tasks_name)
    qsub_script_name = DOE_name + ('_chunk_%04d.sub' % i)
    with open(qsub_script_name, 'w') as f:
        f.write(qsub_script)
    scripts.append(qsub_script_name)

if submit:
    for qsub_script_name in scripts:
        p = Popen('qsub %s' % qsub_script_name, shell=True)
        p.wait()
