"""Reassessment studies of DOE09, one PBS job per run

What b_min_t of the NL cases converges to, see REPORT.md, Reassessment, on
designs A, B and C of the manuscript, the DOE09 cases 0, 1 and 6, with the
options of run_case.py:

- (a) truncation: --distinct 5, 7 and 9, NL, ny = 120 and 160, and 0 LIN and
  6 LIN with 9 at ny = 160;
- (b) axial refinement at ny = 160: --axial-factor 1.5, 2 and 3, LIN and NL,
  a factor that gives the mesh of a smaller one, or of 1, skipped;
- (c) expansion point: --eps1 0.002, 0.001 and 0.0005, cases 1 and 6 NL,
  ny = 120;
- (d) ny = 240 with the setup of the convergence study _k5g;
- (e) the NL mesh sequences with --eps1 0.0005: ny = 120 to 240, and the
  axial factors of (b) at ny = 160. (c) found b to move by 1.9 % (case 1)
  and 0.85 % (case 6) per 0.001 of lambda_b/lambda_c, which the load
  stepping with eps1 = 0.005 leaves anywhere from 0.9951 to 0.9983, more
  than the mesh error being measured;
- (f) more axial meshes of case 1 NL at ny = 160, the candidate DOE mesh
  ny = 240 with --axial-factor 1.5 for cases 1 and 6, and ny = 280 for case
  0, all NL with --eps1 0.0005;
- (g) case 1 NL, --eps1 0.0005, at ny = 120 and 200 with the axial factors
  that give nx = 169 to 331, a grid in ny and nx with (e) and (f);
- (h) quadrature: case 0 NL, ny = 160, --eps1 0.0005, --nint 5 and 6;
- (i) Donnell kinematics, --kinematics donnell, on the ny sequences of (e)
  for cases 0 and 6;
- (j) R/h: case 6 NL, --eps1 0.0005, --thickness-factor 0.7 and 2, R/h
  about 1490 and 520 as cases 0 and 1, at ny giving about 5.3, 6.7 and 8.3
  elements per wave of the critical wave number, as ny = 160, 200 and 240
  do at thickness factor 1; factor 0.7 again with --nxxunit 500, its Pcr
  being below the load of lambda = 1 with the default 1000 N/m;
- (k) simply supported edges with v = w = 0 along the whole edge, on the ny
  sequences of (e) of cases 0, 1 and 6;
- (l) four more designs of the DOE, spanning its R/h, at the candidate
  setup: ny = 160, 200 and 240, --axial-factor 1.5, --eps1 0.0005, and
  --nxxunit 500 against a Pcr below the load of lambda = 1;
- (m) SS4 edges, --edges SS4, u uniform along each edge, from a low ny:
  ny = 40 to 240 (200 for case 1) of cases 0, 1 and 6 NL, --eps1 0.0005,
  the setup of (k), and the SS3 edges of (k) at ny = 40, 60 and 80,
  and 120 for case 6.

(a) to (j) were run with v and w fixed at the edge nodes only, before the
models fixed v,y and w,y too, which is now their only condition; they are
kept as they are and not rerun, and a missing one is only reported.

The reference runs, factor 1 of (b), eps1 = 0.005 of (c) and ny = 120 to
200 of (d), are those of _k5g. Prints the runs with their walltime, memory
and estimated time; set submit = True to submit them
"""
import json
import os
from subprocess import Popen

import run_case

DOE_name = 'DOE09'
python = '/home/saullogiovanip/miniconda3/bin/python3'
submit = False

cases = [0, 1, 6]
#NOTE num_eigvals of a run, raised above the 2K + 8 of run_case.py when
#     --distinct K hits a selection error, keyed by (study, icase, ny,
#     prebuck, tag)
num_eigvals_override = {}

#NOTE rows of DOE09.txt, for the mesh of every axial factor: 0, 1 and 6,
#     designs A, B and C, and the four of (l), the largest R/h of the DOE
#     (1538), its 90th percentile (1461), its median (1148) and its
#     smallest (419), R/h of cases 0, 1 and 6 being 1474, 546 and 1040
designs = {0: (0.090, 10, 0.94, 12.4, 17.0),
           1: (0.148, 4, 0.84, 50.0, 73.4),
           6: (0.171, 8, 0.55, 11.1, 69.1),
           6112: (0.167, 10, 0.53, 0.2, 0.8),
           5517: (0.107, 6, 0.48, 7.5, 25.4),
           3486: (0.191, 3, 0.76, 29.6, 45.7),
           1838: (0.094, 6, 0.08, 74.4, 71.9)}
L, R = 1.2, 0.4

#NOTE time per element on one core, from the _k5g study (12 Koiter modes in
#     the NL runs, 10 in most LIN ones): about 60 ms of Koiter section, and
#     the rest, 20 ms (LIN) and 50 ms (NL), the pre-buckling and eigenvalue
#     analyses. The bordered solves of the second order fields, m(m+1)/2 for
#     m Koiter modes, are most of the Koiter section, which is scaled with
#     them. A run of --distinct K takes 2K + 2 Koiter modes at most, the last
#     group completed
base_time_per_element = dict(LIN=0.020, NL=0.050) # s
koiter_time_per_element_12 = 0.060 # s
#NOTE peak memory against DOF = 10*nx*ny, 37e-6 GB per DOF for case 0 NL at
#     ny = 200 with 12 modes, and phi3, phi30 and cst, (N, m, m) each
mem_per_dof = 37.e-6 # GB


def mesh(icase, ny, axial_factor):
    return run_case.choose_nxt(L, R, ny, *designs[icase],
                               axial_factor=axial_factor)


def estimate(icase, prebuck, ny, distinct, axial_factor):
    """Estimated time in s and peak memory in GB of a run"""
    nx = run_case.estimate_nx(L, R, ny, *designs[icase],
                              axial_factor=axial_factor)
    m = 2*distinct + 2
    elements = (nx - 1)*ny
    dof = 10*nx*ny
    time = elements*(base_time_per_element[prebuck]
                     + koiter_time_per_element_12*m*(m + 1)/(12*13))
    mem = 0.5 + mem_per_dof*dof + max(0, 3*dof*(m**2 - 12**2)*8/1e9)
    return nx, time, mem


runs = []
for ny in [120, 160]:
    for icase in cases:
        for K in [5, 7, 9]:
            runs.append(dict(study='a', icase=icase, prebuck='NL', ny=ny,
                             distinct=K, tag='k%d' % K))
for icase in [0, 6]:
    runs.append(dict(study='a', icase=icase, prebuck='LIN', ny=160,
                     distinct=9, tag='k9'))
for icase in cases:
    for prebuck in ['LIN', 'NL']:
        seen = [mesh(icase, 160, 1.)]
        for F in [1.5, 2., 3.]:
            if mesh(icase, 160, F) in seen:
                print('# (b) case %d %s: axial factor %g skipped, the mesh of '
                      'a smaller factor' % (icase, prebuck, F))
                continue
            seen.append(mesh(icase, 160, F))
            runs.append(dict(study='b', icase=icase, prebuck=prebuck, ny=160,
                             axial_factor=F,
                             tag=('ax%g' % F).replace('.', 'p')))
for icase in [1, 6]:
    for eps1 in [0.002, 0.001, 0.0005]:
        runs.append(dict(study='c', icase=icase, prebuck='NL', ny=120,
                         eps1=eps1, tag=('eps%g' % eps1).replace('.', 'p')))
for icase in cases:
    for prebuck in ['LIN', 'NL']:
        runs.append(dict(study='d', icase=icase, prebuck=prebuck, ny=240,
                         tag='k5'))

#NOTE (e): lambda_b/lambda_c of about 0.9997 in every run, see (c)
for icase in cases:
    for ny in [120, 160, 200, 240]:
        runs.append(dict(study='e', icase=icase, prebuck='NL', ny=ny,
                         eps1=0.0005, tag='eps0p0005'))
    seen = [mesh(icase, 160, 1.)]
    for F in [1.5, 2., 3.]:
        if mesh(icase, 160, F) in seen:
            continue
        seen.append(mesh(icase, 160, F))
        runs.append(dict(study='e', icase=icase, prebuck='NL', ny=160,
                         axial_factor=F, eps1=0.0005,
                         tag=('ax%g_eps0p0005' % F).replace('.', 'p')))

#NOTE (f): case 1 NL gave b of the critical cluster n = 24 not monotone in
#     nx at ny = 160 in (e), -0.2848, -0.2950 and -0.2849 for nx = 127, 169
#     and 275, with the same mass and the same clusters, so more axial
#     meshes; the candidate DOE mesh ny = 240 with factor 1.5 for cases 1
#     and 6, the factor changing nothing for case 0; and a fourth ny for case
#     0, whose order in ny was 2.1 from 120 to 200 and 3.7 from 160 to 240
for F in [2.5, 4., 5.]:
    runs.append(dict(study='f', icase=1, prebuck='NL', ny=160,
                     axial_factor=F, eps1=0.0005,
                     tag=('ax%g_eps0p0005' % F).replace('.', 'p')))
for icase in [1, 6]:
    runs.append(dict(study='f', icase=icase, prebuck='NL', ny=240,
                     axial_factor=1.5, eps1=0.0005, tag='ax1p5_eps0p0005'))
runs.append(dict(study='f', icase=0, prebuck='NL', ny=280, eps1=0.0005,
                 tag='eps0p0005'))

#NOTE (g): case 1 NL on a grid of ny and nx, with (f) and (e), so that
#     b(ny, nx) can be fitted in both directions at once and the mesh of
#     fewest elements for a given error found; nx changed at fixed ny through
#     the axial factor, the element aspect ratio dy/dx up to 4
for ny, F in [(120, 2.), (120, 3.), (120, 5.), (200, 2.), (200, 3.)]:
    runs.append(dict(study='g', icase=1, prebuck='NL', ny=ny,
                     axial_factor=F, eps1=0.0005,
                     tag=('ax%g_eps0p0005' % F).replace('.', 'p')))

#NOTE (h): nint = 4 integrates degree 7 per direction exactly, and phi4 has
#     products of four derivatives of the bicubic w, of degree 8 in x and up
#     to 12 in y; case 0, of slowest convergence in ny, at the mesh of (e)
for nint in [5, 6]:
    runs.append(dict(study='h', icase=0, prebuck='NL', ny=160, eps1=0.0005,
                     nint=nint, tag='nint%d_eps0p0005' % nint))

#NOTE (i): the membrane strains of the two elements are the same, v,y + w/R,
#     the curvatures and the rotation of the nonlinear strains not; the
#     difference of the kinematics, of order 1/n**2, is small at n = 30
for icase, nys in [(0, [120, 160, 200, 240]), (6, [160, 200, 240])]:
    for ny in nys:
        runs.append(dict(study='i', icase=icase, prebuck='NL', ny=ny,
                         eps1=0.0005, kinematics='donnell',
                         tag='donnell_eps0p0005'))

#NOTE (j): the critical wave number of case 6 NL is 29 at thickness factor
#     1, and taken as 29/sqrt(T), as n_c of an axially compressed cylinder
#     goes with sqrt(R/h); the runs print the n reached
for T, nys in [(0.7, [184, 232, 288]), (2., [112, 140, 168])]:
    for ny in nys:
        runs.append(dict(study='j', icase=6, prebuck='NL', ny=ny,
                         eps1=0.0005, thickness_factor=T,
                         tag=('t%g_eps0p0005' % T).replace('.', 'p')))

#NOTE Pcr of about 2470 N at T = 0.7, below the 2513 N of lambda = 1 with
#     Nxxunit = 1000 N/m, where the load stepping starts, so that the
#     expansion point stayed past the bifurcation, lambda_b/lambda_c 1.002 to
#     1.023; 500 N/m starts it at half of Pcr
for ny in [184, 232, 288]:
    runs.append(dict(study='j', icase=6, prebuck='NL', ny=ny, eps1=0.0005,
                     thickness_factor=0.7, nxxunit=500.,
                     tag='t0p7_n500_eps0p0005'))

#NOTE (k): the model fixes v and w at the edge nodes only, v,y and w,y
#     being free there, so that the edges deflect between the nodes, an
#     error that goes only with dy; case 1 at nx = 127 for every ny
for icase, nys in [(0, [120, 160, 200, 240]), (1, [120, 160, 200]),
                   (6, [160, 200, 240])]:
    for ny in nys:
        runs.append(dict(study='k', icase=icase, prebuck='NL', ny=ny,
                         eps1=0.0005, tag='ssfull_eps0p0005'))

for icase in [6112, 5517, 3486, 1838]:
    for ny in [160, 200, 240]:
        runs.append(dict(study='l', icase=icase, prebuck='NL', ny=ny,
                         axial_factor=1.5, eps1=0.0005, nxxunit=500.,
                         tag='ax1p5_n500_eps0p0005'))

#NOTE (m): the low ny end of both edge conditions, to see where the
#     sequences enter their asymptotic range
for icase, nys in [(0, [40, 60, 80, 120, 160, 200, 240]),
                   (1, [40, 60, 80, 120, 160, 200]),
                   (6, [40, 60, 80, 120, 160, 200, 240])]:
    for ny in nys:
        runs.append(dict(study='m', icase=icase, prebuck='NL', ny=ny,
                         eps1=0.0005, edges='SS4', tag='ss4_eps0p0005'))
    #NOTE (k) starts at ny = 160 for case 6
    for ny in [40, 60, 80] + ([120] if icase == 6 else []):
        runs.append(dict(study='m', icase=icase, prebuck='NL', ny=ny,
                         eps1=0.0005, tag='ss3_eps0p0005'))

for r in runs:
    r.setdefault('edges', run_case.edges)
    #NOTE the edges of (a) to (j), see the docstring
    r.setdefault('ss_full', r['study'] not in list('abcdefghij'))
    r.setdefault('nxxunit', run_case.Nxxunit)
    r.setdefault('nint', run_case.nint)
    r.setdefault('kinematics', run_case.kinematics)
    r.setdefault('thickness_factor', run_case.thickness_factor)
    r.setdefault('distinct', run_case.koiter_num_distinct)
    r.setdefault('axial_factor', run_case.axial_factor)
    r.setdefault('eps1', run_case.NLprebuck_eps1)
    key = (r['study'], r['icase'], r['ny'], r['prebuck'], r['tag'])
    r['num_eigvals'] = num_eigvals_override.get(key,
            run_case.num_eigvals if r['distinct'] == run_case.koiter_num_distinct
            else 2*r['distinct'] + 8)
    r['outname'] = DOE_name + ('_reassess_%s_%05d_ny%03d_%s_%s.out'
            % (r['study'], r['icase'], r['ny'], r['prebuck'], r['tag']))


def done(r):
    """True if the output has a RESULT, successful or not, of the options
    requested"""
    if not os.path.isfile(r['outname']):
        return False
    with open(r['outname']) as f:
        for line in f:
            if line.startswith('RESULT '):
                result = json.loads(line[len('RESULT '):])
                return (result.get('ny', r['ny']) == r['ny']
                        and result.get('koiter_num_distinct') == r['distinct']
                        and result.get('num_eigvals') == r['num_eigvals']
                        and result.get('axial_factor') == r['axial_factor']
                        and result.get('NLprebuck_eps1') == r['eps1']
                        and result.get('nint', 4) == r['nint']
                        and result.get('kinematics', 'sanders')
                            == r['kinematics']
                        and result.get('thickness_factor', 1.)
                            == r['thickness_factor']
                        and result.get('Nxxunit', 1000.) == r['nxxunit']
                        and result.get('ss_edge_tangential', False)
                            == r['ss_full']
                        and result.get('edges', 'SS3') == r['edges']
                        and ('subsets' in result or 'error' in result))
    return False


print('%-58s %4s %5s %6s %6s %6s' % ('# output', 'nx', 'm<=', 'est h',
                                     'wall h', 'mem GB'))
scripts = []
total = 0.
for r in runs:
    nx, time, mem = estimate(r['icase'], r['prebuck'], r['ny'], r['distinct'],
                             r['axial_factor'])
    #NOTE the estimates are for one core of the cluster, the walltime and
    #     memory requested at least twice and 1.5 times of them
    walltime = 12 if 2*time < 12*3600 else 24
    mem_gb = 24 if 1.5*mem < 24 else 48
    status = 'done' if done(r) else ''
    print('%-58s %4d %5d %6.1f %6d %6d %s' % (r['outname'], nx,
          2*r['distinct'] + 2, time/3600, walltime, mem_gb, status))
    if status:
        continue
    if not r['ss_full']:
        print('# %s: run with v and w fixed at the edge nodes only, which '
              'the models no longer do, not rerun' % r['outname'])
        continue
    total += time
    options = []
    if r['distinct'] != run_case.koiter_num_distinct:
        options.append('--distinct %d' % r['distinct'])
    if r['num_eigvals'] != run_case.num_eigvals:
        options.append('--num-eigvals %d' % r['num_eigvals'])
    if r['axial_factor'] != run_case.axial_factor:
        options.append('--axial-factor %g' % r['axial_factor'])
    if r['eps1'] != run_case.NLprebuck_eps1:
        options.append('--eps1 %g' % r['eps1'])
    if r['nint'] != run_case.nint:
        options.append('--nint %d' % r['nint'])
    if r['kinematics'] != run_case.kinematics:
        options.append('--kinematics %s' % r['kinematics'])
    if r['thickness_factor'] != run_case.thickness_factor:
        options.append('--thickness-factor %g' % r['thickness_factor'])
    if r['nxxunit'] != run_case.Nxxunit:
        options.append('--nxxunit %g' % r['nxxunit'])
    if r['edges'] != run_case.edges:
        options.append('--edges %s' % r['edges'])
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
{python} -u run_case.py {icase} {prebuck} {ny} {options} > {outname} 2>&1
""".format(mem=mem_gb, walltime=walltime, python=python, icase=r['icase'],
           prebuck=r['prebuck'], ny=r['ny'], options=' '.join(options),
           outname=r['outname'])
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
