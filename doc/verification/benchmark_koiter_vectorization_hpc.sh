#!/bin/sh
#
# Production-size runs of benchmark_koiter_vectorization.py on a PBS cluster,
# one core per run as in the DOE: DOE09 case CASE (default 0), NLprebuck=True
# (PREBUCK=NL, default) or False (PREBUCK=LIN), ny=NY (default 160). Submit
# from the DOE09 directory (run_case.py and DOE09.txt), after setting the
# paths below, for instance
#
#     qsub -v RUN=new_m5 benchmark_koiter_vectorization_hpc.sh
#     qsub -v RUN=new_m0 benchmark_koiter_vectorization_hpc.sh
#     qsub -v RUN=new_m5,CASE=6 benchmark_koiter_vectorization_hpc.sh
#     qsub -v RUN=base_m5 benchmark_koiter_vectorization_hpc.sh  # optional,
#                         # the old element loop alone takes ~5 h for case 0
#
# Each run writes <RUN>_<PREBUCK>_case<CASE>_ny<NY>.json with time_total,
# time_koiter (from the printed buckling load to the return),
# time_element_loop (new version only), time_bordered_solves, peak_mem_gb,
# num_elements and the results (Pcr, load_mult, a_ijk, b_ijkl). The Koiter
# time per element of generate_qsubs.py is time_koiter/num_elements of the
# m=5 run, or (time_total(m=5) - time_total(m=0))/num_elements.
#
#PBS -l nodes=1:ppn=1,mem=16gb,walltime=24:00:00
#
PYTHON=${PYTHON:-/home/saullogiovanip/miniconda3/bin/python3}
#NOTE clones of the vectorize-koiter branch and of version 0.3.2 (3251981)
NEW=${NEW:-$HOME/bfsccylinder_models}
BASE=${BASE:-$HOME/bfsccylinder_models_baseline}
CASE=${CASE:-0}
PREBUCK=${PREBUCK:-NL}
NY=${NY:-160}

cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

case $RUN in
    new_m5)  LIB=$NEW;  M=5 ;;
    new_m0)  LIB=$NEW;  M=0 ;;
    base_m5) LIB=$BASE; M=5 ;;
    base_m0) LIB=$BASE; M=0 ;;
    *) echo "RUN must be new_m5, new_m0, base_m5 or base_m0"; exit 1 ;;
esac
NAME=${RUN}_${PREBUCK}_case${CASE}_ny${NY}

#NOTE run_case.py must find pypardiso as it does for the DOE runs; if it is
#     not installed in $PYTHON, append its --target directory to PYTHONPATH
PYTHONPATH=$LIB${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH} $PYTHON -u \
    $NEW/doc/verification/benchmark_koiter_vectorization.py worker \
    --kind doe --case $CASE --prebuck $PREBUCK --ny $NY --m $M --neig 12 \
    --doe-dir $PBS_O_WORKDIR --out $NAME.json > $NAME.log 2>&1
