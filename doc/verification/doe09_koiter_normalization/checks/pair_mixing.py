"""run_case.py with Koiter modes 0 and 1 rotated by 30 degrees in their plane"""
import sys, runpy
import numpy as np
import run_case
wrapped = run_case.model.canonical_modes
def mix(mu, eigvecsu, bu, axi_order, DOF, deg_rtol=1.e-5):
    v = wrapped(mu, eigvecsu, bu, axi_order, DOF, deg_rtol=deg_rtol)
    a = np.deg2rad(30.)
    v0, v1 = v[:, 0].copy(), v[:, 1].copy()
    v[:, 0], v[:, 1] = np.cos(a)*v0 + np.sin(a)*v1, -np.sin(a)*v0 + np.cos(a)*v1
    return v
orig_use = run_case.use_distinct_modes
def use():
    nd = orig_use()
    inner = run_case.model.canonical_modes
    def outer(*args, **kw):
        v = inner(*args, **kw)
        a = np.deg2rad(30.)
        v0, v1 = v[:, 0].copy(), v[:, 1].copy()
        v[:, 0], v[:, 1] = np.cos(a)*v0 + np.sin(a)*v1, -np.sin(a)*v0 + np.cos(a)*v1
        return v
    run_case.model.canonical_modes = outer
    return nd
run_case.use_distinct_modes = use
sys.argv = ['run_case.py'] + sys.argv[1:]
src = open('run_case.py').read().split("if __name__ == '__main__':")[1]
g = vars(run_case)
exec(compile('if True:' + src, 'run_case_main', 'exec'), g)
