"""run_case.py main, with the raw eigenvectors 1 and 3 replaced by another
member of the same eigenspace when argv[4] == 'slice'"""
import sys
import numpy as np
import run_case as rc
from bfsccylinder_models.cyclic_symmetry import degenerate_partner
if len(sys.argv) > 4 and sys.argv[4] == 'slice':
    original = rc.model.canonical_modes
    def other_slice(mu, v, bu, axi, DOF, deg_rtol=1.e-5):
        v = np.array(v, copy=True)
        a = np.deg2rad(30.)
        for i, j in [(0, 1), (2, 3)]:
            if abs(mu[j] - mu[i]) > 1.e-8*abs(mu[i]):
                continue
            p = np.zeros(bu.shape[0]); p[bu] = v[:, i]
            q = degenerate_partner(p, bu, axi, DOF)
            new = np.cos(a)*v[:, j] + np.sin(a)*q[bu]*np.linalg.norm(v[:, j])/np.linalg.norm(q[bu])
            v[:, j] = new*np.linalg.norm(v[:, j])/np.linalg.norm(new)
        return original(mu, v, bu, axi, DOF, deg_rtol=deg_rtol)
    rc.model.canonical_modes = other_slice
sys.argv = sys.argv[:4]
src = open('run_case.py').read().split("if __name__ == '__main__':")[1]
exec(compile('if True:' + src, 'run_case_main', 'exec'), vars(rc))
