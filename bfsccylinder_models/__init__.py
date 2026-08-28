"""
BFSCCYLINDER MODELS - Models using the BFSC cylinder finite element
===================================================================

Author: Saullo G. P. Castro

.. automodule:: bfsccylinder_models.models
    :members:

.. automodule:: bfsccylinder_models.vatfunctions
    :members:

"""
import ctypes

import numpy as np

if ctypes.sizeof(ctypes.c_long) == 8:
    # here the C long will correspond to np.int64
    INT = np.int64
else:
    # here the C long will correspond to np.int32
    INT = np.int32

DOUBLE = np.float64

