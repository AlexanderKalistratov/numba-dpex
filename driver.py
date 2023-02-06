import ctypes
import glob
import os

import dpnp
import numpy as np

import numba_dpex as dpex

paths = glob.glob(
    os.path.join(
        os.path.dirname(__file__),
        "numba_dpex/core/runtime/_dpexrt_python.cpython-39-x86_64-linux-gnu.so",
    )
)

print("path:", paths[0])

ctypes.cdll.LoadLibrary(paths[0])


@dpex.dpjit
def foo(Arr):
    return Arr


a = dpnp.ones(10)

b = np.ones(10)

print(a)
print(
    "Type of array:",
    type(a),
    " with usm type:",
    a.usm_type,
    " on device ",
    a.sycl_device,
)
c = foo(a)
print(c)
print(
    "Type of array:",
    type(c),
    " with usm type:",
    c.usm_type,
    " on device ",
    c.sycl_device,
)
