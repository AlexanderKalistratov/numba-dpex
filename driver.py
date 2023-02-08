import dpnp
import numpy as np

import numba_dpex as dpex


@dpex.dpjit
def foo(Arr):
    dpnp.empty(10)
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
