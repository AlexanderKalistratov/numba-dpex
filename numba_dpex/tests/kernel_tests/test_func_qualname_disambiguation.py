# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor as dpt
import numpy as np
import pytest

import numba_dpex as ndpx
from numba_dpex.tests._helper import filter_strings


def make_write_values_kernel(n_rows):
    """Uppermost kernel to set 1s in a certain way.
    The uppermost kernel function invokes two levels
    of inner functions to set 1s in an empty matrix
    in a certain way.

    Args:
        n_rows (int): Number of rows to iterate.

    Returns:
        numba_dpex.core.kernel_interface.dispatcher.JitKernel:
            A JitKernel object that encapsulates a @kernel
            decorated numba_dpex compiled kernel object.
    """
    write_values = make_write_values_kernel_func()

    @ndpx.kernel
    def write_values_kernel(array_in):
        for row_idx in range(n_rows):
            is_even = (row_idx % 2) == 0
            write_values(array_in, row_idx, is_even)

    return write_values_kernel[ndpx.NdRange(ndpx.Range(1), ndpx.Range(1))]


def make_write_values_kernel_func():
    """An upper function to set 1 or 3 ones. A function to set
    one or three 1s. If the row index is even it will set three 1s,
    otherwise one 1. It uses the inner function to do this.

    Returns:
        numba_dpex.core.kernel_interface.func.DpexFunctionTemplate:
            A DpexFunctionTemplate that encapsulates a @func decorated
            numba_dpex compiled function object.
    """
    write_when_odd = make_write_values_kernel_func_inner(1)
    write_when_even = make_write_values_kernel_func_inner(3)

    @ndpx.func
    def write_values(array_in, row_idx, is_even):
        if is_even:
            write_when_even(array_in, row_idx)
        else:
            write_when_odd(array_in, row_idx)

    return write_values


def make_write_values_kernel_func_inner(n_cols):
    """Inner function to set 1s. An inner function to set 1s in
    n_cols number of columns.

    Args:
        n_cols (int): Number of columns to be set to 1.

    Returns:
        numba_dpex.core.kernel_interface.func.DpexFunctionTemplate:
            A DpexFunctionTemplate that encapsulates a @func decorated
            numba_dpex compiled function object.
    """

    @ndpx.func
    def write_values_inner(array_in, row_idx):
        for idx in range(n_cols):
            array_in[row_idx, idx] = 1

    return write_values_inner


@pytest.mark.parametrize("offload_device", filter_strings)
def test_qualname_basic(offload_device):
    """A basic test function to test
    qualified name disambiguation.
    """
    ans = np.zeros((10, 10), dtype=np.int64)
    for i in range(ans.shape[0]):
        if i % 2 == 0:
            ans[i, 0:3] = 1
        else:
            ans[i, 0] = 1

    a = np.zeros((10, 10), dtype=dpt.int64)

    device = dpctl.SyclDevice(offload_device)
    queue = dpctl.SyclQueue(device)

    da = dpt.usm_ndarray(
        a.shape,
        dtype=a.dtype,
        buffer="device",
        buffer_ctor_kwargs={"queue": queue},
    )
    da.usm_data.copy_from_host(a.reshape((-1)).view("|u1"))

    kernel = make_write_values_kernel(10)
    kernel(da)

    result = np.zeros_like(a)
    da.usm_data.copy_to_host(result.reshape((-1)).view("|u1"))

    print(ans)
    print(result)

    assert np.array_equal(result, ans)


if __name__ == "__main__":
    test_qualname_basic("level_zero:gpu:0")
    test_qualname_basic("opencl:gpu:0")
    test_qualname_basic("opencl:cpu:0")