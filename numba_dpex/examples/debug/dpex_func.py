# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np

import numba_dpex as ndpx


@ndpx.func(debug=True)
def func_sum(a_in_func, b_in_func):
    result = a_in_func + b_in_func
    return result


@ndpx.kernel(debug=True)
def kernel_sum(a_in_kernel, b_in_kernel, c_in_kernel):
    i = ndpx.get_global_id(0)
    c_in_kernel[i] = func_sum(a_in_kernel[i], b_in_kernel[i])


def driver(a, b, c, global_size):
    print("a = ", a)
    print("b = ", b)
    print("c = ", c)
    kernel_sum[ndpx.Range(global_size)](a, b, c)
    print("a + b = ", c)


def main():
    global_size = 10
    N = global_size
    print("N", N)

    a = np.arange(N, dtype=np.float32)
    b = np.arange(N, dtype=np.float32)
    c = np.empty_like(a)

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        driver(a, b, c, global_size)

    print("Done...")


if __name__ == "__main__":
    main()
