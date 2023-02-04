# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import argparse

import dpctl
import numba
import numpy as np

import numba_dpex as dpex
from numba_dpex.core.kernel_interface.utils import Range


def common_loop_body(i, a, b):
    param_a = a[i]
    param_b = b[i]
    param_c = param_a + 10  # Set breakpoint here
    param_d = param_b * 0.5
    result = param_c + param_d
    return result


def scenario(api):
    print("Using API:", api)

    global_size = 10
    a, b, c = arguments(global_size)

    if api == "numba-dpex-kernel":
        dpex_func_driver(a, b, c)
    else:
        numba_func_driver(a, b, c)

    print(a, b, c, sep="\n")


def arguments(N, dtype=np.float32):
    a = np.arange(N, dtype=dtype)
    b = np.arange(N, dtype=dtype)
    c = np.empty_like(a)
    return a, b, c


@numba.njit(debug=True)
def numba_func_driver(a, b, c):
    for i in range(len(c)):
        c[i] = numba_loop_body(i, a, b)


def dpex_func_driver(a, b, c):
    device = dpctl.select_default_device()
    with dpctl.device_context(device):
        kernel[Range(len(c))](a, b, c)


@dpex.kernel(debug=True)
def kernel(a_in_kernel, b_in_kernel, c_in_kernel):
    i = dpex.get_global_id(0)
    c_in_kernel[i] = dpex_loop_body(i, a_in_kernel, b_in_kernel)


numba_loop_body = numba.njit(debug=True)(common_loop_body)
dpex_loop_body = dpex.func(debug=True)(common_loop_body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api",
        required=False,
        default="numba",
        choices=["numba", "numba-dpex-kernel"],
        help="Start the version of functions using numba or numba-dpex API",
    )

    args = parser.parse_args()

    scenario(args.api)

    print("Done...")


if __name__ == "__main__":
    main()