# Copyright 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import math

import dpctl
import numpy as np

import numba_dpex as dpex
from numba_dpex.tests._helper import skip_no_opencl_gpu


@dpex.kernel
def reduction_kernel(A, R, stride):
    i = dpex.get_global_id(0)
    # sum two element
    R[i] = A[i] + A[i + stride]
    # store the sum to be used in nex iteration
    A[i] = R[i]


@skip_no_opencl_gpu
class TestSumReduction:
    def test_sum_reduction(self):
        # This test will only work for even case
        N = 1024
        assert N % 2 == 0

        A = np.array(np.random.random(N), dtype=np.float32)
        A_copy = A.copy()
        # at max we will require half the size of A to store sum
        R = np.array(np.random.random(math.ceil(N / 2)), dtype=np.float32)

        device = dpctl.SyclDevice("opencl:gpu")
        with dpctl.device_context(device):
            total = N

            while total > 1:
                # call kernel
                global_size = total // 2
                reduction_kernel[global_size, dpex.DEFAULT_LOCAL_SIZE](
                    A, R, global_size
                )
                total = total // 2

            result = A_copy.sum()
            max_abs_err = result - R[0]
            assert max_abs_err < 1e-4
