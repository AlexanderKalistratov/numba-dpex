# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
import pytest
from numba import njit

from numba_dpex.tests._helper import (
    assert_auto_offloading,
    filter_strings,
    is_gen12,
)

list_of_binary_ops = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "true_divide",
    "power",
    "remainder",
    "mod",
    "fmod",
    "hypot",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
]


@pytest.fixture(params=list_of_binary_ops)
def binary_op(request):
    return request.param


list_of_unary_ops = [
    "negative",
    "abs",
    "absolute",
    "fabs",
    "sign",
    "conj",
    "exp",
    "exp2",
    "log",
    "log2",
    "log10",
    "expm1",
    "log1p",
    "sqrt",
    "square",
    "reciprocal",
    "conjugate",
    "floor",
    "ceil",
    "trunc",
]


@pytest.fixture(params=list_of_unary_ops)
def unary_op(request):
    return request.param


list_of_dtypes = [
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    N = 2048
    a = np.array(np.random.random(N), request.param)
    b = np.array(np.random.random(N), request.param)
    return a, b


@pytest.mark.parametrize("filter_str", filter_strings)
def test_binary_ops(filter_str, binary_op, input_arrays):
    a, b = input_arrays
    binop = getattr(np, binary_op)
    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    @njit
    def f(a, b):
        return binop(a, b)

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), assert_auto_offloading():
        actual = f(a, b)

    expected = binop(a, b)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)


@pytest.mark.parametrize("filter_str", filter_strings)
def test_unary_ops(filter_str, unary_op, input_arrays):
    # FIXME: Why does sign fail on Gen12 discrete graphics card?
    skip_ops = ["sign", "log", "log2", "log10", "expm1"]
    if unary_op in skip_ops and is_gen12(filter_str):
        pytest.skip()

    a = input_arrays[0]
    uop = getattr(np, unary_op)
    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    @njit
    def f(a):
        return uop(a)

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), assert_auto_offloading():
        actual = f(a)

    expected = uop(a)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)
