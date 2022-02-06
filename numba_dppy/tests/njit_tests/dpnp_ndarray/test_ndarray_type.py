################################################################################
#                                 Numba-DPPY
#
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import pytest
from dpctl.tensor.numpy_usm_shared import ndarray as dpctl_ndarray
from dpnp import ndarray as dpnp_ndarray
from numba import njit, typeof, types

from numba_dppy.numpy_usm_shared import UsmSharedArrayType
from numba_dppy.types import dpnp_ndarray_Type


@pytest.mark.parametrize(
    "array_type, shape, numba_type",
    [
        (dpctl_ndarray, [1], UsmSharedArrayType(types.float64, 1, "C")),
        (dpctl_ndarray, [1, 1], UsmSharedArrayType(types.float64, 2, "C")),
        (dpnp_ndarray, [1], dpnp_ndarray_Type()),
    ],
)
def test_typeof(array_type, shape, numba_type):
    array = array_type(shape)
    assert typeof(array) == numba_type


dpnp_mark = pytest.mark.xfail(
    raises=AttributeError, reason="No ndim in numba type"
)


@pytest.mark.parametrize(
    "array",
    [
        dpctl_ndarray([1]),
        pytest.param(dpnp_ndarray([1]), marks=dpnp_mark),
    ],
)
def test_njit(array):
    @njit
    def func(a):
        return a

    func(array)