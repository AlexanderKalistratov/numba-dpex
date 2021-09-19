# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dpctl
import numpy as np
import pytest

import numba_dppy as dppy
from numba_dppy.tests._helper import skip_test


def f(a):
    return a


list_of_sig = [
    None,
    ("int32[::1](int32[::1])"),
]


@pytest.fixture(params=list_of_sig)
def sig(request):
    return request.param


def test_return(offload_device, sig):
    if skip_test(offload_device):
        pytest.skip()

    a = np.array(np.random.random(122), np.int32)

    with pytest.raises(TypeError):
        kernel = dppy.kernel(sig)(f)

        device = dpctl.SyclDevice(offload_device)
        with dppy.offload_to_sycl_device(device):
            kernel[a.size, dppy.DEFAULT_LOCAL_SIZE](a)