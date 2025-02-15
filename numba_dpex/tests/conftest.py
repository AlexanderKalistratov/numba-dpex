#! /usr/bin/env python

# Copyright 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import pytest

offload_devices = [
    "opencl:gpu:0",
    "level_zero:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=offload_devices, scope="module")
def offload_device(request):
    return request.param
