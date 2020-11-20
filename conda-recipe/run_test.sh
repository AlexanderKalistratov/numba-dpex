#!/bin/bash

set -e

# For activating OpenCL CPU
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
source ${ONEAPI_ROOT}/tbb/latest/env/vars.sh

set -x

python -m numba.runtests -b -v -m -- numba_dppy.tests

exit 0
