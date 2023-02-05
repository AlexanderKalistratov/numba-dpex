# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


from numba.core import cgutils
from numba.core.errors import NumbaNotImplementedError
from numba.core.pythonapi import NativeValue, PythonAPI, box, unbox
from numba.np import numpy_support

from numba_dpex.core.exceptions import UnreachableError

from .usm_ndarray_type import USMNdArray


class DpnpNdArray(USMNdArray):
    """
    The Numba type to represent an dpnp.ndarray. The type has the same
    structure as USMNdArray used to represnet dpctl.tensor.usm_ndarray.
    """

    pass


# --------------- Boxing/Unboxing logic for dpnp.ndarray ----------------------#


class _DpnpAdapterGenerator:
    """Helper class to translate a dpnp.ndarray Python object to a
    corresponding Numba representation using a dpex runtime C function call.
    """

    def __init__(self):
        self.error = None

    def from_python(self, pyapi: PythonAPI, obj, ptr):
        """Generates a call to dpex_rt_dpnp_ndarray_from_python C function
        defined in the dpex runtime.
        """
        import llvmlite.llvmpy.core as lc
        from llvmlite.llvmpy.core import Type

        fnty = Type.function(Type.int(), [pyapi.pyobj, pyapi.voidptr])
        fn = pyapi._get_function(fnty, "DPEXRT_sycl_usm_ndarray_from_python")
        fn.args[0].add_attribute(lc.ATTR_NO_CAPTURE)
        fn.args[1].add_attribute(lc.ATTR_NO_CAPTURE)

        self.error = pyapi.builder.call(fn, (obj, ptr))

        return self.error

    def to_python(self, pyapi: PythonAPI, aryty, ary, dtypeptr):
        from llvmlite.ir import IntType
        from llvmlite.llvmpy.core import ATTR_NO_CAPTURE, Type
        from numba.core import types

        args = [
            pyapi.voidptr,
            pyapi.pyobj,
            IntType(32),
            IntType(32),
            pyapi.pyobj,
        ]
        fnty = Type.function(pyapi.pyobj, args)
        fn = pyapi._get_function(
            fnty, "DPEXRT_sycl_usm_ndarray_to_python_acqref"
        )
        fn.args[0].add_attribute(ATTR_NO_CAPTURE)

        aryptr = cgutils.alloca_once_value(pyapi.builder, ary)
        ptr = pyapi.builder.bitcast(aryptr, pyapi.voidptr)

        # Embed the Python type of the array (maybe subclass) in the LLVM IR.
        serialized = pyapi.serialize_object(aryty.box_type)
        serial_aryty_pytype = pyapi.unserialize(serialized)

        ndim = pyapi.context.get_constant(types.int32, aryty.ndim)
        writable = pyapi.context.get_constant(types.int32, int(aryty.mutable))

        args = [ptr, serial_aryty_pytype, ndim, writable, dtypeptr]
        return pyapi.builder.call(fn, args)


@unbox(DpnpNdArray)
def unbox_dpnp_nd_array(typ, obj, c):
    """Converts a dpnp.ndarray object to a Numba internal array structure.

    Args:
        typ : The Numba type of the PyObject
        obj : The actual PyObject to be unboxed
        c :

    Returns:
        _type_: _description_
    """
    # Reusing the numba.core.base.BaseContext's make_array function to get a
    # struct allocated. The same struct is used for numpy.ndarray
    # and dpnp.ndarray. It is possible to do so, as the extra information
    # specific to dpnp.ndarray such as sycl_queue is inferred statically and
    # stored as part of the DpnpNdArray type.

    # --------------- Original Numba comment from @ubox(types.Array)
    #
    # This is necessary because unbox_buffer() does not work on some
    # dtypes, e.g. datetime64 and timedelta64.
    # TODO check matching dtype.
    #      currently, mismatching dtype will still work and causes
    #      potential memory corruption
    #
    # --------------- End of Numba comment from @ubox(types.Array)
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder)
    aryptr = nativeary._getpointer()

    ptr = c.builder.bitcast(aryptr, c.pyapi.voidptr)
    # FIXME : We need to check if Numba_RT as well as DPEX RT are enabled.
    if c.context.enable_nrt:
        adapter = _DpnpAdapterGenerator()
        errcode = adapter.from_python(c.pyapi, obj, ptr)
    else:
        raise UnreachableError

    # TODO: here we have minimal typechecking by the itemsize.
    #       need to do better
    try:
        expected_itemsize = numpy_support.as_dtype(typ.dtype).itemsize
    except NumbaNotImplementedError:
        # Don't check types that can't be `as_dtype()`-ed
        itemsize_mismatch = cgutils.false_bit
    else:
        expected_itemsize = nativeary.itemsize.type(expected_itemsize)
        itemsize_mismatch = c.builder.icmp_unsigned(
            "!=",
            nativeary.itemsize,
            expected_itemsize,
        )

    failed = c.builder.or_(
        cgutils.is_not_null(c.builder, errcode),
        itemsize_mismatch,
    )
    # Handle error
    with c.builder.if_then(failed, likely=False):
        c.pyapi.err_set_string(
            "PyExc_TypeError",
            "can't unbox array from PyObject into "
            "native value.  The object maybe of a "
            "different type",
        )
    return NativeValue(c.builder.load(aryptr), is_error=failed)


@box(DpnpNdArray)
def box_array(typ, val, c):
    if c.context.enable_nrt:
        np_dtype = numpy_support.as_dtype(typ.dtype)
        dtypeptr = c.env_manager.read_const(c.env_manager.add_const(np_dtype))

        adapter = _DpnpAdapterGenerator()
        newary = adapter.to_python(c.pyapi, typ, val, dtypeptr)

        # Steals NRT ref
        # Refer:
        #   numba.core.base.nrt -> numba.core.runtime.context -> decref
        #   The `NRT_decref` function is generated directly as LLVM IR inside
        #   numba.core.runtime.nrtdynmod.py
        c.context.nrt.decref(c.builder, typ, val)

        return newary
    else:
        raise UnreachableError
