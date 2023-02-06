# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
from llvmlite import ir
from llvmlite.ir import Constant
from numba import errors, types
from numba.core import cgutils
from numba.core.typing import signature
from numba.core.typing.npydecl import parse_shape
from numba.extending import intrinsic, overload, overload_classmethod
from numba.np.arrayobj import (
    _parse_empty_args,
    get_itemsize,
    make_array,
    populate_array,
)

from numba_dpex.core.types import DpnpNdArray

# ------------------------------------------------------------------------------
# Helps to parse dpnp constructor arguments


def _parse_usm_type(usm_type):
    """
    Returns the usm_type, if it is a string literal.
    """
    from numba.core.errors import TypingError

    if isinstance(usm_type, types.StringLiteral):
        usm_type_str = usm_type.literal_value
        if usm_type_str not in ["shared", "device", "host"]:
            msg = f"Invalid usm_type specified: '{usm_type_str}'"
            raise TypingError(msg)
        return usm_type_str
    else:
        raise TypeError


def _parse_device_filter_string(device):
    """
    Returns the device filter string, if it is a string literal.
    """
    from numba.core.errors import TypingError

    if isinstance(device, types.StringLiteral):
        device_filter_str = device.literal_value
        return device_filter_str
    else:
        raise TypeError


# ------------------------------------------------------------------------------
# Helper functions to support dpnp array constructors

# FIXME: The _empty_nd_impl was copied over *as it is* from numba.np.arrayobj.
# However, we cannot use it yet as the `_call_allocator` function needs to be
# tailored to our needs. Specifically, we need to pass the device string so that
# a correct type of external allocator may be created for the NRT_MemInfo
# object.


def _empty_nd_impl(context, builder, arrtype, shapes):
    """Utility function used for allocating a new array during LLVM code
    generation (lowering).  Given a target context, builder, array
    type, and a tuple or list of lowered dimension sizes, returns a
    LLVM value pointing at a Numba runtime allocated array.
    """
    arycls = make_array(arrtype)
    ary = arycls(context, builder)

    datatype = context.get_data_type(arrtype.dtype)
    itemsize = context.get_constant(types.intp, get_itemsize(context, arrtype))

    # compute array length
    arrlen = context.get_constant(types.intp, 1)
    overflow = Constant(ir.IntType(1), 0)
    for s in shapes:
        arrlen_mult = builder.smul_with_overflow(arrlen, s)
        arrlen = builder.extract_value(arrlen_mult, 0)
        overflow = builder.or_(overflow, builder.extract_value(arrlen_mult, 1))

    if arrtype.ndim == 0:
        strides = ()
    elif arrtype.layout == "C":
        strides = [itemsize]
        for dimension_size in reversed(shapes[1:]):
            strides.append(builder.mul(strides[-1], dimension_size))
        strides = tuple(reversed(strides))
    elif arrtype.layout == "F":
        strides = [itemsize]
        for dimension_size in shapes[:-1]:
            strides.append(builder.mul(strides[-1], dimension_size))
        strides = tuple(strides)
    else:
        raise NotImplementedError(
            "Don't know how to allocate array with layout '{0}'.".format(
                arrtype.layout
            )
        )

    # Check overflow, numpy also does this after checking order
    allocsize_mult = builder.smul_with_overflow(arrlen, itemsize)
    allocsize = builder.extract_value(allocsize_mult, 0)
    overflow = builder.or_(overflow, builder.extract_value(allocsize_mult, 1))

    with builder.if_then(overflow, likely=False):
        # Raise same error as numpy, see:
        # https://github.com/numpy/numpy/blob/2a488fe76a0f732dc418d03b452caace161673da/numpy/core/src/multiarray/ctors.c#L1095-L1101    # noqa: E501
        context.call_conv.return_user_exc(
            builder,
            ValueError,
            (
                "array is too big; `arr.size * arr.dtype.itemsize` is larger "
                "than the maximum possible size.",
            ),
        )

    dtype = arrtype.dtype
    align_val = context.get_preferred_array_alignment(dtype)
    align = context.get_constant(types.uint32, align_val)
    args = (context.get_dummy_value(), allocsize, align)

    mip = types.MemInfoPointer(types.voidptr)
    arytypeclass = types.TypeRef(type(arrtype))
    argtypes = signature(mip, arytypeclass, types.intp, types.uint32)

    meminfo = context.compile_internal(builder, _call_allocator, argtypes, args)
    data = context.nrt.meminfo_data(builder, meminfo)

    intp_t = context.get_value_type(types.intp)
    shape_array = cgutils.pack_array(builder, shapes, ty=intp_t)
    strides_array = cgutils.pack_array(builder, strides, ty=intp_t)

    populate_array(
        ary,
        data=builder.bitcast(data, datatype.as_pointer()),
        shape=shape_array,
        strides=strides_array,
        itemsize=itemsize,
        meminfo=meminfo,
    )

    return ary


@overload_classmethod(DpnpNdArray, "_allocate")
def _ol_array_allocate(cls, allocsize, align):
    """Implements a Numba-only default target (cpu) classmethod on the
    array type.
    """

    def impl(cls, allocsize, align):
        return intrin_alloc(allocsize, align)

    return impl


def _call_allocator(arrtype, size, align):
    """Trampoline to call the intrinsic used for allocation"""
    return arrtype._allocate(size, align)


@intrinsic
def intrin_alloc(typingctx, allocsize, align):
    """Intrinsic to call into the allocator for Array"""

    def codegen(context, builder, signature, args):
        [allocsize, align] = args
        meminfo = context.nrt.meminfo_alloc_aligned(builder, allocsize, align)
        return meminfo

    mip = types.MemInfoPointer(types.voidptr)  # return untyped pointer
    sig = signature(mip, allocsize, align)
    return sig, codegen


@intrinsic
def impl_dpnp_empty(
    tyctx,
    ty_shape,
    ty_dtype,
    ty_usm_type,
    ty_device,
    ty_sycl_queue,
    ty_retty_ref,
):
    ty_retty = ty_retty_ref.instance_type

    sig = ty_retty(
        ty_shape, ty_dtype, ty_usm_type, ty_device, ty_sycl_queue, ty_retty_ref
    )

    def codegen(cgctx, builder, sig, llargs):
        arrtype = _parse_empty_args(cgctx, builder, sig, llargs)
        ary = _empty_nd_impl(cgctx, builder, *arrtype)
        return ary._getvalue()

    return sig, codegen


# ------------------------------------------------------------------------------
# Dpnp array constructor overloads


@overload(dpnp.empty, prefer_literal=True)
def type_dpnp_empty(
    shape, dtype=None, usm_type=None, device=None, order="C", sycl_queue=None
):
    """Implementation of an overload to support dpnp.empty inside a jit
    function.

    Args:
        shape (_type_): _description_
        dtype (_type_, optional): _description_. Defaults to None.
        usm_type (_type_, optional): _description_. Defaults to None.
        device (_type_, optional): _description_. Defaults to None.
        sycl_queue (_type_, optional): _description_. Defaults to None.

    Raises:
        ...: _description_
        errors.TypingError: _description_

    Returns:
        _type_: _description_
    """

    ndim = parse_shape(shape)
    if not ndim:
        raise ...

    if usm_type is not None:
        usm_type = _parse_usm_type(usm_type)
    else:
        usm_type = "device"

    if device is not None:
        device = _parse_device_filter_string(device)

    if ndim is not None:
        retty = DpnpNdArray(
            dtype=dtype,
            ndim=ndim,
            layout=order,
            usm_type=usm_type,
            device=device,
            queue=None,
        )

        def impl(
            shape,
            dtype=None,
            usm_type=None,
            device=None,
            order="C",
            sycl_queue=None,
        ):
            return impl_dpnp_empty(
                shape, dtype, usm_type, device, sycl_queue, retty
            )

        return impl
    else:
        msg = (
            f"Cannot parse input types to function dpnp.empty({shape}, {dtype})"
        )
        raise errors.TypingError(msg)
