# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
from numba import errors, types
from numba.core.typing.npydecl import parse_dtype, parse_shape
from numba.extending import intrinsic, overload
from numba.np.arrayobj import _empty_nd_impl, _parse_shape

from numba_dpex.core.types import DpnpNdArray


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


def _parse_empty_args(context, builder, sig, args):
    """
    Parse the arguments of a dpnp.empty(), .zeros() or .ones() call.
    """
    arrtype = sig.return_type

    arrshapetype = sig.args[0]
    arrshape = args[0]
    shape = _parse_shape(context, builder, arrshapetype, arrshape)

    queue = args[-1]
    return (arrtype, shape, queue)


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
