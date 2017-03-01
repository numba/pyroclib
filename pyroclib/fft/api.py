from __future__ import print_function, absolute_import, division

import ctypes
import numpy as np

from numba import hsa
from numba.hsa.hsadrv import devicearray


def _get_rocfft():
    from .binding import rocfft

    return rocfft


def _auto_device(ary, stream):
    return devicearray.auto_device(ary, context=None, stream=stream)


def _fft_inplace_core(rocfft, dev_buffer, inverse):
    transform_type = (rocfft.transform_type.complex_forward
                      if inverse else rocfft.transform_type.complex_inverse)

    plan = rocfft.plan()

    placement = rocfft.placement.inplace
    transform_type = transform_type
    dtype = dev_buffer.dtype
    if dtype == np.dtype(np.complex64):
        precision = rocfft.precision.single
    else:
        raise TypeError("rocfft doesn't support double-prec type yet")

    dimensions = dev_buffer.ndim

    c_lengths = (ctypes.c_size_t * dimensions)(*reversed(dev_buffer.shape))
    lengths = ctypes.cast(c_lengths, ctypes.POINTER(ctypes.c_size_t))

    number_of_transforms = 1
    description = None
    info = rocfft.execution_info()
    rocfft.execution_info_create(ctypes.byref(info))

    rocfft.plan_create(plan=ctypes.byref(plan),
                       transform_type=transform_type,
                       precision=precision,
                       placement=placement,
                       dimensions=dimensions,
                       lengths=lengths,
                       number_of_transforms=number_of_transforms,
                       description=description)

    rocfft.execute(plan=plan,
                   in_buffer=ctypes.byref(dev_buffer.device_ctypes_pointer),
                   out_buffer=None,
                   info=info)

    rocfft.synchronize()
    rocfft.plan_destroy(plan=plan)
    rocfft.execution_info_destroy(info)
    return dev_buffer

#
# Simple one-off functions
#

# def fft(ary, out, stream=None):
#     '''Perform forward FFT on `ary` and output to `out`.

#     out --- can be a numpy array or a GPU device array with 1 <= ndim <= 3
#     stream --- a CUDA stream
#     '''
#     plan = FFTPlan(ary.shape, ary.dtype, out.dtype, stream=stream)
#     plan.forward(ary, out)
#     return out

# def ifft(ary, out, stream=None):
#     '''Perform inverse FFT on `ary` and output to `out`.

#     out --- can be a numpy array or a GPU device array with 1 <= ndim <= 3
#     stream --- a CUDA stream
#     '''
#     plan = FFTPlan(ary.shape, ary.dtype, out.dtype, stream=stream)
#     plan.inverse(ary, out)
#     return out


def fft_inplace(ary, stream=None):
    '''Perform inplace forward FFT. `ary` must have complex dtype.

    out --- can be a numpy array or a GPU device array with 1 <= ndim <= 3
    stream --- a HSA stream
    '''
    rocfft = _get_rocfft()
    d_ary, conv = _auto_device(ary, stream=stream)
    _fft_inplace_core(rocfft, d_ary, inverse=False)
    if conv:
        d_ary.copy_to_host(ary)
    return ary


def ifft_inplace(ary, stream=None):
    '''Perform inplace inverse FFT. `ary` must have complex dtype.

    out --- can be a numpy array or a GPU device array with 1 <= ndim <= 3
    stream --- a HSA stream
    '''
    rocfft = _get_rocfft()
    d_ary, conv = _auto_device(ary, stream=stream)
    _fft_inplace_core(rocfft, d_ary, inverse=True)
    if conv:
        d_ary.copy_to_host(ary)
    return ary

