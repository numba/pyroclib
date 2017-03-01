from __future__ import print_function, division

import ctypes
from contextlib import contextmanager

import numpy as np
from numba.hsa.hsadrv import devicearray

from ..cwrappers import c_complex, c_doublecomplex


def _get_rocblas():
    from .binding import rocblas

    return rocblas


def _cast_ptr(ptr, cty):
    return ctypes.cast(ptr, ctypes.POINTER(cty))


def _inc_1d(ary):
    assert ary.ndim == 1
    return ary.strides[0] // ary.dtype.itemsize


def _lda(ary):
    "Leading strides"
    assert ary.ndim >= 2
    return ary.strides[-1] // ary.dtype.itemsize


def _bsa(ary):
    "Batch strides"
    assert ary.ndim == 3
    return ary.strides[0] // ary.dtype.itemsize


def _dim_2d_m(ary):
    assert ary.ndim == 2
    return ary.shape[0]


def _dim_2d_n(ary):
    assert ary.ndim == 2
    return ary.shape[1]


def _dim_2d_square(ary):
    assert ary.ndim == 2
    assert ary.shape[0] == ary.shape[1]
    return ary.shape[0]


def _size_1d(ary):
    return ary.size


def _string_arg_matches(arg, match):
    if isinstance(arg, str):
        arg = arg.lower()
        return arg.startswith(match[0]) or arg == match
    return False


def _get_contig(arr):
    if arr.flags.c_contiguous:
        return 'C'
    elif arr.flags.f_contiguous:
        return 'F'
    else:
        return 'A'


_dtype_charcode = {
    np.dtype(np.float32): 's',
    np.dtype(np.float64): 'd',
    np.dtype(np.complex64): 'c',
    np.dtype(np.complex128): 'z',
}


_ctypes_mapping = {
    np.dtype(np.float32): ctypes.c_float,
    np.dtype(np.float64): ctypes.c_double,
    np.dtype(np.complex64): c_complex,
    np.dtype(np.complex128): c_doublecomplex,
}


class _Dispatch(object):
    def __init__(self, blas, fname, dtype):
        self._binding = blas._rocblas
        ch = _dtype_charcode[dtype]
        self.fn = getattr(blas._rocblas, ch + fname)
        self.ctype = _ctypes_mapping[dtype]
        self.stream = blas.stream
        self._auto_devary = []

    def __call__(self, **kwargs):
        return self.fn(**kwargs)

    def parse_trans(self, trans):
        if trans is None:
            return self._binding.operation.none
        elif _string_arg_matches(trans, 'transpose'):
            return self._binding.operation.transpose
        elif _string_arg_matches(trans, 'conjugate'):
            return self._binding.operation.conjugate
        raise ValueError('unknown transformation: {!r}'.format(trans))

    def parse_uplo(self, uplo):
        if uplo is None:
            return self._binding.fill.full
        elif _string_arg_matches(uplo, 'upper'):
            return self._binding.fill.upper
        elif _string_arg_matches(uplo, 'lower'):
            return self._binding.fill.lower

    def parse_unit(self, unit):
        if unit:
            return self._binding.diagonal.unit
        else:
            return self._binding.diagonal.non_unit

    def parse_side(self, side):
        if side is None or _string_arg_matches(side, 'both'):
            return self._binding.side.both
        elif _string_arg_matches(side, 'left'):
            return self._binding.side.left
        elif _string_arg_matches(side, 'right'):
            return self._binding.side.right
        else:
            raise ValueError('unknown side: {!r}'.format(side))

    def host_ptr(self, val):
        return ctypes.byref(self.ctype(val))

    def device_ptr(self, ary):
        devary, conv = devicearray.auto_device(ary, context=None,
                                               stream=self.stream)
        if conv:
            self._auto_devary.append(lambda: devary.copy_to_host(ary))
        return _cast_ptr(devary.device_ctypes_pointer, self.ctype)

    def cleanup(self):
        for fn in self._auto_devary:
            fn()


class Blas(object):
    def __init__(self, stream=None):
        self._rocblas = _get_rocblas()
        self._stream = stream
        self._handle = self._rocblas.handle()
        self._rocblas.create_handle(ctypes.byref(self._handle))
        # XXX add finalizer

    @property
    def stream(self):
        return self._stream

    @contextmanager
    def _get_dispatcher(self, fname, dtype):
        disp = _Dispatch(self, fname, dtype)
        yield disp
        disp.cleanup()
        self._rocblas.synchronize()

    #
    # Level 1
    #

    def scal(self, alpha, x):
        """Compute x = x * alpha
        Returns x
        """
        with self._get_dispatcher('scal', x.dtype) as disp:
            disp(handle=self._handle, alpha=disp.host_ptr(alpha),
                 x=disp.device_ptr(x), incx=_inc_1d(x), n=_size_1d(x))
        return x

    def copy(self, x, y):
        """Copies x to y
        Returns y
        """
        with self._get_dispatcher('copy', y.dtype) as disp:
            disp(handle=self._handle, n=_size_1d(x),
                 x=disp.device_ptr(x), incx=_inc_1d(x),
                 y=disp.device_ptr(y), incy=_inc_1d(y))
        return y

    def dot(self, x, y):
        """Compute the dot product.
        Returns `sum(x * y)`
        """
        with self._get_dispatcher('dot', x.dtype) as disp:
            result = disp.ctype(0)
            disp(handle=self._handle, n=_size_1d(x),
                 x=disp.device_ptr(x), incx=_inc_1d(x),
                 y=disp.device_ptr(y), incy=_inc_1d(y),
                 result=ctypes.byref(result))
        return result.value

    def swap(self, x, y):
        """Swap values in x and y.
        Returns None
        """
        with self._get_dispatcher('swap', x.dtype) as disp:
            disp(handle=self._handle, n=_size_1d(x),
                 x=disp.device_ptr(x), incx=_inc_1d(x),
                 y=disp.device_ptr(y), incy=_inc_1d(y))

    def axpy(self, alpha, x, y):
        """Computes alpha y = alpha * x + y
        Returns y
        """
        with self._get_dispatcher('axpy', x.dtype) as disp:
            disp(handle=self._handle, n=_size_1d(x),
                 alpha=disp.host_ptr(alpha),
                 x=disp.device_ptr(x), incx=_inc_1d(x),
                 y=disp.device_ptr(y), incy=_inc_1d(y))
        return y

    def asum(self, x):
        """
        Returns sum(x)
        """
        with self._get_dispatcher('asum', x.dtype) as disp:
            result = disp.ctype(0)
            disp(handle=self._handle, n=_size_1d(x),
                 x=disp.device_ptr(x), incx=_inc_1d(x),
                 result=ctypes.byref(result))
        return result.value

    def nrm2(self, x):
        """
        Returns L2 norm
        """
        with self._get_dispatcher('nrm2', x.dtype) as disp:
            result = disp.ctype(0)
            disp(handle=self._handle, n=_size_1d(x),
                 x=disp.device_ptr(x), incx=_inc_1d(x),
                 result=ctypes.byref(result))
        return result.value

    def amax(self, x):
        """
        Returns index of maximum value
        """
        with self._get_dispatcher('amax', x.dtype) as disp:
            result = ctypes.c_int(0)
            disp(handle=self._handle, n=_size_1d(x),
                 x=disp.device_ptr(x), incx=_inc_1d(x),
                 result=ctypes.byref(result))
        return result.value

    def amin(self, x):
        """
        Returns index of minimum value
        """
        with self._get_dispatcher('amin', x.dtype) as disp:
            result = ctypes.c_int(0)
            disp(handle=self._handle, n=_size_1d(x),
                 x=disp.device_ptr(x), incx=_inc_1d(x),
                 result=ctypes.byref(result))
        return result.value

    #
    # Level 2
    #

    def gemv(self, alpha, A, x, beta, y, trans=None):
        """Computes matrix-vector multiplication
        y = alpha * A * x + beta * y
        """
        with self._get_dispatcher('gemv', A.dtype) as disp:
            disp(handle=self._handle,
                 trans=disp.parse_trans(trans),
                 m=_dim_2d_m(A), n=_dim_2d_n(A),
                 alpha=disp.host_ptr(alpha),
                 A=disp.device_ptr(A), lda=_lda(A),
                 x=disp.device_ptr(x), incx=_inc_1d(x),
                 beta=disp.host_ptr(beta),
                 y=disp.device_ptr(y), incy=_inc_1d(x))
        return y

    #
    # Level 3
    #

    def trtri(self, A, invA, uplo=None, unit=None):
        """Compute the inverse of A
        Returns invA
        """
        assert _get_contig(A) == 'F'
        with self._get_dispatcher('trtri', A.dtype) as disp:
            disp(handle=self._handle,
                 uplo=disp.parse_uplo(uplo),
                 diag=disp.parse_unit(unit),
                 n=_dim_2d_square(A),
                 A=disp.device_ptr(A), lda=_lda(A),
                 invA=disp.device_ptr(invA), ldinvA=_lda(invA))
        return invA

    def trtri_batched(self, A, invA, uplo=None, unit=None):
        assert A.ndim == 3
        assert A.shape == invA.shape
        assert _get_contig(A[0]) == 'F'
        assert _get_contig(invA[0]) == 'F'
        assert A.strides[0] == max(A.strides)
        assert invA.strides[0] == max(invA.strides)
        with self._get_dispatcher('trtri_batched', A.dtype) as disp:
            disp(handle=self._handle,
                 uplo=disp.parse_uplo(uplo),
                 diag=disp.parse_unit(unit),
                 n=A.shape[-1], batch_count=A.shape[0],
                 A=disp.device_ptr(A), lda=_lda(A), bsa=_bsa(A),
                 invA=disp.device_ptr(invA), ldinvA=_lda(invA),
                 bsinvA=_bsa(invA))
        return invA

    def trsm(self, alpha, A, B, side=None, uplo=None, transA=None, unit=None):
        """Solves A * X = B, by asumming A is a triangular matrix.
        Note: the result X is written to B
        Returns B
        """
        assert _get_contig(A) == 'F'
        with self._get_dispatcher('trsm', A.dtype) as disp:
            disp(handle=self._handle,
                 side=disp.parse_side(side),
                 uplo=disp.parse_uplo(uplo),
                 transA=disp.parse_trans(transA),
                 diag=disp.parse_unit(unit),
                 alpha=disp.host_ptr(alpha),
                 m=_dim_2d_m(B), n=_dim_2d_n(B),
                 A=disp.device_ptr(A), lda=_lda(A),
                 B=disp.device_ptr(B), ldb=_lda(B))
        return B

    def gemm(self, alpha, A, B, beta, C, transa=None, transb=None):
        """Computes alpha * A * B + beta * C
        Returns C
        """
        Acontig = _get_contig(A)
        Bcontig = _get_contig(B)
        Ccontig = _get_contig(C)
        assert Acontig == Bcontig and Bcontig == Ccontig
        assert Acontig == 'F'
        with self._get_dispatcher('gemm', A.dtype) as disp:
            disp(handle=self._handle,
                 order=self._rocblas.order.column,
                 transa=disp.parse_trans(transa),
                 transb=disp.parse_trans(transb),
                 m=_dim_2d_m(C), n=_dim_2d_n(C), k=A.shape[1],
                 lda=_lda(A), ldb=_lda(B), ldc=_lda(C),
                 alpha=disp.host_ptr(alpha),
                 beta=disp.host_ptr(beta),
                 A=disp.device_ptr(A),
                 B=disp.device_ptr(B),
                 C=disp.device_ptr(C))
        return C
