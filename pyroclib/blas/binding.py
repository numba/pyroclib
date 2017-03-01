from __future__ import print_function, division, absolute_import

import ctypes

from ..cwrappers import (enum_int, opaque_pointer, Wrapper, IntEnum,
                         c_complex, c_doublecomplex)


class RocBLASError(RuntimeError):
    pass


def _define_template(apidict, symbol, args, **kwargs):
    assert kwargs  # assert at least one keyword
    paramlens = [len(v) for v in kwargs.values()]
    assert all(paramlens[0] == x for x in paramlens)  # assert equal length
    length = paramlens[0]
    for i in range(length):
        params = dict((k, v[i]) for k, v in kwargs.items())
        apidict[symbol.format(**params)] = args.format(**params)


class RocBLAS(object):
    types = dict(
        # Enums
        rocblas_status=enum_int,
        rocblas_operation=enum_int,
        rocblas_fill=enum_int,
        rocblas_diagonal=enum_int,
        rocblas_side=enum_int,
        rocblas_order=enum_int,

        # Misc
        rocblas_handle=opaque_pointer('rocblas_handle'),
        rocblas_int=ctypes.c_int,
        rocblas_float_complex=c_complex,
        rocblas_double_complex=c_doublecomplex,
    )

    error_class = RocBLASError
    status_type = 'rocblas_status'

    common_prefix = 'rocblas_'

    class status(IntEnum):
        success = 0
        invalid_handle = 1
        not_implemented = 2
        invalid_pointer = 3
        invalid_size = 3
        memory_error = 4
        internal_error = 5

    class operation(IntEnum):
        none = 111
        transpose = 112
        conjugate_transpose = 113

    class fill(IntEnum):
        upper = 121
        lower = 122
        full = 123

    class diagonal(IntEnum):
        non_unit = 131
        unit = 132

    class side(IntEnum):
        # Multiply general matrix by symmetric,
        # Hermitian or triangular matrix on the left.
        left = 141
        # Multiply general matrix by symmetric,
        # Hermitian or triangular matrix on the right.
        right = 142
        both = 143

    class order(IntEnum):
        row = 101
        column = 102

    enums = {}

    api = {
        # Misc
        'rocblas_create_handle': 'rocblas_handle *handle',
        'rocblas_destroy_handle': 'rocblas_handle handle',
    }
    real_char = 'sd'
    real_types = ['float', 'double']

    sdcz_char = 'sdcz'
    sdcz_types = ['float', 'double', 'rocblas_float_complex',
                  'rocblas_double_complex']

    #
    # Level 1
    #
    _define_template(api, 'rocblas_{Tc}scal', '''
        rocblas_handle handle, rocblas_int n,
        {T} *alpha, {T} *x, rocblas_int incx
        ''', Tc=sdcz_char, T=sdcz_types)

    _define_template(api, 'rocblas_{Tc}copy', '''
        rocblas_handle handle, rocblas_int n,
        {T} *x, rocblas_int incx,
        {T} *y, rocblas_int incy
        ''', Tc=sdcz_char, T=sdcz_types)

    _define_template(api, 'rocblas_{Tc}dot', '''
        rocblas_handle handle, rocblas_int n,
        {T} *x, rocblas_int incx,
        {T} *y, rocblas_int incy,
        {T}* result
        ''', Tc=real_char, T=real_types)

    _define_template(api, 'rocblas_{Tc}swap', '''
        rocblas_handle handle, rocblas_int n,
        {T} *x, rocblas_int incx,
        {T} *y, rocblas_int incy
        ''', Tc=sdcz_char, T=sdcz_types)

    _define_template(api, 'rocblas_{Tc}axpy', '''
        rocblas_handle handle, rocblas_int n,
        {T} *alpha, {T} *x, rocblas_int incx,
        {T} *y, rocblas_int incy
        ''', Tc=sdcz_char, T=sdcz_types)

    _define_template(api, 'rocblas_{Tc}asum', '''
        rocblas_handle handle, rocblas_int n,
        {T} *x, rocblas_int incx, {T} *result,
        ''', Tc=real_char, T=real_types)

    _define_template(api, 'rocblas_{Tc}nrm2', '''
        rocblas_handle handle, rocblas_int n,
        {T} *x, rocblas_int incx, {T} *result,
        ''', Tc=real_char, T=real_types)

    _define_template(api, 'rocblas_{Tc}amax', '''
        rocblas_handle handle, rocblas_int n,
        {T} *x, rocblas_int incx,
        rocblas_int *result,
        ''', Tc=real_char, T=real_types)

    _define_template(api, 'rocblas_{Tc}amin', '''
        rocblas_handle handle, rocblas_int n,
        {T} *x, rocblas_int incx,
        rocblas_int *result,
        ''', Tc=real_char, T=real_types)

    #
    # Level 2
    #

    _define_template(api, 'rocblas_{Tc}gemv', '''
        rocblas_handle handle,  rocblas_operation trans,
        rocblas_int m, rocblas_int n,
        {T} *alpha,
        {T} *A, rocblas_int lda,
        {T} *x, rocblas_int incx,
        {T} *beta,
        {T} *y, rocblas_int incy
        ''', Tc=real_char, T=real_types)

    #
    # Level 3
    #

    _define_template(api, 'rocblas_{Tc}trtri', '''
        rocblas_handle handle, rocblas_fill uplo, rocblas_diagonal diag,
        rocblas_int n, {T} *A, rocblas_int lda,
        {T} *invA, rocblas_int ldinvA
        ''', Tc=real_char, T=real_types)

    _define_template(api, 'rocblas_{Tc}trtri_batched', '''
        rocblas_handle handle, rocblas_fill uplo, rocblas_diagonal diag,
        rocblas_int n,
        {T} *A, rocblas_int lda, rocblas_int bsa,
        {T} *invA, rocblas_int ldinvA, rocblas_int bsinvA,
        rocblas_int batch_count
        ''', Tc=real_char, T=real_types)

    _define_template(api, 'rocblas_{Tc}trsm', '''
        rocblas_handle handle,
        rocblas_side side, rocblas_fill uplo,
        rocblas_operation transA, rocblas_diagonal diag,
        rocblas_int m, rocblas_int n,
        {T}* alpha,
        {T}* A, rocblas_int lda,
        {T}* B, rocblas_int ldb
        ''', Tc=real_char, T=real_types)

    _define_template(api, 'rocblas_{Tc}gemm', '''
        rocblas_handle handle,
        rocblas_order order,
        rocblas_operation transa, rocblas_operation transb,
        rocblas_int m, rocblas_int n, rocblas_int k,
        {T} *alpha,
        {T} *A, rocblas_int lda,
        {T} *B, rocblas_int ldb,
        {T} *beta,
        {T} *C, rocblas_int ldc
        ''', Tc=real_char, T=real_types)

    def __init__(self, lib):
        self._wrapper = Wrapper(self, lib)

    def handle(self):
        return self._wrapper.get_type('rocblas_handle')()

    def on_error(self, errcode):
        raise self.error_class(self.status(errcode))

    def synchronize(self):
        self._wrapper.lib.hipDeviceSynchronize()


# XXX hardcoded for now
_path = '/home/amd_user/rocBLAS/build/library-build/src/librocblas-hcc.so'
rocblas = RocBLAS(ctypes.CDLL(_path))
