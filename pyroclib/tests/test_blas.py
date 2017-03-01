from __future__ import print_function, absolute_import, division

import numpy as np
import ctypes

from numba import hsa
import numba.unittest_support as unittest
from numba.hsa.hsadrv.driver import dgpu_present

from pyroc.blas import Blas


@unittest.skipUnless(dgpu_present(), 'test only on dGPU system')
class TestRocBlasBinding(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from pyroc.blas.binding import rocblas

        self.rocblas = rocblas

    def test_handle(self):
        hdl = self.rocblas.handle()
        self.rocblas.create_handle(ctypes.byref(hdl))
        self.rocblas.destroy_handle(hdl)

    def test_sscal(self):
        hdl = self.rocblas.handle()
        self.rocblas.create_handle(ctypes.byref(hdl))

        arr = np.random.random(10).astype(dtype=np.float32)

        dev_buffer = hsa.to_device(arr)
        x = ctypes.cast(dev_buffer.device_ctypes_pointer,
                        ctypes.POINTER(ctypes.c_float))
        n = arr.size
        alpha = ctypes.c_float(0.5)
        incx = 1
        self.rocblas.sscal(hdl, n, ctypes.byref(alpha), x, incx)

        out = dev_buffer.copy_to_host()

        self.rocblas.destroy_handle(hdl)

        np.testing.assert_almost_equal(arr * 0.5, out)


@unittest.skipUnless(dgpu_present(), 'test only on dGPU system')
class TestRocBlasAPI(unittest.TestCase):

    def run_test_real(self, fn):
        for dtype in [np.float32, np.float64]:
            fn(dtype)

    def run_test_complex(self, fn):
        for dtype in [np.complex64, np.complex128]:
            fn(dtype)

    def check_scal(self, dtype):
        blas = Blas()
        arr = np.random.random(10).astype(dtype=dtype)
        alpha = 0.5
        expect = arr * alpha
        blas.scal(alpha=alpha, x=arr)
        np.testing.assert_almost_equal(expect, arr)

    def test_scal_real(self):
        self.run_test_real(self.check_scal)

    @unittest.expectedFailure
    def test_scal_complex(self):
        self.run_test_complex(self.check_scal)

    def check_copy(self, dtype):
        blas = Blas()
        arr = np.random.random(10).astype(dtype=dtype)
        expect = arr.copy()
        got = np.zeros_like(arr)
        blas.copy(x=arr, y=got)
        np.testing.assert_almost_equal(expect, got)

    def test_copy_real(self):
        self.run_test_real(self.check_copy)

    def check_dot(self, dtype):
        blas = Blas()
        x = np.random.random(10).astype(dtype=dtype)
        y = np.random.random(10).astype(dtype=dtype)
        expect = np.dot(x, y)
        got = blas.dot(x=x, y=y)
        np.testing.assert_almost_equal(expect, got, decimal=6)

    def test_dot_real(self):
        self.run_test_real(self.check_dot)

    def check_swap(self, dtype):
        blas = Blas()
        x = np.random.random(10).astype(dtype=dtype)
        y = np.random.random(10).astype(dtype=dtype)
        expect_y, expect_x = x.copy(), y.copy()
        blas.swap(x=x, y=y)
        np.testing.assert_almost_equal(x, expect_x)
        np.testing.assert_almost_equal(y, expect_y)

    def test_swap_real(self):
        self.run_test_real(self.check_swap)

    def check_axpy(self, dtype):
        blas = Blas()
        alpha = 3.23456789
        x = np.random.random(10).astype(dtype=dtype)
        y = np.random.random(10).astype(dtype=dtype)
        expect = alpha * x + y
        blas.axpy(alpha=alpha, x=x, y=y)
        np.testing.assert_almost_equal(expect, y)

    def test_axpy(self):
        self.run_test_real(self.check_axpy)

    def check_asum(self, dtype):
        blas = Blas()
        x = np.random.random(10).astype(dtype=dtype)
        expect = np.sum(x)
        got = blas.asum(x=x)
        np.testing.assert_almost_equal(expect, got, decimal=5)

    def test_asum(self):
        self.run_test_real(self.check_asum)

    def check_nrm2(self, dtype):
        blas = Blas()
        x = np.random.random(10).astype(dtype=dtype)
        expect = np.linalg.norm(x)
        got = blas.nrm2(x=x)
        np.testing.assert_almost_equal(expect, got, decimal=6)

    def test_nrm2(self):
        self.run_test_real(self.check_nrm2)

    def check_amax(self, dtype):
        blas = Blas()
        x = np.random.random(10).astype(dtype=dtype)
        expect = np.argmax(x)
        got = blas.amax(x=x)
        np.testing.assert_almost_equal(expect, got)

    def test_amax(self):
        self.run_test_real(self.check_amax)

    def check_amin(self, dtype):
        blas = Blas()
        x = np.random.random(10).astype(dtype=dtype)
        expect = np.argmax(x)
        got = blas.amin(x=x)
        np.testing.assert_almost_equal(expect, got)

    def test_amin(self):
        self.run_test_real(self.check_amin)

    #
    # Level 2
    #

    def check_gemv(self, dtype):
        blas = Blas()
        m = 3
        n = 4
        x = np.random.random(n).astype(dtype=dtype)
        y = np.random.random(m).astype(dtype=dtype)
        A = np.random.random((m, n)).astype(dtype=dtype, order='F')
        alpha = 0.7
        beta = 0.8
        expect = np.dot(alpha * x, A.T) + beta * y
        blas.gemv(alpha=alpha, beta=beta, x=x, y=y, A=A)
        np.testing.assert_almost_equal(expect, y, decimal=6)

    def test_gemv(self):
        self.run_test_real(self.check_gemv)

    #
    # Level 3
    #

    def check_trtri(self, dtype):
        blas = Blas()
        n = 4
        A = np.asfortranarray(np.triu(np.random.random((n, n)).astype(dtype=dtype)))
        invA = np.zeros_like(A)
        expect = np.linalg.inv(A)
        blas.trtri(A=A, invA=invA, uplo='upper')
        np.testing.assert_almost_equal(expect, invA, decimal=4)

    def test_trtri(self):
        self.run_test_real(self.check_trtri)

    def check_trtri_batched(self, dtype):
        blas = Blas()
        n = 4
        batch_count = 3
        As = [np.asfortranarray(np.triu(np.random.random((n, n)).astype(dtype=dtype)))
              for _ in range(batch_count)]
        A = np.stack(As)
        invA = np.zeros_like(A)
        expect = np.stack([np.linalg.inv(a) for a in A])
        blas.trtri_batched(A=A, invA=invA, uplo='upper')
        np.testing.assert_almost_equal(expect, invA, decimal=4)

    def test_trtri_batched(self):
        self.run_test_real(self.check_trtri_batched)

    def check_trsm(self, dtype):
        import scipy.linalg

        blas = Blas()
        m = 4
        n = 4
        A = np.asfortranarray(np.triu(np.random.random((n, n)).astype(dtype=dtype)))
        B = np.random.random((n, m)).astype(dtype=dtype, order='F')
        alpha = 0.7
        expect = scipy.linalg.solve_triangular(A, alpha * B)
        raise NotImplementedError('segfault!?!')
        #0  __GI___pthread_mutex_lock (mutex=0x0) at ../nptl/pthread_mutex_lock.c:66
        #1  0x00007ffff7b9f9bc in std::__1::mutex::lock() () from /usr/lib/x86_64-linux-gnu/libc++.so
        #2  0x00007fffe8120aeb in LockedAccessor<ihipCtxCriticalBase_t<std::__1::mutex> >::LockedAccessor(ihipCtxCriticalBase_t<std::__1::mutex>&, bool) ()
        # from /home/amd_user/rocBLAS/build/library-build/src/librocblas-hcc.so
        #3  0x00007fffe811cc98 in ihipStream_t::canSeePeerMemory(ihipCtx_t const*, ihipCtx_t*, ihipCtx_t*) ()
        # from /home/amd_user/rocBLAS/build/library-build/src/librocblas-hcc.so
        #4  0x00007fffe811d97f in ihipStream_t::locked_copySync(void*, void const*, unsigned long, unsigned int, bool) () from /home/amd_user/rocBLAS/build/library-build/src/librocblas-hcc.so
        #5  0x00007fffe8128ea6 in hipMemcpy ()
        blas.trsm(alpha=alpha, A=A, B=B, uplo='upper', side='left')
        np.testing.assert_almost_equal(expect, B, decimal=6)

    @unittest.expectedFailure
    def test_trsm(self):
        self.run_test_real(self.check_trsm)

    def check_gemm(self, dtype, order='F'):
        blas = Blas()
        m = 4
        n = 5
        k = 6
        A = np.random.random((n, k)).astype(dtype=dtype, order=order)
        B = np.random.random((k, m)).astype(dtype=dtype, order=order)
        C = np.random.random((n, m)).astype(dtype=dtype, order=order)
        alpha = 0.7
        beta = 0.8
        expect = alpha * np.dot(A, B) + beta * C
        blas.gemm(alpha=alpha, beta=beta, A=A, B=B, C=C)
        np.testing.assert_almost_equal(expect, C, decimal=6)

    def test_gemm(self):
        self.run_test_real(self.check_gemm)


if __name__ == '__main__':
    unittest.main()
