from __future__ import print_function, absolute_import, division

import numpy as np
import ctypes


from numba import hsa
import numba.unittest_support as unittest
from numba.hsa.hsadrv.driver import dgpu_present

from pyroclib.fft.api import _fft_inplace_core, _get_rocfft
from pyroclib.fft import fft_inplace, ifft_inplace


@unittest.skipUnless(dgpu_present(), 'test only on dGPU system')
class TestRocFFT(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.rocfft = _get_rocfft()

    def fft_inplace_core(self, in_array, transform_type):
        return _fft_inplace_core(self.rocfft, in_array, transform_type)

    def fft_outofplace_core(self, in_array, transform_type):
        rocfft = self.rocfft
        plan = rocfft.plan()
        exe_info = rocfft.execution_info()
        placement = rocfft.placement.notinplace
        transform_type = transform_type
        precision = rocfft.precision.single
        dimensions = in_array.ndim

        c_lengths = ctypes.c_size_t(in_array.size)
        lengths = ctypes.byref(c_lengths)

        number_of_transforms = 1
        description = None

        rocfft.plan_create(plan=ctypes.byref(plan),
                           placement=placement,
                           transform_type=transform_type,
                           precision=precision,
                           dimensions=dimensions,
                           lengths=lengths,
                           number_of_transforms=number_of_transforms,
                           description=description)

        rocfft.execution_info_create(ctypes.byref(exe_info))

        dev_buffer = hsa.to_device(in_array)

        out = np.empty_like(in_array)
        dev_out = hsa.to_device(out)

        rocfft.execute(plan=plan,
                       in_buffer=ctypes.byref(dev_buffer.device_ctypes_pointer),
                       out_buffer=ctypes.byref(dev_out.device_ctypes_pointer),
                       info=exe_info)

        rocfft.synchronize()

        result = dev_out.copy_to_host()

        rocfft.execution_info_destroy(exe_info)
        rocfft.plan_destroy(plan=plan)

        return result

    def complex_inplace_forward(self, in_array):
        copied = in_array.copy()
        fft_inplace(copied)
        return copied

    def complex_inplace_inverse(self, in_array):
        copied = in_array.copy()
        ifft_inplace(copied)
        return copied

    def complex_outofplace_forward(self, in_array):
        tt = self.rocfft.transform_type.complex_forward
        return self.fft_outofplace_core(in_array, transform_type=tt)

    def complex_outofplace_inverse(self, in_array):
        tt = self.rocfft.transform_type.complex_inverse
        return self.fft_outofplace_core(in_array, transform_type=tt)

    def inplace_roundtrip(self, size, dtype):
        N = size
        arr = np.ones(N, dtype=dtype)
        got_fwd = self.complex_inplace_forward(arr)
        exp_fwd = np.fft.fft(arr)
        np.testing.assert_allclose(got_fwd, exp_fwd)
        got_inv = self.complex_inplace_inverse(got_fwd) / N
        exp_inv = np.fft.ifft(exp_fwd)
        np.testing.assert_allclose(got_inv, exp_inv)

    def outofplace_roundtrip(self, size, dtype):
        N = size
        arr = np.ones(N, dtype=dtype)
        print('forward')
        got_fwd = self.complex_outofplace_forward(arr)
        exp_fwd = np.fft.fft(arr)
        np.testing.assert_allclose(got_fwd, exp_fwd)
        print('backward')
        got_inv = self.complex_outofplace_inverse(got_fwd) / N
        exp_inv = np.fft.ifft(exp_fwd)
        np.testing.assert_allclose(got_inv, exp_inv)

    def test_inplace_complex64_16_roundtrip(self):
        self.inplace_roundtrip(16, np.complex64)

    def test_inplace_complex64_power2_roundtrip(self):
        for n in range(13):  # rocfft internal assertion error at 2**14 == 8192
            self.inplace_roundtrip(2**n, np.complex64)

    def test_inplace_complex64_device_copy(self):
        arr = np.ones(16, dtype=np.complex64)
        dev_arr = hsa.to_device(arr)
        fwd_out = fft_inplace(dev_arr)
        inv_out = fft_inplace(fwd_out)
        self.assertIs(fwd_out, dev_arr)
        self.assertIs(inv_out, dev_arr)
        out = inv_out.copy_to_host() / arr.size
        np.testing.assert_allclose(arr, out)

    def test_inplace_complex64_100_roundtrip(self):
        """Power of 2 only
        """
        with self.assertRaises(NotImplementedError) as raises:
            self.inplace_roundtrip(100, np.complex64)
        self.assertEqual(str(raises.exception), "non power-of-2 size is not supported")

    @unittest.skip('notinplace is not supported yet')
    def test_outofplace_complex64_16_roundtrip(self):
        self.outofplace_roundtrip(16, np.complex64)

    @unittest.skip('2d is not supported yet')
    def test_inplace_complex64_2d_16_roundtrip(self):
        arr = np.ones((16, 8), dtype=np.complex64)
        got_fwd = self.complex_inplace_forward(arr)
        exp_fwd = np.fft.fft(arr)
        np.testing.assert_allclose(got_fwd, exp_fwd)
        got_inv = self.complex_inplace_inverse(got_fwd) / N
        exp_inv = np.fft.ifft(exp_fwd)
        np.testing.assert_allclose(got_inv, exp_inv)

    @unittest.skip("double precision not working")
    def test_inplace_complex128_16_roundtrip(self):
        self.inplace_roundtrip(16, np.complex128)


if __name__ == '__main__':
    unittest.main()
