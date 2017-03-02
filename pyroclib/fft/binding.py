from __future__ import print_function, division, absolute_import

import ctypes

from ..cwrappers import enum_int, opaque_pointer, Wrapper


class RocFFTError(RuntimeError):
    pass


class RocFFT(object):
    types = dict(
        # Enums
        rocfft_status=enum_int,
        rocfft_transform_type=enum_int,
        rocfft_precision=enum_int,
        rocfft_result_placement=enum_int,
        rocfft_execution_mode=enum_int,
        # Misc
        rocfft_plan=opaque_pointer('rocfft_plan'),
        rocfft_plan_description=opaque_pointer('rocfft_plan_description'),
        rocfft_execution_info=opaque_pointer('rocfft_execution_info'),
    )

    error_class = RocFFTError
    status_type = 'rocfft_status'
    api = {
        'rocfft_plan_create': '''
                                rocfft_plan *plan,
                                rocfft_result_placement placement,
                                rocfft_transform_type transform_type,
                                rocfft_precision precision,
                                size_t dimensions,
                                const size_t *lengths,
                                size_t number_of_transforms,
                                const rocfft_plan_description description
                                ''',

        'rocfft_plan_destroy': 'rocfft_plan plan',

        'rocfft_execute': '''
                            const rocfft_plan plan,
                            void **in_buffer,
                            void **out_buffer,
                            rocfft_execution_info info
                            ''',

        'rocfft_setup': '',

        'rocfft_cleanup': '',

        'rocfft_execution_info_create': 'rocfft_execution_info *info',
        'rocfft_execution_info_destroy': 'rocfft_execution_info info',

    }
    enums = {
        'rocfft_placement': 'inplace, notinplace',
        'rocfft_transform_type': 'complex_forward,complex_inverse,real_forward,real_inverse',
        'rocfft_precision': 'single,double',
        'rocfft_status': '''
                success,
                failure,
                invalid_arg_value,
                invalid_dimensions,
                invalid_array_type,
                invalid_strides,
                invalid_distance,
                invalid_offset
                ''',
        'rocfft_execution_mode': 'nonblocking,nonblocking_with_flush,blocking',
        }
    common_prefix = 'rocfft_'

    def __init__(self, lib):
        self._wrapper = Wrapper(self, lib)

    def plan(self):
        return self._wrapper.get_type('rocfft_plan')()

    def execution_info(self):
        return self._wrapper.get_type('rocfft_execution_info')()

    def on_error(self, errcode):
        raise RocFFTError(self.status(errcode))

    def synchronize(self):
        self._wrapper.lib.hipDeviceSynchronize()


# XXX hardcoded for now
_path = '/root/roclib/librocfft-hcc-d.so'
rocfft = RocFFT(ctypes.CDLL(_path))
