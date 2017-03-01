from __future__ import print_function, division, absolute_import

import re
import ctypes
from enum import IntEnum

from numba import utils


# Opaque structure
class _Opaque(ctypes.Structure):
    _fields = []


def opaque_pointer(name):
    base = type(name, (_Opaque,), {})
    return ctypes.POINTER(base)

# Common types
enum_int = ctypes.c_int

# API
re_arg_decl = re.compile(r"[a-z_][a-z0-9_]*|\*", re.I | re.M)


class Wrapper(object):

    _wrapper_source = """
def {name}_wrapper({args}):
    ret = func({args})
    if ret != 0:
        raise handle_error(ret)
    return
    """

    def __init__(self, obj, lib):
        self.lib = lib
        self.cls = type(obj)
        self.load_enums(obj)
        self.load_api(obj)

    def make_api(self, name, argtypes, handle_error):
        fn = getattr(self.lib, name)
        fn.restype = self.get_type(self.cls.status_type)
        anames, atypes = self.parse_api_types(argtypes)
        fn.argtypes = atypes

        glbls = {'func': fn, 'handle_error': handle_error}
        fvs = {'name': name, 'args': ', '.join(anames)}
        utils.exec_(self._wrapper_source.format(**fvs), glbls)
        return glbls[name + '_wrapper']

    def load_enums(self, obj):
        for k, v in obj.enums.items():
            enumobj = IntEnum(k, v, start=0)
            setattr(obj, self.strip_prefix(k), enumobj)

    def load_api(self, obj):
        for k, v in obj.api.items():
            fn = self.make_api(k, v, handle_error=obj.on_error)
            setattr(obj, self.strip_prefix(k), fn)

    def strip_prefix(self, name):
        if name.startswith(self.cls.common_prefix):
            name = name[len(self.cls.common_prefix):]
        return name

    def parse_arg_decl(self, decl):
        tokens = []
        lastpos = 0
        for m in re.finditer(re_arg_decl, decl):
            cur = m.group(0)
            assert cur
            tokens.append(cur)
            pos = m.start()
            between = decl[lastpos:pos].strip()
            if between:
                raise ValueError('unknown substring {!r}'.format(between))
            lastpos = m.end()

        def f_const(x):
            return x != 'const'

        def append_star_to_previous(tokens):
            tokens = list(tokens)
            pos = 1
            while pos < len(tokens):
                if tokens[pos] == '*':
                    tokens[pos - 1] += '*'
                    del tokens[pos]
                else:
                    pos += 1
            return tuple(tokens)

        return tuple(append_star_to_previous(filter(f_const, tokens)))

    def get_type(self, typename):
        try:
            return self.cls.types[typename]
        except KeyError:
            try:
                return getattr(ctypes, 'c_' + typename)
            except AttributeError:
                raise NameError('unknown type in C decl {!r}'.format(typename))

    def parse_decl_type(self, typ):
        text_voidp = 'void*'
        if typ.startswith(text_voidp):
            ty = ctypes.c_void_p
            typ = typ[len(text_voidp):]
        else:
            ty = self.get_type(typ.rstrip('*'))

        if typ.endswith('*'):
            num_star = typ.count('*')
            for _ in range(num_star):
                ty = ctypes.POINTER(ty)
        return ty

    def parse_api_types(self, descr):
        arguments = []
        for part in descr.split(','):
            part = part.strip()
            if part:
                arguments.append(self.parse_arg_decl(part))
        anames, atypes = [], []
        for typ, name in arguments:
            anames.append(name)
            atypes.append(self.parse_decl_type(typ))

        return anames, atypes


#
# Complex types
#

class c_complex_base(ctypes.Structure):

    def __init__(self, real, imag=0):
        if isinstance(real, complex):
            real = real.real
            imag = real.imag
        super(c_complex_base, self).__init__(real=real, imag=imag)


class c_complex(c_complex_base):
    _fields = [
        ('real', ctypes.c_float),
        ('imag', ctypes.c_float),
    ]


class c_doublecomplex(c_complex_base):
    _fields = [
        ('real', ctypes.c_double),
        ('imag', ctypes.c_double),
    ]
