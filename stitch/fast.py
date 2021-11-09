import numpy as np
import ctypes
import os


def amax0(input):
    dtype = np.dtype("<u2")
    assert input.ndim == 3
    assert input.dtype == dtype
    path = os.path.dirname(os.path.realpath(__file__))
    lib = np.ctypeslib.load_library('stitch0.so', path)
    fun = lib.amax0
    fun.restype = None
    fun.argtypes = [
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        np.ctypeslib.ndpointer(dtype, flags='aligned'),
        ctypes.c_ulong,
        ctypes.c_ulong,
        np.ctypeslib.ndpointer(dtype,
                               flags='aligned, writeable, f_contiguous'),
    ]
    nx, ny, nz = input.shape
    output = np.empty((ny, nz), dtype, 'F')
    fun(*input.shape, *input.strides, input, *output.strides, output)
    return output

def amax1(input):
    dtype = np.dtype("<u2")
    assert input.ndim == 3
    assert input.dtype == dtype
    path = os.path.dirname(os.path.realpath(__file__))
    lib = np.ctypeslib.load_library('stitch0.so', path)
    fun = lib.amax1
    fun.restype = None
    fun.argtypes = [
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        np.ctypeslib.ndpointer(dtype, flags='aligned'),
        ctypes.c_ulong,
        ctypes.c_ulong,
        np.ctypeslib.ndpointer(dtype,
                               flags='aligned, writeable, f_contiguous'),
    ]
    nx, ny, nz = input.shape
    output = np.empty((nx, nz), dtype, 'F')
    fun(*input.shape, *input.strides, input, *output.strides, output)
    return output
