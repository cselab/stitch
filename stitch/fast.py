import numpy as np
import ctypes
import os


def amax(input, axis):
    dtype = np.dtype("<u2")
    if input.dtype != dtype:
        raise TypeError("dtype '%s' is not supported" % str(input.dtype))
    if input.ndim != 3:
        raise TypeError("ndim = '%d' is not supported" % input.ndim)
    if axis == 0:
        return amax0(input)
    elif axis == 1:
        return amax1(input)
    else:
        raise ValueError("axis = %d is not supported" % axis)


def corr(a, b):
    dtype = np.dtype("<u2")
    if a.dtype != dtype:
        raise TypeError("dtype '%s' is not supported" % str(a.dtype))
    if b.dtype != dtype:
        raise TypeError("dtype '%s' is not supported" % str(b.dtype))
    if a.ndim != 3:
        raise TypeError("ndim = '%d' is not supported" % a.ndim)
    if b.ndim != 3:
        raise TypeError("ndim = '%d' is not supported" % b.ndim)
    if np.shape(a) != np.shape(b):
        raise TypeError("np.shape(a) != np.shape(b)" %
                        (str(np.shape(a)), str(np.shape(b))))
    path = os.path.dirname(os.path.realpath(__file__))
    lib = np.ctypeslib.load_library('stitch0.so', path)
    fun = lib.corr
    fun.restype = ctypes.c_double
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
        ctypes.c_ulong,
        np.ctypeslib.ndpointer(dtype, flags='aligned'),
    ]
    return fun(*a.shape, *a.strides, a, *b.strides, b)


def amax0(input):
    dtype = np.dtype("<u2")
    path = os.path.dirname(os.path.realpath(__file__))
    lib = np.ctypeslib.load_library('stitch0.so', path)
    fun = lib.amax0
    fun.restype = ctypes.c_int
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
    path = os.path.dirname(os.path.realpath(__file__))
    lib = np.ctypeslib.load_library('stitch0.so', path)
    fun = lib.amax1
    fun.restype = ctypes.c_int
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
