import numpy as np
import mmap

def pgm(path, a):
    nx, ny = a.shape
    dtype = np.dtype(">u2")
    with open(path, "wb+") as f:
        ma = np.iinfo(dtype).max
        f.write(b"P5\n%d %d\n%d\n" % (nx, ny, ma))
        offset = f.tell()
        f.seek(nx * ny * dtype.itemsize - 1, 1)
        f.write(b'\0')
        f.seek(0, 0)
        buffer = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
    b = np.ndarray((nx, ny), dtype, buffer, offset=offset, order='F')
    np.copyto(b, a)
