import numpy as np
import sys
import poc.pgm

sys.argv.pop(0)
output = sys.argv.pop(0)

tx, ty = 2, 2
nx, ny, nz = 200, 200, 200
dtype = "<u2"
files = [np.memmap(e, dtype, 'r', 0, (nx, ny, nz), order='F') for e in sys.argv]


b = np.ndarray((nx * tx, ny * ty, nz), dtype, order = 'F')

for x in range(tx):
    for y in range(ty):
        f = files.pop(0)
        b[nx * x : nx * (x + 1), ny * y : ny * (y + 1), :] = f

poc.pgm.pgm("a.pgm", b[:, :, nz // 2])        
