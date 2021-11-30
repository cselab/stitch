import sys
import numpy as np
import os
import stitch.mesospim

try:
    (tx, ty), (nx, ny, nz), (ox, oy), path = stitch.mesospim.read_tiles(sys.argv[1::])
except ValueError:
    tx = ty = 2
    nx = ny = nz = 200
    ox = oy = 10
    path = sys.argv[1:]

dtype = np.dtype("<u2")
stride = [8, 8, 8]
# maximim shifts
hx, hy, hz = 5, 5, 5
sx, sy, sz = 2, 2, 2


src = [
    np.memmap(e, dtype, 'r', 0, (nx, ny, nz), order='F')
    for e in path
]
pairs = []
positions = []
for x in range(tx):
    for y in range(ty):
        positions.append([x * (nx - ox), y * (ny - oy), 0])
        i = x * ty + y
        if x + 1 < tx:
            pairs.append((i, (x + 1) * ty + y))
        if y + 1 < ty:
            pairs.append((i, x * ty + y + 1))

def ov(a, b, n):
    l0 = b - a
    h0 = l0 + n
    l1 = a - b
    h1 = l1 + n
    return max(l0, 0), min(h0, n), max(l1, 0), min(h1, n)


mx = my = mz = 5
i, j = pairs[0]
x0, y0, z0 = positions[i]
x1, y1, z1 = positions[j]

x0l, x0h, x1l, x1h = ov(x0, x1 + mx, nx)
y0l, y0h, y1l, y1h = ov(y0, y1 + my, ny)
z0l, z0h, z1l, z1h = ov(z0, z1 + mz, nz)

a = src[i][x0l:x0h:sx,y0l:y0h:sy,z0l:z0h:sz]
b = src[i][x1l:x1h:sx,y1l:y1h:sy,z1l:z1h:sz]


