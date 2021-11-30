import sys
import numpy as np
import os
import stitch.mesospim
import stitch.fast

try:
    (tx, ty), (nx, ny,
               nz), (ox, oy), path = stitch.mesospim.read_tiles(sys.argv[1::])
except ValueError:
    tx = ty = 2
    nx = ny = nz = 200
    ox = oy = 10
    path = sys.argv[1:]

dtype = np.dtype("<u2")
hx, hy, hz = 5, 5, 5
sx, sy, sz = 4, 4, 4

src = [np.memmap(e, dtype, 'r', 0, (nx, ny, nz), order='F') for e in path]
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


def ov(a, b, na, nb):
    l0 = b - a
    h0 = l0 + na
    l1 = a - b
    h1 = l1 + nb
    return max(l0, 0), min(h0, na), max(l1, 0), min(h1, nb)

def ov_roi(x0, x1, n, h):
    x0la, x0ha, x1la, x1ha = ov(x0, x1 + h, n, n)
    x0lb, x0hb, x1lb, x1hb = ov(x0, x1 - h, n, n)
    return min(x0la, x0lb), max(x0ha, x0hb), min(x1la, x1lb), max(x1ha, x1hb)

for i, j in pairs:
    x0, y0, z0 = positions[i]
    x1, y1, z1 = positions[j]
    x0l, x0h, x1l, x1h = ov_roi(x0, x1, nx, hx)
    y0l, y0h, y1l, y1h = ov_roi(y0, y1, ny, hy)
    z0l, z0h, z1l, z1h = ov_roi(z0, z1, nz, hz)

    x0 += x0l
    y0 += y0l
    z0 += z0l

    x1 += x1l
    y1 += y1l
    z1 += z1l

    n0x = x0h - x0l
    n1x = x1h - x1l

    n0y = y0h - y0l
    n1y = y1h - y1l

    n0z = z0h - z0l
    n1z = z1h - z1l

    roi0 = src[i][x0l:x0h, y0l:y0h, z0l:z0h]
    roi1 = src[j][x1l:x1h, y1l:y1h, z1l:z1h]

    print(roi0.shape, roi1.shape)

    roi0 = roi0.copy()
    roi1 = roi1.copy()
    
    m_corr = -1
    for mx in range(-hx, hx + 1):
        for my in range(-hy, hy + 1):
            for mz in range(-hz, hz + 1):

                x0l, x0h, x1l, x1h = ov(x0, x1 + mx, n0x, n1x)
                y0l, y0h, y1l, y1h = ov(y0, y1 + my, n0y, n1y)
                z0l, z0h, z1l, z1h = ov(z0, z1 + mz, n0z, n1z)

                a = roi0[x0l:x0h:sx, y0l:y0h:sy, z0l:z0h:sz]
                b = roi1[x1l:x1h:sx, y1l:y1h:sy, z1l:z1h:sz]

                corr = stitch.fast.corr(a, b)
                if corr > m_corr:
                    ix, iy, iz = mx, my, mz
                    m_corr = corr
    print(ix, iy, iz, m_corr)
