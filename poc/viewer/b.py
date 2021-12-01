import sys
import numpy as np
import os
import stitch.mesospim
import stitch.fast

# STITCH_VERBOSE=1 OMP_NUM_THREADS=1 python3.6 poc/viewer/b.py '/media/user/demeter16TB_3/FCD/FCD/FCD_P-OCX_2.7_NeuN-Cy3 (2)'/*.raw

try:
    (tx, ty), (nx, ny,
               nz), (ox, oy), path = stitch.mesospim.read_tiles(sys.argv[1::])
except ValueError:
    tx = ty = 2
    nx = ny = nz = 200
    ox = oy = 10
    path = sys.argv[1:]

dtype = np.dtype("<u2")
sx = sy = sz = 8
tx = ty = tz = 2
src = [np.memmap(e, dtype, 'r', 0, (nx, ny, nz), order='F')[::sx,::sy,::sz] for e in path]
kx = (nx + sx - 1) // sx
ky = (ny + sy - 1) // sy
kz = (nz + sz - 1) // sz

ox //= sx
oy //= sy
hx, hy, hz = ox//2, oy//2, oy//2

pairs = []
positions = []
for x in range(tx):
    for y in range(ty):
        positions.append([x * (kx - ox), y * (ky - oy), 0])
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
    x0l, x0h, x1l, x1h = ov_roi(x0, x1, kx, hx)
    y0l, y0h, y1l, y1h = ov_roi(y0, y1, ky, hy)
    z0l, z0h, z1l, z1h = ov_roi(z0, z1, kz, hz)

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


    roi0 = np.ndarray((n0x, n0y, n0z), dtype=dtype)
    roi1 = np.ndarray((n0x, n0y, n0z), dtype=dtype)
    print(roi0.shape, roi1.shape)
    np.copyto(roi0, src[i][x0l:x0h, y0l:y0h, z0l:z0h], 'no')
    np.copyto(roi1, src[j][x0l:x0h, y0l:y0h, z0l:z0h], 'no')
    
    m_corr = -1
    for mx in range(-hx, hx + 1):
        for my in range(-hy, hy + 1):
            for mz in range(-hz, hz + 1):
                print(mx, my, mz)
                x0l, x0h, x1l, x1h = ov(x0, x1 + mx, n0x, n1x)
                y0l, y0h, y1l, y1h = ov(y0, y1 + my, n0y, n1y)
                z0l, z0h, z1l, z1h = ov(z0, z1 + mz, n0z, n1z)

                a = roi0[x0l:x0h:tx, y0l:y0h:ty, z0l:z0h:tz]
                b = roi1[x1l:x1h:tx, y1l:y1h:ty, z1l:z1h:tz]

                corr = stitch.fast.corr(a, b)
                if corr > m_corr:
                    ix, iy, iz = mx, my, mz
                    m_corr = corr
    print(ix, iy, iz, m_corr)
