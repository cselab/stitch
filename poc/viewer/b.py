import numpy as np
import os
import stitch.fast
import stitch.glb as glb
import stitch.mesospim
import stitch.Rigid as st
import sys
import multiprocessing

# STITCH_VERBOSE=1 OMP_NUM_THREADS=1 python3.6 poc/viewer/b.py '/media/user/demeter16TB_3/FCD/FCD/FCD_P-OCX_2.7_NeuN-Cy3 (2)'/*.raw

verbose = True
output_path = "rigid.txt"
try:
    (tx, ty), (nx, ny,
               nz), (ox, oy), path = stitch.mesospim.read_tiles(sys.argv[1::])
except ValueError:
    tx = ty = 2
    nx = ny = nz = 200
    ox = oy = 10
    path = sys.argv[1:]

processes = (tx - 1) * ty + tx * (ty - 1)
dtype = np.dtype("<u2")
sx = sy = sz = 1
qx = qy = qz = 32

frac = 20
zl = (frac - 1) * nz // (2 *frac)
zh = (frac + 1) * nz // (2 *frac)

src = [
    np.memmap(e, dtype, 'r', 0, (nx, ny, nz), order='F')[::sx, ::sy, zl:zh:sz]
    for e in path
]
kx, ky, kz = src[0].shape

ox //= sx
oy //= sy
hx = hy = hz = max(ox, oy)//4

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


def ov_roi(r0, r1, n, h):
    r0la, r0ha, r1la, r1ha = ov(r0, r1 + h, n, n)
    r0lb, r0hb, r1lb, r1hb = ov(r0, r1 - h, n, n)
    return min(r0la, r0lb), max(r0ha, r0hb), min(r1la, r1lb), max(r1ha, r1hb)

def pair(i, j):
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
    roi1 = np.ndarray((n1x, n1y, n1z), dtype=dtype)
    np.copyto(roi0, src[i][x0l:x0h, y0l:y0h, z0l:z0h], 'no')
    np.copyto(roi1, src[j][x1l:x1h, y1l:y1h, z1l:z1h], 'no')

    m_corr = -1
    for mx in range(-hx, hx + 1, 1):
        sys.stderr.write("[%d] [%d %d] %d / %d: %.3f\n" % (os.getpid(), i, j, mx + hx + 1, 2 * hx + 1, m_corr))
        for my in range(-hy, hy + 1, 1):
            for mz in range(-hz, hz + 1, 1):
                x0l, x0h, x1l, x1h = ov(x0, x1 + mx, n0x, n1x)
                y0l, y0h, y1l, y1h = ov(y0, y1 + my, n0y, n1y)
                z0l, z0h, z1l, z1h = ov(z0, z1 + mz, n0z, n1z)
                a = roi0[x0l:x0h:qx, y0l:y0h:qy, z0l:z0h:qz]
                b = roi1[x1l:x1h:qx, y1l:y1h:qy, z1l:z1h:qz]
                corr = stitch.fast.corr(a, b)
                if corr > m_corr:
                    m_x, m_y, m_z = mx, my, mz
                    m_corr = corr
    return m_x, m_y, m_z, m_corr

if processes == 0:
    ans = [pair(i, j) for i, j in pairs]
else:
    with multiprocessing.Pool(processes) as pool:
        ans = pool.starmap(pair, pairs)

with open(".shifts", "w") as file:
    for x, y, z, corr in ans:
        file.write("%d %d %d %.16e\n" % (x, y, z, corr))

shifts = [(x, y, z) for x, y, z, corr in ans]
positions = st.place(pairs, positions, shifts)
glb.SRC[:] = (np.memmap(e, dtype, 'r', 0, (nx, ny, nz),
                        order='F')[::sx, ::sy, ::sz] for e in path)
kx, ky, kz = glb.SRC[0].shape
ux, uy, uz = st.shape((kx, ky, kz), positions)
di = os.path.dirname(path[0])
ou = os.path.join(di, 'out')
if not os.path.exists(ou):
    os.makedirs(ou)
output = os.path.join(ou, "%dx%dx%dle.0.raw" % (ux, uy, uz))
sink = np.memmap(output, dtype, 'w+', 0, (ux, uy, uz), order='F')
glb.SINK[:] = [sink]
st.stitch1((kx, ky, kz), positions, verbose=verbose)
sys.stderr.write("%s\n" % output)
