import glob
import mmap
import multiprocessing
import numpy as np
import os
import stitch.glb as glb
import stitch.Rigid as st
import stitch.Wobbly as stw
import sys

me = "dem.py"
# '/media/user/Daten1/ADf_1.2.HC_hFTAA_SMA-Cy3_Pdxl-647/'
di = sys.argv[1]
globs = ('*640*.raw', '*488*.raw', '*561*.raw')
ou = os.path.split(os.path.realpath(di))[-1]
sx = sy = sz = 8


def open(path):
    a = np.memmap(path, dtype, 'r+', 0, (nx, ny, nz),
                  order='F')[::sx, ::sy, ::sz]
    a.setflags(write=False)
    return a


def tile(g):
    g = os.path.join(di, g)
    return tile0(glob.glob(g))


def tile0(path):
    assert path
    lst = []
    for p in path:
        for e in os.path.basename(p).split('_'):
            if len(e) > 0 and e[0] == 'X':
                x = int(e[1:])
            if len(e) > 0 and e[0] == 'Y':
                y = int(e[1:])
        lst.append(((x, y), p))
    lst.sort(reverse=True)
    tile_positions, path = zip(*lst)
    x, y = zip(*tile_positions)
    tx = len(set(x))
    ty = len(set(y))
    return path, tx, ty


verbose = True
dtype = np.dtype("<u2")
nx, ny, nz = 2048, 2048, 4299
path, tx, ty = tile(globs[0])
processes = (tx - 1) * ty + tx * (ty - 1)
sys.stderr.write("%s: processes = %s\n" % (me, processes))
glb.SRC[:] = (open(e) for e in path)
kx, ky, kz = glb.SRC[0].shape
ox = 434 // sx
oy = 425 // sy
tile_positions = []
positions = []
pairs = []
for x in range(tx):
    for y in range(ty):
        tile_positions.append((x, y))
        positions.append((x * (kx - ox), y * (ky - oy), 0))
        i = x * ty + y
        if x + 1 < tx:
            pairs.append((i, (x + 1) * ty + y))
        if y + 1 < ty:
            pairs.append((i, x * ty + y + 1))
shifts, qualities = st.align((kx, ky, kz),
                             pairs,
                             positions,
                             tile_positions,
                             depth=[434 // sx, 425 // sy, None],
                             max_shifts=[(-80 // sx, 80 // sx),
                                         (-80 // sy, 80 // sy),
                                         (-120 // sz, 120 // sz)],
                             clip=25000,
                             processes=processes,
                             verbose=verbose)
positions = st.place(pairs, positions, shifts)
displacements, qualities, status = stw.align(
    (kx, ky, kz),
    pairs,
    positions,
    max_shifts=((-20 // sx, 20 // sx), (-20 // sy, 20 // sy)),
    prepare=True,
    find_shifts=dict(cutoff=3 * np.sqrt(2) / sx),
    processes=processes,
    verbose=verbose)
positions_new, components = stw.place0((kx, ky, kz),
                                       pairs,
                                       positions,
                                       displacements,
                                       qualities,
                                       status,
                                       smooth=dict(method='window',
                                                   window='hamming',
                                                   window_length=100 // sx,
                                                   binary=None),
                                       min_quality=-np.inf,
                                       processes=processes,
                                       verbose=verbose)

wobble, status = stw.place1((kx, ky, kz),
                            positions,
                            positions_new,
                            components,
                            smooth=dict(method='window',
                                        window='bartlett',
                                        window_length=20 // sx,
                                        binary=10),
                            processes=processes,
                            verbose=verbose)

ux, uy, uz = stw.shape_wobbly((kx, ky, kz), positions, wobble)

if not os.path.exists(ou):
    os.makedirs(ou)
for i, g in enumerate(globs):
    path = tile(g)[0]
    output = os.path.join(ou, "%dx%dx%dle.%d.raw" % (ux, uy, uz, i))
    sink = np.memmap(output, dtype, 'w+', 0, (ux, uy, uz), order='F')
    glb.SINK[:] = [sink]
    glb.SRC[:] = (open(e) for e in path)
    stw.stitch((kx, ky, kz),
               positions,
               wobble,
               status,
               processes,
               verbose=verbose)
    sys.stderr.write("%s\n" % output)
