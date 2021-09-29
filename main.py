import glob
import numpy as np
import os
import stitch.glb as glb
import stitch.Rigid as st
import stitch.Wobbly as stw
import sys

me = "main.py"
verbose = False
dtype = np.dtype("<u2")
processes = 4
sys.stderr.write("%s: processes = %s\n" % (me, processes))
tx, ty = 2, 2
nx, ny, nz = 200, 200, 200
sx = sy = sz = 1
path = (
    '200x200x200le.00.00.raw',
    '200x200x200le.00.01.raw',
    '200x200x200le.01.00.raw',
    '200x200x200le.01.01.raw',
)
glb.SRC[:] = (np.memmap(e, dtype, 'r', 0, (nx, ny, nz),
                        order='F')[::sx, ::sy, ::sz] for e in path)
kx, ky, kz = glb.SRC[0].shape
ox = 15 // sx
oy = 15 // sy
positions = []
pairs = []
tile_positions = []
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
                             depth=(20 // sx, 20 // sy, None),
                             max_shifts=[(-20 // sx, 20 // sx),
                                         (-20 // sy, 20 // sy),
                                         (-20 // sz, 20 // sz)],
                             clip=np.iinfo(dtype).max,
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

ux, uy, uz = stw.shape_wobbly(glb.SRC[0].shape, positions, wobble)
output = "%dx%dx%dle.raw" % (ux, uy, uz)
sink = np.memmap(output, dtype, 'w+', 0, (ux, uy, uz), order='F')
glb.SINK[:] = [sink]
stw.stitch(glb.SRC[0].shape,
           positions,
           wobble,
           status,
           processes,
           verbose=verbose)
sys.stderr.write("%.2g%% %s\n" %
                 (100 * np.count_nonzero(sink) / np.size(sink), output))

import poc.pgm

path = "%dx%dx%dle.pgm" % (ux, uy, uz)
poc.pgm.pgm(path, sink[:, :, uz // 2])
