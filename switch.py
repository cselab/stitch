import glob
import numpy as np
import os
import stitch.glb as glb
import stitch.Rigid as st
import stitch.Wobbly as stw
import sys

def open(path):
    return np.memmap(path, dtype, 'r', 0, (nx, ny, nz),
                     order='F')[::sx, ::sy, ::sz]
me = "switch.py"
sx = sy = sz = 4
verbose = True
dtype = np.dtype("<u2")
nx, ny, nz = 2048, 2048, 615
tx, ty = 2, 2
path = (
    "4X4_X2200_Y-77900_488_nm_2x_Right_000009.raw",
    "4X4_X2200_Y-83408_488_nm_2x_Right_000013.raw",
    "4X4_X-2808_Y-77900_488_nm_2x_Right_000001.raw",
    "4X4_X-2808_Y-83408_488_nm_2x_Right_000005.raw",
)
processes = (tx - 1) * ty + tx * (ty - 1)
glb.SRC[:] = (open(e) for e in path)
kx, ky, kz = glb.SRC[0].shape
ox = 205 // sx
oy = 205 // sy
of = 2
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
                             depth=[ox // of, oy // of, None],
                             max_shifts=[(-ox // of, ox // of),
                                         (-oy // of, oy // of),
                                         (-oy // of, oy // of)],
                             clip=25000,
                             processes=processes,
                             verbose=verbose)
positions = st.place(pairs, positions, shifts)
displacements, qualities, status = stw.align(
    (kx, ky, kz),
    pairs,
    positions,
    max_shifts=((-ox // of, ox // of), (-oy // of, oy // of)),
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
output = "%dx%dx%dle.raw" % (ux, uy, uz)
sink = np.memmap(output, dtype, 'w+', 0, (ux, uy, uz), order='F')
glb.SINK[:] = [sink]
stw.stitch((kx, ky, kz), positions, wobble, status, processes, verbose=verbose)
sys.stderr.write("%s\n" % output)
