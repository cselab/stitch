import glob
import numpy as np
import os
import stitch.glb as glb
import stitch.Rigid as st
import stitch.Wobbly as stw
import sys


def open(path):
    path = os.path.join(di, path)
    return np.memmap(path, dtype, 'r', 0, (nx, ny, nz),
                     order='F')[::sx, ::sy, ::sz]


me = "poc/davide.py"
di = "/scratch/lisergey/davide"
sx = sy = sz = 2
verbose = True
dtype = np.dtype("<u2")
nx, ny, nz = 2048, 2048, 3800
tx, ty = 2, 4
path = (
    "7066_3_X5500_Y16500_rot_0_488_nm_498_509-22_2x_Right_000004.raw",
    "7066_3_X5500_Y11000_rot_0_488_nm_498_509-22_2x_Right_000005.raw",
    "7066_3_X5500_Y5500_rot_0_488_nm_498_509-22_2x_Right_000006.raw",
    "7066_3_X5500_Y0_rot_0_488_nm_498_509-22_2x_Right_000007.raw",
    "7066_3_X0_Y16500_rot_0_488_nm_498_509-22_2x_Left_000003.raw",
    "7066_3_X0_Y11000_rot_0_488_nm_498_509-22_2x_Left_000002.raw",
    "7066_3_X0_Y5500_rot_0_488_nm_498_509-22_2x_Left_000001.raw",
    "7066_3_X0_Y0_rot_0_488_nm_498_509-22_2x_Left_000000.raw",
)
processes = (tx - 1) * ty + tx * (ty - 1)
glb.SRC[:] = (open(e) for e in path)
kx, ky, kz = glb.SRC[0].shape
ox = 361 // sx
oy = 361 // sy
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
    find_shifts=dict(cutoff=3 * np.sqrt(2) / sx, min_distance=8 // sx),
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
