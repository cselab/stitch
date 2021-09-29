import stitch.Rigid as st
import stitch.Wobbly as stw
import multiprocessing
import numpy as np
import os
import sys
import stitch.glb as glb

me = "stitch0.py"
verbose = True
dtype = np.dtype("<u2")
#processes = multiprocessing.cpu_count()
processes = 'serial'
sys.stderr.write("%s: processes = %s\n" % (me, processes))
di = '/home/lisergey/stride4'
tx, ty = 2, 2
nx, ny, nz = 2048, 2048, 4299
sx, sy, sz = 4, 4, 4
path = (
    '02x04.raw',
    '02x03.raw',
    # '02x02.raw',
    # '02x01.raw',
    # '02x00.raw',
    '01x04.raw',
    '01x03.raw',
    # '01x02.raw',
    # '01x01.raw',
    # '01x00.raw',

    # '00x04.raw',
    # '00x03.raw',
    # '00x02.raw',
    # '00x01.raw',
    # '00x00.raw',
)
kx = (nx + sx - 1) // sx
ky = (ny + sy - 1) // sy
kz = (nz + sz - 1) // sz
ox = 434 // sx
oy = 425 // sy
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
glb.SRC[:] = (np.memmap(os.path.join(di, e),
                        dtype,
                        'r',
                        0, (kx, ky, kz),
                        order='F') for e in path)
# shifts, qualities = st.align(pairs,
#                              positions,
#                              tile_positions,
#                              depth=[434 // sx, 425 // sy, None],
#                              max_shifts=[(-80 // sx, 80 // sx),
#                                          (-80 // sy, 80 // sy),
#                                          (-120 // sz, 120 // sz)],
#                              clip=25000,
#                              processes=processes,
#                              verbose=verbose)
# positions = st.place(pairs, positions, shifts)
positions = [(8, 0, 24), (0, 406, 26), (412, 10, 0), (416, 416, 0)]
displacements, qualities, status = stw.align(
    (kx, ky, kz),
    pairs,
    positions,
    max_shifts=((-20 // sx, 20 // sx), (-20 // sy, 20 // sy)),
    prepare=True,
    find_shifts=dict(cutoff=3 * np.sqrt(2)),
    processes=processes,
    verbose=verbose)

for d in displacements:
    sys.stderr.write("%.2f %.2f\n" % (np.mean(d), np.std(d)))
