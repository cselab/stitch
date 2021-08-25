import glob
import mmap
import multiprocessing
import numpy as np
import os
import stitch.glb as glb
import stitch.Rigid as st
import stitch.Wobbly as stw
import sys

def open(path):
    return np.memmap(path, dtype, 'r', 0, (nx, ny, nz),
                     order='F')[::sx, ::sy, ::sz]
def key(f):
    x = f.split('_')[-7][1:]
    y = f.split('_')[-6][1:]
    return -int(x), -int(y)

di = '/media/user/Daten1/ADf_1.2.HC_hFTAA_SMA-Cy3_Pdxl-647/'

me = "dem.py"
verbose = True
dtype = np.dtype("<u2")
processes = 22
sys.stderr.write("%s: processes = %s\n" % (me, processes))
tx, ty = 3, 5
nx, ny, nz = 2048, 2048, 4299
sx = sy = sz = 16
path = glob.glob(di + '*640*.raw')
path.sort(key=key)
glb.SRC[:] = (open(e) for e in path)
for p in path:
    print(p)
    
kx, ky, kz = glb.SRC[0].shape
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
shifts, qualities = st.align(pairs,
                             positions,
                             tile_positions,
                             depth=[434 // sx, 425 // sy, None],
                             max_shifts=[(-80 // sx, 80 // sx),
                                         (-80 // sy, 80 // sy),
                                         (-120 // sz, 120 // sz)],
                             background=(100, 120),
                             clip=25000,
                             processes=processes,
                             verbose=verbose)
positions = st.place(pairs, positions, shifts)
displacements, qualities, status = stw.align(
    pairs,
    positions,
    max_shifts=((-20 // sx, 20 // sx), (-20 // sy, 20 // sy)),
    prepare=True,
    find_shifts=dict(method='tracing', cutoff=3 * np.sqrt(2)),
    processes=processes,
    verbose=verbose)
positions_new, components = stw.place0(pairs,
                                       positions,
                                       displacements,
                                       qualities,
                                       status,
                                       smooth=dict(method='window',
                                                   window='hamming',
                                                   window_length=100,
                                                   binary=None),
                                       min_quality=-np.inf,
                                       processes=processes,
                                       verbose=verbose)

wobble, status = stw.place1(positions,
                            positions_new,
                            components,
                            smooth=dict(method='window',
                                        window='bartlett',
                                        window_length=20,
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
sys.stderr.write(
    "[%d %d %d] %.2g%% %s\n" %
    (*sink.shape, 100 * np.count_nonzero(sink) / np.size(sink), output))

#import poc.pgm
#poc.pgm.pgm("a.pgm", sink[:, :, uz//2])
