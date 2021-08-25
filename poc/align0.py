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
sx = sy = sz = 8
path = glob.glob(di + '*640*.raw')
path.sort(key=key)
glb.SRC[:] = (open(e) for e in path)
kx, ky, kz = glb.SRC[0].shape
pairs = []
tile_positions = []
for x in range(tx):
    for y in range(ty):
        tile_positions.append((x, y))
        i = x * ty + y
        if x + 1 < tx:
            pairs.append((i, (x + 1) * ty + y))
        if y + 1 < ty:
            pairs.append((i, x * ty + y + 1))
positions = ((118, 0, 161), (89, 1623, 165), (59, 3247, 167), (29, 4870, 172),
             (0, 6493, 174), (1732, 39, 69),
             (1703, 1661, 68), (1673, 3284, 69), (1644, 4906, 70),
             (1614, 6529, 71), (3346, 69, 1), (3316, 1692, 0), (3287, 3315, 2),
             (3258, 4938, 3), (3228, 6561, 4))
positions = [ (x//sx, y//sy, z//sz) for x, y, z in positions]
displacements, qualities, status = stw.align(
    pairs,
    positions,
    max_shifts=((-20 // sx, 20 // sx), (-20 // sy, 20 // sy)),
    prepare=True,
    find_shifts=dict(method='tracing', cutoff=3 * np.sqrt(2)),
    processes=processes,
    verbose=verbose)
