import Alignment.Rigid as st
import Alignment.Wobbly as stw
import mmap
import multiprocessing
import numpy as np
import os
import sys
import tmp

me = "dem.py"
dtype = np.dtype("<u2")
processes = 16 # multiprocessing.cpu_count()
#processes = 'serial'
sys.stderr.write("%s: processes = %s\n" % (me, processes))
tx, ty = 3, 5
nx, ny, nz = 2048, 2048, 4299
sx, sy, sz = 4, 4, 4
di = '/home/lisergey/s1/AnnaMaria/HuADf2/HuADf1.2/ADf_1.2.HC_hFTAA_SMA-Cy3_Pdxl-647'
path = (
    '1.2.HC_X-5000_Y-10000_640_nm_4x_Left_000044.raw',
    '1.2.HC_X-5000_Y-7500_640_nm_4x_Left_000041.raw',
    '1.2.HC_X-5000_Y-5000_640_nm_4x_Left_000038.raw',
    '1.2.HC_X-5000_Y-2500_640_nm_4x_Left_000035.raw',
    '1.2.HC_X-5000_Y0_640_nm_4x_Left_000032.raw',
    '1.2.HC_X-2500_Y-10000_640_nm_4x_Left_000029.raw',
    '1.2.HC_X-2500_Y-7500_640_nm_4x_Left_000026.raw',
    '1.2.HC_X-2500_Y-5000_640_nm_4x_Left_000023.raw',
    '1.2.HC_X-2500_Y-2500_640_nm_4x_Left_000020.raw',
    '1.2.HC_X-2500_Y0_640_nm_4x_Left_000017.raw',
    '1.2.HC_X0_Y-10000_640_nm_4x_Right_000014.raw',
    '1.2.HC_X0_Y-7500_640_nm_4x_Right_000011.raw',
    '1.2.HC_X0_Y-5000_640_nm_4x_Right_000008.raw',
    '1.2.HC_X0_Y-2500_640_nm_4x_Right_000005.raw',
    '1.2.HC_X0_Y0_640_nm_4x_Right_000002.raw',
)
tmp.SRC = tuple(np.memmap(os.path.join(di, e), dtype, 'r', 0, (nz, ny, nx), 'F')[::sx,::sy,::sz].copy()
            for e in path)
kx, ky, kz = tmp.SRC[0].shape
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
layout = stw.WobblyLayout(tuple(range(len(tmp.SRC))),
                          pairs,
                          tile_positions=tile_positions,
                          positions=positions)
st.align(layout.alignments,
         depth=[434 // sx, 425 // sy, None],
         max_shifts=[(-80 // sx, 80 // sx), (-80 // sy, 80 // sy),
                     (-120 // sz, 120 // sz)],
         background=(100, 120),
         clip=25000,
         processes=processes,
         verbose=True)
st.place(layout.alignments, layout.sources)
stw.align_layout(layout.alignments,
                 max_shifts=[(-20 // sx, 20 // sx), (-20 // sy, 20 // sy),
                             (0, 0)],
                 prepare=True,
                 find_shifts=dict(method='tracing', cutoff=3 * np.sqrt(2)),
                 processes=processes,
                 verbose=True)
stw.place(layout,
          min_quality=-np.inf,
          method='optimization',
          smooth=dict(method='window',
                      window='hamming',
                      window_length=100,
                      binary=None),
          smooth_optimized=dict(method='window',
                                window='bartlett',
                                window_length=20,
                                binary=10),
          fix_isolated=False,
          lower_to_origin=True,
          processes=processes,
          verbose=True)

ux, uy, uz = layout.shape_wobbly()
output = "%dx%dx%dle.raw" % (ux, uy, uz)
sink = np.memmap(output, dtype, 'w+', 0, (ux, uy, uz), order='F')
tmp.SINK[:] = [sink]
stw.stitch(layout, processes, verbose=True)
sys.stderr.write(
    "[%d %d %d] %.2g%% %s\n" %
    (*sink.shape, 100 * np.count_nonzero(sink) / np.size(sink), output))
