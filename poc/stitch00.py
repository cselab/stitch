import Alignment.Rigid as st
import Alignment.Wobbly as stw
import mmap
import multiprocessing
import numpy as np
import os
import sys
import Alignment.glb as glb

me = "stitch0.py"
dtype = np.dtype("<u2")
processes = multiprocessing.cpu_count()
#processes = 'serial'
sys.stderr.write("%s: processes = %s\n" % (me, processes))
di = '/home/lisergey/stride8'
tx, ty = 3, 5
nx, ny, nz = 2048, 2048, 4299
sx, sy, sz = 8, 8, 8
path = (
    '02x04.raw',
    '02x03.raw',
    '02x02.raw',
    '02x01.raw',
    '02x00.raw',
    '01x04.raw',
    '01x03.raw',
    '01x02.raw',
    '01x01.raw',
    '01x00.raw',
    '00x04.raw',
    '00x03.raw',
    '00x02.raw',
    '00x01.raw',
    '00x00.raw',
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
src = tuple(np.memmap(os.path.join(di, e), dtype, 'r', 0, (kx, ky, kz), order='F') for e in path)
glb.SRC[:] = src[:]
layout = stw.WobblyLayout(tuple(range(len(glb.SRC))),
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
glb.SINK[:] = [sink]
stw.stitch(layout, processes, verbose=True)
sys.stderr.write(
    "[%d %d %d] %.2g%% %s\n" %
    (*sink.shape, 100 * np.count_nonzero(sink) / np.size(sink), output))
