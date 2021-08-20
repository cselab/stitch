import adv
import ClearMap.Alignment.Stitching.StitchingRigid as st
import ClearMap.Alignment.Stitching.StitchingWobbly as stw
import mmap
import multiprocessing
import numpy as np
import pdb
import sys

# bsub -I cm.python -u stitch0.py -W 1

me = "stitch.py"
dtype = np.dtype("<u2")
processes = multiprocessing.cpu_count()
sys.stderr.write("%s: processes = %d\n" % (me, processes))
path = [
    '/cluster/scratch/lisergey/stride4/00x00.raw',
    '/cluster/scratch/lisergey/stride4/00x01.raw',
    '/cluster/scratch/lisergey/stride4/00x02.raw',
    #'/cluster/scratch/lisergey/stride4/00x03.raw',
    #'/cluster/scratch/lisergey/stride4/00x04.raw',
    '/cluster/scratch/lisergey/stride4/01x00.raw',
    '/cluster/scratch/lisergey/stride4/01x01.raw',
    '/cluster/scratch/lisergey/stride4/01x02.raw',
    # '/cluster/scratch/lisergey/stride4/01x03.raw',
    # '/cluster/scratch/lisergey/stride4/01x04.raw',

    # '/cluster/scratch/lisergey/stride4/02x00.raw',
    # '/cluster/scratch/lisergey/stride4/02x01.raw',
    # '/cluster/scratch/lisergey/stride4/02x02.raw',
    # '/cluster/scratch/lisergey/stride4/02x03.raw',
    # '/cluster/scratch/lisergey/stride4/02x04.raw',
]
output = "/cluster/scratch/lisergey/stitched.raw"
shape = [2, 3]
nx, ny, nz = 2048, 2048, 4299
sx, sy, sz = 4, 4, 4
kx = (nx + sx - 1) // sx
ky = (ny + sy - 1) // sy
kz = (nz + sz - 1) // sz
src = [adv.array((kx, ky, kz), dtype, e, 'r') for e in path]
ox = 434 // sx
oy = 425 // sy
positions = []
for x in range(shape[0]):
    for y in range(shape[1]):
        positions.append((x * (kx - ox), y * (ky - oy), 0))
layout = stw.WobblyLayout(src,
                          tile_shape=shape,
                          overlaps=(434 // sx, 425 // sy),
                          positions=positions)
st.align(layout,
         depth=[434 // sx, 425 // sy, None],
         max_shifts=[(-80 // sx, 80 // sx), (-80 // sy, 80 // sy),
                     (-120 // sz, 120 // sz)],
         ranges=[None, None, None],
         background=(100, 120),
         clip=25000,
         processes=processes,
         verbose=True)
st.place(layout,
         'optimization',
         min_quality=-np.inf,
         lower_to_origin=True,
         verbose=True)
stw.align_layout(layout,
                 max_shifts=[(-20 // sx, 20 // sx), (-20 // sy, 20 // sy),
                             (0, 0)],
                 axis_mip=None,
                 validate=dict(method='foreground',
                               valid_range=(125, None),
                               size=None),
                 prepare=dict(method='normalization',
                              clip=None,
                              normalize=True),
                 validate_slice=dict(method='foreground',
                                     valid_range=(125, 15000),
                                     size=1500 // sx // sy),
                 prepare_slice=None,
                 find_shifts=dict(method='tracing', cutoff=3 * np.sqrt(2)),
                 processes=2,
                 verbose=True)
sys.exit(0)
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
sink = adv.create(layout.shape, layout.dtype, output, 'w')
stw.stitch(layout, sink, 'interpolation', processes, True)
sys.stderr.write("[%d %d %d] %.2g%%\n" %
                 (*sink.shape, 100 * np.count_nonzero(sink) / np.size(sink)))
