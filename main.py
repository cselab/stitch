import argparse
import numpy as np
import stitch.glb as glb
import stitch.Rigid as st
import stitch.Wobbly as stw
import sys

me = "main.py"
parser = argparse.ArgumentParser(prog=me,
                                 description="stitch a grid of raw tiles",
                                 add_help=False)
parser.add_argument("-h", action="help", help="show this help message and exit")
parser.add_argument("path",
                    nargs="*",
                    help="tiles in x-major order, default: output of tool/gen")
parser.add_argument("-n",
                    nargs=3,
                    type=int,
                    default=(200, 200, 200),
                    metavar=("X", "Y", "Z"),
                    help="tile size (default: 200 200 200)")
parser.add_argument("-t",
                    nargs=2,
                    type=int,
                    default=(2, 2),
                    metavar=("X", "Y"),
                    help="number of tiles (default: 2 2)")
parser.add_argument("-o",
                    nargs=2,
                    type=int,
                    default=(15, 15),
                    metavar=("X", "Y"),
                    help="overlap (default: 15 15)")
parser.add_argument("-s",
                    nargs=3,
                    type=int,
                    default=(1, 1, 1),
                    metavar=("X", "Y", "Z"),
                    help="stride (default: 1 1 1)")
parser.add_argument("-m",
                    type=int,
                    default=20,
                    help="maximum shift (default: 20)")
parser.add_argument("-w",
                    nargs=2,
                    type=int,
                    default=(100, 20),
                    metavar=("A", "B"),
                    help="smoothing windows of shifts and wobble (default: 100 20)")
parser.add_argument("-p",
                    type=int,
                    default=4,
                    help="number of processes (default: 4)")
parser.add_argument("-v", action="store_true", help="verbose")
args = parser.parse_args()
verbose = args.v
dtype = np.dtype("<u2")
processes = args.p
sys.stderr.write("%s: processes = %s\n" % (me, processes))
tx, ty = args.t
nx, ny, nz = args.n
sx, sy, sz = args.s
m = args.m
path = args.path or [
    "%dx%dx%dle.%02d.%02d.raw" % (nx, ny, nz, x, y) for x in range(tx)
    for y in range(ty)
]
if len(path) != tx * ty:
    sys.stderr.write("%s: expect %d tiles, got %d\n" % (me, tx * ty, len(path)))
    sys.exit(2)
glb.SRC[:] = (np.memmap(e, dtype, 'r', 0, (nx, ny, nz),
                        order='F')[::sx, ::sy, ::sz] for e in path)
kx, ky, kz = glb.SRC[0].shape
ox = args.o[0] // sx
oy = args.o[1] // sy
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
                             max_shifts=[(-(m // sx), m // sx),
                                         (-(m // sy), m // sy),
                                         (-(m // sz), m // sz)],
                             clip=np.iinfo(dtype).max,
                             processes=processes,
                             verbose=verbose)
positions = st.place(pairs, positions, shifts)
for name, pos in zip(path, positions):
    sys.stderr.write(f"main.py: {name}: {pos}\n")
displacements, qualities, status = stw.align(
    (kx, ky, kz),
    pairs,
    positions,
    max_shifts=((-(m // sx), m // sx), (-(m // sy), m // sy)),
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
                                                   window_length=args.w[0] // sx,
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
                                        window_length=args.w[1] // sx,
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
import poc.pgm

path = "%dx%d.pgm" % (ux, uy)
poc.pgm.pgm(path, sink[:, :, uz // 2])
sys.stderr.write("%s\n%s\n" % (output, path))
