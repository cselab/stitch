import sys
import os
import glob
import re
import numpy as np

me = "side.py"
di = sys.argv[1]
dtype = np.dtype("<u2")

def tile(g):
    g = os.path.join(di, g)
    return tile0(glob.glob(g))


def meta():
    path = glob.glob(os.path.join(di, "*_000000.raw_meta.txt"))
    if len(path) == 0:
        sys.stderr.write("%s: no meta file\n" % me)
        sys.exit(1)
    with open(path[0]) as file:
        for line in file:
            line = re.sub("\r\n$", "", line)
            key = re.sub("^\[", "", line)
            key = re.sub("\].*", "", key)
            val = re.sub(".*\]", "", line)
            if key == "x_pixels":
                nx = int(val)
            elif key == "y_pixels":
                ny = int(val)
            elif key == "z_planes":
                nz = int(val)
    return nx, ny, nz


def tile0(path):
    assert path
    lst = []
    for p in path:
        for e in os.path.basename(p).split('_'):
            if len(e) > 0 and e[0] == 'X':
                x = int(e[1:])
            if len(e) > 0 and e[0] == 'Y':
                y = int(e[1:])
        lst.append(((x, y), p))
    lst.sort(reverse=True)
    tile_positions, path = zip(*lst)
    x, y = zip(*tile_positions)
    tx = len(set(x))
    ty = len(set(y))
    return path, tx, ty

# python poc/side.py '/media/user/demeter16TB_3/FCD/FCD/FCD_P-OCX_2.7_NeuN-Cy3 (2)'
sx = sy = sz = 8
nx, ny, nz = meta()
path, tx, ty = tile("*.raw")

kx = (nx + sx - 1) // sx
ky = (ny + sy - 1) // sy
kz = (nz + sz - 1) // sy

ox = 4
oy = 4

ux = kx * tx + (tx - 1) * ox
uy = ky * ty + (ty - 1) * oy
uz = kz

output_path = "%dx%dx%dle.raw" % (ux, uy, uz)
output = np.memmap(output_path, dtype, "w+", 0, (ux, uy, uz), 'F')

for x in range(tx):
    for y in range(ty):
        print(x, y)
        i = x * ty + y
        a = np.memmap(path[i], dtype, 'r', 0, (nx, ny, nz), 'F')[::sx, ::sy, ::sz]
        np.copyto(output[x * kx + ox * x : (x + 1) * kx + ox * x,
                         y * ky + oy * y : (y + 1) * ky + oy * y,
                         :], a, 'no')
