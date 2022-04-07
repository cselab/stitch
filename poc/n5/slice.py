import gzip
import itertools
import json
import numpy as np
import os
import sys

me = "n5"


def read(file, dtype):
    buffer = file.read(dtype.itemsize)
    array = np.ndarray(1, dtype, buffer)
    return array[0]


def read16(file):
    return read(file, np.dtype(">u2"))


def read32(file):
    return read(file, np.dtype(">u4"))


def usg():
    sys.stderr.write("%s -i path.n5 -s int [-o .] [-v]\n" % me)
    sys.exit(2)


def one(idx):
    if Verbose:
        sys.stderr.write("%s: %s: %s from %s\n" %
                         (me, os.getpid(), idx, blockNumber))
    lo = tuple(a * b for a, b in zip(idx, blockSize))
    hi = tuple(
        min((a + 1) * b, c) for a, b, c in zip(idx, blockSize, dimensions))
    shape = tuple(b - a for a, b in zip(lo, hi))
    path0 = os.path.join(path, "exported_data", *map(str, idx))
    if not os.path.isfile(path0):
        sys.stderr.write("%s: missing file '%s'\n" % (me, path0))
        sys.exit(2)
    with open(path0, "rb") as file:
        if read16(file) != 0:
            sys.stderr.write("%s: fail to read zero in '%s'\n" % (me, path0))
            sys.exit(2)
        if read16(file) != len(idx):
            sys.stderr.write("%s: fail to number of dimensions in '%s'\n" %
                             (me, path0))
            sys.exit(2)
        if tuple(read32(file) for dim in idx) != shape:
            sys.stderr.write("%s: wrong shape in '%s'\n" % (me, path0))
            sys.exit(2)
        with gzip.GzipFile(fileobj=file) as gz:
            buffer = gz.read()
            array = np.ndarray(shape, np.dtype("<f4"), buffer, order='F')
        for c in range(dimensions[3]):
            np.copyto(output[c][lo[0]:hi[0], lo[1]:hi[1]],
                      array[:, :, Slice - lo[2], c], 'no')


path = None
dir = "."
Verbose = False
Slice = None
while True:
    sys.argv.pop(0)
    if len(sys.argv) and len(sys.argv[0]) > 1 and sys.argv[0][0] == '-':
        if sys.argv[0][1] == 'h':
            usg()
        elif sys.argv[0][1] == 'i':
            sys.argv.pop(0)
            if len(sys.argv) == 0:
                sys.stderr.write("%s: -i needs an argument\n" % me)
                sys.exit(2)
            path = sys.argv[0]
        elif sys.argv[0][1] == 'o':
            sys.argv.pop(0)
            if len(sys.argv) == 0:
                sys.stderr.write("%s: -o needs an argument\n" % me)
                sys.exit(2)
            dir = sys.argv[0]
        elif sys.argv[0][1] == 'v':
            Verbose = True
        elif sys.argv[0][1] == 's':
            sys.argv.pop(0)
            if len(sys.argv) == 0:
                sys.stderr.write("%s: -s needs an argument\n" % me)
                sys.exit(2)
            try:
                Slice = int(sys.argv[0])
            except ValueError:
                sys.stderr.write("%s: -s needs an integer\n" % me)
                sys.exit(2)
        else:
            sys.stderr.write("%s: unknown option '%s'\n" % (me, sys.argv[0]))
            sys.exit(2)
    else:
        break
if path == None:
    sys.stderr.write("%s: -i must be set\n" % me)
    sys.exit(2)
if Slice == None:
    sys.stderr.write("%s: -s must be set\n" % me)
    sys.exit(2)

if not os.path.isdir(path):
    sys.stderr.write("%s: '%s' is not a directory\n" % (me, path))
    sys.exit(2)

with open(os.path.join(path, "attributes.json")) as file:
    js = json.load(file)
    if js != {"n5": "2.0.0"}:
        sys.stderr.write("%s: unknown n5 file\n" % me)
        sys.exit(2)

with open(os.path.join(path, "exported_data", "attributes.json")) as file:
    js = json.load(file)
    dimensions = js["dimensions"]
    blockSize = js["blockSize"]
    if js["dataType"] != 'float32':
        sys.stderr.write("%s: unknown dataType: %s\n" % (me, js["dataType"]))
        sys.exit(2)
    if js["axes"] != ['x', 'y', 'z', 'c']:
        sys.stderr.write("%s: unknown axes: %s\n" % (me, js["axes"]))
        sys.exit(2)

blockNumber = tuple((a + b - 1) // b for a, b in zip(dimensions, blockSize))
dim = dimensions[0], dimensions[1],

if Slice < 0 or Slice >= dimensions[2]:
    sys.stderr.write("%s: invalid slice %d/%d\n" % (me, Slice, dimensions[2]))
    sys.exit(2)

os.makedirs(dir, exist_ok=True)
output_path = [
    os.path.join(dir, "%dx%dbe.%d.raw" % (*dim, c))
    for c in range(dimensions[3])
]
output = [np.memmap(p, "<f4", 'w+', 0, dim, 'F') for p in output_path]
if Verbose:
    for p in output_path:
        sys.stderr.write("%s: %s\n" % (me, p))

z = Slice // blockSize[2]
for x in range(blockNumber[0]):
    for y in range(blockNumber[1]):
        for c in range(blockNumber[3]):
            one((x, y, z, c))
