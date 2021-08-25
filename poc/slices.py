import numpy as np
import os
import poc.pgm
import re
import sys

dtype = np.dtype("<u2")
path = sys.argv[1]
parts = re.sub("le.*", "", os.path.basename(path))
nx, ny, nz = (int(e) for e in parts.split("x"))
a = np.memmap(path, dtype, 'r', 0, (nx, ny, nz), order='F')

for z in range(0, nz, 1):
    poc.pgm.pgm("%04d.pgm" % z, a[:, :, z])
