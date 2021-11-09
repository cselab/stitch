import stitch.fast
import numpy as np

nx, ny, nz = 3, 4, 5
dtype = np.dtype("<u2")

low = np.iinfo(dtype).min
high = np.iinfo(dtype).max

a = np.random.randint(low, high + 1, size=(nx, ny, nz), dtype=dtype)
print(stitch.fast.amax0(a) - np.amax(a, axis=0))
