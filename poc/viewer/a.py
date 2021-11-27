import matplotlib.pyplot
import sys
import numpy as np

dtype = np.dtype("<u2")
nx, ny, nz = 200, 200, 200
tx, ty = 2, 2
sx = sy = sz = 1
ox = 20
oy = 20
path = (
    '200x200x200le.00.00.raw',
    '200x200x200le.00.01.raw',
    '200x200x200le.01.00.raw',
    '200x200x200le.01.01.raw',
)

src = [
    np.memmap(e, dtype, 'r', 0, (nx, ny, nz), order='F')[::sx, ::sy, ::sz]
    for e in path
]
kx, ky, kz = np.shape(src[0])

positions = []
for x in range(tx):
    for y in range(ty):
        positions.append((x * (kx - ox), y * (ky - oy), 0))

src0 = [e[:, :, kz // 2] for e in src]
ax = matplotlib.pyplot.subplot()
ax.set_xlim((0, kx * tx))
ax.set_ylim((0, ky * ty))

for i, (s, (x, y, z)) in enumerate(zip(src0, positions)):
    y = ty * ky - y
    extent = (x, x + kx, y - ky, y)
    cmap = 'Greens' if i == 0 else 'Greys'
    matplotlib.pyplot.imshow(s, alpha=0.5, cmap=cmap, extent=extent)
matplotlib.pyplot.show()
