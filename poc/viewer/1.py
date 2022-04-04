import matplotlib.pyplot
import sys
import numpy as np
import os
import copy
import re

me = "0.py"
shifts = []
dtype = np.dtype(">f4")
stride = [1, 1, 1]

base = os.path.basename(sys.argv[1])
base = base.split("x")
nx = int(base[0])
ny = int(base[1])
nz = int(re.sub("[lb]e.*", "", base[2]))

src = np.memmap(sys.argv[1], dtype, 'r', 0, (nx, ny, nz), order='F')
positions0 = [0, 0, 0]
positions = copy.copy(positions0)

fig, ax = matplotlib.pyplot.subplots()
fig.tight_layout()
fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
fig.canvas.manager.full_screen_toggle()
ax.axis('off')
ax.set_xlim((-nx / 10, 11 * nx / 10))
ax.set_ylim((-ny / 10, 11 * ny / 10))

zslice = [nz // 2]
art = [None for e in src]


def draw(i):
    x, y, z = positions
    z += zslice[0]
    if z < 0:
        z = 0
    elif z > nz - 1:
        z = nz - 1
    s = src[::stride[0], ::stride[1], z]
    y = ny - y
    extent = (x, x + nx, y - ny, y)
    cmap = 'Greens'
    if art[i] is not None:
        art[i].remove()
    vmin = np.quantile(s, 0.1)
    vmax = np.quantile(s, 0.9)
    art[i] = matplotlib.pyplot.imshow(s.T,
                                      alpha=0.5,
                                      cmap=cmap,
                                      vmin=vmin,
                                      vmax=vmax,
                                      extent=extent)


def press(event):
    key = event.key
    if key == "right":
        zslice[0] += stride[2]
        draw(0)
        fig.canvas.draw()
    elif key == "left":
        zslice[0] -= stride[2]
        draw(0)
        fig.canvas.draw()
    elif key == "R":
        stride[:] = [2 * e for e in stride]
        draw(0)
        fig.canvas.draw()
    elif key == "r":
        stride[:] = [max(1, e // 2) for e in stride]
        draw(0)
        fig.canvas.draw()
    elif key == "q":
        sys.exit(0)


draw(0)
fig.canvas.mpl_connect('key_press_event', press)
fig.tight_layout()
matplotlib.pyplot.show()
