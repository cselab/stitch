import matplotlib.pyplot
import sys
import numpy as np
import os
import copy
import stitch.mesospim
import stitch.Rigid

me = "0.py"
shifts = []
nx = ny = nz = 200
dtype = np.dtype("<u2")
stride = [8, 8, 8]
src = np.memmap(sys.argv[1], dtype, 'r', 0, (nx, ny, nz), order='F')
positions0 = [[0, 0, 0]]
positions = copy.copy(positions0)

fig, ax = matplotlib.pyplot.subplots()
fig.tight_layout()
fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
fig.canvas.manager.full_screen_toggle()
ax.axis('off')
lx = nx
ly = ny
ax.set_xlim((-lx / 10, 11 * lx / 10))
ax.set_ylim((-ly / 10, 11 * ly / 10))

zslice = [nz // 2]
se = [0]
art = [None for e in src]


def draw(i):
    x, y, z = positions[i]
    z += zslice[0]
    if z < 0:
        z = 0
    elif z > nz - 1:
        z = nz - 1
    s = src[::stride[0], ::stride[1], z]
    y = ny - y
    extent = (x, x + nx, y - ny, y)
    cmap = 'Greens' if se[0] == i else 'Greys'
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
    n = len(src)
    key = event.key
    if key == "down":
        se[0] += 1
        if se[0] == n:
            se[0] = 0
        for i in range(n):
            draw(i)
        fig.canvas.draw()
    elif key == "up":
        se[0] -= 1
        if se[0] == -1:
            se[0] = n - 1
        for i in range(n):
            draw(i)
        fig.canvas.draw()
    elif key == "right":
        zslice[0] += stride[2]
        for i in range(n):
            draw(i)
        fig.canvas.draw()
    elif key == "left":
        zslice[0] -= stride[2]
        for i in range(n):
            draw(i)
        fig.canvas.draw()
    elif key == "i":
        positions[se[0]][2] -= stride[2]
        draw(se[0])
        fig.canvas.draw()
    elif key == "n":
        positions[se[0]][2] += stride[2]
        draw(se[0])
        fig.canvas.draw()
    elif key == "R":
        stride[:] = [2 * e for e in stride]
        for i in range(n):
            draw(i)
        fig.canvas.draw()
    elif key == "r":
        stride[:] = [max(1, e // 2) for e in stride]
        for i in range(n):
            draw(i)
        fig.canvas.draw()
    elif key == "q":
        sys.exit(0)

draw(0)
fig.canvas.mpl_connect('key_press_event', press)
fig.tight_layout()
matplotlib.pyplot.show()
