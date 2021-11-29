import matplotlib.pyplot
import sys
import numpy as np
import os
import stitch.mesospim

try:
    (tx, ty), (nx, ny, nz), (ox, oy), path = stitch.mesospim.read_tiles(sys.argv[1::])
except ValueError:
    tx = ty = 2
    nx = ny = nz = 200
    ox = oy = 10
    path = sys.argv[1:]

dtype = np.dtype("<u2")
stride = [8, 8, 8]
src = [
    np.memmap(e, dtype, 'r', 0, (nx, ny, nz), order='F')
    for e in path
]
positions = []
for x in range(tx):
    for y in range(ty):
        positions.append([x * (nx - ox), y * (ny - oy), 0])

fig, ax = matplotlib.pyplot.subplots()
fig.tight_layout()
fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
ax.axis('off')
lx = nx * tx
ly = ny * ty
ax.set_xlim((-lx/10, 11*lx/10))
ax.set_ylim((-ly/10, 11*ly/10))

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
    s = src[i][::stride[0], ::stride[1], z]
    y = ty * ny - y
    extent = (x, x + nx, y - ny, y)
    cmap = 'Greens' if se[0] == i else 'Greys'
    if art[i] is not None:
        art[i].remove()
    vmin = np.quantile(s, 0.1)
    vmax = np.quantile(s, 0.9)
    art[i] = matplotlib.pyplot.imshow(s.T, alpha=0.5, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)

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
    elif key == "h":
        positions[se[0]][0] -= stride[0]
        draw(se[0])
        fig.canvas.draw()
    elif key == "l":
        positions[se[0]][0] += stride[0]
        draw(se[0])
        fig.canvas.draw()
    elif key == "j":
        positions[se[0]][1] -= stride[1]
        draw(se[0])
        fig.canvas.draw()
    elif key == "k":
        positions[se[0]][1] += stride[1]
        draw(se[0])
        fig.canvas.draw()
    elif key == "i":
        positions[se[0]][2] -= stride[2]
        draw(se[0])
        fig.canvas.draw()
    elif key == "n":
        positions[se[0]][2] += stride[2]
        draw(se[0])
        fig.canvas.draw()
    elif key == "z":
        if art[se[0]] == None:
            draw(se[0])
            fig.canvas.draw()
        else:
            art[se[0]].remove()
            art[se[0]] = None
            fig.canvas.draw()
    elif key == "R":
        stride[:] = [2*e for e in stride]
        for i in range(n):
            draw(i)
        fig.canvas.draw()
    elif key == "r":
        stride[:] = [max(1, e//2) for e in stride]
        for i in range(n):
            draw(i)
        fig.canvas.draw()
    elif key == "q":
        sys.exit(0)
    elif key == "s":
        print()
        for x, y, z in positions:
            print(x, y, z)

for i in range(len(src)):
    draw(i)
fig.canvas.mpl_connect('key_press_event', press)
matplotlib.pyplot.show()
