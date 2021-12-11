import matplotlib.pyplot
import sys
import numpy as np
import os
import copy
import stitch.mesospim
import stitch.Rigid

me = "b.py"
shifts = []
while True:
    sys.argv.pop(0)
    if len(sys.argv) and len(sys.argv[0]) > 1 and sys.argv[0][0] == '-':
        if sys.argv[0][1] == 'h':
            usg()
        elif sys.argv[0][1] == 'f':
            sys.argv.pop(0)
            if len(sys.argv) == 0:
                sys.stderr.write("%s: not enough arguments for -f\n" % me)
                sys.exit(1)
            shift_path = sys.argv[0]
            with open(shift_path) as file:
                for line in file:
                    x, y, z, corr = line.split()
                    shifts.append((int(x), int(y), int(z)))
        else:
            sys.stderr.write("%s: unknown option '%s'\n" % (me, sys.argv[0]))
            sys.exit(2)
    else:
        break
try:
    (tx, ty), (nx, ny, nz), (ox,
                             oy), path = stitch.mesospim.read_tiles(sys.argv)
except ValueError:
    tx = ty = 2
    nx = ny = nz = 200
    ox = oy = 10
    path = sys.argv
dtype = np.dtype("<u2")
stride = [8, 8, 8]
src = [np.memmap(e, dtype, 'r', 0, (nx, ny, nz), order='F') for e in path]
positions0 = []
for x in range(tx):
    for y in range(ty):
        positions0.append([x * (nx - ox), y * (ny - oy), 0])
if shifts != []:
    pairs = []
    for x in range(tx):
        for y in range(ty):
            i = x * ty + y
            if x + 1 < tx:
                pairs.append((i, (x + 1) * ty + y))
            if y + 1 < ty:
                pairs.append((i, x * ty + y + 1))
    positions = stitch.Rigid.place(pairs, positions0, shifts)
    positions = [list(e) for e in positions]
else:
    positions = copy.copy(positions0)

fig, ax = matplotlib.pyplot.subplots()
fig.tight_layout()
fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
fig.canvas.manager.full_screen_toggle()
ax.axis('off')
lx = (nx - ox) * tx
ly = (ny - oy) * ty
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
    s = src[i][::stride[0], ::stride[1], z]
    y = ty * (ny - oy) - y
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
    elif key == "s":
        print()
        with open(".shifts", "w") as file:
            for (x, y, z), (x0, y0, z0) in zip(positions, positions0):
                print(x, y, z)
                file.write("%d %d %d %.16e\n" % (x - x0, y - y0, z - z0, 0.0))


for i in range(len(src)):
    draw(i)
fig.canvas.mpl_connect('key_press_event', press)
fig.tight_layout()
matplotlib.pyplot.show()
