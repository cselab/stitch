import matplotlib.pyplot
import sys
import numpy as np
import os

dtype = np.dtype("<u2")
nx, ny, nz = 256, 256, 538
tx, ty = 3, 5
sx = sy = sz = 1
ox = 20
oy = 20
path = (
    '02x04.raw',
    '02x03.raw',
    '02x02.raw',
    '02x01.raw',
    '02x00.raw',
    '01x04.raw',
    '01x03.raw',
    '01x02.raw',
    '01x01.raw',
    '01x00.raw',
    '00x04.raw',
    '00x03.raw',
    '00x02.raw',
    '00x01.raw',
    '00x00.raw',
)
di = "/u/stride8"
path = [os.path.join(di, e) for e in path]

src = [
    np.memmap(e, dtype, 'r', 0, (nx, ny, nz), order='F')[::sx, ::sy, ::sz]
    for e in path
]
kx, ky, kz = np.shape(src[0])

positions = []
for x in range(tx):
    for y in range(ty):
        positions.append([x * (kx - ox), y * (ky - oy), 0])

fig, ax = matplotlib.pyplot.subplots()
fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
manager = matplotlib.pyplot.get_current_fig_manager()
manager.full_screen_toggle()
ax.axis('off')
ax.set_xlim((0, kx * tx))
ax.set_ylim((0, ky * ty))

zslice = [kz // 2]
se = [0]
art = [None for e in src]

def draw(i):
    x, y, z = positions[i]
    z += zslice[0]
    if z < 0:
        z = 0
    elif z > kz - 1:
        z = kz - 1
    s = src[i][:, :, z]
    y = ty * ky - y
    extent = (x, x + kx, y - ky, y)
    cmap = 'Greens' if se[0] == i else 'Greys'
    if art[i] is not None:
        art[i].remove()
    vmin = np.quantile(s, 0.1)
    vmax = np.quantile(s, 0.9)
    art[i] = matplotlib.pyplot.imshow(s.T, alpha=0.5, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)

def press(event):
    n = len(src)
    key = event.key
    if key == "right":
        zslice[0] += 10
        for i in range(n):
            draw(i)
        fig.canvas.draw()
    elif key == "left":
        zslice[0] -= 10
        for i in range(n):
            draw(i)
        fig.canvas.draw()
    elif key == "up":
        se[0] += 1
        if se[0] == n:
            se[0] = 0
        for i in range(n):
            draw(i)
        fig.canvas.draw()
    elif key == "down":
        se[0] -= 1
        if se[0] == -1:
            se[0] = n - 1
        for i in range(n):
            draw(i)
        fig.canvas.draw()
    elif key == "h":
        positions[se[0]][0] -= 1
        draw(se[0])
        fig.canvas.draw()
    elif key == "l":
        positions[se[0]][0] += 1
        draw(se[0])
        fig.canvas.draw()
    elif key == "j":
        positions[se[0]][1] -= 1
        draw(se[0])
        fig.canvas.draw()
    elif key == "k":
        positions[se[0]][1] += 1
        draw(se[0])
        fig.canvas.draw()
    elif key == "i":
        positions[se[0]][2] -= 1
        draw(se[0])
        fig.canvas.draw()
    elif key == "n":
        positions[se[0]][2] += 1
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
    elif key == "q":
        sys.exit(0)

for i in range(len(src)):
    draw(i)
fig.canvas.mpl_connect('key_press_event', press)
matplotlib.pyplot.show()
