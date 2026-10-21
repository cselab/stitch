import numpy as np
import stitch.glb as glb
import stitch.Rigid as st
import stitch.Wobbly as stw
import struct
import sys
import zlib


def png(path, a):
    a = np.asarray(a, dtype=np.uint8)
    ny, nx = a.shape[:2]
    color = 2 if a.ndim == 3 else 0
    raw = b''.join(b'\0' + row.tobytes() for row in a)

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack(
            '>I', zlib.crc32(tag + data))

    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', nx, ny, 8, color, 0, 0, 0)))
        f.write(chunk(b'IDAT', zlib.compress(raw, 9)))
        f.write(chunk(b'IEND', b''))


def gray(a, hi):
    return np.clip(255 * np.asarray(a, dtype=float) / hi, 0, 255)


def stitch(positions, wobble):
    status = [np.zeros(kz, int) for _ in positions]
    glb.SINK[:] = [
        np.zeros(stw.shape_wobbly((kx, ky, kz), positions, wobble),
                 dtype,
                 order='F')
    ]
    stw.stitch((kx, ky, kz), positions, wobble, status, 'serial', verbose)
    return glb.SINK[0]


me = "img.py"
verbose = False
dtype = np.dtype("<u2")
processes = 4
tx, ty = 2, 2
kx, ky, kz = 200, 200, 200
ox = oy = 30
ms = 12
path = (
    '200x200x200le.00.00.raw',
    '200x200x200le.00.01.raw',
    '200x200x200le.01.00.raw',
    '200x200x200le.01.01.raw',
)
glb.SRC[:] = (np.memmap(e, dtype, 'r', 0, (kx, ky, kz), order='F')
              for e in path)
positions = []
pairs = []
tile_positions = []
for x in range(tx):
    for y in range(ty):
        tile_positions.append((x, y))
        positions.append((x * (kx - ox), y * (ky - oy), 0))
        i = x * ty + y
        if x + 1 < tx:
            pairs.append((i, (x + 1) * ty + y))
        if y + 1 < ty:
            pairs.append((i, x * ty + y + 1))
nominal = positions
shifts, qualities = st.align((kx, ky, kz),
                             pairs,
                             positions,
                             tile_positions,
                             depth=(20, 20, None),
                             max_shifts=[(-ms, ms), (-ms, ms), (-ms, ms)],
                             clip=np.iinfo(dtype).max,
                             processes=processes,
                             verbose=verbose)
rigid = st.place(pairs, positions, shifts)
displacements, qualities, status = stw.align((kx, ky, kz),
                                             pairs,
                                             rigid,
                                             max_shifts=((-ms, ms), (-ms, ms)),
                                             prepare=True,
                                             find_shifts=dict(cutoff=3 *
                                                              np.sqrt(2)),
                                             processes=processes,
                                             verbose=verbose)
positions_new, components = stw.place0((kx, ky, kz),
                                       pairs,
                                       rigid,
                                       displacements,
                                       qualities,
                                       status,
                                       smooth=dict(method='window',
                                                   window='hamming',
                                                   window_length=10,
                                                   binary=None),
                                       min_quality=-np.inf,
                                       processes=processes,
                                       verbose=verbose)
wobble, status = stw.place1((kx, ky, kz),
                            rigid,
                            positions_new,
                            components,
                            smooth=dict(method='window',
                                        window='bartlett',
                                        window_length=6,
                                        binary=None),
                            processes=processes,
                            verbose=verbose)
flat = lambda p: [np.tile(e[:2], (kz, 1)) for e in p]
xy, xz = [], []
for p, w in ((rigid, flat(rigid)), (rigid, wobble)):
    a = stitch(p, w)
    hi = np.percentile(a, 99)
    o = stw.origin_wobbly(p, w)
    cx, cy = w[0][kz // 2] - o[:2] + (kx - ox // 2, ky - oy // 2)
    cz = p[0][2] + kz // 2 - o[2]
    xy.append(gray(np.max(a[cx - 64:cx + 64, cy - 64:cy + 64, cz - 6:cz + 6], 2).T, hi))
    xz.append(gray(np.max(a[cx - 64:cx + 64, cy - 6:cy + 6, cz - 64:cz + 64], 1).T, hi))
for name, (u, v) in (('xy', xy), ('xz', xz)):
    gap = np.full((u.shape[0], 4), 255)
    png("img/wobble.%s.png" % name, np.kron(np.hstack((u, gap, v)), np.ones((3, 3))))
    png("img/merge.%s.png" % name,
        np.kron(np.dstack((u, v, u)), np.ones((3, 3, 1))))
    sys.stderr.write("%s: img/wobble.%s.png img/merge.%s.png\n" %
                     (me, name, name))
