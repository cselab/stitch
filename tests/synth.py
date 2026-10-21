import numpy as np
import scipy.ndimage as ndi


def world(shape, seed=1, sigma=2.0):
    rng = np.random.default_rng(seed)
    v = ndi.gaussian_filter(rng.random(shape), sigma)
    v = (v - v.min()) / np.ptp(v) * 60000
    return v.astype("<u2")


def cut(vol, origin, shape, wobble=None):
    px, py, pz = origin
    t = np.empty(shape, "<u2")
    for k in range(shape[2]):
        dx, dy = (0, 0) if wobble is None else wobble[k]
        t[:, :, k] = vol[px + dx:px + dx + shape[0], py + dy:py + dy + shape[1], pz + k]
    return np.asfortranarray(t)


def grid(tx, ty):
    tiles = [(x, y) for x in range(tx) for y in range(ty)]
    pairs = []
    for x in range(tx):
        for y in range(ty):
            i = x * ty + y
            if x + 1 < tx:
                pairs.append((i, (x + 1) * ty + y))
            if y + 1 < ty:
                pairs.append((i, x * ty + y + 1))
    return tiles, pairs
