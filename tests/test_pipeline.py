import numpy as np
import pytest

import stitch.glb as glb
import stitch.Rigid as st
import stitch.Wobbly as stw

from synth import cut, grid, world

K = (120, 110, 40)
STEP = (100, 92)


def run(tiles, nominal, pairs, tile_pos, processes="serial", wobble_smooth=6):
    glb.SRC[:] = tiles
    shifts, q = st.align(K, pairs, nominal, tile_pos, depth=(20, 18, None),
                         max_shifts=[(-8, 8), (-8, 8), (-5, 5)], clip=65535,
                         processes=processes, verbose=False)
    rigid = st.place(pairs, nominal, list(shifts))
    d, qual, status = stw.align(K, pairs, rigid, max_shifts=((-8, 8), (-8, 8)), prepare=True,
                                find_shifts=dict(cutoff=3 * np.sqrt(2)), processes=processes,
                                verbose=False)
    newpos, comps = stw.place0(K, pairs, rigid, d, qual, status,
                               smooth=dict(method="window", window="hamming", window_length=10,
                                           binary=None),
                               min_quality=-np.inf, processes=processes, verbose=False)
    wobble, wst = stw.place1(K, rigid, newpos, comps,
                             smooth=dict(method="window", window="bartlett",
                                         window_length=wobble_smooth, binary=None),
                             processes=processes, verbose=False)
    return rigid, wobble, wst


def errors(rigid, wobble, truth, wob_true):
    out = {}
    for i in range(1, len(truth)):
        e = []
        for kz in range(K[2]):
            k0 = rigid[i][2] + kz - rigid[0][2]
            if not 0 <= k0 < K[2]:
                continue
            got = np.array(wobble[i][kz]) - np.array(wobble[0][k0])
            want = (np.array(truth[i][:2]) + wob_true[i][kz]
                    - np.array(truth[0][:2]) - wob_true[0][k0])
            e.append(np.abs(got - want).max())
        out[i] = np.array(e)
    return out


def setup(tx, ty, err, lateral=(0, 0), wob=None, seed=1, zpad=12):
    n = (STEP[0] * (tx - 1) + K[0] + 40, STEP[1] * (ty - 1) + K[1] + 40, K[2] + 2 * zpad)
    vol = world(n, seed=seed)
    tile_pos, pairs = grid(tx, ty)
    tiles, nominal, truth, wobs = [], [], [], []
    for i, (x, y) in enumerate(tile_pos):
        ex, ey, ez = err.get((x, y), (0, 0, 0))
        nx, ny = x * STEP[0] + lateral[0] * y, y * STEP[1] + lateral[1] * x
        o = (10 + nx + ex, 10 + ny + ey, zpad + ez)
        w = np.zeros((K[2], 2), int) if wob is None or (x, y) not in wob else wob[x, y]
        tiles.append(cut(vol, o, K, w))
        nominal.append((nx, ny, 0))
        truth.append(o)
        wobs.append(w)
    return tiles, nominal, truth, wobs, pairs, tile_pos, vol


ERR = {(0, 1): (3, -4, 2), (1, 0): (-5, 2, -1), (1, 1): (4, 5, 1)}


@pytest.mark.parametrize("lateral", [(0, 0), (5, -4)])
def test_rigid_z_offsets(lateral):
    tiles, nominal, truth, wobs, pairs, tp, _ = setup(2, 2, ERR, lateral)
    rigid, _, _ = run(tiles, nominal, pairs, tp)
    dz = [r[2] - rigid[0][2] for r in rigid]
    assert dz == [t[2] - truth[0][2] for t in truth]


@pytest.mark.parametrize("lateral", [(0, 0), (5, -4)])
def test_no_wobble_recovers_positions(lateral):
    tiles, nominal, truth, wobs, pairs, tp, _ = setup(2, 2, ERR, lateral)
    rigid, wobble, _ = run(tiles, nominal, pairs, tp)
    for i, e in errors(rigid, wobble, truth, wobs).items():
        assert e.max() <= 1, (i, e)
        assert np.mean(e == 0) >= 0.9, (i, e)


def test_wobble_is_followed():
    z = np.arange(K[2])
    w = np.stack([np.round(3 * np.sin(z / 8)), np.round(2 * np.cos(z / 10))], axis=1).astype(int)
    tiles, nominal, truth, wobs, pairs, tp, _ = setup(2, 2, ERR, wob={(1, 1): w})
    rigid, wobble, _ = run(tiles, nominal, pairs, tp, wobble_smooth=3)
    e = errors(rigid, wobble, truth, wobs)
    assert e[3].max() <= 1, e[3]


def test_non_square_grid_2x3():
    err = {(0, 1): (2, 3, 1), (0, 2): (-3, 2, 0), (1, 0): (4, -2, -1), (1, 1): (0, 0, 2),
           (1, 2): (-2, -3, 1)}
    tiles, nominal, truth, wobs, pairs, tp, _ = setup(2, 3, err)
    rigid, wobble, _ = run(tiles, nominal, pairs, tp)
    for i, e in errors(rigid, wobble, truth, wobs).items():
        assert e.max() <= 1, (i, e)


def test_parallel_equals_serial():
    tiles, nominal, truth, wobs, pairs, tp, _ = setup(2, 2, ERR)
    a = run(tiles, nominal, pairs, tp, processes="serial")
    b = run(tiles, nominal, pairs, tp, processes=2)
    assert a[0] == b[0]
    assert all(np.array_equal(x, y) for x, y in zip(a[1], b[1]))


def test_stitch_reproduces_world():
    tiles, nominal, truth, wobs, pairs, tp, vol = setup(2, 2, {})
    glb.SRC[:] = tiles
    lo = np.min(truth, axis=0)
    positions = [tuple(np.subtract(t, lo)) for t in truth]
    wobble = [np.tile(p[:2], (K[2], 1)) for p in positions]
    status = [np.zeros(K[2], int) for _ in positions]
    shape = stw.shape_wobbly(K, positions, wobble)
    glb.SINK[:] = [np.zeros(shape, "<u2", order="F")]
    stw.stitch(K, positions, wobble, status, processes="serial", verbose=False)
    out = glb.SINK[0]
    want = vol[lo[0]:lo[0] + shape[0], lo[1]:lo[1] + shape[1], lo[2]:lo[2] + shape[2]]
    assert out.shape == want.shape
    assert np.array_equal(out, want), np.abs(out.astype(int) - want).max()


def test_blank_slices_do_not_corrupt_neighbours():
    tiles, nominal, truth, wobs, pairs, tp, _ = setup(2, 2, ERR)
    for t in tiles:
        t[:, :, 2 * K[2] // 3:] = 100
    rigid, wobble, wst = run(tiles, nominal, pairs, tp)
    e = errors(rigid, wobble, truth, wobs)
    for i in e:
        signal = e[i][:2 * K[2] // 3 - 3]
        assert signal.max() <= 1, (i, signal)


def test_isolated_slices_are_stitched():
    tiles, nominal, truth, wobs, pairs, tp, vol = setup(2, 2, ERR)
    rigid, wobble, wst = run(tiles, nominal, pairs, tp)
    skipped = [(i, kz) for i, s in enumerate(wst) for kz in range(K[2]) if s[kz] != stw.S_VALID]
    assert skipped == [], skipped
