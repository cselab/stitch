import numpy as np
import pytest

import stitch.glb as glb
import stitch.Rigid as st
import stitch.Wobbly as stw

from synth import cut, grid, world

K = (120, 110, 40)
STEP = (100, 92)


def run(tiles, nominal, pairs, tile_pos, processes="serial", wobble_smooth=6, ms=8):
    glb.SRC[:] = tiles
    shifts, q = st.align(K, pairs, nominal, tile_pos, depth=(20, 18, None),
                         max_shifts=[(-ms, ms), (-ms, ms), (-5, 5)], clip=65535,
                         processes=processes, verbose=False)
    rigid = st.place(pairs, nominal, list(shifts))
    d, qual, status = stw.align(K, pairs, rigid, max_shifts=((-ms, ms), (-ms, ms)), prepare=True,
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


def sine(amp_x, amp_y, px, py):
    z = np.arange(K[2])
    return np.stack([np.round(amp_x * np.sin(z / px)), np.round(amp_y * np.cos(z / py))],
                    axis=1).astype(int)


def step(dx, dy, at):
    w = np.zeros((K[2], 2), int)
    w[at:] = dx, dy
    return w


def stitched(rigid, wobble, wst):
    shape = stw.shape_wobbly(K, rigid, wobble)
    glb.SINK[:] = [np.zeros(shape, "<u2", order="F")]
    stw.stitch(K, rigid, wobble, wst, processes="serial", verbose=False)
    return glb.SINK[0]


def test_all_tiles_wobble():
    wob = {(0, 0): sine(2, 1, 9, 13), (0, 1): sine(-3, 2, 7, 11), (1, 0): sine(1, -3, 12, 6),
           (1, 1): sine(3, 3, 8, 10)}
    tiles, nominal, truth, wobs, pairs, tp, _ = setup(2, 2, ERR, wob=wob)
    rigid, wobble, _ = run(tiles, nominal, pairs, tp, wobble_smooth=3, ms=14)
    for i, e in errors(rigid, wobble, truth, wobs).items():
        assert e.max() <= 1, (i, e)
        assert np.mean(e == 0) >= 0.5, (i, e)


def test_large_wobble_near_max_shift():
    wob = {(1, 1): sine(6, 5, 10, 14)}
    tiles, nominal, truth, wobs, pairs, tp, _ = setup(2, 2, ERR, wob=wob)
    rigid, wobble, _ = run(tiles, nominal, pairs, tp, wobble_smooth=3, ms=14)
    e = errors(rigid, wobble, truth, wobs)
    assert e[3].max() <= 1, e[3]


def test_step_jump_is_followed():
    wob = {(1, 0): step(4, -3, K[2] // 2)}
    tiles, nominal, truth, wobs, pairs, tp, _ = setup(2, 2, ERR, wob=wob)
    rigid, wobble, _ = run(tiles, nominal, pairs, tp, wobble_smooth=3)
    e = errors(rigid, wobble, truth, wobs)[2]
    far = np.r_[e[:K[2] // 2 - 3], e[K[2] // 2 + 3:]]
    assert far.max() <= 1, e


def test_wobbly_beats_rigid():
    wob = {(1, 1): sine(5, 4, 8, 10)}
    tiles, nominal, truth, wobs, pairs, tp, _ = setup(2, 2, ERR, wob=wob)
    rigid, wobble, _ = run(tiles, nominal, pairs, tp, wobble_smooth=3, ms=14)
    flat = [np.tile(r[:2], (K[2], 1)) for r in rigid]
    assert errors(rigid, flat, truth, wobs)[3].max() >= 4
    assert errors(rigid, wobble, truth, wobs)[3].max() <= 1


def test_stitch_with_true_wobble_reproduces_world():
    wob = {(0, 1): sine(-3, 2, 7, 11), (1, 1): sine(3, 3, 8, 10), (1, 0): step(2, -2, 15)}
    tiles, nominal, truth, wobs, pairs, tp, vol = setup(2, 2, {}, wob=wob)
    glb.SRC[:] = tiles
    lo = np.min(truth, axis=0)
    positions = [tuple(np.subtract(t, lo)) for t in truth]
    wobble = [np.array(p[:2]) + w for p, w in zip(positions, wobs)]
    m = np.min([w.min(axis=0) for w in wobble], axis=0)
    wobble = [w - m for w in wobble]
    status = [np.zeros(K[2], int) for _ in positions]
    out = stitched(positions, wobble, status)
    want = vol[lo[0] + m[0]:lo[0] + m[0] + out.shape[0], lo[1] + m[1]:lo[1] + m[1] + out.shape[1],
               lo[2]:lo[2] + out.shape[2]]
    covered = np.zeros(out.shape, bool)
    for p, w in zip(positions, wobble):
        for k in range(K[2]):
            x, y = w[k]
            covered[x:x + K[0], y:y + K[1], p[2] + k] = True
    assert np.array_equal(out[covered], want[covered])
    assert np.all(out[~covered] == 0)


def test_full_pipeline_stitched_image_with_wobble():
    wob = {(0, 1): sine(-3, 2, 7, 11), (1, 1): sine(3, 3, 8, 10)}
    tiles, nominal, truth, wobs, pairs, tp, vol = setup(2, 2, ERR, wob=wob)
    rigid, wobble, wst = run(tiles, nominal, pairs, tp, wobble_smooth=3, ms=14)
    out = stitched(rigid, wobble, wst)
    o = stw.origin_wobbly(rigid, wobble)
    match, total = 0, 0
    for k in range(3, K[2] - 3):
        z = rigid[0][2] + k - o[2]
        lx, ly = np.array(truth[0][:2]) + wobs[0][k] + np.array(o[:2]) - wobble[0][k]
        a = out[5:-5, 5:-5, z].astype(int)
        lo = np.full(a.shape, np.iinfo(int).max)
        hi = np.full(a.shape, np.iinfo(int).min)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                b = vol[lx + 5 + dx:lx + out.shape[0] - 5 + dx, ly + 5 + dy:ly + out.shape[1] - 5 + dy,
                        truth[0][2] + k].astype(int)
                lo = np.minimum(lo, b)
                hi = np.maximum(hi, b)
        ok = (lo - 1 <= a) & (a <= hi + 1)
        filled = a > 0
        match += np.sum(ok[filled])
        total += filled.sum()
    assert match == total, match / total


def test_parallel_stitch_equals_serial(tmp_path):
    wob = {(0, 1): sine(-3, 2, 7, 11), (1, 1): sine(3, 3, 8, 10)}
    tiles, nominal, truth, wobs, pairs, tp, _ = setup(2, 2, ERR, wob=wob)
    rigid, wobble, wst = run(tiles, nominal, pairs, tp, wobble_smooth=3, ms=14)
    a = stitched(rigid, wobble, wst)
    sink = np.memmap(tmp_path / "out.raw", "<u2", "w+", 0, a.shape, order="F")
    glb.SINK[:] = [sink]
    stw.stitch(K, rigid, wobble, wst, processes=3, verbose=False)
    assert np.array_equal(a, sink)


def test_noise_slices_are_interpolated():
    tiles, nominal, truth, wobs, pairs, tp, _ = setup(2, 2, ERR)
    rng = np.random.default_rng(3)
    band = slice(15, 25)
    tiles[3][:, :, band] = rng.integers(0, 60000, tiles[3][:, :, band].shape)
    rigid, wobble, wst = run(tiles, nominal, pairs, tp)
    e = errors(rigid, wobble, truth, wobs)
    assert e[3].max() <= 1, e[3]


@pytest.mark.parametrize("sign", [-1, 1])
def test_shift_at_max_shift(sign):
    ms = 6
    vol = world((260, 160, 60))
    glb.SRC[:] = [cut(vol, (10, 10, 5), K), cut(vol, (110 + sign * ms, 10, 5), K)]
    for cutoff in (3 * np.sqrt(2), None):
        d, q, s = stw.align_pair(0, (0, 0, 0), 1, (100, 0, 0), K, ((-ms, ms), (-ms, ms)), True,
                                 dict(cutoff=cutoff), False)
        assert np.all(d == (sign * ms, 0)), d


def test_none_defaults():
    tiles, nominal, truth, wobs, pairs, tp, _ = setup(2, 2, ERR)
    glb.SRC[:] = tiles
    shifts, q = st.align(K, pairs, nominal, tp, depth=(20, 18, None),
                         max_shifts=[(-8, 8), (-8, 8), (-5, 5)], clip=65535,
                         processes="serial", verbose=False)
    rigid = st.place(pairs, nominal, list(shifts))
    d, qual, status = stw.align(K, pairs, rigid, max_shifts=((-8, 8), (-8, 8)), prepare=True,
                                find_shifts=dict(), processes="serial", verbose=False)
    newpos, comps = stw.place0(K, pairs, rigid, d, qual, status, smooth=None, min_quality=None,
                               processes="serial", verbose=False)
    wobble, wst = stw.place1(K, rigid, newpos, comps, smooth=None, processes="serial",
                             verbose=False)
    assert errors(rigid, wobble, truth, wobs)[3].max() <= 1
