import numpy as np
import pytest
import scipy.optimize

import stitch.Rigid as st
import stitch.Tracking as trk
import stitch.Wobbly as stw
import stitch.union_find as uf


def test_union_find_components():
    g = uf.union_find(6)
    g.union(0, 1)
    g.union(3, 4)
    g.union(1, 4)
    assert sorted(map(sorted, g.components())) == [[0, 1, 3, 4], [2], [5]]


def test_union_find_long_chain():
    n = 3000
    g = uf.union_find(n)
    for i in range(1, n):
        g.union(i, i - 1)
    assert len(g.components()) == 1


@pytest.mark.parametrize("seed", range(20))
def test_match_is_optimal(seed):
    rng = np.random.default_rng(seed)
    m, n = rng.integers(1, 7, size=2)
    cost = rng.random((m, n)) * 10
    cost[rng.random((m, n)) < 0.3] = np.inf
    new = 4.0
    padded = np.pad(cost, [(0, 1), (0, 1)], "constant", constant_values=new)
    got = trk.match(padded)
    assert sorted(got) == list(range(m))
    used = [j for j in got.values() if j < n]
    assert len(used) == len(set(used))
    total = sum(padded[i, j] for i, j in got.items())
    total += new * (n - len(used))
    big = np.full((m + n, n + m), 0.0)
    big[:m, :n] = np.where(np.isfinite(cost), cost, 1e9)
    big[:m, n:] = np.where(np.eye(m), new, 1e9)
    big[m:, :n] = np.where(np.eye(n), new, 1e9)
    r, c = scipy.optimize.linear_sum_assignment(big)
    best = big[r, c].sum()
    assert total == pytest.approx(best)


def test_smooth_window_keeps_constant():
    x = np.full(50, 7)
    for w in ("flat", "hanning", "hamming", "bartlett", "blackman"):
        assert np.all(stw.smooth_window(x, window_length=11, window=w) == 7)


def test_smooth_window_short_input():
    assert np.all(stw.smooth_window(np.array([3, 3, 3]), window_length=10) == 3)


def test_smooth_displacements_only_valid_runs():
    d = np.zeros((20, 2), int)
    d[10:] = 5
    valids = np.ones(20, bool)
    valids[9:11] = False
    out = stw.smooth_displacements(d, valids, method="window", window_length=5,
                                   window="flat", binary=None)
    assert np.all(out[:9] == 0) and np.all(out[11:] == 5)


@pytest.mark.parametrize("shift", [-6, -1, 0, 3, 6])
def test_rigid_place_exact(shift):
    positions = [(0, 0, 0), (90, 0, 0), (0, 90, 0), (90, 90, 0)]
    pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]
    true = [(0, 0, 0), (90 + shift, 2, 1), (3, 90, -2), (93 + shift, 92, 0)]
    shifts = [tuple(np.subtract(true[j], true[i]) - np.subtract(positions[j], positions[i]))
              for i, j in pairs]
    got = st.place(pairs, positions, shifts)
    want = np.array(true) - np.min(true, axis=0)
    assert np.array_equal(np.array(got), want)


def test_stitch_weights_symmetric_positive():
    w = stw.stitch_weights(7, 5)
    assert w.min() >= 1 and np.array_equal(w, w[::-1, ::-1])
