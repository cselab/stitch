import math
import numpy as np
import scipy.optimize
import os
import sys


def cdist(a, b, cutoff):
    m = len(a)
    n = len(b)
    dist = np.empty((m, n))
    dist.fill(np.inf)
    xlo, ylo = np.min(b, axis=0)
    xhi, yhi = np.max(b, axis=0)
    D = {}
    for i, (x, y) in enumerate(b):
        u = math.floor((x - xlo) / cutoff)
        v = math.floor((y - ylo) / cutoff)
        if (u, v) not in D:
            D[u, v] = []
        D[u, v].append(i)
    cutoff2 = cutoff * cutoff
    for i, (x, y) in enumerate(a):
        u0 = math.floor((x - xlo) / cutoff)
        v0 = math.floor((y - ylo) / cutoff)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                u = u0 + du
                v = v0 + dv
                if (u, v) in D:
                    for j in D[u, v]:
                        p, q = b[j]
                        d = (x - p)**2 + (y - q)**2
                        if d < cutoff2:
                            dist[i, j] = math.sqrt(d)
    return dist


def track_positions(positions, new_trajectory_cost, cutoff, verbose):
    n_steps = len(positions) - 1
    matches = [None] * n_steps
    for i in range(n_steps):
        if verbose and i % max(n_steps // 100, 1) == 0:
            sys.stderr.write(
                'Wobbly [%d] tracking: %d / %d: patterns: %d %d\n' %
                (os.getpid(), i, n_steps, len(
                    positions[i]), len(positions[i + 1])))
        cost = cdist(positions[i], positions[i + 1], cutoff)
        if new_trajectory_cost is None:
            new_trajectory_cost = np.max(cost) + 1.0
        cost = np.pad(cost, [(0, 1), (0, 1)],
                      'constant',
                      constant_values=new_trajectory_cost)
        matches[i] = match(cost)
    n_pre = len(positions[0])
    trajectories = [[(0, i)] for i in range(n_pre)]
    active = [i for i in range(n_pre)]
    for t in range(1, n_steps + 1):
        n_post = len(positions[t])
        pre = [trajectories[a][-1][1] for a in active]
        m = matches[t - 1]
        post = [m[p] for p in pre]
        p_post = []
        p_end = []
        for a, p in enumerate(post):
            if p == n_post:
                p_end.append(a)
            else:
                trajectories[active[a]].append((t, p))
                p_post.append(p)
        for i in sorted(p_end, reverse=True):
            del active[i]
        new = np.setdiff1d(range(n_post), p_post)
        for p in new:
            active.append(len(trajectories))
            trajectories.append([(t, p)])
    return trajectories


def match(cost):
    osize = cost.shape[0] - 1
    nsize = cost.shape[1] - 1
    big = 1e9 + np.nanmax(np.where(np.isfinite(cost), cost, 0)) * (osize + nsize + 1)
    c = np.zeros((osize + nsize, nsize + osize))
    c[:osize, :nsize] = np.where(np.isfinite(cost[:-1, :-1]), cost[:-1, :-1], big)
    c[:osize, nsize:] = np.where(np.eye(osize, dtype=bool), cost[:-1, nsize][:, None], big)
    c[osize:, :nsize] = np.where(np.eye(nsize, dtype=bool), cost[osize, :-1][None, :], big)
    rows, cols = scipy.optimize.linear_sum_assignment(c)
    return {i: (j if j < nsize else nsize) for i, j in zip(rows, cols) if i < osize}
