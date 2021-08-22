import copy
import functools as ft
import inspect as insp
import itertools as itt
import multiprocessing as mp
import numpy as np
import os
import sys
import stitch.glb as glb
import stitch.union_find


def padding0(s1, p2, s2, minx, maxx):
    o1l = max(0, p2 + minx)
    o1u = min(s1, p2 + s2 + maxx)
    o2l = max(-maxx, p2)
    o2u = min(s1 - minx, p2 + s2)
    maxx += 1
    minx = max(minx, o1l - o2u)
    l_min = min(o1l, o2l + minx)
    l_max = min(o1l, o2l + maxx)
    u_min = max(o1u, o2u + minx)
    u_max = max(o1u, o2u + maxx)
    s_min = u_min - l_min
    s_max = u_max - l_max
    if s_min >= s_max:
        pad1 = o1l - l_min, u_min - o1u
        pad2 = o2l + minx - l_min, u_min - o2u - minx
    else:
        pad1 = o1l - l_max, u_max - o1u
        pad2 = o2l + maxx - l_max, u_max - o2u - maxx

    t = o1u - o1l + pad1[0] + pad1[1]
    np1 = slice(pad1[0], t - pad1[1])
    np2 = slice(pad2[0], t - pad2[1])
    if pad2[0] == 0:
        roil = None
        roiu = maxx - minx
    else:
        roil = -(maxx - minx)
        roiu = None

    return o1l, o1u, o2l - p2, o2u - p2, pad1, pad2, np1, np2, roil, roiu


def align_pair(src1, src2, shift, axis, depth, max_shifts, clip, background,
               verbose):
    if verbose:
        sys.stderr.write('Rigid: %d: start align_pair\n' % os.getpid())
    depth = depth[axis]
    max_shifts = max_shifts[:axis] + max_shifts[axis + 1:]
    s1 = glb.SRC[src1].shape
    s2 = glb.SRC[src2].shape
    d1 = max(0, s1[axis] - depth)
    d2 = min(depth, s2[axis])
    if axis == 0:
        mip1 = np.max(glb.SRC[src1][d1:, :, :], axis=axis)
        mip2 = np.max(glb.SRC[src2][:d2, :, :], axis=axis)
        shift = shift[1], shift[2]
    else:
        mip1 = np.max(glb.SRC[src1][:, d1:, :], axis=axis)
        mip2 = np.max(glb.SRC[src2][:, :d2, :], axis=axis)
        shift = shift[0], shift[2]
    s1lx, s1ux, s2lx, s2ux, pad1x, pad2x, np1x, np2x, roilx, roiux = padding0(
        mip1.shape[0], shift[0], mip1.shape[0], max_shifts[0][0],
        max_shifts[0][1])

    s1ly, s1uy, s2ly, s2uy, pad1y, pad2y, np1y, np2y, roily, roiuy = padding0(
        mip2.shape[1], shift[1], mip2.shape[1], max_shifts[1][0],
        max_shifts[1][1])
    np1 = np1x, np1y
    np2 = np2x, np2y
    max_shifts = np.array(max_shifts, dtype=int)
    i1 = mip1[s1lx:s1ux, s1ly:s1uy]
    i2 = mip2[s2lx:s2ux, s2ly:s2uy]

    i1 = np.asarray(i1, dtype=float)
    i2 = np.asarray(i2, dtype=float)
    i1[i1 > clip] = clip
    i2[i2 > clip] = clip

    i1 = np.pad(i1, (pad1x, pad1y), 'constant')
    i2 = np.pad(i2, (pad2x, pad2y), 'constant')
    w1 = np.zeros(i1.shape)
    w1[np1] = 1
    w1fft = np.fft.fftn(w1)

    if np.any([
            s1.start != s2.start or s1.stop != s2.stop
            for s1, s2 in zip(np1, np2)
    ]):
        w2 = np.zeros(i2.shape)
        w2[np2] = 1
        w2fft = np.fft.fftn(w2)
    else:
        w2 = w1
        w2fft = w1fft
    i1fft = np.fft.fftn(i1)
    i2fft = np.fft.fftn(i2)
    s1fft = np.fft.fftn(i1 * i1)
    s2fft = np.fft.fftn(i2 * i2)
    wssd = w1fft * np.conj(s2fft) + s1fft * np.conj(
        w2fft) - 2 * i1fft * np.conj(i2fft)
    wssd = np.fft.ifftn(wssd)
    nrm = np.fft.ifftn(w1fft * np.conj(w2fft))
    wssd = wssd[roilx:roiux, roily:roiuy]
    nrm = nrm[roilx:roiux, roily:roiuy]
    eps = 2.2204e-16
    nrm[nrm <= eps] = eps
    cc = np.abs(wssd / nrm)
    shift = np.argmin(cc)
    shift = np.unravel_index(shift, cc.shape)
    quality = -(cc[tuple(shift)])
    shift_min = max_shifts[0][0], max_shifts[1][0]
    shift = tuple(s + m for s, m in zip(shift, shift_min))
    if verbose:
        sys.stderr.write('Rigid: %d: shift = %r, quality = %.2e\n' %
                         (os.getpid(), shift, quality))
    if axis == 0:
        return (0, shift[0], shift[1]), quality
    else:
        return (shift[0], 0, shift[1]), quality


def align(pairs, positions, tile_position, depth, max_shifts, clip, background,
          processes, verbose):
    f = ft.partial(align_pair,
                   depth=depth,
                   max_shifts=max_shifts,
                   clip=clip,
                   background=background,
                   verbose=verbose)

    def a2arg(i, j):
        axis = 1 if tile_position[i][0] == tile_position[j][0] else 0
        shift = (positions[i][0] - positions[j][0],
                 positions[i][1] - positions[j][1],
                 positions[i][2] - positions[j][2])
        return i, j, shift, axis

    if processes == 'serial':
        results = [f(*a2arg(i, j)) for i, j in pairs]
    else:
        with mp.Pool(processes) as e:
            results = e.starmap(f, (a2arg(i, j) for i, j in pairs))
    return zip(*results)


def place(pairs, positions, shifts):
    n = 3 * len(shifts)
    m = 3 * (len(positions) - 1)
    s = [ ]
    for (i, j), shift in zip(pairs, shifts):
        s.extend(q + sh - p for p, q, sh in zip(positions[i], positions[j], shift))
    M = np.zeros((n, m))
    k = 0
    for i, j in pairs:
        for d in range(3):
            if i > 0:
                M[k, (i - 1) * 3 + d] = -1
            if j > 0:
                M[k, (j - 1) * 3 + d] = 1
            k += 1
    pos = np.dot(np.linalg.pinv(M), s)
    pos = np.hstack([np.zeros(3), pos])
    pos = np.reshape(pos, (-1, 3))
    pos = np.asarray(np.round(pos), dtype=int)
    pos -= np.min(pos, axis=0)
    return [tuple(p) for p in pos]



def stitch_weights(shape):
    ranges = [np.arange(s) for s in shape]
    mesh = np.meshgrid(*ranges, indexing='ij')
    mesh = [np.min([m, np.max(m) - m], axis=0) for m in mesh]
    weights = np.min(mesh, axis=0) + 1
    return weights


def stitch_by_function_with_weights(sources, position, regions, stitched):
    shapes = [s.shape for s in sources]
    w = stitch_weights(shapes[0])
    for r in regions:
        nsources = len(r.sources)
        if nsources > 1:
            rd = np.zeros((nsources, ) + r.shape)
            wd = np.zeros((nsources, ) + r.shape)
            for i, s, sl in zip(
                    range(len(r.sources)), r.sources,
                    glb.source_slicings0(r.sources, r.lower, r.upper)):
                rd[i] = s[sl]
                wd[i] = w[sl]
            rd = np.average(rd, axis=0, weights=wd)
        else:
            s = r.sources[0]
            rd = s[glb.local_slicing0(r.sources[0].position, r.lower, r.upper)]
        stitched[glb.local_slicing0(position, r.lower, r.upper)] = rd
