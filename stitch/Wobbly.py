import functools as ft
import math
import multiprocessing as mp
import numpy as np
import os
import skimage.feature as skif
import stitch.glb as glb
import stitch.Rigid as strg
import stitch.Tracking as trk
import stitch.union_find as union_find
import sys


class Layout1:
    def __init__(self, sources):
        self.sources = sources
        self.lower = tuple(np.min([s.position for s in sources], axis=0))
        self.upper = tuple(np.max([s.upper for s in sources], axis=0))
        self.origin = tuple(p if p >= 0 else 0 for p in self.lower)
        self.shape = tuple(u - o for u, o in zip(self.upper, self.origin))


class Slice0:
    def __init__(self, source, coordinate, position):
        shape = glb.SRC[source].shape[0], glb.SRC[source].shape[1]
        self.coordinate = coordinate
        self.shape = shape
        self.source = source
        self.upper = position[0] + shape[0], position[1] + shape[1]
        self.position = position

    def __getitem__(self, i):
        return glb.SRC[self.source].__getitem__((*i, self.coordinate))


NOSIGNAL = -5
NOMINIMA = -4
UNALIGNED = -3
UNTRACED = -2
INVALID = -1
VALID = 0
MEASURED = 1
ALIGNED = 2
FIXED = 3

S_ISOLATED = -2
S_INVALID = -1
S_VALID = 0


def fix_unaligned(status, displacements, qualities):
    n_status = len(status)
    unaligned = np.array(status == UNALIGNED, dtype=int)
    unaligned = np.pad(unaligned, (1, 1), 'constant')
    delta = np.diff(unaligned)
    starts = np.where(delta > 0)[0]
    ends = np.where(delta < 0)[0]
    if len(starts) == 0:
        return
    if len(starts) == 1 and starts[0] == 0 and len(
            ends) == 1 and ends[0] == n_status:
        status[:] = self.INVALID
        return
    for s, e in zip(starts, ends):
        if s > 0 and status[s - 1] >= self.VALID:
            left = displacements[[s - 1]]
        else:
            left = None
        if e < n_status and status[e] >= self.VALID:
            right = displacements[[e]]
        else:
            right = None
        if left is None and right is None:
            status[s:e] = self.INVALID
        else:
            if left is None:
                displacements[s:e] = right
                qualities[s:e] = qualities[e]
            elif right is None:
                displacements[s:e] = left
                qualities[s:e] = qualities[s - 1]
            else:
                displacements[s:e] = np.array(
                    np.round((right - left) * 1.0 / (e - s + 1) *
                             np.arange(1, e - s + 1)[:, np.newaxis] + left),
                    dtype=int)
                qs = qualities[s - 1]
                qe = qualities[e]
                if np.isfinite(qs) and np.isfinite(qe):
                    qualities[s:e] = (qe - qs) / (e - s + 1) * np.arange(
                        1, e - s + 1) + qs
                elif np.isfinite(qe):
                    qualities[s:e] = qe
                else:
                    qualities[s:e] = qs
            status[s:e] = FIXED


def shape_wobbly(shape, positions, wobble):
    sx, sy, sz = shape
    lx, ly = np.min(wobble, axis=(0, 1))
    ux, uy = np.max(wobble, axis=(0, 1))
    lz = min(p[2] for p in positions)
    uz = max(p[2] for p in positions)
    lx = max(0, lx)
    ly = max(0, ly)
    lz = max(0, lz)
    return ux - lx + sx, uy - ly + sy, uz - lz + sz


def origin_wobbly(positions, wobble):
    x, y = np.min(wobble, axis=(0, 1))
    z = min(p[2] for p in positions)
    return max(0, x), max(0, y), max(0, z)


def align(pairs, positions, max_shifts, prepare, find_shifts, verbose,
          processes):
    if verbose:
        sys.stderr.write('Wobbly: start align')

    def a2arg(i, j):
        return i, positions[i], j, positions[j]

    f = ft.partial(align_pair,
                   max_shifts=max_shifts,
                   prepare=prepare,
                   find_shifts=find_shifts,
                   verbose=verbose)
    if processes == 'serial':
        results = [f(*a2arg(i, j)) for i, j in pairs]
    else:
        with mp.Pool(processes) as e:
            results = e.starmap(f, (a2arg(i, j) for i, j in pairs))
    return zip(*results)


def align_pair(source1, p1, source2, p2, max_shifts, prepare, find_shifts,
               verbose):
    if verbose:
        sys.stderr.write('Wobbly: align_pair\n')
    s1 = glb.SRC[source1].shape
    s2 = glb.SRC[source2].shape

    z1 = p1[2]
    z2 = p2[2]

    (mlx, mux), (mly, muy) = max_shifts

    start = max(z1, z2)
    stop = min(z1 + s1[2], z2 + s2[2])
    if start > stop:
        raise ValueError('The sources do not overlap!')
    n_slices = stop - start
    s1lx, s1ux, s2lx, s2ux, pad1x, pad2x, np1x, np2x, roilx, roiux = strg.padding0(
        s1[0], p2[0] - p1[0], s2[0], mlx, mux)
    s1ly, s1uy, s2ly, s2uy, pad1y, pad2y, np1y, np2y, roily, roiuy = strg.padding0(
        s1[1], p2[1] - p1[1], s2[1], mly, muy)
    i1 = glb.SRC[source1][s1lx:s1ux, s1ly:s1uy, start - z1:stop - z1]
    i2 = glb.SRC[source2][s2lx:s2ux, s2ly:s2uy, start - z2:stop - z2]
    if prepare:
        b10, b11 = norm_coef(i1)
        b20, b21 = norm_coef(i2)
    status = INVALID * np.ones(n_slices, dtype=int)
    errors = np.zeros((n_slices, roiux if roilx is None else -roilx,
                       roiuy if roily is None else -roily))
    sx = s1ux - s1lx + pad1x[0] + pad1x[1]
    sy = s1uy - s1ly + pad1y[0] + pad1y[1]

    w1 = np.zeros((sx, sy))
    w1[np1x, np1y] = 1
    w1fft = np.fft.fftn(w1)

    w2 = np.zeros((sx, sy))
    w2[np2x, np2y] = 1
    w2fft = np.fft.fftn(w2)
    nrm = np.fft.ifftn(w1fft * np.conj(w2fft))
    nrm = np.abs(nrm[roilx:roiux, roily:roiuy])
    eps = 2.2204e-16
    nrm[nrm < eps] = eps
    for a in range(start, stop):
        if verbose and 100 * (a - start) % n_slices  == 0:
            sys.stderr.write('Wobbly: alignment: slice %d / %d\n' %
                             (a - start, stop - start))
        if prepare:
            i1a = b11 * i1[:, :, a - start] + b10
            i2a = b21 * i2[:, :, a - start] + b20
        else:
            i1a = i1[:, :, a - start]
            i2a = i2[:, :, a - start]
        i1a = np.pad(i1a, (pad1x, pad1y), 'constant')
        i2a = np.pad(i2a, (pad2x, pad2y), 'constant')
        i1fft = np.fft.fftn(i1a)
        i2fft = np.fft.fftn(i2a)
        s1fft = np.fft.fftn(i1a * i1a)
        s2fft = np.fft.fftn(i2a * i2a)
        wssd = w1fft * np.conj(s2fft) + s1fft * np.conj(
            w2fft) - 2 * i1fft * np.conj(i2fft)
        wssd = np.fft.ifftn(wssd)
        wssd = wssd[roilx:roiux, roily:roiuy]
        errors[a - start] = np.abs(wssd / nrm)
        status[a - start] = MEASURED
    shifts, qualities, status = shifts_from_tracing(errors, status, **find_shifts)
    shifts += (mlx, mly)
    return shifts, qualities, status


def norm_coef(a):
    n = a.size
    m = np.mean(np.mean(a, axis=2))
    l = np.sum(np.einsum('ijk,ijk->k', a, a, dtype=float))
    coef = l - m * m * n
    if coef == 0:
        b0 = -m
        b1 = 1
    else:
        k = 1 / math.sqrt(coef)
        b0 = -m * k
        b1 = k
    return b0, b1


def detect_local_minima(error):
    minima = skif.peak_local_max(-error, min_distance=1, exclude_border=True)

    if len(minima) > 0:
        shifts = [tuple(m) for m in minima]
        qualities = [error[s] for s in shifts]
    else:
        shifts = [(0, ) * error.ndim]
        qualities = [-np.inf]

    return shifts, qualities


def shifts_from_tracing(errors,
                        status,
                        new_trajectory_cost = None,
                        cutoff = None):
    n = len(status)
    qualities = -np.inf * np.ones(n)
    shifts = np.zeros((n, errors.ndim - 1), dtype=int)
    measured = np.where(status == MEASURED)[0]
    if len(measured) == 0:
        return shifts, qualities, status
    mins = [detect_local_minima(error) for error in errors[measured]]
    for i, m in zip(measured, mins):
        if len(m[1]) == 1 and not np.isfinite(m[1][0]):
            status[i] = NOMINIMA
    mins = [m for m in mins if np.isfinite(m[1][0])]
    measured = status == MEASURED
    valids = np.logical_or(measured, status == UNALIGNED)
    valids = np.array(valids, dtype=int)
    valids = np.asarray(np.pad(valids, (1, 1), 'constant'))
    starts = np.where(np.diff(valids) > 0)[0]
    ends = np.where(np.diff(valids) < 0)[0]
    if len(starts) == 0:
        return shifts, qualities, status

    if new_trajectory_cost is None:
        new_trajectory_cost = np.sqrt(np.sum(np.power(errors[0].shape, 2)))

    n_measured = 0
    for s, e in zip(starts, ends):
        measured_se = np.where(measured[s:e])[0]
        n_measured_se = len(measured_se)
        if n_measured_se == 0:
            continue

        positions = [
            mins[i][0] for i in range(n_measured, n_measured + n_measured_se)
        ]
        n_measured += n_measured_se

        trajectories = trk.track_positions(
            positions, new_trajectory_cost=new_trajectory_cost, cutoff=cutoff)
        n_opt = 0
        t_opt = []
        while n_opt < n_measured_se:
            lens = np.array([len(t) for t in trajectories])
            iopt = np.where(lens == np.max(lens))[0]
            if len(iopt) > 1:
                q = [
                    np.sum([
                        errors[t[0]][tuple(positions[t[0]][t[1]])]
                        for t in trajectories[i]
                    ]) for i in iopt
                ]
                iopt = iopt[np.argmin(q)]
            else:
                iopt = iopt[0]
            t_opt.append(trajectories[iopt])
            n_opt += len(t_opt[-1])
            ts = t_opt[-1][0][0]
            te = t_opt[-1][-1][0]
            trajectories = [
                t for t in trajectories
                if (t[0][0] < ts and t[-1][0] < ts) or (
                    t[0][0] > te and t[-1][0] > te)
            ]
            if len(trajectories) == 0:
                break
        measured_se += s
        status[measured_se] = UNTRACED
        for t in t_opt:
            for p in t:
                l, m = p
                i = measured_se[l]
                shifts[i] = positions[l][m]
                qualities[i] = -errors[i][tuple(shifts[i])]
                status[i] = ALIGNED

    return shifts, qualities, status


def place0(pairs, positions, displacements, qualities, status, smooth,
           min_quality, processes, verbose):
    lo = min(p[2] for p in positions)
    hi = max(p[2] + glb.SRC[i].shape[2] for i, p in enumerate(positions))
    n_slices = hi - lo
    if verbose:
        sys.stderr.write('Placement: %d slices\n' % (n_slices))
    n_alignments = len(pairs)
    s_displacements = np.full((n_slices, n_alignments, 2), np.nan)
    s_qualities = np.full((n_slices, n_alignments), -np.inf)
    s_status = np.full((n_slices, n_alignments), INVALID, dtype=int)
    for k in range(n_alignments):
        i, j = pairs[k]
        d = displacements[k] + positions[j][:2] - positions[i][:2]
        q = qualities[k]
        s = status[k]
        fix_unaligned(s, d, q)
        l = max(positions[i][2], positions[j][2])
        u = min(positions[i][2] + glb.SRC[i].shape[2],
                positions[j][2] + glb.SRC[j].shape[2])
        if smooth:
            valids = s >= VALID
            if min_quality:
                valids = np.logical_and(valids, q > min_quality)
            s_displacements[l:u, k] = smooth_displacements(d, valids, **smooth)
        else:
            s_displacements[l:u, k] = d
        s_qualities[l:u, k] = q
        s_status[l:u, k] = s
    f = ft.partial(place_slice,
                   positions=[p[:2] for p in positions],
                   alignment_pairs=pairs,
                   min_quality=min_quality)
    if processes == 'serial':
        results = [
            f(d, q, s)
            for d, q, s in zip(s_displacements, s_qualities, s_status)
        ]
    else:
        with mp.Pool(processes) as e:
            results = e.starmap(f, zip(s_displacements, s_qualities, s_status))
    return zip(*results)


def place1(positions, positions_new, components, smooth, processes, verbose):
    n_sources = len(positions)
    wobble = [
        np.zeros((glb.SRC[i].shape[2], 2), dtype=int) for i in range(n_sources)
    ]
    status = [
        np.full(glb.SRC[i].shape[2], S_VALID, dtype=int)
        for i in range(n_sources)
    ]

    positions_new = np.array(positions_new)
    for s, components_slice in enumerate(components):
        for c in components_slice:
            if len(c) == 1:
                i = c[0]
                if 0 <= s - positions[i][2] < glb.SRC[i].shape[2]:
                    z = s - positions[i][2]
                    status[i][z] = S_ISOLATED
    components = [[c for c in components_slice if len(c) > 1]
                  for components_slice in components]
    positions_optimized = _optimize_slice_positions(positions_new,
                                                    components,
                                                    processes=processes,
                                                    verbose=verbose)
    positions_optimized = positions_optimized.swapaxes(0, 1)
    if smooth:
        for p in positions_optimized:
            valids = np.all(np.isfinite(p), axis=1)
            p[:] = smooth_positions(p, valids=valids, **smooth)
    positions_optimized_valid = np.ma.masked_invalid(positions_optimized)
    min_pos = np.array(
        np.min(np.min(positions_optimized_valid, axis=0), axis=0))
    positions_optimized -= min_pos
    for i in range(n_sources):
        po = positions_optimized[i]
        start = positions[i][2]
        stop = start + glb.SRC[i].shape[2]
        wobble[i][:] = po[start:stop]
        finite = np.all(np.isfinite(p[start:stop]), axis=1)
        non_finite = np.logical_not(finite)
        status[i][non_finite] = S_INVALID
    return wobble, status


def place_slice(displacements,
                qualities,
                status,
                positions,
                alignment_pairs,
                min_quality=-np.inf):
    positions = positions[:]
    n_sources = len(positions)
    g = union_find.union_find(n_sources)
    for s, q, a in zip(status, qualities, alignment_pairs):
        if s >= VALID and q > min_quality:
            g.union(a[0], a[1])
    component_ids = g.components()
    component_pairs = []
    component_displacements = []
    for ids in component_ids:
        pairs = []
        displ = []
        for s, q, a, d in zip(status, qualities, alignment_pairs,
                              displacements):
            if s >= VALID and q > min_quality and a[0] in ids:
                pairs.append(a)
                displ.append(d)
        component_pairs.append(pairs)
        component_displacements.append(displ)

    for pairs, displ in zip(component_pairs, component_displacements):
        _place_slice_component(positions, pairs, displ)

    return positions, component_ids


def _place_slice_component(positions, pairs, displacements):
    nalignments = len(pairs)
    if nalignments == 0:
        return positions
    pre_indices = np.unique([p[0] for p in pairs])
    post_indices = np.unique([p[1] for p in pairs])
    node_to_index = np.unique(np.hstack([pre_indices, post_indices]))
    index_to_node = {i: n for n, i in enumerate(node_to_index)}
    nnodes = len(node_to_index)

    n = 2 * nalignments
    m = 2 * (nnodes - 1)
    s = []
    for d in displacements:
        s.extend(d)
    M = np.zeros((n, m))
    k = 0
    for i, j in pairs:
        pre_node = index_to_node[i]
        post_node = index_to_node[j]
        for d in range(2):
            if pre_node > 0:
                M[k, (pre_node - 1) * 2 + d] = -1
            if post_node > 0:
                M[k, (post_node - 1) * 2 + d] = 1
            k = k + 1
    pos = np.dot(np.linalg.pinv(M), s)
    pos = np.hstack([np.zeros(2), pos])
    pos = np.reshape(pos, (-1, 2))
    pos = np.asarray(np.round(pos), dtype=int)
    fixed_id = np.min(pairs)
    fixed_position = positions[fixed_id]
    pos = pos - pos[index_to_node[fixed_id]] + fixed_position
    for i, j in enumerate(node_to_index):
        positions[j] = pos[i]


def _cluster_components(components):
    c_lens = [len(c) for c in components]
    c_ids = np.cumsum(c_lens)
    c_ids = np.hstack([0, c_ids])
    n_components = np.sum(c_lens)

    def is_to_c(s, i):
        return c_ids[s] + i

    def c_to_si(c):
        s = np.searchsorted(c_ids, c, side='right') - 1
        i = c - c_ids[s]
        return s, i

    g = union_find.union_find(n_components)
    for s in range(1, len(components)):
        for i, ci in enumerate(components[s - 1]):
            for j, cj in enumerate(components[s]):
                for c in ci:
                    if c in cj:
                        g.union(is_to_c(s - 1, i), is_to_c(s, j))
                        break
    components_full = tuple(c for c in g.components() if len(c) > 1)
    return components_full, is_to_c, c_to_si


def _optimize_slice_positions(positions, components, processes, verbose):
    n_slices = len(components)
    ndim = len(positions[0, 0])
    cluster_components, si_to_c, c_to_si = _cluster_components(components)
    n_components = len(cluster_components)
    if verbose:
        sys.stderr.write('Placement: found %d components to optimize!\n' %
                         n_components)
    for cci, cluster_component in enumerate(cluster_components):
        n_clusters = len(cluster_component)
        n_s = (n_clusters - 1)
        if verbose:
            sys.stderr.write(
                'Placement: optimizing component %d/%d with %d clusters\n' %
                (cci, n_components, n_clusters))
        slice_to_cluster_ids = [()] * n_slices
        for c in cluster_component:
            s, i = c_to_si(c)
            slice_to_cluster_ids[s] += (i, )
        si_to_d = {}
        d_to_si = {}
        for d, c in enumerate(cluster_component[1:]):
            s, i = c_to_si(c)
            si_to_d[(s, i)] = d
            d_to_si[d] = (s, i)

        s0, i0 = c_to_si(cluster_component[0])
        X = [np.zeros(n_s) for d in range(ndim)]
        M = [np.zeros((n_s, n_s)) for d in range(ndim)]
        for ci, c in enumerate(cluster_component[1:]):
            s, i = c_to_si(c)
            C_si = components[s][i]
            d = si_to_d[(s, i)]
            t = s + 1
            if t < n_slices:
                for j in slice_to_cluster_ids[t]:
                    C_tj = components[t][j]
                    is_first = s0 == t and i0 == j
                    if not is_first:
                        f = si_to_d[(t, j)]
                    for k in C_si:
                        if k in C_tj:
                            for e in range(ndim):
                                X[e][d] += positions[s, k, e] - positions[t, k,
                                                                          e]
                                M[e][d, d] += 1
                                if not is_first:
                                    M[e][d, f] -= 1

            r = s - 1
            if r >= 0:
                for j in slice_to_cluster_ids[r]:
                    C_rj = components[r][j]
                    is_first = s0 == r and i0 == j
                    if not is_first:
                        f = si_to_d[(r, j)]
                    for k in C_si:
                        if k in C_rj:
                            for e in range(ndim):
                                X[e][d] -= positions[r, k, e] - positions[s, k,
                                                                          e]
                                M[e][d, d] += 1
                                if not is_first:
                                    M[e][d, f] -= 1

        if verbose:
            sys.stderr.write(
                'Placement: done constructing constraints for component %d/%d\n'
                % (cci, n_components))
        if isinstance(processes, int):
            glb.MM[:] = M
            glb.XX[:] = X
            with mp.Pool(min(processes, ndim)) as e:
                shifts = e.map(_optimize_shifts, range(len(M)))
            shifts = np.array(shifts).T
        else:
            shifts = [
                np.linalg.lstsq(-M[e], X[e], rcond=None)[0]
                for e in range(ndim)
            ]
            shifts = np.asarray(np.round(shifts), dtype=int).T
        for c in cluster_component[1:]:
            s, i = c_to_si(c)
            C_si = components[s][i]
            d = si_to_d[(s, i)]
            for k in C_si:
                positions[s, k] += shifts[d]

        if verbose:
            sys.stderr.write('Placement: component %d/%d optimized\n' %
                             (cci, n_components))
    return positions


def _optimize_shifts(i):
    ss = np.linalg.lstsq(-glb.MM[i], glb.XX[i], rcond=-1)[0]
    return np.asarray(np.round(ss), dtype=int)


def smooth_binary(x, width=1):
    width = width + 1
    if len(x) < width:
        width = len(x)
    x = x.copy()
    n = len(x)
    x[:width] = np.median(x[:width])
    x[-width:] = np.median(x[-width:])
    for w in range(width, 1, -1):
        starts = range(n - w)
        ends = range(w, n)
        for s, e in zip(starts, ends):
            if x[s] == x[e]:
                x[s:e] = x[s]

    return x


def smooth_window(x, window_length=10, window='bartlett', binary=None):
    if window_length > len(x):
        window_length = len(x)

    if window:
        windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
        if not window in windows:
            raise ValueError('Window not in %r!' % windows)
        if window == 'flat':
            w = np.ones(window_length)
        else:
            w = getattr(np, window)(window_length)
        w /= w.sum()

        x = np.pad(x, (window_length, window_length), 'edge')
        y = np.convolve(w, x, mode='same')[window_length:-window_length]

        y = np.array(np.round(y), dtype=int)
    else:
        y = x.copy()

    if binary:
        y = smooth_binary(y, width=binary)

    return y


def smooth_positions(positions, valids, method='window', **kwargs):
    return smooth_displacements(positions,
                                valids=valids,
                                method=method,
                                **kwargs)


def smooth_displacements(displacements, valids, method='window', **kwargs):
    displacements_smooth = displacements.copy()
    if method is None:
        return displacements_smooth
    valids = np.asarray(np.pad(valids, (1, 1), 'constant'), dtype=int)
    starts = np.where(np.diff(valids) > 0)[0]
    ends = np.where(np.diff(valids) < 0)[0]

    if method == 'window':
        smooth = ft.partial(smooth_window, **kwargs)
    else:
        raise ValueError('Smoothing method %r not valid!' % method)
    ndim = displacements.ndim
    for s, e in zip(starts, ends):
        for d in range(ndim):
            smooth_displacements = smooth(displacements[s:e, d])
            displacements_smooth[s:e, d] = smooth_displacements

    return displacements_smooth


def stitch(shape0, positions, wobble, status, processes, verbose):
    if verbose:
        sys.stderr.write('Stitching: stitching wobbly layout\n')
    origin = origin_wobbly(positions, wobble)
    shape = shape_wobbly(shape0, positions, wobble)
    coordinates = np.arange(origin[2], origin[2] + shape[2])
    layout_slices = []
    for i, c in enumerate(coordinates):
        s = []
        for j, p in enumerate(positions):
            z = c - p[2]
            if 0 <= z < shape0[2] and status[j][z] == S_VALID:
                s.append(Slice0(j, z, wobble[j][z]))
        if s:
            layout_slices.append((i, Layout1(sources=s)))
    if verbose:
        sys.stderr.write('Stitching: stitching %d sliced layouts\n' %
                         len(coordinates))
    f = ft.partial(_stitch_slice,
                   ox=origin[0],
                   oy=origin[1],
                   sx=shape[0],
                   sy=shape[1],
                   verbose=verbose)
    if processes == 'serial':
        for i, l in layout_slices:
            f(i, l)
    else:
        with mp.Pool(processes) as e:
            e.starmap(f, layout_slices)


def _split_region(r, o):
    split = [o]
    rl, ru = r.lower, r.upper
    ol, ou = o.lower, o.upper
    for d in range(2):
        if rl[d] < ol[d]:
            l = ol[:d] + rl[d:]
            u = ou[:d] + (ol[d], ) + ru[d + 1:]
            split.append(glb.Overlap3(lower=l, upper=u))

        if ou[d] < ru[d]:
            l = ol[:d] + (ou[d], ) + rl[d + 1:]
            u = ou[:d] + ru[d:]
            split.append(glb.Overlap3(lower=l, upper=u))
    return split


def _add_overlap_region(regions, region):
    regsadd = [region]
    regscheck = regions
    regsnew = []
    while len(regscheck) > 0 and len(regsadd) > 0:
        rc = regscheck[0]
        found = False
        for a in range(len(regsadd)):
            ra = regsadd[a]
            ovl = np.max([rc.lower, ra.lower], axis=0)
            ovu = np.min([rc.upper, ra.upper], axis=0)
            if np.all(ovu - ovl - 1 >= 0):
                ov = glb.Overlap2(ovl, ovu)
                split = _split_region(rc, ov)
                for s in split:
                    s.sources = rc.sources
                sources = split[0].sources
                sources += tuple(s for s in ra.sources if s not in sources)
                split[0].sources = sources
                regsnew.append(split[0])
                regscheck = split[1:] + regscheck[1:]
                split = _split_region(ra, ov)[1:]
                for s in split:
                    s.sources = ra.sources
                regsadd = regsadd[:a] + split + regsadd[a + 1:]
                found = True
                break
        if not found:
            regsnew.append(rc)
            regscheck = regscheck[1:]
    regsnew = regsnew + regscheck + regsadd
    return regsnew


def _stitch_slice(slice_id, layout, ox, oy, sx, sy, verbose):
    if verbose and slice_id % 100 == 0:
        sys.stderr.write('Stitching: slice %d\n' % slice_id)
    axl, ayl = layout.origin
    axu, ayu = layout.upper
    bxl, byl = ox, oy
    bxu, byu = ox + sx, oy + sy
    xl = max(axl, bxl)
    yl = max(ayl, byl)
    xu = min(axu, bxu)
    yu = min(ayu, byu)
    if xu - xl - 1 < 0 or yu - yl - 1 < 0:
        return

    regions = []
    for s in layout.sources:
        region = glb.Overlap1(lower=s.position, shape=s.shape, sources=(s, ))
        regions = _add_overlap_region(regions, region)
    shape = tuple(max(s, 0) for s in layout.shape)
    new_regions = []
    for i, r in enumerate(regions):
        r.lower = tuple(p if l < p else l
                        for l, p in zip(r.lower, layout.origin))
        r.upper = tuple(p if u < p else u
                        for u, p in zip(r.upper, layout.origin))
        if np.all([u > l for u, l in zip(r.upper, r.lower)]):
            new_regions.append(r)
    regions = new_regions
    new_regions = []
    ps = np.array(layout.origin, dtype=int) + shape
    for i, r in enumerate(regions):
        r.lower = tuple(p if l > p else l for l, p in zip(r.lower, ps))
        r.upper = tuple(p if u > p else u for u, p in zip(r.upper, ps))
        if np.all([u > l for u, l in zip(r.upper, r.lower)]):
            new_regions.append(r)
    regions = new_regions

    sxl, sxu = xl - axl, xu - axl
    syl, syu = yl - ayl, yu - ayl
    fxl, fxu = xl - bxl, xu - bxl
    fyl, fyu = yl - byl, yu - byl
    stitched = np.zeros(shape, dtype='<u2', order='F')
    strg.stitch_by_function_with_weights(sources=layout.sources,
                                         position=layout.origin,
                                         regions=regions,
                                         stitched=stitched)
    np.copyto(glb.SINK[0][fxl:fxu, fyl:fyu, slice_id], stitched[sxl:sxu,
                                                                syl:syu], 'no')
