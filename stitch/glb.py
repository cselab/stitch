import numpy as np

MM = []
XX = []

SRC = []
SINK = []

ALIGNMENTS = []


def local_slicing0(position, lower, upper):
    return tuple(
        slice(l - p, u - p) for l, u, p in zip(lower, upper, position))


def source_slicings0(sources, lower, upper):
    return [local_slicing0(s.position, lower, upper) for s in sources]


class Overlap1:
    def __init__(self, lower, shape, sources):
        self.sources = sources
        self.lower = tuple(lower)
        self.shape = shape
        self.upper = tuple(p + s for p, s in zip(self.lower, self.shape))

class Overlap2:
    def __init__(self, lower, upper):
        self.shape = tuple(u - l for u, l in zip(upper, lower))
        self.lower = tuple(lower)
        self.upper = tuple(upper)

class Overlap3:
    def __init__(self, lower, upper):
        self.shape = tuple(u - l for u, l in zip(upper, lower))
        self.lower = lower
        self.upper = upper


class WobblySource:
    ISOLATED = -2
    INVALID = -1
    VALID = 0
    FIXED = 2

    def __init__(self, source, position, tile_position):
        shape = SRC[source].shape
        self.position = position
        self.tile_position = tuple(tile_position)
        self.source = source
        self._wobble = np.zeros((shape[2], 2), dtype=int)
        self.status = np.full(shape[2], self.VALID, dtype=int)

    @property
    def upper(self):
        return tuple(p + s
                     for p, s in zip(self.position, SRC[self.source].shape))

    def wobble_from_positions(self, positions):
        start = self.position[2]
        stop = start + SRC[self.source].shape[2]
        self._wobble[:] = positions[start:stop]
        finite = np.all(np.isfinite(positions[start:stop]), axis=1)
        non_finite = np.logical_not(finite)
        self.status[non_finite] = self.INVALID

    @property
    def lower_wobbly(self):
        wobble_min = np.min(self._wobble, axis=0)
        return self._wobble_to_position(wobble_min, self.position[2])

    @property
    def upper_wobbly(self):
        wobble_max = np.max(self._wobble, axis=0)
        position = self._wobble_to_position(wobble_max, self.position[2])
        shape = SRC[self.source].shape
        return tuple(p + s for p, s in zip(position, shape))

    def coordinate_to_local(self, coordinate):
        position = self.position[2]
        shape = SRC[self.source].shape[2]
        if coordinate < position or coordinate >= position + shape:
            raise RuntimeError('Coordinate %d out of range (%d,%d)!' %
                               (coordinate, position, position + shape))
        return coordinate - position

    def wobble_at_coordinate(self, coordinate):
        local_coordinate = self.coordinate_to_local(coordinate)
        return self._wobble[local_coordinate]

    def status_at_coordinate(self, coordinate):
        local_coordinate = self.coordinate_to_local(coordinate)
        return self.status[local_coordinate]

    def set_status_at_coordinate(self, coordinate, status):
        local_coordinate = self.coordinate_to_local(coordinate)
        self.status[local_coordinate] = status

    def is_valid(self, coordinate):
        return 0 <= coordinate - self.position[2] < SRC[self.source].shape[
            2] and self.status_at_coordinate(coordinate) >= self.VALID

    def set_invalid(self, coordinate):
        if 0 <= coordinate - self.position[2] < SRC[self.source].shape[2]:
            self.set_status_at_coordinate(coordinate, self.INVALID)

    def set_isolated(self, coordinate):
        if 0 <= coordinate - self.position[2] < SRC[self.source].shape[2]:
            self.set_status_at_coordinate(coordinate, self.ISOLATED)

    def _wobble_to_position(self, wobble, coordinate):
        return tuple(wobble[:2]) + (coordinate, ) + tuple(wobble[2:])
