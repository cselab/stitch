import numpy as np

MM = []
XX = []

SRC = []
SINK = []


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

    def upper(self):
        return tuple(p + s
                     for p, s in zip(self.position, SRC[self.source].shape))

    def is_valid(self, coordinate):
        return 0 <= coordinate - self.position[2] < SRC[self.source].shape[
            2] and self.status[coordinate - self.position[2]] == self.VALID

    def set_invalid(self, coordinate):
        if 0 <= coordinate - self.position[2] < SRC[self.source].shape[2]:
            self.status[coordinate - self.position[2]] = self.INVALID

    def set_isolated(self, coordinate):
        if 0 <= coordinate - self.position[2] < SRC[self.source].shape[2]:
            self.status[coordinate - self.position[2]] = self.ISOLATED
