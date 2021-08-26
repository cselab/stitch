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
        self.upper = tuple(p + s for p, s in zip(self.lower, shape))


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
