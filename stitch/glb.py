import numpy as np

MM = []
XX = []

SRC = []
SINK = []

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
