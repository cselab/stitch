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
    def __init__(self, lx, ly, ux, uy):
        self.lower = lx, ly
        self.upper = ux, uy


class Overlap3:
    def __init__(self, lx, ly, ux, uy):
        self.lower = lx, ly
        self.upper = ux, uy
