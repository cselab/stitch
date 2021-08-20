import stitch.Wobbly
import numpy as np


def prepare(array):
    array -= np.mean(array)
    array *= 1.0 / np.sqrt(np.sum(array * array))


a = np.array([[
    [1, 2, 3],
    [10, 20, 30],
], [
    [-1, -2, -3],
    [10, 20, 30],
]],
             dtype=np.float64)

b0, b1 = Alignment.Wobbly.norm_coef(a)
b = b0 + a * b1
prepare(a)

print(a - b)
