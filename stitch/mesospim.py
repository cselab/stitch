import os
import re
import collections.abc


def sort_tiles(pathes):
    if not isinstance(pathes, collections.abc.Iterable):
        raise TypeError("'%s' is not iterable" % type(pathes))
    lst = []
    for p in pathes:
        x = y = None
        for e in os.path.basename(p).split('_'):
            if len(e) > 0 and e[0] == 'X':
                try:
                    x = int(e[1:])
                except ValueError:
                    raise ValueError("'%s' is not an integer in '%s'" %
                                     (e[:1], p))
            if len(e) > 0 and e[0] == 'Y':
                try:
                    y = int(e[1:])
                except ValueError:
                    raise ValueError("'%s' is not an integer in '%s'" %
                                     (e[:1], p))
        if x == None:
            raise ValueError("no X positon in '%s'" % p)
        if y == None:
            raise ValueError("no Y positon in '%s'" % p)
        lst.append(((x, y), p))
    if lst == []:
        raise ValueError("pathes cannot be empty")
    lst.sort(reverse=True)
    tile_positions, pathes = zip(*lst)
    x, y = zip(*tile_positions)
    tx = len(set(x))
    ty = len(set(y))
    return pathes, tx, ty


def read_meta(path):
    D = {}
    KEY_CONV = (
        ("Pixelsize in um", float),
        ("x_pixels", int),
        ("x_pos", float),
        ("y_pixels", int),
        ("y_pos", float),
        ("z_planes", int),
    )
    with open(path) as file:
        for line in file:
            line = re.sub("\n$", "", line)
            if not re.match(r"^[ \t]*$", line):
                val = re.sub(r".*\]", "", line)
                val = val.strip()
                if val != "":
                    key = re.sub(r"^\[", "", line)
                    key = re.sub(r"\].*", "", key)
                    for key0, conv in KEY_CONV:
                        if key == key0:
                            val = conv(val)
                    D[key] = val
    return D


D = read_meta("poc/meta.txt")
for k, v in D.items():
    print(k, v)
