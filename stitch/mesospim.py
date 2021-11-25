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


def read_sizes(pathes):
    if not isinstance(pathes, collections.abc.Iterable):
        raise TypeError("'%s' is not iterable" % type(pathes))
    nx = ny = nz = psize = None
    x_pos = []
    y_pos = []
    for p in pathes:
        p = p + "_meta.txt"
        meta = read_meta(p)
        nx = meta["x_pixels"]
        ny = meta["y_pixels"]
        nz = meta["z_planes"]
        psize = meta["Pixelsize in um"]
        x_pos.append(meta["x_pos"])
        y_pos.append(meta["y_pos"])
    if nx == None:
        raise ValueError("no 'x_pixels' meta files")
    if ny == None:
        raise ValueError("no 'y_pixels' meta files")
    if nz == None:
        raise ValueError("no 'z_planes' meta files")
    if psize == None:
        raise ValueError("no 'Pixelsize in um' meta files")
    x_pos = sorted(set(x_pos))
    y_pos = sorted(set(y_pos))
    if len(x_pos) > 1:
        lx = abs(x_pos[0] - x_pos[1])
        ox = int((nx * psize - lx) / psize)
    else:
        ox = 0
    if len(y_pos) > 1:
        ly = abs(y_pos[0] - y_pos[1])
        oy = int((ny * psize - ly) / psize)
    else:
        oy = 0
    return nx, ny, nz, ox, oy


def read_tiles(pathes):
    pathes, tx, ty = sort_tiles(pathes)
    nx, ny, nz, ox, oy = read_sizes(pathes)
    return (tx, ty), (nx, ny, nz), (ox, oy), pathes


# import glob
# g = "/media/user/demeter16TB_3/FCD/FCD/FCD_P-OCX_2.7_NeuN-Cy3 (2)/*.raw"
# (tx, ty), (nx, ny, nz), (ox, oy), pathes = read_tiles(glob.glob(g))
# print(tx, ty)
# print(nx, ny, nz)
# print(ox, oy)
# print(pathes)
