import os
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
                    raise ValueError("'%s' is not an integer in '%s'" % (e[:1], p))
            if len(e) > 0 and e[0] == 'Y':
                try:
                    y = int(e[1:])
                except ValueError:
                    raise ValueError("'%s' is not an integer in '%s'" % (e[:1], p))
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
