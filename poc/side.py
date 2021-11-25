di = sys.argv[1]

def tile(g):
    g = os.path.join(di, g)
    return tile0(glob.glob(g))


def tile0(path):
    assert path
    lst = []
    for p in path:
        for e in os.path.basename(p).split('_'):
            if len(e) > 0 and e[0] == 'X':
                x = int(e[1:])
            if len(e) > 0 and e[0] == 'Y':
                y = int(e[1:])
        lst.append(((x, y), p))
    lst.sort(reverse=True)
    tile_positions, path = zip(*lst)
    x, y = zip(*tile_positions)
    tx = len(set(x))
    ty = len(set(y))
    return path, tx, ty
