import ctypes


class union_find:
    def __init__(self, n):
        self.n = n
        self.root = (ctypes.c_uint16 * self.n)()
        self.root[:] = range(self.n)

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)
        if a != b:
            self.root[b] = a

    def find(self, a):
        if a == self.root[a]:
            return a
        else:
            b = self.root[a] = self.find(self.root[a])
            return b

    def components(self):
        lbl = (ctypes.c_uint16 * self.n)()
        ans = tuple([] for i in range(self.n))
        cnt = 0
        for i in range(self.n):
            j = self.find(i)
            if lbl[j] == 0:
                lbl[j] = cnt + 1
                cnt += 1
            ans[lbl[j] - 1].append(i)
        return tuple(e for e in ans if e)
