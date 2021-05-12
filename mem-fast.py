import math
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from core import confirm, BadAlloc
import c_python as cpp

block_size = 50


class Heap:
    n_rows = confirm("Number of rows: ")
    n_blocks = confirm("Number of columns: ")
    blocks_used = np.zeros((n_rows, n_blocks), dtype=bool)
    b = range(n_rows * n_blocks)

    def __init__(self):
        np.set_printoptions(linewidth=400)

    def first(self, obj, force_size=None):
        blocks = force_size
        if not force_size or not isinstance(force_size, int):
            blocks = math.ceil(sys.getsizeof(obj) / block_size)

        used = Heap.blocks_used.flat
        for k in (i for i in self.b if not used[i]):
            ind = np.unravel_index((range(k, k + blocks)), (Heap.n_rows, Heap.n_blocks))
            if (self.blocks_used[ind] == False).all():
                self.blocks_used[ind] = True
                """
                for i in range(blocks - 1, -1, -1):
                    # Converts flatten index to matrix index
                    # x, y = self.flat_to_2D(k + i, self.n_buckets)
                    x, y = np.unravel_index(k + i, (Heap.n_rows, Heap.n_buckets))
                    self.buckets_used[y][x] = 1
                """
                return

        raise BadAlloc("Not enough space")

    def c_first(self, obj, force_size=None):
        blocks = force_size
        if not force_size or not isinstance(force_size, int):
            blocks = math.ceil(sys.getsizeof(obj) / block_size)

        count = cpp.first(Heap.blocks_used, blocks, Heap.n_rows * Heap.n_blocks)
        if count != blocks:
            raise BadAlloc("Not enough space")

    def show(self):
        fig, ax = plt.subplots()
        im = ax.imshow(Heap.blocks_used)
        ax.set_title("Allocated Memory")
        fig.tight_layout()
        plt.show()

    def print(self):
        x = np.where(self.blocks_used == 1, "|x|", '| |')
        if self.n_blocks > 100 or self.n_rows > 100:
            print(x)
        else:
            for row in x:
                print(str(row).replace("'", '').replace('[', '').replace(']', ''))
        print()

    def flat_to_2D(self, index, columns):
        return index % columns, index // columns


heap = Heap()


def new(obj, fit="first", force_size=None):
    return heap.first(obj, force_size)


def main():
    n = confirm("Number of allocs: ")

    start = perf_counter()
    # cpp.multi_first(heap.blocks_used.ctypes.data, n, 1, heap.n_rows * heap.n_blocks)
    for _ in range(n):
        heap.c_first(5)
    print(f"[C-Python Loop] {perf_counter() - start}s")

    heap.blocks_used[:][:] = 0

    start = perf_counter()
    cpp.multi_first(heap.blocks_used, n, 1, heap.n_rows * heap.n_blocks)
    print(f"[C] {perf_counter() - start}s")
    print(heap.blocks_used.sum())

    heap.blocks_used[:][:] = 0

    start = perf_counter()
    for _ in range(n):
        new(5)
    print(f"[Python] {perf_counter() - start}s")

    print(heap.blocks_used.sum())


if __name__ == '__main__':
    main()
