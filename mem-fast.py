import math
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from core import confirm, BadAlloc

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

        raise BadAlloc

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


start = perf_counter()
for _ in range(9):
    new(5)

print(f"{perf_counter() - start}s")

print(heap.blocks_used.sum())
# heap.print()
# print(heap.blocks_used.sum())

# python mem-fast.py build_ext --inplace
