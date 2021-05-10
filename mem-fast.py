import math
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from functools import cache

from core import confirm, BadAlloc

block_size = 50


class Heap:
    n_rows = confirm("Number of rows: ")
    n_buckets = confirm("Number of columns: ")
    buckets_used = np.zeros((n_rows, n_buckets))
    b = np.arange(n_rows * n_buckets)

    def first(self, obj, force_size=None):
        blocks = force_size
        if not force_size or not isinstance(force_size, int):
            blocks = math.ceil(sys.getsizeof(obj) / block_size)

        for k in np.extract(self.buckets_used == 0, self.b):
            ind = np.unravel_index((range(k, k+blocks)), (Heap.n_rows, Heap.n_buckets))
            if (self.buckets_used[ind] == 0).all():
                self.buckets_used[ind] = 1
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
        im = ax.imshow(Heap.buckets_used)
        ax.set_title("Allocated Memory")
        fig.tight_layout()
        plt.show()

    def print(self):
        np.set_printoptions(linewidth=400)
        x = np.where(self.buckets_used == 1, "|x|", '| |')
        print(x)
        print()

    def flat_to_2D(self, index, columns):
        return index % columns, index // columns


heap = Heap()


def new(obj, fit="first", force_size=None):
    return heap.first(obj, force_size)


start = perf_counter()
for i in range(1000):
    new(5)

print(f"{perf_counter() - start}s")
heap.print()