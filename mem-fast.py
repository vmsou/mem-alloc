import math
import sys

import matplotlib.pyplot as plt
import numpy as np

from core import confirm, BadAlloc

block_size = 50


class Heap:
    count = 0
    n_rows = confirm("Number of rows: ")
    n_buckets = confirm("Number of columns: ")
    buckets_used = np.zeros((n_rows, n_buckets))

    def first(self, obj, force_size=None):
        blocks = force_size
        if not force_size or not isinstance(force_size, int):
            blocks = math.ceil(sys.getsizeof(obj) / block_size)

        used = self.buckets_used.flatten()
        for k in range(Heap.n_rows * Heap.n_buckets - blocks):
            if (used[k:k+blocks] == 0).all():
                Heap.count += 1
                for i in range(blocks - 1, -1, -1):
                    # Converts flatten index to matrix index
                    x = (k + i) % Heap.n_buckets
                    y = (k + i) // Heap.n_buckets
                    self.buckets_used[y][x] = 1
                return

        raise BadAlloc

    def show(self):
        fig, ax = plt.subplots()
        im = ax.imshow(Heap.buckets_used)
        ax.set_title("Allocated Memory")
        fig.tight_layout()
        plt.show()

    def print(self):
        np.set_printoptions(linewidth=5000)
        x = np.where(self.buckets_used == 1, "|x|", '| |')
        for row in x:
            print(str(row).replace("'", '').replace('[', '').replace(']', ''))
        print()


heap = Heap()
heap.first(5, 5000 * 5000 - 1)
heap.print()

