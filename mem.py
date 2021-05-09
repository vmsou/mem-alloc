import itertools
import math
import sys

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from core import BadAlloc, Byte, colors


class Bucket:
    size = 50

    def __init__(self):
        self._data = Byte()
        self.type = "void"
        self.id = 0

    def __repr__(self):
        return f"{hex(id(self))}"

    def __iter__(self):
        yield self.data.value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self.data.value = value  # Bucket.data = 1
        self.type = type(value).__name__
        self.id = Heap.count


class Heap:
    count = 0
    n_rows = 5  # int(input("Number of rows: "))
    n_buckets = 10  # int(input("Number of columns: "))
    buckets = np.ndarray((n_rows, n_buckets), dtype=object)
    buckets_used = np.zeros((n_rows, n_buckets))

    def __init__(self):
        self._start()

    def allocate(self, obj):
        blocks = math.ceil(sys.getsizeof(obj) / Bucket.size)
        for i in range(Heap.n_rows):
            for j in range(Heap.n_buckets):
                if (self.buckets_used[i][j:j+blocks] == 0).all():
                    Heap.count += 1
                    self.buckets_used[i][j:j+blocks] = 1
                    for bucket in self.buckets[i][j:j+blocks]:
                        bucket.data = obj
                    return self.buckets[i][j]
        raise BadAlloc("Not enough space.")

    def first(self, obj):
        blocks = math.ceil(sys.getsizeof(obj) / Bucket.size)
        print(blocks)
        used = self.buckets_used.flatten()
        for k in range(Heap.n_rows * Heap.n_buckets - blocks):
            if (used[k:k+blocks] == 0).all():
                Heap.count += 1
                i = 0
                for _ in range(blocks):
                    x, y = np.unravel_index(k + i, (Heap.n_rows, Heap.n_buckets))
                    self.buckets_used[x][y] = 1
                    self.buckets[x][y].data = obj
                    i += 1

                return self.buckets[x][y]

    def free(self, p):
        blocks = math.ceil(sys.getsizeof(p.data.value) / Bucket.size)
        tmp = p.id
        for i in range(Heap.n_rows):
            for j in range(Heap.n_buckets):
                if self.buckets_used[i][j]:
                    if self.buckets[i][j].id == tmp:
                        self.buckets_used[i][j] = 0
                        self.buckets[i][j].id = 0
                        self.buckets[i][j].type = "void"
                        blocks -= 1
                        if blocks <= 0:
                            return

    def show(self):
        fig, ax = plt.subplots()
        im = ax.imshow(Heap.buckets_used)
        ax.set_title("Allocated Memory")
        fig.tight_layout()
        plt.show()

    def show2(self):
        xs = list(range(Heap.n_buckets)) * Heap.n_rows
        ys = np.repeat(np.arange(Heap.n_rows), Heap.n_buckets)[::-1]
        address = Heap.buckets.flatten()
        values = [a.data.value for a in address]
        tlabels = [a.type for a in address]
        zs = [str(a.id) for a in address]

        fig = go.Figure(go.Heatmap(
            x=xs,
            y=ys,
            z=zs,
            hovertemplate="""
            %{text}
            <extra>%{customdata}</extra>
            """,
            customdata=tlabels,
            colorscale=colors,
            text=[f'Address: {a}\t\t<i><b>Value</b></i>: {v}' for a, v in zip(address, values)],
            showlegend=False,
            xgap=1,
            ygap=1,
            showscale=False,
        ))
        fig.update_yaxes(
            showgrid=False,
            scaleanchor='x',
            tickvals=list(range(Heap.n_rows + 1)),

        )
        fig.update_xaxes(
            showgrid=False,
            tickvals=list(range(Heap.n_buckets)),
        )
        fig.update_layout(
            hoverlabel_align='auto',
            title="Alocação de Memória",
            # width=Heap.n_buckets * 1840 / 40, height=Heap.n_rows * 1000 / 20,
            autosize=True,
        )
        fig.show()

    def print(self):
        np.set_printoptions(linewidth=200)
        x = np.where(self.buckets_used == 1, "x", ' ')
        for row in x:
            print(row)

    def _start(self):
        for i in range(Heap.n_rows):
            for j in range(Heap.n_buckets):
                Heap.buckets[i][j] = Bucket()


heap = Heap()


def new(obj):
    return heap.allocate(obj)


def delete(p):
    return heap.free(p)


def main():
    t1 = heap.first(list(range(15)))
    t2 = heap.first(list(range(35)))
    """t1 = new(50)
    print(t1)
    print(*t1)
    t3 = new("Hello World! Testing Block Sizes... This could take 3 blocks")
    t2 = new("Hello World")
    t4 = new([1, 2, 3, 4, 5, 6, 7, 8, 9])
    t5 = new(20)
    delete(t4)
    t6 = new(200)
    heap.print()
    heap.show2()"""
    heap.print()


if __name__ == '__main__':
    main()
