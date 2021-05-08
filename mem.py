import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core import BadAlloc, Byte


class Bucket:
    size = 4096

    def __init__(self):
        self._data = Byte()

    def __repr__(self):
        return f"{hex(id(self))}"

    def __iter__(self):
        yield self.data.value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self.data.value = value


class Heap:
    n_buckets = 3 #int(input("Number of columns: "))
    n_rows = 3 #int(input("Number of rows: "))
    buckets = np.ndarray((n_rows, n_buckets), dtype=object)
    buckets_used = np.zeros((n_rows, n_buckets))

    def __init__(self):
        self._start()

    def allocate(self, obj):
        if sys.getsizeof(obj) > Bucket.size:
            raise BadAlloc("Object too heavy.")
        for i in range(Heap.n_rows):
            for j in range(Heap.n_buckets):
                if not self.buckets_used[i][j]:
                    self.buckets_used[i][j] = 1
                    self.buckets[i][j].data = obj
                    return self.buckets[i][j]
        raise BadAlloc("Not enough space.")

    def free(self, p):
        for i in range(Heap.n_rows):
            for j in range(Heap.n_buckets):
                if self.buckets[j][i] == p:
                    self.buckets_used[j][i] = 0
                    return

    def show(self):
        fig, ax = plt.subplots()
        im = ax.imshow(Heap.buckets_used)
        ax.set_title("Allocated Memory")
        fig.tight_layout()
        plt.show()

    def show2(self):
        xs = list(range(heap.n_buckets)) * heap.n_rows
        ys = [x for x in range(heap.n_rows, 0, -1) for _ in range(heap.n_rows)]
        zs = [heap.buckets_used[i][j] for i in range(heap.n_buckets) for j in range(heap.n_rows)]
        values = [heap.buckets[i][j].data.value for j in range(heap.n_buckets) for i in range(heap.n_rows)]
        address = [heap.buckets[i][j] for j in range(heap.n_buckets) for i in range(heap.n_rows)]

        fig = go.Figure(go.Heatmap(
            x=xs,
            y=ys,
            z=zs,
            hovertemplate="""
            %{text}
            """,
            text=[f'Address: {a}\t\t<i><b>Value</b></i>: {v}' for a, v in zip(address, values)],
            showlegend=False,
        ))
        fig.update_layout(
            hoverlabel_align='right',
            title="Teste",
        )
        fig.update_yaxes(
            scaleanchor='x',
            scaleratio=1,
        )

        fig.show()

    def _start(self):
        for i in range(Heap.n_rows):
            for j in range(Heap.n_buckets):
                Heap.buckets[i][j] = Bucket()


heap = Heap()


def new(obj):
    return heap.allocate(obj)


def delete(p):
    return heap.free(p)


new("Hello")
new(25)
heap.show2()



