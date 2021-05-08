import sys

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from core import BadAlloc, Byte

colors = {"void": 0, "str": 1, 'int': 2}


class Bucket:
    size = 4096

    def __init__(self):
        self._data = Byte()
        self.type = "void"
        self.color = 0

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
        self.type = type(value).__name__
        self.color = Heap.count


class Heap:
    count = 0
    n_buckets = 40  # int(input("Number of columns: "))
    n_rows = 20  # int(input("Number of rows: "))
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
                    Heap.count += 1
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
        xs = list(range(Heap.n_buckets)) * Heap.n_rows
        ys = [y for y in range(Heap.n_rows) for _ in range(Heap.n_buckets)][::-1]
        # zs = [Heap.buckets_used[i][j] for i in range(Heap.n_buckets) for j in range(Heap.n_rows)]
        values = [Heap.buckets[i][j].data.value for i in range(Heap.n_rows) for j in range(Heap.n_buckets)]
        address = [Heap.buckets[i][j] for i in range(Heap.n_rows) for j in range(Heap.n_buckets)]
        tlabels = [Heap.buckets[i][j].type for i in range(Heap.n_rows) for j in range(Heap.n_buckets)]
        zs = [a.color for a in address]

        fig = go.Figure(go.Heatmap(
            x=xs,
            y=ys,
            z=zs,
            hovertemplate="""
            %{text}
            <extra>%{customdata}</extra>
            """,
            customdata=tlabels,
            text=[f'Address: {a}\t\t<i><b>Value</b></i>: {v}' for a, v in zip(address, values)],
            showlegend=False,
            xgap=1,
            ygap=1,
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
            width=Heap.n_buckets * 1840 / 40, height=Heap.n_rows * 1000 / 20,
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


t = new(25)
heap.buckets[0][0:2] = t
heap.buckets_used[0][0:2] = 1
new(10)
new("Hello World")

heap.show2()



