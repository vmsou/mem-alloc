import math
import sys

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from core import BadAlloc, Byte, colors, confirm


class Bucket:
    # TODO: Change size so it won't be const static, so best and first can be differentiated
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
        self.data.value = value
        self.type = type(value).__name__
        self.id = Heap.count


class Heap:
    count = 0
    n_rows = confirm("Number of rows: ")
    n_buckets = confirm("Number of columns: ")
    buckets = np.array([Bucket() for _ in range(n_rows * n_buckets)]).reshape((n_rows, n_buckets))
    buckets_used = np.zeros((n_rows, n_buckets))

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
        used = self.buckets_used.flatten()
        for k in range(Heap.n_rows * Heap.n_buckets - blocks):
            if (used[k:k+blocks] == 0).all():
                Heap.count += 1
                x, y = 0, 0
                for i in range(blocks - 1, -1, -1):
                    # Converts flatten index to matrix index
                    x = (k + i) % Heap.n_buckets
                    y = (k + i) // Heap.n_buckets
                    self.buckets_used[y][x] = 1
                    self.buckets[y][x].data = obj
                return self.buckets[y][x]

        raise BadAlloc("Not enough space")

    def best(self, obj):
        pass

    def worst(self, obj):
        pass

    def free(self, p):
        # Separates id from objects
        a = np.fromiter((a.id for a in Heap.buckets.flatten()), dtype=int)
        # Get only Bucket() object that matches id
        bs = np.extract(a == p.id, Heap.buckets.flatten())
        # Get flatten index from matrix
        c = np.extract(a == p.id, np.arange(Heap.n_rows * Heap.n_buckets))
        for i, b in zip(c, bs):
            Heap.buckets_used[i // Heap.n_buckets][i % Heap.n_buckets] = 0  # flatten index converted to matrix index
            b.id = 0
            b.type = "void"
        return

    def show(self):
        fig, ax = plt.subplots()
        im = ax.imshow(Heap.buckets_used)
        ax.set_title("Allocated Memory")
        fig.tight_layout()
        plt.show()

    def show2(self):
        xs = list(range(Heap.n_buckets)) * Heap.n_rows
        ys = np.repeat(np.arange(Heap.n_rows - 1, -1, -1), Heap.n_buckets)
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
            tickvals=list(range(Heap.n_rows))[::-1],
            ticktext=list(range(Heap.n_rows)),

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
        x = np.where(self.buckets_used == 1, "|x|", '| |')
        for row in x:
            print(str(row).replace("'", '').replace('[', '').replace(']', ''))


heap = Heap()


def new(obj, fit="best"):
    if fit == "best":
        return heap.allocate(obj)
    elif fit == "first":
        return heap.first(obj)
    elif fit == "worst":
        return heap.worst(obj)
    else:
        raise NotImplementedError("fit type not found")


def delete(p):
    return heap.free(p)


def simulate():
    Heap.count = 1
    vazio = []
    vazio.extend((0, i) for i in (1, 3, 9, 11, 14, 15, 17, 18, 19))
    vazio.extend((1, i) for i in (0, 4, 6, 8, 9, 12, 15, 18))
    vazio.extend((2, i) for i in (0, 4, 6, 7, 8, 11, 14, 15, 18))
    vazio.extend((3, i) for i in (5, 10, 11, 12, 14, 18))
    vazio.extend((4, i) for i in (0, 1, 2, 4, 5, 9, 10, 13, 17, 18, 19))
    for i in range(heap.n_rows):
        for j in range(heap.n_buckets):
            if (i, j) not in vazio:
                heap.buckets_used[i][j] = 1
                heap.buckets[i][j].id = 1


def main():
    t1 = new(50, fit='first')
    print(t1)
    print(*t1)
    t3 = new("Hello World! Testing Block Sizes... This could take 3 blocks")
    t2 = new("Hello World")
    t4 = new([1, 2, 3, 4, 5, 6, 7, 8, 9])
    t5 = new(20)
    delete(t4)
    t6 = new(200)
    heap.show2()


if __name__ == '__main__':
    # main()
    # simulate()
    new("Hello World! Testing Block Sizes... This could take 3 blocks", fit='first')
    new("Hello World! Testing Block Sizes... This could take 3 blocks", fit='best')
    heap.print()
    heap.show2()

