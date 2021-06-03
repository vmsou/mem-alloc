import numpy as np

from core import size_confirm, BadAlloc


class Block:
    def __init__(self):
        self._index = None
        self.count = None
        self.total_bytes = None
        self.indexes = None

    def __invert__(self):
        delete(self, show=True)

    def set_data(self, index, count, total_bytes):
        self.count = count
        self.index = index
        self.total_bytes = total_bytes

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self.indexes = np.unravel_index(range(value, value-self.count, -1), (Heap.rows, Heap.columns))
        self._index = value


class Heap:
    rows = size_confirm("Linhas")
    columns = size_confirm("Colunas")
    blocks_used = np.random.choice([0, 1], (rows, columns), p=[0.8, 0.2])
    # blocks_used = np.zeros((rows, columns), dtype=bool)
    bytes_map = np.random.choice([10, 30], p=[0.8, 0.2], size=(rows, columns))
    # bytes_map = np.repeat(30, rows * columns).reshape((rows, columns))
    b = range(rows * columns)

    def first_bitwise(self, min_bytes):
        i, count, soma = 0, 0, 0
        return i, count, soma

    def best_bitwise(self, min_bytes):
        lowest_sum = 1e6
        lowest_count = 1e6
        lowest_idx = None

        b_flat = self.blocks_used.flat
        bm_flat = self.bytes_map.flat

        count = 0
        soma = 0

        for i in self.b:
            if not b_flat[i]:
                count += 1
                soma += bm_flat[i]
            else:
                count = 0
                soma = 0

            if count <= lowest_count and lowest_sum >= soma >= min_bytes:
                lowest_sum = soma
                lowest_count = count
                lowest_idx = i

                count = 0
                soma = 0

            if soma >= min_bytes:
                count = 0
                soma = 0

        if lowest_idx is None:
            raise BadAlloc("Espaço Insuficiente")

        print(f"[Best fit] Bytes: {lowest_sum} Indice: {lowest_idx} Blocos: {lowest_count}")
        return lowest_idx, lowest_count, lowest_sum

    def worst_bitwise(self, min_bytes):
        highest_idx, highest_count, highest_sum = 0, 0, 0
        return highest_idx, highest_count, highest_sum

    def visualize_alloc(self, p):
        ind = p.indexes
        idx = []

        for i, j in zip(*ind):
            idx.append((i, j))
            self.blocks_used[i][j] = 1

        for i in range(Heap.rows):
            for j in range(Heap.columns):
                start = ""
                end = ""
                if (i, j) in idx:
                    start = "\033[92m"
                    end = "\033[0m"
                elif self.blocks_used[i][j]:
                    start = "\033[91m"
                    end = "\033[0m"

                value = self.bytes_map[i][j]
                print(f"{start}{value}{end}", end=' ')
            print()
        print()

    def visualize(self):
        total_bytes = np.sum(self.bytes_map)
        n_allocated = np.sum(self.blocks_used == 1)
        bytes_allocated = np.sum(np.extract(self.blocks_used == 1, self.bytes_map))
        n_free = np.sum(self.blocks_used == 0)
        bytes_free = total_bytes - bytes_allocated
        print(f"[Memory] Total: {total_bytes} bytes | Allocated [{n_allocated}]: {bytes_allocated} bytes | Free [{n_free}]: {bytes_free} bytes")
        for i in range(Heap.rows):
            for j in range(Heap.columns):
                start = ""
                end = ""
                if self.blocks_used[i][j]:
                    start = "\033[91m"
                    end = "\033[0m"
                print(f"{start}{self.bytes_map[i][j]}{end}", end=' ')
            print()
        print()


heap = Heap()


def new(num_bytes, fit="best", show=False):
    idx, count, total_bytes = 0, 0, 0
    block = Block()

    if fit == "best":
        idx, count, total_bytes = heap.best_bitwise(num_bytes)
    elif fit == "first":
        idx, count, total_bytes = heap.first_bitwise(num_bytes)
    elif fit == "worst":
        idx, count, total_bytes = heap.worst_bitwise(num_bytes)
    else:
        raise NotImplementedError("Tipo não encontrado!")

    block.set_data(idx, count, total_bytes)
    if show:
        heap.visualize_alloc(block)
    return block


def delete(p, show=False):
    ind = p.indexes
    idx = list(zip(*ind))

    Heap.blocks_used[ind] = False

    print(f"[Dealloc] Bytes: {p.total_bytes} Indice: {p.index} Blocos: {p.count}")

    if show:
        for i in range(Heap.rows):
            for j in range(Heap.columns):
                start = ""
                end = ""
                if (i, j) in idx:
                    start = "\033[93m"
                    end = "\033[0m"
                elif Heap.blocks_used[i][j]:
                    start = "\033[91m"
                    end = "\033[0m"

                value = Heap.bytes_map[i][j]
                print(f"{start}{value}{end}", end=' ')
            print()
        print()

