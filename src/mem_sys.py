import numpy as np

from core import size_confirm, BadAlloc

print("[ Simulador de Memória ]".center(60, "-"))


class Block:
    """Classe utilizada para representar o número de blocos alocados"""

    def __init__(self):
        self._index = None
        self.count = None
        self.total_bytes = None
        self.indexes = None

    def __repr__(self):
        return f"Block(Bytes: {self.total_bytes}, Indice: {self._index}, Blocos: {self.count})"

    def __invert__(self):
        delete(self, show=True)

    def set_data(self, index, count, total_bytes):
        """Inicializa seus valores na lista, obriga a preencher as informações de forma ordenada"""
        self.count = count
        self.index = index
        self.total_bytes = total_bytes

    @property
    def index(self):
        """A partir da propriedade privada retorna seu valor"""
        return self._index

    @index.setter
    def index(self, value):
        """Quando seu valor for atribuido; gaurda seus valores de modo contiguo em indices bidimensionais"""
        self.indexes = np.unravel_index(range(value, value + self.count), (heap.rows, heap.columns))
        self._index = value


class Heap:
    """Classe utilizada para representar a memória"""

    # Guardando valores de forma estática, para ser compartilhada com outras instâncias
    rows = size_confirm("Linhas")
    columns = size_confirm("Colunas")
    max_size = rows * columns
    blocks_used = np.random.choice([0, 1], (rows, columns), p=[0.8, 0.2])  # Guarda os blocos atualmente utilizados
    # blocks_used = np.zeros((rows, columns), dtype=bool)
    bytes_map = np.random.choice([10, 30], p=[0.8, 0.2],
                                 size=(rows, columns))  # Mapeia a quantidade de bytes em cada posição
    # bytes_map = np.repeat(30, max_size).reshape((rows, columns))
    b = range(max_size)  # Usado para iterar de forma eficiente em uma matriz flat
    allocated = []

    def first_bitwise(self, min_bytes):
        """Concluido
        Encontra o primeiro espaço em memória cujo o tamanho seja igual ou maior que o desejado
        """

        block = Block()
        lowest_sum = 1e6
        lowest_count = 1e6
        lowest_idx = None

        # Converte a matriz de forma que possa ser lida de forma contígua
        b_flat = self.blocks_used.flat
        bm_flat = self.bytes_map.flat
        count = 0
        soma = 0
        for i in self.b:
            if b_flat[i] == 0:
                count += 1
                soma += bm_flat[i]
            else:
                count = 0
                soma = 0
            if min_bytes <= soma <= lowest_sum and count <= lowest_count:
                lowest_idx = i
                break

        if lowest_idx is None:
            raise BadAlloc("Espaço Insuficiente")
        else:
            final_idx = lowest_idx - count + 1
            print(f"[First fit] Bytes: {soma} Indice: {final_idx} Blocos: {count}")
            block.set_data(final_idx, count, soma)
            self.blocks_used[block.indexes] = 1  # Atualiza a matriz para indicar as posições utilizadas

        return block

    def best_bitwise(self, min_bytes):
        """Concluido
        Encontra o melhor espaço que acomode o tamanho requisitado
        """
        block = Block()
        lowest_sum = 1e6
        lowest_count = 1e6
        lowest_idx = None

        # Converte a matriz de forma que possa ser lida de forma contígua
        b_flat = self.blocks_used.flat
        bm_flat = self.bytes_map.flat

        for i in self.b:
            idx = i
            count = 0
            soma = 0
            if b_flat[i] == 0:
                while not b_flat[idx]:
                    count += 1
                    soma += bm_flat[idx]
                    if min_bytes <= soma <= lowest_sum and count <= lowest_count:
                        lowest_idx = idx
                        lowest_count = count
                        lowest_sum = soma
                        break

                    idx += 1
                    if idx >= heap.max_size:
                        break

                    if b_flat[idx] == 1:
                        break

        if lowest_idx is None:
            raise BadAlloc("Espaço Insuficiente")

        final_idx = lowest_idx - lowest_count + 1
        print(f"[Best fit] Bytes: {lowest_sum} Indice: {final_idx} Blocos: {lowest_count}")
        block.set_data(final_idx, lowest_count, lowest_sum)
        self.blocks_used[block.indexes] = 1  # Atualiza a matriz para indicar as posições utilizadas

        return block

    def worst_bitwise(self, min_bytes):
        """Necessário a implementação
        Encontra a região na memória com maior espaço livre
        """
        highest_idx, highest_count, highest_sum = 0, 0, 0
        block = Block()
        block.set_data(highest_idx, highest_count, highest_sum)
        return block

    def free(self, p: Block):
        ind = p.indexes
        heap.blocks_used[ind] = False
        if p in heap.allocated:
            heap.allocated.remove(p)

    def visualize_alloc(self, p: Block):
        """Recebe um objeto de Classe Block e a partir de seus indices imprime suas posições alocadas"""
        ind = p.indexes
        idx = list(zip(*ind))

        for i in range(heap.rows):
            for j in range(heap.columns):
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

    def visualize_dealloc(self, p: Block):
        """Recebe um objeto de Classe Block e a partir de seus indices imprime suas posições dealocadas"""
        idx = list(zip(*p.indexes))

        for i in range(self.rows):
            for j in range(self.columns):
                start = ""
                end = ""
                if (i, j) in idx:
                    start = "\033[93m"
                    end = "\033[0m"
                elif heap.blocks_used[i][j]:
                    start = "\033[91m"
                    end = "\033[0m"

                value = heap.bytes_map[i][j]
                print(f"{start}{value}{end}", end=' ')
            print()
        print()

    def visualize(self):
        """Imprime a matriz com suas informações e condições."""
        total_bytes = np.sum(self.bytes_map)
        n_allocated = np.sum(self.blocks_used == 1)
        bytes_allocated = np.sum(np.extract(self.blocks_used == 1, self.bytes_map))
        n_free = np.sum(self.blocks_used == 0)
        bytes_free = total_bytes - bytes_allocated
        print(
            f"[Memory] Total: {total_bytes} bytes | Allocated [{n_allocated}]: {bytes_allocated} bytes | Free [{n_free}]: {bytes_free} bytes")
        for i in range(heap.rows):
            for j in range(heap.columns):
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
    """Recebe uma quantidade minima de bytes e cria um objeto Block baseado no tipo de fit"""
    block = None
    if fit == "best":
        block = heap.best_bitwise(num_bytes)
    elif fit == "first":
        block = heap.first_bitwise(num_bytes)
    elif fit == "worst":
        block = heap.worst_bitwise(num_bytes)
    else:
        raise NotImplementedError("Tipo não encontrado.", fit)

    if show:
        heap.visualize_alloc(block)
    heap.allocated.append(block)
    return block


def delete(p: Block, show=False):
    """Recebe um objeto de classe Block e a partir da lista de indices alocados; marca esses pontos como não usados"""
    heap.free(p)
    print(f"[Dealloc] Bytes: {p.total_bytes} Indice: {p.index} Blocos: {p.count}")
    if show:
        heap.visualize_dealloc(p)
