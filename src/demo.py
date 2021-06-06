import numpy as np
from mem_sys import heap, new, delete, Block
from core import confirmar, BadAlloc, affirmations, coordinate_to_index


def alocar():
    min_bytes = confirmar("Quantidade minima de bytes: ", tipo=int, confirm=True, goto=menu)
    fit = confirmar("Tipo de fit: ", tipo=str, confirm=True, goto=menu, validation=lambda x: x.lower() in ("first", "best", "worst"))
    show = confirmar("Mostrar alocação: ", tipo=str, confirm=False, goto=menu)
    show = True if show in affirmations else False
    new(min_bytes, fit, show=show)


def desalocar():
    block = Block()
    choice = confirmar("Coordenada ou Indice? ", confirm=False, goto=menu, validation=lambda x: x.lower() in ("coordenada", "indice"))

    if choice.lower() == "coordenada":
        start_row = confirmar("Linha inicial: ", tipo=int, confirm=True, goto=menu, validation=lambda x: x < heap.rows)
        start_column = confirmar("Coluna inicial: ", tipo=int, confirm=True, goto=menu, validation=lambda x: x < heap.columns)
        index = coordinate_to_index([(start_row, start_column)], size=(heap.rows, heap.columns))[0]
    else:
        index = confirmar("Indice: ", tipo=int, confirm=True, goto=menu, validation=lambda x: x <= heap.max_size)

    n_blocks = confirmar("Quantidade de blocos: ", tipo=int, confirm=True, goto=menu)
    block.set_data(index, n_blocks, 0)
    delete(block, show=True)


def simulate():
    """Simula a ocupação de memória mostrado no TDE"""
    matriz = []
    matriz.extend((0, i) for i in (1, 3, 9, 11, 14, 15, 17, 18, 19))
    matriz.extend((1, i) for i in (0, 4, 6, 8, 9, 12, 15, 18))
    matriz.extend((2, i) for i in (0, 4, 6, 7, 8, 11, 14, 15, 18))
    matriz.extend((3, i) for i in (5, 10, 11, 12, 14, 18))
    matriz.extend((4, i) for i in (0, 1, 2, 4, 5, 9, 10, 13, 17, 18, 19))

    x, y = zip(*matriz)
    idx = np.array(x), np.array(y)

    heap.rows = 5
    heap.columns = 20
    heap.max_size = 5 * 20
    heap.b = range(5 * 20)
    heap.blocks_used = np.repeat(1, 5 * 20).reshape(5, 20)
    heap.blocks_used[idx] = 0
    heap.bytes_map = np.repeat(10, 5 * 20).reshape((5, 20))


def menu():
    action_dict = {1: alocar, 2: desalocar, 3: heap.visualize, 4: simulate}
    action_name = ["Alocar", "Desalocar", "Visualizar", "Simular"]
    while True:
        for n, name in enumerate(action_name, start=1):
            print(f"[{n}] {name}")
        try:
            action = confirmar("Ação: ", confirm=False, goto=menu)
            action_dict[int(action)]()
        except ValueError:
            print("Entrada Inválida! Tente Novamente.")
        except BadAlloc:
            print("Espaço insuficiente para Alocação.")


def main():
    menu()


if __name__ == '__main__':
    main()
