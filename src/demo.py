from mem_sys import heap, new, delete, Block
from core import confirmar, BadAlloc, affirmations


def coordinate_to_index(coordinates: list, *, size: tuple):
    rows, columns = size
    indexes = []
    for row, column in coordinates:
        x = row * columns
        y = column % columns
        indexes.append(x + y)
    return indexes


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
        start_row = confirmar("Linha inicial: ", tipo=int, confirm=True, goto=menu)
        start_column = confirmar("Coluna inicial: ", tipo=int, confirm=True, goto=menu)
        index = coordinate_to_index([(start_row, start_column)], size=(heap.rows, heap.columns))[0]
    else:
        index = confirmar("Indice: ", tipo=int, confirm=True, goto=menu, validation=lambda x: x <= heap.max_size)

    n_blocks = confirmar("Quantidade de blocos: ", tipo=int, confirm=True, goto=menu)
    block.set_data(index, n_blocks, 0)
    delete(block, show=True)


def menu():
    action_dict = {1: alocar, 2: desalocar, 3: heap.visualize}
    action_name = ["Alocar", "Desalocar", "Visualizar"]
    while True:
        for n, name in enumerate(action_name, start=1):
            print(f"[{n}] {name}")

        try:
            action = confirmar("Ação: ", confirm=False, goto=menu)
            action_dict[int(action)]()
        except IndexError or ValueError:
            print("Entrada Inválida! Tente Novamente.")
        except BadAlloc:
            print("Espaço insuficiente para Alocação.")


def main():
    """heap.visualize()
    t = new(90, 'best', show=True)
    delete(t, show=True)
    heap.visualize()"""

    menu()


if __name__ == '__main__':
    main()
