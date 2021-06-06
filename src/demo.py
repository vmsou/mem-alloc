from mem_sys import heap, new, delete
from core import confirmar, BadAlloc, affirmations
from validation import validate_fit


def alocar():
    min_bytes = confirmar("Quantidade minima de bytes: ", tipo=int, confirm=True, goto=menu)
    fit = confirmar("Tipo de fit: ", tipo=str, confirm=True, validation=validate_fit, goto=menu)
    show = confirmar("Mostrar alocação: ", tipo=str, confirm=True, goto=menu)
    show = True if show in affirmations else False
    new(min_bytes, fit, show=show)


def desalocar():
    start_row = confirmar("Linha inicial: ", tipo=int, confirm=True, goto=menu)
    start_column = confirmar("Coluna inicial: ", tipo=int, confirm=True, goto=menu)
    start_column = confirmar("Quantidade de blocos: ", tipo=int, confirm=True, goto=menu)


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
