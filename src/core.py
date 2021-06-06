affirmations = ('sim', 's', 'si', 'y', 'yes')
exits = ('exit', 'sair', 'cancelar', 'back', 'voltar')
colors = {"purple": '\033[95m', "blue": '\033[94m', "ciano": '\033[96m',
         "green": '\033[92m', "yellow": '\033[93m', "red": '\033[91m'}


class BadAlloc(Exception):
    pass


class Byte:
    def __init__(self):
        self.value = None

    def __repr__(self):
        return f"{hex(id(self))}: {self.value}"


class Logger:
    enabled = True

    def __init__(self, name):
        self.name = name

    def log(self, message, color=None, bold=False):
        strong = ''
        start = ''
        end = ''
        if bold:
            strong = '\033[1m'
            end = '\033[0m'
        if color in colors.keys():
            end = "\033[0m"
            start = colors[color]
        if self.enabled:
            print(f"[{self.name}]{strong}{start} {message}{end}")


def size_confirm(name):
    while True:
        try:
            value = int(input(name + ": "))
            if value > 0:
                return value
            raise ValueError
        except ValueError:
            print("Entrada inválida. Tente Novamente")


def confirmar(mensagem, tipo=str, confirm=True, validation=None, goto=None):
    while True:
        try:
            valor = input(mensagem)
            if valor.lower() in exits:
                if not goto:
                    return False
                goto()

            valor = tipo(valor)
            if validation is not None:
                if not validation(valor):
                    raise ValueError
        except ValueError:
            print("Entrada Inválida. Tente Novamente.")
        else:
            if confirm:
                if input("Confirmar (s/n): ").lower() in affirmations:
                    break
            else:
                break
    return valor


def coordinate_to_index(coordinates: list, *, size: tuple):
    rows, columns = size
    indexes = []
    for row, column in coordinates:
        x = row * columns
        y = column % columns
        indexes.append(x + y)
    return indexes