import matplotlib.colors


class BadAlloc(Exception):
    pass


class Byte:
    def __init__(self):
        self.value = None

    def __repr__(self):
        return f"{hex(id(self))}: {self.value}"


def confirm(msg="Value: "):
    n = 0
    while n <= 0:
        try:
            n = int(input(msg))
        except ValueError:
            n = 0
        if n <= 0:
            print("Inválido. Tente Novamente")
    return n


colors = ["darkblue"] + list(matplotlib.colors.cnames) + ["yellow"]