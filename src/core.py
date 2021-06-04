class BadAlloc(Exception):
    pass


class Byte:
    def __init__(self):
        self.value = None

    def __repr__(self):
        return f"{hex(id(self))}: {self.value}"


def size_confirm(name):
    while True:
        try:
            value = int(input(name + ": "))
            if value > 0:
                return value
            raise ValueError
        except ValueError:
            print("Entrada inválida. Tente Novamente")
