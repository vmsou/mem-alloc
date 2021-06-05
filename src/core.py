affirmations = ('sim', 's', 'si', 'y', 'yes')
exits = ('exit', 'sair', 'cancelar', 'back', 'voltar')


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
