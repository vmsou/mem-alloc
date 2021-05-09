import matplotlib.colors


class BadAlloc(Exception):
    pass


class Byte:
    def __init__(self):
        self.value = None

    def __repr__(self):
        return f"{hex(id(self))}: {self.value}"


colors = ["darkblue"] + list(matplotlib.colors.cnames) + ["yellow"]