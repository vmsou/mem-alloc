from ctypes import *
from time import perf_counter
import numpy as np
from pathlib import Path

p = Path().absolute() / 'cpp' / 'libtest.so'
lib = cdll.LoadLibrary(str(p))

# Functions
c_first = lib.first
c_multi_first = lib.multi_first

# Argument Types
c_first.argtypes = [c_void_p, c_int, c_int]
c_multi_first.argtypes = [c_void_p, c_int, c_int, c_int]

# Return types
c_first.restype = c_int


def first(data, blocks, n_elements):
    return c_first(data.ctypes.data, blocks, n_elements)


def multi_first(data, n, blocks, n_elements):
    c_multi_first(data.ctypes.data, n, blocks, n_elements)


def main():
    rows = 5000
    columns = 5000
    blocks = 1
    indata = np.random.choice([True, False], (rows, columns), p=[0, 1])

    print(indata.sum())
    print(indata.astype(int))

    start = perf_counter()
    # count = first(indata, blocks, rows * columns)
    multi_first(indata, 20000, blocks, rows * columns)
    print(f"{perf_counter() - start}s")

    print(indata.astype(int))
    print(indata.sum())


if __name__ == '__main__':
    main()
