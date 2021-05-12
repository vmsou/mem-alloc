from ctypes import *
from time import perf_counter
import numpy as np
from pathlib import Path

p = Path().absolute() / 'cpp' / 'libtest.so'
lib = cdll.LoadLibrary(str(p))

# Functions
first = lib.first
multi_first = lib.multi_first

# Argument Types
first.argtypes = [c_void_p, c_int, c_int]
multi_first.argtypes = [c_void_p, c_int, c_int, c_int]

# Return types
first.restype = c_int


def main():
    rows = 5000
    columns = 5000
    blocks = 1
    indata = np.random.choice([True, False], (rows, columns), p=[0, 1])

    print(indata.sum())
    print(indata.astype(int))

    start = perf_counter()
    # count = first(indata.ctypes.data, blocks, rows * columns)
    multi_first(indata.ctypes.data, 20000, blocks, rows * columns)
    print(f"{perf_counter() - start}s")

    print(indata.astype(int))
    print(indata.sum())


if __name__ == '__main__':
    main()
