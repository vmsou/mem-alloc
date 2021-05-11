from ctypes import *
from time import perf_counter
import numpy as np
from pathlib import Path

p = Path().absolute() / 'cpp' / 'libtest.so'
lib = cdll.LoadLibrary(str(p))

# Functions
first = lib.first
multi_first = lib.alloc

# Argument Types
first.argtypes = [c_void_p, c_int, c_int, c_int]
multi_first.argtypes = [c_void_p, c_int, c_int, c_int, c_int]

# Return types
first.restype = c_int


def main():
    rows = 5000
    columns = 5000
    blocks = 1
    indata = np.random.choice([True, False], (rows, columns), p=[0.2, 0.8]).astype(np.bool)

    print(indata.sum())
    print(indata.astype(c_int))

    start = perf_counter()
    # count = first(indata.ctypes.data, blocks, rows, columns)
    multi_first(indata.ctypes.data, 50000, blocks, rows, columns)
    print(f"{perf_counter() - start}s")

    print(indata.astype(c_int))
    print(indata.sum())


if __name__ == '__main__':
    main()
