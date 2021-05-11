from ctypes import *
from time import perf_counter
import numpy as np

lib = cdll.LoadLibrary(r'D:\Documentos\1.Projetos\Python\mem-alloc\cpp\libtest.so')
first = lib.first
first.argtypes = [c_void_p, c_int, c_int, c_int]
first.restype = c_int

multi_first = lib.alloc
multi_first.argtypes = [c_void_p, c_int, c_int, c_int, c_int]


def main():
    rows = 5000
    columns = 5000
    blocks = 1
    indata = np.random.choice([True, False], (rows, columns), p=[0.2, 0.8]).astype(np.bool)

    print(indata.sum())
    print(indata.astype(c_int))

    start = perf_counter()
    # count = first(indata.ctypes.data, blocks, rows, columns)
    multi_first(indata.ctypes.data, 10000, blocks, rows, columns)
    print(f"{perf_counter() - start}s")

    print(indata.astype(c_int))
    print(indata.sum())


if __name__ == '__main__':
    main()
