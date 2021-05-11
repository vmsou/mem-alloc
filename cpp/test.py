from ctypes import *
from time import perf_counter
import numpy as np

lib = cdll.LoadLibrary('./libtest.so')
first = lib.first
first.argtypes = [c_void_p, c_int, c_int, c_int]
first.restype = c_int

multi_first = lib.alloc
multi_first.argtypes = [c_void_p, c_int, c_int, c_int, c_int]

rows = 5000
columns = 5000
blocks = 1
indata = np.random.choice([True, False], (rows, columns), p=[0, 1]).astype(np.bool)

print(indata.sum())
print(indata.astype(c_int))

start = perf_counter()
# count = first(indata.ctypes.data, blocks, rows, columns)
multi_first(indata.ctypes.data, 10000, blocks, rows, columns)
print(f"{perf_counter() - start}s")

print(indata.astype(c_int))
print(indata.sum())
