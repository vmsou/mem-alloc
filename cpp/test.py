from ctypes import *
import numpy as np

rows = 5000
columns = 5000
blocks = 1_000_000
indata = np.random.choice([True, False], (rows, columns), p=[0, 1]).astype(np.bool)
lib = cdll.LoadLibrary('./libtest.so')
first = lib.first

print(indata.sum())
print(indata.astype(int))
first.argtypes = [c_void_p, c_int, c_int, c_int]
first(indata.ctypes.data, blocks, rows, columns)

print()
print(indata.astype(np.int))
print(indata.sum())