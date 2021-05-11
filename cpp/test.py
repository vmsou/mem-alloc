from ctypes import *
import numpy as np

rows = 3
columns = 3
blocks = 2
indata = np.random.choice([True, False], (rows, columns), p=[0.2, 0.8]).astype(c_bool)
lib = cdll.LoadLibrary('./libtest.so')
first = lib.first

print(indata.astype(c_int))
first.argtypes = [c_void_p, c_int, c_int, c_int]
first.restype = c_int
first(indata.ctypes.data, blocks, rows, columns)

print()
print(indata.astype(c_int))