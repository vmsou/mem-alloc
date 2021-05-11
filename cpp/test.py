from ctypes import *
import numpy as np

rows = 3
columns = 3
blocks = 2
indata = np.random.choice([True, False], (rows, columns)).astype(c_bool)
lib = cdll.LoadLibrary('./libtest.so')
first = lib.first

print(indata)
first.argtypes = [c_void_p, c_int, c_int, c_int]
first.restype = c_int
first(indata.ctypes.data, 1, rows, columns)

print()
print(indata)