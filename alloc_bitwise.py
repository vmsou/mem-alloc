import numpy as np

rows = 5
columns = 5

b = range(rows * columns)
bits_map = np.random.choice([50, 100], (rows, columns))
blocks_used = np.random.choice([0, 1], (rows, columns), p=[0.8, 0.2])


min_bits = 300

lowest_sum = 1e6
lowest_count = 1e6
lowest_idx = 0

b_flat = blocks_used.flat
bm_flat = bits_map.flat
indices = []

print(blocks_used.astype(int))
print(bits_map)

count = 0
soma = 0
j = 0

for i in b:
    if not b_flat[i]:
        count += 1
        soma += bm_flat[i]
    else:
        count = 0
        soma = 0

    if count < lowest_count and lowest_sum >= soma >= min_bits:
        lowest_sum = soma
        lowest_count = count
        lowest_idx = i

    if soma >= min_bits:
        count = 0
        soma = 0


print(lowest_idx, lowest_sum, lowest_count)