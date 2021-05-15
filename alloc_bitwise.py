import numpy as np

from core import BadAlloc


def first_bitwise(used_arr, bytes_arr, min_bytes):
    b = range(np.prod(used_arr.shape))
    b_flat = used_arr.flat
    bm_flat = bytes_arr.flat

    count = 0
    soma = 0

    for i in b:
        if not b_flat[i]:
            count += 1
            soma += bm_flat[i]
        else:
            count = 0
            soma = 0

        if soma >= min_bytes:
            print()
            print(f"[First fit] Bytes: {soma} Indice: {i} Blocos: {count}")
            print()
            return i, count

    raise BadAlloc("Not enough space")


def best_bitwise(used_arr, bytes_arr, min_bytes):
    lowest_sum = 1e6
    lowest_count = 1e6
    lowest_idx = None

    b_flat = used_arr.flat
    bm_flat = bytes_arr.flat
    b = range(np.prod(used_arr.shape))

    count = 0
    soma = 0

    for i in b:
        if not b_flat[i]:
            count += 1
            soma += bm_flat[i]
        else:
            count = 0
            soma = 0

        if count <= lowest_count and lowest_sum > soma >= min_bytes:
            lowest_sum = soma
            lowest_count = count
            lowest_idx = i

            count = 0
            soma = 0

        if soma >= min_bytes:
            count = 0
            soma = 0

    if lowest_idx is None:
        raise BadAlloc

    print()
    print(f"[Best fit] Bytes: {soma} Indice: {lowest_idx} Blocos: {lowest_count}")
    print()
    return lowest_idx, lowest_count


def worst_bitwise(used_arr, bytes_arr, min_bytes):
    highest_sum = 0
    highest_count = 0
    highest_idx = None

    b_flat = used_arr.flat
    bm_flat = bytes_arr.flat
    b = range(np.prod(used_arr.shape))

    count = 0
    soma = 0

    for i in b:
        if not b_flat[i]:
            count += 1
            soma += bm_flat[i]
        else:
            count = 0
            soma = 0

        if count >= highest_count and soma >= highest_sum:
            highest_count = count
            highest_sum = soma
            highest_idx = i

    if highest_idx is None:
        raise BadAlloc

    print()
    print(f"[Worst fit] Bytes: {soma} Indice: {highest_idx} Blocos: {highest_count}")
    print()
    return highest_idx, highest_count


def visualize_alloc(used_arr, bytes_arr, idx, blocks):
    rows, columns = used_arr.shape

    teste = [[f"\033[91m{bytes_arr[i][j]}\033[0m" if used_arr[i][j] else f"\033[92m{bytes_arr[i][j]}\033[0m" for j in
              range(columns)] for i in range(rows)]
    visual = np.where(used_arr == 1, teste, bytes_arr)

    ind = np.unravel_index(range(idx, idx - blocks, -1), (rows, columns))

    for i, j in zip(ind[0], ind[1]):
        visual[i][j] = teste[i][j]
        used_arr[i][j] = 1

    for i in range(rows):
        for j in range(columns):
            print(visual[i][j], end=' ')
        print()


def simulate(used_arr):
    vazio = []
    vazio.extend((0, i) for i in (1, 3, 9, 11, 14, 15, 17, 18, 19))
    vazio.extend((1, i) for i in (0, 4, 6, 8, 9, 12, 15, 18))
    vazio.extend((2, i) for i in (0, 4, 6, 7, 8, 11, 14, 15, 18))
    vazio.extend((3, i) for i in (5, 10, 11, 12, 14, 18))
    vazio.extend((4, i) for i in (0, 1, 2, 4, 5, 9, 10, 13, 17, 18, 19))
    for i in range(5):
        for j in range(20):
            if (i, j) not in vazio:
                used_arr[i][j] = 1


def new(num_bytes, blocks_used, bytes_map, fit="best", show=False):
    idx, count = 0, 0
    if fit == "best":
        idx, count = best_bitwise(blocks_used, bytes_map, num_bytes)
    elif fit == "first":
        idx, count = first_bitwise(blocks_used, bytes_map, num_bytes)
    elif fit == "worst":
        idx, count = worst_bitwise(blocks_used, bytes_map, num_bytes)
    if show:
        visualize_alloc(blocks_used, bytes_map, idx, count)


def main():
    rows = 5
    columns = 20

    # blocks_used = np.random.choice([0, 1], (rows, columns), p=[0.8, 0.2])
    blocks_used = np.zeros((rows, columns))
    # bytes_map = np.random.choice([30, 60], (rows, columns))
    bytes_map = np.repeat(30, rows*columns).reshape((rows, columns))

    simulate(blocks_used)

    # new(90, blocks_used, bytes_map, fit="first", show=True)
    # new(90, blocks_used, bytes_map, fit="best", show=True)
    new(60, blocks_used, bytes_map, fit="worst", show=True)


if __name__ == '__main__':
    main()
