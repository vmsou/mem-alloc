import numpy as np

from core import BadAlloc


def first_bitwise(used_arr, bytes_arr, min_bytes):
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

        if count < lowest_count and lowest_sum >= soma >= min_bytes:
            lowest_sum = soma
            lowest_count = count
            lowest_idx = i

        if soma >= min_bytes:
            count = 0
            soma = 0

    if lowest_idx is None:
        raise BadAlloc

    return lowest_idx, lowest_count


def main():
    rows = 5
    columns = 5

    blocks_used = np.random.choice([0, 1], (rows, columns), p=[0.8, 0.2])
    bytes_map = np.random.choice([30, 60], (rows, columns))

    teste = [[f"\033[91m{bytes_map[i][j]}\033[0m" for j in range(columns)] for i in range(rows)]
    visual = np.where(blocks_used == 1, teste, bytes_map)

    num_bytes = 45
    try:
        lowest_idx, lowest_count = first_bitwise(blocks_used, bytes_map, num_bytes)
    except BadAlloc:
        print("Erro de alocação")
    else:
        print(f"Bytes: {num_bytes} Indice: {lowest_idx} Blocos: {lowest_count}")

        print()
        for i in range(rows):
            for j in range(columns):
                print(visual[i][j], end=' ')
            print()


if __name__ == '__main__':
    main()
