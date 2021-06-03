from mem_sys import heap, new, delete


def main():
    heap.visualize()
    t = new(100, show=True)
    heap.visualize()


if __name__ == '__main__':
    main()
