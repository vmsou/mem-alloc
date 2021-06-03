from mem_sys import heap, new, delete


def main():
    heap.visualize()
    t = new(45, 'test', show=True)
    delete(t, show=True)
    heap.visualize()


if __name__ == '__main__':
    main()
