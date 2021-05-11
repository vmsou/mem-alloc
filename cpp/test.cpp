extern "C" {
    int first(const void* indatav, const int blocks, const int rows, const int columns) {
        bool* indata = (bool*) indatav;
        int count = 0;
        for (int i = 0; i < rows * columns; i++) {
           if (indata[i] == 0) {
               count++;
           } else count = 0;

           if (count >= blocks) {
               for (int j = blocks - 1; j > -1; j--) {
                    indata[i-j] = 1;
               }
               return count;
           }
        }
        return count;
    }

    void alloc(const void* indatav, const int n, const int blocks, const int rows, const int columns) {
        for (int i = 0; i < n; i++) {
            first(indatav, blocks, rows, columns);
        }
    }
}