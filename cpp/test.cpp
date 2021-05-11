extern "C" {
    int first(void* indatav, int blocks, int rows, int columns) {
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

    void alloc(void* indatav, int n, int blocks, int rows, int columns) {
        for (int i = 0; i < n; i++) {
            first(indatav, blocks, rows, columns);
        }
    }
}