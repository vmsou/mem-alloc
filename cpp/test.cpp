extern "C" {
    void first(const void* indatav, int blocks, int rows, int columns) {
        bool* indata = (bool*) indatav;
        int count = 0;
        for (int i = 0; i < rows * columns; i++) {
           if (!indata[i]) {
               count++;
           } else count = 0;
           if (count >= blocks) {
               for (int j = 0; j < blocks; j++) {
                   indata[j+i] = 1;
               }
               break;
           }
        }
    }
}