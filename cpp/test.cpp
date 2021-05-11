extern "C" {
    void first(const void* indatav, int blocks, int rows, int columns) {
        bool* indata = (bool*) indatav;
        int count = 0;
        for (int i = 0; i < rows * columns; i++) {
           if (indata[i] == false) {
               count++;
           } else count = 0;
           if (count >= blocks) {
               for (int j = i - count; j < blocks + i; j++) {
                   indata[j] = true;
               }
               break;
           }
        }
    }
}