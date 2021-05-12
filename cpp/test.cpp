extern "C" {
    int first(void* indatav, int blocks, int n_elements) {
        bool* indata = (bool*) indatav;
        int count = 0;
        for (int i = 0; i < n_elements; i++) {
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

    void multi_first(void* indatav, int n, int blocks, int n_elements) {
        for (int i = 0; i < n; i++) {
            first(indatav, blocks, n_elements);
        }
    }
}