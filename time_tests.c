#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <max_cores> <output_base_name>\n", argv[0]);
        return 1;
    }

    int max_cores = atoi(argv[1]);
    const char* base_name = argv[2];

    // Compile the test program
    system("mpicc -o time_tests_single_run ee_algorithm.c time_tests_single_run.c -lm");

    // Test different core counts
    for (int cores = 1; cores <= max_cores; cores++) {
        char output_filename[100];
        sprintf(output_filename, "%s_%dcores.csv", base_name, cores);
        
        char command[200];
        sprintf(command, "mpiexec -n %d ./time_tests_single_run 100 %s", cores, output_filename);
        
        printf("Running with %d cores...\n", cores);
        system(command);
    }

    return 0;
}
