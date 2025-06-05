#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // Added for sqrt()
#include <mpi.h>
#include "ee_algorithm.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: %s <num_tests> <output_filename>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int num_tests = atoi(argv[1]);
    const char* output_filename = argv[2];

    // Algorithm parameters (fixed for testing)
    int num_points = 100;
    Point3D* target_shape = create_sphere_points(num_points);
    
    // Only rank 0 handles file output
    FILE* output_file = NULL;
    if (rank == 0) {
        output_file = fopen(output_filename, "w");
        if (!output_file) {
            fprintf(stderr, "Error opening output file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(output_file, "test_number,evaluation_time_seconds,cores_used\n");
    }

    for (int test = 0; test < num_tests; test++) {
        double total_eval_time = 0;
        
        // Run the algorithm
        Individual best = run_ee_algorithm(
            target_shape, num_points,
            num_points, 500, 100, 20, 30,
            1.0/sqrt(num_points), 1e-5, -1.0, 1.0,
            &total_eval_time, 0
        );

        if (rank == 0) {
            fprintf(output_file, "%d,%.6f,%d\n", 
                   test + 1, 
                   total_eval_time,
                   size);
            fflush(output_file);
        }

        free_individual(&best);
    }

    free(target_shape);
    if (rank == 0) {
        fclose(output_file);
    }
    MPI_Finalize();
    return 0;
}
