#include <stdio.h>
#include <stdlib.h>
#include <math.h>          // Added for sqrt()
#include <time.h>  // For clock()
#include "ee_algorithm.h"  // Added for create_sphere_points()
#include <mpi.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Test parameters
    int test_sizes[] = {0, 10, 20, 40, 60, 100, 200, 300, 400, 500};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    // Fixed parameters
    int num_points = 100;
    int generations = 500;  // Reduced for faster testing
    double box_min = -1, box_max = 1;
    
    printf("population_size, runtime (s), fitness\n");
    
    for (int i = 0; i < num_tests; i++) {
        int population_size = test_sizes[i];
        if (population_size == 0) {
            printf("0, INVALID, INVALID\n");
            continue;
        }
        
        int mu_size = population_size / 5;
        int lambda_size = mu_size * 1.5;
        
        double tau = 1/sqrt(num_points);
        double epsilon0 = 1e-5;
        double execution_time = 0;
        
        Point3D* target_shape = create_sphere_points(num_points);
        
        clock_t start = clock();
        Individual best = run_ee_algorithm(
            target_shape, num_points,
            num_points,
            generations,
            population_size,
            mu_size,
            lambda_size,
            tau,
            epsilon0,
            box_min,
            box_max,
            &execution_time
        );
        clock_t end = clock();
        double runtime = (double)(end - start) / CLOCKS_PER_SEC;

    // Save the solution
    char filename[50];
    sprintf(filename, "parallel4solution_%d.txt", population_size);
    save_individual_to_file(
        &best, 
        filename,
        generations,
        tau,
        epsilon0,
        lambda_size,
        mu_size,
        box_min,
        box_max,
        runtime
    );
        
        printf("%d, %.2f, %.4f\n", population_size, runtime, best.fitness);
        
        free(target_shape);
        free_individual(&best);
    }
    
    MPI_Finalize();
    return 0;
}