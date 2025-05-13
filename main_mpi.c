#include "ee_algorithm_mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    srand(time(NULL) + rank);  // Different seed for each process
    
    // Parameters
    int num_points = 100;
    int generations = 100;
    int population_size = 500;
    int mu_size = population_size / 5;
    int lambda_size = mu_size * 1.5;
    double tau = 1/sqrt(num_points);
    double epsilon0 = 1e-5;
    double box_min = -1;
    double box_max = 1;
    double execution_time = 0;
    
    // Create target shape (root only)
    Point3D* target_shape = NULL;
    if (rank == 0) {
        target_shape = create_sphere_points(num_points);
    }
    
    // Broadcast target shape to all processes
    if (rank != 0) {
        target_shape = (Point3D*)malloc(num_points * sizeof(Point3D));
    }
    MPI_Bcast(target_shape, num_points * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Run the algorithm
    Individual best = run_ee_algorithm_mpi(
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
        &execution_time,
        MPI_COMM_WORLD
    );
    
    // Save results (root only)
    if (rank == 0) {
        printf("Best fitness: %f\n", best.fitness);
        printf("Execution time: %.2f seconds\n", execution_time);
        
        save_individual_to_file(&best, "best_individual_mpi.txt", 
                              generations, tau, epsilon0, 
                              lambda_size, mu_size, box_min, box_max,
                              execution_time);
        
        printf("Best individual saved to best_individual_mpi.txt\n");
    }
    
    // Cleanup
    free(target_shape);
    if (rank == 0) {
        free_individual(&best);
    }
    
    MPI_Finalize();
    return 0;
}
