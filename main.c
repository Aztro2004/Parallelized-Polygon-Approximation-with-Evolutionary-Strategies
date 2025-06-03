#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include "ee_algorithm.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Parameters
    int num_points = 500;
    int generations = num_points * 3;
    int population_size = 1000;
    int save_interval = 50;
    double box_min = -1, box_max = 1;
    
    int mu_size = population_size / 3;
    int lambda_size = mu_size * 3.0;
    
    double tau = 1/sqrt(num_points);
    double epsilon0 = 1e-5;
    double total_eval_time = 0;

    Point3D* target_shape = create_sphere_points(num_points);
    
    if (rank == 0) {
        printf("Running evolution with:\n");
        printf("  Population size: %d\n", population_size);
        printf("  Generations: %d\n", generations);
        printf("  Points per individual: %d\n", num_points);
    }
    
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
        &total_eval_time,
        save_interval
    );
    
    if (rank == 0) {
        printf("\nResults:\n");
        printf("Total parallel evaluation time: %.4f seconds\n", total_eval_time);
        printf("Average per generation: %.4f seconds\n", total_eval_time/generations);
        printf("Final fitness: %.4f\n", best.fitness);
        
        save_individual_to_file(
            &best, 
            "final_solution.txt",
            generations,
            tau,
            epsilon0,
            lambda_size,
            mu_size,
            box_min,
            box_max,
            total_eval_time
        );
    }
    
    free(target_shape);
    free_individual(&best);
    
    MPI_Finalize();
    return 0;
}