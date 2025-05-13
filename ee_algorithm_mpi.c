#include "ee_algorithm_mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <stdbool.h>
#include <mpi.h>

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

/* Comparison function for qsort */
static int compare_individuals(const void* a, const void* b) {
    const Individual* ia = (const Individual*)a;
    const Individual* ib = (const Individual*)b;
    if (ia->fitness < ib->fitness) return 1;
    if (ia->fitness > ib->fitness) return -1;
    return 0;
}

/* Initialize Individual with proper memory allocation */
void init_individual(Individual* ind, int num_points) {
    ind->points = (Point3D*)malloc(num_points * sizeof(Point3D));
    if (!ind->points) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    ind->num_points = num_points;
    ind->fitness = -1.0;
    ind->sigma = 0.1; // Default value
}

/* Safe memory deallocation */
void free_individual(Individual* ind) {
    if (ind && ind->points) {
        free(ind->points);
        ind->points = NULL;
    }
}

/* Helper function to create random points */
Point3D* create_points(double inf, double sup, int n) {
    Point3D* points = (Point3D*)malloc(n * sizeof(Point3D));
    if (!points) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    for (int i = 0; i < n; i++) {
        points[i].x = ((double)rand() / RAND_MAX) * (sup - inf) + inf;
        points[i].y = ((double)rand() / RAND_MAX) * (sup - inf) + inf;
        points[i].z = ((double)rand() / RAND_MAX) * (sup - inf) + inf;
    }
    return points;
}

/* Distance calculation */
double distance(Point3D a, Point3D b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

/* Simplified convex hull volume calculation */
double compute_convex_hull_volume(Point3D* points, int num_points) {
    if (num_points < 4) return 0.0;
    
    double max_dist = 0;
    for (int i = 0; i < num_points; i++) {
        for (int j = i+1; j < num_points; j++) {
            double d = distance(points[i], points[j]);
            if (d > max_dist) max_dist = d;
        }
    }
    return (max_dist * max_dist * max_dist) / 6.0;
}

/* 3D IoU calculation */
double compute_3d_iou(Point3D* points1, int num_points1, Point3D* points2, int num_points2) {
    double vol1 = compute_convex_hull_volume(points1, num_points1);
    double vol2 = compute_convex_hull_volume(points2, num_points2);
    
    if (vol1 == 0 || vol2 == 0) return 0.0;
    
    int intersection_count = 0;
    double threshold = 0.1;
    
    for (int i = 0; i < num_points1; i++) {
        for (int j = 0; j < num_points2; j++) {
            if (distance(points1[i], points2[j]) < threshold) {
                intersection_count++;
                break;
            }
        }
    }
    
    double intersection_vol = (intersection_count / (double)num_points1) * MIN(vol1, vol2);
    double union_vol = vol1 + vol2 - intersection_vol;
    
    return (union_vol > 0) ? (intersection_vol / union_vol) : 0.0;
}

/* Parallel fitness evaluation */
void parallel_evaluate_fitness(Individual* population, int pop_size, 
                             Point3D* target_shape, int target_size,
                             MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // Divide work among processes
    int chunk_size = pop_size / size;
    int remainder = pop_size % size;
    int start = rank * chunk_size + (rank < remainder ? rank : remainder);
    int end = start + chunk_size + (rank < remainder ? 1 : 0);
    
    // Evaluate assigned individuals
    for (int i = start; i < end && i < pop_size; i++) {
        population[i].fitness = compute_3d_iou(
            population[i].points, population[i].num_points,
            target_shape, target_size
        );
    }
    
    // Gather all fitness values to root
    double* all_fitness = NULL;
    if (rank == 0) {
        all_fitness = (double*)malloc(pop_size * sizeof(double));
        if (!all_fitness) {
            fprintf(stderr, "Memory allocation failed\n");
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }
    
    // Prepare gather operation
    int* recvcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    if (!recvcounts || !displs) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }
    
    for (int i = 0; i < size; i++) {
        recvcounts[i] = pop_size / size + (i < (pop_size % size) ? 1 : 0);
        displs[i] = i ? displs[i-1] + recvcounts[i-1] : 0;
    }
    
    // Perform gather
    MPI_Gatherv(
        &(population[start].fitness), end - start, MPI_DOUBLE,
        all_fitness, recvcounts, displs, MPI_DOUBLE,
        0, comm
    );
    
    // Broadcast updated fitness values
    if (rank == 0) {
        for (int i = 0; i < pop_size; i++) {
            population[i].fitness = all_fitness[i];
        }
    }
    MPI_Bcast(&(population[0].fitness), pop_size, MPI_DOUBLE, 0, comm);
    
    // Cleanup
    free(recvcounts);
    free(displs);
    if (rank == 0) free(all_fitness);
}

/* Safe population broadcast */
void broadcast_population(Individual* population, int size, int num_points, int root, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    for (int i = 0; i < size; i++) {
        if (rank != root) {
            init_individual(&population[i], num_points);
        }
        MPI_Bcast(population[i].points, num_points * 3, MPI_DOUBLE, root, comm);
        MPI_Bcast(&population[i].sigma, 1, MPI_DOUBLE, root, comm);
        MPI_Bcast(&population[i].fitness, 1, MPI_DOUBLE, root, comm);
    }
}

/* Main parallel algorithm */
Individual run_ee_algorithm_mpi(
    Point3D* target_shape, int target_size,
    int num_points_in_individual,
    int generations,
    int population_size,
    int mu_size,
    int lambda_size,
    double tau,
    double epsilon0,
    double box_min,
    double box_max,
    double* execution_time,
    MPI_Comm comm
) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    clock_t start_time = clock();
    
    // Initialize population
    Individual* population = NULL;
    if (rank == 0) {
        population = (Individual*)malloc(population_size * sizeof(Individual));
        if (!population) {
            fprintf(stderr, "Memory allocation failed\n");
            MPI_Abort(comm, EXIT_FAILURE);
        }
        
        for (int i = 0; i < population_size; i++) {
            init_individual(&population[i], num_points_in_individual);
            population[i].sigma = ((double)rand() / RAND_MAX) * 0.15 + 0.05;
            population[i].points = create_points(box_min, box_max, num_points_in_individual);
        }
    } else {
        population = (Individual*)malloc(population_size * sizeof(Individual));
        if (!population) {
            fprintf(stderr, "Memory allocation failed\n");
            MPI_Abort(comm, EXIT_FAILURE);
        }
        memset(population, 0, population_size * sizeof(Individual));
    }
    
    // Broadcast initial population
    broadcast_population(population, population_size, num_points_in_individual, 0, comm);
    
    // Main evolutionary loop
    for (int gen = 0; gen < generations; gen++) {
        // Parallel fitness evaluation
        parallel_evaluate_fitness(population, population_size, target_shape, target_size, comm);
        
        // Sort population by fitness (root only)
        if (rank == 0) {
            qsort(population, population_size, sizeof(Individual), compare_individuals);
        }
        
        // Broadcast sorted population
        broadcast_population(population, population_size, num_points_in_individual, 0, comm);
        
        // Create offspring
        Individual* offspring = (Individual*)malloc(lambda_size * sizeof(Individual));
        if (!offspring) {
            fprintf(stderr, "Memory allocation failed\n");
            MPI_Abort(comm, EXIT_FAILURE);
        }
        
        for (int i = 0; i < lambda_size; i++) {
            init_individual(&offspring[i], num_points_in_individual);
            
            // Select parents
            int p1 = rand() % mu_size;
            int p2 = rand() % mu_size;
            
            // Crossover
            for (int j = 0; j < num_points_in_individual; j++) {
                if (((double)rand() / RAND_MAX) < 0.5) {
                    offspring[i].points[j] = population[p1].points[j];
                } else {
                    offspring[i].points[j] = population[p2].points[j];
                }
                
                // Mutation
                offspring[i].points[j].x += population[p1].sigma * ((double)rand() / RAND_MAX - 0.5) * 2.0;
                offspring[i].points[j].y += population[p1].sigma * ((double)rand() / RAND_MAX - 0.5) * 2.0;
                offspring[i].points[j].z += population[p1].sigma * ((double)rand() / RAND_MAX - 0.5) * 2.0;
                
                // Boundary handling
                offspring[i].points[j].x = MIN(MAX(offspring[i].points[j].x, box_min), box_max);
                offspring[i].points[j].y = MIN(MAX(offspring[i].points[j].y, box_min), box_max);
                offspring[i].points[j].z = MIN(MAX(offspring[i].points[j].z, box_min), box_max);
            }
            
            // Update sigma
            double alpha = (double)rand() / RAND_MAX;
            offspring[i].sigma = alpha * population[p1].sigma + (1 - alpha) * population[p2].sigma;
        }
        
        // Evaluate offspring
        parallel_evaluate_fitness(offspring, lambda_size, target_shape, target_size, comm);
        
        // Combine and select (root only)
        if (rank == 0) {
            Individual* combined = (Individual*)malloc((mu_size + lambda_size) * sizeof(Individual));
            if (!combined) {
                fprintf(stderr, "Memory allocation failed\n");
                MPI_Abort(comm, EXIT_FAILURE);
            }
            
            // Combine parents and offspring
            for (int i = 0; i < mu_size; i++) {
                combined[i] = population[i];
            }
            for (int i = 0; i < lambda_size; i++) {
                combined[mu_size + i] = offspring[i];
            }
            
            // Sort combined population
            qsort(combined, mu_size + lambda_size, sizeof(Individual), compare_individuals);
            
            // Select top individuals
            for (int i = 0; i < mu_size; i++) {
                population[i] = combined[i];
            }
            
            free(combined);
        }
        
        // Cleanup offspring
        for (int i = 0; i < lambda_size; i++) {
            free_individual(&offspring[i]);
        }
        free(offspring);
        
        // Broadcast new population
        broadcast_population(population, mu_size, num_points_in_individual, 0, comm);
    }
    
    // Final evaluation
    parallel_evaluate_fitness(population, population_size, target_shape, target_size, comm);
    
    // Get best individual
    Individual best;
    if (rank == 0) {
        best = population[0];
        *execution_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        
        // Make a deep copy of the best individual
        init_individual(&best, best.num_points);
        memcpy(best.points, population[0].points, best.num_points * sizeof(Point3D));
        best.sigma = population[0].sigma;
        best.fitness = population[0].fitness;
    }
    
    // Cleanup
    for (int i = 0; i < population_size; i++) {
        free_individual(&population[i]);
    }
    free(population);
    
    return best;
}

/* Create points on a unit sphere */
Point3D* create_sphere_points(int num_points) {
    Point3D* points = (Point3D*)malloc(num_points * sizeof(Point3D));
    if (!points) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    for (int i = 0; i < num_points; i++) {
        double phi = ((double)rand() / RAND_MAX) * 2 * M_PI;
        double costheta = ((double)rand() / RAND_MAX) * 2 - 1;
        double theta = acos(costheta);
        
        points[i].x = sin(theta) * cos(phi);
        points[i].y = sin(theta) * sin(phi);
        points[i].z = cos(theta);
    }
    return points;
}

/* Save individual to file */
void save_individual_to_file(
    const Individual* ind, const char* filename, 
    int G, double tau, double eps0, 
    int lambda, int mu, double box_min, double box_max,
    double execution_time
) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
        return;
    }

    fprintf(file, "Generations %d\n", G);
    fprintf(file, "Tau %f\n", tau);
    fprintf(file, "Epsilon0 %f\n", eps0);
    fprintf(file, "Lambda %d\n", lambda);
    fprintf(file, "Mu %d\n", mu);
    fprintf(file, "BoxMin %f\n", box_min);
    fprintf(file, "BoxMax %f\n", box_max);
    fprintf(file, "Runtime %.2f\n", execution_time);
    fprintf(file, "Fitness %f\n", ind->fitness);
    fprintf(file, "Sigma %f\n", ind->sigma);
    fprintf(file, "Points %d\n", ind->num_points);
    
    fprintf(file, "\nPoints (x,y,z):\n");
    for (int i = 0; i < ind->num_points; i++) {
        fprintf(file, "%f %f %f\n", ind->points[i].x, ind->points[i].y, ind->points[i].z);
    }
    fclose(file);
}
