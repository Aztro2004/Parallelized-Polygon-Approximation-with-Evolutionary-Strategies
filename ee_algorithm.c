#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <float.h>
#include <string.h>
#include <mpi.h>
#include "ee_algorithm.h"

#define PI 3.14159265358979323846
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

static int compare_individuals(const void* a, const void* b) {
    const Individual* ia = a, *ib = b;
    return (ia->fitness < ib->fitness) ? 1 : ((ia->fitness > ib->fitness) ? -1 : 0);
}

static Point3D* create_points(double min, double max, int n) {
    Point3D* pts = malloc(n * sizeof(Point3D));
    if (!pts) return NULL;
    for (int i = 0; i < n; i++) {
        pts[i].x = ((double)rand()/RAND_MAX)*(max-min)+min;
        pts[i].y = ((double)rand()/RAND_MAX)*(max-min)+min;
        pts[i].z = ((double)rand()/RAND_MAX)*(max-min)+min;
    }
    return pts;
}

static double point_distance(Point3D a, Point3D b) {
    double dx = a.x-b.x, dy = a.y-b.y, dz = a.z-b.z;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

double compute_convex_hull_volume(Point3D* points, int num_points) {
    if (num_points < 4) return 0.0;
    
    double max_dist = 0;
    for (int i = 0; i < num_points; i++) {
        for (int j = i+1; j < num_points; j++) {
            double d = point_distance(points[i], points[j]);
            if (d > max_dist) max_dist = d;
        }
    }
    return (max_dist * max_dist * max_dist) / 6.0;
}

double compute_3d_iou(Point3D* points1, int num_points1, Point3D* points2, int num_points2) {
    double vol1 = compute_convex_hull_volume(points1, num_points1);
    double vol2 = compute_convex_hull_volume(points2, num_points2);
    
    if (vol1 == 0 || vol2 == 0) {
        return 0.0;
    }
    
    int intersection_count = 0;
    double threshold = 0.2;
    
    for (int i = 0; i < num_points1; i++) {
        for (int j = 0; j < num_points2; j++) {
            if (point_distance(points1[i], points2[j]) < threshold) {
                intersection_count++;
                break;
            }
        }
    }
    
    double intersection_vol = (intersection_count / (double)num_points1) * MIN(vol1, vol2);
    double union_vol = vol1 + vol2 - intersection_vol;
    
    return (union_vol > 0) ? (intersection_vol / union_vol) : 0.0;
}

Individual create_individual(double inf, double sup, int n_points, Point3D* target_shape, int target_size) {
    Individual ind;
    ind.points = create_points(inf, sup, n_points);
    ind.num_points = n_points;
    ind.sigma = ((double)rand() / RAND_MAX) * 0.15 + 0.05;
    ind.fitness = compute_3d_iou(ind.points, ind.num_points, target_shape, target_size);
    return ind;
}

Individual copy_individual(const Individual* src) {
    Individual dest;
    dest.num_points = src->num_points;
    dest.sigma = src->sigma;
    dest.fitness = src->fitness;
    dest.points = (Point3D*)malloc(dest.num_points * sizeof(Point3D));
    if (!dest.points) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    memcpy(dest.points, src->points, dest.num_points * sizeof(Point3D));
    return dest;
}

void free_individual(Individual* ind) {
    if (ind && ind->points) {
        free(ind->points);
        ind->points = NULL;
    }
}

void crossover(const Individual* ind1, const Individual* ind2, Individual* child) {
    child->num_points = ind1->num_points;
    child->points = (Point3D*)malloc(child->num_points * sizeof(Point3D));
    if (!child->points) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < child->num_points; i++) {
        if (((double)rand() / RAND_MAX) < 0.5) {
            child->points[i] = ind1->points[i];
        } else {
            child->points[i] = ind2->points[i];
        }
    }
    
    double alpha = (double)rand() / RAND_MAX;
    child->sigma = alpha * ind1->sigma + (1 - alpha) * ind2->sigma;
    child->fitness = -1;
}

void mutate_individual(Individual* ind, double tau, double eps0, double bmin, double bmax) {
    double new_sigma = ind->sigma * exp(tau*(rand()/(double)RAND_MAX - 0.5)*2.0);
    new_sigma = MAX(new_sigma, eps0);

    for (int i = 0; i < ind->num_points; i++) {
        ind->points[i].x += new_sigma*(rand()/(double)RAND_MAX - 0.5)*2.0;
        ind->points[i].y += new_sigma*(rand()/(double)RAND_MAX - 0.5)*2.0;
        ind->points[i].z += new_sigma*(rand()/(double)RAND_MAX - 0.5)*2.0;

        if (ind->points[i].x < bmin) ind->points[i].x = 2*bmin - ind->points[i].x;
        else if (ind->points[i].x > bmax) ind->points[i].x = 2*bmax - ind->points[i].x;

        if (ind->points[i].y < bmin) ind->points[i].y = 2*bmin - ind->points[i].y;
        else if (ind->points[i].y > bmax) ind->points[i].y = 2*bmax - ind->points[i].y;

        if (ind->points[i].z < bmin) ind->points[i].z = 2*bmin - ind->points[i].z;
        else if (ind->points[i].z > bmax) ind->points[i].z = 2*bmax - ind->points[i].z;
    }
    ind->sigma = new_sigma;
}

Point3D* create_sphere_points(int n) {
    Point3D* pts = malloc(n * sizeof(Point3D));
    if (!pts) return NULL;
    for (int i = 0; i < n; i++) {
        double phi = 2*PI*rand()/RAND_MAX;
        double theta = acos(2.0*rand()/RAND_MAX - 1.0);
        pts[i].x = sin(theta)*cos(phi);
        pts[i].y = sin(theta)*sin(phi);
        pts[i].z = cos(theta);
    }
    return pts;
}

double parallel_fitness_evaluation(Individual* offspring, int lambda, Point3D* target_shape, int target_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    int chunk = lambda / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? lambda : start + chunk;

    double* local_fitness = malloc(lambda * sizeof(double));
    for (int i = 0; i < lambda; i++) {
        local_fitness[i] = -1.0;
    }

    for (int i = start; i < end; i++) {
        local_fitness[i] = compute_3d_iou(offspring[i].points, offspring[i].num_points,
                                        target_shape, target_size);
    }

    double* global_fitness = malloc(lambda * sizeof(double));
    MPI_Allreduce(local_fitness, global_fitness, lambda, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    for (int i = 0; i < lambda; i++) {
        offspring[i].fitness = global_fitness[i];
    }

    free(local_fitness);
    free(global_fitness);

    double end_time = MPI_Wtime();
    return end_time - start_time;
}

Individual EEmupluslambda(int G, double tau, double eps0, Individual* population, int pop_size, 
                         int lambda, int mu, double box_min, double box_max, 
                         Point3D* target_shape, int target_size, double* total_eval_time,
                         int save_interval) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    *total_eval_time = 0.0;

    qsort(population, pop_size, sizeof(Individual), compare_individuals);

    for (int gen = 0; gen < G; gen++) {
        Individual* offspring = malloc(lambda * sizeof(Individual));
        for (int i = 0; i < lambda; i++) {
            int p1 = rand() % mu;
            int p2 = rand() % mu;
            crossover(&population[p1], &population[p2], &offspring[i]);
            mutate_individual(&offspring[i], tau, eps0, box_min, box_max);
        }

        double eval_time = parallel_fitness_evaluation(offspring, lambda, target_shape, target_size);
        *total_eval_time += eval_time;

        Individual* new_pop = malloc((mu + lambda) * sizeof(Individual));

        for (int i = 0; i < mu; i++) {
            new_pop[i] = copy_individual(&population[i]);
        }
        for (int i = 0; i < lambda; i++) {
            new_pop[mu + i] = copy_individual(&offspring[i]);
        }

        for (int i = 0; i < lambda; i++) free_individual(&offspring[i]);
        free(offspring);

        qsort(new_pop, mu + lambda, sizeof(Individual), compare_individuals);

        for (int i = 0; i < pop_size; i++) free_individual(&population[i]);
        for (int i = 0; i < mu; i++) population[i] = copy_individual(&new_pop[i]);
        for (int i = mu; i < mu + lambda; i++) free_individual(&new_pop[i]);
        free(new_pop);

        if (save_interval > 0 && (gen % save_interval == 0 || gen == G - 1)) {
            if (rank == 0) {
                char filename[50];
                sprintf(filename, "evolution_generation_%04d.txt", gen);
                save_individual_to_file(
                    &population[0], 
                    filename,
                    gen,
                    tau,
                    eps0,
                    lambda,
                    mu,
                    box_min,
                    box_max,
                    *total_eval_time
                );
            }
        }
    }

    Individual best = copy_individual(&population[0]);
    for (int i = 0; i < pop_size; i++) free_individual(&population[i]);
    return best;
}

void save_individual_to_file(const Individual* ind, const char* filename, 
                            int G, double tau, double eps0, 
                            int lambda, int mu, double box_min, double box_max,
                            double eval_time) {
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
    fprintf(file, "EvaluationTime %.4f\n", eval_time);
    fprintf(file, "Fitness %f\n", ind->fitness);
    fprintf(file, "Sigma %f\n", ind->sigma);
    fprintf(file, "Points %d\n", ind->num_points);
    
    fprintf(file, "\nPoints (x,y,z):\n");
    for (int i = 0; i < ind->num_points; i++) {
        fprintf(file, "%f %f %f\n", ind->points[i].x, ind->points[i].y, ind->points[i].z);
    }

    fclose(file);
}

Individual run_ee_algorithm(
    Point3D* target_shape, 
    int target_size,
    int num_points_in_individual,
    int generations,
    int population_size,
    int mu_size,
    int lambda_size,
    double tau,
    double epsilon0,
    double box_min,
    double box_max,
    double* total_eval_time,
    int save_interval) 
{
    Individual* population = (Individual*)malloc(population_size * sizeof(Individual));
    if (!population) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < population_size; i++) {
        population[i] = create_individual(box_min, box_max, num_points_in_individual, 
                                        target_shape, target_size);
    }
    
    Individual best = EEmupluslambda(
        generations, tau, epsilon0, 
        population, population_size,
        lambda_size, mu_size, 
        box_min, box_max,
        target_shape, target_size,
        total_eval_time,
        save_interval
    );
    
    free(population);
    return best;
}
