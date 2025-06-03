#ifndef EE_ALGORITHM_H
#define EE_ALGORITHM_H

#include <stdbool.h>

typedef struct {
    double x, y, z;
} Point3D;

typedef struct {
    Point3D* points;
    int num_points;
    double sigma;
    double fitness;
} Individual;

Point3D* create_sphere_points(int num_points);

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
    int save_interval
);

void free_individual(Individual* ind);

void save_individual_to_file(
    const Individual* ind, 
    const char* filename, 
    int G, double tau, double eps0, 
    int lambda, int mu, double box_min, double box_max,
    double eval_time
);

#endif // EE_ALGORITHM_H