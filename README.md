# Parallelized-Polygon-Approximation-with-Evolutionary-Strategies

Final proyect for the class of of High Performance Computing at UNAM ENES Unidad Morelia.

## Author

Diego Maldonado Castro: diegomaldonadocastro1805@gmail.com

## Introduction

This project implements a parallel (μ + λ) evolutionary algorithm for 3D shape approximation using MPI. It evaluates the fitness of candidate solutions (individuals) in parallel across multiple processes to accelerate convergence. 



## Features

- Evolutionary strategy: (μ + λ) selection
- Crossover and mutation of 3D shapes (points in space)
- Fitness function using 3D IoU (Intersection over Union)
- MPI parallelism for distributed fitness evaluation
- Output of best individual per run



## Algorithm Overview

The algorithm maintains a population of 3D point clouds (individuals), which are evolved over generations via:
- **Crossover**: Combining traits from two parents.
- **Mutation**: Random variation within defined bounds.
- **Fitness Evaluation**: How well the point cloud matches a target shape through its convexhall.

MPI is used to parallelize the fitness evaluation step, significantly improving runtime on multi-core systems.

## Project Structure

-  main.c # Entry point: initializes data and runs experiments
- ee_algorithm.c # Evolutionary logic with MPI parallelism
- ee_algorithm.h # Algorithm interface
- data/ # Output .txt for best individuals

## Execution
```
mpicc -o ee_algorithm main.c ee_algorithm.c -lm
```

then

```
mpiexec -np #_of_cores ./ee_algorithm
```

## Results

![evolution](https://github.com/user-attachments/assets/e90a3ba1-f013-4e77-b313-6ca254c67cb3)

Example of a 200 point individual approximating a sphere.

![Figure_1](https://github.com/user-attachments/assets/86605c1d-72be-4d4b-837f-bd772858f8ff)

Review of performance by number of threads (100 tests for every number fo threads).

## Dependencies

- Open MPI: https://www.open-mpi.org/
