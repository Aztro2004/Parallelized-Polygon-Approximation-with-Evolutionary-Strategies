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

<img width="747" alt="Screenshot 2025-05-21 at 12 12 59 p m" src="https://github.com/user-attachments/assets/9b790115-b3d7-4543-a9dc-cc020ec6304c" />



Example of a 100 point individual approximating a sphere.

## Dependencies

- Open MPI: https://www.open-mpi.org/
