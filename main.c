#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mpi.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

const unsigned NUM_OF_POINTS = 1024; // Number of points to be generated in one iteration
int myrank; // Current rank
int ranksize; // Number of processes in communicator


typedef struct {
    double x, y, z;
} point_t;

/**
 * @param min lower bound
 * @param max upper bound
 * @return random real value from min to max
 */
double uniformDist(double min, double max);

/**
 * @brief The function generates n points 
 * within a parallelepiped П
 * 
 * @param points the array where the generated 
 * points will be stored
 * @param n number of generated points
 */
void pointsGenerator(point_t *points, unsigned n);

/**
 * @brief Subintegral function
 * 
 * @param p function argument
 * @return function value
 */
double func(point_t *p);

/**
 * @brief Function that determines 
 * whether a point p belongs to the 
 * integration area G
 * 
 * @param p point
 * @return 0 or 1
 */
uint8_t insideArea(point_t *p);

/**
 * @brief Initializing sendcounts and displs 
 * parameters of the MPI_Scatterv function
 * 
 * @param sendcounts an array of values of the number 
 * of elements to be passed to the processes
 * @param displs array of offset values
 */
void initScattervParams(int *sendcounts, int *displs);

/**
 * @brief Numerical Monte Carlo method 
 * for calculating the value of a definite integral.
 * 
 * @param f subintegral function
 * @param inside function that determines 
 * whether a point belongs to the integration area G 
 * @param pointsGen points generator
 * @param vol the volume of the parallelepiped П
 * which completely contains the integration area G
 * @param analytical_res value of the integral 
 * calculated analytically
 * @param eps calculation accuracy
 */
void MonteCarloParallel(double (*f)(point_t *),
                        uint8_t (*inside)(point_t *),
                        void (*pointsGen)(point_t *, unsigned),
                        const double vol,
                        const double analytical_res,
                        const double eps);


int main(int argc, char **argv)
{

    int error;
    error = MPI_Init(&argc, &argv);
    if (error != MPI_SUCCESS) {
        fprintf(stderr, "ERROR: CAN'T MPI INIT\n");
        return 1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
    MPI_Barrier(MPI_COMM_WORLD);
    
    const double analytical_res = 4.0 * M_PI / 3.0; 
    const double volume = 8;
    double eps = 1.0e-4; // default eps value

    if (argc >= 2) sscanf(argv[1], "%lf", &eps);  

    MonteCarloParallel(func, insideArea, pointsGenerator, 
                       volume, analytical_res, eps);

    MPI_Finalize();
    return 0;
}


double uniformDist(double min, double max)
{
    return min + (max - min) * ((double) rand() / RAND_MAX);
}


void pointsGenerator(point_t *points, unsigned n)
{
    for (unsigned i = 0; i < n; ++i) {
        point_t *p = &points[i];
        p->x = 0;
        p->y = uniformDist(0, 1.0);
        p->z = uniformDist(0, 1.0);
    }
}


double func(point_t *p)
{
    return sqrt(p->y * p->y + p->z * p->z);
}


uint8_t insideArea(point_t *p)
{
    return p->y * p->y + p->z * p->z <= 1;
}


void initScattervParams(int *sendcounts, int *displs)
{
    int nmin = (NUM_OF_POINTS / (ranksize - 1)) * 3;
    int nextra = NUM_OF_POINTS % (ranksize - 1);
    int k = 0;

    sendcounts[0] = 0; // master process 0 will not receive data
    displs[0] = 0;
    for(unsigned i = 1; i < ranksize; ++i) {
        sendcounts[i] = i < nextra + 1 ? nmin + 3 : nmin;
        displs[i] = k;
        k = k + sendcounts[i];
    }
}


void MonteCarloParallel(double (*f)(point_t *),
                        uint8_t (*inside)(point_t *),
                        void (*pointsGen)(point_t *, unsigned),
                        const double vol,
                        const double analytical_res,
                        const double eps)
{
    srand(1);
    double global_sum = 0.0; // the sum from all processes on all already generated points
    double global_tmp = 0.0; // the sum from all processes on one portion of points
    double local_sum = 0.0; // the sum from one process on one portion of points
    double mc_res = 0.0; // Monte Carlo result
    unsigned n = 0; // counter of the number of generated points
    int sendcounts[ranksize], displs[ranksize]; // MPI_Scatterv parameters

    double sequential_time = 0.0;
    double sequential_start = 0.0;
    double parallel_time = 0.0;
    double parallel_start = 0.0;
    double allreduce_time = 0.0;
    double allreduce_start = 0.0;

    double start = MPI_Wtime();

    initScattervParams(sendcounts, displs);

    point_t *randPoints = NULL;
    point_t *subRandPoints = NULL;

    if (myrank == 0) {
        randPoints = (point_t *) calloc(NUM_OF_POINTS, sizeof(point_t));
    } else {
        subRandPoints = (point_t *) calloc(sendcounts[myrank], sizeof(double));
    }

    while (fabs(mc_res - analytical_res) > eps) {
        if (myrank == 0) {
            sequential_start = MPI_Wtime();
            pointsGen(randPoints, NUM_OF_POINTS);
            sequential_time += MPI_Wtime() - sequential_start;
        }
        n += NUM_OF_POINTS;
        MPI_Scatterv((double *) randPoints, sendcounts, displs, MPI_DOUBLE, 
                     (double *) subRandPoints, sendcounts[myrank], MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
        
        if (myrank != 0) {
            parallel_start = MPI_Wtime();
            for (unsigned i = 0; i < sendcounts[myrank] / 3; ++i) {
                point_t *p = &subRandPoints[i];
                if (inside(p)) { local_sum += f(p); }
            }
            parallel_time += MPI_Wtime() - parallel_start;
        }

        if (myrank != 0) allreduce_start = MPI_Wtime();
        MPI_Allreduce(&local_sum, &global_tmp, 1, MPI_DOUBLE, 
                      MPI_SUM, MPI_COMM_WORLD);
        if (myrank != 0) allreduce_time += MPI_Wtime() - allreduce_start;
        global_sum += global_tmp;
        mc_res = vol * global_sum / n;
        local_sum = 0.0;
    }
    double time = MPI_Wtime() - start;
    double global_time;
    double global_parallel;
    double global_allreduce;

    MPI_Reduce(&time, &global_time, 1, MPI_DOUBLE, 
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&parallel_time, &global_parallel, 1, MPI_DOUBLE, 
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&allreduce_time, &global_allreduce, 1, MPI_DOUBLE, 
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (myrank == 0) {
        double abs_err = fabs(mc_res - analytical_res);
        printf("Monte Carlo result: %.10lf\n", mc_res);
        printf("Abs error: %.10lf\n", abs_err);
        printf("Number of points: %d\n", n);
        printf("Time: %lf\n", global_time);
        printf("Time points generation: %lf\n", sequential_time);
        printf("Time parallel: %lf\n", global_parallel);
        printf("Time MPI_Allreduce: %lf\n", global_allreduce);
        free(randPoints);
    } else {
        free(subRandPoints);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
