#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mpi.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

const unsigned NUM_OF_POINTS = 65536; // Number of points to be generated in one iteration
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
 * @brief Subintegral function:
 * F(x, y, z) = sqrt(y^2 + z^2) if (x, y, z) in G,
 * 0 otherwise; where G:  0<=x<=2, y^2 + z^2 <= 1
 * 
 * @param p function argument
 * @return function value
 */
double func(point_t *p);

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
 * @param pointsGen points generator
 * @param vol the volume of the parallelepiped П
 * which completely contains the integration area G
 * @param analyticalRes value of the integral 
 * calculated analytically
 * @param eps calculation accuracy
 */
void MonteCarloParallel(double (*f)(point_t *),
                        void (*pointsGen)(point_t *, unsigned),
                        const double vol,
                        const double analyticalRes,
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
    
    const double analyticalRes = 4.0 * M_PI / 3.0; 
    const double volume = 8;
    double eps = 1.0e-5; // default eps value

    if (argc >= 2) sscanf(argv[1], "%lf", &eps);  

    MonteCarloParallel(func, pointsGenerator, 
                       volume, analyticalRes, eps);

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
        p->x = 1.0;
        p->y = uniformDist(0, 1.0);
        p->z = uniformDist(0, 1.0);
    }
}


double func(point_t *p)
{
    if (p->x >= 0. && p->x <= 2. && p->y * p->y + p->z * p->z <= 1.) {
        return sqrt(p->y * p->y + p->z * p->z);
    }
    return 0.0;
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
                        void (*pointsGen)(point_t *, unsigned),
                        const double vol,
                        const double analyticalRes,
                        const double eps)
{
    srand(1);
    double globalSum = 0.0; // the sum from all processes on all already generated points
    double globalSumTmp = 0.0; // the sum from all processes on one portion of points
    double localSum = 0.0; // the sum from one process on one portion of points
    double MCres = 0.0; // Monte Carlo result
    unsigned n = 0; // counter of the number of generated points
    int sendcounts[ranksize], displs[ranksize]; // MPI_Scatterv parameters

    double time = 0.0;

    time -= MPI_Wtime();

    initScattervParams(sendcounts, displs);

    point_t *randPoints = NULL;
    point_t *subRandPoints = NULL;
    int procPointCount = sendcounts[myrank] / 3;

    if (myrank == 0) {
        randPoints = (point_t *) calloc(NUM_OF_POINTS, sizeof(point_t));
    } else {
        subRandPoints = (point_t *) calloc(procPointCount, sizeof(point_t));
    }

    char needBreak = fabs(MCres - analyticalRes) <= eps;

    while (!needBreak) {
        if (myrank == 0) {
            pointsGen(randPoints, NUM_OF_POINTS);
            n += NUM_OF_POINTS;
        }
        MPI_Scatterv((double *) randPoints, sendcounts, displs, MPI_DOUBLE, 
                     (double *) subRandPoints, sendcounts[myrank], MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
        
        if (myrank != 0) {
            localSum = 0.0;
            for (unsigned i=0; i < procPointCount; ++i) {
                point_t *p = &subRandPoints[i];
                localSum += f(p);
            }
        }

        MPI_Reduce(&localSum, &globalSumTmp, 1, MPI_DOUBLE, 
                   MPI_SUM, 0, MPI_COMM_WORLD);
        if (myrank == 0) {
            globalSum += globalSumTmp;
            MCres = vol * globalSum / n;
            needBreak = fabs(MCres - analyticalRes) <= eps;
        }
        MPI_Bcast(&needBreak, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    time += MPI_Wtime();
    double global_time;

    MPI_Reduce(&time, &global_time, 1, MPI_DOUBLE, 
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (myrank == 0) {
        double abs_err = fabs(MCres - analyticalRes);
        printf("Monte Carlo result: %.10lf\n", MCres);
        printf("Abs error: %.10lf\n", abs_err);
        printf("Number of points: %d\n", n);
        printf("Time: %lf\n", global_time);
        free(randPoints);
    } else {
        free(subRandPoints);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
