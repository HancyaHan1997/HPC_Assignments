// g++ -std=c++11 -fopenmp jacobi2D-omp.cpp -o jacobi2D-omp
// OMP_NUM_THREADS=1 ./jacobi2D-omp -n 10

#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0; }
inline omp_int_t omp_get_num_threads() { return 1; }
inline omp_int_t omp_get_max_threads() { return 1; }
#endif

#include <stdio.h>
#include <cmath>
#include <stdexcept>
#include "utils.h"
#include <iostream>
#include <fstream>
using namespace std;

// throughout this program, we represent a N+2 by N+2 matrix as a vector such that the (i,j)-th entry of the matrix is (i* (N+2) + j)-th entry of the vector

double Jacobi_iterate(double *prev_u, long N, long i, long j, double *f)
{
    if (N < 3)
    {
        throw std::invalid_argument("N should be at least 3");
    }
    double new_u_i_j = (f[i * (N + 2) + j] / pow(N + 1, 2) + prev_u[((i - 1) * (N + 2) + (j))] + prev_u[((i) * (N + 2) + (j - 1))] + prev_u[((i + 1) * (N + 2) + (j))] + prev_u[((i) * (N + 2) + (j + 1))]) / 4;
    // printf("i, j, u[i,j] = %ld, %ld, %f \n", i, j, new_u_i_j);
    return new_u_i_j;
}

int main(int argc, char **argv)
{
    Timer t;
    t.tic();
    long N = read_option<long>("-n", argc, argv);
    double *u = (double *)malloc(pow(N + 2, 2) * sizeof(double));
    for (long i = 0; i < pow(N + 2, 2); i++)
    {
        u[i] = 0;
    }

    double *f = (double *)malloc(pow(N + 2, 2) * sizeof(double));
    for (long i = 0; i < pow(N + 2, 2); i++)
    {
        f[i] = 1;
    }

    int num_iter = 1000;
    for (long c = 0; c < num_iter; c++)
    {
        double *new_u = (double *)malloc(pow(N + 2, 2) * sizeof(double));
        // initialize boundary values to zero
        for (long i = 0; i < N + 2; i++)
        {
            new_u[i * (N + 2)] = 0;
            new_u[i * (N + 2) + (N + 1)] = 0;
        }

        for (long j = 0; j < N + 2; j++)
        {
            new_u[j] = 0;
            new_u[(N + 1) * (N + 2) + j] = 0;
        }

#pragma omp parallel for
        for (long i = 1; i < N + 1; i++)
        {
            for (long j = 1; j < N + 1; j++)
            {
                new_u[i * (N + 2) + j] = Jacobi_iterate(u, N, i, j, f);
            }
        }

        free(u);
        u = new_u;
    }

    printf("The total run time is %f. \n", t.toc());
    ofstream myfile;
    string filename = "u_computed_with_" + to_string(omp_get_max_threads()) + "_threads.txt";
    myfile.open(filename);
    for (long i = 0; i < N + 2; i++)
    {
        for (long j = 0; j < N + 2; j++)
        {
            myfile << to_string(u[(i) * (N + 2) + j]) << " ";
        }
        myfile << "\n";
    }
    myfile.close();
    free(u);
    free(f);
    return 0;
}