/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N^2 unknowns, each processor works with its
 * part, which has lN^2 unknowns where lN = N/floor(sqrt(p)) unknowns.
 * Author: Georg Stadler
 * Modified to 2D case by Xiayimei Han
 */

// mpic++ -std=c++11 jacobi_2D.cpp -o jacobi_2D
// mpirun -np 4 ./jacobi_2D 16 10000

#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq)
{
    int i, j;
    double tmp, gres = 0.0, lres = 0.0;

    for (i = 1; i <= lN; i++)
    {
        for (j = 1; j <= lN; j++)
        {
            tmp = ((4.0 * lu[(i * (lN + 2) + j)] - lu[(i - 1) * (lN + 2) + j] - lu[(i + 1) * (lN + 2) + j] - lu[i * (lN + 2) + (j - 1)] - lu[i * (lN + 2) + (j + 1)]) * invhsq - 1);
            lres += tmp * tmp;
        }
    }
    /* use allreduce for convenience; a reduce would also be sufficient */
    MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(gres);
}

int main(int argc, char *argv[])
{
    int mpirank, i, j, p, N, lN, iter, max_iters;
    MPI_Status status;
    MPI_Request request_out1, request_in1;
    MPI_Request request_out2, request_in2;
    MPI_Request request_out3, request_in3;
    MPI_Request request_out4, request_in4;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    /* get name of host running MPI process */
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &max_iters);

    /* compute number of unknowns handled by each process */
    int sqrt_p = int(floor(sqrt(p)));
    if (p != pow(sqrt_p, 2))
    {
        printf("p: %d \n", p);
        printf("Exiting. p must be a square\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    lN = N / sqrt_p;
    if ((N % sqrt_p != 0) && mpirank == 0)
    {
        printf("N: %d, local N: %d\n", N, lN);
        printf("Exiting. N must be a multiple of the square root of p \n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();

    /* Allocation of vectors, including up, down, left and right ghost points */
    double *lu = (double *)calloc(sizeof(double), pow((lN + 2), 2));
    for (i = 0; i < lN + 2; i++)
    {
        for (j = 0; j < lN + 2; j++)
        {
            lu[i * (lN + 2) + j] = 0;
        }
    }
    double *lunew = (double *)calloc(sizeof(double), pow((lN + 2), 2));
    double *lutemp;

    double h = 1.0 / (N + 1);
    double hsq = h * h;
    double invhsq = 1. / hsq;
    double gres, gres0, tol = 1e-5;

    /* initial residual */
    gres0 = compute_residual(lu, lN, invhsq);
    gres = gres0;

    for (iter = 0; iter < max_iters && gres / gres0 > tol; iter++)
    {
        /* interleaf computation and communication: compute the first
         * and last value, which are communicated with non-blocking
         * send/recv. During that communication, do all the local work */

        /* Jacobi step for the up, down, left and right most points */
        // up
        i = 1;
        for (j = 1; j < lN + 1; j++)
        {
            lunew[i * (lN + 2) + j] = (hsq + lu[(i - 1) * (lN + 2) + j] + lu[(i + 1) * (lN + 2) + j] + lu[i * (lN + 2) + (j - 1)] + lu[i * (lN + 2) + (j + 1)]) / 4.0;
        }
        // down
        i = lN;
        for (j = 1; j < lN + 1; j++)
        {
            lunew[i * (lN + 2) + j] = (hsq + lu[(i - 1) * (lN + 2) + j] + lu[(i + 1) * (lN + 2) + j] + lu[i * (lN + 2) + (j - 1)] + lu[i * (lN + 2) + (j + 1)]) / 4.0;
        }
        // left
        j = 1;
        for (i = 1; i < lN + 1; i++)
        {
            lunew[i * (lN + 2) + j] = (hsq + lu[(i - 1) * (lN + 2) + j] + lu[(i + 1) * (lN + 2) + j] + lu[i * (lN + 2) + (j - 1)] + lu[i * (lN + 2) + (j + 1)]) / 4.0;
        }
        // right
        j = lN;
        for (i = 1; i < lN + 1; i++)
        {
            lunew[i * (lN + 2) + j] = (hsq + lu[(i - 1) * (lN + 2) + j] + lu[(i + 1) * (lN + 2) + j] + lu[i * (lN + 2) + (j - 1)] + lu[i * (lN + 2) + (j + 1)]) / 4.0;
        }

        int row_num = mpirank / sqrt_p;
        int col_num = mpirank % sqrt_p;
        if (row_num < sqrt_p - 1)
        {
            /* If not the downmost row, send/recv bdry values to/from the downstairs */
            MPI_Irecv(&(lunew[(lN + 1) * (lN + 2) + 1]), lN, MPI_DOUBLE, mpirank + sqrt_p, 123, MPI_COMM_WORLD, &request_in1);
            MPI_Isend(&(lunew[(lN) * (lN + 2) + 1]), lN, MPI_DOUBLE, mpirank + sqrt_p, 124, MPI_COMM_WORLD, &request_out1);
        }
        if (row_num > 0)
        {
            /* If not the upmost row, send/recv bdry values to/from the upstairs*/
            MPI_Irecv(&(lunew[1]), lN, MPI_DOUBLE, mpirank - sqrt_p, 124, MPI_COMM_WORLD, &request_in2);
            MPI_Isend(&(lunew[lN + 3]), lN, MPI_DOUBLE, mpirank - sqrt_p, 123, MPI_COMM_WORLD, &request_out2);
        }
        if (col_num > 0)
        {
            /* If not the leftmost column, send/recv bdry values to/from the left hand side */
            for (i = 1; i < lN + 1; i++)
            {
                MPI_Irecv(&(lunew[i * (lN + 2)]), 1, MPI_DOUBLE, mpirank - 1, i, MPI_COMM_WORLD, &request_in3);
                MPI_Isend(&(lunew[i * (lN + 2) + 1]), 1, MPI_DOUBLE, mpirank - 1, 200 + i, MPI_COMM_WORLD, &request_out3);
            }
        }
        if (col_num < sqrt_p - 1)
        {
            /* If not the rightmost column, send/recv bdry values to/from the right hand side */
            for (i = 1; i < lN + 1; i++)
            {
                MPI_Irecv(&(lunew[i * (lN + 2) + (lN + 1)]), 1, MPI_DOUBLE, mpirank + 1, 200 + i, MPI_COMM_WORLD, &request_in4);
                MPI_Isend(&(lunew[i * (lN + 2) + lN]), 1, MPI_DOUBLE, mpirank + 1, i, MPI_COMM_WORLD, &request_out4);
            }
        }
        /* Jacobi step for all the inner points */
        for (i = 2; i < lN; i++)
        {
            for (j = 2; j < lN; j++)
            {
                lunew[i * (lN + 2) + j] = (hsq + lu[(i - 1) * (lN + 2) + j] + lu[(i + 1) * (lN + 2) + j] + lu[i * (lN + 2) + (j - 1)] + lu[i * (lN + 2) + (j + 1)]) / 4;
            }
        }

        /* check if Isend/Irecv are done */
        if (row_num < sqrt_p - 1)
        {
            MPI_Wait(&request_out1, &status);
            MPI_Wait(&request_in1, &status);
        }
        if (row_num > 0)
        {
            MPI_Wait(&request_out2, &status);
            MPI_Wait(&request_in2, &status);
        }
        if (col_num > 0)
        {
            MPI_Wait(&request_out3, &status);
            MPI_Wait(&request_in3, &status);
        }
        if (col_num < sqrt_p - 1)
        {
            MPI_Wait(&request_out4, &status);
            MPI_Wait(&request_in4, &status);
        }

        /* copy newu to u using pointer flipping */
        lutemp = lu;
        lu = lunew;
        lunew = lutemp;
        if (0 == (iter % 10))
        {
            gres = compute_residual(lu, lN, invhsq);
            if (0 == mpirank)
            {
                printf("Iter %d: Residual: %g\n", iter, gres);
            }
        }
    }

    /* Clean up */
    free(lu);
    free(lunew);

    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - tt;
    if (0 == mpirank)
    {
        printf("Time elapsed is %f seconds.\n", elapsed);
    }
    MPI_Finalize();
    return 0;
}