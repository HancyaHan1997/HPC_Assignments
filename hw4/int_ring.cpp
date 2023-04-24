// mpic++ -std=c++11 int_ring.c -o int_ring
// mpirun -np 3 ./int_ring 10000

#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

double ring_call(long Nrepeat, long Nsize, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    int total;
    MPI_Comm_size(comm, &total);

    double *msg = (double *)malloc(Nsize * sizeof(double));

    for (long i = 0; i < Nsize; i++)
        msg[i] = rank;

    MPI_Barrier(comm);
    double tt = MPI_Wtime();
    for (long repeat = 0; repeat < Nrepeat; repeat++)
    {
        MPI_Status status;
        if (rank == 0)
        {
            MPI_Send(msg, Nsize, MPI_DOUBLE, 1, repeat, comm);
            MPI_Recv(msg, Nsize, MPI_DOUBLE, total - 1, repeat, comm, &status);
            // printf("rank %d received %f \n", rank, msg[0]);
        }
        else
        {
            MPI_Recv(msg, Nsize, MPI_DOUBLE, rank - 1, repeat, comm, &status);
            // printf("rank %d received %f \n", rank, msg[0]);
            for (long i = 0; i < Nsize; i++)
                msg[i] += rank;
            if (rank < total - 1)
            {
                MPI_Send(msg, Nsize, MPI_DOUBLE, rank + 1, repeat, comm);
            }
            else
            {
                MPI_Send(msg, Nsize, MPI_DOUBLE, 0, repeat, comm);
            }
        }
        // check that every process has sent and received a message before proceeding to the next round
        // MPI_Wait(&request_in1, &status);
        // MPI_Wait(&request_out1, &status);
    }
    tt = MPI_Wtime() - tt;

    // check if all processors have properly added their contribution each time they received and sent the message
    // long single_period_sum = (total-1) * total/2;
    // printf("The total sum should be %ld to %ld \n", single_period_sum * (Nrepeat-1), single_period_sum * Nrepeat);
    printf("At rank %d, the sum is %f \n", rank, msg[0]);

    free(msg);
    return tt;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    if (argc < 2)
    {
        printf("Usage: mpirun ./int_ring Nrepeat \n");
        abort();
    }
    int Nrepeat = atoi(argv[1]);

    int rank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    int total;
    MPI_Comm_size(comm, &total);

    double tt = ring_call(Nrepeat, 1, comm);
    if (!rank)
        printf("ring_call latency: %e ms\n", tt / (total * Nrepeat) * 1000);

    long Nsize = 262144; // 2MB / 8 byte (size of a single double) = 262144
    tt = ring_call(Nrepeat, Nsize, comm);
    if (!rank)
        printf("ring_call bandwidth: %e MB/s\n", (2 * total * Nrepeat) / tt);

    MPI_Finalize();
}