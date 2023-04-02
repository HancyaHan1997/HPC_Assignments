// g++ -std=c++11 -fopenmp omp-scan.cpp -o omp-scan
// OMP_NUM_THREADS=2 ./omp-scan

#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0; }
inline omp_int_t omp_get_num_threads() { return 1; }
inline omp_int_t omp_get_max_threads() { return 1; }
#endif

#include <algorithm>
#include <stdio.h>
#include <math.h>
#include "utils.h"

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long *prefix_sum, const long *A, long n)
{
  if (n == 0)
    return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++)
  {
    prefix_sum[i] = prefix_sum[i - 1] + A[i - 1];
  }
}

void scan_omp(long *prefix_sum, const long *A, long n)
{
  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel
  if (n == 0)
    return;

  int p = omp_get_max_threads();
  double width = floor(n / p);
  double *k = (double *)malloc((p + 1) * sizeof(double));
  for (int i = 0; i < p; i++)
  {
    k[i] = 1 + width * i;
  }
  k[p] = n;

// compute partial sums
#pragma omp parallel
  {
    int j = omp_get_thread_num();
    int p = omp_get_num_threads();

    printf("This is thread %d out of %d threads \n", j, p);
    for (int i = k[j]; i < k[j + 1]; i++)
    {
      if (i == k[j])
      {
        prefix_sum[i] = A[i - 1];
      }
      else
      {
        prefix_sum[i] = prefix_sum[i - 1] + A[i - 1];
      }
    }
  }

  // serially compute previou partial sums
  double *prev_sum = (double *)malloc(p * sizeof(double));
  prev_sum[0] = 0;
  for (int i = 1; i < p; i++)
  {
#pragma omp reduction(+ \
                      : prev_sum)
    prev_sum[i] = prev_sum[i - 1] + prefix_sum[int(k[i]) - 1];
  }

#pragma omp parallel
  {
    int j = omp_get_thread_num();
    int p = omp_get_num_threads();

    printf("This is thread %d out of %d threads \n", j, p);
    for (int i = k[j]; i < k[j + 1]; i++)
    {
      prefix_sum[i] += prev_sum[j];
    }
  }
  free(k);
  free(prev_sum);
}

int main()
{
  long N = 100000000;
  long *A = (long *)malloc(N * sizeof(long));
  long *B0 = (long *)malloc(N * sizeof(long));
  long *B1 = (long *)malloc(N * sizeof(long));
  for (long i = 0; i < N; i++)
    A[i] = rand();
  for (long i = 0; i < N; i++)
    B1[i] = 0;

  Timer t;
  t.tic();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", t.toc());

  t.tic();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", t.toc());

  long err = 0;
  int c = 0;
  for (long i = 0; i < N; i++)
  {
    err = std::max(err, std::abs(B0[i] - B1[i]));
    if (std::abs(B0[i] - B1[i]) > 0 and c < 50)
    {
      c++;
      // printf("i, B0[i], B1[i] = %d, %ld, %ld \n", i, B0[i], B1[i], A[i - 1]);
    }
  }

  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
