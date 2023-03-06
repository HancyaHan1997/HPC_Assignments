// g++ -std=c++11 -O2 -march=native inner_product.cpp -o inner_product

#include <stdio.h>
#include "utils.h"

double inner0(long N, double *a, double *b)
{
    double prod = 0;
    for (long i = 0; i < N; i++)
        prod += a[i] * b[i];
    return prod;
}

double inner1(long N, double *a, double *b)
{
    double sum1 = 0, sum2 = 0;
    for (long i = 0; i < N / 2; i++)
    {
        sum1 += a[2 * i] * b[2 * i];
        sum2 += a[2 * i + 1] * b[2 * i + 1];
    }
    return sum1 + sum2;
}

double inner2(long N, double *a, double *b)
{
    double sum1 = 0, sum2 = 0;
    for (long i = 0; i < N / 2; i++)
    {
        sum1 += *(a + 0) * *(b + 0);
        sum2 += *(a + 1) * *(b + 1);

        a += 2;
        b += 2;
    }
    return sum1 + sum2;
}

double inner3(long N, double *a, double *b)
{
    double sum1 = 0, sum2 = 0, temp1 = 0, temp2 = 0;
    for (long i = 0; i < N / 2; i++)
    {
        temp1 = *(a + 0) * *(b + 0);
        temp2 = *(a + 1) * *(b + 1);
        sum1 += temp1;
        sum2 += temp2;
        a += 2;
        b += 2;
    }
    return sum1 + sum2;
}

double inner4(long N, double *a, double *b)
{
    double sum1 = 0, sum2 = 0, temp1 = 0, temp2 = 0;
    for (long i = 0; i < N / 2; i++)
    {
        sum1 += temp1;
        temp1 = *(a + 0) * *(b + 0);
        sum2 += temp2;
        temp2 = *(a + 1) * *(b + 1);
        a += 2;
        b += 2;
    }
    return sum1 + sum2 + temp1 + temp2;
}

int main(int argc, char **argv)
{
    Timer t;
    long n = read_option<long>("-n", argc, argv);
    long repeat = read_option<long>("-repeat", argc, argv, "1");

    double *x = (double *)malloc(n * sizeof(double));
    double *y = (double *)malloc(n * sizeof(double));
    for (long i = 0; i < n; i++)
    {
        x[i] = i + 1;
        y[i] = 2.0 / (i + 1);
    }

    double *times = new double[5];
    double *dotproducts = new double[5];
    for (int i = 0; i < 5; i++)
    {
        times[i] = 0;
        dotproducts[i] = 0;
    }

    t.tic();
    for (long p = 0; p < repeat; p++)
    {
        dotproducts[0] += inner0(n, x, y);
    }
    times[0] = t.toc();

    t.tic();
    double x_dot_y1 = 0;
    for (long p = 0; p < repeat; p++)
    {
        dotproducts[1] += inner1(n, x, y);
    }
    times[1] = t.toc();

    t.tic();
    double x_dot_y2 = 0;
    for (long p = 0; p < repeat; p++)
    {
        dotproducts[2] += inner2(n, x, y);
    }
    times[2] = t.toc();

    t.tic();
    double x_dot_y3 = 0;
    for (long p = 0; p < repeat; p++)
    {
        dotproducts[3] += inner3(n, x, y);
    }
    times[3] = t.toc();

    t.tic();
    double x_dot_y4 = 0;
    for (long p = 0; p < repeat; p++)
    {
        dotproducts[4] += inner4(n, x, y);
    }
    times[4] = t.toc();

    printf("Inner product result    Runtime\n");
    for (int i = 0; i < 5; i++)
    {
        printf("%f      %f\n", dotproducts[i], times[i]);
    }

    free(x);
    free(y);

    return 0;
}