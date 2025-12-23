// integrate_omp.c
// Composite Simpson rule using OpenMP
// Usage: ./integrate_omp a b N num_threads func_id
// N must be even

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// --- functions to integrate ---
static double f_const(double x)      { return 1.0; }
static double f_x2(double x)         { return x * x; }
static double f_sin(double x)        { return sin(x); }
static double f_exp(double x)        { return exp(x); }
static double f_runge(double x)      { return 1.0 / (1.0 + 25.0 * x * x); }
static double f_inv_sqrt(double x)   { return 1.0 / sqrt(x); }

static double (*select_func(int id))(double)
{
    switch (id)
    {
        case 1: return f_const;
        case 2: return f_x2;
        case 3: return f_sin;
        case 4: return f_exp;
        case 5: return f_runge;
        case 6: return f_inv_sqrt;
        default: return f_sin;
    }
}

int main(int argc, char *argv[])
{
    if (argc != 6) {
        fprintf(stderr,
                "Usage: %s a b N num_threads func_id\n"
                "  a, b          : integration interval\n"
                "  N             : number of subintervals (must be even)\n"
                "  num_threads   : number of OpenMP threads\n"
                "  func_id       : 1=const, 2=x^2, 3=sin, 4=exp, 5=runge, 6=1/sqrt(x)\n",
                argv[0]);
        return 1;
    }

    double a = atof(argv[1]);
    double b = atof(argv[2]);
    long   N = atol(argv[3]);
    int    num_threads = atoi(argv[4]);
    int    func_id = atoi(argv[5]);

    if (N <= 0 || num_threads <= 0 || (N % 2) != 0) {
        fprintf(stderr, "N must be positive even, num_threads must be positive.\n");
        return 1;
    }

    double h = (b - a) / (double)N;
    double (*f)(double) = select_func(func_id);

    // Start timing
    double t_start = omp_get_wtime();

    // Initialize sum with endpoints (like your MPI and pthreads versions)
    double sum = f(a) + f(b);

    omp_set_num_threads(num_threads);

    /* Simpson's rule - parallel loop over interior points */
    #pragma omp parallel for reduction(+:sum)
    for (long i = 1; i < N; ++i) {
        double x = a + i * h;
        if (i % 2 == 1)
            sum += 4.0 * f(x);  // odd indices: coefficient 4
        else
            sum += 2.0 * f(x);  // even indices: coefficient 2
    }

    double result = sum * h / 3.0;
    double t_end = omp_get_wtime();

    printf("Result: %.15f\n", result);
    printf("Time:   %.6f seconds with %d threads\n", t_end - t_start, num_threads);

    return 0;
}