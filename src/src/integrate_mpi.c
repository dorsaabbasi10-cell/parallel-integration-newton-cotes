// integrate_mpi.c
// Composite Simpson rule using MPI
// Usage: mpirun -np P ./integrate_mpi a b N func_id
// N must be even

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

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

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;

    // start timing AFTER MPI_Init
    start_time = MPI_Wtime();

    // ---------- argument check ----------
    if (argc < 5)
    {
        if (rank == 0)
            fprintf(stderr, "Usage: %s a b N func_id\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    double a       = atof(argv[1]);
    double b       = atof(argv[2]);
    long   N       = atol(argv[3]);
    int    func_id = atoi(argv[4]);

    if (N <= 0 || (N % 2) != 0)
    {
        if (rank == 0)
            fprintf(stderr, "N must be positive even.\n");
        MPI_Finalize();
        return 1;
    }

    double h = (b - a) / (double)N;
    double (*f)(double) = select_func(func_id);

    double global_sum = 0.0;
    double local_sum  = 0.0;

    // endpoints contribution only once (rank 0)
    if (rank == 0)
        local_sum = f(a) + f(b);

    // ---------- distribute interior indices among ranks ----------
    long interior  = N - 1; // indices 1..N-1
    long base      = interior / size;
    long rem       = interior % size;

    long start_idx = 1 + rank * base + (rank < rem ? rank : rem);
    long count     = base + (rank < rem ? 1 : 0);
    long end_idx   = start_idx + count - 1;

    if (count <= 0)
    {
        start_idx = 1;
        end_idx   = 0;
    }

    for (long i = start_idx; i <= end_idx; ++i)
    {
        double x = a + i * h;
        if ((i & 1) == 1)
            local_sum += 4.0 * f(x);
        else
            local_sum += 2.0 * f(x);
    }

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double result = 0.0;
    if (rank == 0)
    {
        result = global_sum * h / 3.0;
    }

    // ---------- stop timing BEFORE MPI_Finalize ----------
    end_time = MPI_Wtime();

    if (rank == 0)
    {
        printf("Result = %.15g\n", result);
        printf("Time   = %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
