// integrate_pthreads.c
// Composite Simpson rule using POSIX threads (pthreads)
// Usage: ./integrate_pthreads a b N func_id num_threads
// N must be even
//
// func_id:
//   1 = 1
//   2 = x^2
//   3 = sin(x)
//   4 = exp(x)
//   5 = 1/(1+25x^2)  (Runge)
//   6 = 1/sqrt(x)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

// --- functions to integrate ---
static double f_const(double x)    { return 1.0; }
static double f_x2(double x)       { return x * x; }
static double f_sin(double x)      { return sin(x); }
static double f_exp(double x)      { return exp(x); }
static double f_runge(double x)    { return 1.0 / (1.0 + 25.0 * x * x); }
static double f_inv_sqrt(double x) { return 1.0 / sqrt(x); }

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

// --- wall-clock timer (seconds) ---
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// --- thread data ---
typedef struct
{
    long   start, end;   // indices i in [start, end] (1..N-1)
    double a, h;         // interval start and step size
    int    func_id;      // function id
    double local_sum;    // partial sum
} thread_arg_t;

static void *worker(void *arg)
{
    thread_arg_t *t = (thread_arg_t *)arg;
    double (*f)(double) = select_func(t->func_id);

    double local = 0.0;
    for (long i = t->start; i <= t->end; ++i)
    {
        double x = t->a + (double)i * t->h;
        if (i & 1L) local += 4.0 * f(x);
        else        local += 2.0 * f(x);
    }

    t->local_sum = local;
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 6)
    {
        fprintf(stderr, "Usage: %s a b N func_id num_threads\n", argv[0]);
        return 1;
    }

    double a = atof(argv[1]);
    double b = atof(argv[2]);
    long   N = atol(argv[3]);
    int    func_id = atoi(argv[4]);
    int    num_threads = atoi(argv[5]);

    if (N <= 0 || (N % 2) != 0)
    {
        fprintf(stderr, "N must be positive even.\n");
        return 1;
    }
    if (num_threads <= 0)
    {
        fprintf(stderr, "num_threads must be positive.\n");
        return 1;
    }

    // For func_id=6 (1/sqrt(x)), avoid x=0 singularity
    if (func_id == 6 && a <= 0.0)
    {
        fprintf(stderr, "For func_id=6 (1/sqrt(x)), a must be > 0.\n");
        return 1;
    }

    double h = (b - a) / (double)N;
    double (*f)(double) = select_func(func_id);

    // Start wall time measurement for the integration work
    double t0 = now_sec();

    // Endpoint contribution once
    double sum = f(a) + f(b);

    long interior = N - 1; // indices 1..N-1
    if (interior <= 0)
    {
        double result = sum * h / 3.0;
        double t1 = now_sec();
        printf("Result: %.15g\n", result);
        printf("Time  : %.6f seconds\n", t1 - t0);
        printf("Threads used: %d\n", num_threads);
        return 0;
    }

    if (num_threads > (int)interior)
        num_threads = (int)interior;

    pthread_t    *threads = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)num_threads);
    thread_arg_t *args    = (thread_arg_t *)malloc(sizeof(thread_arg_t) * (size_t)num_threads);
    if (!threads || !args)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        free(threads);
        free(args);
        return 1;
    }

    long base = interior / num_threads;
    long rem  = interior % num_threads;
    long current = 1;

    // Create threads with contiguous blocks
    int created = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        long start = current;
        long count = base + (t < rem ? 1 : 0);
        long end   = start + count - 1;

        args[t].start     = start;
        args[t].end       = end;
        args[t].a         = a;
        args[t].h         = h;
        args[t].func_id   = func_id;
        args[t].local_sum = 0.0;

        if (count <= 0) {
            args[t].start = 1;
            args[t].end   = 0;
        }

        if (pthread_create(&threads[t], NULL, worker, &args[t]) != 0)
        {
            fprintf(stderr, "Error creating thread %d\n", t);
            break;
        }
        created++;

        current = end + 1;
    }

    // Join created threads and accumulate
    for (int t = 0; t < created; ++t)
    {
        pthread_join(threads[t], NULL);
        sum += args[t].local_sum;
    }

    double result = sum * h / 3.0;

    double t1 = now_sec();

    // Print in a parser-friendly, consistent format
    printf("Result: %.15g\n", result);
    printf("Time  : %.6f seconds\n", t1 - t0);
    printf("Threads used: %d\n", created);

    free(threads);
    free(args);

    return 0;
}
