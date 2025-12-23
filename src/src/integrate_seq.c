#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double f(int func_id, double x) {
    switch (func_id) {
        case 1:  // constant
            return 1.0;
        case 2:  // x^2
            return x * x;
        case 3:  // sin(x)
            return sin(x);
        case 4:  // exp(x)
            return exp(x);
        case 5:  // Runge-type
            return 1.0 / (1.0 + 25.0 * x * x);
        case 6:  // 1/sqrt(x) on (0,1], define f(0)=0
            if (x <= 0.0) return 0.0;
            return 1.0 / sqrt(x);
        default:
            return 0.0;
    }
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s a b N func_id\n", prog);
    fprintf(stderr, "  a, b    : integration interval (double)\n");
    fprintf(stderr, "  N       : number of subintervals (integer > 0, must be even)\n");
    fprintf(stderr, "  func_id : 1=const, 2=x^2, 3=sin, 4=exp, 5=runge, 6=1/sqrt(x)\n");
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        print_usage(argv[0]);
        return 1;
    }

    double a = atof(argv[1]);
    double b = atof(argv[2]);
    long long N = atoll(argv[3]);
    int func_id = atoi(argv[4]);

    if (N <= 0) {
        fprintf(stderr, "Error: N must be positive.\n");
        return 1;
    }
    if (N % 2 != 0) {
        fprintf(stderr, "Error: N must be even for Simpson's rule.\n");
        return 1;
    }

    double h = (b - a) / (double)N;

    /* start timing */
    clock_t start = clock();

    // Simpson's rule:
    // integral ≈ h/3 * [f(x0) + f(xN) + 4 * sum f(x_odd) + 2 * sum f(x_even, interior)]
    double sum = f(func_id, a) + f(func_id, b);

    // odd indices: 1,3,5,...,N-1
    for (long long i = 1; i < N; i += 2) {
        double x = a + i * h;
        sum += 4.0 * f(func_id, x);
    }

    // even interior indices: 2,4,6,...,N-2
    for (long long i = 2; i < N; i += 2) {
        double x = a + i * h;
        sum += 2.0 * f(func_id, x);
    }

    double result = (h / 3.0) * sum;

    /* end timing */
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Result: %.15f\n", result);
    printf("Time:   %.6f seconds\n", elapsed);

   return 0;
}
