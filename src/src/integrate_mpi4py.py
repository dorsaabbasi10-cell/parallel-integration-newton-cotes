from mpi4py import MPI
import sys
import math

def f(func_id, x):
    if func_id == 1:      # constant
        return 1.0
    elif func_id == 2:    # x^2
        return x * x
    elif func_id == 3:    # sin(x)
        return math.sin(x)
    elif func_id == 4:    # exp(x)
        return math.exp(x)
    elif func_id == 5:    # Runge-type 1/(1+25x^2)
        return 1.0 / (1.0 + 25.0 * x * x)
    elif func_id == 6:    # 1/sqrt(x) on (0,1], define f(0)=0
        if x <= 0.0:
            return 0.0
        return 1.0 / math.sqrt(x)
    else:
        return 0.0

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ----- parse arguments on rank 0 -----
    if rank == 0:
        if len(sys.argv) != 5:
            print("Usage: python integrate_mpi4py.py a b N func_id")
            print("  a, b    : integration limits (float)")
            print("  N       : number of subintervals (int > 0)")
            print("  func_id : 1=const, 2=x^2, 3=sin, 4=exp, 5=runge, 6=1/sqrt(x)")
            comm.Abort(1)

        a = float(sys.argv[1])
        b = float(sys.argv[2])
        N = int(sys.argv[3])
        func_id = int(sys.argv[4])
    else:
        a = 0.0
        b = 0.0
        N = 0
        func_id = 0

    # broadcast parameters
    a = comm.bcast(a, root=0)
    b = comm.bcast(b, root=0)
    N = comm.bcast(N, root=0)
    func_id = comm.bcast(func_id, root=0)

    if N <= 0:
        if rank == 0:
            print("Error: N must be positive.")
        comm.Abort(1)

    h = (b - a) / float(N)

    # ===== TIMING START =====
    t0 = MPI.Wtime()

    # distribute interior points 1..N-1 across ranks
    interior_points = N - 1 if N > 1 else 0
    base = interior_points // size
    rem = interior_points % size

    if rank < rem:
        start_idx = 1 + rank * (base + 1)
        end_idx = start_idx + (base + 1)
    else:
        start_idx = 1 + rem * (base + 1) + (rank - rem) * base
        end_idx = start_idx + base

    local_sum = 0.0
    for i in range(start_idx, end_idx):
        x = a + i * h
        local_sum += f(func_id, x)

    # reduce interior sums
    total_interior = comm.reduce(local_sum, op=MPI.SUM, root=0)

    if rank == 0:
        sum_all = 0.0
        sum_all += 0.5 * f(func_id, a)
        sum_all += 0.5 * f(func_id, b)
        if N > 1:
            sum_all += total_interior

        result = sum_all * h

        # ===== TIMING END =====
        t1 = MPI.Wtime()
        elapsed = t1 - t0

        print(f"Result: {result:.15f}")
        print(f"Time:   {elapsed:.6f} seconds")

if __name__ == "__main__":
    main()
