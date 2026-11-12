"""One runnable Sparse SPD example from the model used in Example 3."""

import argparse

import numpy as np
from scipy import sparse

from NPDolib import stochastic_npdo


def make_problem(n, k):
    rng = np.random.default_rng(np.random.SeedSequence([0, 1]))
    R = sparse.random(n, n, density=4 / n, format="csr", random_state=rng,
                      data_rvs=lambda size: rng.uniform(-1, 1, size))
    W = (R + R.T).tocsr()
    diagonal = np.asarray(abs(W).sum(axis=1)).ravel() + 0.25
    A = (W + sparse.diags(diagonal, format="csr")).tocsr()
    B = 0.03 * rng.standard_normal((n, k)) / np.sqrt(n)
    initial_rng = np.random.default_rng(np.random.SeedSequence([0, 2]))
    P0 = np.linalg.qr(initial_rng.standard_normal((n, k)), mode="reduced")[0]
    return A, B, P0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-size", action="store_true",
                        help="use (n,k,d)=(1024,512,64), instead of (128,64,8)")
    parser.add_argument("--max-sweeps", type=int, default=1000)
    args = parser.parse_args()
    if args.max_sweeps < 1:
        parser.error("--max-sweeps must be positive")
    n, k, d = (1024, 512, 64) if args.paper_size else (128, 64, 8)
    q = (k + d - 1) // d
    A, B, P0 = make_problem(n, k)
    history = []

    def record(i, residual):
        history.append((i, residual))
        if i == 0 or (i // q) % 100 == 0:
            print(f"sweep={i // q}, normalized KKT={residual:.6e}", flush=True)

    print(f"Sparse SPD: n={n}, k={k}, d={d}", flush=True)
    seed = np.random.SeedSequence([20261025, 1, 0, d])
    P = stochastic_npdo(A, B, P0, d, tol=1e-4,
                        max_iter=args.max_sweeps * q, rng=seed, callback=record)
    updates, residual = history[-1]
    print(f"Completed {updates // q} sweeps ({updates} block updates)")
    print(f"Final normalized KKT: {residual:.6e}")
    print(f"Orthogonality error: {np.linalg.norm(P.T @ P - np.eye(k), 'fro'):.6e}")
    print("Tolerance reached" if residual <= 1e-4 else "Sweep limit reached before tolerance")


if __name__ == "__main__":
    main()
