# NPDo algorithms and one runnable example

Supplementary source code for **A Stochastic NPDo Approach for Quadratic
Trace Maximization on the Stiefel Manifold**.

This small package contains only the authors' NPDo implementation and one
synthetic Sparse SPD example. It does not contain comparison algorithms,
Pymanopt, plotting scripts, saved running results, or benchmark data.
NumPy and SciPy are installed as dependencies, not redistributed source.

## Files

- `NPDolib.py`: projected NPDo (Algorithm 1) and stochastic NPDo
  (Algorithm 2), both using the two-SVD polar update.
- `example.py`: generates one Sparse SPD problem and runs stochastic NPDo.
- `requirements.txt`: the two Python dependencies.
- `README.md`: this guide.

## Run

From this directory:

```text
python -m pip install -r requirements.txt
python example.py
```

If your system uses `python3`, replace `python` with `python3`.
The default small demonstration uses `(n,k,d)=(128,64,8)` so it can be
tried without launching the full experiment. To use the corresponding
paper-size Sparse SPD configuration instead:

```text
python example.py --paper-size
```

This selects `(n,k,d)=(1024,512,64)` and may take several minutes or longer,
depending on the machine and numerical libraries. Both modes use the same
matrix construction, linear-term strength `0.03`, and normalized KKT
tolerance `1e-4`. The default upper limit is 1000 equivalent sweeps.
For a short execution check, use `python example.py --max-sweeps 2`.

One equivalent sweep is `ceil(k/d)` independently sampled block updates.
The program prints the final residual, update count, orthogonality error,
and whether it reached the tolerance. It does not write result files.
No network access or previously saved input matrices are needed.

## Model and library interface

The example constructs a sparse matrix `R` of density `4/n`, sets
`W=R+R.T`, and uses

```text
A = W + diag(abs(W) @ ones(n) + 0.25)
B = 0.03 * Gaussian(n,k) / sqrt(n)
```

The matrix `A` is symmetric positive definite. The library maximizes
`trace(P.T @ A @ P) + 2*trace(P.T @ B)` under `P.T @ P = I`.
Initial points must satisfy the constraints and `A` must be positive
semidefinite. `A` may be dense or sparse; `B` is a dense array.

```python
from NPDolib import projected_npdo, stochastic_npdo

P = stochastic_npdo(A, B, P0, d=64, tol=1e-4, max_iter=8000, rng=0)
Y = projected_npdo(A, B_block, Z, Y0, tol=1e-8, max_iter=1000)
```

In the second call, `Z` contains orthonormal excluded directions and
`Y0.T @ Y0 = I`, `Z.T @ Y0 = 0`. An empty `n x 0` `Z` gives full NPDo.
`stochastic_npdo` requires `2 <= d <= k`; its `max_iter` counts **block
updates**, whereas `projected_npdo` counts complete projected NPDo updates.
Both return only the final matrix. An optional `callback(i, residual)`
receives the normalized KKT residual at each stopping check.

The stochastic routine uses the uncached sparse-problem implementation:
it forms each selected gradient directly, with a global stopping check
once per equivalent sweep. This is a runnable method demonstration, not
the full timing-comparison framework or an archive of the paper's results.
