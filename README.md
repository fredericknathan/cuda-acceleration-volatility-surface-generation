# CUDA Local-Volatility Surface Generator

A GPU-accelerated local volatility surface generator solving the **Dupire equation** for high-frequency options market making. Built with CUDA C++17, targeting Google Colab.

## Features

- **Newton-Raphson + bisection fallback** for implied volatility inversion via `erfcf`-based CDF
- **Parallel Cyclic Reduction (PCR)** tridiagonal solver for constrained cubic splines in `__shared__` memory
- **Dupire PDE** local variance computation with `__shared__`-memory tiling (3 expiry rows)
- **No-arbitrage enforcement**: calendar spread (`∂C/∂T ≥ 0`), butterfly (`∂²C/∂K² ≥ 0`), variance floor/cap
- **Zero-copy I/O** via `cudaHostAlloc` mapped pinned memory
- **Dual `cudaStream_t`** for concurrent H↔D transfer and kernel execution
- **Compile-time precision switch** between `float` (speed) and `double` (precision)

How to run:

```bash
!mkdir -p include src
!nvcc -rdc=true -O3 -use_fast_math -std=c++17 -Iinclude \
      src/iv_kernel.cu src/smooth_kernel.cu src/main.cu \
      -o local_vol_engine
!./local_vol_engine
```

For double precision, add `-DUSE_DOUBLE` to the `nvcc` command.

## Expected Output

```
╔══════════════════════════════════════════════════════╗
║  CUDA Local-Volatility Surface Generator     ║
║  Grid: 100 strikes × 30 expiries                    ║
╚══════════════════════════════════════════════════════╝

[*] Pipeline complete in ~130 ms

══════════════════════════════════════════════════════
  VALIDATION REPORT  (100 strikes × 30 expiries = 3000 cells)
══════════════════════════════════════════════════════
  NaN count           : 0
  Negative-variance   : 0
  PASS
══════════════════════════════════════════════════════
```

## Key Constants

| Constant              | Value | Purpose                                         |
| --------------------- | ----- | ----------------------------------------------- |
| `MAX_STRIKES`       | 256   | Max strikes per surface                         |
| `MAX_EXPIRIES`      | 64    | Max expiries per surface                        |
| `LV_VARIANCE_FLOOR` | 1e-6  | Minimum local variance (prevents negative)      |
| `LV_VARIANCE_CAP`   | 25.0  | Maximum local variance (~500% vol ceiling)      |
| `BOUNDARY_MARGIN`   | 2     | Edge strikes copied from nearest interior point |
| `PCR_BLOCK`         | 256   | Threads per block for PCR solver                |

## Mathematical References

- **Black-Scholes (1973)**: `C = S·e^(-qT)·N(d₁) − K·e^(-rT)·N(d₂)`
- **Dupire (1994)**: `σ²_loc = [∂C/∂T + (r−q)K·∂C/∂K + qC] / [½K²·∂²C/∂K²]`
- **PCR**: Hockney (1965), parallel tridiagonal solve in O(log n) steps
