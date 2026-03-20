#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>


#ifdef USE_DOUBLE
using Real = double;
#define REAL_SQRT sqrt
#define REAL_LOG log
#define REAL_EXP exp
#define REAL_ERFC erfc
#define REAL_FABS fabs
#define REAL_FMAX fmax
#define REAL_POW pow
#define REAL_FMT "%.8f"
#else
using Real = float;
#define REAL_SQRT sqrtf
#define REAL_LOG logf
#define REAL_EXP expf
#define REAL_ERFC erfcf
#define REAL_FABS fabsf
#define REAL_FMAX fmaxf
#define REAL_POW powf
#define REAL_FMT "%.8f"
#endif

constexpr int MAX_STRIKES = 256;
constexpr int MAX_EXPIRIES = 64;

constexpr Real LV_VARIANCE_FLOOR = Real(1e-6);
constexpr Real LV_VARIANCE_CAP = Real(25.0);
constexpr int BOUNDARY_MARGIN = 2;

struct __align__(16) MarketData {
  Real spot;
  int num_strikes;
  int num_expiries;
  int _pad0;
  Real strikes[MAX_STRIKES];
  Real expiries[MAX_EXPIRIES];
  Real market_prices[MAX_STRIKES * MAX_EXPIRIES];
  Real risk_free_rates[MAX_EXPIRIES];
  Real dividend_yields[MAX_EXPIRIES];
};

struct __align__(16) SurfaceOutput {
  Real implied_vols[MAX_STRIKES * MAX_EXPIRIES];
  Real local_vols[MAX_STRIKES * MAX_EXPIRIES];
  Real total_variance[MAX_STRIKES * MAX_EXPIRIES];
  int num_strikes;
  int num_expiries;
  int nan_count;
  int neg_var_count;
};

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d – %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

void upload_rate_constants(const Real *rates, const Real *divs, int n);

void launch_iv_kernel(const Real *d_prices, const Real *d_strikes,
                      const Real *d_expiries, Real spot, int nK, int nT,
                      Real *d_iv, cudaStream_t s);

void launch_smooth_kernel(const Real *d_iv, const Real *d_strikes, int nK,
                          int nT, Real *d_spline_M, cudaStream_t s);

void launch_dupire_kernel(const Real *d_iv, const Real *d_spline_M,
                          const Real *d_strikes, const Real *d_expiries,
                          Real spot, int nK, int nT, Real *d_local_var,
                          Real *d_total_var, cudaStream_t s);
