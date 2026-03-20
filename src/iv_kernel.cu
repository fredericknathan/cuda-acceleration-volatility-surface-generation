#include "types.cuh"
#include <math.h>

__constant__ Real c_rates[MAX_EXPIRIES];
__constant__ Real c_divs[MAX_EXPIRIES];

void upload_rate_constants(const Real *rates, const Real *divs, int n) {
  CUDA_CHECK(cudaMemcpyToSymbol(c_rates, rates, n * sizeof(Real)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_divs, divs, n * sizeof(Real)));
}

__device__ __forceinline__ Real norm_cdf(Real x) {
  return Real(0.5) * REAL_ERFC(-x * Real(0.7071067811865476));
}

__device__ __forceinline__ Real norm_pdf(Real x) {
  return Real(0.3989422804014327) * REAL_EXP(Real(-0.5) * x * x);
}

__device__ Real bs_call_price(Real S, Real K, Real T, Real r, Real q,
                              Real sigma) {
  Real sT = REAL_SQRT(T);
  Real d1 = (REAL_LOG(S / K) + (r - q + Real(0.5) * sigma * sigma) * T) /
            (sigma * sT);
  Real d2 = d1 - sigma * sT;
  return S * REAL_EXP(-q * T) * norm_cdf(d1) -
         K * REAL_EXP(-r * T) * norm_cdf(d2);
}

__device__ Real bs_vega(Real S, Real K, Real T, Real r, Real q, Real sigma) {
  Real sT = REAL_SQRT(T);
  Real d1 = (REAL_LOG(S / K) + (r - q + Real(0.5) * sigma * sigma) * T) /
            (sigma * sT);
  return S * REAL_EXP(-q * T) * sT * norm_pdf(d1);
}

__global__ void iv_kernel(const Real *__restrict__ prices,
                          const Real *__restrict__ strikes,
                          const Real *__restrict__ expiries, Real spot, int nK,
                          int nT, Real *__restrict__ iv_out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nK * nT)
    return;

  int j = idx / nK;
  int i = idx % nK;

  Real S = spot;
  Real K = strikes[i];
  Real T = expiries[j];
  Real r = c_rates[j];
  Real q = c_divs[j];
  Real C_mkt = prices[idx];

  Real intr = REAL_FMAX(S * REAL_EXP(-q * T) - K * REAL_EXP(-r * T), Real(0));
  if (C_mkt <= intr + Real(1e-10)) {
    iv_out[idx] = Real(0.01);
    return;
  }

  Real sigma = Real(0.3);
  Real lo = Real(0.001);
  Real hi = Real(5.0);
  const int MAXITER = 64;
  const Real TOL = Real(1e-8);

  for (int it = 0; it < MAXITER; ++it) {
    Real C = bs_call_price(S, K, T, r, q, sigma);
    Real df = C - C_mkt;
    if (REAL_FABS(df) < TOL)
      break;

    Real vg = bs_vega(S, K, T, r, q, sigma);
    if (vg > Real(1e-12)) {
      Real s_new = sigma - df / vg;
      if (s_new > lo && s_new < hi) {
        if (df > Real(0))
          hi = sigma;
        else
          lo = sigma;
        sigma = s_new;
        continue;
      }
    }
    if (df > Real(0))
      hi = sigma;
    else
      lo = sigma;
    sigma = Real(0.5) * (lo + hi);
  }
  iv_out[idx] = sigma;
}

void launch_iv_kernel(const Real *d_prices, const Real *d_strikes,
                      const Real *d_expiries, Real spot, int nK, int nT,
                      Real *d_iv, cudaStream_t s) {
  int total = nK * nT;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  iv_kernel<<<blocks, threads, 0, s>>>(d_prices, d_strikes, d_expiries, spot,
                                       nK, nT, d_iv);
}
