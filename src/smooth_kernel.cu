#include "types.cuh"
#include <math.h>

constexpr int PCR_BLOCK = 256;

__global__ void smooth_iv_kernel(const Real *__restrict__ iv,
                                 const Real *__restrict__ strikes, int nK,
                                 int nT, Real *__restrict__ spline_M) {
  int ej = blockIdx.x;
  if (ej >= nT)
    return;
  int tid = threadIdx.x;

  __shared__ Real sa[2][PCR_BLOCK];
  __shared__ Real sb[2][PCR_BLOCK];
  __shared__ Real sc[2][PCR_BLOCK];
  __shared__ Real sd[2][PCR_BLOCK];
  __shared__ Real s_iv[PCR_BLOCK];
  __shared__ Real s_K[PCR_BLOCK];

  if (tid < nK) {
    s_iv[tid] = iv[ej * nK + tid];
    s_K[tid] = strikes[tid];
  }
  __syncthreads();

  int m = nK - 2;
  if (m <= 0) {
    if (tid < nK)
      spline_M[ej * nK + tid] = Real(0);
    return;
  }

  int cur = 0;

  if (tid < m) {
    int p = tid;
    Real h0 = s_K[p + 1] - s_K[p];
    Real h1 = s_K[p + 2] - s_K[p + 1];

    sa[0][tid] = (p > 0) ? h0 : Real(0);
    sb[0][tid] = Real(2) * (h0 + h1);
    sc[0][tid] = (p < m - 1) ? h1 : Real(0);
    sd[0][tid] = Real(6) * ((s_iv[p + 2] - s_iv[p + 1]) / h1 -
                            (s_iv[p + 1] - s_iv[p]) / h0);
  } else {
    sa[0][tid] = Real(0);
    sb[0][tid] = Real(1);
    sc[0][tid] = Real(0);
    sd[0][tid] = Real(0);
  }
  __syncthreads();

  int n_pcr = 1;
  while (n_pcr < PCR_BLOCK)
    n_pcr <<= 1;

  for (int stride = 1; stride < n_pcr; stride <<= 1) {
    int nxt = 1 - cur;

    Real a_i = sa[cur][tid];
    Real b_i = sb[cur][tid];
    Real c_i = sc[cur][tid];
    Real d_i = sd[cur][tid];

    Real a_new = a_i, b_new = b_i, c_new = c_i, d_new = d_i;

    if (tid >= stride && tid < n_pcr) {
      int lo = tid - stride;
      Real k1 =
          (REAL_FABS(sb[cur][lo]) > Real(1e-30)) ? a_i / sb[cur][lo] : Real(0);
      a_new = -k1 * sa[cur][lo];
      b_new -= k1 * sc[cur][lo];
      d_new -= k1 * sd[cur][lo];
    }
    if (tid + stride < n_pcr) {
      int hi = tid + stride;
      Real k2 =
          (REAL_FABS(sb[cur][hi]) > Real(1e-30)) ? c_i / sb[cur][hi] : Real(0);
      c_new = -k2 * sc[cur][hi];
      b_new -= k2 * sa[cur][hi];
      d_new -= k2 * sd[cur][hi];
    }

    sa[nxt][tid] = a_new;
    sb[nxt][tid] = b_new;
    sc[nxt][tid] = c_new;
    sd[nxt][tid] = d_new;

    cur = nxt;
    __syncthreads();
  }

  int base = ej * nK;
  if (tid == 0) {
    spline_M[base] = Real(0);
  }
  if (tid == 0) {
    spline_M[base + nK - 1] = Real(0);
  }
  if (tid < m) {
    Real denom = sb[cur][tid];
    spline_M[base + tid + 1] =
        (REAL_FABS(denom) > Real(1e-30)) ? sd[cur][tid] / denom : Real(0);
  }
}

extern __constant__ Real c_rates[MAX_EXPIRIES];
extern __constant__ Real c_divs[MAX_EXPIRIES];

__device__ Real d_norm_cdf(Real x) {
  return Real(0.5) * REAL_ERFC(-x * Real(0.7071067811865476));
}

__device__ Real d_bs_call(Real S, Real K, Real T, Real r, Real q, Real sig) {
  Real sT = REAL_SQRT(T);
  Real d1 =
      (REAL_LOG(S / K) + (r - q + Real(0.5) * sig * sig) * T) / (sig * sT);
  Real d2 = d1 - sig * sT;
  return S * REAL_EXP(-q * T) * d_norm_cdf(d1) -
         K * REAL_EXP(-r * T) * d_norm_cdf(d2);
}

__global__ void dupire_kernel(const Real *__restrict__ iv,
                              const Real *__restrict__ spline_M,
                              const Real *__restrict__ strikes,
                              const Real *__restrict__ expiries, Real spot,
                              int nK, int nT, Real *__restrict__ local_var,
                              Real *__restrict__ total_var) {
  extern __shared__ Real smem[];
  Real *s_prev = smem;
  Real *s_curr = smem + nK;
  Real *s_next = smem + 2 * nK;

  int ej = blockIdx.x;
  int tid = threadIdx.x;
  if (ej >= nT)
    return;

  for (int i = tid; i < nK; i += blockDim.x) {
    s_curr[i] = iv[ej * nK + i];
    s_prev[i] = (ej > 0) ? iv[(ej - 1) * nK + i] : s_curr[i];
    s_next[i] = (ej < nT - 1) ? iv[(ej + 1) * nK + i] : s_curr[i];
  }
  __syncthreads();

  Real S = spot;
  Real T = expiries[ej];
  Real r = c_rates[ej];
  Real q = c_divs[ej];
  Real T_p = (ej > 0) ? expiries[ej - 1] : T;
  Real T_n = (ej < nT - 1) ? expiries[ej + 1] : T;
  Real r_p = (ej > 0) ? c_rates[ej - 1] : r;
  Real q_p = (ej > 0) ? c_divs[ej - 1] : q;
  Real r_n = (ej < nT - 1) ? c_rates[ej + 1] : r;
  Real q_n = (ej < nT - 1) ? c_divs[ej + 1] : q;

  for (int i = tid; i < nK; i += blockDim.x) {
    int idx = ej * nK + i;
    Real sig = s_curr[i];
    total_var[idx] = sig * sig * T;

    if (i < BOUNDARY_MARGIN || i >= nK - BOUNDARY_MARGIN) {
      local_var[idx] = Real(-1);
      continue;
    }

    Real K = strikes[i];
    Real C = d_bs_call(S, K, T, r, q, sig);

    Real dC_dT;
    if (ej == 0) {
      Real C_Tn = d_bs_call(S, K, T_n, r_n, q_n, s_next[i]);
      dC_dT = (C_Tn - C) / REAL_FMAX(T_n - T, Real(1e-10));
    } else if (ej == nT - 1) {
      Real C_Tp = d_bs_call(S, K, T_p, r_p, q_p, s_prev[i]);
      dC_dT = (C - C_Tp) / REAL_FMAX(T - T_p, Real(1e-10));
    } else {
      Real C_Tp = d_bs_call(S, K, T_p, r_p, q_p, s_prev[i]);
      Real C_Tn = d_bs_call(S, K, T_n, r_n, q_n, s_next[i]);
      dC_dT = (C_Tn - C_Tp) / REAL_FMAX(T_n - T_p, Real(1e-10));
    }

    Real Km = strikes[i - 1];
    Real Kp = strikes[i + 1];
    Real C_m = d_bs_call(S, Km, T, r, q, s_curr[i - 1]);
    Real C_p = d_bs_call(S, Kp, T, r, q, s_curr[i + 1]);
    Real dKm = K - Km;
    Real dKp = Kp - K;
    Real dC_dK = (C_p - C_m) / (dKm + dKp);
    Real d2C_dK2 = Real(2) * (C_p * dKm + C_m * dKp - C * (dKm + dKp)) /
                   (dKm * dKp * (dKm + dKp));

    dC_dT = REAL_FMAX(dC_dT, Real(0));
    d2C_dK2 = REAL_FMAX(d2C_dK2, Real(1e-12));

    Real numer = dC_dT + (r - q) * K * dC_dK + q * C;
    numer = REAL_FMAX(numer, Real(0));
    Real denom = Real(0.5) * K * K * d2C_dK2;
    denom = REAL_FMAX(denom, Real(1e-14));

    Real lv2 = numer / denom;
    lv2 = REAL_FMAX(lv2, LV_VARIANCE_FLOOR);
    lv2 = (lv2 > LV_VARIANCE_CAP) ? LV_VARIANCE_CAP : lv2;

    local_var[idx] = lv2;
  }
  __syncthreads();

  for (int i = tid; i < nK; i += blockDim.x) {
    if (i < BOUNDARY_MARGIN) {
      local_var[ej * nK + i] = local_var[ej * nK + BOUNDARY_MARGIN];
    } else if (i >= nK - BOUNDARY_MARGIN) {
      local_var[ej * nK + i] = local_var[ej * nK + nK - BOUNDARY_MARGIN - 1];
    }
  }
}

void launch_smooth_kernel(const Real *d_iv, const Real *d_strikes, int nK,
                          int nT, Real *d_spline_M, cudaStream_t s) {
  dim3 grid(nT);
  dim3 block(PCR_BLOCK);
  smooth_iv_kernel<<<grid, block, 0, s>>>(d_iv, d_strikes, nK, nT, d_spline_M);
}

void launch_dupire_kernel(const Real *d_iv, const Real *d_spline_M,
                          const Real *d_strikes, const Real *d_expiries,
                          Real spot, int nK, int nT, Real *d_local_var,
                          Real *d_total_var, cudaStream_t s) {
  int threads = 128;
  dim3 grid(nT);
  dim3 block(threads);
  size_t smem = 3 * nK * sizeof(Real);
  dupire_kernel<<<grid, block, smem, s>>>(d_iv, d_spline_M, d_strikes,
                                          d_expiries, spot, nK, nT, d_local_var,
                                          d_total_var);
}
