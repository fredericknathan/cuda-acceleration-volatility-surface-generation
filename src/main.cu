#include "engine.cu"
#include "types.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>

static void generate_market_data(MarketData *md, int nK, int nT) {
  Real S = Real(100);
  md->spot = S;
  md->num_strikes = nK;
  md->num_expiries = nT;

  Real K_lo = Real(70), K_hi = Real(130);
  for (int i = 0; i < nK; ++i)
    md->strikes[i] = K_lo + (K_hi - K_lo) * Real(i) / Real(nK - 1);

  Real T_lo = Real(0.04), T_hi = Real(2.0);
  for (int j = 0; j < nT; ++j) {
    Real frac = Real(j) / Real(nT - 1);
    md->expiries[j] = T_lo * REAL_POW(T_hi / T_lo, frac);
  }

  for (int j = 0; j < nT; ++j) {
    md->risk_free_rates[j] = Real(0.05);
    md->dividend_yields[j] = Real(0.02);
  }

  for (int j = 0; j < nT; ++j) {
    Real T = md->expiries[j];
    Real r = md->risk_free_rates[j];
    Real q = md->dividend_yields[j];
    for (int i = 0; i < nK; ++i) {
      Real K = md->strikes[i];
      Real m = K / S - Real(1);
      Real sig = Real(0.20) + Real(0.15) * m * m + Real(0.03) / REAL_SQRT(T);
      Real sT = sig * REAL_SQRT(T);
      Real d1 = (REAL_LOG(S / K) + (r - q + Real(0.5) * sig * sig) * T) / sT;
      Real d2 = d1 - sT;
      Real Nd1 = Real(0.5) * REAL_ERFC(-d1 * Real(0.7071067811865476));
      Real Nd2 = Real(0.5) * REAL_ERFC(-d2 * Real(0.7071067811865476));
      Real C = S * REAL_EXP(-q * T) * Nd1 - K * REAL_EXP(-r * T) * Nd2;
      if (C < Real(1e-10))
        C = Real(1e-10);
      md->market_prices[j * nK + i] = C;
    }
  }
}

static void validate(const SurfaceOutput *out, int nK, int nT) {
  int nan_cnt = 0;
  int neg_cnt = 0;
  Real lv_min = Real(1e30);
  Real lv_max = -Real(1e30);
  Real iv_min = Real(1e30);
  Real iv_max = -Real(1e30);

  for (int idx = 0; idx < nK * nT; ++idx) {
    Real lv = out->local_vols[idx];
    Real iv = out->implied_vols[idx];
    if (lv != lv)
      ++nan_cnt;
    if (iv != iv)
      ++nan_cnt;
    if (lv < Real(0))
      ++neg_cnt;

    if (lv < lv_min)
      lv_min = lv;
    if (lv > lv_max)
      lv_max = lv;
    if (iv < iv_min)
      iv_min = iv;
    if (iv > iv_max)
      iv_max = iv;
  }

  printf("\n══════════════════════════════════════════════════════\n");
  printf("  VALIDATION REPORT  (%d strikes × %d expiries = %d cells)\n", nK, nT,
         nK * nT);
  printf("══════════════════════════════════════════════════════\n");
  printf("  NaN count           : %d\n", nan_cnt);
  printf("  Negative-variance   : %d\n", neg_cnt);
  printf("  Local-var  range    : [" REAL_FMT ", " REAL_FMT "]\n", lv_min,
         lv_max);
  printf("  Implied-vol range   : [" REAL_FMT ", " REAL_FMT "]\n", iv_min,
         iv_max);
  printf("══════════════════════════════════════════════════════\n");

  if (nan_cnt == 0 && neg_cnt == 0)
    printf("PASS\n");
  else
    printf("FAIL");
  printf("══════════════════════════════════════════════════════\n\n");
}

static void print_surface_slice(const SurfaceOutput *out, const MarketData *md,
                                int nK, int nT, int k_step, int t_step) {
  printf("Local-Variance Surface (σ²_loc)  [every %d-th strike, "
         "every %d-th expiry]\n",
         k_step, t_step);
  printf("%10s", "K \\ T");
  for (int j = 0; j < nT; j += t_step)
    printf(" %10.4f", (double)md->expiries[j]);
  printf("\n");

  for (int i = 0; i < nK; i += k_step) {
    printf("%10.2f", (double)md->strikes[i]);
    for (int j = 0; j < nT; j += t_step)
      printf(" %10.6f", (double)out->local_vols[j * nK + i]);
    printf("\n");
  }
  printf("\n");
}

int main() {
  constexpr int NK = 100;
  constexpr int NT = 30;

  printf("╔══════════════════════════════════════════════════════╗\n");
  printf("║  CUDA  Local-Volatility Surface Generator            ║\n");
  printf("║  Grid: %d strikes × %d expiries                      ║\n", NK, NT);
  printf("╚══════════════════════════════════════════════════════╝\n\n");

  Engine eng;
  eng.init(NK, NT);

  generate_market_data(eng.h_market, NK, NT);

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));

  eng.run();

  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  printf("[*] Pipeline complete in %.3f ms\n", ms);
  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));

  validate(eng.h_output, NK, NT);

  print_surface_slice(eng.h_output, eng.h_market, NK, NT, 10, 5);

  eng.cleanup();
  return 0;
}
