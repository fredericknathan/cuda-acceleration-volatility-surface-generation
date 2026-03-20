#include "types.cuh"
#include <cstdio>
#include <cstring>

struct Engine {
  MarketData *h_market;
  SurfaceOutput *h_output;

  Real *d_prices;
  Real *d_strikes;
  Real *d_expiries;
  Real *d_iv;
  Real *d_spline_M;
  Real *d_local_var;
  Real *d_total_var;

  cudaStream_t stream_compute;
  cudaStream_t stream_transfer;

  int nK, nT, total;

  void init(int num_strikes, int num_expiries) {
    nK = num_strikes;
    nT = num_expiries;
    total = nK * nT;

    CUDA_CHECK(
        cudaHostAlloc(&h_market, sizeof(MarketData), cudaHostAllocMapped));
    CUDA_CHECK(
        cudaHostAlloc(&h_output, sizeof(SurfaceOutput), cudaHostAllocMapped));

    CUDA_CHECK(cudaMalloc(&d_prices, total * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_strikes, nK * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_expiries, nT * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_iv, total * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_spline_M, total * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_local_var, total * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_total_var, total * sizeof(Real)));

    CUDA_CHECK(cudaStreamCreate(&stream_compute));
    CUDA_CHECK(cudaStreamCreate(&stream_transfer));
  }

  void run() {
    upload_rate_constants(h_market->risk_free_rates, h_market->dividend_yields,
                          nT);

    CUDA_CHECK(cudaMemcpyAsync(d_prices, h_market->market_prices,
                               total * sizeof(Real), cudaMemcpyHostToDevice,
                               stream_transfer));
    CUDA_CHECK(cudaMemcpyAsync(d_strikes, h_market->strikes, nK * sizeof(Real),
                               cudaMemcpyHostToDevice, stream_transfer));
    CUDA_CHECK(cudaMemcpyAsync(d_expiries, h_market->expiries,
                               nT * sizeof(Real), cudaMemcpyHostToDevice,
                               stream_transfer));

    cudaEvent_t xfer_done;
    CUDA_CHECK(cudaEventCreate(&xfer_done));
    CUDA_CHECK(cudaEventRecord(xfer_done, stream_transfer));
    CUDA_CHECK(cudaStreamWaitEvent(stream_compute, xfer_done, 0));

    launch_iv_kernel(d_prices, d_strikes, d_expiries, h_market->spot, nK, nT,
                     d_iv, stream_compute);

    launch_smooth_kernel(d_iv, d_strikes, nK, nT, d_spline_M, stream_compute);

    launch_dupire_kernel(d_iv, d_spline_M, d_strikes, d_expiries,
                         h_market->spot, nK, nT, d_local_var, d_total_var,
                         stream_compute);

    CUDA_CHECK(cudaMemcpyAsync(h_output->implied_vols, d_iv,
                               total * sizeof(Real), cudaMemcpyDeviceToHost,
                               stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(h_output->local_vols, d_local_var,
                               total * sizeof(Real), cudaMemcpyDeviceToHost,
                               stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(h_output->total_variance, d_total_var,
                               total * sizeof(Real), cudaMemcpyDeviceToHost,
                               stream_compute));

    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    CUDA_CHECK(cudaEventDestroy(xfer_done));

    h_output->num_strikes = nK;
    h_output->num_expiries = nT;
  }

  void cleanup() {
    cudaFree(d_prices);
    cudaFree(d_strikes);
    cudaFree(d_expiries);
    cudaFree(d_iv);
    cudaFree(d_spline_M);
    cudaFree(d_local_var);
    cudaFree(d_total_var);
    cudaFreeHost(h_market);
    cudaFreeHost(h_output);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_transfer);
  }
};
