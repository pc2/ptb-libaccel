/*
 * Copyright (c) 2023 Paderborn Center for Parallel Computing
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "libaccel.h"
#include <dlfcn.h>
#include <iostream>

static decltype(cuda_init) *cuda_init_fn = nullptr;
static decltype(cuda_finalize) *cuda_finalize_fn = nullptr;
static decltype(cuda_dsyevd) *cuda_dsyevd_fn = nullptr;
static decltype(cuda_ssyevd) *cuda_ssyevd_fn = nullptr;
static decltype(cuda_dgemm) *cuda_dgemm_fn = nullptr;
static decltype(cuda_sgemm) *cuda_sgemm_fn = nullptr;

int load_cuda() noexcept {
  auto handle = dlopen("libaccel_cuda.so", RTLD_NOW);

  if (!handle) {
    std::cerr << "libaccel: Could not open CUDA acceleration library."
              << std::endl;
    return 1;
  }

  int err = 0;
  cuda_init_fn =
      reinterpret_cast<decltype(cuda_init_fn)>(dlsym(handle, "cuda_init"));
  err |= !cuda_init_fn;
  cuda_finalize_fn = reinterpret_cast<decltype(cuda_finalize_fn)>(
      dlsym(handle, "cuda_finalize"));
  err |= !cuda_finalize_fn;
  cuda_dsyevd_fn =
      reinterpret_cast<decltype(cuda_dsyevd_fn)>(dlsym(handle, "cuda_dsyevd"));
  err |= !cuda_dsyevd_fn;
  cuda_ssyevd_fn =
      reinterpret_cast<decltype(cuda_ssyevd_fn)>(dlsym(handle, "cuda_ssyevd"));
  err |= !cuda_ssyevd_fn;
  cuda_dgemm_fn =
      reinterpret_cast<decltype(cuda_dgemm_fn)>(dlsym(handle, "cuda_dgemm"));
  err |= !cuda_dgemm_fn;
  cuda_sgemm_fn =
      reinterpret_cast<decltype(cuda_sgemm_fn)>(dlsym(handle, "cuda_sgemm"));
  err |= !cuda_sgemm_fn;

  if (err) {
    std::cerr << "libaccel: Error resolving required symbols." << std::endl;
    return 1;
  }

  return 0;
}

cuda_context_t cuda_init(int *err) noexcept {
  if (!cuda_init_fn) {
    std::cerr << "libaccel: CUDA library not loaded." << std::endl;
    *err = 1;
    return nullptr;
  }
  return cuda_init_fn(err);
}

void cuda_finalize(cuda_context_t ctx, int *err) noexcept {
  if (!cuda_finalize_fn) {
    std::cerr << "libaccel: CUDA library not loaded." << std::endl;
    *err = 1;
  }
  return cuda_finalize_fn(ctx, err);
}

void cuda_dsyevd(cuda_context_t ctx, int64_t n, double *A, int64_t lda,
                 double *W, int *err) noexcept {
  if (!cuda_dsyevd_fn) {
    std::cerr << "libaccel: CUDA library not loaded." << std::endl;
    *err = 1;
  }
  return cuda_dsyevd_fn(ctx, n, A, lda, W, err);
}

void cuda_ssyevd(cuda_context_t ctx, int64_t n, float *A, int64_t lda, float *W,
                 int *err) noexcept {
  if (!cuda_ssyevd_fn) {
    std::cerr << "libaccel: CUDA library not loaded." << std::endl;
    *err = 1;
  }
  return cuda_ssyevd_fn(ctx, n, A, lda, W, err);
}

void cuda_dgemm(cuda_context_t ctx, char trans_a, char trans_b, int64_t m,
                int64_t n, int64_t k, double alpha, const double *A,
                int64_t lda, const double *B, int64_t ldb, double beta,
                double *C, int64_t ldc, int *err) noexcept {
  if (!cuda_dgemm_fn) {
    std::cerr << "libaccel: CUDA library not loaded." << std::endl;
    *err = 1;
  }
  return cuda_dgemm_fn(ctx, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc, err);
}

void cuda_sgemm(cuda_context_t ctx, char trans_a, char trans_b, int64_t m,
                int64_t n, int64_t k, float alpha, const float *A, int64_t lda,
                const float *B, int64_t ldb, float beta, float *C, int64_t ldc,
                int *err) noexcept {
  if (!cuda_sgemm_fn) {
    std::cerr << "libaccel: CUDA library not loaded." << std::endl;
    *err = 1;
  }
  return cuda_sgemm_fn(ctx, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc, err);
}