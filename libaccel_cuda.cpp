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

#include "libaccel_cuda.h"
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <ratio>
#include <stdexcept>
#include <type_traits>

constexpr bool TIMINGS = true;

constexpr uint64_t CTX_MAGIC = 0x80661b813a70c302;

struct context {
  uint64_t magic = CTX_MAGIC;
  cublasHandle_t cublasH = NULL;
  cusolverDnHandle_t cusolverH = NULL;
  cudaStream_t stream = NULL;
  int8_t valid = false;
};

static inline struct context *restore_ctx(cuda_context_t ctx) {
  auto context = reinterpret_cast<struct context *>(ctx);
  if (context->magic != CTX_MAGIC || !context->valid) {
    return nullptr;
  }
  return context;
}

static inline void throwOnErr(cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error("libaccel_cuda: CUDA Error");
  }
}

static inline void throwOnErr(cusolverStatus_t err) {
  if (err != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error("libaccel_cuda: CUSOLVER Error");
  }
}

static inline void throwOnErr(cublasStatus_t err) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("libaccel_cuda: CUBLAS Error");
  }
}

static inline cublasOperation_t cublasOpFromChr(char c) {
  switch (c) {
  case 'N':
  case 'n':
    return CUBLAS_OP_N;
  case 'T':
  case 't':
    return CUBLAS_OP_T;
  default:
    throw std::runtime_error("Invalid character passed to cublasOpFromChr()");
  }
}

template <typename T> static constexpr cudaDataType_t get_cuda_type() {

  static_assert(
      std::disjunction_v<std::is_same<T, double>, std::is_same<T, float>>,
      "Unsupported type given to get_cuda_type()");

  if (std::is_same_v<T, double>) {
    return CUDA_R_64F;
  }
  if (std::is_same_v<T, float>) {
    return CUDA_R_32F;
  }
}

template <typename T>
static void printMatrix_colMajor(size_t m, size_t n, T *A) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      std::cout << A[j * m + i] << " ";
    }
    std::cout << std::endl;
  }
}

template <typename data_type>
static inline void Xgemm(cuda_context_t ctx, char trans_a, char trans_b,
                         int64_t m, int64_t n, int64_t k, data_type alpha,
                         const data_type *A, int64_t lda, const data_type *B,
                         int64_t ldb, data_type beta, data_type *C, int64_t ldc,
                         int *err) noexcept {

  static_assert(std::disjunction_v<std::is_same<data_type, double>,
                                   std::is_same<data_type, float>>,
                "Unsupported type given to Xgemm()");

  cublasOperation_t transa;
  cublasOperation_t transb;

  data_type *d_A = nullptr;
  data_type *d_B = nullptr;
  data_type *d_C = nullptr;

  decltype(std::chrono::high_resolution_clock::now()) tStart_total, tEnd_total;
  cudaEvent_t eStart = nullptr;
  cudaEvent_t eEnd = nullptr;
  float eTime;

  /* print input data
  std::cout << trans_a << " " << trans_b << " " << m << " " << n << " " << k
            << " " << alpha << " " << beta << std::endl;
  std::cout << "----" << std::endl;
  printMatrix_colMajor(m, k, A);
  std::cout << "----" << std::endl;
  printMatrix_colMajor(k, n, B);
  std::cout << "----" << std::endl;
  printMatrix_colMajor(m, n, C);
  std::cout << "----" << std::endl;
  */

  *err = 0;

  auto context = restore_ctx(ctx);
  if (!context) {
    std::cerr << "libaccel_cuda: Invalid context" << std::endl;
    *err = 1;
    return;
  }

  auto &cublasH = context->cublasH;
  auto &stream = context->stream;

  if (TIMINGS) {
    tStart_total = std::chrono::high_resolution_clock::now();
  }

  try {

    transa = cublasOpFromChr(trans_a);
    transb = cublasOpFromChr(trans_b);

    throwOnErr(cudaMalloc(reinterpret_cast<void **>(&d_A),
                          sizeof(data_type) * lda * k));
    throwOnErr(cudaMalloc(reinterpret_cast<void **>(&d_B),
                          sizeof(data_type) * ldb * n));
    throwOnErr(cudaMalloc(reinterpret_cast<void **>(&d_C),
                          sizeof(data_type) * ldc * n));

    if (TIMINGS) {
      throwOnErr(cudaStreamSynchronize(stream));
      throwOnErr(cudaEventCreate(&eStart));
      throwOnErr(cudaEventCreate(&eEnd));
      throwOnErr(cudaEventRecord(eStart, stream));
    }

    throwOnErr(cudaMemcpyAsync(d_A, A, sizeof(data_type) * lda * k,
                               cudaMemcpyHostToDevice, stream));
    throwOnErr(cudaMemcpyAsync(d_B, B, sizeof(data_type) * ldb * n,
                               cudaMemcpyHostToDevice, stream));

    if (beta != 0.0) {
      throwOnErr(cudaMemcpyAsync(d_C, C, sizeof(data_type) * ldc * n,
                                 cudaMemcpyHostToDevice, stream));
    }

    if (TIMINGS) {
      throwOnErr(cudaEventRecord(eEnd, stream));
      throwOnErr(cudaStreamSynchronize(stream));
      cudaEventElapsedTime(&eTime, eStart, eEnd);
      std::cerr << "...CUBLAS Host To Device: " << eTime << " ms" << std::endl;
      throwOnErr(cudaEventRecord(eStart, stream));
    }

    if constexpr (std::is_same_v<data_type, double>) {
      throwOnErr(cublasDgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda,
                             d_B, ldb, &beta, d_C, ldc));
    }

    if constexpr (std::is_same_v<data_type, float>) {
      throwOnErr(cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda,
                             d_B, ldb, &beta, d_C, ldc));
    }

    if (TIMINGS) {
      throwOnErr(cudaEventRecord(eEnd, stream));
      throwOnErr(cudaStreamSynchronize(stream));
      cudaEventElapsedTime(&eTime, eStart, eEnd);
      std::cerr << "...CUBLAS GEMM: " << eTime << " ms" << std::endl;
      throwOnErr(cudaEventRecord(eStart, stream));
    }

    throwOnErr(cudaMemcpyAsync(C, d_C, sizeof(data_type) * ldc * n,
                               cudaMemcpyDeviceToHost, stream));

    throwOnErr(cudaStreamSynchronize(stream));

    if (TIMINGS) {
      throwOnErr(cudaEventRecord(eEnd, stream));
      throwOnErr(cudaStreamSynchronize(stream));
      cudaEventElapsedTime(&eTime, eStart, eEnd);
      std::cerr << "...CUBLAS Device to Host: " << eTime << " ms" << std::endl;
      throwOnErr(cudaEventRecord(eStart, stream));
    }

  } catch (std::runtime_error &ex) {
    std::cerr << ex.what() << std::endl;
    *err = 1;
  }

  if (eStart)
    *err |= cudaEventDestroy(eStart);

  if (eEnd)
    *err |= cudaEventDestroy(eEnd);

  if (d_A)
    *err |= cudaFree(d_A);

  if (d_B)
    *err |= cudaFree(d_B);

  if (d_C)
    *err |= cudaFree(d_C);

  if (TIMINGS) {
    tEnd_total = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        tEnd_total - tStart_total);
    std::cerr << "...Total time spent in C++ function: " << duration.count()
              << " ms" << std::endl;
  }
}

template <typename data_type>
static inline void Xsyevd(cuda_context_t ctx, int64_t n, data_type *A,
                          int64_t lda, data_type *W, int *err) noexcept {
  cudaDataType_t cuda_type = get_cuda_type<data_type>();

  data_type *d_A = nullptr;
  data_type *d_W = nullptr;
  int *d_info = nullptr;
  int info = 0;
  size_t d_lwork = 0;
  void *d_work = nullptr;
  size_t h_lwork = 0;
  void *h_work = nullptr;

  decltype(std::chrono::high_resolution_clock::now()) tStart_total, tEnd_total;
  cudaEvent_t eStart = nullptr;
  cudaEvent_t eEnd = nullptr;
  float eTime;

  *err = 0;

  auto context = restore_ctx(ctx);
  if (!context) {
    std::cerr << "libaccel_cuda: Invalid context" << std::endl;
    *err = 1;
    return;
  }

  auto &cusolverH = context->cusolverH;
  auto &stream = context->stream;

  cusolverEigMode_t jobz =
      CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  if (TIMINGS) {
    tStart_total = std::chrono::high_resolution_clock::now();
  }

  try {
    throwOnErr(cudaMalloc(reinterpret_cast<void **>(&d_A),
                          sizeof(data_type) * n * lda));
    throwOnErr(
        cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(data_type) * n));
    throwOnErr(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    if (TIMINGS) {
      throwOnErr(cudaStreamSynchronize(stream));
      throwOnErr(cudaEventCreate(&eStart));
      throwOnErr(cudaEventCreate(&eEnd));
      throwOnErr(cudaEventRecord(eStart, stream));
    }

    throwOnErr(cudaMemcpyAsync(d_A, A, sizeof(data_type) * n * lda,
                               cudaMemcpyHostToDevice, stream));

    if (TIMINGS) {
      throwOnErr(cudaEventRecord(eEnd, stream));
      throwOnErr(cudaStreamSynchronize(stream));
      cudaEventElapsedTime(&eTime, eStart, eEnd);
      std::cerr << "...CUSOLVER Host To Device: " << eTime << " ms"
                << std::endl;
    }

    throwOnErr(cusolverDnXsyevd_bufferSize(cusolverH, NULL, jobz, uplo, n,
                                           cuda_type, d_A, lda, cuda_type, d_W,
                                           cuda_type, &d_lwork, &h_lwork));

    throwOnErr(cudaMalloc(reinterpret_cast<void **>(&d_work), d_lwork));

    if (TIMINGS) {
      throwOnErr(cudaStreamSynchronize(stream));
      throwOnErr(cudaEventRecord(eStart, stream));
    }

    throwOnErr(cusolverDnXsyevd(cusolverH, NULL, jobz, uplo, n, cuda_type, d_A,
                                lda, cuda_type, d_W, cuda_type, d_work, d_lwork,
                                h_work, h_lwork, d_info));

    if (TIMINGS) {
      throwOnErr(cudaEventRecord(eEnd, stream));
      throwOnErr(cudaStreamSynchronize(stream));
      cudaEventElapsedTime(&eTime, eStart, eEnd);
      std::cerr << "...CUSOLVER SYEVD: " << eTime << " ms" << std::endl;
      throwOnErr(cudaEventRecord(eStart, stream));
    }

    throwOnErr(cudaMemcpyAsync(A, d_A, sizeof(data_type) * n * lda,
                               cudaMemcpyDeviceToHost, stream));
    throwOnErr(cudaMemcpyAsync(W, d_W, sizeof(data_type) * n,
                               cudaMemcpyDeviceToHost, stream));
    throwOnErr(cudaMemcpyAsync(&info, d_info, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));

    throwOnErr(cudaStreamSynchronize(stream));

    if (TIMINGS) {
      throwOnErr(cudaEventRecord(eEnd, stream));
      throwOnErr(cudaStreamSynchronize(stream));
      cudaEventElapsedTime(&eTime, eStart, eEnd);
      std::cerr << "...CUSOLVER Device To Host: " << eTime << " ms"
                << std::endl;
      throwOnErr(cudaEventRecord(eStart, stream));
    }

  } catch (std::runtime_error &ex) {
    std::cerr << ex.what() << std::endl;
    *err = 1;
  }

  if (eStart)
    *err |= cudaEventDestroy(eStart);

  if (eEnd)
    *err |= cudaEventDestroy(eEnd);

  if (d_A)
    *err |= cudaFree(d_A);

  if (d_W)
    *err |= cudaFree(d_W);

  if (d_info)
    *err |= cudaFree(d_info);

  if (d_work)
    *err |= cudaFree(d_work);

  if (TIMINGS) {
    tEnd_total = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        tEnd_total - tStart_total);
    std::cerr << "...Total time spent in C++ function: " << duration.count()
              << " ms" << std::endl;
  }
}

// Public API:

cuda_context_t cuda_init(int *err) noexcept {
  decltype(std::chrono::high_resolution_clock::now()) tStart, tEnd;

  *err = 0;

  auto context = new struct context;

  if (TIMINGS) {
    tStart = std::chrono::high_resolution_clock::now();
  }

  try {
    throwOnErr(cublasCreate(&context->cublasH));
    throwOnErr(cusolverDnCreate(&context->cusolverH));
    throwOnErr(
        cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking));
    throwOnErr(cublasSetStream(context->cublasH, context->stream));
    throwOnErr(cusolverDnSetStream(context->cusolverH, context->stream));
    context->valid = true;
  } catch (std::runtime_error &ex) {
    std::cerr << ex.what() << std::endl;
    *err = 1;
  }

  if (TIMINGS) {
    tEnd = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);
    std::cerr << "...Library Initialization: " << duration.count() << " ms"
              << std::endl;
  }

  return reinterpret_cast<cuda_context_t>(context);
}

void cuda_finalize(cuda_context_t ctx, int *err) noexcept {
  decltype(std::chrono::high_resolution_clock::now()) tStart, tEnd;

  *err = 0;

  auto context = restore_ctx(ctx);
  if (!context) {
    std::cerr << "libaccel_cuda: Invalid context" << std::endl;
    *err = 1;
    return;
  }

  if (TIMINGS) {
    tStart = std::chrono::high_resolution_clock::now();
  }

  if (context->cublasH) {
    *err |= cublasDestroy(context->cublasH);
    context->cublasH = NULL;
  }

  if (context->cusolverH) {
    *err |= cusolverDnDestroy(context->cusolverH);
    context->cusolverH = NULL;
  }

  if (context->stream) {
    *err |= cudaStreamDestroy(context->stream);
    context->stream = NULL;
  }

  // We cannot reset the device here as other contexts may still be valid
  // cudaDeviceReset();

  context->magic = 0;
  context->valid = false;
  atomic_thread_fence(std::memory_order_relaxed);
  delete context;

  if (TIMINGS) {
    tEnd = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);
    std::cerr << "...Library Finalization: " << duration.count() << " ms"
              << std::endl;
  }
}

void cuda_dsyevd(cuda_context_t ctx, int64_t n, double *A, int64_t lda,
                 double *W, int *err) noexcept {
  return Xsyevd(ctx, n, A, lda, W, err);
}

void cuda_ssyevd(cuda_context_t ctx, int64_t n, float *A, int64_t lda, float *W,
                 int *err) noexcept {
  return Xsyevd(ctx, n, A, lda, W, err);
}

void cuda_dgemm(cuda_context_t ctx, char trans_a, char trans_b, int64_t m,
                int64_t n, int64_t k, double alpha, const double *A,
                int64_t lda, const double *B, int64_t ldb, double beta,
                double *C, int64_t ldc, int *err) noexcept {
  return Xgemm(ctx, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C,
               ldc, err);
}

void cuda_sgemm(cuda_context_t ctx, char trans_a, char trans_b, int64_t m,
                int64_t n, int64_t k, float alpha, const float *A, int64_t lda,
                const float *B, int64_t ldb, float beta, float *C, int64_t ldc,
                int *err) noexcept {
  return Xgemm(ctx, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C,
               ldc, err);
}