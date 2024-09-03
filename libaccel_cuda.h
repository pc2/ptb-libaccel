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

#pragma once

#include <stdint.h>
#include <cublas_v2.h>


#ifdef __cplusplus
extern "C" {
#else
#define noexcept
#endif

typedef struct ctx_opaque_ *cuda_context_t;

cuda_context_t cuda_init(int *err) noexcept;
void cuda_finalize(cuda_context_t ctx, int *err) noexcept;

void cuda_dsyevd(cuda_context_t ctx, int64_t n, double *A, int64_t lda,
                 double *W, int *err) noexcept;
void cuda_ssyevd(cuda_context_t ctx, int64_t n, float *A, int64_t lda, float *W,
                 int *err) noexcept;

void cuda_ssygvd(cuda_context_t ctx, int64_t n, float *A, int64_t lda,
                 float *B, int64_t ldb, float *W, int *err) noexcept;
void cuda_dsygvd(cuda_context_t ctx, int64_t n, double *A, int64_t lda,
                 double *B, int64_t ldb, double *W, int *err) noexcept;

void cuda_dgemm(cuda_context_t ctx, char trans_a, char trans_b, int64_t m,
                int64_t n, int64_t k, double alpha, const double *A,
                int64_t lda, const double *B, int64_t ldb, double beta,
                double *C, int64_t ldc, int *err) noexcept;
void cuda_sgemm(cuda_context_t ctx, char trans_a, char trans_b, int64_t m,
                int64_t n, int64_t k, float alpha, const float *A, int64_t lda,
                const float *B, int64_t ldb, float beta, float *C, int64_t ldc,
                int *err) noexcept;

void cuda_gemm_ex(cuda_context_t ctx, char trans_a, char trans_b, int64_t m,
                  int64_t n, int64_t k, const void *alpha, const void *A,
                  int64_t lda, int32_t Atype, const void *B, int64_t ldb,
                  int32_t Btype, const void *beta, void *C, int64_t ldc,
                  int32_t Ctype, int32_t computeType, 
                  int32_t algo, int32_t *err) noexcept;


#ifdef __cplusplus
}
#else
#undef noexcept
#endif