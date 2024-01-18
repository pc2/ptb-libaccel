#pragma once

#include <cstdint>
#include <cusolverDn.h>
#include <cusolver_common.h>
#include <stdexcept>

#if (CUSOLVER_VER_MAJOR < 11)

static cusolverStatus_t CUSOLVERAPI cusolverDnXsyevd_bufferSize(
    cusolverDnHandle_t handle, [[maybe_unused]] cusolverDnParams_t params,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeW,
    const void *W, [[maybe_unused]] cudaDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {

  int lwork = 0;
  cusolverStatus_t ret;

  if (dataTypeA == CUDA_R_32F && dataTypeW == CUDA_R_32F) {
    ret = cusolverDnSsyevd_bufferSize(
        handle, jobz, uplo, static_cast<int>(n), static_cast<const float *>(A),
        static_cast<int>(lda), static_cast<const float *>(W), &lwork);
    *workspaceInBytesOnDevice = sizeof(float) * lwork;
  } else if (dataTypeA == CUDA_R_64F && dataTypeW == CUDA_R_64F) {
    ret = cusolverDnDsyevd_bufferSize(
        handle, jobz, uplo, static_cast<int>(n), static_cast<const double *>(A),
        static_cast<int>(lda), static_cast<const double *>(W), &lwork);
    *workspaceInBytesOnDevice = sizeof(double) * lwork;
  } else {
    throw std::runtime_error(
        "libaccel_cuda: Old CUSOLVER API called with invalid arguments");
  }

  *workspaceInBytesOnHost = 0;
  return ret;
}

cusolverStatus_t CUSOLVERAPI static cusolverDnXsyevd(
    cusolverDnHandle_t handle, [[maybe_unused]] cusolverDnParams_t params,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
    cudaDataType dataTypeA, void *A, int64_t lda, cudaDataType dataTypeW,
    void *W, [[maybe_unused]] cudaDataType computeType, void *bufferOnDevice,
    size_t workspaceInBytesOnDevice, [[maybe_unused]] void *bufferOnHost,
    [[maybe_unused]] size_t workspaceInBytesOnHost, int *info) {

  cusolverStatus_t ret;

  if (dataTypeA == CUDA_R_32F && dataTypeW == CUDA_R_32F) {
    ret = cusolverDnSsyevd(
        handle, jobz, uplo, static_cast<int>(n), static_cast<float *>(A),
        static_cast<int>(lda), static_cast<float *>(W),
        static_cast<float *>(bufferOnDevice),
        static_cast<int>(workspaceInBytesOnDevice / sizeof(float)), info);
  } else if (dataTypeA == CUDA_R_64F && dataTypeW == CUDA_R_64F) {
    ret = cusolverDnDsyevd(
        handle, jobz, uplo, static_cast<int>(n), static_cast<double *>(A),
        static_cast<int>(lda), static_cast<double *>(W),
        static_cast<double *>(bufferOnDevice),
        static_cast<int>(workspaceInBytesOnDevice / sizeof(double)), info);
  } else {
    throw std::runtime_error(
        "libaccel_cuda: Old CUSOLVER API called with invalid arguments");
  }

  return ret;
}

#endif