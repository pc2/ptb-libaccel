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
#include "libaccel_cuda.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int dim = 1024;

void print_matrix_colMajor(double *A, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%f ", A[i + j * rows]);
    }
    printf("\n");
  }
  fflush(stdout);
}

void print_vector(double *A, int entries) {
  for (int i = 0; i < entries; i++) {
    printf("%f ", A[i]);
  }
  printf("\n");
  fflush(stdout);
}

int main() {
  int err;
  cuda_context_t ctx;

  double A[] = {3.5, 0.5, 0.0, 0.5, 3.5, 0.0, 0.0, 0.0, 2.0};
  double W[3];
  double X[] = {1, 2, 3, 4, 5, 6};
  double Y[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  double Z[] = {1, 2, 3, 4, 5, 6, 7, 8};

  err = load_cuda();
  if (err) {
    fprintf(stderr, "CUDA acceleration library could not be loaded.\n");
    exit(1);
  }

  ctx = cuda_init(&err);
  if (err) {
    fprintf(stderr, "Could not initialize GPU acceleration library.\n");
    exit(1);
  }

  printf("\n"
         "############################\n"
         "# Eigenvalue Decomposition #\n"
         "############################\n\n");

  printf("Input matrix:\n");
  print_matrix_colMajor(A, 3, 3);
  printf("\n");

  cuda_dsyevd(ctx, 3, A, 3, W, &err);
  if (err) {
    fprintf(stderr, "An error occured in the CUDA accelerated code.\n");
    exit(1);
  }

  printf("Eigenvectors:\n");
  print_matrix_colMajor(A, 3, 3);
  printf("\n");

  printf("Eigenvalues:\n");
  print_vector(W, 3);
  printf("\n");

  printf("\n"
         "#########################\n"
         "# Matrix Multiplication #\n"
         "#########################\n\n");

  printf("Input matrix X:\n");
  print_matrix_colMajor(X, 2, 3);
  printf("\n");

  printf("Input matrix Y:\n");
  print_matrix_colMajor(Y, 3, 4);
  printf("\n");

  printf("Input matrix Z:\n");
  print_matrix_colMajor(Z, 2, 4);
  printf("\n");

  cuda_dgemm(ctx, 'N', 'N', 2, 4, 3, 1.5, X, 2, Y, 3, -0.3, Z, 2, &err);
  if (err) {
    fprintf(stderr, "An error occured in the CUDA accelerated code.\n");
    exit(1);
  }

  printf("1.5 * X * Y - 0.3 * Z:\n");
  print_matrix_colMajor(Z, 2, 4);
  printf("\n");

  printf("\n"
         "######################################################\n"
         "# Application: Matrix Inverse via Eigendecomposition #\n"
         "######################################################\n\n");

  double *M = calloc(dim * dim, sizeof(*M));
  double *Minv = calloc(dim * dim, sizeof(*M));
  double *eigvecs = calloc(dim * dim, sizeof(*M));
  double *eigvals = calloc(dim, sizeof(*M));
  double *identity = calloc(dim * dim, sizeof(*M));

  printf("Generating random symmetric matrix...\n");
  for (int i = 0; i < dim; i++) {
    for (int j = i; j < dim; j++) {
      M[dim * i + j] = (double)rand() / RAND_MAX;
      M[dim * j + i] = M[dim * i + j];
    }
  }

  printf("Computing eigendecomposition...\n");
  memcpy(eigvecs, M, dim * dim * sizeof(*M));
  cuda_dsyevd(ctx, dim, eigvecs, dim, eigvals, &err);
  if (err) {
    fprintf(stderr, "An error occured in the CUDA accelerated code.\n");
    exit(1);
  }

  printf("Computing inverse based on decomposition...\n");
  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < dim; i++) {
      Minv[i + j * dim] = 1.0 / eigvals[j] * eigvecs[i + j * dim];
    }
  }
  cuda_dgemm(ctx, 'N', 'T', dim, dim, dim, 1.0, Minv, dim, eigvecs, dim, 0.0,
             Minv, dim, &err);
  if (err) {
    fprintf(stderr, "An error occured in the CUDA accelerated code.\n");
    exit(1);
  }

  printf("Computing identity based on matrix and its inverse...\n");
  cuda_dgemm(ctx, 'N', 'N', dim, dim, dim, 1.0, M, dim, Minv, dim, 0.0,
             identity, dim, &err);
  if (err) {
    fprintf(stderr, "An error occured in the CUDA accelerated code.\n");
    exit(1);
  }

  double error = 0;
  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < dim; i++) {
      if (i == j) {
        error += (identity[i + j * dim] - 1) * (identity[i + j * dim] - 1);
      } else {
        error += (identity[i + j * dim]) * (identity[i + j * dim]);
      }
    }
  }
  error = sqrt(error);
  printf("Error (Frobenius norm): %f\n", error);

  free(M);
  free(Minv);
  free(eigvecs);
  free(eigvals);
  free(identity);

  cuda_finalize(ctx, &err);
  if (err) {
    fprintf(stderr, "An error occured in finalize.");
    exit(1);
  }
}