# libaccel

A simple abstraction layer for selected cuBLAS and cuSOLVER routines in C/C++/Fortran. May be extended in future for different accelerator backends (e.g., rocBLAS/rocSOLVER) and additional routines.

Note that this only allows offloading of single cuBLAS/cuSOLVER calls, requiring data transfers between host and GPU for each call. Hence, the performance may be limited by PCIe data transfer rates.

## Currently Implemented Routines
* sgemm / dgemm
* ssyevd / dsyevd

## Usage Model

The core library `libaccel.so` can be compiled without any specific dependencies and hence be linked into any program. The acceleration backend `libaccel_cuda.so` can only be compiled if the CUDA libraries are available. At runtime, libaccel can be asked to load the CUDA backend if available. The calling program should fall back to other implementations (BLAS/LAPACK) if the backend cannot be loaded.

Each calling thread needs to initialize a CUDA context. On failure (e.g., if no CUDA-capable device is available), the calling program should fall back to other implementations (BLAS/LAPACK).

## Building

To build the core library, just run `make` or `make libs`. If CUDA libraries are available, you can build the CUDA backend via `make cuda`. Example programs can be built running `make all`.

## API

General notes:
* The API is not stable and may change at any time.
* Matrices are always considered being stored in column major order, even when using the C API.
* The C API exposes leading dimensions (lda, ldb, ldc) for certain operations. Currently, the Fortran API does not. Here, the leading dimensions are set automatically, assuming a packed storage of all matrix values.

### Fortran

#### Types
* `cuda_context`: Represents a CUDA context.

#### Functions
* `load_cuda()`: Attempts to load the CUDA backend. Returns 0 on success, some integer otherwise.
* `cuda_init(err)`: Attempts to create a CUDA context. Returns the context on success. A return status code is written to `err` and must be queried by the caller. If 0, the returned context is valid. Otherwise, the returned context must not be used.

#### Subroutines
* `cuda_finalize(ctx, err)`: Closes the passed context. The context must not be used again after calling this function. `err` is set to 0 on success, to some integer otherwise.
* `cuda_dsyevd(ctx, n, A, W, err)`
* `cuda_ssyevd(ctx, n, A, W, err)`
* `cuda_dgemm(ctx, trans_a, trans_b, m, n, k, alpha, A, B, beta, C, err)`
* `cuda_sgemm(ctx, trans_a, trans_b, m, n, k, alpha, A, B, beta, C, err)`

#### Example

```Fortran
use, intrinsic :: iso_fortran_env
use accel_lib

type(cuda_context) :: ctx
integer(kind=int32) :: err, N
real(kind=real64), dimension(:, :), allocatable :: A
real(kind=real64), dimension(:), allocatable :: W

! ...

err = load_cuda()
if (err /= 0) then
    stop "CUDA acceleration library could not be loaded."
end if

ctx = cuda_init(err)
if (err /= 0) then
    stop "Could not initialize GPU acceleration library."
end if

call cuda_dsyevd(ctx, N, A, W, err)
if (err /= 0) then
    stop "An error occured in the CUDA accelerated code."
end if

call cuda_finalize(ctx, err)
if (err /= 0) then
    stop "An error occured in finalize."
end if
```

See `test.f90` for a more elaborate test program.

### C

#### Types
* `cuda_context_t`: Represents a CUDA context.

#### Functions

```C
int load_cuda();

cuda_context_t cuda_init(int *err);

void cuda_finalize(cuda_context_t ctx, int *err);

void cuda_dsyevd(cuda_context_t ctx, int64_t n, double *A, int64_t lda,
                double *W, int *err);

void cuda_ssyevd(cuda_context_t ctx, int64_t n, float *A, int64_t lda,
                float *W, int *err);

void cuda_dgemm(cuda_context_t ctx, char trans_a, char trans_b, int64_t m,
                int64_t n, int64_t k, double alpha, const double *A,
                int64_t lda, const double *B, int64_t ldb, double beta,
                double *C, int64_t ldc, int *err);

void cuda_sgemm(cuda_context_t ctx, char trans_a, char trans_b, int64_t m,
                int64_t n, int64_t k, float alpha, const float *A, int64_t lda,
                const float *B, int64_t ldb, float beta, float *C, int64_t ldc,
                int *err);
```

#### Example

```C
int err, N;
cuda_context_t ctx;
double *A, W;

// ...

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

cuda_dsyevd(ctx, N, A, N, W, &err);
if (err) {
    fprintf(stderr, "An error occured in the CUDA accelerated code.\n");
    exit(1);
}

cuda_finalize(ctx, &err);
if (err) {
    fprintf(stderr, "An error occured in finalize.");
    exit(1);
}
```

See `test_c.c` for a more elaborate test program.