! Copyright (c) 2023 Paderborn Center for Parallel Computing
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
! SOFTWARE.

module accel_lib

   use, intrinsic :: iso_c_binding
   use, intrinsic :: iso_fortran_env
   implicit none
   private

   public :: load_cuda, cuda_context, cuda_init, cuda_finalize 
   public :: cuda_dsyevd, cuda_ssyevd, cuda_dgemm, cuda_sgemm, cuda_dsygvd, cuda_ssygvd
   public :: cuda_gemmEx

   type cuda_context
      type(c_ptr) :: c_ctx
   end type cuda_context

   ! Define constants for CUDA data types
   integer(kind=int32), parameter, public :: R_16F = 2
   integer(kind=int32), parameter, public :: R_32F = 0
   integer(kind=int32), parameter, public :: R_64F = 1

   ! Define constants for cublasComputeType_t and cublasGemmAlgo_t
   integer(kind=int32), parameter, public :: COMPUTE_16F = 63
   integer(kind=int32), parameter, public :: COMPUTE_32F = 64
   integer(kind=int32), parameter, public :: COMPUTE_64F = 65
   integer(kind=int32), parameter, public :: GEMM_DEFAULT = -1


   interface
      function c_load_cuda() result(err) bind(c, name="load_cuda")
         use, intrinsic :: iso_c_binding
         implicit none
         integer(kind=c_int) :: err
      end function c_load_cuda
      function c_cuda_init(err) result(ctx) bind(c, name="cuda_init")
         use, intrinsic :: iso_c_binding
         implicit none
         type(c_ptr), intent(in), value :: err
         type(c_ptr) :: ctx
      end function c_cuda_init
      subroutine c_cuda_finalize(ctx, err) bind(c, name="cuda_finalize")
         use, intrinsic :: iso_c_binding
         implicit none
         type(c_ptr), intent(in), value :: ctx, err
      end subroutine c_cuda_finalize
      subroutine c_cuda_dsyevd(ctx, n, A, lda, W, err) bind(c, name="cuda_dsyevd")
         use, intrinsic :: iso_c_binding
         implicit none
         integer(kind=c_int64_t), intent(in), value :: n, lda
         type(c_ptr), intent(in), value :: ctx, A, W, err
      end subroutine c_cuda_dsyevd   

      subroutine c_cuda_ssyevd(ctx, n, A, lda, W, err) bind(c, name="cuda_ssyevd")
         use, intrinsic :: iso_c_binding
         implicit none
         integer(kind=c_int64_t), intent(in), value :: n, lda
         type(c_ptr), intent(in), value :: ctx, A, W, err
      end subroutine c_cuda_ssyevd   

      subroutine c_cuda_ssygvd(ctx, n, A, lda, B, ldb, W, err) bind(c, name="cuda_ssygvd")
         use, intrinsic :: iso_c_binding
         implicit none
         type(c_ptr), intent(in), value :: ctx
         integer(kind=c_int64_t), intent(in), value :: n, lda, ldb
         type(c_ptr), intent(in), value :: A, B, W, err   
      end subroutine c_cuda_ssygvd
           
      subroutine c_cuda_dsygvd(ctx, n, A, lda, B, ldb, W, err) bind(c, name="cuda_dsygvd")
         use, intrinsic :: iso_c_binding
         implicit none
         type(c_ptr), intent(in), value :: ctx
         integer(kind=c_int64_t), intent(in), value :: n, lda, ldb
         type(c_ptr), intent(in), value :: A, B, W, err   
      end subroutine c_cuda_dsygvd

 
      subroutine c_cuda_dgemm(ctx, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, err) bind(c, name="cuda_dgemm")
         use, intrinsic :: iso_c_binding
         implicit none
         integer(kind=c_signed_char), intent(in), value :: trans_a, trans_b
         integer(kind=c_int64_t), intent(in), value :: m, n, k, lda, ldb, ldc
         real(kind=c_double), intent(in), value :: alpha, beta
         type(c_ptr), intent(in), value :: ctx, A, B, C, err
      end subroutine c_cuda_dgemm
      
      subroutine c_cuda_sgemm(ctx, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, err) bind(c, name="cuda_sgemm")
         use, intrinsic :: iso_c_binding
         implicit none
         integer(kind=c_signed_char), intent(in), value :: trans_a, trans_b
         integer(kind=c_int64_t), intent(in), value :: m, n, k, lda, ldb, ldc
         real(kind=c_float), intent(in), value :: alpha, beta
         type(c_ptr), intent(in), value :: ctx, A, B, C, err
      end subroutine c_cuda_sgemm

      subroutine c_cuda_gemmEx(ctx, trans_a, trans_b, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo, err) bind(c, name="cuda_gemm_ex")
         use, intrinsic :: iso_c_binding
         implicit none
         type(c_ptr), intent(in), value :: ctx
         integer(kind=c_signed_char), intent(in), value :: trans_a, trans_b
         integer(kind=c_int64_t), intent(in), value :: m, n, k, lda, ldb, ldc
         type(c_ptr), intent(in), value :: A, B, C
         type(c_ptr), intent(in), value :: alpha, beta, err
         integer(kind=c_int32_t), intent(in), value :: Atype, Btype, Ctype, computeType, algo
       end subroutine c_cuda_gemmEx

   end interface

contains

   function load_cuda() result(err)
      integer(kind=int32) :: err
      err = int(c_load_cuda(), kind=int32)
   end function load_cuda

   function cuda_init(err) result(ctx)
      integer(kind=int32), target, intent(out) :: err
      type(cuda_context) :: ctx

      ctx%c_ctx = c_cuda_init(c_loc(err))
   end function cuda_init

   subroutine cuda_finalize(ctx, err)
      type(cuda_context), intent(in) :: ctx
      integer(kind=int32), target, intent(out) :: err
      call c_cuda_finalize(ctx%c_ctx, c_loc(err))
   end subroutine cuda_finalize

   subroutine cuda_dsyevd(ctx, n, A, W, err)
      type(cuda_context), intent(in) :: ctx
      integer(kind=int64), intent(in) :: n
      real(kind=c_double), dimension(n, n), target, intent(inout) :: A
      real(kind=c_double), dimension(n), target, intent(out) :: W
      integer(kind=int32), target, intent(out) :: err

      call c_cuda_dsyevd(ctx%c_ctx, int(n, kind=c_int64_t), c_loc(A), int(n, kind=c_int64_t), c_loc(W), c_loc(err))
   end subroutine cuda_dsyevd  

   subroutine cuda_ssyevd(ctx, n, A, W, err)
      type(cuda_context), intent(in) :: ctx
      integer(kind=int64), intent(in) :: n
      real(kind=c_float), dimension(n, n), target, intent(inout) :: A
      real(kind=c_float), dimension(n), target, intent(out) :: W
      integer(kind=int32), target, intent(out) :: err

      call c_cuda_ssyevd(ctx%c_ctx, int(n, kind=c_int64_t), c_loc(A), int(n, kind=c_int64_t), c_loc(W), c_loc(err))
   end subroutine cuda_ssyevd

   subroutine cuda_ssygvd(ctx, n, A, B, W, err)
      type(cuda_context), intent(in) :: ctx
      integer(kind=int64), intent(in) :: n
      real(kind=c_float), dimension(n, n), target, intent(inout) :: A
      real(kind=c_float), dimension(n, n), target, intent(inout) :: B
      real(kind=c_float), dimension(n), target, intent(out) :: W
      integer(kind=int32), target, intent(out) :: err
      
      call c_cuda_ssygvd(ctx%c_ctx, int(n, kind=c_int64_t), c_loc(A), int(n,kind=c_int64_t), c_loc(B), &
                                                               int(n, kind=c_int64_t), c_loc(W), c_loc(err))
   end subroutine cuda_ssygvd

   subroutine cuda_dsygvd(ctx, n, A, B, W, err)
      type(cuda_context), intent(in) :: ctx
      integer(kind=int64), intent(in) :: n
      real(kind=c_double), dimension(n, n), target, intent(inout) :: A
      real(kind=c_double), dimension(n, n), target, intent(inout) :: B
      real(kind=c_double), dimension(n), target, intent(out) :: W
      integer(kind=int32), target, intent(out) :: err
      
      call c_cuda_dsygvd(ctx%c_ctx, int(n, kind=c_int64_t), c_loc(A), int(n,kind=c_int64_t), c_loc(B), &
                                                               int(n, kind=c_int64_t), c_loc(W), c_loc(err))
   end subroutine cuda_dsygvd

   subroutine cuda_dgemm(ctx, trans_a, trans_b, m, n, k, alpha, A, B, beta, C, err)
      type(cuda_context), intent(in) :: ctx
      character, intent(in) :: trans_a, trans_b
      integer(kind=int64), intent(in) :: m, n, k
      real(kind=c_double), intent(in) :: alpha, beta
      real(kind=c_double), dimension(m, k), target, intent(in) :: A
      real(kind=c_double), dimension(k, n), target, intent(in) :: B
      real(kind=c_double), dimension(m, n), target, intent(inout) :: C
      integer(kind=int32), target, intent(out) :: err

      call c_cuda_dgemm(ctx%c_ctx, ichar(trans_a, kind=c_signed_char), ichar(trans_b, kind=c_signed_char), &
                        int(m, kind=c_int64_t), int(n, kind=c_int64_t), int(k, kind=c_int64_t), alpha, c_loc(A), &
                        int(m, kind=c_int64_t), c_loc(B), int(k, kind=c_int64_t), beta, c_loc(C), int(m, kind=c_int64_t), &
                        c_loc(err))
   end subroutine cuda_dgemm

   subroutine cuda_sgemm(ctx, trans_a, trans_b, m, n, k, alpha, A, B, beta, C, err)
      type(cuda_context), intent(in) :: ctx
      character, intent(in) :: trans_a, trans_b
      integer(kind=int64), intent(in) :: m, n, k
      real(kind=c_float), intent(in) :: alpha, beta
      real(kind=c_float), dimension(m, k), target, intent(in) :: A
      real(kind=c_float), dimension(k, n), target, intent(in) :: B
      real(kind=c_float), dimension(m, n), target, intent(inout) :: C
      integer(kind=int32), target, intent(out) :: err

      call c_cuda_sgemm(ctx%c_ctx, ichar(trans_a, kind=c_signed_char), ichar(trans_b, kind=c_signed_char), &
                        int(m, kind=c_int64_t), int(n, kind=c_int64_t), int(k, kind=c_int64_t), alpha, c_loc(A), &
                        int(m, kind=c_int64_t), c_loc(B), int(k, kind=c_int64_t), beta, c_loc(C), int(m, kind=c_int64_t), &
                        c_loc(err))
   end subroutine cuda_sgemm


   subroutine cuda_gemmEx(ctx, trans_a, trans_b, m, n, k, alpha, A, B, beta, C, Atype, Btype, Ctype, computeType, algo, err)
      type(cuda_context), intent(in) :: ctx
      character, intent(in) :: trans_a, trans_b
      integer(kind=int64), intent(in) :: m, n, k
      real(kind=c_), target, intent(in) :: alpha, beta
      real(kind=c_double), dimension(:,:), target, intent(in) :: A, B
      real(kind=c_double), dimension(:,:), target, intent(inout) :: C
      integer(kind=int32), target, intent(out) :: err
      integer(kind=int32), intent(in) :: Atype, Btype, Ctype, computeType, algo
      
      ! Call the C function with all necessary parameters
      call c_cuda_gemmEx(ctx%c_ctx, ichar(trans_a, kind=c_signed_char), ichar(trans_b, kind=c_signed_char), &
                         int(m, kind=c_int64_t), int(n, kind=c_int64_t), int(k, kind=c_int64_t), &
                         c_loc(alpha), c_loc(A),  int(m, kind=c_int64_t), int(Atype, kind=c_int32_t),&
                         c_loc(B), int(k, kind=c_int64_t),  int(Btype, kind=c_int32_t), &
                         c_loc(beta), c_loc(C), int(m, kind=c_int64_t), int(Ctype, kind=c_int32_t), &
                         int(computeType,kind=c_int32_t), int(algo, kind=c_int32_t), c_loc(err))
   end subroutine cuda_gemmEx
   

end module accel_lib
