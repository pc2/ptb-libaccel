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

program test

   use accel_lib
   use, intrinsic :: iso_fortran_env, only: real64, real32, int32
   implicit none

   integer, parameter :: dim = 1024

   integer(kind=int32) :: err
   type(cuda_context) :: ctx, anotherctx

   ! For testing cuda_dsyevd
   real(kind=real64), dimension(3, 3) :: A
   real(kind=real64), dimension(3) :: W

   ! For testing syevd
   real(kind=real32), dimension(3, 3) :: Asp
   real(kind=real32), dimension(3) :: Wsp

   ! For testing cuda_dgemm
   real(kind=real64), dimension(2, 3) :: X
   real(kind=real64), dimension(3, 4) :: Y
   real(kind=real64), dimension(2, 4) :: Z

   ! For testing cuda_sgemm
   real(kind=real32), dimension(2, 3) :: Xsp
   real(kind=real32), dimension(3, 4) :: Ysp
   real(kind=real32), dimension(2, 4) :: Zsp

   ! For matrix inversion test
   real(kind=real64), dimension(:, :), allocatable :: M, Minv, eigvecs, identity
   real(kind=real64), dimension(:), allocatable :: eigvals
   real(kind=real64) :: error
   integer :: i, j

   err = load_cuda()
   if (err /= 0) then
      stop "CUDA acceleration library could not be loaded."
   end if

   ctx = cuda_init(err)
   if (err /= 0) then
      stop "Could not initialize GPU acceleration library."
   end if

   print *
   print *, "############################"
   print *, "# Eigenvalue Decomposition #"
   print *, "############################"
   print *

   A = reshape((/3.5, 0.5, 0.0, 0.5, 3.5, 0.0, 0.0, 0.0, 2.0/), shape(A))
   Asp = REAL(A, kind=real32)

   print *, "Input matrix:"
   call print_matrix64(A)
   print *

   call cuda_dsyevd(ctx, 3, A, W, err)

   if (err /= 0) then
      stop "An error occured in the CUDA accelerated code."
   end if

   print *, "Eigenvectors (double prec):"
   call print_matrix64(A)
   print *

   print *, "Eigenvalues (double prec):"
   print *, W
   print *

   call cuda_ssyevd(ctx, 3, Asp, Wsp, err)

   if (err /= 0) then
      stop "An error occured in the CUDA accelerated code."
   end if

   print *, "Eigenvectors (single prec):"
   call print_matrix32(Asp)
   print *

   print *, "Eigenvalues (single prec):"
   print *, Wsp
   print *

   print *
   print *, "#########################"
   print *, "# Matrix Multiplication #"
   print *, "#########################"
   print *

   anotherctx = cuda_init(err)
   if (err /= 0) then
      stop "Could not initialize GPU acceleration library."
   end if

   X = reshape((/1, 2, 3, 4, 5, 6/), shape(X))
   Y = reshape((/1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12/), shape(Y))
   Z = reshape((/1, 2, 3, 4, 5, 6, 7, 8/), shape(Z))
   Xsp = REAL(X, kind=real32)
   Ysp = REAL(Y, kind=real32)
   Zsp = REAL(Z, kind=real32)

   print *, "Input matrix X:"
   call print_matrix64(X)
   print *

   print *, "Input matrix Y:"
   call print_matrix64(Y)
   print *

   print *, "Input matrix Z:"
   call print_matrix64(Z)
   print *

   call cuda_dgemm(anotherctx, 'N', 'N', 2, 4, 3, 1.5_real64, X, Y, -0.3_real64, Z, err)
   if (err /= 0) then
      stop "An error occured in the CUDA accelerated code."
   end if

   print *, "1.5 * X * Y - 0.3 * Z (double prec):"
   call print_matrix64(Z)
   print *

   call cuda_sgemm(anotherctx, 'N', 'N', 2, 4, 3, 1.5, Xsp, Ysp, -0.3, Zsp, err)
   if (err /= 0) then
      stop "An error occured in the CUDA accelerated code."
   end if

   print *, "1.5 * X * Y - 0.3 * Z (single prec):"
   call print_matrix32(Zsp)
   print *

   call cuda_finalize(anotherctx, err)
   if (err /= 0) then
      stop "An error occured in finalize."
   end if

   print *
   print *, "######################################################"
   print *, "# Application: Matrix Inverse via Eigendecomposition #"
   print *, "######################################################"
   print *

   allocate (M(dim, dim), Minv(dim, dim), eigvecs(dim, dim), identity(dim, dim), eigvals(dim))

   print *, "Generating random symmetric matrix..."
   call random_number(M)
   M = M + transpose(M)

   print *, "Computing eigendecomposition..."
   eigvecs = M
   call cuda_dsyevd(ctx, dim, eigvecs, eigvals, err)
   if (err /= 0) then
      stop "An error occured in the CUDA accelerated code."
   end if

   print *, "Computing inverse based on decomposition..."
   do j = 1, dim
      do i = 1, dim
         Minv(i, j) = 1.0_real64/eigvals(j)*eigvecs(i, j)
      end do
   end do
   call cuda_dgemm(ctx, 'N', 'T', dim, dim, dim, 1.0_real64, Minv, eigvecs, &
                   0.0_real64, Minv, err)
   if (err /= 0) then
      stop "An error occured in the CUDA accelerated code."
   end if

   print *, "Computing identity based on matrix and its inverse..."
   call cuda_dgemm(ctx, 'N', 'N', dim, dim, dim, 1.0_real64, M, Minv, &
                   0.0_real64, identity, err)
   if (err /= 0) then
      stop "An error occured in the CUDA accelerated code."
   end if

   error = 0.0_real64
   do j = 1, dim
      do i = 1, dim
         if (i == j) then
            error = error + (identity(i, j) - 1)*(identity(i, j) - 1)
         else
            error = error + identity(i, j)*identity(i, j)
         end if
      end do
   end do
   error = sqrt(error)
   print *, "Error (Frobenius norm): ", error

   deallocate (M, Minv, eigvecs, identity, eigvals)

   call cuda_finalize(ctx, err)
   if (err /= 0) then
      stop "An error occured in finalize."
   end if

contains

   subroutine print_matrix64(input_matrix)
      real(kind=real64), dimension(:, :), intent(in) :: input_matrix
      integer :: N(2), i
      N = shape(input_matrix)
      do i = 1, N(1)
         print *, input_matrix(i, :)
      end do
   end subroutine print_matrix64

   subroutine print_matrix32(input_matrix)
      real(kind=real32), dimension(:, :), intent(in) :: input_matrix
      integer :: N(2), i
      N = shape(input_matrix)
      do i = 1, N(1)
         print *, input_matrix(i, :)
      end do
   end subroutine print_matrix32

end program test
