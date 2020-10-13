#ifdef _OPENACC

c-----------------------------------------------------------------------
      subroutine ax_acc(w,u,gxyz,n) ! Matrix-vector product: w=A*u

#ifdef _CUDA
      use cudafor
#endif

      include 'SIZE'
      include 'TOTAL'

      real w(nx1,ny1,nz1,nelt)
      real u(nx1,ny1,nz1,nelt)
      real gxyz(nx1,ny1,nz1,2*ldim,lelt)
      integer i,j,k,l,e,n

      integer lt

      integer cuda_err

      lt = nx1*ny1*nz1*nelt
      !print *, nelt
!$ACC DATA PRESENT(w,u(:,:,:,:),gxyz,dxm1,dxtm1)


!$ACC HOST_DATA USE_DEVICE(w,u(:,:,:,:),gxyz,dxm1,dxtm1)
         call ax_cuda2(w, u,
     $     gxyz,dxm1,dxtm1,nelt)
!$ACC END HOST_DATA

       cuda_err = cudaGetLastError()
       if (cuda_err /= cudaSuccess) then
         write(6, 815) cuda_err, cudaGetErrorString(cuda_err)
         call exitt
       endif

       istat = cudaDeviceSynchronize()

       cuda_err = cudaGetLastError()
       if (cuda_err /= cudaSuccess) then
         write(6, 815) cuda_err, cudaGetErrorString(cuda_err)
         call exitt
       endif

  815    format('CUDA ERROR', I3, ': ', A)

!$ACC END DATA

      nxyz=nx1*ny1*nz1
      flop_a = flop_a + (19*nxyz+12*nx1*nxyz)*nelt

      return
      end
c-----------------------------------------------------------------------
      subroutine bk5_acc(g,w,p,n,niter)
 
      include 'SIZE'

c     Solve Ax=f where A is SPD and is invoked by ax()
c
c     Output:  x - vector of length n
c
c     Input:   f - vector of length n
c     Input:   g - geometric factors for SEM operator
c     Input:   c - inverse of the counting matrix
c
c     Work arrays:   r,w,p,z  - vectors of length n
c
c     User-provided ax(w,z,n) returns  w := Az,
c
c     User-provided solveM(z,r,n) ) returns  z := M^-1 r,
c

      common /mymask/cmask(-1:lx1*ly1*lz1*lelt)
      parameter (lt=lx1*ly1*lz1*lelt)

      real w(n)
      real p(lx1,lx1,lx1,lelt)

      real g(2*ldim,lt)
      integer cuda_err
      character*1 ans

      pap = 0.0



!$ACC DATA PRESENT(g,w,p)
      call rone_acc(w, n)
      call rone_acc(p, n)

      call set_timer_flop_cnt(0)
      call apibegin(nelt, lx1)
      do iter = 1, niter         
         call ax_acc(w,p,g,n)   ! flopa
      end do
      call apiend
      call set_timer_flop_cnt(1)

!$ACC END DATA


      return
      end
c-----------------------------------------------------------------------
      subroutine solveM_acc(z,r,n)
      include 'INPUT'
      real z(n),r(n)

      nn = n
      call h1mg_solve_acc(z,r,nn)
      return
      end
c-----------------------------------------------------------------------

#endif
