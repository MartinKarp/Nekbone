c-----------------------------------------------------------------------
      subroutine cg(x,f,g,c,r,w,p,z,n,niter,flop_cg)
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
      real ur(lt),us(lt),ut(lt),wk(lt)

     
      real x(n),f(n),r(n),w(n),p(n),z(n),g(1),c(n)

      character*1 ans
      pap = 0.0
c     set machine tolerances
      one = 1.
      eps = 1.e-20
      if (one+eps .eq. one) eps = 1.e-14
      if (one+eps .eq. one) eps = 1.e-7
      rtz1=1.0
      call rzero(x,n)
      call copy (r,f,n)
      call maskit (r,cmask,nx1,ny1,nz1) ! Zero out Dirichlet conditions
      rnorm = sqrt(glsc3(r,c,r,n))
      iter = 0
      if (nid.eq.0)  write(6,6) iter,rnorm

      miter = niter
c     call tester(z,r,n)  
      do iter=1,miter
         call solveM(z,r,n)    ! preconditioner here

         rtz2=rtz1                                                       ! OPS
         rtz1=glsc3(r,c,z,n)   ! parallel weighted inner product r^T C z ! 3n

         beta = rtz1/rtz2
         if (iter.eq.1) beta=0.0
         call add2s1(p,z,beta,n)                                         ! 2n

         call ax(w,p,g,ur,us,ut,wk,n)                                    ! flopa
         pap=glsc3(w,c,p,n)                                              ! 3n

         alpha=rtz1/pap
         alphm=-alpha
         call add2s2(x,p,alpha,n)                                        ! 2n
         call add2s2(r,w,alphm,n)                                        ! 2n

         rtr = glsc3(r,c,r,n)                                            ! 3n
         if (iter.eq.1) rlim2 = rtr*eps**2
         if (iter.eq.1) rtr0  = rtr
         rnorm = sqrt(rtr)
c        if (nid.eq.0.and.mod(iter,100).eq.0) 
c    $      write(6,6) iter,rnorm,alpha,beta,pap
    6    format('cg:',i4,1p4e12.4)
c        if (rtr.le.rlim2) goto 1001

      enddo

 1001 continue

      if (nid.eq.0) write(6,6) iter,rnorm,alpha,beta,pap

      flop_cg = flop_cg + iter*15.*n

      return
      end
c-----------------------------------------------------------------------
      subroutine solveM(z,r,n)
      include 'INPUT'
      real z(n),r(n)

      nn = n
      call h1mg_solve(z,r,nn)

      return
      end
c-----------------------------------------------------------------------
      subroutine ax(w,u,gxyz,ur,us,ut,wk,n) ! Matrix-vector product: w=A*u

      include 'SIZE'
      include 'TOTAL'

      real w(nx1*ny1*nz1,nelt),u(nx1*ny1*nz1,nelt)
      real gxyz(2*ldim,nx1*ny1*nz1,nelt)

      parameter (lt=lx1*ly1*lz1*lelt)
      real ur(lt),us(lt),ut(lt),wk(lt)
      common /mymask/cmask(-1:lx1*ly1*lz1*lelt)

      integer e


      do e=1,nelt                                ! ~
         call ax_e( w(1,e),u(1,e),gxyz(1,1,e)    ! w   = A  u
     $                             ,ur,us,ut,wk) !  L     L  L
      enddo                                      ! 
      
      call dssum(w)         ! Gather-scatter operation  ! w   = QQ  w
                                                           !            L
      call add2s2(w,u,.1,n)   !2n
      call maskit(w,cmask,nx1,ny1,nz1)  ! Zero out Dirichlet conditions

      nxyz=nx1*ny1*nz1
      flop_a = flop_a + (19*nxyz+12*nx1*nxyz)*nelt

      return
      end
c-------------------------------------------------------------------------
      subroutine ax1(w,u,n)
      include 'SIZE'
      real w(n),u(n)
      real h2i
  
      h2i = (n+1)*(n+1)  
      do i = 2,n-1
         w(i)=h2i*(2*u(i)-u(i-1)-u(i+1))
      enddo
      w(1)  = h2i*(2*u(1)-u(2  ))
      w(n)  = h2i*(2*u(n)-u(n-1))

      return
      end
c-------------------------------------------------------------------------
      subroutine ax_e(w,u,g,ur,us,ut,wk) ! Local matrix-vector product
      include 'SIZE'
      include 'TOTAL'

      parameter (lxyz=lx1*ly1*lz1)
      real ur(lxyz),us(lxyz),ut(lxyz),wk(lxyz)
      real w(nx1*ny1*nz1),u(nx1*ny1*nz1),g(2*ldim,nx1*ny1*nz1)


      nxyz = nx1*ny1*nz1
      n    = nx1-1

      call local_grad3(ur,us,ut,u,n,dxm1,dxtm1)
      do i=1,nxyz
         wr = g(1,i)*ur(i) + g(2,i)*us(i) + g(3,i)*ut(i)
         ws = g(2,i)*ur(i) + g(4,i)*us(i) + g(5,i)*ut(i)
         wt = g(3,i)*ur(i) + g(5,i)*us(i) + g(6,i)*ut(i)
         us(i) = ws
         ut(i) = wt
         ur(i) = wr
      enddo

      call local_grad3_t(w,ur,us,ut,n,dxm1,dxtm1,wk)

      return
      end
c-------------------------------------------------------------------------
      subroutine local_grad3(ur,us,ut,u,n,D,Dt)
c     Output: ur,us,ut         Input:u,n,D,Dt
      real ur(0:n,0:n,0:n),us(0:n,0:n,0:n),ut(0:n,0:n,0:n)
      real u (0:n,0:n,0:n)
      real D (0:n,0:n),Dt(0:n,0:n)
      integer e

      m1 = n+1
      m2 = m1*m1

      call mxm(D ,m1,u,m1,ur,m2)
      do k=0,n
         call mxm(u(0,0,k),m1,Dt,m1,us(0,0,k),m1)
      enddo
      call mxm(u,m2,Dt,m1,ut,m1)

      return
      end
c-----------------------------------------------------------------------
      subroutine local_grad3_t(u,ur,us,ut,N,D,Dt,w)
c     Output: ur,us,ut         Input:u,N,D,Dt
      real u (0:N,0:N,0:N)
      real ur(0:N,0:N,0:N),us(0:N,0:N,0:N),ut(0:N,0:N,0:N)
      real D (0:N,0:N),Dt(0:N,0:N)
      real w (0:N,0:N,0:N)
      integer e

      m1 = N+1
      m2 = m1*m1
      m3 = m1*m1*m1

      call mxm(Dt,m1,ur,m1,u,m2)

      do k=0,N
         call mxm(us(0,0,k),m1,D ,m1,w(0,0,k),m1)
      enddo
      call add2(u,w,m3)

      call mxm(ut,m2,D ,m1,w,m1)
      call add2(u,w,m3)

      return
      end
c-----------------------------------------------------------------------
      subroutine maskit(w,pmask,nx,ny,nz)   ! Zero out Dirichlet conditions
      include 'SIZE'
      include 'PARALLEL'

      real pmask(-1:lx1*ly1*lz1*lelt)
      real w(1)
      integer e

      nxyz = nx*ny*nz
      nxy  = nx*ny
      if(pmask(-1).lt.0) then
        j=pmask(0)
        do i = 1,j
           k = pmask(i)
           w(k)=0.0
        enddo
      else
c         Zero out Dirichlet boundaries.
c
c                      +------+     ^ Y
c                     /   3  /|     |
c               4--> /      / |     |
c                   +------+ 2 +    +----> X
c                   |   5  |  /    /
c                   |      | /    /
c                   +------+     Z   
c

        nn = 0
        do e  = 1,nelt
          call get_face(w,nx,e)
          do i = 1,nxyz
             if(w(i).eq.0) then
               nn=nn+1
               pmask(nn)=i
             endif
          enddo
        enddo     
        pmask(-1) = -1.
        pmask(0) = nn
      endif


      return
      end
c-----------------------------------------------------------------------
      subroutine masko(w)   ! Old 'mask'
      include 'SIZE'
      real w(1)

      if (nid.eq.0) w(1) = 0.  ! suitable for solvability

      return
      end
c-----------------------------------------------------------------------
      subroutine masking(w,nx,e,x0,x1,y0,y1,z0,z1)
c     Zeros out boundary
      include 'SIZE'
      integer e,x0,x1,y0,y1,z0,z1
      real w(nx,nx,nx,nelt)
      
c       write(6,*) x0,x1,y0,y1,z0,z1
      do k=z0,z1
      do j=y0,y1
      do i=x0,x1
          w(i,j,k,e)=0.0
      enddo
      enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine tester(z,r,n)
c     Used to test if solution to precond. is SPD
      real r(n),z(n)

      do j=1,n
         call rzero(r,n)
         r(j) = 1.0
         call solveM(z,r,n)
         do i=1,n
            write(79,*) z(i)
         enddo
      enddo
      call exitt0
      return
      end
c-----------------------------------------------------------------------
      subroutine get_face(w,nx,ie)
c     zero out all boundaries as Dirichlet
c     to change, change this routine to only zero out 
c     the nodes desired to be Dirichlet, and leave others alone.
      include 'SIZE'
      include 'PARALLEL'
      real w(1)
      integer nx,ie,nelx,nely,nelz
      integer x0,x1,y0,y1,z0,z1
      
      x0=1
      y0=1
      z0=1
      x1=nx
      y1=nx
      z1=nx
      
      nelxy=nelx*nely
      ngl = lglel(ie)        !global element number

      ir = 1+(ngl-1)/nelxy   !global z-count
      iq = mod1(ngl,nelxy)   !global y-count
      iq = 1+(iq-1)/nelx     
      ip = mod1(ngl,nelx)    !global x-count

c     write(6,1) ip,iq,ir,nelx,nely,nelz, nelt,' test it'
c  1  format(7i7,a8)

      if(mod(ip,nelx).eq.1.or.nelx.eq.1)   then  ! Face4
         x0=1
         x1=1
         call masking(w,nx,ie,x0,x1,y0,y1,z0,z1)
      endif
      if(mod(ip,nelx).eq.0)               then   ! Face2
         x0=nx
         x1=nx
         call masking(w,nx,ie,x0,x1,y0,y1,z0,z1)
      endif

      x0=1
      x1=nx
      if(mod(iq,nely).eq.1.or.nely.eq.1) then    ! Face1
         y0=1
         y1=1
         call masking(w,nx,ie,x0,x1,y0,y1,z0,z1)
      endif
      if(mod(iq,nely).eq.0)              then    ! Face3
         y0=nx
         y1=nx
         call masking(w,nx,ie,x0,x1,y0,y1,z0,z1)
      endif

      y0=1
      y1=nx
      if(mod(ir,nelz).eq.1.or.nelz.eq.1) then    ! Face5
         z0=1
         z1=1
         call masking(w,nx,ie,x0,x1,y0,y1,z0,z1)
      endif
      if(mod(ir,nelz).eq.0)              then    ! Face6
         z1=nx
         z0=nx
         call masking(w,nx,ie,x0,x1,y0,y1,z0,z1)
      endif

      return
      end
c-----------------------------------------------------------------------
#ifdef FPGA

      subroutine cg_fpga(x,f,g,c,r,w,p,z,n,niter,flop_cg,flop_a)
      use clfortran
      use clroutines
      use ISO_C_BINDING  
      include 'SIZE'
      include 'DXYZ'
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
      real, target :: cmask
      common /mymask/cmask(-1:lx1*ly1*lz1*lelt)
      parameter (lt=lx1*ly1*lz1*lelt)
      real ur(lt),us(lt),ut(lt),wk(lt)

     
c   Target necessary for OpenCL
      real, target :: x(n),f(n),r(n),w(n),p(n),z(n),g(6*n),c(n),
     $ dxm1,dxtm1
!      real x(n),f(n),r(n),w(n),p(n),z(n),g(1),c(n)

      integer(c_intptr_t), target ::
     $ cl_x,cl_f,cl_r1,cl_w,cl_p,cl_z,cl_g,cl_c,cl_rtr, cl_rtz1,
     $ cl_dxm1, cl_dxtm1, cl_ur, cl_us, cl_ut, cl_wk, cl_cmask
      integer, target :: n, iter
      character*1 ans
c OPencl things
      integer(c_size_t) :: byte_size, element_size, ret, g_size
      integer(c_int32_t) :: err
      character(len=1024) :: options, kernel_name
      character(len=1, kind=c_char),allocatable :: kernel_str(:)
      !integer, target :: np
      integer(c_size_t),target :: globalsize,localsize, length
      integer(c_intptr_t), target :: 
     $ cmd_queue,prog,context,ax_kernel,binary_status
      integer(c_intptr_t), allocatable, target :: 
     $ platform_ids(:), device_ids(:)
      integer :: iplatform
      integer(c_int) :: num_platforms
      integer :: idevice 
      integer :: filesize 
      character(len=1,kind=c_char), allocatable, target :: binary(:)
      character(len=1,kind=c_char), target :: c_kernel_name(1:1024)
      type(c_ptr), target :: psource
      real, target ::  rtr, rtz1
      kernel_name = "ax"
      idevice = 1
      iplatform = 2
      byte_size= 8_8 * int(n,8)
      g_size= 8_8 * int(6*n,8)
      element_size = 8_8*int(lx1*ly1,8)

  
      call create_device_context(iplatform, platform_ids,
     $                           device_ids, context, cmd_queue)
      call query_platform_info(platform_ids(iplatform))
      call read_file("ax.aocx",binary,filesize)
      length = filesize
      psource = C_LOC(binary)
      prog = clCreateProgramWithBinary(
     $ context,1, C_LOC(device_ids(idevice)),
     $ C_LOC(length),C_LOC(psource),C_NULL_PTR,err)
    
      err=clBuildProgram(prog,
     $ 0, C_NULL_PTR,C_NULL_PTR,C_NULL_FUNPTR,C_NULL_PTR)
     
      irec=len(trim(kernel_name))
      do i=1,irec
         c_kernel_name(i)=kernel_name(i:i)
      enddo
      c_kernel_name(irec+1)=C_NULL_CHAR
      ax_kernel=clCreateKernel(prog,C_LOC(c_kernel_name),err)
      if (err.ne.0) stop 'clCreateKernel'
      

      err = clReleaseProgram(prog)
      if (err.ne.0) stop 'clReleaseProgram'
c Init arrays
      cl_p = clCreateBuffer(context, 
     $      CL_MEM_READ_WRITE,byte_size,C_NULL_PTR, err)
      if (err.ne.0) stop 'clCreateBuffer'
      cl_w = clCreateBuffer(context, 
     $      CL_MEM_READ_WRITE,byte_size,C_NULL_PTR, err)
      if (err.ne.0) stop 'clCreateBuffer'
      
      cl_g = clCreateBuffer(context, 
     $      CL_MEM_READ_WRITE,g_size,C_NULL_PTR, err)
      if (err.ne.0) stop 'clCreateBuffer'
 
      cl_dxm1 = clCreateBuffer(context, 
     $      CL_MEM_READ_WRITE,element_size,C_NULL_PTR, err)
      if (err.ne.0) stop 'clCreateBuffer'
     

      err=clSetKernelArg(ax_kernel,0,sizeof(8_8),C_LOC(cl_w))
      if (err.ne.0) stop 'clSetKernelArg'
      err=clSetKernelArg(ax_kernel,1,sizeof(8_8),C_LOC(cl_p))
      if (err.ne.0) stop 'clSetKernelArg'
      err=clSetKernelArg(ax_kernel,2,sizeof(8_8),C_LOC(cl_g))
      if (err.ne.0) stop 'clSetKernelArg'
      err=clSetKernelArg(ax_kernel,3,sizeof(8_8),C_LOC(cl_dxm1))
      if (err.ne.0) stop 'clSetKernelArg'
      err=clSetKernelArg(ax_kernel,4,sizeof(8_4),C_LOC(n))
      if (err.ne.0) stop 'clSetKernelArg'
      err=clSetKernelArg(ax_kernel,5,sizeof(8_4),C_LOC(iter))
      if (err.ne.0) stop 'clSetKernelArg'
      
      globalsize=int(n,8)
       
      pap = 0.0
c     set machine tolerances
      one = 1.
      eps = 1.e-20
      if (one+eps .eq. one) eps = 1.e-14
      if (one+eps .eq. one) eps = 1.e-7
      rtz1=1.0
      iter = 0
      miter = niter
      err=clEnqueueWriteBuffer(cmd_queue,cl_w,CL_TRUE,0_8,
     $    byte_size,C_LOC(w), 0,C_NULL_PTR,C_NULL_PTR)
      if (err.ne.0) stop 'clEnqueueWriteBuffer'
      err=clEnqueueWriteBuffer(cmd_queue,cl_p,CL_TRUE,0_8,
     $    byte_size,C_LOC(p), 0,C_NULL_PTR,C_NULL_PTR)
      if (err.ne.0) stop 'clEnqueueWriteBuffer'
      err=clEnqueueWriteBuffer(cmd_queue,cl_g,CL_TRUE,0_8,
     $    g_size,C_LOC(g), 0,C_NULL_PTR,C_NULL_PTR)
      if (err.ne.0) stop 'clEnqueueWriteBuffer'
      err=clEnqueueWriteBuffer(cmd_queue,cl_dxm1,CL_TRUE,0_8,
     $    element_size,C_LOC(dxm1), 0,C_NULL_PTR,C_NULL_PTR)
      if (err.ne.0) stop 'clEnqueueWriteBuffer'
c     call tester(z,r,n)  
      do iter=1,miter
         !call solveM(z,r,n)    ! preconditioner here

         !rtz2=rtz1                                                       ! OPS
         !rtz1=glsc3(r,c,z,n)   ! parallel weighted inner product r^T C z ! 3n

         !beta = rtz1/rtz2
         !if (iter.eq.1) beta=0.0
         !call add2s1(p,z,beta,n)                                         ! 2n
         err=clEnqueueTask(cmd_queue,ax_kernel,
     $    0,C_NULL_PTR,C_NULL_PTR)
         if (err.ne.0) stop 'clEnqueueEnqueueTask'
        
!         err = clEnqueueReadBuffer(cmd_queue,cl_w,CL_TRUE,
!     $      0_8,byte_size,C_LOC(w),0,C_NULL_PTR,C_NULL_PTR)
!         if (err.ne.0) stop 'clEnqueueReadBuffer'
         !print *, g(n*6-4:n*6) 
!         err=clFinish(cmd_queue)
!         if (err.ne.0) stop 'clFinish'
         !call ax_fpga(w,p,g,ur,us,ut,wk,n)                                    ! flopa
 
!         call dssum(w)         ! Gather-scatter operation  ! w   = QQ  w
                                                              !            L
!         err=clEnqueueWriteBuffer(cmd_queue,cl_w,CL_TRUE,0_8,
!     $    byte_size,C_LOC(w), 0,C_NULL_PTR,C_NULL_PTR)
!         if (err.ne.0) stop 'clEnqueueWriteBuffer'
         !call add2s2(w,p,.1,n)   !2n
         !call maskit(w,cmask,nx1,ny1,nz1)  ! Zero out Dirichlet conditions
   

c         err=clEnqueueNDRangeKernel(cmd_queue,kernel,
c     $    1,C_NULL_PTR,C_LOC(globalsize),C_NULL_PTR,
c     $    0,C_NULL_PTR,C_NULL_PTR)
c         if (err.ne.0) stop 'clEnqueueNDRangeKernel'
!         err=clFinish(cmd_queue)
!         if (err.ne.0) stop 'clFinish'
!         err=clEnqueueTask(cmd_queue,kernel,
!     $    0,C_NULL_PTR,C_NULL_PTR)
!         if (err.ne.0) stop 'clEnqueueEnqueueTask'
         !pap=glsc3(w,c,p,n)                                              ! 3n

         !alpha=rtz1/pap
         !alphm=-alpha
 
         !call add2s2(x,p,alpha,n)                                        ! 2n
         !call add2s2(r,w,alphm,n)                                        ! 2n
         
         !rtr = glsc3(r,c,r,n)                                            ! 3n
         
!         err = clEnqueueReadBuffer(cmd_queue,cl_rtr,CL_TRUE,
!     $      0_8,8_8,C_LOC(rtr),0,C_NULL_PTR,C_NULL_PTR)
!         if (err.ne.0) stop 'clEnqueueReadBuffer'
!         err=clFinish(cmd_queue)
!         if (err.ne.0) stop 'clFinish'
!         
!         if (iter.eq.1) rlim2 = rtr*eps**2
!         if (iter.eq.1) rtr0  = rtr
!         rnorm = sqrt(rtr)
c        if (nid.eq.0.and.mod(iter,100).eq.0) 
c    $      write(6,6) iter,rnorm,alpha,beta,pap
!    6    format('cg:',i4,1p4e12.4)
c        if (rtr.le.rlim2) goto 1001

      enddo

 1001 continue
      nxyz=nx1*ny1*nz1
      flop_a = flop_a 
     $       + (dble(19) * dble(nxyz) + dble(12) *dble(nx1)* dble(nxyz))
     $       * dble(nelt) * dble(miter)
!      if (nid.eq.0) write(6,6) iter,rnorm,alpha,beta,pap
      err=clFinish(cmd_queue)
      if (err.ne.0) stop 'clFinish'
      err = clReleaseCommandQueue(cmd_queue) 
      err = clReleaseContext(context)  
 
      return
      end
c      ---------------------------------------------------------------

      subroutine ax_fpga(w,u,gxyz,ur,us,ut,wk,n) ! Matrix-vector product: w=A*u

      include 'SIZE'
      include 'TOTAL'

      real w(nx1*ny1*nz1,nelt),u(nx1*ny1*nz1,nelt)
      real gxyz(2*ldim,nx1*ny1*nz1,nelt)

      parameter (lt=lx1*ly1*lz1*lelt)
      real ur(lt),us(lt),ut(lt),wk(lt)
      common /mymask/cmask(-1:lx1*ly1*lz1*lelt)

      integer e


      do e=1,nelt                                ! ~
         call ax_e( w(1,e),u(1,e),gxyz(1,1,e)    ! w   = A  u
     $                             ,ur,us,ut,wk) !  L     L  L
      enddo                                      ! 
      return
      end


#endif FPGA
