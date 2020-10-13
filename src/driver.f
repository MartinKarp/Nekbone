c-----------------------------------------------------------------------
      program nekbone

      include 'SIZE'
      include 'TOTAL'
      include 'SEMHAT'
      include 'mpif.h'

      common /mymask/cmask(-1:lx1*ly1*lz1*lelt)
      parameter (lxyz = lx1*ly1*lz1)
      parameter (lt=lxyz*lelt)

      real w(lt)
      real p(lx1,ly1,lz1,lelt)

      real g(6,lt)
      real mfloplist(1024), avmflop
      integer icount

      logical ifbrick
      integer iel0,ielN,ielD   ! element range per proc.
      integer nx0,nxN,nxD      ! poly. order range
      integer npx,npy,npz      ! processor decomp
      integer mx ,my ,mz        ! element decomp
      integer iter


      call iniproc(mpi_comm_world)    ! has nekmpi common block
      call init_delay

      call read_param(ifbrick,iel0,ielN,ielD,nx0,nxN,nxD,
     $                               npx,npy,npz,mx,my,mz)

c     GET PLATFORM CHARACTERISTICS
c      iverbose = 1
c     call platform_timer(iverbose)   ! iverbose=0 or 1

      icount = 0

c     SET UP and RUN NEKBONE

#ifdef _OPENACC

!$ACC  DATA CREATE(w,p)
!$ACC&      CREATE(g,dxm1,dxtm1)
      do nx1=nx0,nxN,nxD
         call init_dim
         do iter=iel0,ielN,ielD
            nelt = 2**iter
           call init_mesh(ifbrick,cmask,npx,npy,npz,mx,my,mz)

           call rone(w,n)
           call rone(p,n)
           call proxy_setup(ah,bh,ch,dh,zh,wh,g)
!$ACC UPDATE DEVICE(w,p,g,dxm1,dxtm1)
           niter = 50000
           n     = nx1*ny1*nz1*nelt

           if(nid.eq.0) write(6,*)

           call nekgsync()

           call bk5_acc(g,w,p,n,niter)


           call nekgsync()
           
           icount = icount + 1
           mfloplist(icount) = mflops*np
         enddo
      enddo
!$ACC END DATA
#endif
      avmflop = 0.0
      do i = 1,icount
         avmflop = avmflop+mfloplist(i)
      enddo
      if(icount.ne.0) then
         avmflop=avmflop/icount
      endif
      if(nid.eq.0) then
         write(6,1) avmflop
      endif
  1   format('Avg MFlops = ',1pe12.4)
c     TEST BANDWIDTH BISECTION CAPACITY
c     call xfer(np,cr_h)

      call exitt0

      end
c--------------------------------------------------------------
      subroutine init_dim

C     Transfer array dimensions to common

      include 'SIZE'

      ny1=nx1
      nz1=nx1

      ndim=ldim

      return
      end
c-----------------------------------------------------------------------
      subroutine init_mesh(ifbrick,cmask,npx,npy,npz,mx,my,mz)
      include 'SIZE'
      include 'TOTAL'
      real cmask(-1:nx1*ny1*nz1*nelt)
      logical ifbrick
      integer e,eg,offs,npx,npy,npz,mx,my,mz

c     Trigger reset of mask
      cmask(-1) = 1.0

      if(.not.ifbrick) then   ! A 1-D array of elements of length P*lelt
         nelx = nelt*np
         nely = 1
         nelz = 1

         npx=np
         npy=1
         npz=1

         mx=nelt
         my=1
         mz=1

         if(nid.eq.0) then
           write(6,*)
           write(6,*) 'Processor Distribution:  npx,npy,npz=',npx,
     $                npy,npz
           write(6,*) 'Element Distribution: nelx,nely,nelz=',nelx,
     $                nely,nelz
           write(6,*) 'Local Element Distribution: mx,my,mz=',mx,
     $                my,mz
         endif

         do e=1,nelt
            eg = e + nid*nelt
            lglel(e) = eg
         enddo
      else              ! A 3-D block of elements
         !xyz distribution of total proc if user-provided isn't valid
         if(npx*npy*npz.ne.np) then
            call cubic(npx,npy,npz,np)
         endif

         !xyz distribution of total NELT if user-provided isn't valid
         if(mx*my*mz.ne.nelt)  then
            call cubic(mx,my,mz,nelt)
         endif

         nelx = mx*npx
         nely = my*npy
         nelz = mz*npz

         if(nid.eq.0) then
           write(6,*)
           write(6,*) 'Processor Distribution:  npx,npy,npz=',npx,
     $                npy,npz
           write(6,*) 'Element Distribution: nelx,nely,nelz=',nelx,
     $                nely,nelz
           write(6,*) 'Local Element Distribution: mx,my,mz=',mx,
     $                my,mz
         endif

         e = 1
         offs = (mod(nid,npx)*mx) + npx*(my*mx)*(mod(nid/npx,npy))
     $       + (npx*npy)*(mx*my*mz)*(nid/(npx*npy))
         do k = 0,mz-1
         do j = 0,my-1
         do i = 0,mx-1
            eg = offs+i+(j*nelx)+(k*nelx*nely)+1
            lglel(e) = eg
            e        = e+1
         enddo
         enddo
         enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine cubic(mx,my,mz,np)

      mx = np
      my = 1
      mz = 1
      ratio = np

      iroot3 = np**(1./3.) + 0.000001
      do i = iroot3,1,-1
        iz = i
        myx = np/iz
        nrem = np-myx*iz

        if (nrem.eq.0) then
          iroot2 = myx**(1./2.) + 0.000001
          do j=iroot2,1,-1
            iy = j
            ix = myx/iy
            nrem = myx-ix*iy
            if (nrem.eq.0) goto 20
          enddo
   20     continue

          if (ix.lt.iy) then
            it = ix
            ix = iy
            iy = it
          endif

          if (ix.lt.iz) then
            it = ix
            ix = iz
            iz = it
          endif

          if (iy.lt.iz) then
            it = iy
            iy = iz
            iz = it
          endif

          if ( REAL(ix)/iz.lt.ratio) then
            ratio = REAL(ix)/iz
            mx = ix
            my = iy
            mz = iz
          endif
        endif
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine cubic0(mx,my,mz,np)

        rp = np**(1./3.)
        mz = rp*(1.01)

        do iz=mz,1,-1
           myx = np/iz
           nrem = np-myx*iz
           if (nrem.eq.0) goto 10
        enddo
   10   mz = iz
        rq = myx**(1./2.)
        my = rq*(1.01)
        do iy=my,1,-1
           mx = myx/iy
           nrem = myx-mx*iy
           if (nrem.eq.0) goto 20
        enddo
   20   my = iy

        mx = np/(mz*my)

      return
      end

c-----------------------------------------------------------------------
      subroutine set_timer_flop_cnt(iset)
      include 'SIZE'
      include 'TOTAL'

      real time0,time1
      save time0,time1

      if (iset.eq.0) then
         flop_a  = 0
         flop_cg = 0
         time0   = dnekclock()
      else
        time1   = dnekclock()-time0
        if (time1.gt.0) mflops = (flop_a+flop_cg)/(1.d6*time1)
        if (nid.eq.0) then
          write(6,*)
          write(6,1) nelt,np,nx1, nelt*np
          write(6,2) mflops*dble(np), mflops
          write(6,3) flop_a,flop_cg
          write(6,4) time1
        endif
    1   format('nelt = ',i7, ', np = ', i9,', nx1 = ', i7,
     &         ', elements =', i10 )
    2   format('Tot MFlops = ', 1pe12.4, ', MFlops      = ', e12.4)
    3   format('Setup Flop = ', 1pe12.4, ', Solver Flop = ', e12.4)
    4   format('Solve Time = ', e12.4)
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine load_points(buffer,nppp,npoints,npts,nbuf)
      include 'SIZE'
      include 'PARALLEL'

      real buffer(2,nbuf)

      nppp=0
      if(nbuf.gt.npts) then
       npass = 1+npoints/nbuf

       do ipass = 1,npass
          if(nid.eq.ipass.and.ipass.ne.npass) then
            do i = 1,nbuf
             buffer(1,i)=i
             buffer(2,i)=nid
            enddo
            nppp=nbuf
          elseif (npass.eq.ipass.and.nid.eq.0) then
            mbuf=mod1(npoints,nbuf)
            do i=1,mbuf
               buffer(1,i)=i
               buffer(2,i)=nid
            enddo
            nppp=mbuf
          endif
       enddo
      else
       do i = 1,npts
          buffer(1,i)=i
          buffer(2,i)=nid
       enddo
       nppp=npts
      endif

      return
      end
c----------------------------------------------------------------------
      subroutine read_param(ifbrick,iel0,ielN,ielD,nx0,nxN,nxD,
     $                                     npx,npy,npz,mx,my,mz)
      include 'SIZE'
      include 'INPUT'
      include 'HSMG'
      logical ifbrick
      integer iel0,ielN,ielD,nx0,nxN,nxD,npx,npy,npz,mx,my,mz

      !open .rea
      ifbrick = .false.
      ifmgrid = .false.  !initialize to false
      npx=0
      npy=0
      npz=0
      mx =0
      my =0
      mz =0

      if(nid.eq.0) then
         open(unit=9,file='data.rea',status='old')
         read(9,*,err=100) ifbrick
         read(9,*,err=100) iel0,ielN,ielD
         read(9,*,err=100) nx0,nxN,nxD
         read(9,*,err=100,iostat=ii) npx,npy,npz !optional
         read(9,*,err=100,iostat=ii) mx,my,mz    !optional
         close(9)
      endif
      call bcast(ifbrick,4)

      call bcast(iel0,4)! ELEMENT range
      call bcast(ielN,4)
      call bcast(ielD,4)

      call bcast(nx0,4) ! POLY Order range
      call bcast(nxN,4)
      call bcast(nxD,4)

      call bcast(npx,4) ! PROC decomp
      call bcast(npy,4)
      call bcast(npz,4)

      call bcast(mx,4)  ! NELT decomp
      call bcast(my,4)
      call bcast(mz,4)

#ifdef MGRID
      ifmgrid = .true.
#endif

      if(iel0.gt.ielN.or.nx0 .gt.nxN)    goto 200
      if(ielN.gt.lelt.or.nxN .gt.lx1)    goto 210
      if(ielD.gt.ielN.or.nxD.gt.nxN)     goto 220
      if(nx0.lt.4.and.ifmgrid)           goto 230

      if(nid.eq.0) write(6,*) "ifmgrid    :",ifmgrid
     $                   ,"    ifbrick    :",ifbrick

      return

  100 continue
      write(6,*) "ERROR READING    data.rea.....ABORT"
      write(6,*) "CHECK PARAMETERS data.rea.....ABORT"
      call exitt0

  200 continue
      write(6,*) "ERROR data.rea :: iel0 > ielN or nx0 > nxN :: ABORT"
      call exitt0

  210 continue
      write(6,*) "ERROR data.rea : ielN>lelt or nxN>lx1(SIZE) :: ABORT"
      call exitt0

  220 continue
      write(6,*) "WARNING data.rea : STRIDE   nxD>nxN or ielD>ielN !!!"

  230 continue
      write(6,*) "ERROR data.rea nx0 must be greater than or equal to 4"
     $          ," ...setting nx0=4"
      nx0=4


      return
      end
c-----------------------------------------------------------------------

