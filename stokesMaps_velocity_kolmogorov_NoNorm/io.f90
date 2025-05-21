MODULE io
USE nrtype
IMPLICIT NONE

CONTAINS

      subroutine writer_r2D(dfile,field,n1,n2)
      real(SP),     dimension(:,:), intent(in)  :: field
      character(len=*),             intent(in)  :: dfile
      integer(I4B),                 intent(in)  :: n1,n2

      integer(I4B)                  :: i1,i2

      write(0,*)'Write down 2D array '
      open(1,file=TRIM(ADJUSTL(dfile)),status='unknown',form='formatted')
      do i1=1,n1
         write(1,*)(field(i1,i2),i2=1,n2)
      enddo
      close(1)
      return
      end subroutine writer_r2D

      subroutine writer_r(field,n1,n2,n3)
      real(SP), dimension(:,:,:), intent(in)  :: field
      integer(I4B),               intent(in)  :: n1,n2,n3

      integer(I4B)                  :: i1,i2,i3

      write(0,*)'Write down array in real space'
      do i3=1,1
         do i2=1,1
            write(*,*)(field(i1,i2,i3),i1=1,n1)
         enddo
      enddo
      return
      end subroutine writer_r

      subroutine writer_f(field,speq,n1,n2,n3)
      complex(SPC), dimension(:,:,:), intent(in)  :: field
      complex(SPC), dimension(:,:),   intent(in)  :: speq
      integer(I4B),                   intent(in)  :: n1,n2,n3

      complex(SPC)  :: temp
      real(SP)      :: magn
      integer(I4B)  :: i1,i2,i3

      write(0,*)'Write down array in Fourier space'
      do i3=1,n3
         do i2=1,n2
            do i1=1,n1/2
               temp = field(i1,i2,i3)
               magn = temp*conjg(temp)
               write(*,*)magn
            enddo
         enddo
      enddo
      return
      end subroutine writer_f

      subroutine write_spec_file(dfile,spec,l1,l2,l3,closed)
      complex(SPC),      intent(in), dimension(:,:,:)  :: spec
      character(len=*),  intent(in)  :: dfile
      integer(I4B),      intent(in)  :: l1,l2,l3
      logical,           intent(in)  :: closed

      integer(I4B), parameter        :: iformat=0
      integer(I4B)                   :: i,j,k

      if (iformat.eq.0) then
         open(1,file=dfile,status='unknown',form='unformatted')
         write(1) l1,l2,l3
         do k=1,l3
            write(1) ((spec(i,j,k),i=1,l1/2),j=1,l2)
         enddo
      else
         open(1,file=dfile,status='unknown',form='formatted')
         do k=1,l3
            do j=1,l2
               do i=1,l1/2
                  write(1,*)spec(i,j,k)
               enddo
            enddo
         enddo
      endif

      if ( closed ) close(1)

      return
      end subroutine write_spec_file

      subroutine write_3D_file(dfile,den,l1,l2,l3,closed,islice_request)
      real(SP),     dimension(:,:,:), intent(in)  :: den
      character(len=*),                  intent(in)  :: dfile
      integer(I4B),                      intent(in)  :: l1,l2,l3
      logical,                           intent(in)  :: closed
      integer(I4B), optional,            intent(in)  :: islice_request

      ! islice controls writing by (1) rows (needed for MKLFFT), 
      ! (2) planes (usual default), or (3), in one record
      integer(I4B), parameter        :: iformat=0
      integer(I4B)                   :: i,j,k,ndims=3, islice=2

      if (iformat.eq.0) then
         if ( present(islice_request) ) islice = islice_request
         open(1,file=dfile,status='unknown',form='unformatted')
         write(1) ndims
         write(1) l1,l2,l3,islice
         if ( islice == 1 ) then
            do k=1,l3
               do j=1,l2
                  write(1) (den(i,j,k),i=1,l1)
               enddo
            enddo
         else if ( islice == 3 ) then
            write(1) (((den(i,j,k),i=1,l1),j=1,l2),k=1,l3)
         else
            do k=1,l3
               write(1) ((den(i,j,k),i=1,l1),j=1,l2)
            enddo
         endif
      else
         open(1,file=dfile,status='unknown',form='formatted')
         do k=1,l3
            do j=1,l2
               do i=1,l1
                  if ( den(i,j,k) < 1. ) then
                     write(1,*) den(i,j,k),i,j,k
                  else
                     write(1,*)den(i,j,k),i,j
                  endif
               enddo
            enddo
         enddo
      endif

      if ( closed ) close(1)

      return
      end subroutine write_3D_file

      subroutine write_2D_file(dfile,den,l1,l2,closed,islice_request)
      real(SP),     dimension(:,:), intent(in)  :: den
      character(len=*),             intent(in)  :: dfile
      integer(I4B),                 intent(in)  :: l1,l2
      logical,                      intent(in)  :: closed
      integer(I4B), optional,       intent(in)  :: islice_request

      integer(I4B), parameter        :: iformat=0
      integer(I4B)                   :: i,j,ndims=2,islice=2
      integer(I4B)                   :: ios

      ! islice controls writing by (1) rows (needed for MKLFFT), 
      ! or (2), default,  in one record

      if (iformat.eq.0) then
         if ( present(islice_request) ) islice=islice_request
         open(1,file=dfile,status='unknown',form='unformatted')
         write(1) ndims
         write(1) l1,l2
         if ( islice == 1 ) then
            ! For MKL CCE format we need to store and read individual rows
            ! since first index range of den array can be larger than data
            do j=1,l2
               write(1,iostat=ios,err=100) (den(i,j),i=1,l1)
            enddo
         else
            ! Single record
            write(1,iostat=ios,err=100) ((den(i,j),i=1,l1),j=1,l2)
         endif
      else
         open(1,file=dfile,status='unknown',form='formatted')
         do j=1,l2
            do i=1,l1
               write(1,*)den(i,j),i,j
            enddo
         enddo
      endif

      if ( closed ) close(1)

100   if (ios /= 0) then
         write(0,*) 'Write out error'
         stop
      endif
      return
      end subroutine write_2D_file

      subroutine read_file_size(dfile,n1,n2,n3,ndim)
      character(len=*), intent(in)           :: dfile
      integer(I4B),     intent(out)          :: n1,n2
      integer(I4B),     intent(out),optional :: n3,ndim

      integer(I4B)                           :: ndimin
      integer(I4B)                           :: ios

      open(1,file=dfile,status='old',form='unformatted',iostat=ios)
      if (ios /= 0) then
         write(0,*)'Unable to open density input file, stop',dfile
         stop
      endif

      read(1) ndimin
      if ( ndimin == 3 ) then
         if ( .not.present(n3) ) then
            write(0,*) 'n3 output is not expected, stop'
            stop
         endif
         read(1) n1,n2,n3
      elseif ( ndimin == 2 ) then
         read(1) n1,n2
      else
         write(0,*)'Neither 2D nor 3D file, ndim=',ndim
         stop
      endif
      close(1)

      if ( present(ndim) ) ndim=ndimin
      return
      end subroutine read_file_size

      subroutine read_3D_file(dfile,den,n1,n2,n3)
      character(len=*),  intent(in)      :: dfile
      real(SP),          intent(inout)   :: den(:,:,:)
      integer(I4B),      intent(in)      :: n1,n2,n3

      integer(I4B)      :: ndim,i,j,k,ni1,ni2,ni3,ios
      integer(I4B)      :: islice

      open(1,file=dfile,status='old',form='unformatted',iostat=ios)
      if (ios /= 0) then
         write(0,*)'Unable to open input file, stop ',dfile
         stop
      endif

      ! Raw format
      read(1,iostat=ios,err=100) ndim
      read(1,iostat=ios,err=100) ni1,ni2,ni3,islice

      if ( (n1 /= ni1).or.(n2 /= ni2) ) then
         write(0,*)'The assumed read-in size of array mismatches'
         write(0,*) n1,n2,'differ from',ni1,ni2
         stop
      endif

      if ( n3 > ni3 ) then
         write(0,*) 'Not enough records', ni3, 'to read ', n3
         stop
      endif

      if ( islice == 1 ) then
         ! For MKL CCE format we need to store and read individual rows
         ! since first index range of den array can be larger than data
         do k=1,n3
            do j=1,n2
               read(1,iostat=ios,err=100) (den(i,j,k),i=1,n1)
            enddo
         enddo
      elseif ( islice == 3 ) then
         ! Single record
         read(1,iostat=ios,err=100) (((den(i,j,k),i=1,n1),j=1,n2),k=1,n3)
      else
         ! default, store and read 2D planes
         do k=1,n3
            read(1,iostat=ios,err=100) ((den(i,j,k),i=1,n1),j=1,n2)
         enddo
      endif

      close(1)
      return

100   if (ios /= 0) then
         write(0,*) 'Read-in error'
         stop
      endif

      end subroutine read_3D_file

      subroutine read_2D_file(dfile,den,n1,n2)
      character(len=*),         intent(in)      :: dfile
      real(SP), dimension(:,:), intent(inout)   :: den
      integer(I4B),             intent(in)      :: n1,n2

      integer(I4B)      :: ndim,i,j,ni1,ni2,ios
      integer(I4B)      :: islice=2   ! To be used later

      open(1,file=dfile,status='old',form='unformatted',iostat=ios)
      if (ios /= 0) then
         write(0,*)'Unable to open input file, stop ',dfile
         stop
      endif

      ! Raw format
      read(1,iostat=ios,err=100) ndim
      read(1,iostat=ios,err=100) ni1,ni2

      if ( (n1 /= ni1).or.(n2 /= ni2) ) then
         write(0,*)'The assumed read-in size of array mismatches'
         write(0,*) n1,n2,'differ from',ni1,ni2
         stop
      endif

      if ( islice == 1 ) then
         ! For MKL CCE format we need to store and read individual rows
         ! since first index range of den array can be larger than data
         do j=1,n2
            read(1,iostat=ios,err=100) (den(i,j),i=1,n1)
         enddo
      else
         ! Single record
         read(1,iostat=ios,err=100) ((den(i,j),i=1,n1),j=1,n2)
      endif

      close(1)
      return

100   if (ios /= 0) then
         write(0,*) 'Read-in error'
         stop
      endif

      end subroutine read_2D_file

      subroutine read_file_formatted(dfile,den,n1,n2,n3)
      character(len=*),  intent(in)    :: dfile
      real(SP),          intent(inout) :: den(:,:,:)
      integer(I4B),      intent(in)    :: n1,n2,n3

      integer(I4B)       :: i,j,k
      integer(I4B)       :: ios

      open(1,file=dfile,status='old',form='formatted',iostat=ios)
      if (ios /= 0) then
         write(0,*)'Unable to open density input file, stop',dfile
         stop
      endif

      do i=1,n1
         do j=1,n2
            do k=1,n3
               read(1,*) den(i,j,k)
            enddo
         enddo
      enddo

      close(1)
      return
      end subroutine read_file_formatted


END MODULE io
