      program vladimir               ! Иванов Владимир Александрович 09.09.2023 г.
      implicit real*8 (A-H,O-Z)
      parameter (nn=202,mm=202)
      dimension A0(nn,mm), A1(nn,mm), A2(nn,mm), A3(nn,mm), A4(nn,mm),
     1          A5(nn,mm), A6(nn,mm), A7(nn,mm), A8(nn,mm), X (nn,mm),
     2          F (nn,mm), Xn(nn,mm), XX(nn)   , YY(mm) 
      DIMENSION AL0(NN,MM),  AL5(NN,MM),  AL7(NN,MM),
     2    PP(NN,MM),FIL(NN,MM),YC(NN,MM),R(NN,MM)
      
      open(unit=9,file='output.plt')
      open(unit=10,file='iter.dat')
      open(unit=11,file='X(t0).plt')
      pi = 3.1415926535D0
      EPSH = 1.0D-12
      ALOK = 1.0D-02
      tau = 0.001D0
      np = 100
      timemax = 5.0D0
      ifile = 20
      ifile1 = 200
      ifile2 = 300
      N1 = 16
      N2 = 17
      M1 = 16
      M2 = 17
c
      N1 = 201
      N2 = 202
      M1 = 201
      M2 = 202
      hx = 1.0D0/(dfloat(N1)-0.5D0)
      do i=1,N2
         XX(i) = (dfloat(i)-1.5D0)*hx
      end do
      do j=1,M2
         YY(j) = (dfloat(j)-1.5D0)*hx
      end do
!      
      do i=1,N2
          do j=1,M2
              A0(i,j)  =  4.0D0 + hx**2/tau
              AL0(i,j) =  4.0D0
              AL5(i,j) =  0.0D0
              AL7(i,j) =  0.0D0
              A1(i,j)  = -1.0D0   
              A3(i,j)  = -1.0D0
              A5(i,j)  = -1.0D0   
              A7(i,j)  = -1.0D0   
              A2(i,j)  =  0.0D0   
              A4(i,j)  =  0.0D0
              A6(i,j)  =  0.0D0   
              A8(i,j)  =  0.0D0
              F(i,j)   =  0.0D0
              X (i,j)  =  dsin(pi/2.0D0*(1.0D0-XX(i))) *
     1                    dsin(pi/2.0D0*(1.0D0-YY(j)))    
              Xn(i,j)  =  X (i,j)
          end do
      end do
c       PLOT
       write(11,10)
       write(11,11) N2,M2
      do i=1,N2
          do j=1,M2
              write(11,12) XX(i),YY(j),Xn(i,j)
          end do
      end do

c 
c      B.C.
c
c                T=1
c      _______________________
c     |                       |
c     |                       |
c     |                       |
c     |                       |
c  T=0|                       |  T=0
c     |                       |
c     |                       |
c     |                       |
c     |_______________________|
c                 T=0
      IGC = 1
      IGC = 2
      go to (30,40),IGC
c
c         Задача Дирихле на всех границах
!
30    continue
      j=2
      jj = M2
      do i=1,N2
          A7(i,j)  = 0.0D0
          A7(i,jj) = 0.0D0
          F(i,M1)  = 1.0D0
      end do
c
      i=2
      ii = N2
      do j=2,M1
          A5(i,j)  = 0.0D0
          A5(ii,j) = 0.0D0
      end do
      go to 50
c
c         Задача Нэймана на левой и ни нижней границе границах
!
40    continue
      do i=1,N2
          do j=1,M2
              A0(i,j)  =  2.0D0 + hx**2/tau
              A5(i,j)  = -0.5D0   
              A7(i,j)  = -0.5D0   
              A2(i,j)  =  0.0D0   
              A4(i,j)  =  0.0D0
              A6(i,j)  =  0.0D0   
              A8(i,j)  =  0.0D0
              F(i,j)   =  0.0D0
              X (i,j)  =  dsin(pi/2.0D0*(1.0D0-XX(i))) *
     1                    dsin(pi/2.0D0*(1.0D0-YY(j)))    
              Xn(i,j)  =  X (i,j)
          end do
      end do
c             UP   |~~~~~~|
c                  |      |
!
      jj = M2
      do i=1,N2
          A7(i,jj) = 0.0D0
      end do
!                  |      |
!     Du           |______|
!
      j=2
      do i=1,N2
          A0(i,j) = A0(i,j) + A7(i,j)
          A7(i,j) = 0.0D0
      end do
c           R           ___
!                          |
!                          |
!                       ---
      ii = N2
      do j=2,M1
          A5(ii,j) = 0.0D0
      end do
!           L
!       ----
!       |
!       |
!       ----
      i=2
      do j=2,M1
          A0(i,j) = A0(i,j) + A5(i,j)
          A5(i,j)  = 0.0D0
      end do
!     
50    continue
!
      CALL ALT(A0, A5, A7, AL0, AL5, AL7, ALOK, N2,M2)
!
      nstep = 0
      time  = 0.0D0
200   continue
      nstep = nstep + 1
      time  = time  + tau
!-----------------------------------------------------------------------
      do i=2,N1
          do j=2,M1         
             F(i,j)  =  2.0D0 * hx**2 * Xn(i,j)/tau 
     1  -(A0(i,j)*Xn(i,j)+A5(i+1,j)*Xn(i+1,j)+A5(i,j)*Xn(i-1,j)
     2                   +A7(i,j+1)*Xn(i,j+1)+A7(i,j)*Xn(i,j-1))
          end do
      end do
!-----------------------------------------------------------------------      
!      SUBROUTINE SOLVE
!     1             (A0,A5,   A7,   AL0,AL5,    AL7,    HEB,PP,R,DP,
!     2              FIL,EPSH,YY,N,M)
!------------------------------------------------------------

      call SOLVE (A0,A5,   A7,   AL0,AL5,    AL7,F,PP,R,Xn,
     2              FIL,EPSH,YC,N2,M2)
      do i=1,N2
          do j=1,M2
              X(i,j) = Xn(i,j)
          end do
      end do
      j=1
       jj=M2
      do i=1,N2
          X(i,j)  = X(i,j+1)
          X(i,jj) = 0.0D0
      end do
c
       i=1
       ii=M2
      do j=1,M1
          X(i,j)  = X(i+1,j)
          X(ii,j) = 0.0D0
      end do
!
      do i=1,N2
          do j=1,M2
              Xn(i,j) = X(i,j)
          end do
      end do
c
      if(nstep/np*np .eq. nstep) then
          ifile = ifile + 1
          ifile1 = ifile1 + 1
          ifile2 = ifile2 + 1
         write(ifile2,10)
         write(ifile2,11) N2,M2
         write(ifile1,10)
         write(ifile1,11) N2,M2
         write(ifile ,10)
         write(ifile ,11) N2,M2
      do i=1,N2
          xa = XX(i)
          do j=1,M2
              ya  = YY(j)
              sol = DEXP(-pi**2/2.0D0*time)*
     2        dsin(pi/2.0D0*(1.0D0-XX(i))) *
     1        dsin(pi/2.0D0*(1.0D0-YY(j)))
              dsol = DABS(sol-Xn(i,j))
              write(ifile ,12) xa,ya,X(i,j)
              write(ifile1,12) xa,ya,dsol
              write(ifile2,12) xa,ya, sol
          end do
      end do
         write(ifile2,*) ' time = ',time
         write(ifile1,*) ' time = ',time
         write(ifile ,*) ' time = ',time
       end if
      if(time .le. timemax) go to 200
      stop
10    format('VARIABLS="X","Y","Z"')
11    format('ZONE I=    ',I5' ,','J=    ',I5,',','F=POINT')
12    format(3(1PE14.5))
      end
      