!
!------------------------------------------------------------
!
      SUBROUTINE SOLVE
     1             (A0,A5,   A7,   AL0,AL5,    AL7,    HEB,PP,R,DP,
     2              FIL,EPSH,YY,N,M)
!      include 'real8.com'
      implicit real*8 (A-H,O-Z)
      parameter (nn=202,mm=202)
      DIMENSION A0(NN,MM),  A5(NN,MM),  A7(NN,MM),
     1           AL0(NN,MM),  AL5(NN,MM),  AL7(NN,MM),
     2    HEB(NN,MM),PP(NN,MM),DP(NN,MM),FIL(NN,MM),YY(NN,MM),R(NN,MM)
      COMMON /  NIT   / NIT
      N2 = N-1
      M2 = M-1
            zmaxheb = 0.0D0
            zmaxhe  = 0.0D0
      DO I=2,N2
          DO J=2,M2
             HEB(I,J)=
     *  A0(I,J)*DP(I,J)+A5(I+1,J)*DP(I+1,J)+ A5(I,J)*DP(I-1,J)+
     * A7(I,J+1)*DP(I,J+1)+ A7(I,J)*DP(I,J-1)-HEB(I,J)
      if(DABS(HEB(I,J)) .gt. zmaxhe) zmaxhe=DABS(HEB(I,J))
            zmaxheb = zmaxheb + HEB(I,J)*HEB(I,J)
c
          end do
      end do
      zmaxheb = DSQRT(zmaxheb/N2/M2)
c
      CALL LTL(HEB ,PP,FIL,AL0 ,AL5 ,AL7 ,N2,M2)
C
C              4.        iTEPAciOHHyj ciKl
C
      NIT = 0
 400  CONTINUE
      NIT = NIT+1
      write(10,410) nit, zmaxheb,zmaxhe
410   format('nit = ',I7, ' Heb = ',1PE14.5,' Heb1 = ',1PE14.5)
C
C              4.1.      By~iClEHiE MACCiBA R i CKAlqPHyX pPOizBEdEHij
C
      ZPHEB = 0.0D0
      ZPPR  = 0.0D0
      DO I=2,N2
         DO J=2,M2
            R(I,J)=A0(I,J)*PP(I,J)+A5(I+1,J)*PP(I+1,J)+
     *A7(I,J+1)*PP(I,J+1)                        +A5(I,J)*PP(I-1,J)+
     *                     A7(I,J)*PP(I,J-1)
          ZPHEB = ZPHEB+HEB(I,J)*PP(I,J)
          ZPPR  = ZPPR+R(I,J)*PP(I,J)
         end do
      end do
c
      ZAK=ZPHEB/(ZPPR+1.0D-20)
      IPK=0
      zmaxheb = 0.0D0
      zmaxhe  = 0.0D0
      DO I=2,N2
          DO J=2,M2
             ZAKP=ZAK*PP(I,J)
             DP(I,J)=DP(I,J)-ZAKP
             ZDPA = DABS(DP(I,J))
!             IF(ZDPA .GT. 1.) ZAKP = ZAKP / ZDPA
!             IF(DABS(ZAKP) .GT. EPSH) IPK=1
                HEB(I,J)=HEB(I,J)-ZAK*R(I,J)
                if(dabs(HEB(I,J)) .gt. zmaxhe) zmaxhe = DABS(HEB(I,J))
                zmaxheb = zmaxheb + HEB(I,J)*HEB(I,J)      
          end do
      end do
c
      zmaxheb = DSQRT(zmaxheb/N2/M2)
      if(zmaxheb .gt. EPSH) IPK=1
      IF(IPK.EQ.0) RETURN
C
C                        ByzOB  LTL
C
      CALL LTL(HEB ,YY,FIL,AL0 ,AL5 ,  AL7 ,   N2,M2)
       ZYYR = 0.
      DO I=2,N2
         DO J=2,M2
            ZYYR=ZYYR+YY(I,J)*R(I,J)
             ZBK=ZYYR/(ZPPR+1.0D-20)
         end do
      end do
c      
      DO I=2,N2
         DO J=2,M2
            PP(I,J)=YY(I,J)-ZBK*PP(I,J)
         end do
      end do
c      
      GO TO 400
      END
!
!-----------------------------------------------------------------
!
      SUBROUTINE LTL(F,X,FIL,AL0,AL5,    AL7,    N2,M2)
      implicit real*8 (A-H,O-Z)
      parameter (nn=202,mm=202)
      DIMENSION F(NN,MM),X(NN,MM),FIL(NN,MM),AL0(NN,MM),AL5(NN,MM),
     1          AL7(NN,MM)
C
C         1.    PEШEHИE  CИCTEMЫ LLT*X =F
C
      M1 = M2
      N1 = N2
      DO J=2,M1
         DO I=2,N1
            FIL(I,J)=(F(I,J)-AL5(I,J)*FIL(I-1,J)-
     *            AL7(I,J)*FIL(I,J-1)                      )/AL0(I,J)
         end do
      end do
c      
      DO JJ=2,M1
           J=M2-JJ+2
          DO II=2,N1
             I=N2-II+2
             X(I,J)=(FIL(I,J)-AL5(I+1,J)*X(I+1,J)-
     *         AL7(I,J+1)*X(I,J+1)                        )/AL0(I,J)
         end do
      end do
c      
      RETURN
      END
c---------------------------------------------------------------------
      SUBROUTINE ALT(A0,A5,A7,AL0,AL5,    AL7, ALOK,   N2,M2)
      implicit real*8 (A-H,O-Z)
      parameter (nn=202,mm=202)
      DIMENSION A0(NN,MM),A5(NN,MM), A7(NN,MM), AL0(NN,MM), AL5(NN,MM),
     1          AL7(NN,MM)
      N1 = N2-1
      M1 = M2-1
C
C          Неполная декомпозиция  холецкого  A = L * LT,
C              L - нижняя треугольная матрица
C              ( переход к  AL0, AL5, AL7  )
C-------------------------------------------------------
!      DO 21 I=1,N2
!         DO 21 J=1,M2
!            AL0(I,J) = 1.0D0
!            AL5(I,J) = 0.0D0
!            AL6(I,J) = 0.0D0
!            AL7(I,J) = 0.0D0
!            AL8(I,J) = 0.0D0
! 21   CONTINUE
!      ALOK=1.0D-03
!      ALOK=2.0D-01
!      ALOK = 1.0D0
!-----------------------------------------------------------------------
!
!          Нумерация вдоль I.
!
!      DO J=2,M1
!         DO I=2,N1
!      AL6(I,J)=A6(I,J)/(AL0(I-1,J-1)+ALOK)
!      AL7(I,J) = A7(I,J) / (AL0(I,J-1)+ALOK)
!      AL8(I,J)=(A8(I,J)-AL7(I,J)*AL5(I+1,J-1))/AL0(I+1,J-1)
!      AL5(I,J)=  A5(I,J) / (AL0(I-1,J)+ALOK)
!     1         /AL0(I-1,J)
!      AL0(I,J)=DSQRT(A0(I,J) - AL7(I,J)**2 - AL5(I,J)**2)
!
!
!       Для 5 - ти диаг. А.
!
!
!      Write(22,23) i,j,AL0(I,J),AL7(I,J),AL5(I,J),A0(I,J),A7(I,J)
!     1                ,A5(I,J)
!      IF(AL0(I,J) .LE. 0.0D0) then
!      Write(22,23) i,j,AL0(I,J),AL7(I,J),AL5(I,J),A0(I,J),A7(I,J)
!     1                ,A5(I,J)
!	AL0(I,J) = ALOK
!      stop
!                              end if
!         end do
!      end do
!
C
C
C---------------------------------------------------------------------
C              неполная декомпозиция Холецкго  A = L * LT,
C              L - нижняя треугольная матрица, LT - транспонированная к ней
C              ( переход к  AL0, AL5, AL7  )
C-------------------------------------------------------
      DO J=1,M
         DO I=1,N
            AL0(I,J) = 1.0D0
            AL5(I,J) = 0.0D0
            AL7(I,J) = 0.0D0
         END DO
	END DO
C
!      ALOK = 1.0D0
      DO J=2,M1
         DO I=2,N1
C
C---------------------------------------------------------------------
C
            AL7(I,J) = A7(I,J)
            AL5(I,J) = A5(I,J)
            AL0(I,J) = A0(I,J) - AL7(I,J)**2 / AL0(I,J-1)
     *                         - AL5(I,J)**2 / AL0(I-1,J)
            IF(AL0(I,J) .LE. 0.0D0) AL0(I,J) = ALOK
C
C---------------------------------------------------------------------
C
         END DO
	END DO
C
C--------------------------------------------------------------
C
      DO J=3,M1
         DO I=2,N1
            ZSQ1         = 1.0D0/DSQRT(AL0(I,J))
            AL0(I  ,J  ) = AL0(I  ,J  ) * ZSQ1
            AL5(I+1,J  ) = AL5(I+1,J  ) * ZSQ1
            AL7(I  ,J+1) = AL7(I  ,J+1) * ZSQ1
         END DO
      END DO
C
C----------------------------------------------------------------------
C
      
      return
      end