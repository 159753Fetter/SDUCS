;PROGRAM 	TITLE	GOES	HERE	--	HANOI
;Solves tower of HANOI puzzle.Printout sequence of moves
;of N discs from initial spindle X to final spindle Z.
;Using spindle Y for temporery storage.
;***********************************************************************
datarea			segment							;define data segment
  message1		db				'N=?',0AH,0DH,'$'
  message2		db				'What is the name of spindle X?'
  				db				0AH,0DH,'$'
  message3		db				'What is the name of spindle Y?'
  				db				0AH,0DH,'$'
  message4		db				'What is the name of spindle Z?'
  				db				0AH,0DH,'$'
  flag			dw				0
  constant		dw				10000,1000,100,10,1
datarea			ends
;***********************************************************************
prognam			segment							;define code segment
;-----------------------------------------------------------------------
main			proc			far
				assume cs:prognam,ds:datarea
start:
;set up stack for return
				push			ds
				sub				ax,ax
				push			ax
;set DS register to current data segment
				mov				ax,datarea
				mov				ds,ax
;main PART OF PROGRAM GOES HERE
				lea 			dx,message1		;N = ?
				mov				AH,09H
				int				21h
				call 			decibin			;read N into BX
				call  			crlf
;
				cmp				bx,0			;if N = 0
				jz				exit			;exit
;
				lea				dx,message2		;X = ?
				mov				AH,09H
				int				21H
				mov				AH,01H			;read X's name into CX
				int				21H
				mov				AH,0
				mov				cx,ax
				call			crlf
;
				lea				dx,message3		;Y = ?
				mov				AH,09H
				int				21H
				mov				AH,01H			;read Y's name into CX
				int				21H
				mov				AH,0
				mov				si,ax
				call			crlf
;
				lea				dx,message4		;Z = ?
				mov				AH,09H
				int				21H
				mov				AH,01H			;read Z's name into CX
				int				21H
				mov				AH,0
				mov				di,ax
				call			crlf
;
				call			hanoi			;call HANOI(N,X,Y,Z)
;
exit:			ret								;return ro DOS
;
main			endp
;-----------------------------------------------------------------------
hanoi			proc			near			;define subprocedure 
;Solve tower of hanoi puzzle.
;Argement: (BX) = N, (CX) = X, (SI) = Y,(DI) = Z.
				cmp				bx,1			;if N = 1,execute basis 
				je				basis
				call			save			;save N,X,Y,Z
				dec				bx
				xchg			si,di
				call			hanoi			;call HANOI(N-1,X,Z,Y)
				call			restor			;restore N,X,Y,Z
				call			print
				dec				bx
				xchg			cx,si
				call			hanoi			;call HANOI(X01,Y,X,Z)
				jmp				return
basis:			call			print			
return:			ret								;return
hanoi			endp							;end subprocedure
;-----------------------------------------------------------------------
print			proc			near
;print XNZ
				mov				dx,cx			;print X
				mov				AH,02H		
				int 			21H
				call			binidec			;print N
				mov				dx,di			;print Z
				mov				AH,02H
				int				21h
				call			crlf			;skip to next line
				ret
print			endp
;-----------------------------------------------------------------------
save			proc			near
;push N,X,Y,Z onto stack
				pop				bp
				push			bx
				push			cx
				push			si
				push			di
				push			bp
				ret
save			endp
;-----------------------------------------------------------------------
restor			proc			near
;pop Z,Y,X,N from stack
				pop				bp
				pop				di
				pop				si
				pop				cx
				pop				bx
				push			bp
				ret
restor			endp
;-----------------------------------------------------------------------
decibin			proc			near
;procedure to convert decimal on keybd to binary.
;result is left in BX register.
				mov				bx,0			;clear BX for number
;get digit from keyboard,convert to binart
newchar:	
				mov				AH,1			;keyboard input
				int				21H				;call DOS
				sub				AL,30h			;ASCII to binary
				jl				exit1			;jump if < 0
				cmp				AL,9D			;is it > 9D
				jg				exit1			;yes,not dec digit
				cbw								;BYTE in AL to WORD in AX
;(digit is now in AX)
;multiply number in BX by 10 decimal.
				xchg			AX,BX			;trade digit & number
				mov				cx,10D			;put 10 dec in CX
				mul				cx
				xchg			AX,BX			;trade number & digit
;add digit in AX to number in BX
				add				BX,AX			;add digit to number
				jmp				newchar			;get next digit
exit1:			ret								;return from decibin
decibin			endp							;end of decibin	proc
;-----------------------------------------------------------------------
binidec			proc			near
;procedure to convert binart number in BX to decimal
;on console screen
				push			BX
				push			CX
				push			SI
				push			DI
				mov				flag,0
				mov				cx,5
				lea				si,constant
;
dec_div:	
				mov				AX,BX			;number high half
				mov				dx,0			;zero out low half
				div				WORD ptr[si]	;divide by constant
				mov				bx,dx			;remainder into BX
				mov				DL,AL			;quotient into DL
;
				cmp				flag,0			;have not leading zero
				jnz				print1
				cmp				DL,0
				je				skip
				mov				flag,1
;print	the contents of DL on screen
print1:			add				DL,30H			;convert to ASCII
				mov				AH,02H			;display function
				int				21h				;call DOS
skip:			add				si,2
				loop			dec_div
				pop				di
				pop				si
				pop				cx
				pop				bx
				ret								;return from dec_div
binidec			endp
;-----------------------------------------------------------------------
crlf			proc			near
;print carriage return and linefeed
				mov				DL,0AH			;linefeed
				mov				AH,02H			;display function
				int				21H
;
				mov				DL,0DH			;carriage return
				mov				AH,02H			;display function
				int				21h
;
				ret
crlf			endp
;-----------------------------------------------------------------------
prognam			ends							;end of code segment
;***********************************************************************
				end		start					;end assembly