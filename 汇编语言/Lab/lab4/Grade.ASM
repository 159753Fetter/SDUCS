;PROGRAM TITLE GOES HERE--RANK
;****************************************************************************
DATAREA				SEGMENT										;define data segment
GRADE				DW					50 DUP(?)
RANK				DW					50 DUP(?)
COUNT				DW					?
MESS1				DB					'GRADE? $'
MESS2				DB					13,10,'Input Error!',13,10,'$'
MESS3				DB					'RANK:$'
DATAREA				ENDS
;****************************************************************************
PROGNAM				SEGMENT										;define code segment
;----------------------------------------------------------------------------
MAIN				PROC				FAR						;main part of program
					ASSUME				CS:PROGNAM,DS:DATAREA
START:															;starting execution address														;starting execution address
;set up stack for return
					PUSH				DS						;save old data segment
					SUB					AX, AX					;put zero in AX
					PUSH				AX						;save it on stack
;set DS register to current data segment
					MOV					AX, DATAREA				;datarea segment addr
					MOV					DS, AX					;into DS register
;MAIN PART OF PROGRAM GOES HERE
					CALL				INPUT
					CALL				RANKP
					CALL				OUTPUT
					RET
MAIN				ENDP
;----------------------------------------------------------------------------
INPUT				PROC				NEAR
					LEA					DX, MESS1
					MOV					AH, 09
					INT					21H
					MOV					SI, 0
					MOV					COUNT, 0
ENTERING:
					CALL				DECIBIN
					INC					COUNT
					CMP					DL, ','					;is it ','?
					JE					STORE
					CMP					DL, 13					;is it'return?
					JE					EXIT2
					JNE					ERROR
STORE:
					MOV					GRADE[SI], BX
					ADD					SI, 2
					JMP					ENTERING
ERROR:
					LEA					DX, MESS2
					MOV					AH, 09
					INT					21H
EXIT2:
					MOV					GRADE[SI], BX
					CALL				CRLF
					RET
INPUT				ENDP
;----------------------------------------------------------------------------
RANKP				PROC				NEAR
					MOV					DI, COUNT
					MOV					BX, 0
LOOP1:
					MOV					AX, GRADE[BX]
					MOV					WORD PTR RANK[BX], 0
					MOV					CX, COUNT
					LEA					SI, GRADE
NEXT:
					CMP					AX, [SI]
					JG					NO_COUNT
					INC					WORD PTR RANK[BX]
NO_COUNT:
					ADD					SI, 2
					LOOP				NEXT
					ADD					BX, 2
					DEC					DI
					JNE					LOOP1
					RET											;return to DOS
RANKP				ENDP
;----------------------------------------------------------------------------
OUTPUT				PROC				NEAR
					LEA					DX, MESS3
					MOV					AH, 09
					INT					21H
					
					MOV					SI, 0
					MOV					DI, COUNT
NEXT1:
					MOV					BX, RANK[SI]
					CALL				BINIDEC					;display the rank
					MOV					DL, ','					;of students
					MOV					AH, 02
					INT					21H
					ADD					SI, 2
					DEC					DI
					JNZ					NEXT1
					CALL				CRLF
					RET
OUTPUT				ENDP
;----------------------------------------------------------------------------
DECIBIN				PROC				NEAR
;convert decimal on keybd to binary
;result is left in bx register
					MOV					BX, 0					;clear BX
NEWCHAR:
					MOV					AH, 1					;keyboard input
					INT					21H						;call DOS
					MOV					DL, AL
					SUB					AL, 30H					;ASCII TO binary
					JL					EXIT1					;if < 0 jump to exit1
					CMP					AL, 9D
					JG					EXIT1					;if it > 9d ,not dec digit
					CBW											;bute in AL to WORD in AX
;digit is now in AX
;multiply number in BX by decimal
					XCHG				AX, BX					;trade digit & number
					MOV					CX, 10D					;put 10 dec in CX
					MUL					CX						;number times 10
					XCHG				AX,BX					;trade number & digit
;add digit in AX to number in BX
					ADD					BX,AX
					JMP					NEWCHAR					;get next digit
EXIT1:				RET
DECIBIN				ENDP
;----------------------------------------------------------------------------
BINIDEC				PROC				NEAR
;convert binary number in BX to decimal
					PUSH				BX
					PUSH				CX
					PUSH				SI
					PUSH				DI
					MOV					CX, 100D				;divide by 100
					CALL				DEC_DIV
					MOV					CX, 10D					;divide by 10
					CALL				DEC_DIV
					MOV					CX, 1D					;divide by 1
					CALL				DEC_DIV
					POP					DI
					POP					SI
					POP					CX
					POP					BX
					RET											;return from binidec
BINIDEC				ENDP
;----------------------------------------------------------------------------
DEC_DIV				PROC				NEAR
;sub_subroutine to divide number in BX by number in CX
					MOV					AX, BX					;number high half
					MOV					DX, 0					;zero out low half
					DIV					CX						;divide by CX
					MOV					BX, DX					;remainder into BX
					MOV					DL, AL					;quotient into DL
					
					ADD					DL, 30H					;将BX的数字除以CX中的数字，在屏幕上显示商
					MOV					AH, 02H					;display function
					INT					21H						;call DOS
					RET											;return from dec_div
DEC_DIV				ENDP
;----------------------------------------------------------------------------
CRLF				PROC				NEAR
;print carriage return and linefeed
					MOV					DL, 0AH					;linefeed
					MOV					AH, 02H					;display function
					INT					21H
					
					MOV					DL, 0DH					;carriage return
					MOV					AH, 02H					;display function
					INT					21H
					
					RET
CRLF				ENDP
;----------------------------------------------------------------------------
PROGNAM				ENDS										;end of code segment
					END					START					;end assembly