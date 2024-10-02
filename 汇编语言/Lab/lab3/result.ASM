;PROGRAM TITLE GOES HERE -- RESULT
;********************************************************
DATAREA		SEGMENT						  ;define data segment
  GRADE		DW			56,69,84,82,73,88,99,63,100,80
  S5		DW			0
  S6		DW			0
  S7		DW			0
  S8		DW			0
  S9		DW			0
  S10		DW			0
DATAREA		ENDS
;*******************************************************
PROGNAM		SEGMENT						;define code segment
;-------------------------------------------------------
MAIN		PROC		FAR
			ASSUME CS:PROGNAM,DS:DATAREA
START:									;starting execution address
;set up stack for return 
			PUSH		DS				;save old data segment
			SUB			AX,AX			;put zero in AX
			PUSH		AX				;save it on stack 
;set DS register to current data segment
			MOV			AX,DATAREA		;datarea segment addr
			MOV			DS,AX			;into DS register
;MAIN PART OF PROGRAM GOES HERE
			MOV			S5,0			;initialize counter
			MOV			S6,0
			MOV			S7,0
			MOV			S8,0
			MOV			S9,0
			MOV			S10,0
			MOV			CX,10			;initialize loop count value
			MOV			BX,OFFSET GRADE	;initialize first addr
COMPARE:
			MOV			AX,[BX]			;get a result
			CMP			AX,60			; < 60?
			JL			FIVE
			CMP			AX,70			; < 70?
			JL			SIX
			CMP			AX,80			; < 80?
			JL			SEVEN
			CMP			AX,90			; < 90?
			JL			EIGHT
			CMP			AX,100			; = 100?
			JNE			NINE
			INC			S10
			JMP			SHORT CHANGE_ADDR		
NINE:
			INC			S9
			JMP			SHORT CHANGE_ADDR
EIGHT:
			INC			S8
			JMP			SHORT CHANGE_ADDR
SEVEN:
			INC			S7
			JMP			SHORT CHANGE_ADDR
SIX:
			INC			S6
			JMP			SHORT CHANGE_ADDR
FIVE:
			INC			S5
CHANGE_ADDR:
			ADD			BX,2
			LOOP		COMPARE
			RET							;return to DOS
MAIN		ENDP						;end of MAIN part of program
;-----------------------------------------------------------------
PROGNAM		ENDS
			END			START			;end assembly