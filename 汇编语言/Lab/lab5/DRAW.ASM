TITLE DRAW -- Program to draw on screen with
;	cursor arrows,character write to video memory
;-----------------------------------------------------------------------
READ_C			EQU				0H				;read character code
KEY_ROM			EQU				16H				;ROM keyboard routine
UP				EQU				48H				;scan code for up arrow
DOWN			EQU				50H				;scan code for down arrow
RIGHT			EQU				4DH				;scan code for rigth arrow
LEFT			EQU				4BH				;scan code for left arrow
BLOCKING		EQU				0DBH			;solid graphics character
ESCING			EQU				1BH				;Escape key
;***********************************************************************
VIDEO			SEGMENT			AT 0B800H		;define extra seg
WD_BUFF			LABEL			WORD
V_BUFF			DB				25 * 80 * 2	DUP(?)
VIDEO			ENDS
;***********************************************************************
PRO_NAM			SEGMENT							;define code segment
;-----------------------------------------------------------------------
MAIN			PROC			FAR				;MAIN part of program
                ASSUME			CS:PRO_NAM, ES:VIDEO
;set UP			stack for return
START:
                PUSH			DS				;save DS
                SUB				AX,AX			;set AX	to zers
                PUSH			AX				;put it on stack
;set ES to extra segment
                MOV				AX,VIDEO
                MOV				ES,AX
;clear screent by writng zeors to it
;even bytes get 0
;odd bytes get 7
				MOV				CX, 80 * 25		;count
                MOV				BX, 0			;start of buff
CLEAR:
                MOV				ES:[WD_BUFF + BX], 0700H
                INC				BX				;incr pointer
                INC				BX				;twice
                LOOP			CLEAR			;do again
;screen pointer will be in CX register
;row number (0 to 24d) in CH
;column number (0 to 79d) in CL
;set screen pointer to center of screen			
				MOV				CH, 12D			;#rows divided by 2
                MOV				CL, 40D			;#columns div by 2
GET_CHAR:
                MOV				AH,READ_C		;code for read char
                INT				KEY_ROM			;keyboard I/O ROM call
                CMP				AL, ESCING		;is it escape?
                JZ				EXIT			;yes
                MOV				AL, AH			;put scan code in AL
                CMP				AL, UP			;is it UP arrow
                JNZ				NOT_UP			;no
                DEC				CH				;yes , decrement row
NOT_UP:
                CMP				AL, DOWN		;is it DOWN arrow?
                JNZ				NOT_DOWN		;no
                INC				CH				;yes, increment row
NOT_DOWN:
                CMP				AL, RIGHT		;is it RIGHT arrow?
                JNZ				NOT_RIGHT		;no
                INC				CL				;yes,increment column
NOT_RIGHT:
                CMP				AL, LEFT		;is it LEFT arrow?
                JNZ				LITE_IT			;no
                DEC				CL				;yes,decrement column
LITE_IT:
                MOV				AL ,160D		;bytes per row into AL
                MUL				CH				;time # of rows
                MOV				BL, CL			;# of columns in BL
                ROL				BL, 1			;times 2 to get bytes
                MOV				BH, 0			;clear top part of BX
                ADD				BX, AX			;gives ADDRESS offset
;ADDRESS offset in BX . put block char there
                MOV				AL, BLOCKING
                MOV				ES:[V_BUFF + BX], AL
                JMP				GET_CHAR		;go get next arrow
EXIT:
                RET								;return to DOS
MAIN			ENDP							;end of MAIN part of program
;-----------------------------------------------------------------------
PRO_NAM			ENDS
;***********************************************************************
                END				START